import os
import asyncio
import streamlit as st
import semantic_kernel as sk
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path
from typing import List, Dict
import pickle
import time

# Try to import optional dependencies for vector store
try:
    import faiss
    import numpy as np
    import PyPDF2
    VECTOR_STORE_AVAILABLE = True
    
    # Check for GPU support
    try:
        GPU_AVAILABLE = faiss.get_num_gpus() > 0
    except:
        GPU_AVAILABLE = False
        
    # Try to import advanced RAG dependencies
    try:
        from rank_bm25 import BM25Okapi
        import re
        from collections import Counter
        BM25_AVAILABLE = True
    except ImportError:
        BM25_AVAILABLE = False
        
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    GPU_AVAILABLE = False
    BM25_AVAILABLE = False

class HybridRAGManager:
    """Advanced Hybrid RAG with Intelligent Pre-processing and Context Compression."""
    
    def __init__(self, openai_client, model_name, data_folder="data", vector_store_path="vector_store"):
        if not VECTOR_STORE_AVAILABLE:
            raise ImportError("Vector store dependencies not available. Install with: pip install faiss-cpu PyPDF2 rank-bm25")
        
        self.openai_client = openai_client
        self.model_name = model_name
        self.data_folder = Path(data_folder)
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True)
        self.data_folder.mkdir(exist_ok=True)
        
        # Advanced settings
        self.use_gpu = GPU_AVAILABLE
        self.use_bm25 = BM25_AVAILABLE
        self.embedding_model = "text-embedding-ada-002"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_chunks_per_file = 50
        
        if self.use_gpu:
            st.info(f"🚀 GPU acceleration enabled! Found {faiss.get_num_gpus()} GPU(s)")
        
        if self.use_bm25:
            st.info("🔍 Hybrid retrieval available (Vector + BM25)")
        else:
            st.warning("⚠️ BM25 not available. Install: pip install rank-bm25")
    
    def intelligent_text_preprocessing(self, text: str) -> Dict:
        """Advanced text preprocessing with structure awareness."""
        if not text.strip():
            return {"sections": [], "metadata": {}}
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        
        # Extract structure markers
        sections = []
        current_section = {"title": "", "content": "", "type": "content"}
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect headers/sections
            if (len(line) < 100 and 
                (line.isupper() or 
                 re.match(r'^[A-Z][a-z\s]+:?\s*$', line) or
                 re.match(r'^\d+\.\s+[A-Z]', line) or
                 line.endswith(':'))):
                
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section.copy())
                
                # Start new section
                current_section = {
                    "title": line,
                    "content": "",
                    "type": "header" if line.isupper() else "section"
                }
            else:
                current_section["content"] += " " + line
        
        # Add final section
        if current_section["content"]:
            sections.append(current_section)
        
        # Extract metadata
        metadata = self._extract_metadata(text)
        
        return {"sections": sections, "metadata": metadata}
    
    def _extract_metadata(self, text: str) -> Dict:
        """Extract key metadata from text."""
        metadata = {}
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            metadata["emails"] = emails
        
        # Extract phone numbers
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        if phones:
            metadata["phones"] = phones
        
        # Extract years of experience
        experience = re.findall(r'(\d+)\s+years?\s+of\s+experience', text, re.IGNORECASE)
        if experience:
            metadata["experience_years"] = experience
        
        # Extract education keywords
        education_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'diploma']
        found_education = [word for word in education_keywords if word in text.lower()]
        if found_education:
            metadata["education_terms"] = found_education
        
        # Extract technology keywords
        tech_keywords = ['python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 'azure', 'docker', 'kubernetes']
        found_tech = [word for word in tech_keywords if word.lower() in text.lower()]
        if found_tech:
            metadata["technologies"] = found_tech
        
        return metadata
    
    def intelligent_chunking(self, sections: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Intelligent chunking with context preservation."""
        chunks = []
        
        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")
            section_type = section.get("type", "content")
            
            if not content.strip():
                continue
            
            # For small sections, keep whole
            if len(content) <= chunk_size:
                chunks.append({
                    "content": f"{title}\n{content}".strip(),
                    "title": title,
                    "type": section_type,
                    "tokens": self._estimate_tokens(content)
                })
                continue
            
            # Split large sections intelligently
            sentences = re.split(r'(?<=[.!?])\s+', content)
            current_chunk = title + "\n" if title else ""
            current_size = len(current_chunk)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding sentence exceeds chunk size
                if current_size + len(sentence) > chunk_size and current_chunk.strip():
                    # Save current chunk
                    chunks.append({
                        "content": current_chunk.strip(),
                        "title": title,
                        "type": section_type,
                        "tokens": self._estimate_tokens(current_chunk)
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = f"{title}\n{overlap_text}" if title else overlap_text
                    current_size = len(current_chunk)
                
                current_chunk += " " + sentence
                current_size += len(sentence) + 1
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    "content": current_chunk.strip(),
                    "title": title,
                    "type": section_type,
                    "tokens": self._estimate_tokens(current_chunk)
                })
        
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return len(text.split()) * 1.3  # Approximate
    
    def create_hybrid_search_index(self, chunks: List[Dict]) -> Dict:
        """Create both vector and BM25 indices."""
        if not chunks:
            return {"vector_index": None, "bm25_index": None, "metadata": []}
        
        # Prepare texts and metadata
        texts = [chunk["content"] for chunk in chunks]
        metadata = [
            {
                "content": chunk["content"],  # Store full content now
                "title": chunk.get("title", ""),
                "type": chunk.get("type", "content"),
                "tokens": chunk.get("tokens", 0),
                "chunk_id": i
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Create vector index
        st.info("🔄 Creating vector embeddings...")
        embeddings = self.get_embeddings_optimized(texts)
        vector_index = self.create_faiss_index(embeddings)
        
        # Create BM25 index
        bm25_index = None
        tokenized_docs = None
        if self.use_bm25:
            st.info("🔄 Creating BM25 sparse index...")
            tokenized_docs = [self._tokenize_for_bm25(text) for text in texts]
            bm25_index = BM25Okapi(tokenized_docs)
        
        return {
            "vector_index": vector_index,
            "bm25_index": bm25_index,
            "metadata": metadata,
            "tokenized_docs": tokenized_docs if self.use_bm25 else None
        }
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def _advanced_tokenize_for_bm25(self, text: str) -> List[str]:
        """Advanced tokenization for better BM25 matching."""
        # Simple tokenization with basic preprocessing
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [token for token in text.split() if len(token) > 1]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        return re.split(r'(?<=[.!?])\s+', text)
    
    def get_embeddings_optimized(self, texts: List[str], batch_size: int = 40) -> np.ndarray:
        """Optimized embedding generation."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Process batch
                batch_texts = [text[:8000] for text in batch]  # Truncate
                
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Progress
                progress = min(i + batch_size, len(texts))
                if progress % 20 == 0:
                    st.write(f"✅ Processed {progress}/{len(texts)} embeddings")
                
            except Exception as e:
                st.warning(f"Error in batch {i//batch_size + 1}: {str(e)}")
                # Add zero vectors for failed batch
                for _ in batch:
                    embeddings.append([0.0] * 1536)
        
        return np.array(embeddings, dtype=np.float32)
    
    def create_faiss_index(self, embeddings: np.ndarray):
        """Create optimized FAISS index."""
        if embeddings.size == 0:
            return None
        
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        # Choose index type
        if n_vectors < 1000:
            index = faiss.IndexFlatL2(dimension)
        else:
            nlist = min(100, n_vectors // 10)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings)
        
        # GPU acceleration
        if self.use_gpu and n_vectors > 500:
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu_index.add(embeddings)
                index = faiss.index_gpu_to_cpu(gpu_index)
                st.success("🚀 Used GPU acceleration!")
            except:
                index.add(embeddings)
        else:
            index.add(embeddings)
        
        return index
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Dict]:
        """Enhanced hybrid search with query preprocessing."""
        indices = self.load_hybrid_indices()
        if not indices or not indices["metadata"]:
            return []
        
        vector_index = indices["vector_index"]
        bm25_index = indices["bm25_index"]
        metadata = indices["metadata"]
        
        # Preprocess query for better matching
        processed_query = self._preprocess_query(query)
        
        all_results = []
        
        # Vector search with query variations
        if vector_index:
            # Original query
            vector_results = self._vector_search_internal(query, vector_index, metadata, k)
            for result in vector_results:
                result["search_type"] = "vector"
                result["query_variant"] = "original"
            all_results.extend(vector_results)
            
            # Processed query if different
            if processed_query != query:
                processed_results = self._vector_search_internal(processed_query, vector_index, metadata, k//2)
                for result in processed_results:
                    result["search_type"] = "vector"
                    result["query_variant"] = "processed"
                all_results.extend(processed_results)
        
        # BM25 search
        if bm25_index and self.use_bm25:
            bm25_results = self._bm25_search(processed_query, bm25_index, metadata, k)
            for result in bm25_results:
                result["search_type"] = "bm25"
            all_results.extend(bm25_results)
        
        # Enhanced reciprocal rank fusion
        combined_results = self._enhanced_reciprocal_rank_fusion(all_results, k)
        
        return combined_results[:k]
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better search performance."""
        # Convert to lowercase for better matching
        processed = query.lower()
        
        # Expand common abbreviations
        abbreviations = {
            'yrs': 'years',
            'exp': 'experience',
            'tech': 'technology',
            'dev': 'development',
            'mgmt': 'management',
            'eng': 'engineering',
            'comp': 'computer',
            'sci': 'science',
            'univ': 'university'
        }
        
        for abbr, full in abbreviations.items():
            processed = re.sub(r'\b' + abbr + r'\b', full, processed)
        
        # Add context for experience queries
        if 'experience' in processed and 'years' not in processed:
            processed += ' years'
        
        return processed
    
    def _vector_search_internal(self, query: str, index, metadata: List[Dict], k: int) -> List[Dict]:
        """Enhanced vector search with better scoring."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[query[:8000]]
            )
            
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            
            # Search with more candidates for better selection
            search_k = min(k * 3, index.ntotal)
            distances, indices = index.search(query_embedding, search_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata) and idx >= 0:
                    result = metadata[idx].copy()
                    # Convert L2 distance to similarity score (0-1, higher is better)
                    similarity = 1.0 / (1.0 + distances[0][i])
                    result["similarity_score"] = float(distances[0][i])  # Keep original for compatibility
                    result["normalized_score"] = similarity
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Vector search error: {str(e)}")
            return []
    
    def _bm25_search(self, query: str, bm25_index, metadata: List[Dict], k: int) -> List[Dict]:
        """Enhanced BM25 search with better scoring."""
        try:
            query_tokens = self._advanced_tokenize_for_bm25(query)
            
            if not query_tokens:
                return []
            
            scores = bm25_index.get_scores(query_tokens)
            
            # Get top k indices with non-zero scores
            scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for idx, score in scored_indices[:k * 2]:  # Get more candidates
                if idx < len(metadata):
                    result = metadata[idx].copy()
                    # Normalize BM25 score to similarity format
                    normalized_score = min(score / 10.0, 1.0)  # Adjust scaling as needed
                    result["similarity_score"] = float(1.0 - normalized_score)  # For compatibility
                    result["bm25_score"] = float(score)
                    result["normalized_score"] = normalized_score
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"BM25 search error: {str(e)}")
            return []
    
    def _enhanced_reciprocal_rank_fusion(self, results: List[Dict], k: int) -> List[Dict]:
        """Enhanced RRF with query variant and search type weighting."""
        rrf_scores = {}
        
        # Group results by search type and query variant
        vector_results = [r for r in results if r.get("search_type") == "vector"]
        bm25_results = [r for r in results if r.get("search_type") == "bm25"]
        
        # Sort each group by their respective scores
        vector_results.sort(key=lambda x: x.get("similarity_score", float('inf')))
        bm25_results.sort(key=lambda x: x.get("similarity_score", float('inf')))
        
        # RRF parameters
        rank_constant = 60
        vector_weight = 0.7  # Slightly favor vector search
        bm25_weight = 0.3
        
        # Calculate RRF scores for vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"result": result, "score": 0, "vector_rank": None, "bm25_rank": None}
            
            score_contribution = vector_weight / (rank_constant + rank + 1)
            rrf_scores[chunk_id]["score"] += score_contribution
            rrf_scores[chunk_id]["vector_rank"] = rank + 1
        
        # Calculate RRF scores for BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_id = result["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"result": result, "score": 0, "vector_rank": None, "bm25_rank": None}
            
            score_contribution = bm25_weight / (rank_constant + rank + 1)
            rrf_scores[chunk_id]["score"] += score_contribution
            rrf_scores[chunk_id]["bm25_rank"] = rank + 1
        
        # Sort by RRF score and prepare final results
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        final_results = []
        for chunk_id, data in sorted_results:
            result = data["result"].copy()
            result["rrf_score"] = data["score"]
            result["vector_rank"] = data["vector_rank"]
            result["bm25_rank"] = data["bm25_rank"]
            result["similarity_score"] = 1.0 - data["score"]  # Convert for compatibility
            final_results.append(result)
        
        return final_results
    
    def context_compression(self, results: List[Dict], query: str, max_tokens: int = 4000) -> str:
        """Advanced context compression with query-aware selection."""
        if not results:
            return ""
        
        # Sort by RRF score if available, otherwise by similarity
        results = sorted(results, 
                        key=lambda x: x.get("rrf_score", 1.0 - x.get("similarity_score", 1.0)), 
                        reverse=True)
        
        compressed_sections = []
        total_tokens = 0
        seen_content_hashes = set()
        query_keywords = set(self._advanced_tokenize_for_bm25(query.lower()))
        
        for result in results:
            content = result.get("content", "")
            title = result.get("title", "")
            
            # Skip very similar content using content hashing
            content_hash = hash(content[:300])
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            
            # Estimate tokens
            content_tokens = self._estimate_tokens(content)
            title_tokens = self._estimate_tokens(title) if title else 0
            section_tokens = content_tokens + title_tokens
            
            # Check if we can fit this section
            if total_tokens + section_tokens > max_tokens:
                # Try to include partial content with query-relevant excerpts
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Only if we have meaningful space
                    partial_content = self._extract_query_relevant_excerpt(
                        content, query_keywords, remaining_tokens * 4
                    )
                    if partial_content:
                        section_text = f"**{title}**\n{partial_content}..." if title else f"{partial_content}..."
                        compressed_sections.append(section_text)
                break
            
            # Add full content with formatting
            if title:
                section_text = f"**{title}**\n{content}"
            else:
                section_text = content
                
            compressed_sections.append(section_text)
            total_tokens += section_tokens
        
        return "\n\n---\n\n".join(compressed_sections)
    
    def _extract_query_relevant_excerpt(self, content: str, query_keywords: set, max_chars: int) -> str:
        """Extract the most relevant excerpt from content based on query keywords."""
        if not query_keywords or not content:
            return content[:max_chars]
        
        sentences = self._split_into_sentences(content)
        
        # Score sentences by keyword relevance
        scored_sentences = []
        for sentence in sentences:
            sentence_tokens = set(self._advanced_tokenize_for_bm25(sentence.lower()))
            overlap = len(query_keywords.intersection(sentence_tokens))
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        if not scored_sentences:
            # No keyword matches, return beginning
            return content[:max_chars]
        
        # Sort by relevance score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Build excerpt from most relevant sentences
        excerpt_parts = []
        current_length = 0
        
        for sentence, score in scored_sentences:
            if current_length + len(sentence) > max_chars:
                break
            excerpt_parts.append(sentence)
            current_length += len(sentence) + 1
        
        return " ".join(excerpt_parts)
    
    def create_vector_store(self) -> Dict:
        """Create advanced hybrid RAG system with enhanced processing."""
        pdf_files = list(self.data_folder.glob("*.pdf"))
        
        if not pdf_files:
            return {"success": False, "message": "No PDF files found in data folder"}
        
        # Enhanced capacity with hybrid approach
        max_files = 20  # Increased from 15
        if len(pdf_files) > max_files:
            st.warning(f"Processing first {max_files} of {len(pdf_files)} files")
            pdf_files = pdf_files[:max_files]
        
        all_chunks = []
        processing_stats = {"files": 0, "sections": 0, "chunks": 0, "metadata_items": 0}
        
        # Progress tracking
        main_progress = st.progress(0)
        main_status = st.empty()
        start_time = time.time()
        
        # Phase 1: Intelligent preprocessing (35% of progress)
        main_status.text("🧠 Advanced preprocessing and structure extraction...")
        
        for file_idx, pdf_file in enumerate(pdf_files):
            # Extract text with better handling
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                continue
            
            # Advanced preprocessing
            processed = self.intelligent_text_preprocessing(text)
            processing_stats["sections"] += len(processed["sections"])
            processing_stats["metadata_items"] += len(processed["metadata"])
            
            # Enhanced chunking
            chunks = self.intelligent_chunking(processed["sections"])
            processing_stats["chunks"] += len(chunks)
            
            # Add enriched metadata
            for chunk in chunks:
                chunk["source"] = pdf_file.name
                chunk["file_metadata"] = processed["metadata"]
                chunk["file_index"] = file_idx
            
            all_chunks.extend(chunks)
            processing_stats["files"] += 1
            
            # Update progress
            file_progress = (file_idx + 1) / len(pdf_files) * 0.35
            main_progress.progress(file_progress)
        
        if not all_chunks:
            return {"success": False, "message": "No content extracted from PDF files"}
        
        # Limit chunks if too many
        if len(all_chunks) > self.max_chunks_per_file * len(pdf_files):
            max_total_chunks = self.max_chunks_per_file * len(pdf_files)
            st.info(f"Limiting to {max_total_chunks} chunks for optimal performance")
            all_chunks = all_chunks[:max_total_chunks]
            processing_stats["chunks"] = len(all_chunks)
        
        main_status.text(f"📊 Creating hybrid indices for {len(all_chunks)} chunks...")
        main_progress.progress(0.35)
        
        # Phase 2: Create hybrid indices (55% of progress)
        try:
            indices = self.create_hybrid_search_index(all_chunks)
            main_progress.progress(0.9)
        except Exception as e:
            return {"success": False, "message": f"Error creating indices: {str(e)}"}
        
        # Phase 3: Save everything (10% of progress)
        main_status.text("💾 Saving hybrid RAG system...")
        
        try:
            self.save_hybrid_indices(indices)
        except Exception as e:
            return {"success": False, "message": f"Error saving: {str(e)}"}
        
        # Complete with detailed stats
        total_time = time.time() - start_time
        main_progress.progress(1.0)
        main_status.text(f"✅ Hybrid RAG system created in {total_time:.1f}s!")
        
        search_types = ["Dense Vector"]
        if self.use_bm25:
            search_types.append("Sparse BM25")
        
        return {
            "success": True,
            "message": f"Hybrid RAG created: {processing_stats['chunks']} chunks from {processing_stats['files']} files ({', '.join(search_types)})",
            "chunks": processing_stats["chunks"],
            "files": processing_stats["files"],
            "sections": processing_stats["sections"],
            "metadata_items": processing_stats["metadata_items"],
            "time": total_time,
            "search_types": search_types
        }
    
    def save_hybrid_indices(self, indices: Dict):
        """Save hybrid indices with enhanced metadata."""
        # Save vector index
        if indices["vector_index"]:
            vector_path = self.vector_store_path / "faiss_index.bin"
            faiss.write_index(indices["vector_index"], str(vector_path))
        
        # Save comprehensive metadata
        metadata_path = self.vector_store_path / "hybrid_metadata.pkl"
        save_data = {
            "metadata": indices["metadata"],
            "bm25_index": indices["bm25_index"],
            "tokenized_docs": indices.get("tokenized_docs"),
            "creation_time": time.time(),
            "system_info": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "use_bm25": self.use_bm25,
                "embedding_model": self.embedding_model
            }
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_hybrid_indices(self) -> Dict:
        """Load hybrid indices with validation."""
        vector_path = self.vector_store_path / "faiss_index.bin"
        metadata_path = self.vector_store_path / "hybrid_metadata.pkl"
        
        if not (vector_path.exists() and metadata_path.exists()):
            return {}
        
        try:
            # Load vector index
            vector_index = faiss.read_index(str(vector_path))
            
            # Load metadata with validation
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure
            metadata = data.get("metadata", [])
            if not isinstance(metadata, list):
                raise ValueError("Invalid metadata format")
            
            return {
                "vector_index": vector_index,
                "bm25_index": data.get("bm25_index"),
                "metadata": metadata,
                "tokenized_docs": data.get("tokenized_docs"),
                "system_info": data.get("system_info", {})
            }
        except Exception as e:
            st.error(f"Error loading hybrid indices: {str(e)}")
            return {}
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Enhanced PDF text extraction with better error handling."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Increased page limit for hybrid system
                max_pages = min(200, len(pdf_reader.pages))
                text_parts = []
                
                for i, page in enumerate(pdf_reader.pages[:max_pages]):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            # Basic cleaning
                            text = re.sub(r'\s+', ' ', text.strip())
                            text_parts.append(text)
                    except Exception as page_error:
                        # Continue with other pages if one fails
                        continue
                
                if not text_parts:
                    return ""
                
                full_text = "\n".join(text_parts)
                
                if len(pdf_reader.pages) > max_pages:
                    st.info(f"Processed {max_pages} of {len(pdf_reader.pages)} pages from {pdf_path.name}")
                
                return full_text
                
        except Exception as e:
            st.error(f"Error reading PDF {pdf_path.name}: {str(e)}")
            return ""
    
    def search_similar(self, query: str, k: int = 8) -> List[Dict]:
        """Main search interface using advanced hybrid approach."""
        try:
            # Use enhanced hybrid search
            results = self.hybrid_search(query, k)
            
            # Ensure full content is available for display
            indices = self.load_hybrid_indices()
            if indices and indices["metadata"]:
                metadata = indices["metadata"]
                for result in results:
                    chunk_id = result.get("chunk_id", -1)
                    if 0 <= chunk_id < len(metadata):
                        # Metadata now stores full content
                        result["content"] = metadata[chunk_id].get("content", result.get("content", ""))
            
            return results
            
        except Exception as e:
            st.error(f"Hybrid search error: {str(e)}")
            return []

class ToolPlugin:
    """A plugin that exposes tools for the chatbot."""
    @kernel_function(
        description="Logs a request to report sick.",
        name="report_sick"
    )
    def report_sick(self):
        """Placeholder function to handle reporting sick."""
        return "Request to report sick has been logged."

    @kernel_function(
        description="Logs a request to book a hotel.",
        name="book_hotel"
    )
    def book_hotel(self):
        """Placeholder function to handle booking a hotel."""
        return "Request to book a hotel has been logged."

    @kernel_function(
        description="Logs a request to book a limo.",
        name="book_limo"
    )
    def book_limo(self):
        """Placeholder function to handle booking a limo."""
        return "Request to book a limo has been logged."

    @kernel_function(
        description="Logs a request to report fatigue.",
        name="report_fatigue"
    )
    def report_fatigue(self):
        """Placeholder function to handle reporting fatigue."""
        return "Request to report fatigue has been logged."

    @kernel_function(
        description="Handles knowledge base queries and document searches.",
        name="search_knowledge_base"
    )
    def search_knowledge_base(self):
        """Placeholder function to handle knowledge base searches."""
        return "Knowledge base search request has been logged."

def create_tool_dict(metadata):
    """Convert KernelFunctionMetadata to dictionary format for tools."""
    return {
        "type": "function",
        "function": {
            "name": metadata.fully_qualified_name,
            "description": metadata.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

def categorize_with_direct_openai(openai_client, user_input, model_name):
    """Use direct OpenAI client to categorize user input with context awareness."""
    
    # Check conversation context - if last interaction was Knowledge Base, be more likely to route there
    last_category = st.session_state.get('last_successful_category', None)
    context_hint = ""
    
    if last_category == "Search Knowledge Base":
        context_hint = "\n\nNote: The previous interaction was about searching documents/knowledge base, so follow-up questions about the same topic should likely be categorized as 'Search Knowledge Base'."
    
    categorization_prompt = f"""You are a categorization assistant. Analyze the user's request and determine which category it belongs to.

Available categories:
1. Report Sick - for health issues, illness, feeling unwell, sickness, medical problems, pain, headaches
2. Report Fatigue - for tiredness, exhaustion, fatigue, being worn out, sleepiness, energy issues
3. Book Hotel - for hotel bookings, accommodation requests, room reservations, lodging
4. Book Limo - for transportation requests, ride bookings, car services, taxi requests, travel
5. Search Knowledge Base - for questions about documents, PDFs, information lookup, research queries, asking about specific content, "what does the document say", "find information about", "search for", follow-up questions about people/entities mentioned in documents, questions using pronouns like "he/she/they" that refer to document content, company information, personal details, qualifications, skills, experience details

User request: "{user_input}"{context_hint}

Rules:
- Respond with ONLY the exact category name (e.g., "Report Sick" or "Search Knowledge Base")
- If the request doesn't clearly match any category, respond with "no_match"
- Do not provide explanations or additional text
- Be flexible in understanding different ways people might express these needs
- Pay special attention to follow-up questions that might refer to document content
- Questions about specific people, their details, work, skills, etc. should go to "Search Knowledge Base"

Category:"""

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": categorization_prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        category = response.choices[0].message.content.strip()
        
        # Validate the response is one of our expected categories
        valid_categories = ["Report Sick", "Report Fatigue", "Book Hotel", "Book Limo", "Search Knowledge Base", "no_match"]
        
        if category in valid_categories:
            return category if category != "no_match" else None
        else:
            # If LLM returns something unexpected, try to match it to our categories
            category_lower = category.lower()
            if "sick" in category_lower or "health" in category_lower:
                return "Report Sick"
            elif "fatigue" in category_lower or "tired" in category_lower:
                return "Report Fatigue"
            elif "hotel" in category_lower or "accommodation" in category_lower:
                return "Book Hotel"
            elif "limo" in category_lower or "transport" in category_lower or "ride" in category_lower:
                return "Book Limo"
            elif "search" in category_lower or "document" in category_lower or "find" in category_lower or "knowledge" in category_lower or "what" in category_lower:
                return "Search Knowledge Base"
            else:
                # Enhanced fallback logic with context awareness
                if last_category == "Search Knowledge Base":
                    # If previous was KB and current has question words or pronouns, likely KB
                    question_indicators = ["what", "who", "where", "when", "how", "which", "he", "she", "they", "company", "work", "job", "does", "is", "are"]
                    if any(indicator in user_input.lower().split() for indicator in question_indicators):
                        return "Search Knowledge Base"
                return None
                
    except Exception as e:
        st.error(f"LLM categorization failed: {str(e)}")
        return None

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components."""
    load_dotenv()

    # For Azure OpenAI, we need different environment variables
    endpoint = os.getenv("ENDPOINT_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    model_name = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")

    # Extract the base endpoint (remove the full path)
    if endpoint and "/openai/deployments/" in endpoint:
        base_endpoint = endpoint.split("/openai/deployments/")[0]
    else:
        base_endpoint = endpoint

    # Validate configuration
    if not base_endpoint or not api_key or not model_name:
        st.error("Configuration error: Please check your environment variables.")
        return None, None, None

    # Initialize direct OpenAI client for categorization
    openai_client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version or "2024-12-01-preview",
        azure_endpoint=base_endpoint
    )

    # Initialize Semantic Kernel for function calling
    kernel = sk.Kernel()
    
    # Use Azure AI Inference connector for function calling (we'll bypass LLM categorization)
    chat_service = AzureAIInferenceChatCompletion(
        ai_model_id=model_name,
        endpoint=base_endpoint,
        api_key=api_key,
    )
    kernel.add_service(chat_service)
    kernel.add_plugin(ToolPlugin(), "ToolPlugin")

    return kernel, chat_service, openai_client

def show_debug_panel(openai_client, model_name, base_endpoint, api_version, api_key):
    """Show debug panel with configuration and test connection."""
    with st.sidebar:
        with st.expander("🔧 Configuration Debug", expanded=False):
            st.write(f"**Endpoint:** {base_endpoint}")
            st.write(f"**Model:** {model_name}")
            st.write(f"**API Version:** {api_version}")
            st.write(f"**API Key:** {'✅ Present' if api_key else '❌ Missing'}")
            
            # Test connection
            if st.button("🧪 Test Connection"):
                try:
                    test_response = openai_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    st.success("✅ Connection successful!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
                    if "404" in str(e):
                        st.warning("💡 Check your DEPLOYMENT_NAME in environment variables")
                    elif "401" in str(e):
                        st.warning("💡 Check your API key and endpoint URL")

async def process_request(kernel, category, function_name):
    """Process the user request and call the appropriate function."""
    try:
        tool_result = await kernel.invoke(
            plugin_name="ToolPlugin",
            function_name=function_name,
            arguments=KernelArguments(),
        )
        return str(tool_result)
    except Exception as e:
        st.error(f"Error calling function: {e}")
        return f"Request logged successfully (local fallback)."

def create_hybrid_vector_store_ui(openai_client, model_name):
    """Hybrid RAG UI for vector store management."""
    st.sidebar.markdown("---")
    st.sidebar.header("📚 Hybrid RAG Knowledge Base")
    
    # Check if vector store dependencies are available
    if not VECTOR_STORE_AVAILABLE:
        st.sidebar.error("❌ Vector store dependencies not installed")
        st.sidebar.code("pip install faiss-cpu PyPDF2 rank-bm25")
        return None
    
    # Initialize hybrid RAG manager
    if "vector_manager" not in st.session_state:
        try:
            st.session_state.vector_manager = HybridRAGManager(openai_client, model_name)
        except Exception as e:
            st.sidebar.error(f"Error initializing hybrid RAG: {str(e)}")
            return None
    
    vector_manager = st.session_state.vector_manager
    
    # Check if data folder exists and show PDF count
    data_folder = Path("data")
    pdf_files = list(data_folder.glob("*.pdf")) if data_folder.exists() else []
    
    st.sidebar.markdown(f"**PDF Files in Data Folder:** {len(pdf_files)}")
    
    # Show improved limits
    if len(pdf_files) > 15:
        st.sidebar.warning(f"⚠️ {len(pdf_files)} PDF files found. First 15 will be processed for optimal performance.")
    
    if pdf_files:
        with st.sidebar.expander("📄 Available PDF Files", expanded=False):
            for i, pdf_file in enumerate(pdf_files[:20]):  # Show more in list
                status = "✅" if i < 15 else "⏳"
                file_size = pdf_file.stat().st_size / 1024  # KB
                st.write(f"{status} {pdf_file.name} ({file_size:.0f}KB)")
            if len(pdf_files) > 20:
                st.write(f"... and {len(pdf_files) - 20} more files")
    else:
        st.sidebar.warning("⚠️ No PDF files found in 'data' folder")
        st.sidebar.markdown("*Place PDF files in the 'data' folder to create hybrid RAG*")
    
    # Hybrid RAG features info
    st.sidebar.info("⚡ **Hybrid RAG Features:**\n- Context-aware follow-up questions\n- Vector + BM25 sparse search\n- Intelligent preprocessing\n- Context compression\n- Structure-aware chunking\n- Process up to 15 files\n- Files up to 15MB supported\n- Reciprocal rank fusion")
    
    # Check if hybrid indices exist
    indices = vector_manager.load_hybrid_indices()
    vector_store_exists = bool(indices and indices.get("metadata"))
    
    if vector_store_exists:
        st.sidebar.success("✅ Hybrid RAG system exists")
        
        # Show hybrid RAG info
        try:
            metadata = indices.get("metadata", [])
            if metadata:
                st.sidebar.markdown(f"**Chunks:** {len(metadata)}")
                
                # Get unique sources
                sources = set(item["source"] for item in metadata if "source" in item)
                st.sidebar.markdown(f"**Sources:** {len(sources)}")
                
                # Show search capabilities
                search_types = ["Vector Search"]
                if indices.get("bm25_index") and BM25_AVAILABLE:
                    search_types.append("BM25 Sparse")
                st.sidebar.markdown(f"**Search Types:** {', '.join(search_types)}")
        except:
            pass
    else:
        st.sidebar.info("ℹ️ No hybrid RAG system found")
    
    # Create/Recreate hybrid RAG button
    button_text = "🔄 Recreate Hybrid RAG" if vector_store_exists else "⚡ Create Hybrid RAG"
    
    if st.sidebar.button(button_text, disabled=(len(pdf_files) == 0)):
        if len(pdf_files) == 0:
            st.sidebar.error("❌ No PDF files found to process")
        else:
            with st.sidebar:
                st.markdown("### 🚀 Creating Hybrid RAG System")
                try:
                    result = vector_manager.create_vector_store()
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.balloons()
                        
                        # Show search capabilities
                        if "search_types" in result:
                            st.info(f"🔍 Search types: {', '.join(result['search_types'])}")
                    else:
                        st.error(f"❌ {result['message']}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Hybrid RAG search interface
    if vector_store_exists:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Hybrid RAG Search")
        
        search_query = st.sidebar.text_input("Search query:", placeholder="Search in PDFs...")
        
        if st.sidebar.button("🔍 Hybrid Search") and search_query:
            with st.sidebar:
                with st.spinner("Searching..."):
                    try:
                        start_time = time.time()
                        results = vector_manager.search_similar(search_query, k=5)
                        search_time = time.time() - start_time
                        
                        if results:
                            st.success(f"Found {len(results)} results in {search_time:.2f}s")
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Result {i} - {result.get('source', 'Unknown')}", expanded=(i==1)):
                                    st.write(f"**Source:** {result.get('source', 'Unknown')}")
                                    st.write(f"**Content:** {result.get('content', '')[:200]}...")
                                    
                                    # Show search type if available
                                    search_type = result.get('search_type', 'hybrid')
                                    rrf_score = result.get('rrf_score')
                                    if rrf_score:
                                        st.write(f"**Type:** {search_type} (RRF: {rrf_score:.4f})")
                                    else:
                                        st.write(f"**Similarity:** {result.get('similarity_score', 0):.4f}")
                        else:
                            st.info("No relevant results found")
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
    
    return vector_manager

def handle_agent_conversation(category):
    """Handle specialized agent conversation based on category."""
    
    if category == "Report Sick":
        st.markdown("🏥 **Specialized Agent** is now assisting you.")
        st.markdown("I'll help you report your illness. I need to collect some basic information.")
        
        # Create form for date inputs
        with st.form("sick_report_form"):
            st.markdown("**Please provide the following information:**")
            from_date = st.date_input("From Date (sick leave start):")
            to_date = st.date_input("To Date (sick leave end):")
            
            submitted = st.form_submit_button("Submit Report Sick Request")
            
            if submitted:
                st.success(f"Thank you. I have recorded your sick leave from {from_date} to {to_date}.")
                st.session_state.agent_complete = True
                st.session_state.agent_result = f"✅ Sick report has been logged successfully from {from_date} to {to_date}."
                return True
                
    elif category == "Book Hotel":
        st.markdown("🏨 **Specialized Agent** is now assisting you.")
        st.markdown("I'll help you book a hotel. I need to collect some basic information.")
        
        # Create form for date inputs
        with st.form("hotel_booking_form"):
            st.markdown("**Please provide the following information:**")
            from_date = st.date_input("From Date (check-in):")
            to_date = st.date_input("To Date (check-out):")
            
            submitted = st.form_submit_button("Submit Book Hotel Request")
            
            if submitted:
                st.success(f"Thank you. I have recorded your hotel booking from {from_date} to {to_date}.")
                st.session_state.agent_complete = True
                st.session_state.agent_result = f"✅ Hotel booking request has been logged successfully from {from_date} to {to_date}."
                return True
                
    elif category == "Report Fatigue":
        st.markdown("😴 **Specialized Agent** is now assisting you.")
        st.markdown("I'll help you report your fatigue.")
        
        if st.button("Submit Report Fatigue Request", type="primary"):
            st.success("Thank you. I have recorded your fatigue report.")
            st.session_state.agent_complete = True
            st.session_state.agent_result = "✅ Fatigue report has been logged successfully."
            return True
            
    elif category == "Book Limo":
        st.markdown("🚗 **Specialized Agent** is now assisting you.")
        st.markdown("I'll help you book transportation.")
        
        if st.button("Submit Limo Booking Request", type="primary"):
            st.success("Thank you. I have recorded your transportation booking request.")
            st.session_state.agent_complete = True
            st.session_state.agent_result = "✅ Transportation booking request has been logged successfully."
            return True
            
    elif category == "Search Knowledge Base":
        st.markdown("📚 **Knowledge Base Agent** is now assisting you.")
        
        # Check if vector store exists
        if "vector_manager" not in st.session_state:
            st.error("❌ Hybrid RAG not initialized. Please create a hybrid RAG system first.")
            if st.button("Return to Main Agent"):
                return True
            return False
        
        vector_manager = st.session_state.vector_manager
        indices = vector_manager.load_hybrid_indices()
        
        if not indices or not indices.get("metadata"):
            st.error("❌ No hybrid RAG system found. Please create one first by uploading PDFs.")
            if st.button("Return to Main Agent"):
                return True
            return False
        
        # Check if we have the original user query from routing
        original_query = st.session_state.get('original_user_query', '')
        
        if original_query and not st.session_state.get('kb_query_processed', False):
            # Process the original query directly with hybrid RAG
            with st.spinner("🔍 Hybrid searching..."):
                try:
                    start_time = time.time()
                    
                    # Check if this is a general query about the knowledge base
                    general_queries = ["what information", "what's in", "what is in", "what does the document contain", "what topics", "summarize", "overview", "what's available"]
                    is_general_query = any(phrase in original_query.lower() for phrase in general_queries)
                    
                    # Use hybrid search with appropriate parameters
                    if is_general_query:
                        results = vector_manager.search_similar(original_query, k=12)
                    else:
                        results = vector_manager.search_similar(original_query, k=8)
                    
                    search_time = time.time() - start_time
                    
                    # Always try to use results if we have any
                    if results:
                        # Use context compression for better LLM input
                        max_tokens = 6000 if is_general_query else 4000
                        compressed_context = vector_manager.context_compression(results, original_query, max_tokens)
                        
                        # Create enhanced answer with hybrid RAG
                        answer_parts = []
                        
                        # Try to generate AI summary with compressed context
                        ai_summary_generated = False
                        try:
                            openai_client = st.session_state.get('openai_client')
                            model_name = os.getenv("DEPLOYMENT_NAME")
                            
                            if openai_client and model_name and compressed_context:
                                if is_general_query:
                                    summary_prompt = f"""Based on the following content from documents, provide a comprehensive overview of what information is available in the knowledge base.

User Question: {original_query}

Document Content:
{compressed_context}

Please provide a helpful summary of the main topics, types of information, and key content available in these documents. Be comprehensive and informative."""
                                else:
                                    summary_prompt = f"""Based on the following content from documents, provide a clear and direct answer to the user's question. Extract specific facts and details that answer the question.

User Question: {original_query}

Relevant Content:
{compressed_context}

Please provide a direct, accurate answer based on the content above. Focus on answering the specific question asked with concrete details and facts from the documents."""

                                response = openai_client.chat.completions.create(
                                    model=model_name,
                                    messages=[{"role": "user", "content": summary_prompt}],
                                    temperature=0.05,  # Very low temperature for factual accuracy
                                    max_tokens=800
                                )
                                
                                ai_summary = response.choices[0].message.content.strip()
                                
                                # Add search metadata
                                search_info = f"\n\n*📊 Hybrid search found {len(results)} relevant sections in {search_time:.2f}s*"
                                #answer_parts.append(ai_summary + search_info)
                                answer_parts.append(ai_summary)
                                ai_summary_generated = True
                                
                        except Exception as e:
                            # Silently fall back to showing compressed content
                            pass
                        
                        # If no AI summary, show compressed content directly
                        if not ai_summary_generated:
                            if is_general_query:
                                answer_parts.append("Based on your documents, here's what information is available:")
                                answer_parts.append(compressed_context)
                            else:
                                answer_parts.append("Based on your documents:")
                                answer_parts.append(compressed_context)
                            
                            # Add search metadata
                            search_info = f"\n\n*📊 Hybrid search found {len(results)} relevant sections in {search_time:.2f}s*"
                            answer_parts.append(search_info)
                        
                        final_answer = "\n".join(answer_parts)
                        
                        st.session_state.agent_complete = True
                        st.session_state.agent_result = final_answer
                        st.session_state.kb_query_processed = True
                        return True
                        
                    else:
                        # If no results found, show sample content from the knowledge base
                        try:
                            indices = vector_manager.load_hybrid_indices()
                            metadata = indices.get("metadata", [])
                            if metadata and len(metadata) > 0:
                                sample_content = []
                                sample_content.append("I couldn't find specific matches for your query, but here's a sample of what's in your knowledge base:")
                                
                                # Show first few chunks as samples
                                for i, item in enumerate(metadata[:4], 1):
                                    sample_content.append(f"\n**Sample {i} (from {item.get('source', 'Unknown')}):**")
                                    content = item.get('content', '')[:400] + "..." if len(item.get('content', '')) > 400 else item.get('content', '')
                                    sample_content.append(content)
                                
                                sample_content.append(f"\n*Total content: {len(metadata)} sections across your documents*")
                                sample_content.append("\nTry asking more specific questions about topics that interest you.")
                                
                                final_answer = "\n".join(sample_content)
                            else:
                                final_answer = "I couldn't find relevant information in your documents for that question. Try rephrasing or asking about different topics."
                        except:
                            final_answer = "I couldn't find relevant information in your documents for that question. Try rephrasing or asking about different topics."
                        
                        st.session_state.agent_complete = True
                        st.session_state.agent_result = final_answer
                        st.session_state.kb_query_processed = True
                        return True
                        
                except Exception as e:
                    # Handle search failures gracefully
                    error_msg = str(e)
                    if "404" in error_msg or "DeploymentNotFound" in error_msg:
                        fallback_answer = "I'm unable to search your documents due to a configuration issue with the AI service. Please check your Azure OpenAI deployment settings."
                    elif "401" in error_msg:
                        fallback_answer = "I'm unable to search your documents due to authentication issues. Please check your API credentials."
                    else:
                        fallback_answer = f"I encountered an error while searching your documents: {error_msg}"
                    
                    st.session_state.agent_complete = True
                    st.session_state.agent_result = fallback_answer
                    st.session_state.kb_query_processed = True
                    return True
        
        else:
            # Show form for additional queries
            with st.form("knowledge_query_form"):
                st.markdown("**Enter your question:**")
                query = st.text_area("Query:", placeholder="Ask any question about your documents...")
                
                submitted = st.form_submit_button("🔍 Search", type="primary")
                
                if submitted and query.strip():
                    with st.spinner("🔍 Hybrid searching..."):
                        try:
                            start_time = time.time()
                            results = vector_manager.search_similar(query.strip(), k=8)
                            search_time = time.time() - start_time
                            
                            # Always try to use results if we have any
                            if results:
                                # Use context compression
                                compressed_context = vector_manager.context_compression(results, query.strip(), 4000)
                                
                                answer_parts = []
                                
                                # Try AI summary with compressed context
                                ai_summary_generated = False
                                try:
                                    openai_client = st.session_state.get('openai_client')
                                    model_name = os.getenv("DEPLOYMENT_NAME")
                                    
                                    if openai_client and model_name and compressed_context:
                                        summary_prompt = f"""Based on the following content from documents, provide a clear and helpful answer to the user's question. Focus on directly answering what was asked.

User Question: {query}

Relevant Content:
{compressed_context}

Please provide a direct, accurate answer based on the content above. Extract specific facts and details that answer the question."""

                                        response = openai_client.chat.completions.create(
                                            model=model_name,
                                            messages=[{"role": "user", "content": summary_prompt}],
                                            temperature=0.05,
                                            max_tokens=700
                                        )
                                        
                                        ai_summary = response.choices[0].message.content.strip()
                                        
                                        # Add search metadata
                                        search_info = f"\n\n*📊 Hybrid search found {len(results)} relevant sections in {search_time:.2f}s*"
                                        answer_parts.append(ai_summary + search_info)
                                        ai_summary_generated = True
                                        
                                except:
                                    pass
                                
                                # Fallback to compressed content
                                if not ai_summary_generated:
                                    answer_parts.append("Based on your documents:")
                                    answer_parts.append(compressed_context)
                                    
                                    # Add search metadata
                                    search_info = f"\n\n*📊 Hybrid search found {len(results)} relevant sections in {search_time:.2f}s*"
                                    answer_parts.append(search_info)
                                
                                final_answer = "\n".join(answer_parts)
                                st.session_state.agent_complete = True
                                st.session_state.agent_result = final_answer
                                return True
                                
                            else:
                                st.session_state.agent_complete = True
                                st.session_state.agent_result = "I couldn't find relevant information in your documents for that question."
                                return True
                                
                        except Exception as e:
                            error_msg = str(e)
                            if "404" in error_msg or "DeploymentNotFound" in error_msg:
                                fallback_answer = "I'm unable to search your documents due to a configuration issue with the AI service."
                            elif "401" in error_msg:
                                fallback_answer = "I'm unable to search your documents due to authentication issues."
                            else:
                                fallback_answer = f"I encountered an error while searching: {error_msg}"
                            
                            st.session_state.agent_complete = True
                            st.session_state.agent_result = fallback_answer
                            return True
        
        # Option to ask another question
        if st.button("🔄 Ask Another Question"):
            st.session_state.kb_query_processed = False
            st.rerun()
    
    return False

def main():
    st.set_page_config(
        page_title="Multi-Agent Assistant (Optimized)",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Multi-Agent Assistant with Hybrid RAG")
    st.markdown("Featuring **ultra-fast hybrid RAG creation**, **vector + sparse search**, **intelligent preprocessing**, **context compression**, and optimized processing for all your requests.")

    # Initialize chatbot
    kernel, chat_service, openai_client = initialize_chatbot()
    
    if kernel is None:
        st.stop()
    
    # Show debug panel (outside cached function)
    endpoint = os.getenv("ENDPOINT_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    model_name = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")
    
    # Extract the base endpoint (remove the full path)
    if endpoint and "/openai/deployments/" in endpoint:
        base_endpoint = endpoint.split("/openai/deployments/")[0]
    else:
        base_endpoint = endpoint
    
    show_debug_panel(openai_client, model_name, base_endpoint, api_version, api_key)

    # Add hybrid RAG UI to sidebar
    vector_manager = create_hybrid_vector_store_ui(openai_client, model_name)
    
    # Store openai_client in session state for knowledge base agent
    st.session_state.openai_client = openai_client

    # Initialize session state for chat history and agent states
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.waiting_for_confirmation = False
        st.session_state.pending_category = None
        st.session_state.pending_function = None
        st.session_state.agent_active = False
        st.session_state.agent_complete = False
        st.session_state.agent_result = ""
        st.session_state.kb_query_processed = False
        st.session_state.original_user_query = ""
        st.session_state.last_successful_category = None

    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Handle different states
    if st.session_state.agent_active:
        # Agent is handling the conversation
        st.markdown("---")
        
        if handle_agent_conversation(st.session_state.pending_category):
            # Agent completed - store successful category for context
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.agent_result})
            st.session_state.last_successful_category = st.session_state.pending_category
            st.session_state.agent_active = False
            st.session_state.waiting_for_confirmation = False
            st.session_state.pending_category = None
            st.session_state.pending_function = None
            st.session_state.agent_complete = False
            st.session_state.agent_result = ""
            st.session_state.kb_query_processed = False
            st.session_state.original_user_query = ""
            st.rerun()
        
        # Add back to main button
        if st.button("🔄 Return to Main Agent"):
            st.session_state.agent_active = False
            st.session_state.waiting_for_confirmation = False
            st.session_state.pending_category = None
            st.session_state.pending_function = None
            st.session_state.kb_query_processed = False
            st.session_state.original_user_query = ""
            # Keep last_successful_category to maintain context for next question
            st.rerun()
            
    elif st.session_state.waiting_for_confirmation:
        # Show confirmation buttons
        st.markdown("### Confirmation Required")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Connect with Specialized Agent", use_container_width=True):
                # Connect to specialized agent
                st.session_state.messages.append({"role": "user", "content": "Yes"})
                st.session_state.agent_active = True
                st.rerun()
        
        with col2:
            if st.button("❌ No, cancel request", use_container_width=True):
                # Cancel the request
                st.session_state.messages.append({"role": "user", "content": "No"})
                st.session_state.messages.append({"role": "assistant", "content": "No problem! Feel free to ask about something else."})
                
                # Reset confirmation state
                st.session_state.waiting_for_confirmation = False
                st.session_state.pending_category = None
                st.session_state.pending_function = None
                st.session_state.kb_query_processed = False
                st.session_state.original_user_query = ""
                # Don't reset last_successful_category here to maintain context
                
                st.rerun()
    
    else:
        # Regular chat input
        if prompt := st.chat_input("What can I help you with today?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Show analyzing message
            with st.spinner("🔍 Analyzing your request..."):
                # Categorize using direct OpenAI client
                category = categorize_with_direct_openai(openai_client, prompt, model_name)
            
            # Define valid categories
            valid_categories = {
                "Report Sick": ("report_sick", "report that you're not feeling well"),
                "Report Fatigue": ("report_fatigue", "report fatigue or tiredness"), 
                "Book Hotel": ("book_hotel", "book a hotel"),
                "Book Limo": ("book_limo", "book a limo/transportation"),
                "Search Knowledge Base": ("search_knowledge_base", "search documents and knowledge base")
            }
            
            if category in valid_categories:
                function_name, description = valid_categories[category]
                
                # Store original query for Knowledge Base agent
                if category == "Search Knowledge Base":
                    st.session_state.original_user_query = prompt
                    st.session_state.kb_query_processed = False
                
                # Show identified category
                category_message = f"🎯 I've identified your request as: **{category}**\n\nWould you like me to connect you with our Specialized Agent?"
                st.session_state.messages.append({"role": "assistant", "content": category_message})
                
                # Set up confirmation state
                st.session_state.waiting_for_confirmation = True
                st.session_state.pending_category = category
                st.session_state.pending_function = function_name
                
            else:
                # Unable to categorize
                error_message = "🤔 I'm sorry, I couldn't categorize your request. I can help you with reporting sickness, reporting fatigue, booking hotels, or booking transportation. Could you please clarify what you need help with?"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            st.rerun()

    # Enhanced sidebar with performance info
    with st.sidebar:
        st.header("🚀 Advanced Multi-Agent Services")
        st.markdown("""
        **I can help you with:**
        
        🏥 **Report Sick**
        - Health issues, illness, not feeling well
        - *Requires: From Date, To Date*
        
        😴 **Report Fatigue** 
        - Tiredness, exhaustion, being worn out
        - *No additional info required*
        
        🏨 **Book Hotel**
        - Hotel bookings, accommodation, rooms
        - *Requires: From Date, To Date*
        
        🚗 **Book Limo**
        - Transportation, rides, car services
        - *No additional info required*
        
        📚 **Hybrid RAG Search**
        - Advanced hybrid retrieval (Vector + BM25 sparse search)
        - Intelligent preprocessing and context compression
        - Context-aware follow-up questions
        - Ask questions about uploaded PDFs with superior accuracy
        - Get AI-powered summaries with optimal context
        - Works with difficult and complex queries
        - *Requires: Hybrid RAG system with PDFs*
        """)
        
        st.header("💡 Example Requests")
        st.markdown("""
        **Service Requests:**
        - "I have a headache"
        - "Need somewhere to stay tonight"
        - "Feeling really drained"
        - "Can you get me a ride?"
        
        **Hybrid RAG Queries:**
        - "How many years of experience does [person] have?"
        - "What skills are mentioned in the resume?"
        - "Find information about XYZ"
        - "What are the technical skills listed?"
        - "Summarize the main qualifications"
        - "What does the document say about education?"
        
        **📋 Context-Aware Follow-ups:**
        - After asking about someone: "Which company does he work for?"
        - "What about his education background?"
        - "Does he have any certifications?"
        - "Where is he located?"
        """)
        
        # Show context status
        if st.session_state.get('last_successful_category') == "Search Knowledge Base":
            st.info("🔗 **Context Active**: Follow-up questions will automatically use the Knowledge Base")
        
        st.header("⚡ Hybrid RAG Features")
        st.markdown("""
        - **Hybrid retrieval**: Vector search + BM25 sparse search
        - **Intelligent preprocessing**: Structure-aware text processing
        - **Context compression**: Optimal context for LLM generation
        - **Reciprocal rank fusion**: Combines multiple search strategies
        - **Context-aware conversations**: Follow-up questions automatically use Knowledge Base
        - **Enhanced chunking**: Better overlap and sizing for RAG
        - **Intelligent prompting**: Optimized for factual extraction
        - **Query expansion**: Automatically tries related terms
        - **Ultra-fast** hybrid system creation
        - **Batch processing** for embeddings
        - **GPU acceleration** when available
        - **Smart memory management**
        - **Process up to 15 PDF files**
        - **Files up to 15MB supported**
        """)
        
        # Show knowledge base status
        if VECTOR_STORE_AVAILABLE and "vector_manager" in st.session_state:
            vector_manager = st.session_state.vector_manager
            indices = vector_manager.load_hybrid_indices()
            if indices and indices.get("metadata"):
                metadata = indices["metadata"]
                sources = set(item.get("source", "") for item in metadata)
                search_types = ["Vector"]
                if indices.get("bm25_index") and BM25_AVAILABLE:
                    search_types.append("BM25")
                st.success(f"📚 Hybrid RAG Ready: {len(sources)} documents, {len(metadata)} chunks ({'+'.join(search_types)})")
                
                # Show context status in sidebar too
                if st.session_state.get('last_successful_category') == "Search Knowledge Base":
                    st.info("🔗 Context Active: Next questions will use Knowledge Base")
            else:
                st.info("📚 Hybrid RAG: Not created yet")
        
        st.markdown("---")
        st.header("🔧 Installation & Setup")
        st.markdown("""
        **For Hybrid RAG (Recommended):**
        ```bash
        pip install faiss-cpu PyPDF2 rank-bm25
        ```
        
        **For GPU acceleration:**
        ```bash
        pip install faiss-gpu PyPDF2 rank-bm25
        ```
        
        **Environment Variables Required:**
        ```bash
        ENDPOINT_URL=https://your-resource.openai.azure.com
        AZURE_OPENAI_API_KEY=your-api-key
        DEPLOYMENT_NAME=your-model-deployment-name
        API_VERSION=2024-12-01-preview
        ```
        
        **Troubleshooting 404 Errors:**
        - ✅ Verify DEPLOYMENT_NAME matches your Azure model deployment
        - ✅ Check ENDPOINT_URL format (no trailing slash)
        - ✅ Ensure deployment is active in Azure Portal
        - ✅ Test connection using the debug panel above
        
        **Hybrid RAG Features:**
        - Use PDF files under 15MB for best performance
        - Process up to 15 files at once  
        - Hybrid search combines vector similarity with BM25 sparse search
        - Intelligent preprocessing extracts structure and metadata
        - Context compression optimizes LLM input for better answers
        - GPU acceleration used automatically when available
        - Reciprocal rank fusion combines multiple search strategies
        """)
        
        # Show current performance status
        if VECTOR_STORE_AVAILABLE:
            gpu_status = "🚀 GPU Accelerated" if GPU_AVAILABLE else "💻 CPU Optimized"
            bm25_status = "🔍 BM25 Available" if BM25_AVAILABLE else "❌ BM25 Missing"
            st.markdown(f"**Status:** {gpu_status} | {bm25_status}")
            
        # Show AI summary status
        if "openai_client" in st.session_state:
            st.markdown("**Hybrid RAG:** ⚡ Full system available")
        else:
            st.markdown("**Hybrid RAG:** ❌ Configuration issue")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.waiting_for_confirmation = False
            st.session_state.pending_category = None
            st.session_state.pending_function = None
            st.session_state.agent_active = False
            st.session_state.agent_complete = False
            st.session_state.agent_result = ""
            st.session_state.kb_query_processed = False
            st.session_state.original_user_query = ""
            st.session_state.last_successful_category = None
            st.rerun()

if __name__ == "__main__":
    main()