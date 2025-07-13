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
from typing import List, Dict, Tuple, Optional, Set
import pickle
import time
from io import BytesIO
import json
import re
import math
from collections import Counter, defaultdict

# Try to import dependencies for modern RAG
try:
    import faiss
    import numpy as np
    import PyPDF2
    import tiktoken
    VECTOR_STORE_AVAILABLE = True
    
    # Check for GPU support
    try:
        GPU_AVAILABLE = faiss.get_num_gpus() > 0
    except:
        GPU_AVAILABLE = False
        
    # Try to import BM25 for hybrid search
    try:
        from rank_bm25 import BM25Okapi
        BM25_AVAILABLE = True
    except ImportError:
        BM25_AVAILABLE = False
        
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    GPU_AVAILABLE = False
    BM25_AVAILABLE = False

class ModernRAGManager:
    """Modern RAG system with latest recommended techniques."""
    
    def __init__(self, openai_client, model_name, data_folder="data", vector_store_path="vector_store"):
        if not VECTOR_STORE_AVAILABLE:
            raise ImportError("Vector store dependencies not available. Install with: pip install faiss-cpu PyPDF2 rank-bm25 tiktoken")
        
        self.openai_client = openai_client
        self.model_name = model_name
        self.data_folder = Path(data_folder)
        self.vector_store_path = Path(vector_store_path)
        
        # Create directories
        self.vector_store_path.mkdir(exist_ok=True)
        self.data_folder.mkdir(exist_ok=True)
        
        # Modern RAG settings
        self.use_gpu = GPU_AVAILABLE
        self.use_bm25 = BM25_AVAILABLE
        self.text_embedding_model = "text-embedding-3-large"
        
        # Token-based chunking (modern approach)
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        self.chunk_size = 512  # tokens (optimal for most models)
        self.chunk_overlap = 64  # tokens overlap
        self.max_tokens_per_chunk = 8000  # embedding model limit
        
        # Advanced retrieval settings
        self.sentence_window_size = 3  # for sentence window retrieval
        self.parent_chunk_size = 2048  # larger parent chunks
        self.use_hyde = True  # Hypothetical Document Embeddings
        self.use_query_expansion = True
        self.use_mmr = True  # Maximal Marginal Relevance
        self.mmr_diversity_score = 0.3
        
        # Quality settings
        self.min_chunk_tokens = 32
        self.max_chunks_per_file = 150
        
        self._initialize_tokenizer()
        self._show_capabilities()
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for proper token counting."""
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            st.info("✅ Initialized tiktoken tokenizer for GPT-4")
        except Exception as e:
            st.warning(f"⚠️ Tokenizer initialization failed: {e}")
            self.tokenizer = None
    
    def _show_capabilities(self):
        """Show modern RAG capabilities."""
        if self.use_gpu:
            st.info(f"🚀 GPU acceleration enabled! Found {faiss.get_num_gpus()} GPU(s)")
        
        if self.use_bm25:
            st.info("🔍 Hybrid retrieval available (Vector + BM25)")
        else:
            st.warning("⚠️ BM25 not available. Install: pip install rank-bm25")
        
        techniques = []
        if self.use_hyde:
            techniques.append("HyDE")
        if self.use_query_expansion:
            techniques.append("Query Expansion")
        if self.use_mmr:
            techniques.append("MMR")
        
        st.info(f"✨ Modern RAG techniques: {', '.join(techniques)}, Token-based chunking, Sentence Window")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback estimation (rough)
            return len(text.split()) * 1.3
    
    def recursive_character_text_splitter(self, text: str, separators: List[str] = None) -> List[str]:
        """Modern recursive text splitting approach (LangChain-style)."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_text(text: str, separators: List[str]) -> List[str]:
            if not separators:
                return [text]
            
            separator = separators[0]
            if separator == "":
                # Character-level splitting
                return list(text)
            
            splits = text.split(separator)
            if len(splits) == 1:
                # Separator not found, try next separator
                return split_text(text, separators[1:])
            
            # Process splits recursively
            final_splits = []
            for split in splits:
                if self.count_tokens(split) <= self.chunk_size:
                    final_splits.append(split)
                else:
                    # Split is still too large, recursively split further
                    sub_splits = split_text(split, separators[1:])
                    final_splits.extend(sub_splits)
            
            return final_splits
        
        return split_text(text, separators)
    
    def create_chunks_with_overlap(self, text: str) -> List[Dict]:
        """Create chunks with token-based overlap (modern approach)."""
        # First, split into sections by double newlines
        sections = text.split('\n\n')
        all_chunks = []
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Use recursive splitting for this section
            splits = self.recursive_character_text_splitter(section)
            
            # Combine splits into properly sized chunks with overlap
            current_chunk = ""
            current_tokens = 0
            
            for split in splits:
                split = split.strip()
                if not split:
                    continue
                
                split_tokens = self.count_tokens(split)
                
                # Check if adding this split would exceed chunk size
                if current_tokens + split_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    if current_tokens >= self.min_chunk_tokens:
                        chunk_dict = self._create_chunk_dict(current_chunk, section_idx)
                        all_chunks.append(chunk_dict)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + split if overlap_text else split
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    # Add to current chunk
                    current_chunk = current_chunk + " " + split if current_chunk else split
                    current_tokens = self.count_tokens(current_chunk)
            
            # Add final chunk
            if current_chunk.strip() and current_tokens >= self.min_chunk_tokens:
                chunk_dict = self._create_chunk_dict(current_chunk, section_idx)
                all_chunks.append(chunk_dict)
        
        return all_chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for chunk continuity."""
        tokens = self.count_tokens(text)
        if tokens <= self.chunk_overlap:
            return text
        
        # Take last portion for overlap
        words = text.split()
        overlap_words = []
        overlap_tokens = 0
        
        for word in reversed(words):
            word_tokens = self.count_tokens(word)
            if overlap_tokens + word_tokens <= self.chunk_overlap:
                overlap_words.insert(0, word)
                overlap_tokens += word_tokens
            else:
                break
        
        return " ".join(overlap_words)
    
    def _create_chunk_dict(self, content: str, section_idx: int) -> Dict:
        """Create standardized chunk dictionary."""
        return {
            "content": content.strip(),
            "tokens": self.count_tokens(content),
            "char_count": len(content),
            "section_idx": section_idx,
            "chunk_type": "regular"
        }
    
    def create_parent_child_chunks(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Create parent-child chunk hierarchy for better retrieval."""
        # Create large parent chunks
        parent_chunks = []
        words = text.split()
        
        current_parent = ""
        current_tokens = 0
        parent_idx = 0
        
        for word in words:
            word_tokens = self.count_tokens(word)
            
            if current_tokens + word_tokens > self.parent_chunk_size and current_parent:
                # Save parent chunk
                parent_chunks.append({
                    "content": current_parent.strip(),
                    "tokens": current_tokens,
                    "parent_id": parent_idx,
                    "chunk_type": "parent"
                })
                parent_idx += 1
                current_parent = word
                current_tokens = word_tokens
            else:
                current_parent += " " + word if current_parent else word
                current_tokens += word_tokens
        
        # Add final parent
        if current_parent.strip():
            parent_chunks.append({
                "content": current_parent.strip(),
                "tokens": current_tokens,
                "parent_id": parent_idx,
                "chunk_type": "parent"
            })
        
        # Create child chunks from each parent
        child_chunks = []
        child_id = 0
        
        for parent in parent_chunks:
            children = self.create_chunks_with_overlap(parent["content"])
            for child in children:
                child["parent_id"] = parent["parent_id"]
                child["child_id"] = child_id
                child["chunk_type"] = "child"
                child_chunks.append(child)
                child_id += 1
        
        return parent_chunks, child_chunks
    
    def create_sentence_windows(self, chunks: List[Dict], full_text: str) -> List[Dict]:
        """Create sentence windows for better context retrieval."""
        # Split full text into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Map chunks to sentence positions
        enhanced_chunks = []
        
        for chunk in chunks:
            chunk_content = chunk["content"]
            
            # Find sentences that overlap with this chunk
            chunk_sentences = []
            for i, sentence in enumerate(sentences):
                if sentence in chunk_content or chunk_content in sentence:
                    chunk_sentences.append((i, sentence))
            
            if chunk_sentences:
                # Get sentence window around the chunk
                center_idx = chunk_sentences[len(chunk_sentences)//2][0]
                start_idx = max(0, center_idx - self.sentence_window_size)
                end_idx = min(len(sentences), center_idx + self.sentence_window_size + 1)
                
                window_sentences = sentences[start_idx:end_idx]
                window_content = " ".join(window_sentences)
                
                # Create enhanced chunk
                enhanced_chunk = chunk.copy()
                enhanced_chunk["window_content"] = window_content
                enhanced_chunk["window_tokens"] = self.count_tokens(window_content)
                enhanced_chunk["sentence_range"] = (start_idx, end_idx)
                enhanced_chunks.append(enhanced_chunk)
            else:
                # Keep original chunk if no sentence mapping found
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def extract_and_process_text(self, pdf_path: Path) -> Dict:
        """Extract and process text with modern techniques."""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"success": False, "message": f"No text extracted from {pdf_path.name}"}
        
        # Clean and preprocess
        text = self._clean_text(text)
        
        # Extract metadata
        metadata = self._extract_metadata(text)
        
        # Create parent-child chunks
        parent_chunks, child_chunks = self.create_parent_child_chunks(text)
        
        # Add sentence windows to child chunks
        enhanced_chunks = self.create_sentence_windows(child_chunks, text)
        
        # Add source information
        for chunk in enhanced_chunks:
            chunk["source"] = pdf_path.name
        
        for chunk in parent_chunks:
            chunk["source"] = pdf_path.name
        
        return {
            "success": True,
            "text": text,
            "metadata": metadata,
            "parent_chunks": parent_chunks,
            "child_chunks": enhanced_chunks,
            "total_tokens": self.count_tokens(text)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better processing with improved section handling."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)
        
        # Fix sentence boundaries
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up common document structure artifacts
        # Remove standalone section markers that might confuse processing
        text = re.sub(r'\b(Appendix|Index|Bibliography|References|Summary|Abstract|Introduction|Overview)\s*:\s*(?=\w)', 
                     r'\1 - ', text, flags=re.IGNORECASE)
        
        # Fix numbered lists that might get confused with section headers
        text = re.sub(r'^\s*(\d+)\.\s*([A-Z][a-z]+)\s*:\s*', r'\1. \2: ', text, flags=re.MULTILINE)
        
        # Remove page numbers and headers/footers that might interfere
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Remove standalone page numbers
        text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata using modern patterns."""
        metadata = {}
        
        # Basic statistics
        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())
        metadata["token_count"] = self.count_tokens(text)
        
        # Extract patterns
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            metadata["emails"] = list(set(emails[:10]))
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if urls:
            metadata["urls"] = list(set(urls[:5]))
        
        # Extract phone numbers
        phones = re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text)
        if phones:
            metadata["phone_numbers"] = list(set(phones[:5]))
        
        # Extract dates
        dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
        if dates:
            metadata["dates"] = list(set(dates[:10]))
        
        # Extract entities (capitalized terms)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if entities:
            common_words = {'The', 'This', 'That', 'They', 'These', 'Those', 'And', 'But', 'Or', 'For', 'In', 'On', 'At', 'To', 'From', 'With', 'By', 'Of', 'As', 'Is', 'Was', 'Are', 'Were', 'Be', 'Been', 'Being'}
            unique_entities = [e for e in set(entities) if e not in common_words and len(e) > 2]
            if unique_entities:
                metadata["entities"] = sorted(unique_entities[:25])
        
        # Extract keywords using frequency analysis
        keywords = self._extract_keywords(text)
        if keywords:
            metadata["keywords"] = keywords
        
        # Detect content type
        metadata["content_type"] = self._detect_content_type(text)
        
        return metadata
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF-like approach."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stop words to filter
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'a', 'an', 'the', 'you', 'your', 'they', 'their', 'we', 'our', 'he', 'his', 'she', 'her',
            'it', 'its', 'me', 'my', 'us', 'him', 'them', 'all', 'any', 'some', 'each', 'every',
            'other', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
        }
        
        # Filter and count
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        word_freq = Counter(filtered_words)
        
        # Return top keywords
        return [word for word, freq in word_freq.most_common(20) if freq > 1]
    
    def _detect_content_type(self, text: str) -> str:
        """Detect document content type."""
        text_lower = text.lower()
        
        patterns = {
            "resume": ["resume", "cv", "curriculum vitae", "experience", "education", "skills", "employment", "work history"],
            "legal": ["contract", "agreement", "terms", "conditions", "legal", "whereas", "party", "jurisdiction"],
            "financial": ["financial", "revenue", "profit", "budget", "expense", "investment", "earnings", "fiscal"],
            "technical": ["technical", "specification", "manual", "guide", "documentation", "api", "system", "software"],
            "report": ["report", "analysis", "summary", "findings", "conclusion", "executive summary", "overview"],
            "academic": ["research", "study", "paper", "journal", "university", "professor", "thesis", "dissertation"],
            "medical": ["medical", "health", "patient", "diagnosis", "treatment", "clinical", "hospital", "doctor"]
        }
        
        scores = {}
        for doc_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[doc_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "general"
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """Create embeddings in batches with proper token handling."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Truncate texts to fit embedding model limits
                truncated_batch = []
                for text in batch:
                    if self.count_tokens(text) > self.max_tokens_per_chunk:
                        # Truncate to fit
                        words = text.split()
                        truncated_text = ""
                        tokens = 0
                        
                        for word in words:
                            word_tokens = self.count_tokens(word)
                            if tokens + word_tokens <= self.max_tokens_per_chunk - 10:  # Safety margin
                                truncated_text += " " + word if truncated_text else word
                                tokens += word_tokens
                            else:
                                break
                        
                        truncated_batch.append(truncated_text)
                    else:
                        truncated_batch.append(text)
                
                # Create embeddings
                response = self.openai_client.embeddings.create(
                    model=self.text_embedding_model,
                    input=truncated_batch
                )
                
                batch_embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
                embeddings.extend(batch_embeddings)
                
                # Progress update
                progress = min(i + batch_size, len(texts))
                if progress % 100 == 0 or progress == len(texts):
                    st.write(f"✅ Created embeddings for {progress}/{len(texts)} chunks")
                
            except Exception as e:
                st.warning(f"Error creating embeddings for batch {i//batch_size + 1}: {str(e)}")
                # Add zero vectors for failed batch
                for _ in batch:
                    embeddings.append(np.zeros(3072, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
    
    def hyde_query_expansion(self, query: str) -> str:
        """HyDE: Generate hypothetical document for better retrieval."""
        if not self.use_hyde:
            return query
        
        try:
            hyde_prompt = f"""Write a detailed, informative paragraph that would answer this question: "{query}"
            
Write as if you're providing the actual answer from a document. Be specific and include relevant details, terminology, and context that would typically appear in documents about this topic.

Hypothetical answer:"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": hyde_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            
            # Combine original query with hypothetical document
            expanded_query = f"{query} {hypothetical_doc}"
            return expanded_query
            
        except Exception as e:
            st.warning(f"HyDE generation failed: {e}")
            return query
    
    def expand_query(self, query: str) -> str:
        """Expand query with related terms."""
        if not self.use_query_expansion:
            return query
        
        try:
            expansion_prompt = f"""Given this search query: "{query}"

Generate 3-5 related terms, synonyms, or phrases that would help find relevant documents. Focus on:
- Alternative terminology
- Related concepts  
- Professional/technical terms
- Common synonyms

Return only the additional terms, separated by spaces:"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.3,
                max_tokens=50
            )
            
            expansion_terms = response.choices[0].message.content.strip()
            
            # Combine with original query
            expanded_query = f"{query} {expansion_terms}"
            return expanded_query
            
        except Exception as e:
            st.warning(f"Query expansion failed: {e}")
            return query
    
    def maximal_marginal_relevance(self, query_embedding: np.ndarray, 
                                  embeddings: np.ndarray, 
                                  metadata: List[Dict], 
                                  k: int, 
                                  lambda_param: float = 0.7) -> List[Dict]:
        """MMR for diverse result selection."""
        if not self.use_mmr or len(embeddings) == 0:
            return []
        
        # Calculate similarities to query
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        selected_indices = []
        remaining_indices = list(range(len(embeddings)))
        
        # Select first document (highest similarity)
        if remaining_indices:
            best_idx = remaining_indices[np.argmax(similarities[remaining_indices])]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Select subsequent documents using MMR
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance score
                relevance = similarities[idx]
                
                # Diversity score (max similarity to already selected)
                if selected_indices:
                    selected_embeddings = embeddings[selected_indices]
                    current_embedding = embeddings[idx]
                    
                    diversities = np.dot(selected_embeddings, current_embedding) / (
                        np.linalg.norm(selected_embeddings, axis=1) * np.linalg.norm(current_embedding)
                    )
                    max_similarity = np.max(diversities)
                else:
                    max_similarity = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            if mmr_scores:
                best_mmr_idx = np.argmax(mmr_scores)
                selected_idx = remaining_indices[best_mmr_idx]
                selected_indices.append(selected_idx)
                remaining_indices.remove(selected_idx)
        
        # Return selected documents
        results = []
        for idx in selected_indices:
            if idx < len(metadata):
                result = metadata[idx].copy()
                result["similarity_score"] = float(similarities[idx])
                result["mmr_selected"] = True
                results.append(result)
        
        return results
    
    def create_vector_store(self) -> Dict:
        """Create modern vector store with latest techniques."""
        pdf_files = list(self.data_folder.glob("*.pdf"))
        
        if not pdf_files:
            return {"success": False, "message": "No PDF files found in data folder"}
        
        max_files = 25
        if len(pdf_files) > max_files:
            st.warning(f"Processing first {max_files} of {len(pdf_files)} files")
            pdf_files = pdf_files[:max_files]
        
        all_child_chunks = []
        all_parent_chunks = []
        processing_stats = {"files": 0, "chunks": 0, "tokens": 0}
        
        # Progress tracking
        main_progress = st.progress(0)
        main_status = st.empty()
        start_time = time.time()
        
        main_status.text("📄 Processing PDFs with modern RAG techniques...")
        
        # Process each PDF
        for file_idx, pdf_file in enumerate(pdf_files):
            st.write(f"\n### Processing {pdf_file.name}")
            
            result = self.extract_and_process_text(pdf_file)
            
            if not result["success"]:
                st.warning(result["message"])
                continue
            
            # Add file metadata
            for chunk in result["child_chunks"]:
                chunk["file_index"] = file_idx
                chunk["document_metadata"] = result["metadata"]
            
            for chunk in result["parent_chunks"]:
                chunk["file_index"] = file_idx
                chunk["document_metadata"] = result["metadata"]
            
            all_child_chunks.extend(result["child_chunks"])
            all_parent_chunks.extend(result["parent_chunks"])
            
            processing_stats["files"] += 1
            processing_stats["chunks"] += len(result["child_chunks"])
            processing_stats["tokens"] += result["total_tokens"]
            
            st.success(f"✅ Processed {pdf_file.name}: {len(result['child_chunks'])} chunks, {result['total_tokens']} tokens")
            
            # Update progress
            file_progress = (file_idx + 1) / len(pdf_files) * 0.4
            main_progress.progress(file_progress)
        
        if not all_child_chunks:
            return {"success": False, "message": "No content extracted from any PDF files"}
        
        # Limit chunks for performance
        max_total_chunks = self.max_chunks_per_file * len(pdf_files)
        if len(all_child_chunks) > max_total_chunks:
            st.info(f"Limiting to {max_total_chunks} chunks for optimal performance")
            # Sort by token count (prefer longer, more informative chunks)
            all_child_chunks.sort(key=lambda x: x.get("tokens", 0), reverse=True)
            all_child_chunks = all_child_chunks[:max_total_chunks]
            processing_stats["chunks"] = len(all_child_chunks)
        
        main_status.text("🧠 Creating modern embeddings...")
        main_progress.progress(0.4)
        
        # Create embeddings for child chunks (used for retrieval)
        try:
            # Prepare texts for embedding (use window content if available)
            embed_texts = []
            for chunk in all_child_chunks:
                text = chunk.get("window_content", chunk["content"])
                embed_texts.append(text)
            
            embeddings = self.create_embeddings_batch(embed_texts)
            main_progress.progress(0.7)
        except Exception as e:
            return {"success": False, "message": f"Error creating embeddings: {str(e)}"}
        
        main_status.text("📊 Building modern search indices...")
        
        # Create search indices
        try:
            indices = self.create_search_indices(embeddings, all_child_chunks, all_parent_chunks)
            main_progress.progress(0.9)
        except Exception as e:
            return {"success": False, "message": f"Error creating indices: {str(e)}"}
        
        main_status.text("💾 Saving modern vector store...")
        
        # Save to disk
        try:
            self.save_indices(indices)
        except Exception as e:
            return {"success": False, "message": f"Error saving: {str(e)}"}
        
        # Complete
        total_time = time.time() - start_time
        main_progress.progress(1.0)
        main_status.text(f"✅ Modern RAG system created in {total_time:.1f}s!")
        
        techniques = ["Token-based Chunking", "Sentence Windows", "Parent-Child"]
        if self.use_hyde:
            techniques.append("HyDE")
        if self.use_query_expansion:
            techniques.append("Query Expansion") 
        if self.use_mmr:
            techniques.append("MMR")
        if self.use_bm25:
            techniques.append("BM25")
        
        return {
            "success": True,
            "message": f"Modern RAG created: {processing_stats['chunks']} chunks from {processing_stats['files']} files ({processing_stats['tokens']:,} tokens)",
            "chunks": processing_stats["chunks"],
            "files": processing_stats["files"],
            "tokens": processing_stats["tokens"],
            "time": total_time,
            "techniques": techniques
        }
    
    def create_search_indices(self, embeddings: np.ndarray, child_chunks: List[Dict], parent_chunks: List[Dict]) -> Dict:
        """Create optimized search indices."""
        if not child_chunks:
            return {"vector_index": None, "bm25_index": None, "metadata": [], "parent_metadata": []}
        
        # Prepare child metadata (for retrieval)
        child_metadata = []
        for i, chunk in enumerate(child_chunks):
            metadata = {
                "content": chunk["content"],
                "source": chunk.get("source", ""),
                "chunk_id": i,
                "tokens": chunk.get("tokens", 0),
                "char_count": chunk.get("char_count", 0),
                "chunk_type": chunk.get("chunk_type", "child"),
                "parent_id": chunk.get("parent_id", -1),
                "child_id": chunk.get("child_id", i),
                "window_content": chunk.get("window_content", ""),
                "sentence_range": chunk.get("sentence_range", (0, 0)),
                "document_metadata": chunk.get("document_metadata", {}),
                "file_index": chunk.get("file_index", 0)
            }
            child_metadata.append(metadata)
        
        # Prepare parent metadata
        parent_metadata = []
        for chunk in parent_chunks:
            metadata = {
                "content": chunk["content"],
                "source": chunk.get("source", ""),
                "parent_id": chunk.get("parent_id", -1),
                "tokens": chunk.get("tokens", 0),
                "chunk_type": "parent",
                "document_metadata": chunk.get("document_metadata", {}),
                "file_index": chunk.get("file_index", 0)
            }
            parent_metadata.append(metadata)
        
        # Create FAISS vector index
        st.info("🔄 Creating optimized FAISS vector index...")
        vector_index = self.create_faiss_index(embeddings)
        
        # Create BM25 index
        bm25_index = None
        tokenized_docs = None
        if self.use_bm25:
            st.info("🔄 Creating BM25 index...")
            texts = [chunk["content"] for chunk in child_chunks]
            tokenized_docs = [self._tokenize_for_bm25(text) for text in texts]
            bm25_index = BM25Okapi(tokenized_docs)
        
        return {
            "vector_index": vector_index,
            "bm25_index": bm25_index,
            "metadata": child_metadata,
            "parent_metadata": parent_metadata,
            "tokenized_docs": tokenized_docs
        }
    
    def create_faiss_index(self, embeddings: np.ndarray):
        """Create optimized FAISS index."""
        if embeddings.size == 0:
            return None
        
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Choose index type
        if n_vectors < 1000:
            index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        elif n_vectors < 50000:
            nlist = min(int(math.sqrt(n_vectors)), 1000)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
        else:
            # Large dataset - use PQ compression
            nlist = min(int(math.sqrt(n_vectors)), 2000)
            m = 16  # Subquantizers
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            index.train(embeddings)
        
        # GPU acceleration
        if self.use_gpu and n_vectors > 1000:
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu_index.add(embeddings)
                index = faiss.index_gpu_to_cpu(gpu_index)
                st.success("🚀 Used GPU acceleration for indexing!")
            except Exception as e:
                st.write(f"GPU indexing failed, using CPU: {str(e)}")
                index.add(embeddings)
        else:
            index.add(embeddings)
        
        return index
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple but effective tokenization
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [token for token in text.split() if len(token) > 2]
        return tokens
    
    def modern_search(self, query: str, k: int = 8) -> List[Dict]:
        """Perform modern RAG search with latest techniques."""
        indices = self.load_indices()
        if not indices or not indices["metadata"]:
            return []
        
        vector_index = indices["vector_index"]
        bm25_index = indices["bm25_index"]
        metadata = indices["metadata"]
        parent_metadata = indices.get("parent_metadata", [])
        
        # Query enhancement
        enhanced_query = self.hyde_query_expansion(query)
        expanded_query = self.expand_query(enhanced_query)
        
        all_results = []
        
        # Vector search with enhanced query
        if vector_index:
            vector_results = self._modern_vector_search(expanded_query, vector_index, metadata, k * 3)
            all_results.extend(vector_results)
        
        # BM25 search
        if bm25_index and self.use_bm25:
            bm25_results = self._bm25_search(query, bm25_index, metadata, k * 2)
            all_results.extend(bm25_results)
        
        # Apply MMR for diversity
        if self.use_mmr and vector_index and all_results:
            try:
                # Create query embedding for MMR
                response = self.openai_client.embeddings.create(
                    model=self.text_embedding_model,
                    input=expanded_query[:8000]
                )
                query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
                
                # Get embeddings for all results
                result_indices = [r["chunk_id"] for r in all_results if "chunk_id" in r]
                if result_indices:
                    # Get embeddings from FAISS index
                    result_embeddings = np.array([vector_index.reconstruct(i) for i in result_indices])
                    
                    # Apply MMR
                    mmr_results = self.maximal_marginal_relevance(
                        query_embedding, result_embeddings, all_results, k, self.mmr_diversity_score
                    )
                    
                    if mmr_results:
                        return mmr_results
            except Exception as e:
                st.warning(f"MMR failed, using standard ranking: {e}")
        
        # Fallback: combine and rank results
        if all_results:
            combined_results = self._combine_results(all_results, k)
            return combined_results[:k]
        
        return []
    
    def _modern_vector_search(self, query: str, index, metadata: List[Dict], k: int) -> List[Dict]:
        """Modern vector search with enhancements."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.text_embedding_model,
                input=query[:8000]
            )
            
            query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            search_k = min(k, index.ntotal)
            scores, indices = index.search(query_embedding.reshape(1, -1), search_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata) and idx >= 0:
                    result = metadata[idx].copy()
                    result["similarity_score"] = float(scores[0][i])  # Inner product score
                    result["search_type"] = "vector"
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Vector search error: {str(e)}")
            return []
    
    def _bm25_search(self, query: str, bm25_index, metadata: List[Dict], k: int) -> List[Dict]:
        """BM25 search."""
        try:
            query_tokens = self._tokenize_for_bm25(query)
            if not query_tokens:
                return []
            
            scores = bm25_index.get_scores(query_tokens)
            scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for idx, score in scored_indices[:k]:
                if idx < len(metadata):
                    result = metadata[idx].copy()
                    result["bm25_score"] = float(score)
                    result["search_type"] = "bm25"
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"BM25 search error: {str(e)}")
            return []
    
    def _combine_results(self, results: List[Dict], k: int) -> List[Dict]:
        """Combine and rank results."""
        # Remove duplicates based on chunk_id
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            chunk_id = result.get("chunk_id", -1)
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        # Sort by best available score
        def get_score(result):
            if "similarity_score" in result:
                return result["similarity_score"]
            elif "bm25_score" in result:
                return result["bm25_score"] / 10.0  # Normalize BM25
            return 0
        
        unique_results.sort(key=get_score, reverse=True)
        return unique_results
    
    def get_parent_context(self, child_result: Dict, parent_metadata: List[Dict]) -> str:
        """Get parent context for a child chunk."""
        parent_id = child_result.get("parent_id", -1)
        if parent_id >= 0:
            for parent in parent_metadata:
                if parent.get("parent_id") == parent_id:
                    return parent.get("content", "")
        return child_result.get("content", "")
    
    def display_modern_results(self, results: List[Dict]):
        """Display modern search results with context."""
        if not results:
            st.info("No results found.")
            return
        
        indices = self.load_indices()
        parent_metadata = indices.get("parent_metadata", [])
        
        for i, result in enumerate(results, 1):
            source = result.get('source', 'Unknown')
            
            # Create enhanced title
            title = f"📄 Result {i} - {source}"
            techniques = []
            if result.get("mmr_selected"):
                techniques.append("MMR")
            if result.get("search_type"):
                techniques.append(result["search_type"].upper())
            
            if techniques:
                title += f" ({', '.join(techniques)})"
            
            with st.expander(title, expanded=(i <= 3)):
                # Display main content
                content = result.get("content", "")
                st.write("**Chunk Content:**")
                st.write(content)
                
                # Show window content if different
                window_content = result.get("window_content", "")
                if window_content and window_content != content:
                    st.write("**Sentence Window Context:**")
                    st.write(window_content)
                
                # Show parent context if available (without nested expander)
                if parent_metadata:
                    parent_context = self.get_parent_context(result, parent_metadata)
                    if parent_context and parent_context != content and len(parent_context) > len(content) * 1.5:
                        st.write("**Full Parent Context:**")
                        # Show first part of parent context with option to show more
                        preview_length = 500
                        if len(parent_context) > preview_length:
                            st.write(parent_context[:preview_length] + "...")
                            if st.button(f"📖 Show Full Parent Context", key=f"parent_{i}"):
                                st.write("**Complete Parent Context:**")
                                st.write(parent_context)
                        else:
                            st.write(parent_context)
                
                # Show metadata
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result.get("tokens"):
                        st.caption(f"**Tokens:** {result['tokens']}")
                    if result.get("chunk_type"):
                        st.caption(f"**Type:** {result['chunk_type']}")
                
                with col2:
                    # Show scores
                    if result.get("similarity_score") is not None:
                        st.caption(f"**Similarity:** {result['similarity_score']:.4f}")
                    if result.get("bm25_score"):
                        st.caption(f"**BM25:** {result['bm25_score']:.2f}")
                
                with col3:
                    # Show IDs
                    if result.get("parent_id", -1) >= 0:
                        st.caption(f"**Parent ID:** {result['parent_id']}")
                    if result.get("sentence_range"):
                        start, end = result["sentence_range"]
                        st.caption(f"**Sentences:** {start}-{end}")
                
                st.markdown("---")
    
    def save_indices(self, indices: Dict):
        """Save modern indices to disk."""
        try:
            # Save vector index
            if indices["vector_index"]:
                vector_path = self.vector_store_path / "faiss_index.bin"
                faiss.write_index(indices["vector_index"], str(vector_path))
            
            # Save metadata and other components
            metadata_path = self.vector_store_path / "modern_metadata.pkl"
            save_data = {
                "metadata": indices["metadata"],
                "parent_metadata": indices.get("parent_metadata", []),
                "bm25_index": indices["bm25_index"],
                "tokenized_docs": indices.get("tokenized_docs"),
                "creation_time": time.time(),
                "system_info": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "use_bm25": self.use_bm25,
                    "use_hyde": self.use_hyde,
                    "use_query_expansion": self.use_query_expansion,
                    "use_mmr": self.use_mmr,
                    "text_embedding_model": self.text_embedding_model,
                    "system_type": "modern_rag",
                    "tokenizer": "tiktoken",
                    "parent_chunk_size": self.parent_chunk_size,
                    "sentence_window_size": self.sentence_window_size
                }
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            st.success("✅ Successfully saved modern RAG system")
            
        except Exception as e:
            st.error(f"Error saving indices: {str(e)}")
            raise
    
    def load_indices(self) -> Dict:
        """Load modern indices from disk."""
        vector_path = self.vector_store_path / "faiss_index.bin"
        metadata_path = self.vector_store_path / "modern_metadata.pkl"
        
        if not (vector_path.exists() and metadata_path.exists()):
            return {}
        
        try:
            # Load vector index
            vector_index = faiss.read_index(str(vector_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            
            return {
                "vector_index": vector_index,
                "bm25_index": data.get("bm25_index"),
                "metadata": data.get("metadata", []),
                "parent_metadata": data.get("parent_metadata", []),
                "tokenized_docs": data.get("tokenized_docs"),
                "system_info": data.get("system_info", {})
            }
            
        except Exception as e:
            st.error(f"Error loading search indices: {str(e)}")
            return {}
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF with quality improvements."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                max_pages = min(250, len(pdf_reader.pages))
                text_parts = []
                
                for i, page in enumerate(pdf_reader.pages[:max_pages]):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            cleaned_text = self._clean_text(text)
                            if len(cleaned_text) > 50:
                                text_parts.append(cleaned_text)
                    except Exception:
                        continue
                
                if not text_parts:
                    return ""
                
                full_text = "\n\n".join(text_parts)
                
                if len(pdf_reader.pages) > max_pages:
                    st.info(f"Processed {max_pages} of {len(pdf_reader.pages)} pages from {pdf_path.name}")
                
                return full_text
                
        except Exception as e:
            st.error(f"Error reading PDF {pdf_path.name}: {str(e)}")
            return ""


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


def categorize_with_direct_openai(openai_client, user_input, model_name):
    """Use direct OpenAI client to categorize user input with context awareness."""
    
    # Check conversation context
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
            elif "search" in category_lower or "document" in category_lower or "find" in category_lower or "knowledge" in category_lower or "what" in category_lower or "how" in category_lower:
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
    
    # Use Azure AI Inference connector for function calling
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


def create_modern_rag_ui(openai_client, model_name):
    """Modern RAG UI with latest techniques."""
    st.sidebar.markdown("---")
    st.sidebar.header("🚀 Modern RAG System")
    
    if not VECTOR_STORE_AVAILABLE:
        st.sidebar.error("❌ Dependencies not installed")
        st.sidebar.code("pip install faiss-cpu PyPDF2 rank-bm25 tiktoken")
        return None
    
    # Initialize manager
    if "modern_rag_manager" not in st.session_state:
        try:
            st.session_state.modern_rag_manager = ModernRAGManager(openai_client, model_name)
        except Exception as e:
            st.sidebar.error(f"Error initializing: {str(e)}")
            return None
    
    manager = st.session_state.modern_rag_manager
    
    # Check data folder
    data_folder = Path("data")
    pdf_files = list(data_folder.glob("*.pdf")) if data_folder.exists() else []
    
    st.sidebar.markdown(f"**PDF Files:** {len(pdf_files)}")
    
    if pdf_files:
        with st.sidebar.expander("📄 PDF Files", expanded=False):
            for pdf_file in pdf_files[:25]:
                file_size = pdf_file.stat().st_size / (1024 * 1024)
                status = "✅" if file_size <= 30 else "⚠️"
                st.write(f"{status} {pdf_file.name} ({file_size:.1f}MB)")
    else:
        st.sidebar.warning("⚠️ No PDF files found in 'data' folder")
    
    # Features
    st.sidebar.info("""⚡ **Modern RAG Features:**
- **Token-based chunking** with tiktoken
- **Recursive text splitting** (LangChain-style)
- **Parent-child chunks** for context
- **Sentence window** retrieval
- **HyDE** (Hypothetical Document Embeddings)
- **Query expansion** with LLM
- **MMR** (Maximal Marginal Relevance)
- **Hybrid search** (Vector + BM25)
- **Cosine similarity** with normalized embeddings
- **Support up to 25 PDF files**
- **GPU acceleration** when available
""")
    
    # Check if system exists
    indices = manager.load_indices()
    system_exists = bool(indices and indices.get("metadata"))
    
    if system_exists:
        st.sidebar.success("✅ Modern RAG System exists")
        
        metadata = indices.get("metadata", [])
        parent_metadata = indices.get("parent_metadata", [])
        
        if metadata:
            st.sidebar.markdown(f"**Child Chunks:** {len(metadata)}")
            st.sidebar.markdown(f"**Parent Chunks:** {len(parent_metadata)}")
            
            # Show statistics
            total_tokens = sum(item.get("tokens", 0) for item in metadata)
            st.sidebar.markdown(f"**Total Tokens:** {total_tokens:,}")
            
            sources = set(item.get("source", "") for item in metadata)
            st.sidebar.markdown(f"**Sources:** {len(sources)}")
            
            # Show techniques used
            system_info = indices.get("system_info", {})
            techniques = []
            if system_info.get("use_hyde"):
                techniques.append("HyDE")
            if system_info.get("use_query_expansion"):
                techniques.append("Query Expansion")
            if system_info.get("use_mmr"):
                techniques.append("MMR")
            if system_info.get("use_bm25"):
                techniques.append("BM25")
            
            if techniques:
                st.sidebar.markdown(f"**Techniques:** {', '.join(techniques)}")
    else:
        st.sidebar.info("ℹ️ No modern RAG system found")
    
    # Create/Recreate button
    button_text = "🔄 Recreate Modern RAG" if system_exists else "🚀 Create Modern RAG System"
    
    if st.sidebar.button(button_text, disabled=(len(pdf_files) == 0)):
        if len(pdf_files) == 0:
            st.sidebar.error("❌ No PDF files found")
        else:
            with st.sidebar:
                st.markdown("### 🚀 Creating Modern RAG System")
                try:
                    result = manager.create_vector_store()
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.info(f"🎯 Techniques: {', '.join(result['techniques'])}")
                        st.info(f"⏱️ Processing: {result.get('time', 0):.1f}s for {result.get('tokens', 0):,} tokens")
                        st.balloons()
                    else:
                        st.error(f"❌ {result['message']}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Search interface
    if system_exists:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Modern RAG Search")
        
        search_query = st.sidebar.text_input("Search:", placeholder="Ask about document content...")
        
        if st.sidebar.button("🚀 Modern Search") and search_query:
            with st.sidebar:
                with st.spinner("Searching with modern techniques..."):
                    try:
                        start_time = time.time()
                        results = manager.modern_search(search_query, k=6)
                        search_time = time.time() - start_time
                        
                        if results:
                            st.success(f"Found {len(results)} results in {search_time:.2f}s")
                            
                            # Show preview
                            for i, result in enumerate(results, 1):
                                source = result.get("source", "Unknown")
                                techniques = []
                                if result.get("mmr_selected"):
                                    techniques.append("MMR")
                                if result.get("search_type"):
                                    techniques.append(result["search_type"].upper())
                                
                                technique_str = f" ({', '.join(techniques)})" if techniques else ""
                                
                                with st.expander(f"Preview {i} - {source}{technique_str}", expanded=(i==1)):
                                    content = result.get("content", "")[:200]
                                    if len(result.get("content", "")) > 200:
                                        content += "..."
                                    
                                    st.write(f"**Text:** {content}")
                                    
                                    # Show scores
                                    if result.get("similarity_score") is not None:
                                        st.caption(f"Similarity: {result['similarity_score']:.4f}")
                                    if result.get("bm25_score"):
                                        st.caption(f"BM25: {result['bm25_score']:.2f}")
                        else:
                            st.info("No results found")
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
    
    return manager


def handle_modern_knowledge_base(query, manager):
    """Handle Knowledge Base queries with modern RAG techniques."""
    st.markdown("🚀 **Modern RAG Knowledge Base Agent** is assisting you.")
    
    # Check if system exists
    indices = manager.load_indices()
    if not indices or not indices.get("metadata"):
        st.error("❌ No modern RAG system found. Please create one first.")
        return "No system found."
    
    # Show system status
    metadata = indices.get("metadata", [])
    parent_metadata = indices.get("parent_metadata", [])
    total_tokens = sum(item.get("tokens", 0) for item in metadata)
    
    system_info = indices.get("system_info", {})
    techniques = []
    if system_info.get("use_hyde"):
        techniques.append("HyDE")
    if system_info.get("use_query_expansion"):
        techniques.append("Query Expansion")
    if system_info.get("use_mmr"):
        techniques.append("MMR")
    if system_info.get("use_bm25"):
        techniques.append("BM25")
    
    st.success(f"🚀 Using Modern RAG: {len(metadata)} chunks, {len(parent_metadata)} parents, {total_tokens:,} tokens ({', '.join(techniques)})")
    
    # Process the query
    with st.spinner("🔍 Searching with modern RAG techniques..."):
        try:
            start_time = time.time()
            
            # Determine search scope
            general_queries = ["what information", "what's in", "what is in", "what does the document contain", 
                             "what topics", "summarize", "overview", "what's available"]
            is_general_query = any(phrase in query.lower() for phrase in general_queries)
            
            # Search with appropriate scope
            if is_general_query:
                results = manager.modern_search(query, k=10)
            else:
                results = manager.modern_search(query, k=8)
            
            search_time = time.time() - start_time
            
            if results:
                # Display results
                st.markdown(f"### 🚀 Modern RAG Search Results")
                st.write(f"*Found {len(results)} relevant sections in {search_time:.2f}s*")
                
                # Show techniques used
                techniques_used = set()
                for result in results:
                    if result.get("mmr_selected"):
                        techniques_used.add("MMR")
                    if result.get("search_type"):
                        techniques_used.add(result["search_type"].upper())
                
                if techniques_used:
                    st.info(f"🎯 Applied techniques: {', '.join(techniques_used)}")
                
                manager.display_modern_results(results)
                
                # Generate AI summary
                try:
                    openai_client = st.session_state.get('openai_client')
                    model_name = os.getenv("DEPLOYMENT_NAME")
                    
                    if openai_client and model_name:
                        # Create context from results
                        context_parts = []
                        for result in results[:5]:  # Use top 5 results
                            # Use window content if available for better context
                            content = result.get("window_content", result.get("content", ""))
                            context_parts.append(content)
                        
                        context = "\n\n".join(context_parts)
                        
                        if context:
                            if is_general_query:
                                summary_prompt = f"""Based on the following content from documents, provide a comprehensive overview of what information is available.

User Question: {query}

Document Content:
{context}

Provide a helpful summary of the main topics, types of information, and key content available."""
                            else:
                                summary_prompt = f"""Based on the following content from documents, answer the user's question directly and comprehensively.

User Question: {query}

Document Content:
{context}

Provide a clear, detailed answer based on the content above."""

                            response = openai_client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": summary_prompt}],
                                temperature=0.05,
                                max_tokens=800
                            )
                            
                            ai_summary = response.choices[0].message.content.strip()
                            
                            # Display AI summary
                            st.markdown("### 🤖 AI Summary")
                            st.write(ai_summary)
                            
                            final_result = f"{ai_summary}\n\n*🚀 Modern RAG search completed in {search_time:.2f}s*"
                            return final_result
                            
                except Exception as e:
                    st.warning(f"AI summary generation failed: {str(e)}")
                
                return f"Found {len(results)} relevant sections with modern RAG techniques in {search_time:.2f}s"
                
            else:
                # Show sample content if no results
                try:
                    if metadata:
                        st.markdown("No specific matches found, but here's sample content from your modern RAG system:")
                        
                        sample_items = metadata[:4]
                        for i, item in enumerate(sample_items, 1):
                            with st.expander(f"Sample {i} (from {item.get('source', 'Unknown')})", expanded=(i==1)):
                                content = item.get('content', '')[:400]
                                if len(item.get('content', '')) > 400:
                                    content += "..."
                                st.write(content)
                                
                                # Show metadata
                                if item.get('tokens'):
                                    st.caption(f"Tokens: {item['tokens']}, Type: {item.get('chunk_type', 'child')}")
                        
                        return f"Sample content shown. Total: {len(metadata)} chunks with {total_tokens:,} tokens."
                    else:
                        return "No relevant information found. Try rephrasing your question."
                except:
                    return "No relevant information found. Try rephrasing your question."
                
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                return "Unable to search due to AI service configuration issue."
            elif "401" in error_msg:
                return "Unable to search due to authentication issues."
            else:
                return f"Search error: {error_msg}"


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
        st.markdown("🚀 **Modern RAG Knowledge Base Agent** is now assisting you.")
        
        # Check if modern manager exists
        if "modern_rag_manager" not in st.session_state or st.session_state.modern_rag_manager is None:
            st.error("❌ Modern RAG not initialized. Please create a system first.")
            if st.button("Return to Main Agent"):
                return True
            return False
        
        manager = st.session_state.modern_rag_manager
        indices = manager.load_indices()
        
        if not indices or not indices.get("metadata"):
            st.error("❌ No modern RAG system found. Please create one first.")
            if st.button("Return to Main Agent"):
                return True
            return False
        
        # Show system status
        system_info = indices.get("system_info", {})
        techniques = []
        if system_info.get("use_hyde"):
            techniques.append("HyDE")
        if system_info.get("use_query_expansion"):
            techniques.append("Query Exp")
        if system_info.get("use_mmr"):
            techniques.append("MMR")
        if system_info.get("use_bm25"):
            techniques.append("BM25")
        
        st.success(f"🚀 Using Modern RAG ({', '.join(techniques)})")
        
        # Check for original query from routing
        original_query = st.session_state.get('original_user_query', '')
        
        if original_query and not st.session_state.get('kb_query_processed', False):
            # Process the original query
            result = handle_modern_knowledge_base(original_query, manager)
            
            st.session_state.agent_complete = True
            st.session_state.agent_result = result
            st.session_state.kb_query_processed = True
            return True
        
        else:
            # Show form for additional queries
            with st.form("modern_rag_query_form"):
                st.markdown("**Enter your question:**")
                query = st.text_area("Query:", placeholder="Ask about document content...")
                
                submitted = st.form_submit_button("🚀 Modern RAG Search", type="primary")
                
                if submitted and query.strip():
                    result = handle_modern_knowledge_base(query.strip(), manager)
                    
                    st.session_state.agent_complete = True
                    st.session_state.agent_result = result
                    return True
        
        # Option to ask another question
        if st.button("🔄 Ask Another Question"):
            st.session_state.kb_query_processed = False
            st.rerun()
    
    return False


def main():
    st.set_page_config(
        page_title="Modern RAG Multi-Agent Assistant",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Modern RAG Multi-Agent Assistant")
    st.markdown("Featuring **latest RAG techniques**: **Token-based chunking**, **HyDE**, **Query expansion**, **MMR**, **Sentence windows**, **Parent-child chunks**, **Hybrid search**, and **Advanced retrieval** for superior document understanding.")

    # Initialize chatbot
    kernel, chat_service, openai_client = initialize_chatbot()
    
    if kernel is None:
        st.stop()
    
    # Show debug panel
    endpoint = os.getenv("ENDPOINT_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    model_name = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")
    
    if endpoint and "/openai/deployments/" in endpoint:
        base_endpoint = endpoint.split("/openai/deployments/")[0]
    else:
        base_endpoint = endpoint
    
    show_debug_panel(openai_client, model_name, base_endpoint, api_version, api_key)

    # Add modern RAG UI to sidebar
    modern_manager = create_modern_rag_ui(openai_client, model_name)
    
    # Store managers in session state
    st.session_state.openai_client = openai_client
    st.session_state.modern_rag_manager = modern_manager

    # Initialize session state
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
            # Agent completed
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
            st.rerun()
            
    elif st.session_state.waiting_for_confirmation:
        # Show confirmation buttons
        st.markdown("### Confirmation Required")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Connect with Specialized Agent", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Yes"})
                st.session_state.agent_active = True
                st.rerun()
        
        with col2:
            if st.button("❌ No, cancel request", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "No"})
                st.session_state.messages.append({"role": "assistant", "content": "No problem! Feel free to ask about something else."})
                
                # Reset confirmation state
                st.session_state.waiting_for_confirmation = False
                st.session_state.pending_category = None
                st.session_state.pending_function = None
                st.session_state.kb_query_processed = False
                st.session_state.original_user_query = ""
                
                st.rerun()
    
    else:
        # Regular chat input
        if prompt := st.chat_input("What can I help you with today?"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Analyze request
            with st.spinner("🔍 Analyzing your request..."):
                category = categorize_with_direct_openai(openai_client, prompt, model_name)
            
            # Define valid categories
            valid_categories = {
                "Report Sick": ("report_sick", "report that you're not feeling well"),
                "Report Fatigue": ("report_fatigue", "report fatigue or tiredness"), 
                "Book Hotel": ("book_hotel", "book a hotel"),
                "Book Limo": ("book_limo", "book a limo/transportation"),
                "Search Knowledge Base": ("search_knowledge_base", "search documents with modern RAG techniques")
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
                error_message = "🤔 I'm sorry, I couldn't categorize your request. I can help you with reporting sickness, reporting fatigue, booking hotels, booking transportation, or searching your documents with modern RAG techniques. Could you please clarify what you need help with?"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            st.rerun()

    # Enhanced sidebar with modern RAG info
    with st.sidebar:
        st.header("🚀 Modern RAG Multi-Agent Services")
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
        
        🚀 **Modern RAG Search**
        - **Token-based chunking** with tiktoken
        - **HyDE** (Hypothetical Document Embeddings)
        - **Query expansion** with LLM
        - **MMR** (Maximal Marginal Relevance) for diversity
        - **Sentence window** retrieval for context
        - **Parent-child chunks** for better understanding
        - **Hybrid search** (Vector + BM25)
        - **Perfect for complex document queries**
        - Ask questions about PDFs and get comprehensive answers
        - *Requires: Modern RAG system with PDFs*
        """)
        
        st.header("💡 Example Requests")
        st.markdown("""
        **Service Requests:**
        - "I have a headache"
        - "Need somewhere to stay tonight"  
        - "Feeling really drained"
        - "Can you get me a ride?"
        
        **Modern RAG Queries:**
        - "What qualifications does [person] have?"
        - "Summarize the main points of the document"
        - "Find information about [specific topic]"
        - "What experience does the candidate have?"
        - "Tell me about the company's background"
        - "What are the key findings in the report?"
        
        **📋 Context-Aware Follow-ups:**
        - After asking about someone: "What company does he work for?"
        - "What about his education?"
        - "What other skills does she have?"
        - "Tell me more about their experience"
        """)
        
        # Show context status
        if st.session_state.get('last_successful_category') == "Search Knowledge Base":
            st.info("🔗 **Context Active**: Follow-up questions will search documents")
        
        st.header("🚀 Modern RAG Techniques")
        modern_features = """
        **🎯 LATEST RAG TECHNIQUES:**
        - **Token-based chunking** with tiktoken for proper token counting
        - **Recursive text splitting** (LangChain-style) with multiple separators
        - **Parent-child chunks** for hierarchical context
        - **Sentence window retrieval** for better context boundaries
        - **HyDE** (Hypothetical Document Embeddings) for query enhancement
        - **Query expansion** with LLM for better matching
        - **MMR** (Maximal Marginal Relevance) for diverse results
        - **Normalized embeddings** with cosine similarity
        
        **🔍 ADVANCED SEARCH:**
        - **Hybrid retrieval**: Optimized Vector + BM25 search
        - **Query enhancement** pipeline with multiple techniques
        - **Context-aware results** with sentence windows
        - **Parent context** retrieval for comprehensive answers
        - **Quality-weighted scoring** and ranking
        - **GPU acceleration** when available
        
        **📊 MODERN ARCHITECTURE:**
        - **Optimized FAISS indices** with proper metrics
        - **Batch embedding** creation for efficiency
        - **Smart chunk validation** and filtering
        - **Comprehensive metadata** extraction
        - **Advanced tokenization** with tiktoken
        """
        
        st.markdown(modern_features)
        
        # Show modern RAG status
        if VECTOR_STORE_AVAILABLE and "modern_rag_manager" in st.session_state:
            manager = st.session_state.modern_rag_manager
            if manager:
                indices = manager.load_indices()
                if indices and indices.get("metadata"):
                    metadata = indices["metadata"]
                    parent_metadata = indices.get("parent_metadata", [])
                    sources = set(item.get("source", "") for item in metadata)
                    total_tokens = sum(item.get("tokens", 0) for item in metadata)
                    
                    # Check system features
                    system_info = indices.get("system_info", {})
                    techniques = []
                    if system_info.get("use_hyde"):
                        techniques.append("HyDE")
                    if system_info.get("use_query_expansion"):
                        techniques.append("Query Exp")
                    if system_info.get("use_mmr"):
                        techniques.append("MMR")
                    if system_info.get("use_bm25"):
                        techniques.append("BM25")
                        
                    st.success(f"🚀 Modern RAG Ready: {len(sources)} documents, {len(metadata)} chunks, {len(parent_metadata)} parents, {total_tokens:,} tokens ({'+'.join(techniques)})")
                    
                    if st.session_state.get('last_successful_category') == "Search Knowledge Base":
                        st.info("🔗 Context Active: Next questions will search documents")
                else:
                    st.info("🚀 Modern RAG: Not created yet")
        
        st.markdown("---")
        st.header("🔧 Installation & Setup")
        st.markdown("""
        **For Modern RAG (Required):**
        ```bash
        pip install faiss-cpu PyPDF2 rank-bm25 tiktoken
        ```
        
        **For GPU acceleration:**
        ```bash
        pip install faiss-gpu PyPDF2 rank-bm25 tiktoken
        ```
        
        **Environment Variables:**
        ```bash
        ENDPOINT_URL=https://your-resource.openai.azure.com
        AZURE_OPENAI_API_KEY=your-api-key
        DEPLOYMENT_NAME=your-deployment-name
        API_VERSION=2024-12-01-preview
        ```
        
        **🚀 MODERN RAG ADVANTAGES:**
        - ✅ **Token-based chunking** with proper token counting
        - ✅ **HyDE** for enhanced query understanding
        - ✅ **Query expansion** with LLM assistance
        - ✅ **MMR** for diverse, non-redundant results
        - ✅ **Sentence windows** for better context boundaries
        - ✅ **Parent-child chunks** for comprehensive answers
        - ✅ **Hybrid search** combining vector and keyword search
        - ✅ **Normalized embeddings** with cosine similarity
        - ✅ **Perfect for complex document analysis**
        """)
        
        # Show current status
        if VECTOR_STORE_AVAILABLE:
            gpu_status = "🚀 GPU" if GPU_AVAILABLE else "💻 CPU"
            bm25_status = "🔍 BM25" if BM25_AVAILABLE else "❌ BM25"
            try:
                import tiktoken
                tiktoken_status = "🔢 tiktoken"
            except:
                tiktoken_status = "❌ tiktoken"
            st.markdown(f"**Status:** {gpu_status} | {bm25_status} | {tiktoken_status}")
            
        if "openai_client" in st.session_state and "modern_rag_manager" in st.session_state:
            st.markdown("**🚀 Modern RAG:** Full system ready")
        else:
            st.markdown("**❌ Configuration issue**")
        
        if st.button("🗑️ Clear Chat History"):
            # Clear all session state except managers
            for key in list(st.session_state.keys()):
                if key not in ['modern_rag_manager', 'openai_client']:  # Keep these
                    del st.session_state[key]
            
            # Reinitialize required session state
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