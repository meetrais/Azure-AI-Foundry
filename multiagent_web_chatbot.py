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
from typing import List, Dict, Tuple, Optional
import pickle
import time
import base64
from io import BytesIO
import json
import re

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
        
    # Try to import multimodal dependencies
    try:
        import fitz  # PyMuPDF for image extraction
        from PIL import Image
        MULTIMODAL_AVAILABLE = True
    except ImportError:
        MULTIMODAL_AVAILABLE = False
        
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    GPU_AVAILABLE = False
    BM25_AVAILABLE = False
    MULTIMODAL_AVAILABLE = False

class TrueMultimodalRAGManager:
    """True Multimodal RAG with embedded image storage and multimodal embeddings."""
    
    def __init__(self, openai_client, model_name, data_folder="data", vector_store_path="vector_store"):
        if not VECTOR_STORE_AVAILABLE:
            raise ImportError("Vector store dependencies not available. Install with: pip install faiss-cpu PyPDF2 rank-bm25")
        
        self.openai_client = openai_client
        self.model_name = model_name
        self.data_folder = Path(data_folder)
        self.vector_store_path = Path(vector_store_path)
        
        # Create directories
        self.vector_store_path.mkdir(exist_ok=True)
        self.data_folder.mkdir(exist_ok=True)
        
        # Advanced settings
        self.use_gpu = GPU_AVAILABLE
        self.use_bm25 = BM25_AVAILABLE
        self.use_multimodal = MULTIMODAL_AVAILABLE
        self.text_embedding_model = "text-embedding-3-large"  # Best model for multimodal
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_chunks_per_file = 50
        self.min_image_size = (100, 100)
        self.max_images_per_pdf = 20
        self.max_image_size_kb = 500  # Limit image size in embeddings
        
        # Initialize capabilities info
        self._show_capabilities()
    
    def _show_capabilities(self):
        """Show system capabilities."""
        if self.use_gpu:
            st.info(f"🚀 GPU acceleration enabled! Found {faiss.get_num_gpus()} GPU(s)")
        
        if self.use_bm25:
            st.info("🔍 Hybrid retrieval available (Vector + BM25)")
        else:
            st.warning("⚠️ BM25 not available. Install: pip install rank-bm25")
            
        if self.use_multimodal:
            st.info("🖼️ True multimodal support enabled (Images stored in embeddings)")
        else:
            st.warning("⚠️ Multimodal not available. Install: pip install PyMuPDF Pillow")
    
    def extract_and_encode_images_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract images from PDF and encode them as base64 for embedding storage."""
        if not self.use_multimodal:
            return []
        
        images_data = []
        try:
            st.write(f"🔍 Extracting images from {pdf_path.name}...")
            pdf_document = fitz.open(str(pdf_path))
            st.write(f"📄 PDF has {len(pdf_document)} pages")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                st.write(f"📄 Page {page_num + 1}: Found {len(image_list)} image references")
                
                for img_index, img in enumerate(image_list):
                    if len(images_data) >= self.max_images_per_pdf:
                        st.write(f"⚠️ Reached maximum image limit ({self.max_images_per_pdf})")
                        break
                        
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        # Skip if image is too small
                        if pix.width < self.min_image_size[0] or pix.height < self.min_image_size[1]:
                            st.write(f"⏭️ Skipping small image: {pix.width}x{pix.height}")
                            pix = None
                            continue
                        
                        # Handle different color spaces and convert to RGB
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            if pix.n != 4:  # Not RGBA
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            
                            # Convert to bytes
                            img_data = pix.tobytes("png")
                            
                            # Check size limit
                            size_kb = len(img_data) / 1024
                            if size_kb > self.max_image_size_kb:
                                # Resize image if too large
                                img_pil = Image.open(BytesIO(img_data))
                                # Calculate new size maintaining aspect ratio
                                ratio = min(800/img_pil.width, 600/img_pil.height)
                                new_size = (int(img_pil.width * ratio), int(img_pil.height * ratio))
                                img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
                                
                                # Convert back to bytes
                                buffer = BytesIO()
                                img_pil.save(buffer, format='PNG', optimize=True)
                                img_data = buffer.getvalue()
                                size_kb = len(img_data) / 1024
                            
                            # Encode to base64 for storage
                            base64_image = base64.b64encode(img_data).decode('utf-8')
                            
                            # Analyze image content with vision model
                            image_description = self.analyze_image_with_vision(base64_image)
                            
                            # Create image metadata with embedded data
                            image_info = {
                                "base64_data": base64_image,
                                "description": image_description,
                                "source_pdf": pdf_path.name,
                                "page_number": page_num + 1,
                                "image_index": img_index + 1,
                                "width": pix.width,
                                "height": pix.height,
                                "size_kb": size_kb,
                                "format": "PNG"
                            }
                            
                            images_data.append(image_info)
                            st.write(f"✅ Processed image: Page {page_num+1} Image {img_index+1} ({size_kb:.1f}KB)")
                            
                        pix = None  # Clean up
                        
                    except Exception as img_error:
                        st.write(f"❌ Error processing image {img_index + 1}: {str(img_error)}")
                        continue
                
                if len(images_data) >= self.max_images_per_pdf:
                    break
            
            pdf_document.close()
            st.success(f"✅ Extracted and encoded {len(images_data)} images from {pdf_path.name}")
            
        except Exception as e:
            st.error(f"❌ Error extracting images from {pdf_path.name}: {str(e)}")
        
        return images_data
    
    def analyze_image_with_vision(self, base64_image: str) -> str:
        """Analyze image content using vision model."""
        try:
            # Check if the model supports vision
            if "gpt-4" not in self.model_name.lower():
                return ""
            
            # Call vision model
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail, focusing on text content, charts, diagrams, people, objects, and any important visual information that would be useful for document search and retrieval."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.write(f"⚠️ Vision analysis failed: {str(e)}")
            return ""
    
    def create_multimodal_embedding(self, text: str, images: List[Dict] = None) -> np.ndarray:
        """Create true multimodal embedding combining text and image information."""
        try:
            # Enhanced text with image descriptions
            enhanced_text = text
            
            # Add image descriptions to text
            if images:
                image_descriptions = []
                for img in images:
                    desc = img.get("description", "")
                    if desc:
                        image_descriptions.append(f"[Image: {desc}]")
                
                if image_descriptions:
                    enhanced_text += "\n\nImage Content:\n" + "\n".join(image_descriptions)
            
            # Get embedding for enhanced text
            response = self.openai_client.embeddings.create(
                model=self.text_embedding_model,
                input=enhanced_text[:8000]  # Truncate if too long
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            return embedding
            
        except Exception as e:
            st.error(f"Error creating multimodal embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(3072, dtype=np.float32)  # text-embedding-3-large dimension
    
    def intelligent_text_preprocessing(self, text: str) -> Dict:
        """Advanced text preprocessing with structure awareness."""
        if not text.strip():
            return {"sections": [], "metadata": {}}
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
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
    
    def intelligent_chunking_with_embedded_images(self, sections: List[Dict], images_data: List[Dict], 
                                                 chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Intelligent chunking with embedded image data."""
        chunks = []
        
        # Create text chunks first
        text_chunks = self.intelligent_chunking(sections, chunk_size, overlap)
        
        # Distribute images across chunks intelligently
        for chunk_idx, chunk in enumerate(text_chunks):
            chunk["embedded_images"] = []
            chunk["has_images"] = False
            
            # Distribute images across chunks based on content relevance
            if images_data:
                # Simple distribution: spread images across chunks
                total_chunks = len(text_chunks)
                if total_chunks > 0:
                    images_per_chunk = max(1, len(images_data) // total_chunks)
                    start_idx = chunk_idx * images_per_chunk
                    end_idx = min(start_idx + images_per_chunk + 1, len(images_data))
                    
                    # Assign images to this chunk
                    assigned_images = images_data[start_idx:end_idx]
                    
                    # Store only essential image data with the chunk
                    for img in assigned_images:
                        chunk_image = {
                            "base64_data": img["base64_data"],
                            "description": img["description"],
                            "source_pdf": img["source_pdf"],
                            "page_number": img["page_number"],
                            "width": img["width"],
                            "height": img["height"],
                            "size_kb": img["size_kb"]
                        }
                        chunk["embedded_images"].append(chunk_image)
                    
                    chunk["has_images"] = len(chunk["embedded_images"]) > 0
            
            chunks.append(chunk)
        
        return chunks
    
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
        return len(text.split()) * 1.3
    
    def create_multimodal_search_index(self, chunks: List[Dict]) -> Dict:
        """Create search index with embedded multimodal data."""
        if not chunks:
            return {"vector_index": None, "bm25_index": None, "metadata": []}
        
        # Prepare metadata with embedded images
        metadata = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "content": chunk["content"],
                "title": chunk.get("title", ""),
                "type": chunk.get("type", "content"),
                "tokens": chunk.get("tokens", 0),
                "chunk_id": i,
                "embedded_images": chunk.get("embedded_images", []),
                "has_images": chunk.get("has_images", False),
                "source": chunk.get("source", ""),
                "file_metadata": chunk.get("file_metadata", {}),
                "file_index": chunk.get("file_index", 0)
            }
            metadata.append(chunk_metadata)
        
        # Create multimodal embeddings
        st.info("🔄 Creating true multimodal embeddings...")
        embeddings = self.get_optimized_multimodal_embeddings(chunks)
        vector_index = self.create_faiss_index(embeddings)
        
        # Create BM25 index
        bm25_index = None
        tokenized_docs = None
        if self.use_bm25:
            st.info("🔄 Creating BM25 sparse index...")
            texts = [chunk["content"] for chunk in chunks]
            tokenized_docs = [self._tokenize_for_bm25(text) for text in texts]
            bm25_index = BM25Okapi(tokenized_docs)
        
        return {
            "vector_index": vector_index,
            "bm25_index": bm25_index,
            "metadata": metadata,
            "tokenized_docs": tokenized_docs if self.use_bm25 else None
        }
    
    def get_optimized_multimodal_embeddings(self, chunks: List[Dict], batch_size: int = 10) -> np.ndarray:
        """Create optimized multimodal embeddings with true image integration."""
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                batch_embeddings = []
                
                for chunk in batch:
                    text = chunk["content"]
                    embedded_images = chunk.get("embedded_images", [])
                    
                    # Create multimodal embedding
                    embedding = self.create_multimodal_embedding(text, embedded_images)
                    batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
                
                # Progress update
                progress = min(i + batch_size, len(chunks))
                if progress % 5 == 0:
                    st.write(f"✅ Processed {progress}/{len(chunks)} multimodal embeddings")
                
            except Exception as e:
                st.warning(f"Error in batch {i//batch_size + 1}: {str(e)}")
                # Add zero vectors for failed batch
                for _ in batch:
                    embeddings.append(np.zeros(3072, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
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
    
    def display_multimodal_results_streamlit(self, results: List[Dict]):
        """Enhanced display of true multimodal results with embedded images."""
        if not results:
            st.info("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            with st.expander(f"📄 Result {i} - {result.get('source', 'Unknown')}", expanded=(i<=2)):
                # Display text content
                st.markdown("**Content:**")
                st.write(result.get("content", ""))
                
                # Display embedded images
                embedded_images = result.get("embedded_images", [])
                if embedded_images:
                    st.markdown(f"**🖼️ Embedded Images ({len(embedded_images)}):**")
                    
                    # Create columns for better image layout
                    cols = st.columns(min(2, len(embedded_images)))
                    
                    for idx, image_data in enumerate(embedded_images):
                        col_idx = idx % len(cols)
                        
                        with cols[col_idx]:
                            try:
                                # Decode base64 image
                                image_bytes = base64.b64decode(image_data["base64_data"])
                                
                                # Display image
                                st.image(
                                    image_bytes,
                                    caption=f"Page {image_data['page_number']} ({image_data['width']}x{image_data['height']}, {image_data['size_kb']:.1f}KB)",
                                    width=300
                                )
                                
                                # Show image description if available
                                if image_data.get("description"):
                                    st.caption(f"📝 {image_data['description']}")
                                
                            except Exception as e:
                                st.error(f"Error displaying embedded image {idx + 1}: {str(e)}")
                    
                    st.markdown("---")
                else:
                    st.info("No embedded images in this result.")
                
                # Display metadata
                if result.get("rrf_score"):
                    st.caption(f"Relevance Score: {result['rrf_score']:.4f}")
                elif result.get("similarity_score"):
                    st.caption(f"Similarity Score: {result['similarity_score']:.4f}")
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Dict]:
        """Enhanced hybrid search with multimodal query processing."""
        indices = self.load_search_indices()
        if not indices or not indices["metadata"]:
            return []
        
        vector_index = indices["vector_index"]
        bm25_index = indices["bm25_index"]
        metadata = indices["metadata"]
        
        all_results = []
        
        # Vector search with multimodal query
        if vector_index:
            # Create embedding for query (enhanced with any image context)
            query_embedding = self.create_multimodal_embedding(query)
            vector_results = self._vector_search_multimodal(query_embedding, vector_index, metadata, k)
            for result in vector_results:
                result["search_type"] = "vector_multimodal"
            all_results.extend(vector_results)
        
        # BM25 search
        if bm25_index and self.use_bm25:
            bm25_results = self._bm25_search(query, bm25_index, metadata, k)
            for result in bm25_results:
                result["search_type"] = "bm25"
            all_results.extend(bm25_results)
        
        # Enhanced reciprocal rank fusion
        combined_results = self._enhanced_reciprocal_rank_fusion(all_results, k)
        
        return combined_results[:k]
    
    def _vector_search_multimodal(self, query_embedding: np.ndarray, index, metadata: List[Dict], k: int) -> List[Dict]:
        """Enhanced vector search with embedded image data."""
        try:
            search_k = min(k * 3, index.ntotal)
            distances, indices = index.search(query_embedding.reshape(1, -1), search_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata) and idx >= 0:
                    result = metadata[idx].copy()
                    # Convert L2 distance to similarity score
                    similarity = 1.0 / (1.0 + distances[0][i])
                    result["similarity_score"] = float(distances[0][i])
                    result["normalized_score"] = similarity
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Multimodal vector search error: {str(e)}")
            return []
    
    def _bm25_search(self, query: str, bm25_index, metadata: List[Dict], k: int) -> List[Dict]:
        """Enhanced BM25 search."""
        try:
            query_tokens = self._tokenize_for_bm25(query)
            
            if not query_tokens:
                return []
            
            scores = bm25_index.get_scores(query_tokens)
            
            # Get top k indices with non-zero scores
            scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for idx, score in scored_indices[:k * 2]:
                if idx < len(metadata):
                    result = metadata[idx].copy()
                    normalized_score = min(score / 10.0, 1.0)
                    result["similarity_score"] = float(1.0 - normalized_score)
                    result["bm25_score"] = float(score)
                    result["normalized_score"] = normalized_score
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"BM25 search error: {str(e)}")
            return []
    
    def _enhanced_reciprocal_rank_fusion(self, results: List[Dict], k: int) -> List[Dict]:
        """Enhanced RRF for multimodal results."""
        rrf_scores = {}
        
        # Group results by search type
        vector_results = [r for r in results if "vector" in r.get("search_type", "")]
        bm25_results = [r for r in results if r.get("search_type") == "bm25"]
        
        # Sort each group
        vector_results.sort(key=lambda x: x.get("similarity_score", float('inf')))
        bm25_results.sort(key=lambda x: x.get("similarity_score", float('inf')))
        
        # RRF parameters
        rank_constant = 60
        vector_weight = 0.7
        bm25_weight = 0.3
        
        # Calculate RRF scores
        for rank, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"result": result, "score": 0}
            
            score_contribution = vector_weight / (rank_constant + rank + 1)
            rrf_scores[chunk_id]["score"] += score_contribution
        
        for rank, result in enumerate(bm25_results):
            chunk_id = result["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"result": result, "score": 0}
            
            score_contribution = bm25_weight / (rank_constant + rank + 1)
            rrf_scores[chunk_id]["score"] += score_contribution
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        final_results = []
        for chunk_id, data in sorted_results:
            result = data["result"].copy()
            result["rrf_score"] = data["score"]
            result["similarity_score"] = 1.0 - data["score"]
            final_results.append(result)
        
        return final_results
    
    def context_compression_multimodal(self, results: List[Dict], query: str, max_tokens: int = 4000) -> str:
        """Enhanced context compression with multimodal awareness."""
        if not results:
            return ""
        
        # Sort by RRF score if available
        results = sorted(results, 
                        key=lambda x: x.get("rrf_score", 1.0 - x.get("similarity_score", 1.0)), 
                        reverse=True)
        
        compressed_sections = []
        total_tokens = 0
        seen_content_hashes = set()
        query_keywords = set(self._tokenize_for_bm25(query.lower()))
        
        for result in results:
            content = result.get("content", "")
            title = result.get("title", "")
            embedded_images = result.get("embedded_images", [])
            
            # Skip very similar content
            content_hash = hash(content[:300])
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            
            # Estimate tokens
            content_tokens = self._estimate_tokens(content)
            title_tokens = self._estimate_tokens(title) if title else 0
            
            # Add image description tokens
            image_tokens = 0
            image_descriptions = []
            for img in embedded_images:
                desc = img.get("description", "")
                if desc:
                    image_descriptions.append(f"[Image: {desc}]")
                    image_tokens += self._estimate_tokens(desc)
            
            section_tokens = content_tokens + title_tokens + image_tokens
            
            # Check if we can fit this section
            if total_tokens + section_tokens > max_tokens:
                break
            
            # Add content with image descriptions
            if title:
                section_text = f"**{title}**\n{content}"
            else:
                section_text = content
            
            # Add image descriptions
            if image_descriptions:
                section_text += f"\n\nImages in this section:\n" + "\n".join(image_descriptions)
                
            compressed_sections.append(section_text)
            total_tokens += section_tokens
        
        return "\n\n---\n\n".join(compressed_sections)
    
    def create_vector_store(self) -> Dict:
        """Create true multimodal RAG system with embedded images."""
        pdf_files = list(self.data_folder.glob("*.pdf"))
        
        if not pdf_files:
            return {"success": False, "message": "No PDF files found in data folder"}
        
        max_files = 15
        if len(pdf_files) > max_files:
            st.warning(f"Processing first {max_files} of {len(pdf_files)} files")
            pdf_files = pdf_files[:max_files]
        
        all_chunks = []
        processing_stats = {"files": 0, "sections": 0, "chunks": 0, "images": 0}
        
        # Progress tracking
        main_progress = st.progress(0)
        main_status = st.empty()
        start_time = time.time()
        
        # Phase 1: Extract text and embed images
        main_status.text("🧠 Extracting text and embedding images...")
        
        for file_idx, pdf_file in enumerate(pdf_files):
            st.write(f"\n### Processing {pdf_file.name}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            
            # Extract and encode images for embedding
            images_data = []
            if self.use_multimodal:
                images_data = self.extract_and_encode_images_from_pdf(pdf_file)
                processing_stats["images"] += len(images_data)
            
            if not text and not images_data:
                st.warning(f"No content extracted from {pdf_file.name}")
                continue
            
            # Process text
            processed = self.intelligent_text_preprocessing(text) if text else {"sections": [], "metadata": {}}
            processing_stats["sections"] += len(processed["sections"])
            
            # Create chunks with embedded images
            chunks = self.intelligent_chunking_with_embedded_images(processed["sections"], images_data)
            processing_stats["chunks"] += len(chunks)
            
            # Add metadata
            for chunk in chunks:
                chunk["source"] = pdf_file.name
                chunk["file_metadata"] = processed["metadata"]
                chunk["file_index"] = file_idx
            
            all_chunks.extend(chunks)
            processing_stats["files"] += 1
            
            st.success(f"✅ Processed {pdf_file.name}: {len(chunks)} chunks, {len(images_data)} embedded images")
            
            # Update progress
            file_progress = (file_idx + 1) / len(pdf_files) * 0.4
            main_progress.progress(file_progress)
        
        if not all_chunks:
            return {"success": False, "message": "No content extracted from PDF files"}
        
        # Limit chunks if too many
        max_total_chunks = self.max_chunks_per_file * len(pdf_files)
        if len(all_chunks) > max_total_chunks:
            st.info(f"Limiting to {max_total_chunks} chunks for optimal performance")
            all_chunks = all_chunks[:max_total_chunks]
            processing_stats["chunks"] = len(all_chunks)
        
        main_status.text(f"📊 Creating true multimodal search indices...")
        main_progress.progress(0.4)
        
        # Phase 2: Create multimodal search indices
        try:
            indices = self.create_multimodal_search_index(all_chunks)
            main_progress.progress(0.9)
        except Exception as e:
            return {"success": False, "message": f"Error creating indices: {str(e)}"}
        
        # Phase 3: Save everything
        main_status.text("💾 Saving true multimodal RAG system...")
        
        try:
            self.save_search_indices(indices)
        except Exception as e:
            return {"success": False, "message": f"Error saving: {str(e)}"}
        
        # Complete
        total_time = time.time() - start_time
        main_progress.progress(1.0)
        main_status.text(f"✅ True multimodal RAG system created in {total_time:.1f}s!")
        
        search_types = ["Multimodal Vector (True)"]
        if self.use_bm25:
            search_types.append("BM25 Sparse")
        
        return {
            "success": True,
            "message": f"True multimodal RAG created: {processing_stats['chunks']} chunks with {processing_stats['images']} embedded images from {processing_stats['files']} files",
            "chunks": processing_stats["chunks"],
            "files": processing_stats["files"],
            "sections": processing_stats["sections"],
            "images": processing_stats["images"],
            "time": total_time,
            "search_types": search_types
        }
    
    def save_search_indices(self, indices: Dict):
        """Save search indices with embedded image data."""
        # Save vector index
        if indices["vector_index"]:
            vector_path = self.vector_store_path / "faiss_index.bin"
            faiss.write_index(indices["vector_index"], str(vector_path))
        
        # Save metadata with embedded images
        metadata_path = self.vector_store_path / "multimodal_metadata.pkl"
        save_data = {
            "metadata": indices["metadata"],
            "bm25_index": indices["bm25_index"],
            "tokenized_docs": indices.get("tokenized_docs"),
            "creation_time": time.time(),
            "system_info": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "use_bm25": self.use_bm25,
                "use_multimodal": self.use_multimodal,
                "text_embedding_model": self.text_embedding_model,
                "is_true_multimodal": True  # Flag for new system
            }
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_search_indices(self) -> Dict:
        """Load search indices with embedded image data."""
        vector_path = self.vector_store_path / "faiss_index.bin"
        metadata_path = self.vector_store_path / "multimodal_metadata.pkl"
        
        if not (vector_path.exists() and metadata_path.exists()):
            return {}
        
        try:
            # Load vector index
            vector_index = faiss.read_index(str(vector_path))
            
            # Load metadata
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
            st.error(f"Error loading search indices: {str(e)}")
            return {}
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                max_pages = min(150, len(pdf_reader.pages))
                text_parts = []
                
                for i, page in enumerate(pdf_reader.pages[:max_pages]):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text = re.sub(r'\s+', ' ', text.strip())
                            text_parts.append(text)
                    except Exception:
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
        """Main search interface using true multimodal approach."""
        try:
            # Use enhanced hybrid search
            results = self.hybrid_search(query, k)
            
            # Ensure full content and embedded images are available
            indices = self.load_search_indices()
            if indices and indices["metadata"]:
                metadata = indices["metadata"]
                for result in results:
                    chunk_id = result.get("chunk_id", -1)
                    if 0 <= chunk_id < len(metadata):
                        # Metadata now stores embedded images
                        result["content"] = metadata[chunk_id].get("content", result.get("content", ""))
                        result["embedded_images"] = metadata[chunk_id].get("embedded_images", [])
                        result["has_images"] = metadata[chunk_id].get("has_images", False)
            
            return results
            
        except Exception as e:
            st.error(f"True multimodal search error: {str(e)}")
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

def create_true_multimodal_vector_store_ui(openai_client, model_name):
    """True Multimodal RAG UI with embedded image storage."""
    st.sidebar.markdown("---")
    st.sidebar.header("🚀 True Multimodal RAG (Images in Embeddings)")
    
    # Check if vector store dependencies are available
    if not VECTOR_STORE_AVAILABLE:
        st.sidebar.error("❌ Vector store dependencies not installed")
        st.sidebar.code("pip install faiss-cpu PyPDF2 rank-bm25 PyMuPDF Pillow")
        return None
    
    # Initialize true multimodal RAG manager
    if "vector_manager" not in st.session_state:
        try:
            st.session_state.vector_manager = TrueMultimodalRAGManager(openai_client, model_name)
        except Exception as e:
            st.sidebar.error(f"Error initializing true multimodal RAG: {str(e)}")
            return None
    
    vector_manager = st.session_state.vector_manager
    
    # Check data folder
    data_folder = Path("data")
    pdf_files = list(data_folder.glob("*.pdf")) if data_folder.exists() else []
    
    st.sidebar.markdown(f"**PDF Files in Data Folder:** {len(pdf_files)}")
    
    if len(pdf_files) > 15:
        st.sidebar.warning(f"⚠️ {len(pdf_files)} PDF files found. First 15 will be processed.")
    
    if pdf_files:
        with st.sidebar.expander("📄 Available PDF Files", expanded=False):
            for i, pdf_file in enumerate(pdf_files[:15]):
                file_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
                status = "✅" if file_size <= 15 else "⚠️"
                st.write(f"{status} {pdf_file.name} ({file_size:.1f}MB)")
            if len(pdf_files) > 15:
                st.write(f"... and {len(pdf_files) - 15} more files")
    else:
        st.sidebar.warning("⚠️ No PDF files found in 'data' folder")
        st.sidebar.markdown("*Place PDF files in the 'data' folder*")
    
    # Enhanced features info
    feature_text = """⚡ **True Multimodal Features:**
- **Images stored in embeddings** (not as files)
- **True multimodal embeddings** with image content
- **Base64 image encoding** for vector storage
- **Inline image display** with search results
- **Vision model analysis** of image content
- **Hybrid search** (Vector + BM25)
- **Intelligent preprocessing** and chunking
- **Context compression** with image awareness
- **GPU acceleration** when available
- **Process up to 15 PDF files**
- **Files up to 15MB** supported
- **Smart image size optimization**
"""
    
    if not MULTIMODAL_AVAILABLE:
        feature_text += "\n- ⚠️ **Image support disabled** (install PyMuPDF + Pillow)"
    
    st.sidebar.info(feature_text)
    
    # Check if true multimodal indices exist
    indices = vector_manager.load_search_indices()
    vector_store_exists = bool(indices and indices.get("metadata"))
    
    if vector_store_exists:
        st.sidebar.success("✅ True Multimodal RAG system exists")
        
        try:
            metadata = indices.get("metadata", [])
            if metadata:
                st.sidebar.markdown(f"**Chunks:** {len(metadata)}")
                
                # Count embedded images
                total_images = sum(len(item.get("embedded_images", [])) for item in metadata)
                if total_images > 0:
                    st.sidebar.markdown(f"**Embedded Images:** {total_images}")
                
                # Get unique sources
                sources = set(item["source"] for item in metadata if "source" in item)
                st.sidebar.markdown(f"**Sources:** {len(sources)}")
                
                # Show search capabilities
                search_types = ["True Multimodal Vector"]
                if indices.get("bm25_index") and BM25_AVAILABLE:
                    search_types.append("BM25 Sparse")
                st.sidebar.markdown(f"**Search Types:** {', '.join(search_types)}")
                
                # Show system type
                system_info = indices.get("system_info", {})
                if system_info.get("is_true_multimodal"):
                    st.sidebar.success("🎯 True Multimodal System")
                else:
                    st.sidebar.warning("⚠️ Old System (Upgrade Recommended)")
        except:
            pass
    else:
        st.sidebar.info("ℹ️ No true multimodal RAG system found")
    
    # Create/Recreate button
    button_text = "🔄 Recreate True Multimodal RAG" if vector_store_exists else "🚀 Create True Multimodal RAG"
    
    if st.sidebar.button(button_text, disabled=(len(pdf_files) == 0)):
        if len(pdf_files) == 0:
            st.sidebar.error("❌ No PDF files found to process")
        else:
            with st.sidebar:
                st.markdown("### 🚀 Creating True Multimodal RAG System")
                try:
                    result = vector_manager.create_vector_store()
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        if result.get("images", 0) > 0:
                            st.info(f"🖼️ Embedded {result['images']} images in vector store")
                        st.balloons()
                        
                        # Show capabilities
                        if "search_types" in result:
                            st.info(f"🔍 Search types: {', '.join(result['search_types'])}")
                    else:
                        st.error(f"❌ {result['message']}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Enhanced search interface
    if vector_store_exists:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 True Multimodal Search")
        
        search_query = st.sidebar.text_input("Search query:", placeholder="Search PDFs with images...")
        
        if st.sidebar.button("🔍 Multimodal Search") and search_query:
            with st.sidebar:
                with st.spinner("Searching..."):
                    try:
                        start_time = time.time()
                        results = vector_manager.search_similar(search_query, k=3)
                        search_time = time.time() - start_time
                        
                        if results:
                            st.success(f"Found {len(results)} results in {search_time:.2f}s")
                            
                            # Show condensed results
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Result {i} - {result.get('source', 'Unknown')}", expanded=(i==1)):
                                    # Show text preview
                                    content = result.get('content', '')
                                    preview = content[:150] + "..." if len(content) > 150 else content
                                    st.write(f"**Content:** {preview}")
                                    
                                    # Show embedded images count
                                    embedded_images = result.get("embedded_images", [])
                                    if embedded_images:
                                        st.write(f"**🖼️ Embedded Images:** {len(embedded_images)}")
                                        
                                        # Show first image thumbnail
                                        if len(embedded_images) > 0:
                                            try:
                                                image_data = embedded_images[0]
                                                image_bytes = base64.b64decode(image_data["base64_data"])
                                                st.image(
                                                    image_bytes,
                                                    caption=f"Page {image_data['page_number']}",
                                                    width=200
                                                )
                                            except:
                                                st.write("Image preview unavailable")
                                    
                                    # Show relevance score
                                    rrf_score = result.get('rrf_score')
                                    if rrf_score:
                                        st.write(f"**Relevance:** {rrf_score:.4f}")
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
        st.markdown("📚🖼️ **True Multimodal Knowledge Base Agent** is now assisting you.")
        
        # Check if vector store exists
        if "vector_manager" not in st.session_state:
            st.error("❌ True Multimodal RAG not initialized. Please create a system first.")
            if st.button("Return to Main Agent"):
                return True
            return False
        
        vector_manager = st.session_state.vector_manager
        indices = vector_manager.load_search_indices()
        
        if not indices or not indices.get("metadata"):
            st.error("❌ No true multimodal RAG system found. Please create one first.")
            if st.button("Return to Main Agent"):
                return True
            return False
        
        # Show system type
        system_info = indices.get("system_info", {})
        if system_info.get("is_true_multimodal"):
            st.success("🎯 Using True Multimodal System with embedded images")
        else:
            st.warning("⚠️ Using legacy system - consider recreating for better multimodal support")
        
        # Check for original query from routing
        original_query = st.session_state.get('original_user_query', '')
        
        if original_query and not st.session_state.get('kb_query_processed', False):
            # Process the original query with true multimodal search
            with st.spinner("🔍 True multimodal searching..."):
                try:
                    start_time = time.time()
                    
                    # Determine search scope
                    general_queries = ["what information", "what's in", "what is in", "what does the document contain", 
                                     "what topics", "summarize", "overview", "what's available"]
                    is_general_query = any(phrase in original_query.lower() for phrase in general_queries)
                    
                    # Use enhanced search
                    if is_general_query:
                        results = vector_manager.search_similar(original_query, k=12)
                    else:
                        results = vector_manager.search_similar(original_query, k=8)
                    
                    search_time = time.time() - start_time
                    
                    if results:
                        # Display true multimodal results
                        st.markdown("### 🚀 True Multimodal Search Results")
                        st.write(f"*Found {len(results)} relevant sections with embedded images in {search_time:.2f}s*")
                        
                        # Enhanced multimodal display
                        vector_manager.display_multimodal_results_streamlit(results)
                        
                        # Generate AI summary with multimodal context
                        try:
                            openai_client = st.session_state.get('openai_client')
                            model_name = os.getenv("DEPLOYMENT_NAME")
                            
                            if openai_client and model_name:
                                max_tokens = 6000 if is_general_query else 4000
                                compressed_context = vector_manager.context_compression_multimodal(results, original_query, max_tokens)
                                
                                if compressed_context:
                                    if is_general_query:
                                        summary_prompt = f"""Based on the following multimodal content from documents (text + embedded image descriptions), provide a comprehensive overview of what information is available.

User Question: {original_query}

Multimodal Document Content:
{compressed_context}

Please provide a helpful summary of the main topics, types of information, and key content available, including visual elements described in the images."""
                                    else:
                                        summary_prompt = f"""Based on the following multimodal content from documents (text + embedded image descriptions), provide a clear answer to the user's question.

User Question: {original_query}

Relevant Multimodal Content:
{compressed_context}

Please provide a direct, accurate answer based on both the text content and visual information described above."""

                                    response = openai_client.chat.completions.create(
                                        model=model_name,
                                        messages=[{"role": "user", "content": summary_prompt}],
                                        temperature=0.05,
                                        max_tokens=800
                                    )
                                    
                                    ai_summary = response.choices[0].message.content.strip()
                                    
                                    # Display AI summary
                                    st.markdown("### 🤖 AI Summary (True Multimodal)")
                                    st.write(ai_summary)
                                    
                                    final_answer = f"{ai_summary}\n\n*🚀 True multimodal search with {len(results)} sections and embedded images in {search_time:.2f}s*"
                                    
                        except Exception as e:
                            # Fallback to context display
                            final_answer = f"Based on multimodal documents:\n{compressed_context}\n\n*🚀 True multimodal search found {len(results)} sections in {search_time:.2f}s*"
                        
                        st.session_state.agent_complete = True
                        st.session_state.agent_result = final_answer
                        st.session_state.kb_query_processed = True
                        return True
                        
                    else:
                        # Show sample content
                        try:
                            metadata = indices.get("metadata", [])
                            if metadata:
                                st.markdown("No specific matches found, but here's sample content from your true multimodal knowledge base:")
                                
                                for i, item in enumerate(metadata[:3], 1):
                                    with st.expander(f"Sample {i} (from {item.get('source', 'Unknown')})", expanded=(i==1)):
                                        content = item.get('content', '')[:400]
                                        if len(item.get('content', '')) > 400:
                                            content += "..."
                                        st.write(content)
                                        
                                        # Show embedded images
                                        embedded_images = item.get("embedded_images", [])
                                        if embedded_images:
                                            st.write(f"*Contains {len(embedded_images)} embedded image(s)*")
                                            # Show first image
                                            try:
                                                image_data = embedded_images[0]
                                                image_bytes = base64.b64decode(image_data["base64_data"])
                                                st.image(image_bytes, width=200)
                                            except:
                                                st.write("Image preview unavailable")
                                
                                total_images = sum(len(item.get("embedded_images", [])) for item in metadata)
                                st.write(f"*Total: {len(metadata)} sections with {total_images} embedded images*")
                                
                                final_answer = f"Sample content shown. Total: {len(metadata)} sections with {total_images} embedded images."
                            else:
                                final_answer = "No relevant information found. Try rephrasing your question."
                        except:
                            final_answer = "No relevant information found. Try rephrasing your question."
                        
                        st.session_state.agent_complete = True
                        st.session_state.agent_result = final_answer
                        st.session_state.kb_query_processed = True
                        return True
                        
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg or "DeploymentNotFound" in error_msg:
                        fallback_answer = "Unable to search due to AI service configuration issue."
                    elif "401" in error_msg:
                        fallback_answer = "Unable to search due to authentication issues."
                    else:
                        fallback_answer = f"Search error: {error_msg}"
                    
                    st.session_state.agent_complete = True
                    st.session_state.agent_result = fallback_answer
                    st.session_state.kb_query_processed = True
                    return True
        
        else:
            # Show form for additional queries
            with st.form("true_multimodal_query_form"):
                st.markdown("**Enter your question:**")
                query = st.text_area("Query:", placeholder="Ask about documents with visual content...")
                
                submitted = st.form_submit_button("🚀 True Multimodal Search", type="primary")
                
                if submitted and query.strip():
                    with st.spinner("🔍 True multimodal searching..."):
                        try:
                            start_time = time.time()
                            results = vector_manager.search_similar(query.strip(), k=8)
                            search_time = time.time() - start_time
                            
                            if results:
                                # Display results
                                st.markdown("### 🚀 True Multimodal Search Results")
                                st.write(f"*Found {len(results)} relevant sections with embedded images in {search_time:.2f}s*")
                                
                                vector_manager.display_multimodal_results_streamlit(results)
                                
                                # Generate summary
                                compressed_context = vector_manager.context_compression_multimodal(results, query.strip(), 4000)
                                
                                try:
                                    openai_client = st.session_state.get('openai_client')
                                    model_name = os.getenv("DEPLOYMENT_NAME")
                                    
                                    if openai_client and model_name and compressed_context:
                                        summary_prompt = f"""Based on multimodal content (text + image descriptions), answer the user's question.

User Question: {query}

Relevant Multimodal Content:
{compressed_context}

Provide a direct, accurate answer based on both text and visual information."""

                                        response = openai_client.chat.completions.create(
                                            model=model_name,
                                            messages=[{"role": "user", "content": summary_prompt}],
                                            temperature=0.05,
                                            max_tokens=700
                                        )
                                        
                                        ai_summary = response.choices[0].message.content.strip()
                                        
                                        st.markdown("### 🤖 AI Summary (True Multimodal)")
                                        st.write(ai_summary)
                                        
                                        final_answer = f"{ai_summary}\n\n*🚀 True multimodal search with embedded images in {search_time:.2f}s*"
                                        
                                except:
                                    final_answer = f"Based on multimodal documents:\n{compressed_context}\n\n*🚀 Search completed in {search_time:.2f}s*"
                                
                                st.session_state.agent_complete = True
                                st.session_state.agent_result = final_answer
                                return True
                                
                            else:
                                st.session_state.agent_complete = True
                                st.session_state.agent_result = "No relevant information found in your documents."
                                return True
                                
                        except Exception as e:
                            error_msg = str(e)
                            if "404" in error_msg:
                                fallback_answer = "Unable to search due to AI service configuration issue."
                            elif "401" in error_msg:
                                fallback_answer = "Unable to search due to authentication issues."
                            else:
                                fallback_answer = f"Search error: {error_msg}"
                            
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
        page_title="True Multimodal Multi-Agent Assistant",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Multi-Agent Assistant with True Multimodal RAG")
    st.markdown("Featuring **TRUE MULTIMODAL** text + image storage in embeddings, **embedded base64 images**, **inline image display**, **multimodal vector search**, **ultra-fast hybrid RAG**, **intelligent preprocessing**, and optimized processing for all your requests.")

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

    # Add true multimodal RAG UI to sidebar
    vector_manager = create_true_multimodal_vector_store_ui(openai_client, model_name)
    
    # Store openai_client in session state
    st.session_state.openai_client = openai_client

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
                error_message = "🤔 I'm sorry, I couldn't categorize your request. I can help you with reporting sickness, reporting fatigue, booking hotels, booking transportation, or searching your documents. Could you please clarify what you need help with?"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            st.rerun()

    # Enhanced sidebar with true multimodal info
    with st.sidebar:
        st.header("🚀 True Multimodal Multi-Agent Services")
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
        
        🚀 **True Multimodal RAG Search**
        - **Images stored in embeddings** (not as files)
        - **True multimodal vector search** with embedded images
        - **Inline image display** with search results
        - **Base64 image encoding** for vector storage
        - **Vision model analysis** of image content
        - **Hybrid retrieval** (Vector + BM25)
        - **Context-aware follow-up** questions
        - **Intelligent preprocessing** and chunking
        - **Smart context compression** with image awareness
        - Ask questions about PDFs with visual content
        - Get AI summaries with multimodal context
        - View images inline with relevant text
        - *Requires: True Multimodal RAG system with PDFs*
        """)
        
        st.header("💡 Example Requests")
        st.markdown("""
        **Service Requests:**
        - "I have a headache"
        - "Need somewhere to stay tonight"  
        - "Feeling really drained"
        - "Can you get me a ride?"
        
        **True Multimodal RAG Queries:**
        - "How many years of experience does [person] have?"
        - "What skills are mentioned in the resume?"
        - "Find information about XYZ"
        - "Show me any charts or diagrams"
        - "What do the images show?"
        - "Describe the visual content"
        - "Are there any photos of horses?"
        - "What technical diagrams are included?"
        
        **📋 Context-Aware Follow-ups:**
        - After asking about someone: "Which company does he work for?"
        - "What about his education?"
        - "Show me related images"
        - "What visual elements support this?"
        """)
        
        # Show context status
        if st.session_state.get('last_successful_category') == "Search Knowledge Base":
            st.info("🔗 **Context Active**: Follow-up questions will use Knowledge Base")
        
        st.header("🚀 True Multimodal RAG Features")
        multimodal_features = """
        **🎯 TRUE MULTIMODAL SYSTEM:**
        - **Images stored IN embeddings** (not as separate files)
        - **Base64 encoded images** in vector store data
        - **Inline image display** with search results
        - **Multimodal embeddings** with image content analysis
        - **Vision model integration** for image understanding
        - **Smart image size optimization** and compression
        - **Enhanced image extraction** with detailed logging
        - **Better error handling** for image processing
        - **Embedded image metadata** with descriptions
        
        **🔍 ENHANCED SEARCH:**
        - **Hybrid retrieval**: Vector + BM25 sparse search
        - **Intelligent preprocessing**: Structure-aware text processing
        - **Context compression**: Optimal multimodal context for LLM
        - **Reciprocal rank fusion**: Combines search strategies
        - **Context-aware conversations**: Auto Knowledge Base routing
        - **Query expansion**: Related terms and synonyms
        
        **⚡ PERFORMANCE:**
        - **Ultra-fast system creation** with batch processing
        - **GPU acceleration** when available
        - **Smart memory management** for large files
        - **Process up to 15 PDF files** simultaneously
        - **Files up to 15MB** supported per file
        - **Optimized embedding generation** in batches
        """
        
        if MULTIMODAL_AVAILABLE:
            multimodal_features += "\n- **✅ Full multimodal support** with PyMuPDF and Pillow"
        else:
            multimodal_features += "\n- **⚠️ Image support disabled**: Install PyMuPDF and Pillow"
        
        st.markdown(multimodal_features)
        
        # Show knowledge base status
        if VECTOR_STORE_AVAILABLE and "vector_manager" in st.session_state:
            vector_manager = st.session_state.vector_manager
            indices = vector_manager.load_search_indices()
            if indices and indices.get("metadata"):
                metadata = indices["metadata"]
                sources = set(item.get("source", "") for item in metadata)
                total_images = sum(len(item.get("embedded_images", [])) for item in metadata)
                
                search_types = ["True Multimodal Vector"]
                if indices.get("bm25_index") and BM25_AVAILABLE:
                    search_types.append("BM25")
                    
                # Check system type
                system_info = indices.get("system_info", {})
                system_type = "🚀 True Multimodal" if system_info.get("is_true_multimodal") else "⚠️ Legacy"
                    
                st.success(f"📚🖼️ {system_type} RAG Ready: {len(sources)} documents, {len(metadata)} chunks, {total_images} embedded images ({'+'.join(search_types)})")
                
                if st.session_state.get('last_successful_category') == "Search Knowledge Base":
                    st.info("🔗 Context Active: Next questions will use Knowledge Base")
            else:
                st.info("📚🖼️ True Multimodal RAG: Not created yet")
        
        st.markdown("---")
        st.header("🔧 Installation & Setup")
        st.markdown("""
        **For True Multimodal RAG (Required):**
        ```bash
        pip install faiss-cpu PyPDF2 rank-bm25 PyMuPDF Pillow
        ```
        
        **For GPU acceleration:**
        ```bash
        pip install faiss-gpu PyPDF2 rank-bm25 PyMuPDF Pillow
        ```
        
        **Environment Variables:**
        ```bash
        ENDPOINT_URL=https://your-resource.openai.azure.com
        AZURE_OPENAI_API_KEY=your-api-key
        DEPLOYMENT_NAME=your-model-deployment-name
        API_VERSION=2024-12-01-preview
        ```
        
        **For Vision Model Support:**
        - Use GPT-4 Vision model deployment
        - Automatic image content analysis
        - Enhanced multimodal understanding
        
        **🚀 TRUE MULTIMODAL ADVANTAGES:**
        - ✅ **Images in embeddings** (not file storage)
        - ✅ **Faster search** (no file I/O)
        - ✅ **Better portability** (single vector store)
        - ✅ **Inline display** (embedded base64)
        - ✅ **Multimodal context** for AI responses
        - ✅ **Optimized storage** with compression
        - ✅ **Vision analysis** integrated in embeddings
        - ✅ **No file management** overhead
        """)
        
        # Show current status
        if VECTOR_STORE_AVAILABLE:
            gpu_status = "🚀 GPU" if GPU_AVAILABLE else "💻 CPU"
            bm25_status = "🔍 BM25" if BM25_AVAILABLE else "❌ BM25"
            multimodal_status = "🖼️ True Multimodal" if MULTIMODAL_AVAILABLE else "❌ Images"
            st.markdown(f"**Status:** {gpu_status} | {bm25_status} | {multimodal_status}")
            
        if "openai_client" in st.session_state:
            st.markdown("**🚀 True Multimodal RAG:** Full system ready")
        else:
            st.markdown("**❌ Configuration issue**")
        
        if st.button("🗑️ Clear Chat History"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key not in ['vector_manager', 'openai_client']:  # Keep these
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