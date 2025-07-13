import os
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
import time
import json
import re
import math
from collections import Counter

# RAG dependencies
try:
    import faiss
    import numpy as np
    import PyPDF2
    import tiktoken
    from rank_bm25 import BM25Okapi
    VECTOR_STORE_AVAILABLE = True
    GPU_AVAILABLE = faiss.get_num_gpus() > 0 if 'faiss' in locals() else False
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    GPU_AVAILABLE = False

class ModernRAGManager:
    def __init__(self, openai_client, model_name, data_folder="data", vector_store_path="vector_store"):
        if not VECTOR_STORE_AVAILABLE:
            raise ImportError("Install: pip install faiss-cpu PyPDF2 rank-bm25 tiktoken")
        
        self.openai_client = openai_client
        self.model_name = model_name
        self.data_folder = Path(data_folder)
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True)
        self.data_folder.mkdir(exist_ok=True)
        
        self.use_gpu = GPU_AVAILABLE
        self.text_embedding_model = "text-embedding-3-large"
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.max_tokens_per_chunk = 8000
        self.sentence_window_size = 3
        self.parent_chunk_size = 2048
        self.use_hyde = True
        self.use_query_expansion = True
        self.use_mmr = True
        self.mmr_diversity_score = 0.3
        self.min_chunk_tokens = 32
        self.max_chunks_per_file = 150
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if self.tokenizer else len(text.split()) * 1.3
    
    def recursive_character_text_splitter(self, text: str, separators: List[str] = None) -> List[str]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_text(text: str, separators: List[str]) -> List[str]:
            if not separators:
                return [text]
            
            separator = separators[0]
            if separator == "":
                return list(text)
            
            splits = text.split(separator)
            if len(splits) == 1:
                return split_text(text, separators[1:])
            
            final_splits = []
            for split in splits:
                if self.count_tokens(split) <= self.chunk_size:
                    final_splits.append(split)
                else:
                    sub_splits = split_text(split, separators[1:])
                    final_splits.extend(sub_splits)
            return final_splits
        
        return split_text(text, separators)
    
    def create_chunks_with_overlap(self, text: str) -> List[Dict]:
        sections = text.split('\n\n')
        all_chunks = []
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            splits = self.recursive_character_text_splitter(section)
            current_chunk = ""
            current_tokens = 0
            
            for split in splits:
                split = split.strip()
                if not split:
                    continue
                
                split_tokens = self.count_tokens(split)
                
                if current_tokens + split_tokens > self.chunk_size and current_chunk:
                    if current_tokens >= self.min_chunk_tokens:
                        all_chunks.append(self._create_chunk_dict(current_chunk, section_idx))
                    
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + split if overlap_text else split
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = current_chunk + " " + split if current_chunk else split
                    current_tokens = self.count_tokens(current_chunk)
            
            if current_chunk.strip() and current_tokens >= self.min_chunk_tokens:
                all_chunks.append(self._create_chunk_dict(current_chunk, section_idx))
        
        return all_chunks
    
    def _get_overlap_text(self, text: str) -> str:
        tokens = self.count_tokens(text)
        if tokens <= self.chunk_overlap:
            return text
        
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
        return {
            "content": content.strip(),
            "tokens": self.count_tokens(content),
            "char_count": len(content),
            "section_idx": section_idx,
            "chunk_type": "regular"
        }
    
    def create_parent_child_chunks(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        parent_chunks = []
        words = text.split()
        current_parent = ""
        current_tokens = 0
        parent_idx = 0
        
        for word in words:
            word_tokens = self.count_tokens(word)
            
            if current_tokens + word_tokens > self.parent_chunk_size and current_parent:
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
        
        if current_parent.strip():
            parent_chunks.append({
                "content": current_parent.strip(),
                "tokens": current_tokens,
                "parent_id": parent_idx,
                "chunk_type": "parent"
            })
        
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
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        enhanced_chunks = []
        
        for chunk in chunks:
            chunk_content = chunk["content"]
            chunk_sentences = []
            
            for i, sentence in enumerate(sentences):
                if sentence in chunk_content or chunk_content in sentence:
                    chunk_sentences.append((i, sentence))
            
            if chunk_sentences:
                center_idx = chunk_sentences[len(chunk_sentences)//2][0]
                start_idx = max(0, center_idx - self.sentence_window_size)
                end_idx = min(len(sentences), center_idx + self.sentence_window_size + 1)
                
                window_sentences = sentences[start_idx:end_idx]
                window_content = " ".join(window_sentences)
                
                enhanced_chunk = chunk.copy()
                enhanced_chunk["window_content"] = window_content
                enhanced_chunk["window_tokens"] = self.count_tokens(window_content)
                enhanced_chunk["sentence_range"] = (start_idx, end_idx)
                enhanced_chunks.append(enhanced_chunk)
            else:
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def extract_and_process_text(self, pdf_path: Path) -> Dict:
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"success": False, "message": f"No text extracted from {pdf_path.name}"}
        
        text = self._clean_text(text)
        metadata = self._extract_metadata(text)
        parent_chunks, child_chunks = self.create_parent_child_chunks(text)
        enhanced_chunks = self.create_sentence_windows(child_chunks, text)
        
        for chunk in enhanced_chunks + parent_chunks:
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
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _extract_metadata(self, text: str) -> Dict:
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "token_count": self.count_tokens(text),
            "content_type": self._detect_content_type(text)
        }
    
    def _detect_content_type(self, text: str) -> str:
        text_lower = text.lower()
        patterns = {
            "resume": ["resume", "cv", "experience", "education", "skills"],
            "legal": ["contract", "agreement", "terms", "conditions", "legal"],
            "financial": ["financial", "revenue", "profit", "budget", "investment"],
            "technical": ["technical", "specification", "manual", "guide", "api"],
            "report": ["report", "analysis", "summary", "findings", "conclusion"],
            "academic": ["research", "study", "paper", "journal", "university"],
            "medical": ["medical", "health", "patient", "diagnosis", "treatment"]
        }
        
        scores = {}
        for doc_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[doc_type] = score
        
        return max(scores, key=scores.get) if scores else "general"
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                truncated_batch = []
                for text in batch:
                    if self.count_tokens(text) > self.max_tokens_per_chunk:
                        words = text.split()
                        truncated_text = ""
                        tokens = 0
                        
                        for word in words:
                            word_tokens = self.count_tokens(word)
                            if tokens + word_tokens <= self.max_tokens_per_chunk - 10:
                                truncated_text += " " + word if truncated_text else word
                                tokens += word_tokens
                            else:
                                break
                        
                        truncated_batch.append(truncated_text)
                    else:
                        truncated_batch.append(text)
                
                response = self.openai_client.embeddings.create(
                    model=self.text_embedding_model,
                    input=truncated_batch
                )
                
                batch_embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                for _ in batch:
                    embeddings.append(np.zeros(3072, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
    
    def hyde_query_expansion(self, query: str) -> str:
        if not self.use_hyde:
            return query
        
        try:
            hyde_prompt = f"""Write a detailed paragraph answering: "{query}"
Write as if from a document with specific details and context.
Answer:"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": hyde_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            return f"{query} {hypothetical_doc}"
            
        except:
            return query
    
    def expand_query(self, query: str) -> str:
        if not self.use_query_expansion:
            return query
        
        try:
            expansion_prompt = f"""For query "{query}", provide 3-5 related terms/synonyms:"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.3,
                max_tokens=50
            )
            
            expansion_terms = response.choices[0].message.content.strip()
            return f"{query} {expansion_terms}"
            
        except:
            return query
    
    def maximal_marginal_relevance(self, query_embedding: np.ndarray, embeddings: np.ndarray, 
                                  metadata: List[Dict], k: int, lambda_param: float = 0.7) -> List[Dict]:
        if not self.use_mmr or len(embeddings) == 0:
            return []
        
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        selected_indices = []
        remaining_indices = list(range(len(embeddings)))
        
        if remaining_indices:
            best_idx = remaining_indices[np.argmax(similarities[remaining_indices])]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                relevance = similarities[idx]
                
                if selected_indices:
                    selected_embeddings = embeddings[selected_indices]
                    current_embedding = embeddings[idx]
                    
                    diversities = np.dot(selected_embeddings, current_embedding) / (
                        np.linalg.norm(selected_embeddings, axis=1) * np.linalg.norm(current_embedding)
                    )
                    max_similarity = np.max(diversities)
                else:
                    max_similarity = 0
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)
            
            if mmr_scores:
                best_mmr_idx = np.argmax(mmr_scores)
                selected_idx = remaining_indices[best_mmr_idx]
                selected_indices.append(selected_idx)
                remaining_indices.remove(selected_idx)
        
        results = []
        for idx in selected_indices:
            if idx < len(metadata):
                result = metadata[idx].copy()
                result["similarity_score"] = float(similarities[idx])
                result["mmr_selected"] = True
                results.append(result)
        
        return results
    
    def create_vector_store(self) -> Dict:
        pdf_files = list(self.data_folder.glob("*.pdf"))
        
        if not pdf_files:
            return {"success": False, "message": "No PDF files found"}
        
        max_files = 25
        if len(pdf_files) > max_files:
            pdf_files = pdf_files[:max_files]
        
        all_child_chunks = []
        all_parent_chunks = []
        processing_stats = {"files": 0, "chunks": 0, "tokens": 0}
        
        for file_idx, pdf_file in enumerate(pdf_files):
            result = self.extract_and_process_text(pdf_file)
            
            if not result["success"]:
                continue
            
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
        
        if not all_child_chunks:
            return {"success": False, "message": "No content extracted"}
        
        max_total_chunks = self.max_chunks_per_file * len(pdf_files)
        if len(all_child_chunks) > max_total_chunks:
            all_child_chunks.sort(key=lambda x: x.get("tokens", 0), reverse=True)
            all_child_chunks = all_child_chunks[:max_total_chunks]
            processing_stats["chunks"] = len(all_child_chunks)
        
        try:
            embed_texts = []
            for chunk in all_child_chunks:
                text = chunk.get("window_content", chunk["content"])
                embed_texts.append(text)
            
            embeddings = self.create_embeddings_batch(embed_texts)
            indices = self.create_search_indices(embeddings, all_child_chunks, all_parent_chunks)
            self.save_indices(indices)
            
            return {
                "success": True,
                "message": f"Created: {processing_stats['chunks']} chunks from {processing_stats['files']} files",
                "chunks": processing_stats["chunks"],
                "files": processing_stats["files"],
                "tokens": processing_stats["tokens"]
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def create_search_indices(self, embeddings: np.ndarray, child_chunks: List[Dict], parent_chunks: List[Dict]) -> Dict:
        if not child_chunks:
            return {"vector_index": None, "bm25_index": None, "metadata": [], "parent_metadata": []}
        
        child_metadata = []
        for i, chunk in enumerate(child_chunks):
            metadata = {
                "content": chunk["content"],
                "source": chunk.get("source", ""),
                "chunk_id": i,
                "tokens": chunk.get("tokens", 0),
                "chunk_type": chunk.get("chunk_type", "child"),
                "parent_id": chunk.get("parent_id", -1),
                "window_content": chunk.get("window_content", ""),
                "document_metadata": chunk.get("document_metadata", {}),
                "file_index": chunk.get("file_index", 0)
            }
            child_metadata.append(metadata)
        
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
        
        vector_index = self.create_faiss_index(embeddings)
        
        bm25_index = None
        tokenized_docs = None
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
        if embeddings.size == 0:
            return None
        
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        faiss.normalize_L2(embeddings)
        
        if n_vectors < 1000:
            index = faiss.IndexFlatIP(dimension)
        elif n_vectors < 50000:
            nlist = min(int(math.sqrt(n_vectors)), 1000)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
        else:
            nlist = min(int(math.sqrt(n_vectors)), 2000)
            m = 16
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            index.train(embeddings)
        
        if self.use_gpu and n_vectors > 1000:
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu_index.add(embeddings)
                index = faiss.index_gpu_to_cpu(gpu_index)
            except:
                index.add(embeddings)
        else:
            index.add(embeddings)
        
        return index
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [token for token in text.split() if len(token) > 2]
        return tokens
    
    def modern_search(self, query: str, k: int = 8) -> List[Dict]:
        indices = self.load_indices()
        if not indices or not indices["metadata"]:
            return []
        
        vector_index = indices["vector_index"]
        bm25_index = indices["bm25_index"]
        metadata = indices["metadata"]
        
        enhanced_query = self.hyde_query_expansion(query)
        expanded_query = self.expand_query(enhanced_query)
        
        all_results = []
        
        if vector_index:
            vector_results = self._modern_vector_search(expanded_query, vector_index, metadata, k * 3)
            all_results.extend(vector_results)
        
        if bm25_index:
            bm25_results = self._bm25_search(query, bm25_index, metadata, k * 2)
            all_results.extend(bm25_results)
        
        if self.use_mmr and vector_index and all_results:
            try:
                response = self.openai_client.embeddings.create(
                    model=self.text_embedding_model,
                    input=expanded_query[:8000]
                )
                query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
                
                result_indices = [r["chunk_id"] for r in all_results if "chunk_id" in r]
                if result_indices:
                    result_embeddings = np.array([vector_index.reconstruct(i) for i in result_indices])
                    mmr_results = self.maximal_marginal_relevance(
                        query_embedding, result_embeddings, all_results, k, self.mmr_diversity_score
                    )
                    
                    if mmr_results:
                        return mmr_results
            except:
                pass
        
        if all_results:
            return self._combine_results(all_results, k)[:k]
        
        return []
    
    def _modern_vector_search(self, query: str, index, metadata: List[Dict], k: int) -> List[Dict]:
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
                    result["similarity_score"] = float(scores[0][i])
                    result["search_type"] = "vector"
                    results.append(result)
            
            return results
            
        except:
            return []
    
    def _bm25_search(self, query: str, bm25_index, metadata: List[Dict], k: int) -> List[Dict]:
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
            
        except:
            return []
    
    def _combine_results(self, results: List[Dict], k: int) -> List[Dict]:
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            chunk_id = result.get("chunk_id", -1)
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        def get_score(result):
            if "similarity_score" in result:
                return result["similarity_score"]
            elif "bm25_score" in result:
                return result["bm25_score"] / 10.0
            return 0
        
        unique_results.sort(key=get_score, reverse=True)
        return unique_results
    
    def get_parent_context(self, child_result: Dict, parent_metadata: List[Dict]) -> str:
        parent_id = child_result.get("parent_id", -1)
        if parent_id >= 0:
            for parent in parent_metadata:
                if parent.get("parent_id") == parent_id:
                    return parent.get("content", "")
        return child_result.get("content", "")
    
    def display_results(self, results: List[Dict]):
        if not results:
            print("No results found.")
            return
        
        indices = self.load_indices()
        parent_metadata = indices.get("parent_metadata", [])
        
        for i, result in enumerate(results[:3], 1):
            source = result.get('source', 'Unknown')
            print(f"\n--- Result {i} ({source}) ---")
            
            content = result.get("content", "")
            print(content[:300] + "..." if len(content) > 300 else content)
            
            if result.get("similarity_score"):
                print(f"Score: {result['similarity_score']:.3f}")
    
    def save_indices(self, indices: Dict):
        try:
            if indices["vector_index"]:
                vector_path = self.vector_store_path / "faiss_index.bin"
                faiss.write_index(indices["vector_index"], str(vector_path))
            
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
                    "use_hyde": self.use_hyde,
                    "use_query_expansion": self.use_query_expansion,
                    "use_mmr": self.use_mmr,
                    "text_embedding_model": self.text_embedding_model
                }
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(save_data, f)
            
        except Exception as e:
            raise
    
    def load_indices(self) -> Dict:
        vector_path = self.vector_store_path / "faiss_index.bin"
        metadata_path = self.vector_store_path / "modern_metadata.pkl"
        
        if not (vector_path.exists() and metadata_path.exists()):
            return {}
        
        try:
            vector_index = faiss.read_index(str(vector_path))
            
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
            
        except:
            return {}
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                max_pages = min(250, len(pdf_reader.pages))
                text_parts = []
                
                for page in pdf_reader.pages[:max_pages]:
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            cleaned_text = self._clean_text(text)
                            if len(cleaned_text) > 50:
                                text_parts.append(cleaned_text)
                    except:
                        continue
                
                return "\n\n".join(text_parts) if text_parts else ""
                
        except:
            return ""

class ToolPlugin:
    @kernel_function(description="Logs a request to report sick.", name="report_sick")
    def report_sick(self):
        return "Request to report sick has been logged."

    @kernel_function(description="Logs a request to book a hotel.", name="book_hotel")
    def book_hotel(self):
        return "Request to book a hotel has been logged."

    @kernel_function(description="Logs a request to book a limo.", name="book_limo")
    def book_limo(self):
        return "Request to book a limo has been logged."

    @kernel_function(description="Logs a request to report fatigue.", name="report_fatigue")
    def report_fatigue(self):
        return "Request to report fatigue has been logged."

    @kernel_function(description="Handles knowledge base queries.", name="search_knowledge_base")
    def search_knowledge_base(self):
        return "Knowledge base search request has been logged."

def categorize_with_openai(openai_client, user_input, model_name, last_category=None):
    context_hint = ""
    if last_category == "Search Knowledge Base":
        context_hint = "\n\nNote: Previous was knowledge base search, follow-ups likely same category."
    
    categorization_prompt = f"""Categorize this request:

Categories:
1. Report Sick - health, illness, unwell, pain
2. Report Fatigue - tired, exhaustion, fatigue  
3. Book Hotel - hotel, accommodation, rooms
4. Book Limo - transportation, rides, taxi
5. Search Knowledge Base - documents, PDFs, information lookup, research

Request: "{user_input}"{context_hint}

Reply with exact category name or "no_match":"""

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": categorization_prompt}],
            temperature=0.1,
            max_tokens=50
        )
        
        category = response.choices[0].message.content.strip()
        valid_categories = ["Report Sick", "Report Fatigue", "Book Hotel", "Book Limo", "Search Knowledge Base", "no_match"]
        
        if category in valid_categories:
            return category if category != "no_match" else None
        
        category_lower = category.lower()
        if "sick" in category_lower or "health" in category_lower:
            return "Report Sick"
        elif "fatigue" in category_lower or "tired" in category_lower:
            return "Report Fatigue"
        elif "hotel" in category_lower:
            return "Book Hotel"
        elif "limo" in category_lower or "transport" in category_lower:
            return "Book Limo"
        elif any(word in category_lower for word in ["search", "document", "find", "knowledge", "what", "how"]):
            return "Search Knowledge Base"
        
        if last_category == "Search Knowledge Base":
            question_indicators = ["what", "who", "where", "when", "how", "which", "he", "she", "they"]
            if any(indicator in user_input.lower().split() for indicator in question_indicators):
                return "Search Knowledge Base"
        
        return None
        
    except:
        return None

def initialize_chatbot():
    load_dotenv()
    
    endpoint = os.getenv("ENDPOINT_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    model_name = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")
    
    if endpoint and "/openai/deployments/" in endpoint:
        base_endpoint = endpoint.split("/openai/deployments/")[0]
    else:
        base_endpoint = endpoint
    
    if not base_endpoint or not api_key or not model_name:
        return None, None, None
    
    openai_client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version or "2024-12-01-preview",
        azure_endpoint=base_endpoint
    )
    
    kernel = sk.Kernel()
    chat_service = AzureAIInferenceChatCompletion(
        ai_model_id=model_name,
        endpoint=base_endpoint,
        api_key=api_key,
    )
    kernel.add_service(chat_service)
    kernel.add_plugin(ToolPlugin(), "ToolPlugin")
    
    return kernel, chat_service, openai_client

def handle_knowledge_base(query, manager, openai_client, model_name):
    indices = manager.load_indices()
    if not indices or not indices.get("metadata"):
        return "No RAG system found. Create one first with 'setup'."
    
    results = manager.modern_search(query, k=5)
    
    if results:
        manager.display_results(results)
        
        try:
            context_parts = []
            for result in results[:3]:
                content = result.get("window_content", result.get("content", ""))
                context_parts.append(content)
            
            context = "\n\n".join(context_parts)
            
            if context:
                summary_prompt = f"""Based on this content, answer: "{query}"

Content:
{context}

Answer:"""

                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.05,
                    max_tokens=400
                )
                
                ai_summary = response.choices[0].message.content.strip()
                print(f"\nAnswer: {ai_summary}")
                return ai_summary
                
        except:
            pass
        
        return f"Found {len(results)} relevant sections."
    else:
        return "No relevant information found."

def handle_agent(category, original_query=""):
    if category == "Report Sick":
        from_date = input("From date (YYYY-MM-DD): ").strip()
        to_date = input("To date (YYYY-MM-DD): ").strip()
        if from_date and to_date:
            return f"Sick leave logged: {from_date} to {to_date}"
        return "Invalid dates provided."
        
    elif category == "Book Hotel":
        from_date = input("Check-in (YYYY-MM-DD): ").strip()
        to_date = input("Check-out (YYYY-MM-DD): ").strip()
        if from_date and to_date:
            return f"Hotel booking logged: {from_date} to {to_date}"
        return "Invalid dates provided."
        
    elif category == "Report Fatigue":
        confirm = input("Submit fatigue report? (y/n): ").strip().lower()
        return "Fatigue report logged." if confirm == 'y' else "Cancelled."
        
    elif category == "Book Limo":
        confirm = input("Submit limo booking? (y/n): ").strip().lower()
        return "Limo booking logged." if confirm == 'y' else "Cancelled."
        
    elif category == "Search Knowledge Base":
        return "Knowledge base search initiated."
    
    return "Unknown category."

def main():
    kernel, chat_service, openai_client = initialize_chatbot()
    
    if kernel is None:
        print("Failed to initialize. Check environment variables.")
        return
    
    model_name = os.getenv("DEPLOYMENT_NAME")
    
    try:
        test_response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        return
    
    if VECTOR_STORE_AVAILABLE:
        try:
            modern_manager = ModernRAGManager(openai_client, model_name)
        except:
            modern_manager = None
    else:
        modern_manager = None
    
    last_category = None
    
    print("Multi-Agent CLI Chatbot")
    print("Commands: 'setup' (create RAG), 'quit' (exit)")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() == 'quit':
            break
            
        if user_input.lower() == 'setup':
            if modern_manager is None:
                print("RAG dependencies not available.")
                continue
                
            data_folder = Path("data")
            pdf_files = list(data_folder.glob("*.pdf")) if data_folder.exists() else []
            
            if not pdf_files:
                print("No PDF files in 'data' folder.")
                continue
                
            print("Creating RAG system...")
            result = modern_manager.create_vector_store()
            print(result["message"])
            continue
        
        category = categorize_with_openai(openai_client, user_input, model_name, last_category)
        
        if category:
            print(f"Category: {category}")
            confirm = input("Connect to agent? (y/n): ").strip().lower()
            
            if confirm == 'y':
                if category == "Search Knowledge Base" and modern_manager:
                    result = handle_knowledge_base(user_input, modern_manager, openai_client, model_name)
                else:
                    result = handle_agent(category, user_input)
                
                print(f"Result: {result}")
                last_category = category
            else:
                print("Cancelled.")
        else:
            print("Cannot categorize request. Try: sick, tired, hotel, transport, or document questions.")

if __name__ == "__main__":
    main()