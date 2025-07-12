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
        
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    GPU_AVAILABLE = False

class OptimizedVectorStoreManager:
    """Highly optimized FAISS vector store manager with performance improvements."""
    
    def __init__(self, openai_client, model_name, data_folder="data", vector_store_path="vector_store"):
        if not VECTOR_STORE_AVAILABLE:
            raise ImportError("Vector store dependencies not available. Install with: pip install faiss-cpu PyPDF2")
        
        self.openai_client = openai_client
        self.model_name = model_name
        self.data_folder = Path(data_folder)
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True)
        self.data_folder.mkdir(exist_ok=True)
        
        # Optimized settings
        self.use_gpu = GPU_AVAILABLE
        self.embedding_model = "text-embedding-ada-002"
        
        if self.use_gpu:
            st.info(f"🚀 GPU acceleration enabled! Found {faiss.get_num_gpus()} GPU(s)")
        else:
            st.info("💻 Using CPU for vector operations")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Optimized PDF text extraction."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Process more pages but efficiently
                max_pages = min(100, len(pdf_reader.pages))  # Increased from 50
                text_parts = []
                
                for page in pdf_reader.pages[:max_pages]:
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_parts.append(text.strip())
                    except:
                        continue
                
                full_text = "\n".join(text_parts)
                
                if len(pdf_reader.pages) > max_pages:
                    st.info(f"Processed {max_pages} of {len(pdf_reader.pages)} pages")
                
                return full_text
                
        except Exception as e:
            st.error(f"Error reading PDF {pdf_path.name}: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 300) -> List[str]:
        """Optimized text chunking with better parameters for RAG."""
        if not text.strip():
            return []
        
        # More reasonable limits
        max_total_length = 100000
        if len(text) > max_total_length:
            text = text[:max_total_length]
            st.info(f"Text truncated to {max_total_length} characters")
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Find a good break point (sentence or paragraph end)
            if end < text_len:
                for break_char in ['\n\n', '. ', '! ', '? ', '\n']:
                    break_pos = text.rfind(break_char, start + chunk_size//2, end)
                    if break_pos > start:
                        end = break_pos + len(break_char)
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:  # Lower minimum chunk size
                chunks.append(chunk)
            
            if end >= text_len:
                break
                
            start = end - overlap
        
        # More reasonable chunk limit
        max_chunks = 150  # Increased for better coverage
        if len(chunks) > max_chunks:
            st.warning(f"Using first {max_chunks} chunks out of {len(chunks)}")
            chunks = chunks[:max_chunks]
        
        return chunks
    
    def get_embeddings_batch_optimized(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """Highly optimized batch embedding generation."""
        if not texts:
            return np.array([])
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Create a single progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i:i + batch_size]
            
            # Update status less frequently
            status_text.text(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            try:
                # Prepare batch with length limits
                processed_batch = []
                for text in batch:
                    # More generous token limit
                    if len(text) > 8000:
                        text = text[:8000]
                    processed_batch.append(text)
                
                # Single API call for entire batch
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=processed_batch
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = min((i + batch_size) / len(texts), 1.0)
                progress_bar.progress(progress)
                
            except Exception as e:
                st.warning(f"Error in batch {batch_num}: {str(e)}")
                # Add zero vectors for failed batch
                for _ in batch:
                    embeddings.append([0.0] * 1536)
        
        # Show completion stats
        elapsed_time = time.time() - start_time
        status_text.text(f"✅ Generated {len(embeddings)} embeddings in {elapsed_time:.1f}s")
        progress_bar.progress(1.0)
        
        return np.array(embeddings, dtype=np.float32)
    
    def create_optimized_index(self, embeddings: np.ndarray):
        """Create optimized FAISS index."""
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        # Choose index type based on data size
        if n_vectors < 1000:
            # Use simple flat index for small datasets
            index = faiss.IndexFlatL2(dimension)
        else:
            # Use IVF index for larger datasets
            nlist = min(100, n_vectors // 10)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            index.train(embeddings)
        
        # Add vectors to index
        if self.use_gpu and n_vectors > 500:  # Use GPU for larger datasets
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu_index.add(embeddings)
                # Convert back to CPU for saving
                index = faiss.index_gpu_to_cpu(gpu_index)
                st.success("🚀 Used GPU acceleration for indexing!")
            except Exception as e:
                st.warning(f"GPU indexing failed: {str(e)}, using CPU")
                index.add(embeddings)
        else:
            index.add(embeddings)
        
        return index
    
    def create_vector_store(self) -> Dict:
        """Optimized vector store creation with performance improvements."""
        pdf_files = list(self.data_folder.glob("*.pdf"))
        
        if not pdf_files:
            return {"success": False, "message": "No PDF files found in data folder"}
        
        # Process more files but with better memory management
        max_files = 10  # Increased from 1
        if len(pdf_files) > max_files:
            st.warning(f"Processing first {max_files} of {len(pdf_files)} files for optimal performance")
            pdf_files = pdf_files[:max_files]
        
        all_texts = []
        all_metadata = []
        
        # Main progress tracking
        main_progress = st.progress(0)
        main_status = st.empty()
        
        start_time = time.time()
        
        # Phase 1: Extract and chunk text (30% of progress)
        main_status.text("📄 Extracting and chunking text from PDFs...")
        
        for file_idx, pdf_file in enumerate(pdf_files):
            # Check reasonable file size limit
            file_size = pdf_file.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB limit (increased from 1MB)
                st.warning(f"Large file {pdf_file.name} ({file_size//1024//1024}MB) - processing first 100 pages")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                continue
            
            # Chunk text with optimized parameters
            chunks = self.chunk_text(text)
            
            # Add chunks and metadata
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadata.append({
                    "source": pdf_file.name,
                    "chunk_id": i,
                    "content": chunk[:300]  # Store more content in metadata
                })
            
            # Update progress
            file_progress = (file_idx + 1) / len(pdf_files) * 0.3
            main_progress.progress(file_progress)
        
        if not all_texts:
            return {"success": False, "message": "No text extracted from PDF files"}
        
        main_status.text(f"📊 Processing {len(all_texts)} text chunks...")
        
        # Phase 2: Generate embeddings (60% of progress)
        main_progress.progress(0.3)
        
        try:
            embeddings = self.get_embeddings_batch_optimized(all_texts)
            main_progress.progress(0.9)
        except Exception as e:
            return {"success": False, "message": f"Error generating embeddings: {str(e)}"}
        
        # Phase 3: Create index (10% of progress)
        main_status.text("🔧 Creating optimized search index...")
        
        try:
            index = self.create_optimized_index(embeddings)
            main_progress.progress(0.95)
        except Exception as e:
            return {"success": False, "message": f"Error creating index: {str(e)}"}
        
        # Phase 4: Save everything
        main_status.text("💾 Saving vector store...")
        
        try:
            index_path = self.vector_store_path / "faiss_index.bin"
            metadata_path = self.vector_store_path / "metadata.pkl"
            
            faiss.write_index(index, str(index_path))
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(all_metadata, f)
                
        except Exception as e:
            return {"success": False, "message": f"Error saving: {str(e)}"}
        
        # Complete
        total_time = time.time() - start_time
        main_progress.progress(1.0)
        main_status.text(f"✅ Vector store created in {total_time:.1f}s!")
        
        return {
            "success": True, 
            "message": f"Vector store created with {len(all_texts)} chunks from {len(pdf_files)} files in {total_time:.1f}s",
            "chunks": len(all_texts),
            "files": len(pdf_files),
            "time": total_time
        }
    
    def load_vector_store(self) -> tuple:
        """Load existing vector store."""
        index_path = self.vector_store_path / "faiss_index.bin"
        metadata_path = self.vector_store_path / "metadata.pkl"
        
        if not (index_path.exists() and metadata_path.exists()):
            return None, None
        
        try:
            index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            return index, metadata
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None, None
    
    def enhanced_search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Enhanced similarity search with multiple strategies for better RAG quality."""
        index, metadata = self.load_vector_store()
        
        if index is None or metadata is None:
            return []
        
        try:
            # Strategy 1: Direct vector search with original query
            results = self._vector_search(query, k * 2)  # Get more results initially
            
            # Strategy 2: Query expansion for better matching
            expanded_queries = self._expand_query(query)
            for expanded_query in expanded_queries:
                expanded_results = self._vector_search(expanded_query, k)
                results.extend(expanded_results)
            
            # Strategy 3: Keyword-based fallback search
            keyword_results = self._keyword_search(query, metadata, k)
            results.extend(keyword_results)
            
            # Remove duplicates and sort by relevance
            seen_indices = set()
            unique_results = []
            for result in results:
                idx = result.get('chunk_id', -1)
                source = result.get('source', '')
                key = f"{source}_{idx}"
                if key not in seen_indices:
                    seen_indices.add(key)
                    unique_results.append(result)
            
            # Sort by similarity score (lower is better for L2 distance)
            unique_results.sort(key=lambda x: x.get('similarity_score', float('inf')))
            
            # Return top k results with more lenient threshold
            return unique_results[:k]
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "DeploymentNotFound" in error_msg:
                st.error("❌ Embedding service unavailable - check your Azure OpenAI deployment configuration")
            elif "401" in error_msg:
                st.error("❌ Authentication failed - check your API key and endpoint")
            else:
                st.error(f"❌ Search error: {error_msg}")
            return []
    
    def _vector_search(self, query: str, k: int) -> List[Dict]:
        """Perform vector similarity search."""
        index, metadata = self.load_vector_store()
        
        if index is None or metadata is None:
            return []
        
        try:
            # Get query embedding
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[query[:8000]]
            )
            
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            
            # Search with larger k to get more candidates
            search_k = min(k * 3, index.ntotal)
            distances, indices = index.search(query_embedding, search_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata) and idx >= 0:
                    result = metadata[idx].copy()
                    result["similarity_score"] = float(distances[0][i])
                    results.append(result)
            
            return results
            
        except Exception:
            return []
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and variations for better matching."""
        expanded = []
        
        # Common expansions for CV/resume queries
        experience_terms = ["experience", "years", "work", "career", "professional", "background"]
        skill_terms = ["skills", "technologies", "expertise", "proficient", "knowledge"]
        education_terms = ["education", "degree", "university", "college", "study"]
        
        query_lower = query.lower()
        
        # Expand experience-related queries
        if any(term in query_lower for term in ["experience", "years"]):
            expanded.extend([
                "professional experience years",
                "work experience career",
                "years of experience",
                "professional background"
            ])
        
        # Expand skill-related queries  
        if any(term in query_lower for term in ["skill", "technology", "tech"]):
            expanded.extend([
                "technical skills technologies",
                "expertise proficient",
                "programming languages tools"
            ])
        
        # Expand education queries
        if any(term in query_lower for term in ["education", "degree", "study"]):
            expanded.extend([
                "education degree university",
                "academic background",
                "qualifications studies"
            ])
        
        # Add simplified versions
        words = query.split()
        if len(words) > 2:
            # Try with just the key terms
            key_words = [w for w in words if len(w) > 3 and w.lower() not in ["what", "how", "many", "does", "have", "the", "is"]]
            if key_words:
                expanded.append(" ".join(key_words))
        
        return expanded[:3]  # Limit to 3 expansions
    
    def _keyword_search(self, query: str, metadata: List[Dict], k: int) -> List[Dict]:
        """Fallback keyword-based search for when vector search fails."""
        query_words = query.lower().split()
        query_words = [w for w in query_words if len(w) > 2]  # Filter short words
        
        scored_results = []
        
        for i, item in enumerate(metadata):
            content = item.get('content', '').lower()
            
            # Simple keyword matching score
            score = 0
            for word in query_words:
                if word in content:
                    score += content.count(word)
            
            if score > 0:
                result = item.copy()
                result['similarity_score'] = 1.0 / (score + 1)  # Convert to similarity format (lower is better)
                result['search_type'] = 'keyword'
                scored_results.append(result)
        
        # Sort by score (lower similarity_score is better)
        scored_results.sort(key=lambda x: x['similarity_score'])
        
        return scored_results[:k]
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Main search function - use enhanced search."""
        return self.enhanced_search_similar(query, k)

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

def create_optimized_vector_store_ui(openai_client, model_name):
    """Enhanced RAG UI for vector store management."""
    st.sidebar.markdown("---")
    st.sidebar.header("📚 Enhanced RAG Knowledge Base")
    
    # Check if vector store dependencies are available
    if not VECTOR_STORE_AVAILABLE:
        st.sidebar.error("❌ Vector store dependencies not installed")
        st.sidebar.code("pip install faiss-cpu PyPDF2")
        return None
    
    # Initialize optimized vector store manager
    if "vector_manager" not in st.session_state:
        try:
            st.session_state.vector_manager = OptimizedVectorStoreManager(openai_client, model_name)
        except Exception as e:
            st.sidebar.error(f"Error initializing vector store: {str(e)}")
            return None
    
    vector_manager = st.session_state.vector_manager
    
    # Check if data folder exists and show PDF count
    data_folder = Path("data")
    pdf_files = list(data_folder.glob("*.pdf")) if data_folder.exists() else []
    
    st.sidebar.markdown(f"**PDF Files in Data Folder:** {len(pdf_files)}")
    
    # Show improved limits
    if len(pdf_files) > 10:
        st.sidebar.warning(f"⚠️ {len(pdf_files)} PDF files found. First 10 will be processed for optimal performance.")
    
    if pdf_files:
        with st.sidebar.expander("📄 Available PDF Files", expanded=False):
            for i, pdf_file in enumerate(pdf_files[:15]):  # Show more in list
                status = "✅" if i < 10 else "⏳"
                file_size = pdf_file.stat().st_size / 1024  # KB
                st.write(f"{status} {pdf_file.name} ({file_size:.0f}KB)")
            if len(pdf_files) > 15:
                st.write(f"... and {len(pdf_files) - 15} more files")
    else:
        st.sidebar.warning("⚠️ No PDF files found in 'data' folder")
        st.sidebar.markdown("*Place PDF files in the 'data' folder to create vector store*")
    
    # Enhanced RAG performance tips
    st.sidebar.info("⚡ **Enhanced RAG Features:**\n- Context-aware follow-up questions\n- Multi-strategy search (vector + keyword + expansion)\n- Process up to 10 files\n- Files up to 10MB supported\n- Better chunking with overlap\n- Query expansion for better matching\n- Lenient similarity thresholds")
    
    # Check if vector store exists
    vector_store_exists = vector_manager.load_vector_store()[0] is not None
    
    if vector_store_exists:
        st.sidebar.success("✅ Vector store exists")
        
        # Show vector store info
        try:
            index, metadata = vector_manager.load_vector_store()
            if index and metadata:
                st.sidebar.markdown(f"**Chunks:** {len(metadata)}")
                st.sidebar.markdown(f"**Index Size:** {index.ntotal} vectors")
                
                # Get unique sources
                sources = set(item["source"] for item in metadata)
                st.sidebar.markdown(f"**Sources:** {len(sources)}")
        except:
            pass
    else:
        st.sidebar.info("ℹ️ No vector store found")
    
    # Create/Recreate vector store button
    button_text = "🔄 Recreate Enhanced RAG Store" if vector_store_exists else "⚡ Create Enhanced RAG Store"
    
    if st.sidebar.button(button_text, disabled=(len(pdf_files) == 0)):
        if len(pdf_files) == 0:
            st.sidebar.error("❌ No PDF files found to process")
        else:
            with st.sidebar:
                st.markdown("### 🚀 Creating Enhanced RAG System")
                try:
                    result = vector_manager.create_vector_store()
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.balloons()
                    else:
                        st.error(f"❌ {result['message']}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Enhanced RAG search interface
    if vector_store_exists:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Enhanced RAG Search")
        
        search_query = st.sidebar.text_input("Search query:", placeholder="Search in PDFs...")
        
        if st.sidebar.button("🔍 Enhanced Search") and search_query:
            with st.sidebar:
                with st.spinner("Searching..."):
                    try:
                        start_time = time.time()
                        results = vector_manager.search_similar(search_query, k=3)
                        search_time = time.time() - start_time
                        
                        if results:
                            st.success(f"Found {len(results)} results in {search_time:.2f}s")
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Result {i} - {result['source']}", expanded=(i==1)):
                                    st.write(f"**Source:** {result['source']}")
                                    st.write(f"**Content:** {result['content']}")
                                    st.write(f"**Similarity:** {result['similarity_score']:.4f}")
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
            st.error("❌ Vector store not initialized. Please create a vector store first.")
            if st.button("Return to Main Agent"):
                return True
            return False
        
        vector_manager = st.session_state.vector_manager
        index, metadata = vector_manager.load_vector_store()
        
        if index is None or metadata is None:
            st.error("❌ No vector store found. Please create a vector store first by uploading PDFs.")
            if st.button("Return to Main Agent"):
                return True
            return False
        
        # Check if we have the original user query from routing
        original_query = st.session_state.get('original_user_query', '')
        
        if original_query and not st.session_state.get('kb_query_processed', False):
            # Process the original query directly
            with st.spinner("🔍 Searching..."):
                try:
                    start_time = time.time()
                    
                    # Check if this is a general query about the knowledge base
                    general_queries = ["what information", "what's in", "what is in", "what does the document contain", "what topics", "summarize", "overview", "what's available"]
                    is_general_query = any(phrase in original_query.lower() for phrase in general_queries)
                    
                    if is_general_query:
                        # For general queries, get more results
                        results = vector_manager.search_similar(original_query, k=15)
                        similarity_threshold = 5.0  # Very lenient for general queries
                    else:
                        # For specific queries, use enhanced search
                        results = vector_manager.search_similar(original_query, k=10)
                        similarity_threshold = 3.0  # Much more lenient
                    
                    search_time = time.time() - start_time
                    
                    # Always try to use results if we have any, regardless of similarity score
                    if results:
                        # Create a simple, clean answer
                        answer_parts = []
                        
                        # Try to generate AI summary first
                        ai_summary_generated = False
                        try:
                            openai_client = st.session_state.get('openai_client')
                            model_name = os.getenv("DEPLOYMENT_NAME")
                            
                            if openai_client and model_name:
                                # Use more content for better context
                                num_contexts = 6 if is_general_query else 4
                                context = "\n\n".join([r['content'] for r in results[:num_contexts]])
                                if len(context) > 8000:  # Increased limit
                                    context = context[:8000] + "..."
                                
                                if is_general_query:
                                    summary_prompt = f"""Based on the following content from documents, provide a comprehensive overview of what information is available in the knowledge base.

User Question: {original_query}

Document Content:
{context}

Please provide a helpful summary of the main topics, types of information, and key content available in these documents. Be comprehensive and informative."""
                                else:
                                    summary_prompt = f"""Based on the following content from documents, provide a clear and helpful answer to the user's question. Focus on directly answering what was asked.

User Question: {original_query}

Relevant Content:
{context}

Please provide a direct, accurate answer based on the content above. Extract specific facts and details that answer the question."""

                                response = openai_client.chat.completions.create(
                                    model=model_name,
                                    messages=[{"role": "user", "content": summary_prompt}],
                                    temperature=0.1,  # Lower temperature for more factual responses
                                    max_tokens=700
                                )
                                
                                ai_summary = response.choices[0].message.content.strip()
                                answer_parts.append(ai_summary)
                                ai_summary_generated = True
                                
                        except Exception as e:
                            # Silently fall back to showing raw results
                            pass
                        
                        # If no AI summary, show the most relevant content
                        if not ai_summary_generated:
                            if is_general_query:
                                answer_parts.append("Based on your documents, here's what information is available:")
                                # Show more results for general queries
                                for i, result in enumerate(results[:5], 1):
                                    answer_parts.append(f"\n**Section {i} (from {result['source']}):**")
                                    # Show more content for overview
                                    content = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
                                    answer_parts.append(content)
                            else:
                                answer_parts.append("Based on your documents:")
                                # Show top 3 results with full content
                                for i, result in enumerate(results[:3], 1):
                                    answer_parts.append(f"\n**From {result['source']}:**")
                                    answer_parts.append(result['content'])
                        
                        final_answer = "\n".join(answer_parts)
                        
                        st.session_state.agent_complete = True
                        st.session_state.agent_result = final_answer
                        st.session_state.kb_query_processed = True
                        return True
                        
                    else:
                        # If no results found, show sample content from the knowledge base
                        try:
                            index, metadata = vector_manager.load_vector_store()
                            if metadata and len(metadata) > 0:
                                sample_content = []
                                sample_content.append("I couldn't find specific matches for your query, but here's a sample of what's in your knowledge base:")
                                
                                # Show first few chunks as samples
                                for i, item in enumerate(metadata[:4], 1):
                                    sample_content.append(f"\n**Sample {i} (from {item['source']}):**")
                                    content = item['content'][:400] + "..." if len(item['content']) > 400 else item['content']
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
                    with st.spinner("🔍 Searching..."):
                        try:
                            results = vector_manager.search_similar(query.strip(), k=8)
                            
                            # Always try to use results if we have any
                            if results:
                                answer_parts = []
                                
                                # Try AI summary first
                                ai_summary_generated = False
                                try:
                                    openai_client = st.session_state.get('openai_client')
                                    model_name = os.getenv("DEPLOYMENT_NAME")
                                    
                                    if openai_client and model_name:
                                        context = "\n\n".join([r['content'] for r in results[:4]])
                                        if len(context) > 8000:
                                            context = context[:8000] + "..."
                                        
                                        summary_prompt = f"""Based on the following content from documents, provide a clear and helpful answer to the user's question. Focus on directly answering what was asked.

User Question: {query}

Relevant Content:
{context}

Please provide a direct, accurate answer based on the content above. Extract specific facts and details that answer the question."""

                                        response = openai_client.chat.completions.create(
                                            model=model_name,
                                            messages=[{"role": "user", "content": summary_prompt}],
                                            temperature=0.1,
                                            max_tokens=700
                                        )
                                        
                                        ai_summary = response.choices[0].message.content.strip()
                                        answer_parts.append(ai_summary)
                                        ai_summary_generated = True
                                        
                                except:
                                    pass
                                
                                # Fallback to raw results
                                if not ai_summary_generated:
                                    answer_parts.append("Based on your documents:")
                                    for i, result in enumerate(results[:3], 1):
                                        answer_parts.append(f"\n**From {result['source']}:**")
                                        answer_parts.append(result['content'])
                                
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

    st.title("🚀 Multi-Agent Assistant with Enhanced RAG")
    st.markdown("Featuring **ultra-fast vector store creation**, **advanced multi-strategy RAG search**, **context-aware conversations**, and optimized processing for all your requests.")

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

    # Add optimized vector store UI to sidebar
    vector_manager = create_optimized_vector_store_ui(openai_client, model_name)
    
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
        st.header("🚀 Enhanced Multi-Agent Services")
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
        
        📚 **Enhanced RAG Search**
        - Multi-strategy document search (vector + keyword + expansion)
        - Context-aware follow-up questions
        - Ask questions about uploaded PDFs with improved accuracy
        - Get AI-powered summaries with better context
        - Works even with difficult queries
        - *Requires: Vector store with PDFs*
        """)
        
        st.header("💡 Example Requests")
        st.markdown("""
        **Service Requests:**
        - "I have a headache"
        - "Need somewhere to stay tonight"
        - "Feeling really drained"
        - "Can you get me a ride?"
        
        **Enhanced RAG Queries:**
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
        
        st.header("⚡ Advanced RAG Features")
        st.markdown("""
        - **Context-aware conversations**: Follow-up questions automatically use Knowledge Base
        - **Multi-strategy search**: Vector + keyword + query expansion
        - **Enhanced chunking**: Better overlap and sizing for RAG
        - **Intelligent prompting**: Optimized for factual extraction
        - **Lenient matching**: Finds relevant content even with low similarity
        - **Query expansion**: Automatically tries related terms
        - **Fallback search**: Keyword matching when vector search fails
        - **Ultra-fast** vector store creation
        - **Batch processing** for embeddings
        - **GPU acceleration** when available
        - **Smart memory management**
        - **Process up to 10 PDF files**
        - **Files up to 10MB supported**
        """)
        
        # Show knowledge base status
        if VECTOR_STORE_AVAILABLE and "vector_manager" in st.session_state:
            vector_manager = st.session_state.vector_manager
            index, metadata = vector_manager.load_vector_store()
            if index is not None and metadata is not None:
                sources = set(item["source"] for item in metadata)
                st.success(f"📚 Enhanced RAG Ready: {len(sources)} documents, {len(metadata)} chunks")
                
                # Show context status in sidebar too
                if st.session_state.get('last_successful_category') == "Search Knowledge Base":
                    st.info("🔗 Context Active: Next questions will use Knowledge Base")
            else:
                st.info("📚 Knowledge Base: Not created yet")
        
        st.markdown("---")
        st.header("🔧 Installation & Setup")
        st.markdown("""
        **For CPU-only (Recommended):**
        ```bash
        pip install faiss-cpu PyPDF2
        ```
        
        **For GPU acceleration:**
        ```bash
        pip install faiss-gpu PyPDF2
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
        
        **Performance Tips:**
        - Use PDF files under 10MB for best performance
        - Process up to 10 files at once  
        - Enhanced RAG uses multi-strategy search for better results
        - GPU acceleration used automatically when available
        - Optimized batch processing for fast embedding generation
        - Better chunking strategy improves answer quality
        """)
        
        # Show current performance status
        if VECTOR_STORE_AVAILABLE:
            gpu_status = "🚀 GPU Accelerated" if GPU_AVAILABLE else "💻 CPU Optimized"
            st.markdown(f"**Status:** {gpu_status}")
            
        # Show AI summary status
        if "openai_client" in st.session_state:
            st.markdown("**Enhanced RAG:** ⚡ Multi-strategy search available")
        else:
            st.markdown("**Enhanced RAG:** ❌ Configuration issue")
        
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