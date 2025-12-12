"""
RAG Q&A Demo - A Retrieval-Augmented Generation system using Streamlit.

This app allows users to:
1. Upload PDF or TXT documents
2. Ask questions about the content
3. Get AI-generated answers with source citations
"""

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import Embedder, VectorStore, Generator


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide"
)


# =============================================================================
# Initialize Session State
# =============================================================================
@st.cache_resource
def load_embedder():
    """Load embedder model (cached to avoid reloading)."""
    return Embedder(model_name="all-MiniLM-L6-v2")


def init_session_state():
    """Initialize session state variables."""
    if "vector_store" not in st.session_state:
        embedder = load_embedder()
        st.session_state.vector_store = VectorStore(dimension=embedder.dimension)
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# =============================================================================
# Document Processing Functions
# =============================================================================
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from a PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_txt(txt_file) -> str:
    """Extract text content from a TXT file."""
    return txt_file.read().decode("utf-8")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def process_uploaded_files(uploaded_files) -> int:
    """Process uploaded files and add to vector store."""
    embedder = load_embedder()
    total_chunks = 0
    
    for uploaded_file in uploaded_files:
        # Extract text based on file type
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            text = extract_text_from_txt(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue
        
        # Chunk the text
        chunks = chunk_text(text)
        
        if not chunks:
            st.warning(f"No text extracted from: {uploaded_file.name}")
            continue
        
        # Create embeddings
        embeddings = embedder.embed_texts(chunks)
        
        # Create metadata for each chunk
        metadata = [{"source": uploaded_file.name} for _ in chunks]
        
        # Add to vector store
        st.session_state.vector_store.add_texts(chunks, embeddings, metadata)
        total_chunks += len(chunks)
    
    return total_chunks


# =============================================================================
# Main Application UI
# =============================================================================
def main():
    init_session_state()
    
    # Header
    st.title("📚 RAG Q&A System")
    st.markdown("""
    Upload documents and ask questions! This system uses:
    - **Sentence Transformers** for text embeddings (local)
    - **FAISS** for vector similarity search (local)  
    - **HuggingFace Inference API** for answer generation (free tier)
    """)
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # HuggingFace Token
        hf_token = st.text_input(
            "HuggingFace Token",
            type="password",
            help="Get your free token at huggingface.co/settings/tokens"
        )
        
        # Check for token in secrets (for Streamlit Cloud deployment)
        if not hf_token:
            hf_token = st.secrets.get("HF_TOKEN", "")
        
        # Model selection
        model_choice = st.selectbox(
            "LLM Model",
            options=list(Generator.AVAILABLE_MODELS.keys()),
            index=2,  # Default to qwen-2.5
            help="Select the model for answer generation"
        )
        
        st.divider()
        
        # File upload section
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Clear existing data
                    st.session_state.vector_store.clear()
                    st.session_state.chat_history = []
                    
                    # Process new files
                    num_chunks = process_uploaded_files(uploaded_files)
                    
                    if num_chunks > 0:
                        st.session_state.documents_loaded = True
                        st.success(f"✅ Processed {num_chunks} text chunks!")
                    else:
                        st.error("No text could be extracted from the files.")
        
        # Show current status
        st.divider()
        st.header("📊 Status")
        chunk_count = st.session_state.vector_store.count
        st.metric("Chunks in Database", chunk_count)
        
        # Clear button
        if chunk_count > 0:
            if st.button("🗑️ Clear Database"):
                st.session_state.vector_store.clear()
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.rerun()
    
    # Main content area
    if not st.session_state.documents_loaded:
        st.info("👈 Upload documents in the sidebar to get started!")
        
        # Show example usage
        with st.expander("📖 How to use this app"):
            st.markdown("""
            1. **Get a HuggingFace token** (free): Go to [huggingface.co](https://huggingface.co), 
               create an account, and generate a token in Settings → Access Tokens
            2. **Enter your token** in the sidebar
            3. **Upload documents** (PDF or TXT files)
            4. **Click "Process Documents"** to index them
            5. **Ask questions** about your documents!
            
            The system will find relevant passages and generate answers using AI.
            """)
    else:
        # Q&A Interface
        st.header("💬 Ask Questions")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("📎 Sources"):
                        for src in message["sources"]:
                            st.markdown(f"**{src['source']}** (relevance: {src['score']:.2f})")
                            st.markdown(f"> {src['text'][:300]}...")
        
        # Question input
        if question := st.chat_input("Ask a question about your documents..."):
            if not hf_token:
                st.error("Please enter your HuggingFace token in the sidebar!")
            else:
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })
                
                with st.chat_message("user"):
                    st.markdown(question)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Retrieve relevant chunks
                        embedder = load_embedder()
                        query_embedding = embedder.embed_query(question)
                        results = st.session_state.vector_store.search(
                            query_embedding, k=3
                        )
                        
                        if not results:
                            response = "I couldn't find any relevant information in the documents."
                            sources = []
                        else:
                            # Generate answer
                            generator = Generator(hf_token, model_key=model_choice)
                            response = generator.generate_with_chat(
                                question, results
                            )
                            
                            # Prepare sources for display
                            sources = [
                                {
                                    "source": meta.get("source", "Unknown"),
                                    "text": text,
                                    "score": 1 / (1 + dist)  # Convert distance to similarity
                                }
                                for text, dist, meta in results
                            ]
                        
                        st.markdown(response)
                        
                        if sources:
                            with st.expander("📎 Sources"):
                                for src in sources:
                                    st.markdown(f"**{src['source']}** (relevance: {src['score']:.2f})")
                                    st.markdown(f"> {src['text'][:300]}...")
                
                # Add assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources if sources else []
                })


if __name__ == "__main__":
    main()
