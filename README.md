# RAG Q&A System

A Retrieval-Augmented Generation (RAG) question-answering system built with Streamlit.

## Abstract

Retrieval-Augmented Generation (RAG) represents a significant advancement in making Large Language Models (LLMs) more accurate and contextually relevant by grounding their responses in external knowledge sources. This project implements a complete RAG pipeline that enables users to upload documents and ask natural language questions, receiving AI-generated answers with source citations.

The system architecture consists of three core components working in harmony. First, the **embedding layer** utilizes Sentence Transformers (all-MiniLM-L6-v2) to convert text chunks into dense 384-dimensional vector representations, capturing semantic meaning rather than simple keyword matching. Second, the **retrieval layer** employs FAISS (Facebook AI Similarity Search) as an efficient vector store, enabling fast similarity search to find the most relevant document passages for any given query. Third, the **generation layer** leverages HuggingFace's Inference API to access state-of-the-art language models like Qwen-2.5 and Mistral-7B, which synthesize retrieved context into coherent, accurate answers.

The implementation addresses several practical challenges in RAG system design. Document processing handles both PDF and plain text formats, with intelligent text chunking using recursive character splitting to maintain semantic coherence across chunk boundaries. The overlapping chunk strategy ensures that context is not lost at split points. The user interface, built with Streamlit, provides an intuitive chat-based interaction pattern with transparent source attribution, allowing users to verify the provenance of generated answers.

This project demonstrates how modern NLP techniques can be combined to create practical AI applications without requiring expensive computational resources. By utilizing free-tier APIs and efficient local processing for embeddings and retrieval, the system achieves a balance between capability and accessibility, making advanced RAG functionality available for educational and prototyping purposes.

**Keywords:** RAG, Retrieval-Augmented Generation, Vector Search, LLM, Sentence Transformers, FAISS, Streamlit

---

## Features

- 📄 **Document Upload**: Support for PDF and TXT files
- 🔍 **Semantic Search**: Find relevant passages using vector similarity
- 🤖 **AI Answers**: Generate contextual answers using LLMs
- 📎 **Source Citations**: Transparent attribution to source documents
- 💬 **Chat Interface**: Intuitive conversation-style interaction

## Architecture

```
┌─────────────────┐
│  User Upload    │
│  (PDF/TXT)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Chunking  │
│  (RecursiveChar)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │
│ (MiniLM-L6-v2)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │
│    (FAISS)      │
└────────┬────────┘
         │
    User Query
         │
         ▼
┌─────────────────┐
│   Retrieval     │
│  (Top-K Search) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation    │
│  (HF Inference) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Answer + Sources│
└─────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Web UI |
| Embeddings | sentence-transformers | Text → Vectors |
| Vector Store | FAISS (CPU) | Similarity Search |
| LLM | HuggingFace Inference API | Answer Generation |
| PDF Parsing | pypdf | Document Processing |
| Chunking | langchain-text-splitters | Text Segmentation |

## Local Development

### Prerequisites

- Python 3.11.14
- uv package manager
- HuggingFace account (free)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qa-llm.git
cd qa-llm

# Create virtual environment with uv
uv venv --python 3.11.14

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt

# Create secrets file for local development
mkdir -p .streamlit
echo 'HF_TOKEN = "your_token_here"' > .streamlit/secrets.toml

# Run the app
streamlit run app.py
```

### Environment Variables

Create `.streamlit/secrets.toml` for local development:

```toml
HF_TOKEN = "your_huggingface_token"
```

## Deployment to Streamlit Cloud

1. Push this repository to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and `app.py`
5. Add secrets in the app settings:
   - Go to "Advanced settings" → "Secrets"
   - Add: `HF_TOKEN = "your_huggingface_token"`
6. Deploy!

## Usage

1. Enter your HuggingFace token in the sidebar
2. Upload PDF or TXT documents
3. Click "Process Documents"
4. Ask questions in the chat input
5. View answers with source citations

## Project Structure

```
qa-llm/
├── .streamlit/
│   └── config.toml          # Streamlit theme/config
├── rag/
│   ├── __init__.py          # Package exports
│   ├── embedder.py          # Text embedding
│   ├── vectorstore.py       # FAISS operations
│   └── generator.py         # LLM inference
├── app.py                   # Main Streamlit app
├── requirements.txt         # Dependencies
├── .gitignore
├── .env.example
└── README.md
```

## References

- [Demo06 - RAG System](https://github.com/yenlung/AI-Demo) by Professor Yen-Lung Tsai
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Inference API](https://huggingface.co/inference-api)

## License

MIT License - Feel free to use for educational purposes.
