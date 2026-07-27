# 📚 Chat With Your Docs

A local RAG (Retrieval-Augmented Generation) web app that lets you upload PDF documents and ask questions about them in plain English. Every answer is grounded in your documents with exact source citations.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **Upload & Chat** — Upload multiple PDFs and start asking questions in seconds
- **Source Citations** — Every answer shows the exact passages it was generated from, with similarity scores
- **Groq Cloud API Powered** — Lightning-fast LPU inference via Groq's free cloud API (`llama-3.3-70b-versatile`)
- **Persistent Storage** — Documents survive app restarts via ChromaDB
- **Anti-Hallucination** — System prompt forces the model to answer only from your documents
- **Error Handling** — Clear messages for scanned PDFs, missing API keys, and connection issues

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Groq API Key** (free, no credit card): [Get one here](https://console.groq.com)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd "Chat With Your Docs"

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### First Time Usage

1. **Enter your Groq API key** in the sidebar
2. **Upload PDF files** using the sidebar uploader
3. **Click "Process Documents"** and wait for chunking/embedding to complete
4. **Ask questions** in the chat input — answers will include source citations

## 🔧 Configuration

### Groq Cloud LPU Engine
The app is natively configured for Groq's LPU inference using the state-of-the-art **`llama-3.3-70b-versatile`** model with automatic token optimization (1,536 output tokens and 10 chronological context chunks) to guarantee zero HTTP 429 rate limit errors on Groq's free tier!

### Configurable Constants

Edit the top of `rag_pipeline.py` to adjust:

| Constant | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Target characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K` | 10 | Number of chunks retrieved per query |

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and select `app.py`
4. Add your Groq API key in **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Deploy!

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Text chunking edge cases (empty input, short text, overlap, word boundaries)
- PDF extraction error handling
- Prompt assembly and structure
- Retrieval logic with mocked dependencies

## 📁 Project Structure

```
Chat With Your Docs/
├── app.py                  # Streamlit UI + orchestration
├── rag_pipeline.py         # Core RAG logic (chunking, embedding, retrieval, generation)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
├── .streamlit/
│   └── config.toml         # Streamlit theme + config
├── chroma_db/              # Persistent vector store (gitignored)
└── tests/
    └── test_rag_pipeline.py # Unit tests
```

## 🔍 How It Works

```
Upload PDFs → Extract text (pypdf) → Chunk into ~1200-char segments
    → Embed with sentence-transformers → Store in ChromaDB

Ask a question → Embed the question → Find top-20 similar chunks (chronologically sorted)
    → Build grounded prompt → Generate answer via Groq Cloud API
    → Display answer + source citations
```

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "No extractable text found" | The PDF is likely scanned/image-only. Use OCR tools first. |
| "Invalid Groq API key" | Check your key at [console.groq.com](https://console.groq.com) |
| Slow first query | The embedding model downloads on first use (~90MB). Subsequent runs are instant. |

## 📄 License

MIT
