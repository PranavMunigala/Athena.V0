# Athena v0 — Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your PDFs
Create a `corpus/` folder and add your lecture PDFs:
```bash
mkdir corpus
# Copy your PDF files here
```

### 3. Ingest PDFs (One-Time Setup)
```bash
python ingest.py
```

Expected output:
```
Loading embedding model...
Initializing Chroma persistent client at ./chroma_db...

Processing: lecture-1.pdf
  Page 1: 5 chunks added
  Page 2: 4 chunks added
...
✓ Ingestion complete! Total chunks added: 127
```

### 4. Start the App
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# Run the Streamlit app
streamlit run app.py
```

The app opens at **http://localhost:8501/**

### 5. Ask Questions!
- Type a question in the text input
- See the answer in the main area
- Check the sidebar for retrieved context and similarity scores

---

## Using Local Models (Ollama)

Instead of OpenAI, you can use local LLMs like Ollama:

```bash
# 1. Install and start Ollama (https://ollama.ai)
ollama serve

# 2. In another terminal, run Athena
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="llama3"  # or mistral, neural-chat, etc.
streamlit run app.py
```

---

## Key Features

✅ **Smart PDF Chunking** — 400-token chunks with 50-token overlap  
✅ **Local Embeddings** — Fast, private, no API calls for embeddings  
✅ **Persistent Storage** — Chroma DB saves your embeddings locally  
✅ **Confidence Filtering** — Refuses to answer low-confidence queries  
✅ **Full Transparency** — See retrieved context in the sidebar  
✅ **Citation Support** — LLM cites sources inline  

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No PDFs found" | Create `corpus/` folder and add PDFs |
| "API key not found" | Set `OPENAI_API_KEY` environment variable |
| "No documents in DB" | Run `python ingest.py` |
| Slow response | Check internet/API rate limits, or use Ollama locally |

---

## What Each File Does

| File | Purpose |
|------|---------|
| `ingest.py` | Reads PDFs, chunks text, generates embeddings, stores in Chroma |
| `rag.py` | Retrieves chunks, generates LLM responses with citations |
| `app.py` | Streamlit web UI with sidebar analytics |

---

## Next Steps

- Read **README.md** for detailed documentation
- Adjust chunking parameters in `ingest.py` if needed
- Customize the LLM prompt in `rag.py`
- Deploy to the cloud (Streamlit Cloud, AWS, etc.)

Happy studying! 🏛️✨
