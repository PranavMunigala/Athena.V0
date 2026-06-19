# Athena v0 — RAG Study Assistant

A clean, modular Retrieval-Augmented Generation (RAG) app for querying your own PDF
lecture notes using OpenAI embeddings, Chroma vector search, and an LLM backend.

For internals and design rationale, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Overview

| File | Purpose |
|------|---------|
| `ingest.py` | Reads PDFs from `corpus/`, chunks text (~400 tokens, 50-token overlap), embeds in batches, and stores vectors in Chroma. |
| `embeddings.py` | `OpenAIEmbedder` — wraps the OpenAI embeddings API (`text-embedding-3-small`) behind a simple `encode()` interface. |
| `rag.py` | Retrieves relevant chunks and generates an LLM answer with a strict, citation-enforcing prompt and a confidence threshold. |
| `app.py` | Streamlit web UI with sidebar analytics (retrieved chunks + similarity scores). |

## Key Features

- **Cloud embeddings** — OpenAI `text-embedding-3-small` (1536-dim) for both documents and queries.
- **Batched ingestion** — chunks are embedded/inserted in batches of 100, minimizing API round-trips.
- **Smart chunking** — ~400 tokens per chunk with ~50-token overlap, respecting word/sentence boundaries.
- **Confidence-based refusal** — if max similarity < 15%, replies *"I don't see this in your notes."*
- **Transparent retrieval** — sidebar shows the top retrieved chunks with scores and source/page.
- **Strict prompting** — answers only from context, with inline `[filename.pdf p.X]` citations.

## Installation

```bash
cd Athena.V0
pip install -r requirements.txt
```

Requirements: Python 3.12+, an OpenAI API key.

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY="sk-..."
# Optional — point the chat LLM at a local endpoint (embeddings always use OpenAI):
# LLM_BASE_URL="http://localhost:11434/v1"
# LLM_MODEL="llama3"
```

> **Note:** `OPENAI_API_KEY` is always required — embeddings go to OpenAI even when the
> chat LLM points at a local endpoint via `LLM_BASE_URL`.

## Quick Start

### 1. Add your PDFs

```
Athena.V0/
├── ingest.py
├── rag.py
├── app.py
└── corpus/
    ├── lecture-1.pdf
    └── notes.pdf
```

### 2. Ingest (one-time)

```bash
python ingest.py
```

```
Initializing OpenAI embedder...
Initializing Chroma persistent client at ./chroma_db...

Processing: lecture-1.pdf
  Page 1: 5 chunks prepared
  ...
✓ Ingestion complete! Total chunks added: 127
```

### 3. Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501/**. Type a question, read the answer, and inspect the
retrieved context in the sidebar.

## Configuration

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI key — required (embeddings + default LLM) | — |
| `LLM_BASE_URL` | Base URL for a local/OpenAI-compatible chat LLM (e.g. Ollama) | — |
| `LLM_MODEL` | Chat LLM model name | `gpt-4o-mini` |

### Code constants

```python
# ingest.py
CHUNK_TOKEN_SIZE = 400        # chunk size in tokens
CHUNK_OVERLAP_TOKENS = 50     # overlap in tokens
EMBED_BATCH_SIZE = 100        # chunks per embedding request

# rag.py
SIMILARITY_THRESHOLD = 0.15   # min confidence to answer (0–1)

# embeddings.py
EMBEDDING_MODEL = "text-embedding-3-small"
```

Changing the embedding model or chunking requires deleting `./chroma_db/` and
re-running `python ingest.py` (vector dimensions must stay consistent).

## Cost

`text-embedding-3-small` is ~$0.02 per 1M tokens. Embedding a typical notes corpus
costs a few cents; each query embeds one short string. The chat LLM is billed separately.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No PDF files found in corpus/" | Create `corpus/` and add PDFs, then re-run `python ingest.py`. |
| "OPENAI_API_KEY is not set" | Add it to `.env` (required even with a local chat LLM). |
| "No documents found in the database" | Run `python ingest.py`. |
| Dimension/embedding mismatch | Delete `./chroma_db/` and re-ingest. |
| Slow / timing-out answers | Check network and OpenAI rate limits, or use a local LLM via `LLM_BASE_URL`. |

---

**Happy studying with Athena v0!** 🏛️✨
