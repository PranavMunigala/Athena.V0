# Athena v0 — RAG Study Assistant

A clean, modular Retrieval-Augmented Generation (RAG) application that allows you to query your own PDF lecture notes using embeddings, vector search, and an LLM backend.

## Overview

Athena v0 consists of three main modules:

- **`ingest.py`** — Processes PDFs from the `corpus/` folder, chunks text intelligently (~400 tokens with 50-token overlap), generates embeddings using `sentence-transformers`, and populates a persistent Chroma vector database.
- **`rag.py`** — Backend logic for retrieving relevant context chunks and generating LLM responses with a strict system prompt and confidence threshold.
- **`app.py`** — Interactive Streamlit web UI with sidebar analytics showing retrieved context and similarity scores.

## Architecture & Design Choices

### 1. **Embedding Model**
- Uses **`sentence-transformers/all-MiniLM-L6-v2`** (384-dimensional vectors) locally.
- The **exact same model instance** is used for both document and query embeddings.
- No external API calls for embeddings; everything runs locally.

### 2. **Vector Store**
- **Chroma** (persistent client) stores embeddings in `./chroma_db/`.
- Uses **cosine similarity** for retrieval.
- Automatic L2-to-cosine conversion applied by the backend.

### 3. **PDF Parsing & Chunking**
- **`pymupdf` (fitz)** extracts text page-by-page.
- **Smart chunking**: ~400 tokens (~1600 characters) per chunk with ~50-token (~200 character) overlap.
- Respects word and sentence boundaries when possible.
- **Metadata tracking**: Each chunk stores source filename and page number.

### 4. **LLM Backend**
- Uses the **OpenAI API** (or any OpenAI-compatible endpoint like Ollama).
- Configurable model (default: `gpt-4o-mini`; also supports local models like `llama3`).
- **Strict system prompt** enforces citations and refusals when context is insufficient.
- **Confidence threshold**: Max similarity score must ≥ 0.5 (50%) to attempt answering.

## Installation

```bash
# Clone or navigate to the project directory
cd Athena.V0

# Install dependencies
pip install -r requirements.txt
```

**System Requirements:**
- Python 3.8+
- ~500MB disk space for the embedding model and vector database
- OpenAI API key (or access to a local Ollama/Omnix instance)

## Quick Start

### Step 1: Prepare Your PDFs
Create a `corpus/` folder in the project root and place your PDF lecture notes inside:

```
Athena.V0/
├── ingest.py
├── rag.py
├── app.py
├── requirements.txt
└── corpus/
    ├── lecture-1.pdf
    ├── lecture-2.pdf
    └── notes.pdf
```

### Step 2: Ingest PDFs

```bash
python ingest.py
```

Output:
```
Loading embedding model...
Initializing Chroma persistent client at ./chroma_db...

Processing: lecture-1.pdf
  Page 1: 5 chunks added
  Page 2: 4 chunks added
  Page 3: 6 chunks added
...
✓ Ingestion complete! Total chunks added: 127
```

The script will:
- Load all `.pdf` files from `corpus/`
- Extract and chunk the text intelligently
- Generate embeddings for each chunk
- Store everything in `./chroma_db/` (persistent, survives restarts)

### Step 3: Run the Streamlit App

```bash
# Set your OpenAI API key (or leave unset if in environment)
export OPENAI_API_KEY="sk-..."

# Or for local models (Ollama):
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="llama3"

# Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501/` by default.

## Usage

1. **Enter a question** in the main text input.
2. **View retrieved context** in the sidebar under "Retrieved Context Analytics" — the top 3 chunks with their similarity scores.
3. **Read the generated answer** in the main area.
   - If confidence is low (max similarity < 0.5), the system returns: *"I don't see this in your notes."*
   - Otherwise, the LLM generates a response with inline citations.

### Example Queries

- "What is photosynthesis?"
- "Explain the theory of relativity."
- "Summarize the key concepts from chapter 3."
- "How do mitochondria function?"

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required for OpenAI models) | ` ` |
| `LLM_BASE_URL` | Base URL for local LLM (e.g., Ollama) | ` ` |
| `LLM_MODEL` | LLM model name (e.g., `gpt-4o-mini`, `llama3`) | `gpt-4o-mini` |

### Code Configuration

Edit the constants at the top of each file:

**`ingest.py`:**
```python
CORPUS_DIR = "./corpus"              # PDF folder location
CHROMA_DB_DIR = "./chroma_db"       # Vector database location
CHUNK_TOKEN_SIZE = 400              # Chunk size in tokens
CHUNK_OVERLAP_TOKENS = 50           # Overlap in tokens
```

**`rag.py`:**
```python
SIMILARITY_THRESHOLD = 0.5           # Min confidence to answer (0-1)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
```

## Key Features

### 1. **Smart Text Chunking**
- ~400 tokens per chunk with 50-token overlap prevents information loss at chunk boundaries.
- Respects sentence and word boundaries for readability.

### 2. **Confidence-Based Refusal**
- If the maximum similarity score is < 0.5, returns: *"I don't see this in your notes."*
- User can inspect the retrieved context in the sidebar to understand why.

### 3. **Transparent Context Retrieval**
- Sidebar displays **top 3 retrieved chunks** with exact similarity scores and source metadata.
- User can see what the system "knew" when generating the answer.

### 4. **Strict LLM Prompting**
- System prompt forces:
  - Only answering from provided context
  - Inline citations in `[filename.pdf p.X]` format
  - Refusal when uncertain

### 5. **Cached Resources**
- Streamlit's `@st.cache_resource` ensures the embedding model and Chroma client are loaded once per session, not on every interaction.
- Dramatically improves responsiveness.

## File Structure

```
Athena.V0/
├── ingest.py          # PDF ingestion and embedding pipeline
├── rag.py             # RAG backend (retrieval + generation)
├── app.py             # Streamlit UI
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── corpus/            # Input folder (your PDFs go here)
├── chroma_db/         # Vector database (auto-created, persistent)
└── .git/              # Version control
```

## Troubleshooting

### "No PDF files found in corpus/"
- Create a `corpus/` folder in the project root.
- Add your PDF files to it.
- Run `python ingest.py` again.

### "Error: OPENAI_API_KEY not set"
- Set your API key as an environment variable:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
  Or enter it in the Streamlit sidebar.

### "No documents found in the database"
- Run `python ingest.py` to populate the database.

### LLM is slow / timing out
- For OpenAI models, check your internet connection and API rate limits.
- For local models, ensure Ollama is running:
  ```bash
  ollama serve
  ```
  Then set `LLM_BASE_URL="http://localhost:11434/v1"` and `LLM_MODEL="llama3"`.

### Chroma database errors
- The `./chroma_db/` folder is auto-created and persistent.
- To reset: delete the folder and run `python ingest.py` again.

## Advanced Customization

### Change the Embedding Model
Edit `ingest.py` and `rag.py`:
```python
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Larger, slower, more accurate
```
Then re-run `python ingest.py` to rebuild the database.

### Adjust Chunking Parameters
Edit `ingest.py`:
```python
CHUNK_TOKEN_SIZE = 600              # Larger chunks
CHUNK_OVERLAP_TOKENS = 100          # More overlap
```
Then re-run `python ingest.py`.

### Change the Similarity Threshold
Edit `rag.py` or pass it when calling `generate_answer()`:
```python
answer = rag.generate_answer(query, retrieved_chunks, similarity_threshold=0.3)
```

### Use a Different LLM
Edit the `LLM_MODEL` environment variable:
```bash
# Anthropic Claude (requires API key)
export LLM_MODEL="claude-3-opus-20240229"

# Local Ollama
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="mistral"
```

## Performance Notes

- **First run of `ingest.py`**: ~2-5 minutes (depends on PDF size and embedding model download).
- **Subsequent ingestions**: ~seconds to minutes (data is loaded from `./chroma_db/`).
- **Query latency**: ~2-5 seconds (embedding + retrieval + LLM generation).
  - Retrieval is instant (~<100ms).
  - LLM generation is the bottleneck (depends on model and network).

## Error Handling

All three modules include robust error handling:
- Missing PDFs or empty pages are skipped with warnings.
- Chroma connection failures are caught and reported.
- LLM API errors are gracefully surfaced to the user.
- Invalid metadata is handled gracefully.

## Example Workflow

```bash
# 1. Set up dependencies
pip install -r requirements.txt

# 2. Add your PDFs to corpus/

# 3. Ingest them
python ingest.py

# 4. Start the app
export OPENAI_API_KEY="sk-..."
streamlit run app.py

# 5. Open browser to http://localhost:8501/
# 6. Ask questions!
```

## License & Attribution

This is a template for a RAG study assistant. Use it freely for personal or educational purposes.

---

**Happy studying with Athena v0!** 🏛️✨
