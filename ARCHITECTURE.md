# Athena v0 — Architecture & Design

## System Overview

```
PDF Files (corpus/)
    ↓
[ingest.py]
    ├─ Extract text (pymupdf)
    ├─ Chunk text (~400 tokens)
    └─ Generate embeddings (sentence-transformers)
    ↓
Chroma DB (./chroma_db/)
    ├─ Persistent vector store
    ├─ Cosine similarity index
    └─ Metadata (source, page)
    ↓
[rag.py] → Retrieve + Generate
    ├─ Embed query
    ├─ Retrieve top-k chunks
    ├─ Confidence check (≥0.5 similarity)
    └─ LLM generation with citations
    ↓
[app.py] (Streamlit UI)
    ├─ User input
    ├─ Display answer
    └─ Sidebar: context analytics
```

## Component Details

### 1. ingest.py — Document Processing Pipeline

**Responsibilities:**
- Walk `corpus/` directory for PDFs
- Extract text page-by-page using `pymupdf`
- Chunk text intelligently with overlap
- Generate embeddings using `sentence-transformers`
- Store in Chroma DB with metadata

**Key Functions:**
- `extract_text_from_pdf(pdf_path)` → Returns list of (text, page_number) tuples
- `chunk_text_with_overlap(text)` → Splits text respecting word boundaries
- `ingest_pdfs()` → Main pipeline orchestrator

**Chunking Strategy:**
- Target: ~400 tokens per chunk (~1600 characters)
- Overlap: ~50 tokens (~200 characters)
- Respects word/sentence boundaries when possible
- Prevents information loss at chunk boundaries

**Metadata Structure:**
```python
{
    "source": "lecture-1.pdf",
    "page": 3
}
```

---

### 2. rag.py — Retrieval & Generation Backend

**Responsibilities:**
- Retrieve semantically similar chunks
- Enforce confidence thresholds
- Generate LLM responses with strict prompting
- Handle L2-to-cosine similarity conversion

**Key Class:**
`RAGBackend` — Encapsulates retrieval and generation logic

**Key Methods:**
- `retrieve(query, k=5)` → Returns list of (text, metadata, similarity_score)
- `generate_answer(query, chunks, threshold=0.5)` → Returns LLM response or refusal

**Similarity Scoring:**
- Chroma uses **L2 (Euclidean) distance** by default
- For normalized vectors: `similarity = 1 - (distance² / 2)`
- Clamped to [0, 1] range
- Score of 1.0 = identical vectors
- Score of 0.0 = orthogonal vectors

**Confidence Threshold Logic:**
```python
max_similarity = max(chunk["similarity_score"] for chunk in retrieved_chunks)
if max_similarity < 0.5:
    return "I don't see this in your notes."
else:
    # Generate answer using LLM
```

**LLM System Prompt:**
```
"You are Athena v0, a strict study assistant. Answer the user's question using ONLY 
the provided context blocks. Do not invent facts. Every claim you make must be 
accompanied by an inline citation format referencing its source file and page, 
exactly like this: [filename.pdf p.X]. If the context does not contain enough 
information to answer the question, or if you are unsure, reply exactly with: 
'I don't see this in your notes.'"
```

---

### 3. app.py — Streamlit Web UI

**Responsibilities:**
- Provide user interface for querying
- Display answers with formatting
- Show retrieved context in sidebar
- Cache expensive resources

**Key Features:**

#### Caching with `@st.cache_resource`
```python
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)
```
**Why:** Streamlit re-runs scripts on every interaction. Without caching, the model and DB would reload each time, causing huge latency.

#### Main UI Elements
1. **Title & Description**
   - "Athena v0 — RAG Study Assistant"
   - Clean, professional layout

2. **Query Input**
   - Text input: "Ask a question about your notes:"
   - Single line for simplicity

3. **Sidebar Configuration**
   - LLM model selector
   - Base URL for local models
   - API key input (password field)

4. **Sidebar Context Analytics**
   - Title: "Retrieved Context Analytics"
   - Top 3 chunks in expandable sections
   - For each chunk:
     - Similarity score (e.g., "75.4%")
     - Full text
     - Source metadata (filename, page)

5. **Main Answer Display**
   - Blue box for successful answers
   - Yellow box for refusals (< 0.5 confidence)
   - Info message explaining why refusal occurred

#### UI/UX Design Choices
- **Wide layout** → More space for context chunks
- **Expanders** → Hide/show chunks for cleanliness
- **Inline badges** → Quick visual reference for scores
- **Color coding** → Blue (answer), yellow (refusal)
- **Spinner messages** → Feedback during processing

---

## Data Flow Example

### User asks: "What is photosynthesis?"

```
1. User Input
   └─ Query: "What is photosynthesis?"

2. Embedding [app.py → rag.py]
   └─ Query embedding (384-dim vector)

3. Retrieval [rag.py → Chroma DB]
   └─ Cosine similarity search
   └─ Returns top 5 chunks with L2 distances

4. Distance Conversion [rag.py]
   └─ L2 distances → Cosine similarities
   └─ chunk_1: 0.87, chunk_2: 0.76, ..., chunk_5: 0.43

5. Confidence Check [rag.py]
   └─ max_similarity = 0.87
   └─ 0.87 ≥ 0.5? YES → Proceed to LLM

6. Context Assembly [rag.py]
   └─ Build context string from top chunks
   └─ Include citations: [biology-101.pdf p.5], etc.

7. LLM Generation [rag.py → OpenAI/Local]
   └─ System prompt: strict, citation-enforcing
   └─ User message: context + question
   └─ Response: "Photosynthesis is... [biology-101.pdf p.5]..."

8. UI Rendering [app.py]
   ├─ Main: Display answer in blue box
   └─ Sidebar: Show top 3 chunks with scores
      ├─ Chunk 1: 87.2% | biology-101.pdf, Page 5
      ├─ Chunk 2: 76.1% | biology-101.pdf, Page 6
      └─ Chunk 3: 68.5% | biology-notes.pdf, Page 2
```

---

## Key Design Decisions

### 1. **Local Embeddings (sentence-transformers)**
✅ **Pro:** No API calls, privacy, speed, cost-free
✅ **Pro:** Exact model consistency between docs and queries
❌ **Con:** Lower quality than larger models (e.g., OpenAI)

**Choice:** Justified for educational use where privacy/cost outweigh quality

### 2. **Persistent Chroma DB**
✅ **Pro:** Fast subsequent queries, offline capability
✅ **Pro:** No rebuilding needed
❌ **Con:** Single-machine only (not distributed)

**Choice:** Perfect for personal study assistant

### 3. **Similarity Threshold (0.5)**
✅ **Pro:** Avoids hallucinations
✅ **Pro:** Transparent refusals
❌ **Con:** May miss valid answers with lower scores

**Choice:** Conservative default; adjustable per use case

### 4. **Strict LLM Prompting**
✅ **Pro:** Forces citations and responsibility
✅ **Pro:** Clear refusal messages
❌ **Con:** Slightly more restrictive than open-ended responses

**Choice:** Essential for study assistants (must not invent)

### 5. **Smart Chunking with Overlap**
✅ **Pro:** Preserves context at boundaries
✅ **Pro:** Increases retrieval recall
❌ **Con:** Increases storage size (~50% more)

**Choice:** Trade-off favors quality over storage

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Embedding model load | ~2-3s | One-time, cached |
| Single PDF ingest | ~1-5s | Depends on size |
| Query embedding | ~100ms | Fast, local |
| Similarity search | ~50-100ms | Chroma index lookup |
| LLM generation | ~2-5s | OpenAI API latency |
| **Total query latency** | ~3-6s | Dominated by LLM |

---

## Extensibility & Customization

### Add a New Embedding Model
```python
# ingest.py & rag.py
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# Re-run ingest.py to rebuild database
```

### Increase Chunk Size
```python
# ingest.py
CHUNK_TOKEN_SIZE = 600  # Instead of 400
CHUNK_OVERLAP_TOKENS = 100  # Instead of 50
# Re-run ingest.py
```

### Adjust Confidence Threshold
```python
# rag.py or app.py
rag.generate_answer(query, chunks, similarity_threshold=0.3)
```

### Switch LLM Providers
```python
# app.py environment variables
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="mistral"
```

### Add Filtering (e.g., by date)
```python
# rag.py - modify retrieve() to add where clause
# Chroma supports metadata filtering
results = self.collection.query(
    query_embeddings=[...],
    where={"page": {"$gt": 5}}  # Only pages > 5
)
```

---

## Future Enhancements

1. **Reranking** — Use a second model to rerank top-k chunks before LLM
2. **Conversation Memory** — Track conversation history for multi-turn queries
3. **Hybrid Search** — Combine semantic + BM25 keyword search
4. **Citation Verification** — Extract and verify citations programmatically
5. **Multi-PDF Collections** — Manage multiple document sets
6. **Web Interface** → FastAPI + React instead of Streamlit
7. **Distributed Storage** → Move from Chroma to Pinecone/Weaviate
8. **Fine-tuning** — Custom embedding model for domain-specific knowledge

---

## Security & Privacy

- **Embeddings:** Generated locally, never sent to third parties
- **PDF Storage:** Local `corpus/` folder, not uploaded
- **Vector DB:** Local `chroma_db/` folder, not cloud-synced
- **API Keys:** Only sent to configured LLM provider (OpenAI or local)
- **User Queries:** Sent to LLM provider (respect their privacy policy)

---

## Testing Strategy

```python
# Suggested test cases (manual testing for now)
1. Empty corpus/ → Graceful error
2. Invalid PDF → Skip, continue
3. Very long document → Proper chunking
4. Low-similarity query → Proper refusal
5. Multi-page query → Correct citations
6. API timeout → Graceful error message
```

---

## Deployment Options

### Option 1: Local Development
```bash
streamlit run app.py
```

### Option 2: Streamlit Cloud
```bash
streamlit cloud deploy app.py
```
*(Requires git repo + Streamlit account)*

### Option 3: Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

### Option 4: Cloud Deployment (AWS, GCP, Azure)
```bash
# Package as Docker container
# Deploy to CloudRun, Lambda, etc.
```

---

## Troubleshooting Guide

See **QUICKSTART.md** and **README.md** for user-facing troubleshooting.

For developers:
1. Check logs: `streamlit run app.py --logger.level=debug`
2. Verify Chroma DB: `ls -la chroma_db/`
3. Test embeddings: Run `ingest.py` with verbose output
4. Test LLM: Use `curl` to test OpenAI endpoint directly

---

**Architecture designed for simplicity, transparency, and educational use.** 🏛️
