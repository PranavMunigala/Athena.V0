"""
Athena v0 - Document Ingestion Module

This module processes PDF files from the corpus/ directory, chunks them,
and populates a persistent Chroma vector database with embeddings.

Required packages:
    pip install pymupdf chromadb openai streamlit python-dotenv
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import fitz  # pymupdf
import chromadb
from dotenv import load_dotenv
from v1.embeddings import OpenAIEmbedder


# Load environment variables (OPENAI_API_KEY) from .env if present
load_dotenv()


# Configuration
CORPUS_DIR = "./corpus"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "athena_notes"
CHUNK_TOKEN_SIZE = 400
CHUNK_OVERLAP_TOKENS = 50
# Number of chunks to embed/insert per OpenAI request (well within API limits).
EMBED_BATCH_SIZE = 100
# Rough approximation: 1 token is about 4 characters
CHAR_PER_TOKEN = 4
CHUNK_CHAR_SIZE = CHUNK_TOKEN_SIZE * CHAR_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHAR_PER_TOKEN


def count_tokens_approx(text: str) -> int:
    """Approximate token count using character-based heuristic."""
    return len(text) // CHAR_PER_TOKEN


def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extract text from PDF file page by page.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples (text, page_number) where page_number starts at 1
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append((text, page_num))
        doc.close()
        return pages
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return []


def chunk_text_with_overlap(text: str, chunk_size: int = CHUNK_CHAR_SIZE, 
                           overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """
    Split text into chunks with overlap, respecting word boundaries.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap size in characters
        
    Returns:
        List of text chunks
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # If not at end of text, try to break at sentence or word boundary
        if end < len(text):
            # Look for period + space within last 100 chars
            last_period = text.rfind(". ", max(start, end - 100), end)
            if last_period != -1:
                end = last_period + 2
            else:
                # Look for space within last 50 chars
                last_space = text.rfind(" ", max(start, end - 50), end)
                if last_space != -1:
                    end = last_space + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        
        # Move start position with overlap while guaranteeing forward progress.
        # Boundary-aware splits can move ``end`` inside the overlap window.
        next_start = end - overlap
        start = next_start if next_start > start else end
    
    return chunks


def generate_unique_id(filename: str, page: int, chunk_idx: int) -> str:
    """Generate a unique ID for a chunk."""
    clean_filename = Path(filename).stem
    return f"{clean_filename}_p{page}_c{chunk_idx}"


def flush_batch(model, collection, ids: List[str], texts: List[str],
                metadatas: List[Dict]) -> int:
    """
    Embed a batch of chunks in a single API call and add them to the collection.

    Returns the number of chunks written and leaves the input lists for the
    caller to clear.
    """
    if not texts:
        return 0
    embeddings = model.encode(texts, convert_to_numpy=True)
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    return len(texts)


def ingest_pdfs():
    """
    Main ingestion function: load PDFs, chunk them, embed, and store in Chroma.
    """
    # Initialize embedder and Chroma client
    print("Initializing OpenAI embedder...")
    model = OpenAIEmbedder()
    
    print(f"Initializing Chroma persistent client at {CHROMA_DB_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Find all PDF files in corpus directory
    corpus_path = Path(CORPUS_DIR)
    if not corpus_path.exists():
        print(f"Corpus directory '{CORPUS_DIR}' does not exist. Creating it...")
        corpus_path.mkdir(parents=True, exist_ok=True)
        print("Please add your PDF files to the corpus/ directory and run again.")
        return
    
    pdf_files = sorted(corpus_path.glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {CORPUS_DIR}")
        return
    
    total_chunks_added = 0

    # Pending batch buffers, flushed once they reach EMBED_BATCH_SIZE.
    batch_ids: List[str] = []
    batch_texts: List[str] = []
    batch_metadatas: List[Dict] = []

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        pages = extract_text_from_pdf(str(pdf_file))

        if not pages:
            print(f"  No text extracted from {pdf_file.name}")
            continue

        for text, page_num in pages:
            chunks = chunk_text_with_overlap(text)

            for chunk_idx, chunk in enumerate(chunks):
                batch_ids.append(generate_unique_id(pdf_file.name, page_num, chunk_idx))
                batch_texts.append(chunk)
                batch_metadatas.append({"source": pdf_file.name, "page": page_num})

                # Flush once the batch is full.
                if len(batch_texts) >= EMBED_BATCH_SIZE:
                    total_chunks_added += flush_batch(
                        model, collection, batch_ids, batch_texts, batch_metadatas
                    )
                    batch_ids.clear()
                    batch_texts.clear()
                    batch_metadatas.clear()

            print(f"  Page {page_num}: {len(chunks)} chunks prepared")

    # Flush any remaining chunks in the final partial batch.
    total_chunks_added += flush_batch(
        model, collection, batch_ids, batch_texts, batch_metadatas
    )

    print(f"\nIngestion complete! Total chunks added: {total_chunks_added}")


if __name__ == "__main__":
    ingest_pdfs()
