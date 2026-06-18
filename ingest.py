"""
Athena v0 - Document Ingestion Module

This module processes PDF files from the corpus/ directory, chunks them,
and populates a persistent Chroma vector database with embeddings.

Required packages:
    pip install pymupdf sentence-transformers chromadb openai streamlit
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # pymupdf
import chromadb
from sentence_transformers import SentenceTransformer


# Configuration
CORPUS_DIR = "./corpus"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "athena_notes"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKEN_SIZE = 400
CHUNK_OVERLAP_TOKENS = 50
# Rough approximation: 1 token ≈ 4 characters
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
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def generate_unique_id(filename: str, page: int, chunk_idx: int) -> str:
    """Generate a unique ID for a chunk."""
    clean_filename = Path(filename).stem
    return f"{clean_filename}_p{page}_c{chunk_idx}"


def ingest_pdfs():
    """
    Main ingestion function: load PDFs, chunk them, embed, and store in Chroma.
    """
    # Initialize model and Chroma client
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    
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
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        pages = extract_text_from_pdf(str(pdf_file))
        
        if not pages:
            print(f"  No text extracted from {pdf_file.name}")
            continue
        
        for text, page_num in pages:
            chunks = chunk_text_with_overlap(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = generate_unique_id(pdf_file.name, page_num, chunk_idx)
                metadata = {
                    "source": pdf_file.name,
                    "page": page_num
                }
                
                # Embed the chunk
                embedding = model.encode(chunk, convert_to_numpy=True)
                
                # Add to collection
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding.tolist()],
                    documents=[chunk],
                    metadatas=[metadata]
                )
                
                total_chunks_added += 1
            
            print(f"  Page {page_num}: {len(chunks)} chunks added")
    
    print(f"\n✓ Ingestion complete! Total chunks added: {total_chunks_added}")


if __name__ == "__main__":
    ingest_pdfs()
