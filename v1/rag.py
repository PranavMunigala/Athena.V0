"""
Athena v0 - RAG Backend Module

This module handles retrieval from the vector store and answer generation
using the configured LLM with strict system prompts.

Required packages:
    pip install pymupdf chromadb openai streamlit python-dotenv
"""

import math
from typing import List, Dict, Optional
import chromadb
from openai import OpenAI, APIError
from v1.embeddings import OpenAIEmbedder


# Configuration
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "athena_notes"
SIMILARITY_THRESHOLD = 0.15


class RAGBackend:
    """Main RAG backend for retrieval and generation."""
    
    def __init__(
        self,
        model: OpenAIEmbedder,
        chroma_client: chromadb.PersistentClient,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize RAG backend.
        
        Args:
            model: OpenAIEmbedder instance for embeddings
            chroma_client: Chroma PersistentClient instance
            llm_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            llm_base_url: Base URL for LLM endpoint (for local models like Ollama)
            llm_model: Model name to use for LLM (default: gpt-4o-mini)
        """
        self.model = model
        self.collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self.llm_model = llm_model
        
        # Initialize OpenAI client
        if llm_base_url:
            self.llm_client = OpenAI(api_key=llm_api_key or "not-needed", base_url=llm_base_url)
        else:
            self.llm_client = OpenAI(api_key=llm_api_key)
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert Chroma's returned cosine distance to a cosine similarity.

        The collection is created with ``hnsw:space="cosine"`` (see ingest.py
        and __init__), so Chroma returns cosine distance = 1 - cosine_similarity.
        Therefore cosine_similarity = 1 - distance.
        """
        similarity = 1 - distance
        return max(0.0, min(1.0, similarity))
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k similar chunks from the vector store.
        
        Args:
            query: User query string
            k: Number of results to retrieve
            
        Returns:
            List of dicts with keys: 'text', 'metadata', 'similarity_score'
        """
        # Embed the query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Query the collection with distances
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_chunks = []
        
        if results and results["documents"] and len(results["documents"]) > 0:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if results["distances"] else [float('inf')] * len(documents)
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                similarity_score = self._distance_to_similarity(distance)
                retrieved_chunks.append({
                    "text": doc,
                    "metadata": metadata,
                    "similarity_score": similarity_score
                })
        
        return retrieved_chunks
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ) -> str:
        """
        Generate an answer using the LLM based on retrieved chunks.
        
        If max similarity is below threshold, returns refusal message immediately.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved chunks from retrieve()
            similarity_threshold: Minimum similarity score to attempt answering
            
        Returns:
            Generated answer string
        """
        if not retrieved_chunks:
            return "I don't see this in your notes."
        
        # Check max similarity score
        max_similarity = max(chunk["similarity_score"] for chunk in retrieved_chunks)
        
        if max_similarity < similarity_threshold:
            return "I don't see this in your notes."
        
        # Build context from retrieved chunks
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk["metadata"].get("source", "unknown")
            page = chunk["metadata"].get("page", "?")
            context_blocks.append(f"[Block {i}] Source: {source}, Page {page}\n{chunk['text']}")
        
        context_str = "\n\n---\n\n".join(context_blocks)
        
        # Build the system prompt
        system_prompt = (
            "You are Athena v0, a helpful study assistant. Answer the user's question using the "
            "provided context blocks as your source of truth. Do not invent facts that contradict "
            "the context. Cite the source for your claims using inline citations like [filename.pdf p.X]. "
            "If the question is brief or a single keyword (e.g. a topic name), interpret it generously "
            "and summarize what the context says about that topic. Only reply exactly with "
            "'I don't see this in your notes.' when the context blocks contain nothing relevant to the "
            "question at all."
        )
        
        # Call LLM
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context blocks:\n\n{context_str}\n\nQuestion: {query}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except APIError as e:
            return f"Error calling LLM: {str(e)}"


def create_rag_backend(
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_model: str = "gpt-4o-mini"
) -> RAGBackend:
    """
    Factory function to create a RAGBackend with initialized components.
    
    Args:
        llm_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        llm_base_url: Base URL for LLM endpoint (for local models like Ollama)
        llm_model: Model name to use for LLM
        
    Returns:
        Initialized RAGBackend instance
    """
    model = OpenAIEmbedder()
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    return RAGBackend(
        model=model,
        chroma_client=client,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model
    )
