"""Tool wrappers reused by the Athena v3 planning loop."""

from __future__ import annotations

import os
from typing import Dict, List

import chromadb
from openai import OpenAI

from v2.chain import CHROMA_DB_DIR, COLLECTION_NAME, _distance_to_similarity, get_athena_v2_engine


def search_notes(query: str, k: int = 5) -> Dict[str, object]:
    """Search Athena's existing Chroma notes through the v2 vector store."""
    try:
        engine = get_athena_v2_engine()
        results = engine.vector_store.similarity_search_with_score(query, k=k)
        chunks: List[Dict[str, object]] = []
        for doc, distance in results:
            source = str(doc.metadata.get("source", "unknown"))
            page = str(doc.metadata.get("page", "?"))
            similarity = _distance_to_similarity(distance)
            chunks.append(
                {
                    "text": doc.page_content,
                    "source": source,
                    "page": page,
                    "similarity_score": similarity,
                    "citation": f"[{source} p.{page}]",
                }
            )
        return {"query": query, "chunks": chunks, "mode": "vector"}
    except Exception as exc:
        return _keyword_search_notes(query, k=k, error=str(exc))


def web_search(query: str) -> Dict[str, object]:
    """
    Search the web using OpenAI's hosted web-search tool when available.

    The repository did not contain an existing web_search implementation, so this
    is the smallest local wrapper around the provider tool. It degrades to a
    normal model response if the hosted search tool is unavailable.
    """
    model = os.getenv("ATHENA_WEB_MODEL", os.getenv("ATHENA_CHAT_MODEL", "gpt-4o-mini"))
    client = OpenAI()
    try:
        response = client.responses.create(
            model=model,
            input=f"Search the web and summarize current evidence for: {query}",
            tools=[{"type": "web_search_preview"}],
        )
        text = getattr(response, "output_text", "") or str(response)
        return {"query": query, "summary": text, "sources": _extract_urls(text)}
    except Exception as exc:
        try:
            fallback = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer from general knowledge only if web search is unavailable. "
                            "State that live web search failed."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
                max_tokens=700,
            )
            text = fallback.choices[0].message.content or ""
            return {
                "query": query,
                "summary": f"Live web search failed: {exc}\n\nFallback summary:\n{text}",
                "sources": [],
            }
        except Exception as fallback_exc:
            return {
                "query": query,
                "summary": (
                    "Live web search failed and no chat fallback was available. "
                    f"Search error: {exc}; fallback error: {fallback_exc}"
                ),
                "sources": [],
            }


def _extract_urls(text: str) -> List[str]:
    """Extract simple URL-like citations from a text blob."""
    urls = []
    for token in text.replace(")", " ").replace("]", " ").split():
        if token.startswith(("http://", "https://")):
            urls.append(token.rstrip(".,;"))
    return list(dict.fromkeys(urls))


def _keyword_search_notes(query: str, k: int, error: str) -> Dict[str, object]:
    """Local fallback over stored Chroma documents when embedding search fails."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    stored = collection.get(include=["documents", "metadatas"])
    terms = {term.strip(".,?:;!()[]").lower() for term in query.split()}
    terms = {term for term in terms if len(term) > 2}
    scored = []
    for doc, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
        lowered = doc.lower()
        score = sum(1 for term in terms if term in lowered)
        if score:
            scored.append((score, doc, metadata or {}))
    scored.sort(key=lambda item: item[0], reverse=True)

    chunks: List[Dict[str, object]] = []
    for score, doc, metadata in scored[:k]:
        source = str(metadata.get("source", "unknown"))
        page = str(metadata.get("page", "?"))
        chunks.append(
            {
                "text": doc,
                "source": source,
                "page": page,
                "similarity_score": min(1.0, score / max(len(terms), 1)),
                "citation": f"[{source} p.{page}]",
            }
        )
    return {"query": query, "chunks": chunks, "mode": "keyword_fallback", "error": error}
