"""
Athena v0 - Embedding Model

Wraps OpenAI's embeddings API behind a SentenceTransformer-like ``encode()``
interface, so the rest of the pipeline (ingest.py, rag.py) stays unchanged.

Embeddings always go to OpenAI proper (using OPENAI_API_KEY), independent of
the chat LLM — which may point at a local endpoint that has no embeddings API.
"""

from typing import List, Optional, Union
import numpy as np
from openai import OpenAI


# text-embedding-3-small returns 1536-dimensional vectors and is L2-normalized.
EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    """Minimal embedding client exposing a SentenceTransformer-style ``encode()``."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
    ):
        """
        Args:
            api_key: OpenAI API key. If None, the OpenAI client falls back to
                the OPENAI_API_KEY environment variable.
            model: Embedding model name (default: text-embedding-3-small).
        """
        self.model = model
        # No base_url override: embeddings must hit OpenAI, not a local LLM.
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, List[float], List[List[float]]]:
        """
        Embed one string or a list of strings.

        Mirrors SentenceTransformer.encode: a single string returns a single
        vector, a list returns a list/array of vectors.
        """
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        # OpenAI rejects empty strings; substitute a single space.
        inputs = [t if t and t.strip() else " " for t in inputs]

        response = self.client.embeddings.create(model=self.model, input=inputs)
        # Sort by index to guarantee alignment with the input order.
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]

        if convert_to_numpy:
            arr = np.array(vectors, dtype=np.float32)
            return arr[0] if single else arr
        return vectors[0] if single else vectors
