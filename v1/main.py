"""Public Athena v1 entry points used by the v2 evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from v1.rag import create_rag_backend


_BACKEND = None


def _get_backend():
    """Reuse the hand-rolled v1 backend across repeated eval questions."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = create_rag_backend()
    return _BACKEND


def query_athena_v1(query: str, chat_history: Optional[str] = None) -> str:
    """Answer a query through the preserved v1 hand-rolled RAG pipeline."""
    backend = _get_backend()
    chunks = backend.retrieve(query)
    return backend.generate_answer(query, chunks)


def main():
    print(query_athena_v1("What is Synchron?"))


if __name__ == "__main__":
    main()
