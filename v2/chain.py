"""
Athena v2 LCEL RAG chain.

This module keeps the Week 4 retrieval behavior, but moves composition onto
LangChain Expression Language (LCEL). The Chroma store and corpus remain shared
at the repository root so v1 and v2 can query the same local notes index.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"langchain.*")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.globals import set_debug
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_classic.memory import ConversationBufferWindowMemory
except ModuleNotFoundError:

    class ConversationBufferWindowMemory:
        """Compatibility implementation for LangChain's removed window memory API."""

        def __init__(
            self,
            *,
            k: int,
            memory_key: str,
            input_key: str,
            output_key: str,
            return_messages: bool = False,
        ) -> None:
            self.k = k
            self.memory_key = memory_key
            self.input_key = input_key
            self.output_key = output_key
            self.return_messages = return_messages
            self._turns: List[Dict[str, str]] = []

        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
            lines = []
            for turn in self._turns[-self.k :]:
                lines.append(f"Human: {turn[self.input_key]}")
                lines.append(f"Athena: {turn[self.output_key]}")
            return {self.memory_key: "\n".join(lines)}

        def save_context(
            self,
            inputs: Dict[str, str],
            outputs: Dict[str, str],
        ) -> None:
            self._turns.append(
                {
                    self.input_key: inputs[self.input_key],
                    self.output_key: outputs[self.output_key],
                }
            )


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parents[1]
CHROMA_DB_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "athena_notes"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = os.getenv("ATHENA_CHAT_MODEL", "gpt-4o-mini")
SIMILARITY_THRESHOLD = 0.15
REFUSAL_MESSAGE = "I don't see this in your notes."
CITATION_PAGE_SPACE_RE = re.compile(r"(\[[^\]]+?\.pdf p\.)\s+(\d+\])")
CITATION_PAGE_WORD_RE = re.compile(r"\[([^,\]]+?\.pdf),\s*Page\s+(\d+)\]")


def _configure_langsmith_tracing() -> None:
    """Honor the requested LANGCHAIN_TRACING_V2 flag and map it to LangSmith."""
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    if tracing_enabled:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "athena-v2-rag")
        os.environ.setdefault("LANGSMITH_PROJECT", "athena-v2-rag")


def configure_langchain_debug(enabled: bool = True) -> None:
    """Turn on LangChain's local debug stream for prompt and runnable wiring."""
    set_debug(enabled)


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine distance into the v1 cosine similarity convention."""
    similarity = 1.0 - float(distance)
    return max(0.0, min(1.0, similarity))


def _format_chat_history(chat_history: Any) -> str:
    """Normalize memory output into prompt-friendly text."""
    if not chat_history:
        return "No prior conversation."
    if isinstance(chat_history, str):
        return chat_history
    if isinstance(chat_history, list):
        return "\n".join(str(message) for message in chat_history)
    return str(chat_history)


def _format_documents(docs: List[Document]) -> str:
    """Render retrieved Documents with citation metadata visible to the model."""
    blocks = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[Block {index}] Source: {source}, Page {page}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def _normalize_inline_citations(answer: str) -> str:
    """Normalize common model citation variants to [filename.pdf p.X]."""
    answer = CITATION_PAGE_SPACE_RE.sub(r"\1\2", answer)
    return CITATION_PAGE_WORD_RE.sub(r"[\1 p.\2]", answer)


def _build_memory() -> ConversationBufferWindowMemory:
    """Create the exact six-turn buffer required by the v2 spec."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return ConversationBufferWindowMemory(
            k=6,
            memory_key="chat_history",
            input_key="query",
            output_key="answer",
            return_messages=False,
        )


class AthenaLCELRAG:
    """Small service wrapper around the LCEL retrieval, refusal, and answer path."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_CHAT_MODEL,
        temperature: float = 0.3,
        retrieval_k: int = 5,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        _configure_langsmith_tracing()

        if os.getenv("ATHENA_LANGCHAIN_DEBUG", "").lower() == "true":
            configure_langchain_debug(True)

        self.retrieval_k = retrieval_k
        self.similarity_threshold = similarity_threshold
        self.memory = _build_memory()
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR),
        )
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.parser = StrOutputParser()
        self.chain = self._build_chain()

    def _retrieve_documents(self, inputs: Dict[str, Any]) -> List[Document]:
        """Retrieve notes and attach v1-compatible similarity scores to metadata."""
        query = inputs["query"]
        results = self.vector_store.similarity_search_with_score(
            query,
            k=self.retrieval_k,
        )

        docs: List[Document] = []
        for doc, distance in results:
            metadata = dict(doc.metadata)
            metadata["similarity_score"] = _distance_to_similarity(distance)
            docs.append(Document(page_content=doc.page_content, metadata=metadata))
        return docs

    def _max_similarity(self, inputs: Dict[str, Any]) -> float:
        """Read the strongest retrieved similarity from the LCEL state."""
        docs = inputs.get("retrieved_docs", [])
        if not docs:
            return 0.0
        return max(float(doc.metadata.get("similarity_score", 0.0)) for doc in docs)

    def _build_chain(self):
        """Compose retrieval, refusal, and the canonical prompt | model | parser chain."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Athena v2, a precise study assistant. Use only the "
                    "provided notes context as your source of truth. Cite claims "
                    "inline exactly as [filename.pdf p.X], with no space between "
                    "p. and the page number. If the context does not support "
                    "the answer, reply exactly: "
                    f"{REFUSAL_MESSAGE}",
                ),
                (
                    "human",
                    "Recent chat history:\n{chat_history}\n\n"
                    "Context blocks:\n{context}\n\n"
                    "Question: {question}",
                ),
            ]
        )

        # Required LCEL core: prompt | model | parser.
        answer_chain = prompt | self.model | self.parser

        # RunnablePassthrough keeps original inputs while the retriever component
        # injects matching Chroma documents for downstream formatting and gating.
        retrieval_stage = RunnablePassthrough.assign(
            retrieved_docs=RunnableLambda(self._retrieve_documents)
        ).assign(max_similarity=RunnableLambda(self._max_similarity))

        prepare_prompt_inputs = {
            "question": RunnableLambda(lambda x: x["query"]),
            "chat_history": RunnableLambda(
                lambda x: _format_chat_history(x.get("chat_history"))
            ),
            "context": RunnableLambda(lambda x: _format_documents(x["retrieved_docs"])),
        }

        grounded_answer_chain = prepare_prompt_inputs | answer_chain

        def answer_or_refuse(inputs: Dict[str, Any]) -> str:
            if inputs["max_similarity"] < self.similarity_threshold:
                return REFUSAL_MESSAGE
            return grounded_answer_chain.invoke(inputs)

        return retrieval_stage | RunnableLambda(answer_or_refuse)

    def query(
        self,
        query: str,
        *,
        chat_history: Optional[str] = None,
        remember: bool = True,
    ) -> str:
        """Answer one question, using memory history unless explicit history is passed."""
        memory_vars = self.memory.load_memory_variables({})
        history = chat_history if chat_history is not None else memory_vars["chat_history"]
        answer = self.chain.invoke({"query": query, "chat_history": history})
        answer = _normalize_inline_citations(answer)

        if remember:
            self.memory.save_context({"query": query}, {"answer": answer})
        return answer


_ENGINE: Optional[AthenaLCELRAG] = None


def get_athena_v2_engine() -> AthenaLCELRAG:
    """Return a process-local engine so Streamlit and evals reuse memory and clients."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = AthenaLCELRAG()
    return _ENGINE


def query_athena_v2(query: str, chat_history: Optional[str] = None) -> str:
    """Public query function used by the evaluation harness and future app code."""
    return get_athena_v2_engine().query(query, chat_history=chat_history)
