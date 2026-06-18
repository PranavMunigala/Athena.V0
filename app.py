"""
Athena v0 - Streamlit Web UI

Interactive web interface for the RAG system with sidebar analytics
and retrieved context visibility.

Required packages:
    pip install pymupdf sentence-transformers chromadb openai streamlit
"""

import os
import streamlit as st
from dotenv import load_dotenv
st.set_page_config(
    page_title="Athena v0 - RAG Study Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from rag import RAGBackend


# Load environment variables from .env if present
load_dotenv()


# Configuration
CHROMA_DB_DIR = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD_PERCENT = 15

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .subtitle {
        color: #666;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    .chunk-container {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .similarity-badge {
        background-color: #e8f4f8;
        color: #1f77b4;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9em;
        display: inline-block;
        margin-top: 8px;
    }
    .source-badge {
        background-color: #f0f0f0;
        color: #333;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        display: inline-block;
        margin-top: 6px;
    }
    .answer-container {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 20px 0;
    }
    .refusal-container {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ffc107;
        margin: 20px 0;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedding_model():
    """Load the embedding model (cached to avoid reinitialization)."""
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_chroma_client():
    """Load the Chroma persistent client (cached to avoid reinitialization)."""
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)


@st.cache_resource
def initialize_rag_backend():
    """Initialize the RAG backend with cached components."""
    model = load_embedding_model()
    chroma_client = load_chroma_client()
    
    # Get API key and base URL from environment
    llm_api_key = os.getenv("OPENAI_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", None)
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    return RAGBackend(
        model=model,
        chroma_client=chroma_client,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model
    )


def check_database_populated():
    """Check if the Chroma database has any documents."""
    try:
        chroma_client = load_chroma_client()
        collection = chroma_client.get_or_create_collection(name="athena_notes")
        count = collection.count()
        return count > 0
    except Exception as e:
        st.error(f"Error checking database: {e}")
        return False


def format_similarity_score(score: float) -> str:
    """Format similarity score as percentage."""
    return f"{score * 100:.1f}%"


def main():
    """Main Streamlit application."""
    
    # Main title
    st.markdown('<div class="main-title">🏛️ Athena v0</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">RAG Study Assistant — Query Your Lecture Notes</div>', unsafe_allow_html=True)
    
    # Check if database is populated
    if not check_database_populated():
        st.warning(
            "⚠️ No documents found in the database. "
            "Please run `python ingest.py` to populate the database with your PDF files from the `corpus/` folder."
        )
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        llm_model = st.text_input(
            "LLM Model Name",
            value=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            help="e.g., 'gpt-4o-mini', 'gpt-4', or 'llama3' for local Ollama"
        )
        
        st.divider()
        st.header("📊 Retrieved Context Analytics")
        context_placeholder = st.empty()

    llm_api_key = os.getenv("OPENAI_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", None)
    if not llm_api_key and not llm_base_url:
        st.error(
            "OPENAI_API_KEY is not set in your environment or .env file. "
            "Add OPENAI_API_KEY to .env before launching Athena."
        )
        st.stop()
    
    # Main input area
    query = st.text_input(
        "Ask a question about your notes:",
        placeholder="e.g., What is the definition of photosynthesis?",
        help="Your question will be answered based on the retrieved context from your PDFs."
    )
    
    answer_placeholder = st.empty()
    
    if query:
        # Initialize RAG backend
        try:
            # Create a new backend instance with user-provided credentials
            model = load_embedding_model()
            chroma_client = load_chroma_client()
            
            rag = RAGBackend(
                model=model,
                chroma_client=chroma_client,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model=llm_model
            )
            
            # Retrieve relevant chunks
            with st.spinner("🔍 Retrieving relevant context..."):
                retrieved_chunks = rag.retrieve(query, k=5)
            
            # Display retrieved context in sidebar
            if retrieved_chunks:
                with context_placeholder.container():
                    st.subheader("Top 3 Retrieved Chunks")
                    
                    for i, chunk in enumerate(retrieved_chunks[:3], 1):
                        with st.expander(
                            f"📄 Chunk {i} — "
                            f"Similarity: {format_similarity_score(chunk['similarity_score'])}"
                        ):
                            st.write(chunk["text"])
                            st.markdown(
                                f"**Source:** {chunk['metadata'].get('source', 'unknown')} | "
                                f"**Page:** {chunk['metadata'].get('page', '?')}",
                                help="Source file and page number of this context block"
                            )
            
            # Generate answer
            with st.spinner("✍️ Generating answer..."):
                answer = rag.generate_answer(query, retrieved_chunks)
            
            # Display answer
            if answer == "I don't see this in your notes.":
                with answer_placeholder.container():
                    st.markdown(
                        f'<div class="refusal-container">'
                        f'<strong>⚠️ Low Confidence</strong><br>{answer}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    if retrieved_chunks:
                        max_sim = max(c["similarity_score"] for c in retrieved_chunks)
                        st.info(
                            f"The highest similarity score ({format_similarity_score(max_sim)}) "
                            f"fell below the confidence threshold ({SIMILARITY_THRESHOLD_PERCENT}%). "
                            f"Check the retrieved context in the sidebar to see what was found."
                        )
            else:
                with answer_placeholder.container():
                    st.markdown(
                        f'<div class="answer-container">{answer}</div>',
                        unsafe_allow_html=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error(
                "Please ensure:\n"
                "- Your OpenAI API key is correct\n"
                "- The database has been populated with `python ingest.py`\n"
                "- All dependencies are installed"
            )


if __name__ == "__main__":
    main()
