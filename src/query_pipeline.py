# query_pipeline.py
# Updated: Use professional chunking, fix build_vector_store redundancy, add logging/metrics
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Optional
import logging

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Existing modules
from src.loader import load_pdfs
from src.pro_chunking import professional_chunk_documents  # Unified to professional
from src.embedding_pipeline import EmbeddingPipeline
from src.vector_store import ChromaVectorStore  # Custom wrapper for consistency

# Utilities
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Set it with your Groq API key.")

# Configurable parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
LLM_MODEL = "openai/gpt-oss-20b"
MAX_TOKENS = 1024
TEMPERATURE = 0.5
MAX_CONTEXT_LENGTH = 8000  # Approximate token limit for context

def truncate_context(context: str, max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """Truncate context to fit within token limits (approximate via chars)."""
    if len(context) > max_length:
        return context[:max_length] + "... [truncated]"
    return context

def process_documents(pdf_dir: str) -> List[Document]:
    """Load and chunk PDFs into documents using professional chunking."""
    logger.info(f"Processing PDFs from {pdf_dir}…")
    docs = load_pdfs(directory=pdf_dir, recursive=True)
    chunks = professional_chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def build_vector_store(chunks: List[Document]) -> Chroma:
    """Build and return Chroma vector store with embeddings."""
    logger.info("Building vector store…")
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    embedder = EmbeddingPipeline(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = embedder.embed_texts(texts)
    
    # Use custom store to add without re-embedding
    vector_store = ChromaVectorStore(
        persist_dir="D:/RAG-PIPELINE/vector_db",
        collection_name="rag_pipeline_collection",
        similarity="cosine"
    )
    vector_store.add_embeddings(texts, vectors, metadatas)
    vector_store.persist()
    
    # Return LangChain Chroma wrapper for retrieval (loads existing)
    return Chroma(
        persist_directory="D:/RAG-PIPELINE/vector_db",
        collection_name="rag_pipeline_collection",
        embedding_function=embedder.model
    )

if __name__ == "__main__":
    pdf_dir = r"D:/RAG-PIPELINE/data"

    # Step 1: Process documents
    chunks = process_documents(pdf_dir)

    # Step 2: Build vector store
    vector_store = build_vector_store(chunks)

    # Step 3: Initialize Groq LLM
    logger.info("Initializing Groq LLM…")
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            base_url="https://api.groq.com/openai/v1"  # Matches your curl endpoint
        )
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}")
        sys.exit(1)

    # Step 4: Set up RetrievalQA chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5, "mmr": True, "mmr_diversity": 0.7})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple concatenation of context
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    # Step 5: Query example
    query = "Explain the importance of fast language models"
    logger.info(f"Processing query: '{query}'")
    try:
        response = qa_chain({"query": query})
        answer = response["result"]
        context = "\n\n".join([f"Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page_number')}\n{doc.page_content}" for doc in response["source_documents"]])
        context = truncate_context(context)

        logger.info("Query completed successfully.")
        print(f"\nQuery: {query}")
        print(f"\nAnswer: {answer}")
        print(f"\nContext:\n{context}")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        sys.exit(1)