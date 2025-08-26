# index_pipeline.py
# Updated: Add force_reindex option, logging
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loader import load_pdfs
from src.pro_chunking import professional_chunk_documents
from src.embedding_pipeline import EmbeddingPipeline
from src.vector_store import ChromaVectorStore
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    pdf_dir = r"D:/RAG-PIPELINE/data"

    logger.info("[STEP 1] Loading PDFs…")
    docs = load_pdfs(directory=pdf_dir, recursive=True)

    logger.info("\n[STEP 2] Chunking documents…")
    chunks = professional_chunk_documents(docs, chunk_size=1000, chunk_overlap=150)
    logger.info(f"Chunks ready: {len(chunks)}")

    logger.info("\n[STEP 3] Generating embeddings…")
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    emb_pipeline = EmbeddingPipeline(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = emb_pipeline.embed_texts(texts)

    logger.info("\n[STEP 4] Storing embeddings in Chroma…")
    store = ChromaVectorStore(
        persist_dir="D:/RAG-PIPELINE/vector_db",
        collection_name="rag_pipeline_collection",
        similarity="cosine"
    )
    store.add_embeddings(texts, vectors, metadatas, force_reindex=False)  # Added force option
    store.persist()