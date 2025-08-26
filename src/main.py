# app.py (Updated for frontend connection)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Adjust if needed for imports

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware  # For frontend cross-origin requests
from pydantic import BaseModel
from typing import Dict, Any, Optional
import tempfile
from contextlib import asynccontextmanager

# Your RAG imports (adapt paths as needed)
from src.query_pipeline import process_documents, build_vector_store, truncate_context, GROQ_API_KEY, LLM_MODEL, TEMPERATURE, MAX_TOKENS
from src.loader import load_pdfs
from src.pro_chunking import professional_chunk_documents
from src.embedding_pipeline import EmbeddingPipeline
from src.vector_store import ChromaVectorStore
from src.hybrid_retriever import HybridRetriever  # For hybrid (optional)
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

app = FastAPI(title="RAG Pipeline API")

# CORS for frontend (allow all origins for dev; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vars (load once on startup)
VECTOR_STORE = None
QA_CHAIN = None
HYBRID_RETRIEVER = None  # Optional
EMBEDDER = None
CHROMA_STORE = None  # Custom store for adding new docs

@asynccontextmanager
async def lifespan(app: FastAPI):
    global VECTOR_STORE, QA_CHAIN, HYBRID_RETRIEVER, EMBEDDER, CHROMA_STORE
    pdf_dir = r"D:/RAG-PIPELINE/data"
    
    # Build or load vector store (run indexing if DB empty)
    chunks = process_documents(pdf_dir)
    VECTOR_STORE = build_vector_store(chunks)
    
    EMBEDDER = EmbeddingPipeline(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    CHROMA_STORE = ChromaVectorStore(
        persist_dir="D:/RAG-PIPELINE/vector_db",
        collection_name="rag_pipeline_collection",
        similarity="cosine"
    )
    
    # Optional: Use hybrid retriever (uncomment to enable)
    # HYBRID_RETRIEVER = HybridRetriever(reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")  # Example reranker
    
    # Initialize LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        base_url="https://api.groq.com/openai/v1"
    )
    
    # Set up chain (use hybrid if enabled)
    if HYBRID_RETRIEVER:
        # Custom retrieval logic in /api/rag endpoint
        QA_CHAIN = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=None,  # We'll handle retrieval manually
            return_source_documents=True,
            verbose=True
        )
    else:
        retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 5, "mmr": True, "mmr_diversity": 0.7})
        QA_CHAIN = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
    yield  # No shutdown code needed unless you add some

app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)

@app.post("/api/rag", response_model=Dict[str, Any])
async def rag_query(
    query: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    if not QA_CHAIN:
        raise HTTPException(status_code=500, detail="RAG chain not initialized")
    
    try:
        # Handle uploaded file if present
        if file and file.filename.lower().endswith('.pdf'):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            
            # Load, chunk, embed, add to store
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            chunks = professional_chunk_documents(docs, chunk_size=1000, chunk_overlap=150)
            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            vectors = EMBEDDER.embed_texts(texts)
            CHROMA_STORE.add_embeddings(texts, vectors, metadatas)
            CHROMA_STORE.persist()
            
            # Clean up temp file
            os.unlink(tmp_path)
        
        # Perform query
        if HYBRID_RETRIEVER:
            # Use hybrid retrieval
            results = HYBRID_RETRIEVER.retrieve(query, k=5)
            docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in results]
            response = QA_CHAIN({"query": query, "input_documents": docs})
        else:
            response = QA_CHAIN({"query": query})
        
        answer = response["result"]
        sources = response["source_documents"]
        
        # Format response as markdown
        formatted_response = f"{answer}\n\n**Sources:**\n" + "\n".join(
            [f"- {doc.metadata.get('source')} (Page {doc.metadata.get('page_number')})" for doc in sources]
        )
        
        return {"response": formatted_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)