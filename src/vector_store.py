# vector_store.py
# Updated: Removed self.client.persist() as persistence is automatic with PersistentClient
from __future__ import annotations
import hashlib
from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ChromaVectorStore:
    """
    Professional Chroma Vector Store for RAG pipelines.
    Supports:
    - Persistent embeddings
    - Metadata-rich indexing
    - Duplicate prevention via checksums
    - Configurable similarity retrieval
    """

    def __init__(
        self,
        persist_dir: str = "D:/RAG-PIPELINE/vector_db",
        collection_name: str = "rag_collection",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity: str = "l2"  # "l2", "ip", or "cosine"
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ✅ New PersistentClient API
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # ✅ Built-in embedding function
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

        metadata = {"hnsw:space": similarity} if similarity in ["l2", "ip", "cosine"] else None
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata=metadata
        )
        print(f"[Chroma] Collection: {collection_name} (persisted at {self.persist_dir}, similarity={similarity})")

    @staticmethod
    def compute_checksum(text: str) -> str:
        """Generate a stable checksum for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict],
        force_reindex: bool = False
    ):
        """
        Add texts to the vector store, skipping already indexed chunks.
        Embeddings are computed automatically by the collection's embedding_function.
        """
        if len(texts) != len(metadatas):
            raise ValueError("Mismatch between texts and metadatas count!")

        ids, final_texts, final_metadata = [], [], []

        for txt, meta in zip(texts, metadatas):
            checksum = self.compute_checksum(txt)
            existing = self.collection.get(ids=[checksum])
            if existing["ids"] and not force_reindex:
                continue  # skip already indexed
            ids.append(checksum)
            final_texts.append(txt)
            final_metadata.append(meta)

        if final_texts:
            self.collection.add(
                documents=final_texts,
                metadatas=final_metadata,
                ids=ids
            )
            print(f"[Chroma] Added {len(final_texts)} new texts.")
        else:
            print("[Chroma] No new chunks to add.")

    def add_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        force_reindex: bool = False
    ):
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("Mismatch in lengths of texts, embeddings, and metadatas!")
        
        ids, final_texts, final_embeddings, final_metadatas = [], [], [], []
        for txt, emb, meta in zip(texts, embeddings, metadatas):
            checksum = self.compute_checksum(txt)
            existing = self.collection.get(ids=[checksum])
            if existing["ids"] and not force_reindex:
                continue
            ids.append(checksum)
            final_texts.append(txt)
            final_embeddings.append(emb)
            final_metadatas.append(meta)
        
        if final_texts:
            self.collection.add(
                documents=final_texts,
                embeddings=final_embeddings,
                metadatas=final_metadatas,
                ids=ids
            )
            print(f"[Chroma] Added {len(final_texts)} new embeddings.")
        else:
            print("[Chroma] No new chunks to add.")

    def search(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        filter_by: Optional[Dict] = None
    ):
        """
        Retrieve top_k most similar documents to the query text or embedding.
        Optional metadata filtering supported.
        """
        if query is not None:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_by
            )
        elif query_embedding is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_by
            )
        else:
            raise ValueError("Provide either query (str) or query_embedding (list[float])")
        return results

    def persist(self):
        """Persist database changes to disk."""
        # Persistence is automatic with PersistentClient, no need to call persist()
        print("[Chroma] Database persisted automatically.")