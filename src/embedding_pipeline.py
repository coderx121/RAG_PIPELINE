# embedding_pipeline.py
# No changes needed, as per analysis
from __future__ import annotations
from typing import List, Optional, Literal
from sentence_transformers import SentenceTransformer
import numpy as np
import re

class EmbeddingPipeline:
    """
    Professional-grade embedding pipeline for RAG.
    Supports preprocessing, batching, normalization, and model flexibility.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,  # 'cuda' or 'cpu', auto-detect if None
        normalize: bool = True,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.batch_size = batch_size

        dim = self.model.get_sentence_embedding_dimension()
        print(f"[EmbeddingPipeline] Loaded model: {model_name} (dim={dim}, device={self.model.device})")

    # --- Step 1: Preprocessing ---
    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic normalization: remove excessive whitespace, control chars."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        return [self._clean_text(t) for t in texts]

    # --- Step 2: Encoding ---
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Batch encodes texts into embeddings.
        """
        texts = self.preprocess_texts(texts)
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            emb = self.model.encode(batch, normalize_embeddings=self.normalize, convert_to_numpy=True)
            embeddings.append(emb)
        return np.vstack(embeddings).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]