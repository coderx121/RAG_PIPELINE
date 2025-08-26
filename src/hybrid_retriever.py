# hybrid_retriever.py
# Updated: Load corpus from Chroma in init for alignment
from __future__ import annotations  # Must be the first import!

import sys, os
# --- FIX: Add project root to sys.path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------------------

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder  # optional reranker

from src.embedding_pipeline import EmbeddingPipeline
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _zscore(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    x = np.array(x, dtype=float)
    mu = x.mean()
    sigma = x.std()
    if sigma == 0:
        return x - mu
    return (x - mu) / (sigma + 1e-12)


def mmr_rerank(query_vec: np.ndarray, cand_vecs: np.ndarray, top_k: int, diversity: float = 0.7) -> List[int]:
    """Maximal Marginal Relevance selection."""
    if len(cand_vecs) == 0:
        return []
    selected = []
    candidates = list(range(len(cand_vecs)))
    sims = (cand_vecs @ query_vec).flatten()
    first = int(np.argmax(sims))
    selected.append(first)
    candidates.remove(first)
    while len(selected) < min(top_k, len(cand_vecs)):
        best_score = -1e9
        best_idx = None
        for c in candidates:
            sim_to_query = sims[c]
            sim_to_selected = max((cand_vecs[c] @ cand_vecs[s]).item() for s in selected)
            score = (1 - diversity) * sim_to_query - diversity * sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = c
        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected


class HybridRetriever:
    def __init__(
        self,
        embedder: Optional[EmbeddingPipeline] = None,
        chroma_persist_dir: str = "D:/RAG-PIPELINE/vector_db",
        chroma_collection: str = "rag_pipeline_collection",
        bm25_tokenizer=None,
        reranker_model_name: Optional[str] = None,
    ):
        """
        HybridRetriever combines BM25 (lexical) and Chroma (vector) retrieval.
        """
        self.embedder = embedder or EmbeddingPipeline()
        self.vs = ChromaVectorStore(persist_dir=chroma_persist_dir, collection_name=chroma_collection)
        
        # Load corpus from Chroma for BM25 alignment
        all_docs = self.vs.collection.get(include=['documents', 'metadatas'])
        chunks = [{'text': doc, 'metadata': meta, 'id': id} for id, doc, meta in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas'])]
        self.build_bm25_from_chunks(chunks)

        # Optional reranker
        self.reranker = None
        if reranker_model_name:
            try:
                self.reranker = CrossEncoder(reranker_model_name)
                logger.info(f"[HybridRetriever] Loaded CrossEncoder: {reranker_model_name}")
            except Exception as e:
                logger.warning(f"[HybridRetriever] Failed to load reranker '{reranker_model_name}': {e}")

    def build_bm25_from_chunks(self, chunks: List[Dict[str, Any]]):
        """Build BM25 index from chunks."""
        tokenized, texts, metas, ids = [], [], [], []
        for i, c in enumerate(chunks):
            text = c.get("text") if isinstance(c, dict) else getattr(c, "page_content", "")
            texts.append(text or "")
            metas.append(c.get("metadata", {}) if isinstance(c, dict) else getattr(c, "metadata", {}))
            ids.append(c.get("id", f"chunk_{i}"))
            tokenized.append((text or "").split())

        if not tokenized:
            raise ValueError("No chunks provided to build BM25 index.")

        self.bm25 = BM25Okapi(tokenized)
        self.corpus_texts = texts
        self.corpus_metas = metas
        self.corpus_ids = ids
        logger.info(f"[HybridRetriever] BM25 built on {len(texts)} chunks")

    def _bm25_search(self, query: str, top_n: int):
        if not self.bm25:
            return [], np.array([])
        scores = np.array(self.bm25.get_scores(query.split()), dtype=float)
        top_idx = np.argsort(scores)[::-1][:top_n]
        return top_idx.tolist(), scores[top_idx]

    def _vector_search(self, query_embedding: List[float], top_n: int):
        res = self.vs.search(query_embedding=query_embedding, top_k=top_n)
        ids = res.get("ids", [[]])[0]
        dists = np.array(res.get("distances", [[]])[0], dtype=float)
        idxs = []
        for _id in ids:
            try:
                idxs.append(self.corpus_ids.index(_id))
            except ValueError:
                idxs.append(-1)
        return idxs, dists

    def retrieve(
        self,
        query: str,
        k: int = 10,
        bm25_k: int = 50,
        vector_k: int = 50,
        fusion_alpha: float = 0.5,
        rerank_top_k: int = 10,
        use_mmr: bool = True,
        mmr_diversity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval pipeline."""
        bm25_idxs, bm25_scores = self._bm25_search(query, bm25_k)

        qvec = self.embedder.embed_query(query)
        vec_idxs, vec_dists = self._vector_search(qvec, vector_k)
        vec_sims = 1 - vec_dists  # convert cosine distance to similarity (updated fix)

        candidate_set = set(bm25_idxs) | set(vec_idxs)
        candidate_set = [c for c in candidate_set if c >= 0]
        if not candidate_set:
            return []

        bm_scores = np.array([bm25_scores[bm25_idxs.index(c)] if c in bm25_idxs else 0.0 for c in candidate_set], float)
        vec_scores = np.array([vec_sims[vec_idxs.index(c)] if c in vec_idxs else 0.0 for c in candidate_set], float)

        bm_z = _zscore(bm_scores)
        vec_z = _zscore(vec_scores)
        fused = fusion_alpha * vec_z + (1 - fusion_alpha) * bm_z

        candidates = []
        for i, idx in enumerate(candidate_set):
            candidates.append({
                "corpus_index": idx,
                "text": self.corpus_texts[idx],
                "metadata": self.corpus_metas[idx],
                "bm25_score": float(bm_scores[i]),
                "vector_score": float(vec_scores[i]),
                "fused_score": float(fused[i])
            })

        candidates.sort(key=lambda x: x["fused_score"], reverse=True)
        top_candidates = candidates[:max(rerank_top_k, k)]

        # Optional reranking
        if self.reranker:
            pairs = [(query, c["text"]) for c in top_candidates]
            try:
                rerank_scores = self.reranker.predict(pairs)
                for c, s in zip(top_candidates, rerank_scores):
                    c["rerank_score"] = float(s)
                top_candidates.sort(key=lambda x: x.get("rerank_score", x["fused_score"]), reverse=True)
            except Exception as e:
                logger.warning(f"[HybridRetriever] Reranker failed: {e}")

        if use_mmr:
            cand_vecs = np.vstack([self.embedder.embed_texts([c["text"]])[0] for c in top_candidates])
            selected_idxs = mmr_rerank(np.array(qvec, float), cand_vecs, top_k=k, diversity=mmr_diversity)
            final = [top_candidates[i] for i in selected_idxs]
        else:
            final = top_candidates[:k]

        for rank, item in enumerate(final, start=1):
            item["rank"] = rank

        return final