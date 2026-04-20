from __future__ import annotations

from typing import Any

import numpy as np

from rag.embed import embed_query
from rag.schemas import RetrievedChunk


def _search_numpy(index_vectors: np.ndarray, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    q = query_vec.astype("float32")
    q = q / (np.linalg.norm(q) + 1e-12)
    scores = index_vectors @ q
    top_idx = np.argsort(scores)[::-1][:top_k]
    return scores[top_idx], top_idx


def retrieve_top_k(loaded_index: dict[str, Any], query: str, top_k: int = 3) -> list[RetrievedChunk]:
    vectorizer = loaded_index["vectorizer"]
    query_vec = embed_query(vectorizer, query)
    index = loaded_index["index"]
    chunks: list[dict[str, Any]] = loaded_index["chunks"]

    if loaded_index["meta"]["index_backend"] == "faiss":
        q = query_vec.astype("float32")[None, :]
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        scores, ids = index.search(q, top_k)
        raw_scores = scores[0]
        raw_ids = ids[0]
    else:
        raw_scores, raw_ids = _search_numpy(index, query_vec, top_k=top_k)

    results: list[RetrievedChunk] = []
    for score, idx in zip(raw_scores, raw_ids):
        idx = int(idx)
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                source_uri=chunk["source_uri"],
                score=float(score),
                text=chunk["text"],
            )
        )

    return results

