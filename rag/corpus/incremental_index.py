from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rag.indexing.chunk import chunk_document
from rag.corpus.drift import combined_doc_fingerprint
from rag.indexing.embed import embed_passages
from rag.indexing.index import build_and_save_index, load_index
from rag.indexing.ingest import load_documents
from rag.core.schemas import Chunk


def _norm_path(p: str | Path) -> str:
    return str(Path(p).resolve())


def _extract_embedding_matrix(loaded: dict[str, Any]) -> np.ndarray:
    """Row-aligned with chunks.json (same normalization as stored in index)."""
    idx = loaded["index"]
    meta = loaded["meta"]
    if meta["index_backend"] == "faiss":
        d = int(idx.d)
        n = int(idx.ntotal)
        out = np.empty((n, d), dtype=np.float32)
        for i in range(n):
            out[i] = idx.reconstruct(i)
        return out
    return np.asarray(idx, dtype=np.float32)


def incremental_reindex(
    docs_dir: str | Path,
    index_dir: str | Path,
    changed_rel_paths: list[str],
) -> dict[str, Any]:
    """
    Remove chunks for changed files, re-chunk and TF-IDF-transform only those documents,
    then rebuild the vector index and chunks.json. Other documents' chunks and vectors
    are preserved (same vocabulary / vectorizer).
    """
    docs_root = Path(docs_dir).resolve()
    idx_root = Path(index_dir).resolve()
    if not changed_rel_paths:
        return {"skipped": True, "reason": "no_changed_paths"}

    loaded = load_index(idx_root)
    meta = loaded["meta"]
    if meta.get("embedding_model_id") != "tfidf-ngram-1-2":
        raise ValueError("incremental_reindex only supports tfidf-ngram-1-2 indexes")

    chunk_params = meta.get("chunk_params") or {}
    chunk_size = int(chunk_params.get("chunk_size", 500))
    overlap = int(chunk_params.get("overlap", 100))

    chunks_json: list[dict[str, Any]] = loaded["chunks"]
    vectorizer = loaded["vectorizer"]
    old_matrix = _extract_embedding_matrix(loaded)

    if len(chunks_json) != old_matrix.shape[0]:
        raise ValueError("chunks.json length does not match index row count; run full ingest_pipeline")

    changed_abs = {_norm_path(docs_root / rel) for rel in changed_rel_paths}
    kept_indices = [i for i, c in enumerate(chunks_json) if _norm_path(c["source_uri"]) not in changed_abs]

    all_docs = load_documents(docs_dir)
    want_paths = changed_abs & {_norm_path(d.source_uri) for d in all_docs}
    updated_docs = [d for d in all_docs if _norm_path(d.source_uri) in want_paths]

    new_chunks: list[Chunk] = []
    for doc in updated_docs:
        new_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))

    kept_chunks = [Chunk(**chunks_json[i]) for i in kept_indices]
    merged_chunks: list[Chunk] = kept_chunks + new_chunks

    kept_matrix = old_matrix[kept_indices] if kept_indices else np.zeros((0, old_matrix.shape[1]), dtype=np.float32)
    new_texts = [c.text for c in new_chunks]
    new_matrix = embed_passages(vectorizer, new_texts) if new_texts else np.zeros((0, old_matrix.shape[1]), dtype=np.float32)

    full_matrix = np.vstack([kept_matrix, new_matrix]).astype(np.float32)
    fingerprint = combined_doc_fingerprint([d.metadata["fingerprint"] for d in all_docs])

    out_meta = build_and_save_index(
        embeddings=full_matrix,
        chunks=merged_chunks,
        vectorizer=vectorizer,
        index_dir=idx_root,
        embedding_model_id=str(meta.get("embedding_model_id", "tfidf-ngram-1-2")),
        chunk_params={"chunk_size": chunk_size, "overlap": overlap},
        doc_fingerprint=fingerprint,
        docs_dir=docs_root,
    )
    return {
        "skipped": False,
        "num_changed_files": len(changed_rel_paths),
        "num_kept_chunks": len(kept_chunks),
        "num_new_chunks": len(new_chunks),
        "num_total_chunks": len(merged_chunks),
        "meta": out_meta,
    }
