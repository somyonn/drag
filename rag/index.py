from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rag.schemas import Chunk, utc_now_iso

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_and_save_index(
    embeddings: np.ndarray,
    chunks: list[Chunk],
    vectorizer: Any,
    index_dir: str | Path,
    embedding_model_id: str,
    chunk_params: dict[str, Any],
    doc_fingerprint: str,
) -> dict[str, Any]:
    index_path = Path(index_dir)
    _ensure_dir(index_path)

    meta = {
        "embedding_model_id": embedding_model_id,
        "chunk_params": chunk_params,
        "doc_fingerprint": doc_fingerprint,
        "created_at": utc_now_iso(),
        "index_backend": "faiss" if FAISS_AVAILABLE else "numpy",
        "num_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
    }

    # Normalize for cosine similarity via inner product.
    vectors = embeddings.copy().astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    vectors = vectors / norms

    if FAISS_AVAILABLE:
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)
        faiss.write_index(faiss_index, str(index_path / "index.faiss"))
    else:
        np.save(index_path / "index.npy", vectors)

    with (index_path / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    with (index_path / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    with (index_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def load_index(index_dir: str | Path) -> dict[str, Any]:
    path = Path(index_dir)
    if not path.exists():
        raise FileNotFoundError(f"Index dir not found: {path}")

    with (path / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with (path / "chunks.json").open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    with (path / "vectorizer.pkl").open("rb") as f:
        vectorizer = pickle.load(f)

    backend = meta["index_backend"]
    if backend == "faiss":
        idx = faiss.read_index(str(path / "index.faiss"))
    else:
        idx = np.load(path / "index.npy")

    return {"meta": meta, "chunks": chunks, "vectorizer": vectorizer, "index": idx}

