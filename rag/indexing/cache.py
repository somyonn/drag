"""In-memory cache for loaded vector indexes (avoid re-reading chunks.json every query)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from rag.indexing.index import load_index
from rag.indexing.lock import index_read_lock

_cache: dict[str, tuple[str, dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def _index_fingerprint(index_dir: Path) -> str:
    """Change when any on-disk index artifact is replaced."""
    parts: list[str] = []
    for name in ("meta.json", "chunks.json", "vectorizer.pkl", "index.faiss", "index.npy"):
        p = index_dir / name
        if p.is_file():
            st = p.stat()
            parts.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
    return "|".join(parts) if parts else "missing"


def invalidate_index_cache(index_dir: str | Path | None = None) -> None:
    with _cache_lock:
        if index_dir is None:
            _cache.clear()
            return
        key = str(Path(index_dir).resolve())
        _cache.pop(key, None)


def load_index_cached(index_dir: str | Path) -> dict[str, Any]:
    path = Path(index_dir).resolve()
    key = str(path)
    fp = _index_fingerprint(path)

    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == fp:
            return hit[1]

    with index_read_lock():
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None and hit[0] == fp:
                return hit[1]
        loaded = load_index(path)
        fp = _index_fingerprint(path)

    with _cache_lock:
        _cache[key] = (fp, loaded)
    return loaded


def warm_index_cache(index_dir: str | Path) -> None:
    """Load index at startup (e.g. web server lifespan)."""
    load_index_cached(index_dir)
