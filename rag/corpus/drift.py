"""Doc fingerprint helpers (used when rebuilding the index)."""

from __future__ import annotations

import hashlib


def combined_doc_fingerprint(file_fingerprints: list[str]) -> str:
    joined = "|".join(sorted(file_fingerprints))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
