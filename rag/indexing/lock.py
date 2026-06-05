"""Process-wide lock while the index files are being rewritten (background sync)."""

from __future__ import annotations

import threading

_index_lock = threading.Lock()


def index_read_lock():
    return _index_lock
