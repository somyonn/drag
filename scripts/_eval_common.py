"""Shared helpers for the evaluation scripts (query/label IO, reporting, logging).

Imported by run_kb_eval / run_judge_eval / run_privacy_eval / run_freshness_eval
to avoid duplicating the same boilerplate in each entry point.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


def load_queries(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Query file not found: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of records. Missing file -> empty list."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_labels(path: str | Path, key: str = "query") -> dict[str, dict]:
    """Load JSONL labels into a dict keyed by ``key`` (default 'query')."""
    return {record[key]: record for record in load_jsonl(path)}


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def install_thread_safe_logging() -> None:
    """Serialize jsonl appends so concurrent workers don't interleave log lines.

    Patches the shared ``_append_log`` used by both the baseline pipeline and the
    profile-query path. Call once before launching a ThreadPoolExecutor.
    """
    import rag.pipeline as rag_pipeline
    import rag.profiles.query as rag_profile

    lock = threading.Lock()
    original = rag_pipeline._append_log

    def locked_append(log_path, payload):  # type: ignore[no-untyped-def]
        with lock:
            original(log_path, payload)

    rag_pipeline._append_log = locked_append
    rag_profile._append_log = locked_append
