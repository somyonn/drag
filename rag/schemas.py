from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Document:
    doc_id: str
    source_uri: str
    text: str
    metadata: dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_uri: str
    text: str
    start_char: int
    end_char: int


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_uri: str
    score: float
    text: str


@dataclass
class QueryLog:
    timestamp: str
    query: str
    answer: str
    top_k: int
    doc_ids: list[str]
    chunk_ids: list[str]
    scores: list[float]
    source_uris: list[str]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

