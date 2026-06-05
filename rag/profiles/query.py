"""Domain-profile RAG queries (low_latency, privacy, freshness_accuracy)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.corpus.external_sync import is_external_http_sync_allowed, sync_external_docs_with_index
from rag.llm.generate import CloudThenMockLLMClient, MockLLMClient, OpenAIChatClient, build_prompt
from rag.indexing.cache import load_index_cached
from rag.core.metrics import Timer
from rag.pipeline import _append_log
from rag.indexing.retrieve import retrieve_top_k
from rag.core.schemas import RetrievedChunk, utc_now_iso
from rag.core.timing import build_latency_payload

DEFAULT_PROFILES_PATH = Path("data/config/domain_profiles.json")
OFFICIAL_INDEX_DIR = "data/index/official"
PROFILE_NAMES = ("low_latency", "privacy", "freshness_accuracy")

PROFILE_LABELS: dict[str, str] = {
    "low_latency": "Low latency",
    "privacy": "Privacy",
    "freshness_accuracy": "Freshness & accuracy",
}


def load_profiles(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_PROFILES_PATH
    if not p.exists():
        raise FileNotFoundError(f"Profile config not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# Replacement tokens per PII type.
PII_REPLACEMENTS: dict[str, str] = {
    "email": "[REDACTED_EMAIL]",
    "aws_key": "[REDACTED_AWS_KEY]",
    "rrn": "[REDACTED_RRN]",
    "credit_card": "[REDACTED_CREDIT_CARD]",
    "ip": "[REDACTED_IP]",
    "phone": "[REDACTED_PHONE]",
}

# Strict, well-formed IPv4 octets (0-255) to avoid matching version strings.
_IPV4_STRICT = r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"

# Three detector operating points. Patterns are ordered so the specific numeric
# patterns (rrn/credit_card/ip) run before the greedy phone pattern, preventing
# phone from swallowing their digits.
#   conservative: strict formats -> high precision, lower recall (misses obfuscation)
#   balanced:     default production behavior
#   aggressive:   loose/obfuscation-aware -> higher recall, more false positives
PII_PATTERN_SETS: dict[str, list[tuple[str, re.Pattern[str], str]]] = {
    "conservative": [
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[REDACTED_AWS_KEY]"),
        ("rrn", re.compile(r"\b\d{6}-\d{7}\b"), "[REDACTED_RRN]"),
        ("credit_card", re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b"), "[REDACTED_CREDIT_CARD]"),
        ("ip", re.compile(_IPV4_STRICT), "[REDACTED_IP]"),
        ("phone", re.compile(r"\b(?:\+\d{1,3}[-.\s])?0\d{1,2}[-.\s]\d{3,4}[-.\s]\d{4}\b"), "[REDACTED_PHONE]"),
    ],
    "balanced": [
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[REDACTED_AWS_KEY]"),
        ("rrn", re.compile(r"\b\d{6}-\d{7}\b"), "[REDACTED_RRN]"),
        ("credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[REDACTED_CREDIT_CARD]"),
        ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
        ("phone", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}\b"), "[REDACTED_PHONE]"),
    ],
    "aggressive": [
        # Standard + obfuscated emails (name [at] domain [dot] com).
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        (
            "email",
            re.compile(
                r"[A-Za-z0-9._%+-]+\s*(?:\[at\]|\(at\)|\s+at\s+)\s*[A-Za-z0-9.-]+\s*(?:\[dot\]|\(dot\)|\s+dot\s+)\s*[A-Za-z]{2,}",
                re.IGNORECASE,
            ),
            "[REDACTED_EMAIL]",
        ),
        ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", re.IGNORECASE), "[REDACTED_AWS_KEY]"),
        # Hyphenated and bare 13-digit resident registration numbers.
        ("rrn", re.compile(r"\b\d{6}-?\d{7}\b"), "[REDACTED_RRN]"),
        ("credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b"), "[REDACTED_CREDIT_CARD]"),
        # IPv4 (loose) + IPv6.
        ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
        ("ip", re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b"), "[REDACTED_IP]"),
        # Korean mobile without separators + separated groups.
        ("phone", re.compile(r"\b01\d[-.\s]?\d{3,4}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
        ("phone", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}\b"), "[REDACTED_PHONE]"),
    ],
}

DETECTOR_LEVELS: tuple[str, ...] = ("conservative", "balanced", "aggressive")

# Default production pattern set (backward-compatible name).
PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = PII_PATTERN_SETS["balanced"]

PII_TYPES: tuple[str, ...] = tuple(PII_REPLACEMENTS.keys())

PII_LABELS_KO: dict[str, str] = {
    "email": "이메일",
    "phone": "전화",
    "aws_key": "AWS키",
    "rrn": "주민번호",
    "credit_card": "신용카드",
    "ip": "IP",
}


def redact_with_counts(text: str, level: str = "balanced") -> tuple[str, dict[str, int]]:
    """Mask PII patterns and return (masked_text, counts_by_type).

    ``level`` selects a detector operating point: conservative/balanced/aggressive.
    """
    patterns = PII_PATTERN_SETS.get(level, PII_PATTERN_SETS["balanced"])
    counts: dict[str, int] = {}
    for name, pattern, replacement in patterns:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts


def redact_text(text: str, level: str = "balanced") -> str:
    return redact_with_counts(text, level=level)[0]


def merge_counts(target: dict[str, int], src: dict[str, int]) -> None:
    for k, v in src.items():
        target[k] = target.get(k, 0) + v


def format_masking_summary(total: dict[str, int]) -> str:
    items = [f"{PII_LABELS_KO.get(k, k)} {v}" for k, v in total.items() if v]
    body = ", ".join(items) if items else "마스킹된 개인정보 없음"
    return f"[마스킹 요약] {body}"


def apply_privacy_guard_with_counts(
    chunks: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], dict[str, int]]:
    guarded: list[RetrievedChunk] = []
    counts: dict[str, int] = {}
    for c in chunks:
        masked, c_counts = redact_with_counts(c.text)
        merge_counts(counts, c_counts)
        guarded.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                source_uri=c.source_uri,
                score=c.score,
                text=masked,
            )
        )
    return guarded, counts


def apply_privacy_guard(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return apply_privacy_guard_with_counts(chunks)[0]


# Prefer an explicit "Last updated" line, then fall back to the first ISO date.
_DATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)last\s+updated[:\s]+(\d{4})-(\d{2})-(\d{2})"),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
]


def _date_to_epoch(year: str, month: str, day: str) -> float | None:
    try:
        return datetime(int(year), int(month), int(day), tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _extract_doc_date(text: str) -> float | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            ts = _date_to_epoch(*match.groups())
            if ts is not None:
                return ts
    return None


def _doc_timestamp(source_uri: str, cache: dict[str, float]) -> float:
    """Freshness timestamp: content-embedded date if present, else file mtime."""
    if source_uri in cache:
        return cache[source_uri]
    ts = 0.0
    p = Path(source_uri)
    if p.exists():
        try:
            content_date = _extract_doc_date(p.read_text(encoding="utf-8"))
            ts = content_date if content_date is not None else p.stat().st_mtime
        except OSError:
            ts = 0.0
    cache[source_uri] = ts
    return ts


DEFAULT_FRESHNESS_WEIGHT = 0.15


def rerank_with_freshness(
    chunks: list[RetrievedChunk], weight: float = DEFAULT_FRESHNESS_WEIGHT
) -> list[RetrievedChunk]:
    """Re-rank by ``score + weight * normalized_freshness``.

    ``weight=0`` reproduces pure relevance ordering; higher weight pushes
    newer documents up at the expense of relevance.
    """
    if not chunks:
        return chunks

    cache: dict[str, float] = {}
    timestamps: dict[str, float] = {
        c.chunk_id: _doc_timestamp(c.source_uri, cache) for c in chunks
    }

    min_ts = min(timestamps.values())
    max_ts = max(timestamps.values())
    span = (max_ts - min_ts) if max_ts != min_ts else 1.0

    def sort_key(chunk: RetrievedChunk) -> float:
        freshness = (timestamps[chunk.chunk_id] - min_ts) / span
        return chunk.score + (weight * freshness)

    return sorted(chunks, key=sort_key, reverse=True)


def pick_llm(name: str = "auto"):
    """Select an LLM client by condition.

    - "mock": deterministic Mock LLM (no network, reproducible)
    - "cloud": OpenAI-compatible API only (raises on failure, no mock fallback)
    - "auto" (default): cloud first, mock fallback on failure
    """
    if name == "mock":
        return MockLLMClient()
    if name == "cloud":
        return OpenAIChatClient()
    return CloudThenMockLLMClient()


def infer_docs_dir_from_index(index_dir: str) -> str:
    p = Path(index_dir).resolve()
    return str(p.parent.parent / "docs" / p.name)


def resolve_docs_dir(profile: dict[str, Any], index_dir: str, meta: dict[str, Any]) -> Path:
    if profile.get("docs_dir"):
        return Path(str(profile["docs_dir"])).resolve()
    if meta.get("docs_dir"):
        return Path(str(meta["docs_dir"])).resolve()
    return Path(infer_docs_dir_from_index(index_dir)).resolve()


def external_sync_settings(profile: dict[str, Any]) -> dict[str, Any]:
    raw = profile.get("external_sync")
    if raw is None:
        return {"enabled": False, "sync_on_query": False, "timeout_sec": 15.0, "delay_sec": 0.05}
    if isinstance(raw, bool):
        return {"enabled": raw, "timeout_sec": 15.0, "delay_sec": 0.05}
    if isinstance(raw, dict):
        return {
            "enabled": bool(raw.get("enabled", False)),
            "sync_on_query": bool(raw.get("sync_on_query", False)),
            "timeout_sec": float(raw.get("timeout_sec", 15.0)),
            "delay_sec": float(raw.get("delay_sec", 0.05)),
        }
    return {"enabled": False, "sync_on_query": False, "timeout_sec": 15.0, "delay_sec": 0.05}


def run_profile_query(
    query: str,
    profile_name: str,
    profiles: dict[str, Any],
    llm_name: str,
    log_path: Path | str = "runs/domain_logs.jsonl",
    *,
    skip_external_sync: bool = False,
    sync_timeout_sec: float | None = None,
    sync_delay_sec: float | None = None,
    top_k: int | None = None,
    retrieval_k: int | None = None,
    freshness_weight: float | None = None,
) -> dict[str, Any]:
    if profile_name not in profiles:
        raise KeyError(f"Unknown profile: {profile_name}")

    profile = profiles[profile_name]
    top_k = int(top_k if top_k is not None else profile["top_k"])
    retrieval_k = int(retrieval_k if retrieval_k is not None else profile.get("retrieval_k", top_k))
    if retrieval_k < top_k:
        retrieval_k = top_k
    index_dir = str(profile.get("index_dir") or OFFICIAL_INDEX_DIR)

    total_timer = Timer()

    index_load_timer = Timer()
    loaded_index = load_index_cached(index_dir)
    index_load_ms = index_load_timer.elapsed_ms()
    meta = loaded_index["meta"]

    external_sync_summary: dict[str, Any] | None = None
    external_sync_ms = 0.0
    index_reload_ms = 0.0

    sync_cfg = external_sync_settings(profile) if profile.get("external_sync") is not None else None
    will_sync = bool(
        is_external_http_sync_allowed()
        and sync_cfg
        and sync_cfg["enabled"]
        and sync_cfg.get("sync_on_query")
        and not skip_external_sync
    )
    if will_sync:
        docs_dir = resolve_docs_dir(profile, index_dir, meta)
        sync_timer = Timer()
        external_sync_summary = sync_external_docs_with_index(
            docs_dir,
            index_dir,
            timeout_sec=sync_timeout_sec if sync_timeout_sec is not None else sync_cfg["timeout_sec"],
            delay_sec=sync_delay_sec if sync_delay_sec is not None else sync_cfg["delay_sec"],
        )
        external_sync_ms = sync_timer.elapsed_ms()
        if external_sync_summary.get("changed_rel_paths"):
            reload_timer = Timer()
            loaded_index = load_index_cached(index_dir)
            index_reload_ms = reload_timer.elapsed_ms()

    vector_search_timer = Timer()
    retrieved = retrieve_top_k(loaded_index, query=query, top_k=retrieval_k)
    vector_search_ms = vector_search_timer.elapsed_ms()

    profile_postprocess_ms = 0.0
    privacy_context_mask_ms = 0.0
    privacy_answer_mask_ms = 0.0
    privacy_query_mask_ms = 0.0
    context_counts: dict[str, int] = {}
    masking_summary: dict[str, Any] | None = None

    postprocess_timer = Timer()
    if profile_name == "low_latency":
        retrieved = retrieved[:top_k]
        profile_postprocess_ms = postprocess_timer.elapsed_ms()
    elif profile_name == "privacy":
        mask_timer = Timer()
        retrieved, context_counts = apply_privacy_guard_with_counts(retrieved[:top_k])
        privacy_context_mask_ms = mask_timer.elapsed_ms()
    elif profile_name == "freshness_accuracy":
        w = (
            freshness_weight
            if freshness_weight is not None
            else float(profile.get("freshness_weight", DEFAULT_FRESHNESS_WEIGHT))
        )
        retrieved = rerank_with_freshness(retrieved, weight=w)[:top_k]
        profile_postprocess_ms = postprocess_timer.elapsed_ms()
    else:
        retrieved = retrieved[:top_k]
        profile_postprocess_ms = postprocess_timer.elapsed_ms()

    # The privacy profile masks the user query before it ever reaches the LLM;
    # retrieval still uses the original query to preserve recall.
    prompt_query = query
    query_counts: dict[str, int] = {}
    if profile_name == "privacy":
        query_mask_timer = Timer()
        prompt_query, query_counts = redact_with_counts(query)
        privacy_query_mask_ms = query_mask_timer.elapsed_ms()

    prompt_build_timer = Timer()
    llm = pick_llm(llm_name)
    prompt = build_prompt(prompt_query, retrieved)
    prompt_build_ms = prompt_build_timer.elapsed_ms()
    prompt_chars = len(prompt)
    chunks_sent_to_llm = len(retrieved)

    answer, llm_request_ms = llm.generate_with_timing(prompt, retrieved)
    is_mock_answer = answer.startswith("[MOCK ANSWER]")

    answer_postprocess_ms = 0.0
    if profile_name == "privacy":
        answer_mask_timer = Timer()
        answer, answer_counts = redact_with_counts(answer)
        privacy_answer_mask_ms = answer_mask_timer.elapsed_ms()

        total_counts: dict[str, int] = {}
        merge_counts(total_counts, query_counts)
        merge_counts(total_counts, context_counts)
        merge_counts(total_counts, answer_counts)
        masking_summary = {
            "query": query_counts,
            "context": context_counts,
            "answer": answer_counts,
            "total": total_counts,
        }
        answer = f"{answer}\n\n{format_masking_summary(total_counts)}"

    total_latency = total_timer.elapsed_ms()
    latency_ms = build_latency_payload(
        total_ms=total_latency,
        index_load_ms=index_load_ms,
        index_reload_ms=index_reload_ms,
        external_sync_ms=external_sync_ms,
        vector_search_ms=vector_search_ms,
        profile_postprocess_ms=profile_postprocess_ms,
        privacy_context_mask_ms=privacy_context_mask_ms,
        privacy_query_mask_ms=privacy_query_mask_ms,
        prompt_build_ms=prompt_build_ms,
        llm_request_ms=llm_request_ms,
        answer_postprocess_ms=answer_postprocess_ms,
        privacy_answer_mask_ms=privacy_answer_mask_ms,
        chunks_sent_to_llm=chunks_sent_to_llm,
        prompt_chars=prompt_chars,
    )

    payload = {
        "timestamp": utc_now_iso(),
        "profile": profile_name,
        "llm_mode_requested": llm_name,
        "is_mock_answer": is_mock_answer,
        "query": query,
        "answer": answer,
        "top_k": top_k,
        "retrieval_k": retrieval_k,
        "doc_ids": [x.doc_id for x in retrieved],
        "chunk_ids": [x.chunk_id for x in retrieved],
        "scores": [x.score for x in retrieved],
        "source_uris": [x.source_uri for x in retrieved],
        "latency_ms": latency_ms,
        "masking_summary": masking_summary,
        "external_sync": external_sync_summary,
    }
    _append_log(log_path, payload)

    return {
        "profile": profile_name,
        "llm_mode_requested": llm_name,
        "is_mock_answer": is_mock_answer,
        "query": query,
        "answer": answer,
        "top_k": top_k,
        "retrieval_k": retrieval_k,
        "retrieved": [r.__dict__ for r in retrieved],
        "latency_ms": latency_ms,
        "masking_summary": masking_summary,
        "external_sync": external_sync_summary,
    }
