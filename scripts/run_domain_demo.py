#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.generate import CloudThenMockLLMClient, build_prompt
from rag.index import load_index
from rag.metrics import Timer
from rag.pipeline import _append_log
from rag.retrieve import retrieve_top_k
from rag.schemas import RetrievedChunk, utc_now_iso


PROFILE_PATH = Path("data/config/domain_profiles.json")


def load_profiles(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Profile config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def route_services(query: str) -> list[str]:
    q = query.lower()
    selected: list[str] = []
    if any(k in q for k in ("iam", "role", "policy", "permission", "access key")):
        selected.append("iam")
    if any(k in q for k in ("s3", "bucket", "object", "lifecycle", "presigned")):
        selected.append("s3")
    if any(k in q for k in ("ec2", "instance", "ami", "eip", "autoscaling")):
        selected.append("ec2")
    return selected


def filter_chunks_by_service(chunks: list[RetrievedChunk], services: list[str]) -> list[RetrievedChunk]:
    if not services:
        return chunks
    lowered = [s.lower() for s in services]
    return [c for c in chunks if any(f"/{svc}/" in c.source_uri.lower() for svc in lowered)]


def redact_text(text: str) -> str:
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    phone_pattern = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}\b")
    account_pattern = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
    text = email_pattern.sub("[REDACTED_EMAIL]", text)
    text = phone_pattern.sub("[REDACTED_PHONE]", text)
    text = account_pattern.sub("[REDACTED_AWS_KEY]", text)
    return text


def apply_privacy_guard(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    guarded: list[RetrievedChunk] = []
    for c in chunks:
        guarded.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                source_uri=c.source_uri,
                score=c.score,
                text=redact_text(c.text),
            )
        )
    return guarded


def rerank_with_freshness(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return chunks

    timestamps: dict[str, float] = {}
    for c in chunks:
        p = Path(c.source_uri)
        if p.exists():
            timestamps[c.chunk_id] = p.stat().st_mtime
        else:
            timestamps[c.chunk_id] = 0.0

    min_ts = min(timestamps.values())
    max_ts = max(timestamps.values())
    span = (max_ts - min_ts) if max_ts != min_ts else 1.0

    def sort_key(chunk: RetrievedChunk) -> float:
        freshness = (timestamps[chunk.chunk_id] - min_ts) / span
        return chunk.score + (0.15 * freshness)

    return sorted(chunks, key=sort_key, reverse=True)


def pick_llm(name: str):
    if name in {"openai", "auto"}:
        # Always fall back to mock on cloud call failure.
        return CloudThenMockLLMClient()
    raise ValueError(f"Unsupported llm mode: {name}")


def run_profile_query(
    query: str,
    profile_name: str,
    profiles: dict[str, Any],
    llm_name: str,
    log_path: Path,
) -> dict[str, Any]:
    profile = profiles[profile_name]
    top_k = int(profile["top_k"])
    retrieval_k = int(profile.get("retrieval_k", top_k))
    index_dir = profile["index_dir"]

    total_timer = Timer()
    retrieval_timer = Timer()
    loaded_index = load_index(index_dir)
    retrieved = retrieve_top_k(loaded_index, query=query, top_k=retrieval_k)

    if profile_name == "low_latency":
        routed_services = route_services(query)
        filtered = filter_chunks_by_service(retrieved, routed_services)
        if filtered:
            retrieved = filtered
        retrieved = retrieved[:top_k]
    elif profile_name == "privacy":
        retrieved = apply_privacy_guard(retrieved[:top_k])
    elif profile_name == "freshness_accuracy":
        retrieved = rerank_with_freshness(retrieved)[:top_k]
    else:
        retrieved = retrieved[:top_k]

    retrieval_latency = retrieval_timer.elapsed_ms()

    generation_timer = Timer()
    llm = pick_llm(llm_name)
    prompt = build_prompt(query, retrieved)
    answer = llm.generate(prompt, retrieved)
    is_mock_answer = answer.startswith("[MOCK ANSWER]")

    if profile_name == "privacy":
        answer = redact_text(answer)
    if profile_name == "freshness_accuracy":
        citations = "\n".join(f"- {r.source_uri}" for r in retrieved)
        answer = f"{answer}\n\nSources:\n{citations}"

    generation_latency = generation_timer.elapsed_ms()
    total_latency = total_timer.elapsed_ms()

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
        "retrieval_latency_ms": retrieval_latency,
        "generation_latency_ms": generation_latency,
        "total_latency_ms": total_latency,
    }
    _append_log(log_path, payload)

    return {
        "profile": profile_name,
        "llm_mode_requested": llm_name,
        "is_mock_answer": is_mock_answer,
        "query": query,
        "answer": answer,
        "retrieved": [r.__dict__ for r in retrieved],
        "latency_ms": {
            "retrieval": retrieval_latency,
            "generation": generation_latency,
            "total": total_latency,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG query using domain-specific policy profiles.")
    parser.add_argument("--profile", choices=["low_latency", "privacy", "freshness_accuracy"], required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--llm", choices=["auto", "openai"], default="auto")
    parser.add_argument("--profiles-path", default=str(PROFILE_PATH))
    parser.add_argument("--log-path", default="runs/domain_logs.jsonl")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    profiles = load_profiles(Path(args.profiles_path))
    result = run_profile_query(
        query=args.query,
        profile_name=args.profile,
        profiles=profiles,
        llm_name=args.llm,
        log_path=Path(args.log_path),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
