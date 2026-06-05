#!/usr/bin/env python3
"""Batch eval over KB test queries: baseline + domain profiles with latency and retrieval stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, pstdev

from dotenv import load_dotenv

from scripts._eval_common import (
    install_thread_safe_logging,
    load_labels,
    load_queries,
    write_report,
)
from rag.indexing.cache import load_index_cached, warm_index_cache
from rag.core.metrics import summarize_latencies_ms
from rag.pipeline import query_pipeline
from rag.profiles.query import (
    DEFAULT_PROFILES_PATH,
    PROFILE_NAMES,
    load_profiles,
    pick_llm,
    run_profile_query,
)


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def corpus_from_source_uri(uri: str) -> str:
    normalized = uri.replace("\\", "/")
    for tag in ("aws", "docker", "google_drive"):
        if f"/docs/{tag}/" in normalized:
            return tag
    return "other"


def top_corpus(source_uris: list[str]) -> str:
    if not source_uris:
        return "none"
    counts = Counter(corpus_from_source_uri(u) for u in source_uris)
    return counts.most_common(1)[0][0]


def uri_matches_groups(uri_lc: str, groups: list[list[str]]) -> bool:
    """Relevance rule: a doc is relevant if it matches ANY group, where a group
    matches when ALL of its substrings are present (OR-of-AND)."""
    return any(all(token in uri_lc for token in group) for group in groups)


def count_relevant_in_corpus(chunks: list[dict], groups: list[list[str]]) -> int:
    """Number of unique documents in the corpus that satisfy the relevance rule."""
    unique_uris = {c.get("source_uri", "").lower() for c in chunks}
    return sum(1 for u in unique_uris if u and uri_matches_groups(u, groups))


def dedupe_preserve_order(source_uris: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for u in source_uris:
        key = u.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(u)
    return ordered


def ranking_metrics(
    source_uris: list[str], groups: list[list[str]], total_relevant: int
) -> dict[str, float | None]:
    """Document-level ranking metrics against a curated gold relevant set.

    - mrr: reciprocal rank of the first relevant doc in the ranked list (0 if none).
    - ndcg_at_k: binary-gain nDCG over the ranked (deduped) docs.
    - recall_at_k: unique relevant docs retrieved / total relevant docs in corpus.
    """
    if not groups or total_relevant <= 0:
        return {"mrr": None, "ndcg_at_k": None, "recall_at_k": None}

    ranked = dedupe_preserve_order(source_uris)
    rels = [1 if uri_matches_groups(u.lower(), groups) else 0 for u in ranked]

    mrr = 0.0
    for i, rel in enumerate(rels, start=1):
        if rel:
            mrr = 1.0 / i
            break

    dcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(rels, start=1))
    ideal_hits = min(total_relevant, len(rels))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    retrieved_relevant = sum(rels)
    recall = retrieved_relevant / total_relevant

    return {
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg, 4),
        "recall_at_k": round(recall, 4),
    }


def evaluate_quality(label: dict, source_uris: list[str], answer: str) -> dict:
    """Compute golden-label quality metrics for one query result.

    - corpus_correct: top retrieved corpus matches expected_corpus
    - hit_at_k: any top-k source path contains an expected source substring
    - keyword_recall: fraction of expected answer keywords present (heuristic proxy)
    - mrr / ndcg_at_k / recall_at_k: document-level ranking vs curated gold set
      (only when the label provides relevant_uri_substrings + precomputed
      _relevant_count from the corpus).
    """
    expected_corpus = label.get("expected_corpus")
    corpus_correct = (top_corpus(source_uris) == expected_corpus) if expected_corpus else None

    substrings = [s.lower() for s in label.get("expected_source_substrings", [])]
    lowered_uris = [u.lower() for u in source_uris]
    hit_at_k = (any(s in u for s in substrings for u in lowered_uris)) if substrings else None

    keywords = [k.lower() for k in label.get("expected_keywords", [])]
    answer_lc = (answer or "").lower()
    keyword_recall = (
        round(sum(1 for k in keywords if k in answer_lc) / len(keywords), 4) if keywords else None
    )

    groups = [[t.lower() for t in g] for g in label.get("relevant_uri_substrings", [])]
    total_relevant = int(label.get("_relevant_count", 0))
    ranks = ranking_metrics(source_uris, groups, total_relevant)

    return {
        "corpus_correct": corpus_correct,
        "hit_at_k": hit_at_k,
        "keyword_recall": keyword_recall,
        "mrr": ranks["mrr"],
        "ndcg_at_k": ranks["ndcg_at_k"],
        "recall_at_k": ranks["recall_at_k"],
    }


def row_from_result(mode: str, query: str, result: dict, label: dict | None = None) -> dict:
    latency = result.get("latency_ms") or {}
    retrieved = result.get("retrieved") or []
    source_uris = [r.get("source_uri", "") for r in retrieved]
    scores = [float(r.get("score", 0.0)) for r in retrieved]
    answer = result.get("answer", "") or ""
    citation_present = ("sources:" in answer.lower()) or ("http" in answer.lower())
    row = {
        "mode": mode,
        "query": query,
        "top_k": result.get("top_k"),
        "retrieval_k": result.get("retrieval_k"),
        "is_mock_answer": bool(result.get("is_mock_answer")),
        "top_corpus": top_corpus(source_uris),
        "source_corpora": dict(Counter(corpus_from_source_uri(u) for u in source_uris)),
        "mean_retrieval_score": round(mean(scores), 4) if scores else 0.0,
        "top1_score": round(scores[0], 4) if scores else 0.0,
        "citation_present": citation_present,
        "latency_ms": latency,
        "source_uris": source_uris[:3],
    }
    if label is not None:
        row["quality"] = evaluate_quality(label, source_uris, answer)
    return row


def summarize_rows(rows: list[dict]) -> dict:
    totals = [float(r["latency_ms"]["total"]) for r in rows if r.get("latency_ms")]
    retrievals = [float(r["latency_ms"]["retrieval"]) for r in rows if r.get("latency_ms")]
    generations = [float(r["latency_ms"]["generation"]) for r in rows if r.get("latency_ms")]
    index_loads = [float(r["latency_ms"].get("index_load_ms", 0.0)) for r in rows if r.get("latency_ms")]
    mock_rate = sum(1 for r in rows if r.get("is_mock_answer")) / len(rows) if rows else 0.0
    corpus_hits = Counter(r.get("top_corpus", "none") for r in rows)
    citation_rate = (
        round(sum(1 for r in rows if r.get("citation_present")) / len(rows), 3) if rows else 0.0
    )

    labeled = [r for r in rows if "quality" in r]
    quality: dict | None = None
    if labeled:
        corpus_flags = [1.0 if r["quality"]["corpus_correct"] else 0.0 for r in labeled if r["quality"]["corpus_correct"] is not None]
        hit_flags = [1.0 if r["quality"]["hit_at_k"] else 0.0 for r in labeled if r["quality"]["hit_at_k"] is not None]
        kw_vals = [r["quality"]["keyword_recall"] for r in labeled if r["quality"]["keyword_recall"] is not None]
        mrr_vals = [r["quality"]["mrr"] for r in labeled if r["quality"].get("mrr") is not None]
        ndcg_vals = [r["quality"]["ndcg_at_k"] for r in labeled if r["quality"].get("ndcg_at_k") is not None]
        recall_vals = [r["quality"]["recall_at_k"] for r in labeled if r["quality"].get("recall_at_k") is not None]
        quality = {
            "num_labeled": len(labeled),
            "corpus_accuracy": _avg(corpus_flags),
            "hit_at_k": _avg(hit_flags),
            "keyword_recall": _avg(kw_vals),
            "mrr": _avg(mrr_vals),
            "ndcg_at_k": _avg(ndcg_vals),
            "recall_at_k": _avg(recall_vals),
        }

    summary = {
        "num_queries": len(rows),
        "mock_answer_rate": round(mock_rate, 3),
        "top_corpus_distribution": dict(corpus_hits),
        "mean_retrieval_score": round(mean([r["mean_retrieval_score"] for r in rows]), 4) if rows else 0.0,
        "citation_rate": citation_rate,
        "total_latency_ms": summarize_latencies_ms(totals),
        "retrieval_latency_ms": summarize_latencies_ms(retrievals),
        "generation_latency_ms": summarize_latencies_ms(generations),
        "index_load_ms": summarize_latencies_ms(index_loads),
    }
    if quality is not None:
        summary["quality"] = quality
    return summary


def per_query_latency_stats(rows: list[dict]) -> list[dict]:
    """Within-query latency variation across trials (mean/std of total_ms)."""
    grouped: dict[str, list[float]] = defaultdict(list)
    label: dict[str, str] = {}
    for r in rows:
        latency = r.get("latency_ms") or {}
        if "total" not in latency:
            continue
        key = r["query"]
        grouped[key].append(float(latency["total"]))
        label[key] = r["query"]
    stats: list[dict] = []
    for key, totals in grouped.items():
        stats.append(
            {
                "query": label[key],
                "trials": len(totals),
                "total_mean_ms": round(mean(totals), 4),
                "total_std_ms": round(pstdev(totals), 4) if len(totals) > 1 else 0.0,
                "total_min_ms": round(min(totals), 4),
                "total_max_ms": round(max(totals), 4),
            }
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KB batch eval (baseline + domain profiles)")
    parser.add_argument("--queries", default="data/queries/kb_eval.txt")
    parser.add_argument("--labels", default="data/queries/kb_eval_labeled.jsonl")
    parser.add_argument("--index-dir", default="data/index/official")
    parser.add_argument("--profiles-path", default=str(DEFAULT_PROFILES_PATH))
    parser.add_argument(
        "--report-path",
        default=None,
        help="Report output path. Default: runs/kb_eval_report_{llm}.json (auto -> runs/kb_eval_report.json).",
    )
    parser.add_argument("--log-baseline", default="runs/kb_eval_baseline.jsonl")
    parser.add_argument("--log-domain", default="runs/kb_eval_domain.jsonl")
    parser.add_argument("--modes", default="baseline,low_latency,privacy,freshness_accuracy")
    parser.add_argument(
        "--llm",
        choices=("auto", "mock", "cloud"),
        default="auto",
        help="LLM condition. mock=deterministic Mock LLM (no network, reproducible); "
        "cloud=OpenAI API only (no fallback); auto=cloud first, mock fallback.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Recorded repetitions per (mode, query). >1 enables latency distribution stats.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Unrecorded warmup repetitions per (mode, query) before timed trials.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of queries to run in parallel (1 = sequential). LLM calls are I/O-bound; "
        "parallel runs finish faster but per-query latency reflects concurrent load. "
        "Use --concurrency 1 for clean latency distributions.",
    )
    return parser.parse_args()


def default_report_path(llm_condition: str) -> str:
    if llm_condition == "auto":
        return "runs/kb_eval_report.json"
    return f"runs/kb_eval_report_{llm_condition}.json"


def resolve_model_id(llm_condition: str) -> str | None:
    if llm_condition == "mock":
        return "mock"
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def run_one(
    mode: str,
    query: str,
    *,
    args: argparse.Namespace,
    profiles: dict,
    llm,
) -> dict:
    if mode == "baseline":
        return query_pipeline(
            query=query,
            index_dir=args.index_dir,
            log_path=args.log_baseline,
            llm_client=llm,
        )
    if mode in PROFILE_NAMES:
        return run_profile_query(
            query=query,
            profile_name=mode,
            profiles=profiles,
            llm_name=args.llm,
            log_path=args.log_domain,
            skip_external_sync=True,
        )
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    queries = load_queries(Path(args.queries))
    labels = load_labels(Path(args.labels))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    profiles = load_profiles(Path(args.profiles_path))
    # Construct once for the baseline path; the profile path builds per call via pick_llm.
    llm = pick_llm(args.llm)
    concurrency = max(1, int(args.concurrency))
    trials = max(1, int(args.trials))
    warmup = max(0, int(args.warmup))

    warm_index_cache(args.index_dir)

    # Precompute the corpus-wide relevant-document universe per labeled query so
    # ranking metrics (recall@k) have a stable denominator across all trials.
    loaded_index = load_index_cached(args.index_dir)
    corpus_chunks = loaded_index.get("chunks", [])
    for label in labels.values():
        groups = [[t.lower() for t in g] for g in label.get("relevant_uri_substrings", [])]
        label["_relevant_count"] = count_relevant_in_corpus(corpus_chunks, groups) if groups else 0

    if concurrency > 1:
        install_thread_safe_logging()

    # Optional unrecorded warmup pass (sequential) to prime caches / connections.
    for _ in range(warmup):
        for mode in modes:
            for query in queries:
                run_one(mode, query, args=args, profiles=profiles, llm=llm)

    def run_task(task: tuple[int, str, int, int, str]) -> tuple[int, str, int, int, dict]:
        order, mode, q_idx, trial, query = task
        result = run_one(mode, query, args=args, profiles=profiles, llm=llm)
        return order, mode, q_idx, trial, row_from_result(mode, query, result, labels.get(query))

    tasks: list[tuple[int, str, int, int, str]] = []
    for mode in modes:
        for q_idx, query in enumerate(queries):
            for trial in range(trials):
                tasks.append((len(tasks), mode, q_idx, trial, query))

    started = time.perf_counter()
    if concurrency == 1:
        results = [run_task(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(run_task, tasks))
    wall_clock_sec = round(time.perf_counter() - started, 3)

    by_mode_clean: dict[str, list[dict]] = {mode: [] for mode in modes}
    for order, mode, _q_idx, _trial, row in sorted(results, key=lambda x: x[0]):
        by_mode_clean[mode].append(row)
    all_rows = [row for mode in modes for row in by_mode_clean[mode]]

    report = {
        "queries_file": args.queries,
        "index_dir": args.index_dir,
        "modes": modes,
        "llm_condition": args.llm,
        "model_id": resolve_model_id(args.llm),
        "trials": trials,
        "warmup": warmup,
        "concurrency": concurrency,
        "wall_clock_sec": wall_clock_sec,
        "per_mode": {mode: summarize_rows(rows) for mode, rows in by_mode_clean.items()},
        "per_mode_query_latency": {
            mode: per_query_latency_stats(rows) for mode, rows in by_mode_clean.items()
        },
        "overall": summarize_rows(all_rows),
        "runs": all_rows,
    }

    report_path = write_report(args.report_path or default_report_path(args.llm), report)

    summary_keys = (
        "queries_file", "index_dir", "modes", "llm_condition", "model_id",
        "trials", "warmup", "concurrency", "wall_clock_sec", "per_mode", "overall",
    )
    print(json.dumps({k: report[k] for k in summary_keys}, ensure_ascii=False, indent=2))
    print(f"\nFull per-query details: {report_path}")


if __name__ == "__main__":
    main()
