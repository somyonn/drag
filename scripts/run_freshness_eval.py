#!/usr/bin/env python3
"""Freshness-profile evaluation.

Three analyses:

A. Controlled ordering: near-identical documents differing only by an embedded
   date verify that the rerank orders results newest-first when relevance is
   equal (mechanism check).

B. Relevance-freshness trade-off (ablation): a realistic corpus with varied
   relevance scores plus a fresh-but-irrelevant distractor. Sweeping the
   freshness weight shows freshness ordering (Kendall tau) improving while
   relevance@k eventually degrades.

C. mtime fallback degradation: quantifies how the freshness signal collapses
   when documents lack a parseable content date (uniform crawl mtime).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripts._eval_common import load_jsonl, write_report
from rag.pipeline import ingest_pipeline
from rag.profiles.query import (
    DEFAULT_PROFILES_PATH,
    _doc_timestamp,
    _extract_doc_date,
    load_profiles,
    run_profile_query,
)

DEFAULT_REPORT = "runs/freshness_eval_report.json"
DEFAULT_WEIGHTS = [0.0, 0.05, 0.15, 0.3, 0.5, 1.0]

CONTROLLED_DOCS = "data/docs_freshness_test"
CONTROLLED_INDEX = "data/index/freshness_test"
CONTROLLED_LABELS = "data/queries/freshness_eval_labeled.jsonl"

HARD_DOCS = "data/docs_freshness_hard"
HARD_INDEX = "data/index/freshness_hard"
HARD_LABELS = "data/queries/freshness_hard_labeled.jsonl"

NODATE_DOCS = "data/docs_freshness_nodate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate freshness ordering, trade-off, and fallback.")
    parser.add_argument("--llm", default="mock")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--log-path", default="runs/freshness_eval_logs.jsonl")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def ensure_index(docs_dir: str, index_dir: str, rebuild: bool) -> None:
    meta = Path(index_dir) / "meta.json"
    if rebuild or not meta.exists():
        ingest_pipeline(docs_dir=docs_dir, index_dir=index_dir)


def order_metrics(ranked_dates: list[date]) -> dict[str, Any]:
    n = len(ranked_dates)
    if n < 2:
        return {"top1_newest_correct": True, "pairwise_concordance": 1.0, "kendall_tau": 1.0, "exact_order_match": True}
    concordant = 0
    pairs = 0
    for i, j in combinations(range(n), 2):
        pairs += 1
        if ranked_dates[i] >= ranked_dates[j]:
            concordant += 1
    concordance = concordant / pairs
    return {
        "top1_newest_correct": ranked_dates[0] == max(ranked_dates),
        "pairwise_concordance": round(concordance, 4),
        "kendall_tau": round(2 * concordance - 1, 4),
        "exact_order_match": ranked_dates == sorted(ranked_dates, reverse=True),
    }


# ---------------------------------------------------------------------------
# A. Controlled ordering
# ---------------------------------------------------------------------------

def run_controlled(args: argparse.Namespace) -> dict[str, Any]:
    ensure_index(CONTROLLED_DOCS, CONTROLLED_INDEX, args.rebuild_index)
    profiles = load_profiles(Path(DEFAULT_PROFILES_PATH))
    profiles["freshness_accuracy"]["index_dir"] = CONTROLLED_INDEX
    labels = load_jsonl(CONTROLLED_LABELS)
    cases = []
    for label in labels:
        doc_dates = {k: date.fromisoformat(v) for k, v in label["doc_dates"].items()}
        result = run_profile_query(
            query=label["query"], profile_name="freshness_accuracy", profiles=profiles,
            llm_name=args.llm, log_path=Path(args.log_path), skip_external_sync=True,
            top_k=label.get("top_k"), retrieval_k=label.get("retrieval_k"),
        )
        ranked = [r["source_uri"] for r in result["retrieved"]]
        ranked_dates = [doc_dates[u] for u in ranked if u in doc_dates]
        cases.append({"id": label.get("id"), "ordering": order_metrics(ranked_dates),
                      "has_appended_citation": "Sources:" in result["answer"]})
    return {"num_cases": len(cases), "cases": cases}


# ---------------------------------------------------------------------------
# B. Relevance-freshness trade-off (weight ablation)
# ---------------------------------------------------------------------------

def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    ensure_index(HARD_DOCS, HARD_INDEX, args.rebuild_index)
    profiles = load_profiles(Path(DEFAULT_PROFILES_PATH))
    profiles["freshness_accuracy"]["index_dir"] = HARD_INDEX
    labels = load_jsonl(HARD_LABELS)

    curves: list[dict[str, Any]] = []
    for label in labels:
        doc_dates = {k: date.fromisoformat(v) for k, v in label["doc_dates"].items()}
        relevant = set(label["relevant"])
        top_k = int(label.get("top_k", 4))
        points = []
        for w in DEFAULT_WEIGHTS:
            result = run_profile_query(
                query=label["query"], profile_name="freshness_accuracy", profiles=profiles,
                llm_name=args.llm, log_path=Path(args.log_path), skip_external_sync=True,
                top_k=top_k, retrieval_k=label.get("retrieval_k"), freshness_weight=w,
            )
            ranked = [r["source_uri"] for r in result["retrieved"]]
            ranked_dates = [doc_dates[u] for u in ranked if u in doc_dates]
            relevance_at_k = round(sum(1 for u in ranked if u in relevant) / len(ranked), 4) if ranked else 0.0
            metrics = order_metrics(ranked_dates)
            points.append({
                "weight": w,
                "relevance_at_k": relevance_at_k,
                "freshness_kendall_tau": metrics["kendall_tau"],
                "top1_newest": metrics["top1_newest_correct"],
                "ranked": [Path(u).name for u in ranked],
            })
        curves.append({"id": label.get("id"), "top_k": top_k, "operating_points": points})
    return {"weights": DEFAULT_WEIGHTS, "num_cases": len(curves), "curves": curves}


# ---------------------------------------------------------------------------
# C. mtime fallback degradation
# ---------------------------------------------------------------------------

def timestamp_profile(docs_dir: str) -> dict[str, Any]:
    cache: dict[str, float] = {}
    content_dated = 0
    mtime_fallback = 0
    timestamps: list[float] = []
    for p in sorted(Path(docs_dir).glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        has_date = _extract_doc_date(text) is not None
        content_dated += int(has_date)
        mtime_fallback += int(not has_date)
        timestamps.append(_doc_timestamp(str(p), cache))
    span = (max(timestamps) - min(timestamps)) if timestamps else 0.0
    return {
        "docs_dir": docs_dir,
        "num_docs": len(timestamps),
        "content_dated": content_dated,
        "mtime_fallback": mtime_fallback,
        "timestamp_span_sec": round(span, 3),
        # A usable freshness signal needs the docs to be at least ~1 day apart.
        "freshness_signal_usable": span > 86400.0,
    }


def main() -> None:
    load_dotenv()
    args = parse_args()

    report = {
        "controlled": run_controlled(args),
        "tradeoff_ablation": run_ablation(args),
        "fallback": {
            "with_content_dates": timestamp_profile(HARD_DOCS),
            "without_content_dates": timestamp_profile(NODATE_DOCS),
        },
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, report)

    print("Controlled ordering (case 0):", json.dumps(report["controlled"]["cases"][0]["ordering"], ensure_ascii=False))
    print("\nTrade-off ablation (relevance@k vs freshness tau):")
    for pt in report["tradeoff_ablation"]["curves"][0]["operating_points"]:
        print(f"  w={pt['weight']:<5} relevance@k={pt['relevance_at_k']:<6} tau={pt['freshness_kendall_tau']}")
    print("\nFallback:", json.dumps(report["fallback"], ensure_ascii=False))
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
