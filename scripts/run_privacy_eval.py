#!/usr/bin/env python3
"""Privacy-profile evaluation.

Two complementary analyses:

A. End-to-end masking (pipeline integrity): runs the privacy profile over a
   labeled query and verifies the input query and retrieved context are masked
   before reaching the LLM, with type-wise masking counts and per-stage latency.

B. Detector precision/recall (honest robustness): applies the regex detector at
   three operating points (conservative/balanced/aggressive) to a labeled corpus
   that includes obfuscated PII (hard positives) and PII-like decoys (hard
   negatives), producing per-type and overall precision/recall/F1 and a PR curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripts._eval_common import load_jsonl, write_report
from rag.pipeline import ingest_pipeline
from rag.profiles.query import (
    DEFAULT_PROFILES_PATH,
    DETECTOR_LEVELS,
    PII_TYPES,
    load_profiles,
    redact_with_counts,
    run_profile_query,
)

DEFAULT_DOCS_DIR = "data/docs_privacy_test"
DEFAULT_INDEX_DIR = "data/index/privacy_test"
DEFAULT_LABELS = "data/queries/privacy_eval_labeled.jsonl"
DEFAULT_TRUTH = "data/queries/privacy_pii_truth.jsonl"
DEFAULT_REPORT = "runs/privacy_eval_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the privacy profile end-to-end and detector PR.")
    parser.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--truth", default=DEFAULT_TRUTH)
    parser.add_argument("--llm", default="mock", help="LLM mode (mock keeps the eval deterministic)")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--log-path", default="runs/privacy_eval_logs.jsonl")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def ensure_index(docs_dir: str, index_dir: str, rebuild: bool) -> None:
    meta = Path(index_dir) / "meta.json"
    if rebuild or not meta.exists():
        ingest_pipeline(docs_dir=docs_dir, index_dir=index_dir)


# ---------------------------------------------------------------------------
# Section A: end-to-end pipeline masking
# ---------------------------------------------------------------------------

def recall_by_type(detected: dict[str, int], expected: dict[str, int]) -> dict[str, Any]:
    per_type: dict[str, Any] = {}
    total_expected = 0
    total_hit = 0
    for t in PII_TYPES:
        exp = int(expected.get(t, 0))
        det = int(detected.get(t, 0))
        hit = min(det, exp)
        per_type[t] = {"expected": exp, "detected": det, "recall": round(hit / exp, 4) if exp else 1.0}
        total_expected += exp
        total_hit += hit
    return {"per_type": per_type, "micro_recall": round(total_hit / total_expected, 4) if total_expected else 1.0}


def run_e2e(args: argparse.Namespace, labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = load_profiles(Path(DEFAULT_PROFILES_PATH))
    profiles["privacy"]["index_dir"] = args.index_dir
    cases: list[dict[str, Any]] = []
    for label in labels:
        result = run_profile_query(
            query=label["query"],
            profile_name="privacy",
            profiles=profiles,
            llm_name=args.llm,
            log_path=Path(args.log_path),
            skip_external_sync=True,
            top_k=label.get("top_k"),
            retrieval_k=label.get("retrieval_k"),
        )
        summary = result.get("masking_summary") or {}
        latency = result["latency_ms"]
        retrieval_ms = latency["retrieval_ms"]
        generation_ms = latency["generation_ms"]

        # Leakage check: re-scan the already-masked context with the most
        # aggressive detector. Anything still found is PII that the production
        # (balanced) masker left behind -> an honest leakage measurement.
        residual: dict[str, int] = {}
        for r in result["retrieved"]:
            _, c = redact_with_counts(r["text"], level="aggressive")
            for k, v in c.items():
                residual[k] = residual.get(k, 0) + v
        residual = {k: v for k, v in residual.items() if v}

        cases.append(
            {
                "id": label.get("id"),
                "is_mock_answer": result["is_mock_answer"],
                "query_masking": recall_by_type(summary.get("query", {}), label.get("expected_query_pii", {})),
                "context_residual_pii": residual,
                "context_no_leakage": not residual,
                "stage_counts": {
                    "query": summary.get("query", {}),
                    "context": summary.get("context", {}),
                    "answer": summary.get("answer", {}),
                },
                "masking_latency_ms": {
                    "query_mask_ms": retrieval_ms.get("privacy_query_mask_ms", 0.0),
                    "context_mask_ms": retrieval_ms.get("privacy_context_mask_ms", 0.0),
                    "answer_mask_ms": generation_ms.get("privacy_answer_mask_ms", 0.0),
                },
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Section B: detector precision/recall over a labeled corpus
# ---------------------------------------------------------------------------

def evaluate_level(level: str, truth: list[dict[str, Any]]) -> dict[str, Any]:
    """Count-based TP/FP/FN per type across labeled docs (no span alignment)."""
    tp = {t: 0 for t in PII_TYPES}
    fp = {t: 0 for t in PII_TYPES}
    fn = {t: 0 for t in PII_TYPES}
    fp_examples: list[dict[str, Any]] = []
    fn_examples: list[dict[str, Any]] = []

    for row in truth:
        text = Path(row["doc"]).read_text(encoding="utf-8")
        _, detected = redact_with_counts(text, level=level)
        true_counts = row.get("true_counts", {})
        doc_name = Path(row["doc"]).name
        for t in PII_TYPES:
            d = int(detected.get(t, 0))
            tr = int(true_counts.get(t, 0))
            tp[t] += min(d, tr)
            over = max(d - tr, 0)
            under = max(tr - d, 0)
            fp[t] += over
            fn[t] += under
            if over and len(fp_examples) < 20:
                fp_examples.append({"doc": doc_name, "type": t, "false_positives": over})
            if under and len(fn_examples) < 20:
                fn_examples.append({"doc": doc_name, "type": t, "missed": under})

    def prf(tp_n: int, fp_n: int, fn_n: int) -> dict[str, float]:
        precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) else 1.0
        recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    per_type = {t: {"tp": tp[t], "fp": fp[t], "fn": fn[t], **prf(tp[t], fp[t], fn[t])} for t in PII_TYPES}
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    return {
        "level": level,
        "micro": {"tp": total_tp, "fp": total_fp, "fn": total_fn, **prf(total_tp, total_fp, total_fn)},
        "macro": {
            "precision": round(sum(per_type[t]["precision"] for t in PII_TYPES) / len(PII_TYPES), 4),
            "recall": round(sum(per_type[t]["recall"] for t in PII_TYPES) / len(PII_TYPES), 4),
            "f1": round(sum(per_type[t]["f1"] for t in PII_TYPES) / len(PII_TYPES), 4),
        },
        "per_type": per_type,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    ensure_index(args.docs_dir, args.index_dir, args.rebuild_index)

    labels = load_jsonl(args.labels)
    truth = load_jsonl(args.truth)

    e2e_cases = run_e2e(args, labels)
    pr_by_level = {level: evaluate_level(level, truth) for level in DETECTOR_LEVELS}
    # PR curve operating points (micro), ordered conservative -> aggressive.
    pr_curve = [
        {
            "level": level,
            "precision": pr_by_level[level]["micro"]["precision"],
            "recall": pr_by_level[level]["micro"]["recall"],
            "f1": pr_by_level[level]["micro"]["f1"],
        }
        for level in DETECTOR_LEVELS
    ]

    report = {
        "index_dir": args.index_dir,
        "docs_dir": args.docs_dir,
        "llm_condition": args.llm,
        "end_to_end": {
            "num_cases": len(e2e_cases),
            "query_micro_recall": round(
                sum(c["query_masking"]["micro_recall"] for c in e2e_cases) / len(e2e_cases), 4
            ) if e2e_cases else 0.0,
            "context_no_leakage_rate": round(
                sum(1 for c in e2e_cases if c["context_no_leakage"]) / len(e2e_cases), 4
            ) if e2e_cases else 0.0,
            "cases": e2e_cases,
        },
        "detector_pr": {
            "num_docs": len(truth),
            "curve": pr_curve,
            "by_level": pr_by_level,
        },
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, report)
    print(json.dumps({"end_to_end_query_recall": report["end_to_end"]["query_micro_recall"],
                      "pr_curve": pr_curve}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
