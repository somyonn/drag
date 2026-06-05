#!/usr/bin/env python3
"""LLM-as-judge answer-accuracy eval (RAGAS-style, no gold answers required).

For each (mode, query) the generator LLM produces an answer from retrieved
context; a separate, stronger judge LLM scores the answer on three axes:

  - answer_relevance (1-5): does the answer address the question?
  - faithfulness     (1-5): is every claim grounded in the provided context?
  - correctness      (1-5): is the answer factually correct?

Using a different judge model than the generator (e.g. judge=gpt-4.1,
generator=gpt-4.1-mini) reduces self-evaluation bias. temperature=0 on both
sides keeps the scoring reproducible. Answers/contexts are generated with the
real cloud LLM only (no mock).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests
from dotenv import load_dotenv

from scripts._eval_common import (
    install_thread_safe_logging,
    load_queries,
    write_report,
)
from rag.indexing.cache import warm_index_cache
from rag.pipeline import query_pipeline
from rag.profiles.query import (
    DEFAULT_PROFILES_PATH,
    PROFILE_NAMES,
    load_profiles,
    pick_llm,
    run_profile_query,
)

JUDGE_AXES = ("answer_relevance", "faithfulness", "correctness")
JUDGE_PASS_THRESHOLD = 4  # score >= 4 counts as a "pass" for binary rates.

SYSTEM_PROMPT = (
    "You are a strict evaluation judge for a retrieval-augmented QA system. "
    "You grade an answer ONLY against the provided context and the question. "
    "Respond with a single JSON object and nothing else."
)

JUDGE_TEMPLATE = """Evaluate the assistant answer on three axes, each an integer from 1 (worst) to 5 (best).

Definitions:
- answer_relevance: Does the answer directly address the question that was asked?
- faithfulness: Is every factual claim in the answer supported by the CONTEXT below? Penalize claims not present in the context (hallucination). If the answer correctly says the context is insufficient, that is faithful.
- correctness: Is the answer factually correct for the question (using the context as evidence)?

Return strict JSON with exactly these keys:
{{"answer_relevance": int, "faithfulness": int, "correctness": int, "rationale": "one short sentence"}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


def build_context_block(retrieved: list[dict], max_chunks: int, max_chars: int) -> str:
    blocks: list[str] = []
    for i, item in enumerate(retrieved[:max_chunks], start=1):
        text = (item.get("text") or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        blocks.append(f"[Context {i}] source={item.get('source_uri', '')}\n{text}")
    return "\n\n".join(blocks) if blocks else "No context."


def generate_answer(mode: str, query: str, args: argparse.Namespace, profiles: dict) -> dict:
    if mode == "baseline":
        return query_pipeline(
            query=query,
            index_dir=args.index_dir,
            log_path=args.log,
            llm_client=pick_llm(args.llm),
        )
    if mode in PROFILE_NAMES:
        return run_profile_query(
            query=query,
            profile_name=mode,
            profiles=profiles,
            llm_name=args.llm,
            log_path=args.log,
            skip_external_sync=True,
        )
    raise ValueError(f"Unknown mode: {mode}")


def extract_json(text: str) -> dict | None:
    """Parse a JSON object out of the judge response, tolerating stray prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def coerce_scores(obj: dict) -> dict | None:
    """Validate and clamp judge scores to integers in [1, 5]."""
    scores: dict[str, object] = {}
    for axis in JUDGE_AXES:
        if axis not in obj:
            return None
        try:
            val = int(round(float(obj[axis])))
        except (TypeError, ValueError):
            return None
        scores[axis] = max(1, min(5, val))
    scores["rationale"] = str(obj.get("rationale", ""))[:300]
    return scores


def call_judge(question: str, answer: str, context: str, args: argparse.Namespace) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user_prompt = JUDGE_TEMPLATE.format(question=question, context=context, answer=answer)
    payload = {
        "model": args.judge_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_err = ""
    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=args.judge_timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            parsed = extract_json(content)
            scores = coerce_scores(parsed) if parsed is not None else None
            if scores is not None:
                return scores
            last_err = f"unparseable judge output: {content[:120]}"
        except requests.RequestException as exc:
            last_err = f"request error: {exc}"
        time.sleep(0.5 * (attempt + 1))
    return {"judge_error": last_err}


def evaluate_one(mode: str, query: str, args: argparse.Namespace, profiles: dict) -> dict:
    result = generate_answer(mode, query, args, profiles)
    answer = result.get("answer", "") or ""
    retrieved = result.get("retrieved") or []
    context = build_context_block(retrieved, args.max_context_chunks, args.max_context_chars)
    is_mock = bool(result.get("is_mock_answer"))

    judged = call_judge(query, answer, context, args)
    row = {
        "mode": mode,
        "query": query,
        "is_mock_answer": is_mock,
        "answer": answer,
        "num_context_chunks": len(retrieved),
        "judge": judged,
    }
    return row


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if "judge_error" not in r["judge"]]
    errors = len(rows) - len(scored)
    summary: dict[str, object] = {
        "num_queries": len(rows),
        "num_judged": len(scored),
        "num_judge_errors": errors,
        "any_mock_answer": any(r["is_mock_answer"] for r in rows),
    }
    for axis in JUDGE_AXES:
        vals = [r["judge"][axis] for r in scored]
        summary[f"{axis}_mean"] = round(mean(vals), 3) if vals else None
        pass_rate = (sum(1 for v in vals if v >= JUDGE_PASS_THRESHOLD) / len(vals)) if vals else None
        summary[f"{axis}_pass_rate"] = round(pass_rate, 3) if pass_rate is not None else None
    return summary


def lowest_examples(rows: list[dict], n: int = 3) -> list[dict]:
    scored = [r for r in rows if "judge_error" not in r["judge"]]

    def composite(r: dict) -> float:
        return mean(r["judge"][a] for a in JUDGE_AXES)

    worst = sorted(scored, key=composite)[:n]
    return [
        {
            "mode": r["mode"],
            "query": r["query"],
            "scores": {a: r["judge"][a] for a in JUDGE_AXES},
            "rationale": r["judge"].get("rationale", ""),
            "answer_preview": (r["answer"][:200] + " ...") if len(r["answer"]) > 200 else r["answer"],
        }
        for r in worst
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-judge answer-accuracy eval")
    parser.add_argument("--queries", default="data/queries/kb_eval.txt")
    parser.add_argument("--index-dir", default="data/index/official")
    parser.add_argument("--profiles-path", default=str(DEFAULT_PROFILES_PATH))
    parser.add_argument("--modes", default="baseline,low_latency,privacy,freshness_accuracy")
    parser.add_argument(
        "--llm",
        choices=("cloud", "auto", "mock"),
        default="cloud",
        help="Generator LLM condition. Default cloud (real LLM, no mock).",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1"),
        help="Judge model (stronger than the generator to reduce self-judge bias).",
    )
    parser.add_argument("--judge-timeout", type=int, default=40)
    parser.add_argument("--max-context-chunks", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=1200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--report-path", default="runs/judge_eval_report.json")
    parser.add_argument("--log", default="runs/judge_eval_gen.jsonl")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    queries = load_queries(Path(args.queries))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    profiles = load_profiles(Path(args.profiles_path))
    warm_index_cache(args.index_dir)

    generator_model = "mock" if args.llm == "mock" else os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    tasks = [(idx, mode, query) for idx, (mode, query) in enumerate(
        (m, q) for m in modes for q in queries
    )]

    def run_task(task: tuple[int, str, str]) -> tuple[int, dict]:
        order, mode, query = task
        return order, evaluate_one(mode, query, args, profiles)

    started = time.perf_counter()
    concurrency = max(1, int(args.concurrency))
    if concurrency == 1:
        results = [run_task(t) for t in tasks]
    else:
        install_thread_safe_logging()  # generation appends to runs/*.jsonl concurrently
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(run_task, tasks))
    wall = round(time.perf_counter() - started, 3)

    by_mode: dict[str, list[dict]] = {m: [] for m in modes}
    for _order, row in sorted(results, key=lambda x: x[0]):
        by_mode[row["mode"]].append(row)
    all_rows = [r for m in modes for r in by_mode[m]]

    report = {
        "queries_file": args.queries,
        "index_dir": args.index_dir,
        "modes": modes,
        "llm_condition": args.llm,
        "generator_model": generator_model,
        "judge_model": args.judge_model,
        "pass_threshold": JUDGE_PASS_THRESHOLD,
        "wall_clock_sec": wall,
        "per_mode": {m: summarize(rows) for m, rows in by_mode.items()},
        "overall": summarize(all_rows),
        "lowest_examples": lowest_examples(all_rows, n=5),
        "runs": all_rows,
    }

    report_path = write_report(args.report_path, report)

    summary_keys = (
        "queries_file", "modes", "llm_condition", "generator_model", "judge_model",
        "pass_threshold", "wall_clock_sec", "per_mode", "overall",
    )
    print(json.dumps({k: report[k] for k in summary_keys}, ensure_ascii=False, indent=2))
    print(f"\nFull per-query judgments: {report_path}")


if __name__ == "__main__":
    main()
