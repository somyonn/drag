#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from rag.generate import CloudThenMockLLMClient, MockLLMClient, OpenAIChatClient
from rag.metrics import summarize_latencies_ms
from rag.pipeline import query_pipeline


def load_queries(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Query file not found: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Batch evaluation for baseline RAG")
    parser.add_argument("--queries", required=True, help="Text file with one query per line")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--log-path", default="runs/logs.jsonl")
    parser.add_argument("--llm", choices=["auto", "mock", "openai"], default="auto")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    latencies: list[float] = []
    if args.llm == "mock":
        llm_client = MockLLMClient()
    elif args.llm == "openai":
        llm_client = OpenAIChatClient()
    else:
        llm_client = CloudThenMockLLMClient()

    for query in queries:
        result = query_pipeline(
            query=query,
            index_dir=args.index_dir,
            top_k=args.top_k,
            log_path=args.log_path,
            llm_client=llm_client,
        )
        latencies.append(result["latency_ms"]["total"])

    summary = {
        "num_queries": len(queries),
        "total_latency_ms": summarize_latencies_ms(latencies),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

