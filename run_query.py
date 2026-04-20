#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv
from rag.generate import MockLLMClient, OpenAIChatClient
from rag.pipeline import query_pipeline


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run a baseline RAG query")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--log-path", default="runs/logs.jsonl")
    parser.add_argument("--llm", choices=["mock", "openai"], default="mock")
    args = parser.parse_args()

    llm_client = MockLLMClient() if args.llm == "mock" else OpenAIChatClient()
    result = query_pipeline(
        query=args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
        log_path=args.log_path,
        llm_client=llm_client,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

