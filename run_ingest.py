#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from rag.pipeline import ingest_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline RAG index")
    parser.add_argument("--docs-dir", default="data/docs")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    result = ingest_pipeline(
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

