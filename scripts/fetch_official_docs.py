#!/usr/bin/env python3
"""Fetch official docs (AWS services, Docker, Google Drive/Workspace) into data/docs/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.corpus.doc_crawl import crawl_source
from rag.corpus.doc_sources import (
    ALL_SOURCE_IDS,
    AWS_SERVICE_SOURCES,
    OTHER_SOURCES,
    DocSourceSpec,
)


def resolve_sources(
    source_names: list[str],
    aws_services: list[str] | None,
) -> list[DocSourceSpec]:
    specs: list[DocSourceSpec] = []
    names = list(source_names)

    if "all" in names:
        for key in sorted(AWS_SERVICE_SOURCES.keys()):
            specs.append(AWS_SERVICE_SOURCES[key])
        for key in sorted(OTHER_SOURCES.keys()):
            specs.append(OTHER_SOURCES[key])
        return specs

    if "aws" in names:
        keys = aws_services if aws_services else sorted(AWS_SERVICE_SOURCES.keys())
        for key in keys:
            if key not in AWS_SERVICE_SOURCES:
                raise ValueError(f"Unknown AWS service: {key}. Choose from: {sorted(AWS_SERVICE_SOURCES)}")
            specs.append(AWS_SERVICE_SOURCES[key])
        names = [n for n in names if n != "aws"]

    for key in names:
        if key in OTHER_SOURCES:
            specs.append(OTHER_SOURCES[key])
        elif key in AWS_SERVICE_SOURCES:
            specs.append(AWS_SERVICE_SOURCES[key])
        else:
            raise ValueError(f"Unknown source: {key}. Use --sources all or one of: {', '.join(ALL_SOURCE_IDS[:8])}…")

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["all"],
        help=f"Sources: all, aws, docker, google_drive, or AWS keys ({', '.join(sorted(AWS_SERVICE_SOURCES)[:5])}…)",
    )
    parser.add_argument(
        "--aws-services",
        nargs="+",
        default=None,
        help="When using 'aws', limit to these services (default: all AWS services in catalog)",
    )
    parser.add_argument("--output-dir", default="data/docs", help="Root output directory")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=120,
        help="Max pages to visit per source (each AWS service counts separately)",
    )
    parser.add_argument("--delay-sec", type=float, default=0.08)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    specs = resolve_sources(args.sources, args.aws_services)
    if not specs:
        raise SystemExit("No sources selected.")

    results = []
    for spec in specs:
        print(f"Crawling {spec.source_id} (max {args.max_pages} pages)…", flush=True)
        results.append(
            crawl_source(
                spec,
                output_root=output_root,
                max_pages=args.max_pages,
                delay_sec=args.delay_sec,
                timeout_sec=args.timeout_sec,
            )
        )

    total_saved = sum(int(r["saved_docs"]) for r in results)
    print(json.dumps({"results": results, "total_saved_docs": total_saved}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
