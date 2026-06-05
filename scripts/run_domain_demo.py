#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from rag.profiles.query import DEFAULT_PROFILES_PATH, PROFILE_NAMES, load_profiles, run_profile_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG query using domain-specific policy profiles.")
    parser.add_argument("--profile", choices=list(PROFILE_NAMES), required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--profiles-path", default=str(DEFAULT_PROFILES_PATH))
    parser.add_argument("--log-path", default="runs/domain_logs.jsonl")
    parser.add_argument(
        "--sync-on-query",
        action="store_true",
        help="Run external doc sync before query (default: off; requires allow_network in scripts/sync_external_docs.py)",
    )
    parser.add_argument("--sync-timeout-sec", type=float, default=None)
    parser.add_argument("--sync-delay-sec", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    profiles = load_profiles(Path(args.profiles_path))
    result = run_profile_query(
        query=args.query,
        profile_name=args.profile,
        profiles=profiles,
        llm_name="auto",
        log_path=Path(args.log_path),
        skip_external_sync=not args.sync_on_query,
        sync_timeout_sec=args.sync_timeout_sec,
        sync_delay_sec=args.sync_delay_sec,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
