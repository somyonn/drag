#!/usr/bin/env python3
"""Check AWS doc URLs (Source: line) for updates; refresh changed files; merge index incrementally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.corpus.external_sync import sync_external_docs, sync_external_docs_with_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", default="data/docs", help="Local corpus root")
    parser.add_argument("--index-dir", default="data/index/official", help="Index directory (manifest stored here)")
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument("--delay-sec", type=float, default=0.05, help="Delay after each successful write")
    parser.add_argument("--dry-run", action="store_true", help="No disk/index writes; report would-update paths")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore validators; GET and compare body hash")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Only update text files + manifest; do not run incremental TF-IDF merge",
    )
    args = parser.parse_args()

    idx = Path(args.index_dir)
    meta_path = idx / "meta.json"
    if not args.skip_index and not meta_path.exists():
        print(
            json.dumps(
                {
                    "warning": f"No index at {meta_path}; run: python run_ingest.py --docs-dir {args.docs_dir} --index-dir {args.index_dir}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.skip_index:
        result = sync_external_docs(
            args.docs_dir,
            args.index_dir,
            timeout_sec=args.timeout_sec,
            delay_sec=args.delay_sec,
            dry_run=args.dry_run,
            force_refresh=args.force_refresh,
            allow_network=True,
        )
        out = {
            "checked": result.checked,
            "skipped_no_source_url": result.skipped_no_url,
            "unchanged": result.unchanged,
            "updated_or_would_update": result.updated_files,
            "changed_rel_paths": result.changed_rel_paths,
            "errors": result.errors,
        }
    else:
        out = sync_external_docs_with_index(
            args.docs_dir,
            args.index_dir,
            timeout_sec=args.timeout_sec,
            delay_sec=args.delay_sec,
            dry_run=args.dry_run,
            force_refresh=args.force_refresh,
            allow_network=True,
        )
        out["updated_or_would_update"] = out.pop("updated_files")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
