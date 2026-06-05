#!/usr/bin/env python3
"""Start the RAG web UI (FastAPI + static frontend)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description="Run DRAG RAG web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()
    uvicorn.run(
        "web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
