"""Crawl official documentation sites into plain-text files for RAG ingestion."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from rag.corpus.aws_fetch import build_output_name, fetch_html, format_doc_text, normalize_url, parse_html
from rag.corpus.doc_sources import DocSourceSpec


def is_allowed_url(url: str, spec: DocSourceSpec) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc not in spec.allowed_netlocs:
        return False
    if not spec.allowed_path_prefixes:
        return True
    return any(parsed.path.startswith(prefix) for prefix in spec.allowed_path_prefixes)


def crawl_source(
    spec: DocSourceSpec,
    output_root: Path,
    max_pages: int,
    delay_sec: float,
    timeout_sec: float,
    user_agent: str = "drag-rag-official-docs/0.2",
) -> dict[str, object]:
    start_urls = [normalize_url(u) for u in spec.start_urls]
    queue: deque[str] = deque(start_urls)
    visited: set[str] = set()
    written_files: list[str] = []

    out_dir = output_root / spec.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = user_agent

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        if not is_allowed_url(url, spec):
            continue
        visited.add(url)

        html = fetch_html(session, url, timeout_sec=timeout_sec)
        if html is None:
            continue

        title, body, hrefs = parse_html(html)
        if body:
            title = title or spec.default_title
            out_path = out_dir / build_output_name(url)
            out_path.write_text(format_doc_text(title, url, body), encoding="utf-8")
            written_files.append(str(out_path))

        for href in hrefs:
            next_url = normalize_url(urljoin(url, href))
            if next_url not in visited and is_allowed_url(next_url, spec):
                queue.append(next_url)

        if delay_sec > 0:
            time.sleep(delay_sec)

    return {
        "source_id": spec.source_id,
        "output_dir": str(out_dir),
        "visited_pages": len(visited),
        "saved_docs": len(written_files),
    }
