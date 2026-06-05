"""
Detect remote updates for locally stored AWS-style docs (Source: URL header)
and refresh disk + optionally merge into the TF-IDF index incrementally.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import requests

from rag.corpus.aws_fetch import fetch_html_with_headers, format_doc_text, parse_html

MANIFEST_FILENAME = "external_source_manifest.json"

# Off for web UI and profile queries. Manual runs: scripts/sync_external_docs.py (allow_network=True).
EXTERNAL_HTTP_SYNC_ALLOWED = False

SYNC_ALLOWED_HOSTS = (
    "docs.aws.amazon.com",
    "docs.docker.com",
    "developers.google.com",
)

SOURCE_LINE_RE = re.compile(r"^Source:\s*(https?://\S+)\s*$", re.MULTILINE)


def is_external_http_sync_allowed() -> bool:
    return EXTERNAL_HTTP_SYNC_ALLOWED


def is_sync_allowed_url(url: str) -> bool:
    from urllib.parse import urlparse

    host = urlparse(url).netloc
    return any(host == allowed or host.endswith("." + allowed) for allowed in SYNC_ALLOWED_HOSTS)


def manifest_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / MANIFEST_FILENAME


def load_manifest(index_dir: str | Path) -> dict[str, Any]:
    p = manifest_path(index_dir)
    if not p.exists():
        return {"version": 1, "entries": {}}
    data = json.loads(p.read_text(encoding="utf-8"))
    if "entries" not in data:
        data["entries"] = {}
    return data


def save_manifest(index_dir: str | Path, data: dict[str, Any]) -> None:
    p = manifest_path(index_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_source_url_from_doc(file_path: Path, read_bytes: int = 12000) -> str | None:
    raw = file_path.read_text(encoding="utf-8")[:read_bytes]
    m = SOURCE_LINE_RE.search(raw)
    return m.group(1).strip() if m else None


def discover_tracked_files(docs_dir: Path) -> list[Path]:
    out: list[Path] = []
    for fp in sorted(docs_dir.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".txt", ".md"}:
            continue
        if fp.name.startswith("."):
            continue
        if fp.name == MANIFEST_FILENAME:
            continue
        out.append(fp)
    return out


@dataclass
class ExternalSyncResult:
    changed_rel_paths: list[str] = field(default_factory=list)
    checked: int = 0
    skipped_no_url: int = 0
    unchanged: int = 0
    updated_files: int = 0
    errors: list[str] = field(default_factory=list)
    disabled: bool = False


def _rel_key(docs_dir: Path, file_path: Path) -> str:
    return file_path.resolve().relative_to(docs_dir.resolve()).as_posix()


def sync_external_docs(
    docs_dir: str | Path,
    index_dir: str | Path,
    *,
    timeout_sec: float = 15.0,
    delay_sec: float = 0.0,
    dry_run: bool = False,
    force_refresh: bool = False,
    allow_network: bool = False,
) -> ExternalSyncResult:
    """
    HEAD/GET with validators against docs.aws.amazon.com Source URLs.
    Writes updated text files when remote body (normalized hash) changes.
    """
    if not allow_network and not is_external_http_sync_allowed():
        return ExternalSyncResult(disabled=True)

    docs_root = Path(docs_dir).resolve()
    idx_root = Path(index_dir).resolve()
    result = ExternalSyncResult()
    manifest = load_manifest(idx_root)
    entries: dict[str, Any] = manifest.setdefault("entries", {})

    session = requests.Session()
    session.headers.setdefault("User-Agent", "drag-rag-external-sync/0.1")

    for fp in discover_tracked_files(docs_root):
        rel = _rel_key(docs_root, fp)
        url = parse_source_url_from_doc(fp)
        if not url:
            result.skipped_no_url += 1
            continue
        if not is_sync_allowed_url(url):
            result.errors.append(f"{rel}: unsupported sync host ({url})")
            continue

        result.checked += 1
        entry = entries.get(rel) or {}

        req_headers: dict[str, str] = {}
        if not force_refresh:
            if entry.get("etag"):
                req_headers["If-None-Match"] = str(entry["etag"])
            if entry.get("last_modified"):
                req_headers["If-Modified-Since"] = str(entry["last_modified"])

        status, html, rh = fetch_html_with_headers(session, url, timeout_sec, req_headers or None)
        if status == 304:
            result.unchanged += 1
            continue
        if status != 200 or html is None:
            result.errors.append(f"{rel}: HTTP {status} for {url}")
            continue

        title, body, _hrefs = parse_html(html)
        if not body.strip():
            result.errors.append(f"{rel}: empty body from {url}")
            continue

        title = title or "AWS documentation"
        new_text = format_doc_text(title, url, body)
        new_hash = _sha256_text(new_text)
        if entry.get("body_sha256") == new_hash and not force_refresh:
            result.unchanged += 1
            continue

        if dry_run:
            result.changed_rel_paths.append(rel)
            result.updated_files += 1
            continue

        fp.write_text(new_text, encoding="utf-8")
        result.changed_rel_paths.append(rel)
        result.updated_files += 1

        etag = rh.get("etag")
        last_mod = rh.get("last-modified")
        entries[rel] = {
            "url": url,
            "etag": etag,
            "last_modified": last_mod,
            "body_sha256": new_hash,
        }
        save_manifest(idx_root, manifest)

        if delay_sec > 0:
            import time

            time.sleep(delay_sec)

    return result


def sync_external_docs_with_index(
    docs_dir: str | Path,
    index_dir: str | Path,
    *,
    timeout_sec: float = 15.0,
    delay_sec: float = 0.05,
    dry_run: bool = False,
    force_refresh: bool = False,
    allow_network: bool = False,
) -> dict[str, Any]:
    """
    Run sync_external_docs; if files changed and meta.json exists, incremental_reindex.
    Returns a summary dict suitable for query logs.
    """
    result = sync_external_docs(
        docs_dir,
        index_dir,
        timeout_sec=timeout_sec,
        delay_sec=delay_sec,
        dry_run=dry_run,
        force_refresh=force_refresh,
        allow_network=allow_network,
    )
    summary: dict[str, Any] = {
        "checked": result.checked,
        "skipped_no_source_url": result.skipped_no_url,
        "unchanged": result.unchanged,
        "updated_files": result.updated_files,
        "changed_rel_paths": result.changed_rel_paths,
        "errors": result.errors,
    }
    if result.disabled:
        summary["disabled"] = True
        summary["message"] = (
            "External HTTP sync is disabled in this process. "
            "Use: python scripts/sync_external_docs.py"
        )
        return summary
    meta_path = Path(index_dir) / "meta.json"
    if not dry_run and result.changed_rel_paths and meta_path.exists():
        from rag.corpus.incremental_index import incremental_reindex
        from rag.indexing.lock import index_read_lock

        with index_read_lock():
            summary["incremental_reindex"] = incremental_reindex(
                docs_dir, index_dir, result.changed_rel_paths
            )
    return summary
