"""Shared AWS documentation HTML fetch + text extraction for crawlers and sync jobs."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="", query="").geturl()


def build_output_name(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.strip("/").replace("/", "__")
    if not slug:
        slug = "index"
    if not slug.endswith(".html"):
        slug = f"{slug}.html"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}.txt"


def fetch_html(session: requests.Session, url: str, timeout_sec: float) -> str | None:
    try:
        res = session.get(url, timeout=timeout_sec)
    except requests.RequestException:
        return None
    if res.status_code == 200:
        content_type = (res.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            return None
        res.encoding = res.encoding or "utf-8"
        return res.text
    return None


def fetch_html_with_headers(
    session: requests.Session,
    url: str,
    timeout_sec: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, str | None, dict[str, str]]:
    """Returns (status_code, html_or_none, response_headers_lower_keys)."""
    try:
        res = session.get(url, timeout=timeout_sec, headers=headers or {})
    except requests.RequestException:
        return 0, None, {}
    hdrs = {k.lower(): v for k, v in res.headers.items()}
    if res.status_code == 304:
        return 304, None, hdrs
    if res.status_code != 200:
        return res.status_code, None, hdrs
    content_type = (hdrs.get("content-type") or "").lower()
    if "text/html" not in content_type:
        return res.status_code, None, hdrs
    res.encoding = res.encoding or "utf-8"
    return res.status_code, res.text, hdrs


def parse_html(html: str) -> tuple[str, str, set[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "AWS documentation"

    content_root = (
        soup.find("main")
        or soup.find(id="main-col-body")
        or soup.find("article")
        or soup.body
        or soup
    )

    for tag in content_root.find_all(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()

    text = content_root.get_text(separator="\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    hrefs = {a["href"] for a in content_root.find_all("a", href=True)}
    return title, text, hrefs


def format_doc_text(title: str, url: str, body: str) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            f"Source: {url}",
            "",
            body,
            "",
        ]
    )
