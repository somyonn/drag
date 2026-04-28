#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


SERVICE_CONFIG: dict[str, dict[str, object]] = {
    "ec2": {
        "start_urls": ["https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html"],
        "allowed_prefixes": ["/AWSEC2/", "/ec2/"],
    },
    "s3": {
        "start_urls": ["https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html"],
        "allowed_prefixes": ["/AmazonS3/"],
    },
    "iam": {
        "start_urls": ["https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html"],
        "allowed_prefixes": ["/IAM/"],
    },
}


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="", query="").geturl()


def is_allowed_aws_doc(url: str, allowed_prefixes: list[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc != "docs.aws.amazon.com":
        return False
    return any(parsed.path.startswith(prefix) for prefix in allowed_prefixes)


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
    if res.status_code != 200:
        return None
    content_type = (res.headers.get("Content-Type") or "").lower()
    if "text/html" not in content_type:
        return None
    res.encoding = res.encoding or "utf-8"
    return res.text


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


def crawl_service(
    service: str,
    output_root: Path,
    max_pages: int,
    delay_sec: float,
    timeout_sec: float,
) -> dict[str, object]:
    config = SERVICE_CONFIG[service]
    start_urls = [normalize_url(u) for u in config["start_urls"]]  # type: ignore[index]
    allowed_prefixes = list(config["allowed_prefixes"])  # type: ignore[index]

    queue: deque[str] = deque(start_urls)
    visited: set[str] = set()
    written_files: list[str] = []
    service_out = output_root / service
    service_out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "drag-rag-aws-docs-loader/0.1"

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        if not is_allowed_aws_doc(url, allowed_prefixes):
            continue
        visited.add(url)

        html = fetch_html(session, url, timeout_sec=timeout_sec)
        if html is None:
            continue

        title, body, hrefs = parse_html(html)
        if body:
            title = title or f"AWS {service.upper()} document"
            out_name = build_output_name(url)
            out_path = service_out / out_name
            out_path.write_text(
                "\n".join(
                    [
                        f"# {title}",
                        "",
                        f"Source: {url}",
                        "",
                        body,
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            written_files.append(str(out_path))

        for href in hrefs:
            next_url = normalize_url(urljoin(url, href))
            if next_url not in visited and is_allowed_aws_doc(next_url, allowed_prefixes):
                queue.append(next_url)

        if delay_sec > 0:
            time.sleep(delay_sec)

    return {
        "service": service,
        "visited_pages": len(visited),
        "saved_docs": len(written_files),
        "output_dir": str(service_out),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl AWS docs for EC2, S3, IAM and store text files for RAG ingestion."
    )
    parser.add_argument(
        "--services",
        nargs="+",
        default=["ec2", "s3", "iam"],
        choices=sorted(SERVICE_CONFIG.keys()),
        help="AWS services to crawl",
    )
    parser.add_argument("--output-dir", default="data/docs/aws", help="Output directory for fetched docs")
    parser.add_argument("--max-pages-per-service", type=int, default=30, help="Max pages to crawl per service")
    parser.add_argument("--delay-sec", type=float, default=0.1, help="Delay between requests per service")
    parser.add_argument("--timeout-sec", type=float, default=10.0, help="HTTP request timeout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for service in args.services:
        results.append(
            crawl_service(
                service=service,
                output_root=output_root,
                max_pages=args.max_pages_per_service,
                delay_sec=args.delay_sec,
                timeout_sec=args.timeout_sec,
            )
        )

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
