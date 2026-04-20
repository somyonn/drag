from __future__ import annotations

import hashlib
from pathlib import Path

from rag.schemas import Document


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_documents(docs_dir: str | Path) -> list[Document]:
    root = Path(docs_dir)
    if not root.exists():
        return []

    documents: list[Document] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        relative_path = file_path.relative_to(root).as_posix()
        doc_id = f"doc::{relative_path}"
        documents.append(
            Document(
                doc_id=doc_id,
                source_uri=str(file_path),
                text=text,
                metadata={
                    "fingerprint": fingerprint_text(text),
                    "relative_path": relative_path,
                    "char_count": len(text),
                },
            )
        )

    return documents

