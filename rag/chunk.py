from __future__ import annotations

from rag.schemas import Chunk, Document


def chunk_document(doc: Document, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    text = doc.text

    for i, start in enumerate(range(0, len(text), step)):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::chunk::{i}",
                doc_id=doc.doc_id,
                source_uri=doc.source_uri,
                text=chunk_text,
                start_char=start,
                end_char=end,
            )
        )
        if end >= len(text):
            break
    return chunks


def chunk_documents(documents: list[Document], chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    result: list[Chunk] = []
    for doc in documents:
        result.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return result

