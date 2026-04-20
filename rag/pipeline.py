from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag.chunk import chunk_documents
from rag.embed import fit_tfidf_embeddings
from rag.generate import LLMClient, MockLLMClient, build_prompt
from rag.index import build_and_save_index, load_index
from rag.ingest import load_documents
from rag.metrics import Timer
from rag.retrieve import retrieve_top_k
from rag.schemas import QueryLog, utc_now_iso


def _combined_doc_fingerprint(doc_fingerprints: list[str]) -> str:
    joined = "|".join(sorted(doc_fingerprints))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def ingest_pipeline(
    docs_dir: str | Path = "data/docs",
    index_dir: str | Path = "data/index",
    chunk_size: int = 500,
    overlap: int = 100,
) -> dict[str, Any]:
    docs = load_documents(docs_dir)
    if not docs:
        raise ValueError(f"No documents found in {docs_dir}")

    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No chunks were produced from documents")

    embeddings = fit_tfidf_embeddings([c.text for c in chunks])
    fingerprint = _combined_doc_fingerprint([d.metadata["fingerprint"] for d in docs])

    meta = build_and_save_index(
        embeddings=embeddings.matrix,
        chunks=chunks,
        vectorizer=embeddings.vectorizer,
        index_dir=index_dir,
        embedding_model_id=embeddings.model_id,
        chunk_params={"chunk_size": chunk_size, "overlap": overlap},
        doc_fingerprint=fingerprint,
    )
    return {"num_docs": len(docs), "num_chunks": len(chunks), "meta": meta}


def _append_log(log_path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def query_pipeline(
    query: str,
    index_dir: str | Path = "data/index",
    top_k: int = 3,
    log_path: str | Path = "runs/logs.jsonl",
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    llm = llm_client or MockLLMClient()

    total_timer = Timer()
    retrieval_timer = Timer()
    loaded_index = load_index(index_dir)
    retrieved = retrieve_top_k(loaded_index, query=query, top_k=top_k)
    retrieval_latency = retrieval_timer.elapsed_ms()

    generation_timer = Timer()
    prompt = build_prompt(query, retrieved)
    answer = llm.generate(prompt, retrieved)
    generation_latency = generation_timer.elapsed_ms()
    total_latency = total_timer.elapsed_ms()

    log = QueryLog(
        timestamp=utc_now_iso(),
        query=query,
        answer=answer,
        top_k=top_k,
        doc_ids=[x.doc_id for x in retrieved],
        chunk_ids=[x.chunk_id for x in retrieved],
        scores=[x.score for x in retrieved],
        source_uris=[x.source_uri for x in retrieved],
        retrieval_latency_ms=retrieval_latency,
        generation_latency_ms=generation_latency,
        total_latency_ms=total_latency,
    )
    _append_log(log_path, log.to_dict())

    return {
        "query": query,
        "answer": answer,
        "prompt": prompt,
        "retrieved": [r.__dict__ for r in retrieved],
        "latency_ms": {
            "retrieval": retrieval_latency,
            "generation": generation_latency,
            "total": total_latency,
        },
    }

