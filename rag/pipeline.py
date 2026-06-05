from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag.indexing.chunk import chunk_documents
from rag.indexing.embed import fit_tfidf_embeddings
from rag.llm.generate import LLMClient, MockLLMClient, build_prompt
from rag.indexing.index import build_and_save_index
from rag.indexing.cache import load_index_cached
from rag.indexing.ingest import load_documents
from rag.core.metrics import Timer
from rag.indexing.retrieve import retrieve_top_k
from rag.core.schemas import QueryLog, utc_now_iso
from rag.core.timing import build_latency_payload


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
        docs_dir=docs_dir,
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
    retrieval_k: int | None = None,
    log_path: str | Path = "runs/logs.jsonl",
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    llm = llm_client or MockLLMClient()
    rk = int(retrieval_k if retrieval_k is not None else top_k)
    if rk < top_k:
        rk = top_k

    total_timer = Timer()

    index_load_timer = Timer()
    loaded_index = load_index_cached(index_dir)
    index_load_ms = index_load_timer.elapsed_ms()

    vector_search_timer = Timer()
    retrieved = retrieve_top_k(loaded_index, query=query, top_k=rk)
    retrieved = retrieved[:top_k]
    vector_search_ms = vector_search_timer.elapsed_ms()

    prompt_build_timer = Timer()
    prompt = build_prompt(query, retrieved)
    prompt_build_ms = prompt_build_timer.elapsed_ms()
    prompt_chars = len(prompt)
    chunks_sent_to_llm = len(retrieved)

    answer, llm_request_ms = llm.generate_with_timing(prompt, retrieved)
    is_mock_answer = answer.startswith("[MOCK ANSWER]")

    total_latency = total_timer.elapsed_ms()
    latency_ms = build_latency_payload(
        total_ms=total_latency,
        index_load_ms=index_load_ms,
        vector_search_ms=vector_search_ms,
        prompt_build_ms=prompt_build_ms,
        llm_request_ms=llm_request_ms,
        chunks_sent_to_llm=chunks_sent_to_llm,
        prompt_chars=prompt_chars,
    )

    log = QueryLog(
        timestamp=utc_now_iso(),
        query=query,
        answer=answer,
        top_k=top_k,
        doc_ids=[x.doc_id for x in retrieved],
        chunk_ids=[x.chunk_id for x in retrieved],
        scores=[x.score for x in retrieved],
        source_uris=[x.source_uri for x in retrieved],
        retrieval_latency_ms=latency_ms["retrieval"],
        generation_latency_ms=latency_ms["generation"],
        total_latency_ms=latency_ms["total"],
    )
    log_dict = log.to_dict()
    log_dict["latency_ms"] = latency_ms
    log_dict["is_mock_answer"] = is_mock_answer
    _append_log(log_path, log_dict)

    return {
        "query": query,
        "answer": answer,
        "prompt": prompt,
        "is_mock_answer": is_mock_answer,
        "top_k": top_k,
        "retrieval_k": rk,
        "retrieved": [r.__dict__ for r in retrieved],
        "latency_ms": latency_ms,
    }

