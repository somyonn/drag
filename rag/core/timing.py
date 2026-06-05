from __future__ import annotations

from typing import Any


def round_ms(value: float) -> float:
    return round(float(value), 3)


def build_latency_payload(
    *,
    total_ms: float,
    index_load_ms: float = 0.0,
    index_reload_ms: float = 0.0,
    external_sync_ms: float = 0.0,
    vector_search_ms: float = 0.0,
    profile_postprocess_ms: float = 0.0,
    privacy_context_mask_ms: float = 0.0,
    privacy_query_mask_ms: float = 0.0,
    prompt_build_ms: float = 0.0,
    llm_request_ms: float = 0.0,
    answer_postprocess_ms: float = 0.0,
    privacy_answer_mask_ms: float = 0.0,
    chunks_sent_to_llm: int = 0,
    prompt_chars: int = 0,
) -> dict[str, Any]:
    retrieval_total_ms = (
        vector_search_ms
        + profile_postprocess_ms
        + privacy_context_mask_ms
        + privacy_query_mask_ms
    )
    return {
        "total": round_ms(total_ms),
        "index_load_ms": round_ms(index_load_ms),
        "index_reload_ms": round_ms(index_reload_ms),
        "external_sync_ms": round_ms(external_sync_ms),
        "retrieval": round_ms(retrieval_total_ms),
        "generation": round_ms(llm_request_ms),
        "retrieval_ms": {
            "total": round_ms(retrieval_total_ms),
            "vector_search_ms": round_ms(vector_search_ms),
            "profile_postprocess_ms": round_ms(profile_postprocess_ms),
            "privacy_context_mask_ms": round_ms(privacy_context_mask_ms),
            "privacy_query_mask_ms": round_ms(privacy_query_mask_ms),
        },
        "generation_ms": {
            "llm_request_ms": round_ms(llm_request_ms),
            "prompt_build_ms": round_ms(prompt_build_ms),
            "answer_postprocess_ms": round_ms(answer_postprocess_ms),
            "privacy_answer_mask_ms": round_ms(privacy_answer_mask_ms),
            "chunks_sent_to_llm": chunks_sent_to_llm,
            "prompt_chars": prompt_chars,
        },
    }
