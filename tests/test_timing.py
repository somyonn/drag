"""Unit tests for latency payload composition (rag.core.timing.build_latency_payload)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.core.timing import build_latency_payload


class BuildLatencyPayloadTest(unittest.TestCase):
    def test_retrieval_total_is_sum_of_components(self) -> None:
        payload = build_latency_payload(
            total_ms=100.0,
            vector_search_ms=7.0,
            profile_postprocess_ms=2.0,
            privacy_context_mask_ms=1.0,
            llm_request_ms=80.0,
        )
        self.assertAlmostEqual(payload["retrieval"], 10.0, places=3)
        self.assertEqual(payload["retrieval_ms"]["total"], payload["retrieval"])

    def test_generation_maps_to_llm_request(self) -> None:
        payload = build_latency_payload(total_ms=50.0, llm_request_ms=42.5)
        self.assertEqual(payload["generation"], 42.5)
        self.assertEqual(payload["generation_ms"]["llm_request_ms"], 42.5)

    def test_values_are_rounded_to_three_decimals(self) -> None:
        payload = build_latency_payload(total_ms=1.23456, vector_search_ms=0.00049)
        self.assertEqual(payload["total"], 1.235)
        self.assertEqual(payload["retrieval_ms"]["vector_search_ms"], 0.0)

    def test_prompt_metadata_is_passed_through(self) -> None:
        payload = build_latency_payload(total_ms=1.0, chunks_sent_to_llm=3, prompt_chars=2048)
        self.assertEqual(payload["generation_ms"]["chunks_sent_to_llm"], 3)
        self.assertEqual(payload["generation_ms"]["prompt_chars"], 2048)


if __name__ == "__main__":
    unittest.main()
