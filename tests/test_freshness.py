"""Unit tests for freshness rerank weight behavior (rag.profiles.query)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.profiles.query import rerank_with_freshness
from rag.core.schemas import RetrievedChunk


class FreshnessRerankTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        # Two docs: "old" is more relevant (higher score), "new" is fresher.
        self.old = root / "old.txt"
        self.new = root / "new.txt"
        self.old.write_text("Topic doc.\nLast updated: 2020-01-01\n", encoding="utf-8")
        self.new.write_text("Topic doc.\nLast updated: 2026-01-01\n", encoding="utf-8")
        self.chunks = [
            RetrievedChunk(chunk_id="c_old", doc_id="d_old", source_uri=str(self.old), score=0.90, text="old"),
            RetrievedChunk(chunk_id="c_new", doc_id="d_new", source_uri=str(self.new), score=0.80, text="new"),
        ]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_weight_zero_keeps_relevance_order(self) -> None:
        ranked = rerank_with_freshness(self.chunks, weight=0.0)
        self.assertEqual(ranked[0].chunk_id, "c_old")

    def test_large_weight_promotes_fresher_doc(self) -> None:
        ranked = rerank_with_freshness(self.chunks, weight=1.0)
        self.assertEqual(ranked[0].chunk_id, "c_new")

    def test_empty_input(self) -> None:
        self.assertEqual(rerank_with_freshness([], weight=0.5), [])


if __name__ == "__main__":
    unittest.main()
