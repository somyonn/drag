"""Integration test: ingest a tiny corpus then query it with the Mock LLM."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.llm.generate import MockLLMClient
from rag.pipeline import ingest_pipeline, query_pipeline

DOCS = {
    "compose.txt": (
        "Docker Compose defines and runs multi-container applications. "
        "You describe services in a compose.yaml file and start them with docker compose up."
    ),
    "iam.txt": (
        "An IAM role grants temporary permissions to an EC2 instance through an instance profile. "
        "Attach the role so the workload can call AWS APIs without long-lived keys."
    ),
    "drive.txt": (
        "The Google Drive API files.list method searches files and folders using a query string. "
        "Use the q parameter to filter by name, mimeType, or parents."
    ),
}


class PipelineIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.docs_dir = root / "docs"
        self.index_dir = root / "index"
        self.docs_dir.mkdir(parents=True)
        for name, text in DOCS.items():
            (self.docs_dir / name).write_text(text, encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_ingest_then_query(self) -> None:
        ingest = ingest_pipeline(docs_dir=self.docs_dir, index_dir=self.index_dir)
        self.assertEqual(ingest["num_docs"], len(DOCS))
        self.assertGreater(ingest["num_chunks"], 0)

        result = query_pipeline(
            query="How do I run a multi-container app with Docker Compose?",
            index_dir=self.index_dir,
            top_k=2,
            log_path=Path(self._tmp.name) / "logs.jsonl",
            llm_client=MockLLMClient(),
        )

        self.assertTrue(result["is_mock_answer"])
        self.assertTrue(result["answer"].startswith("[MOCK ANSWER]"))
        self.assertEqual(len(result["retrieved"]), 2)
        # Compose query should surface the compose document as the top hit.
        self.assertIn("compose.txt", result["retrieved"][0]["source_uri"])

        latency = result["latency_ms"]
        for key in ("total", "retrieval", "generation"):
            self.assertIn(key, latency)
            self.assertGreaterEqual(latency[key], 0.0)


if __name__ == "__main__":
    unittest.main()
