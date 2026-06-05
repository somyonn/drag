"""Unit tests for latency summary statistics (rag.core.metrics)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.core.metrics import percentile, summarize_latencies_ms


class SummarizeLatenciesTest(unittest.TestCase):
    def test_empty_returns_zeroed_keys(self) -> None:
        summary = summarize_latencies_ms([])
        for key in ("mean_ms", "std_ms", "min_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"):
            self.assertIn(key, summary)
            self.assertEqual(summary[key], 0.0)

    def test_percentiles_are_monotonic(self) -> None:
        values = [float(i) for i in range(1, 101)]
        s = summarize_latencies_ms(values)
        self.assertLessEqual(s["p50_ms"], s["p90_ms"])
        self.assertLessEqual(s["p90_ms"], s["p95_ms"])
        self.assertLessEqual(s["p95_ms"], s["p99_ms"])
        self.assertLessEqual(s["p99_ms"], s["max_ms"])
        self.assertEqual(s["min_ms"], 1.0)
        self.assertEqual(s["max_ms"], 100.0)

    def test_mean_and_std(self) -> None:
        s = summarize_latencies_ms([10.0, 20.0, 30.0])
        self.assertAlmostEqual(s["mean_ms"], 20.0, places=4)
        self.assertGreater(s["std_ms"], 0.0)

    def test_single_sample_has_zero_std(self) -> None:
        s = summarize_latencies_ms([42.0])
        self.assertEqual(s["std_ms"], 0.0)
        self.assertEqual(s["mean_ms"], 42.0)
        self.assertEqual(s["p99_ms"], 42.0)

    def test_percentile_bounds(self) -> None:
        self.assertEqual(percentile([], 50), 0.0)
        self.assertEqual(percentile([5.0], 50), 5.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0), 1.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 100), 3.0)


if __name__ == "__main__":
    unittest.main()
