import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_kb_eval import (
    count_relevant_in_corpus,
    ranking_metrics,
    uri_matches_groups,
)


class TestUriMatchesGroups(unittest.TestCase):
    def test_and_within_group(self):
        self.assertTrue(uri_matches_groups("aws/iam/ec2-roles.txt", [["iam", "ec2"]]))
        self.assertFalse(uri_matches_groups("aws/iam/roles.txt", [["iam", "ec2"]]))

    def test_or_across_groups(self):
        groups = [["multi-stage-builds"], ["multi_stage"]]
        self.assertTrue(uri_matches_groups("docker/multi-stage-builds.txt", groups))
        self.assertTrue(uri_matches_groups("docker/multi_stage.txt", groups))
        self.assertFalse(uri_matches_groups("docker/compose.txt", groups))


class TestCountRelevantInCorpus(unittest.TestCase):
    def test_unique_doc_count(self):
        chunks = [
            {"source_uri": "/d/AWS/EC2/AllocateHosts.txt"},
            {"source_uri": "/d/AWS/EC2/AllocateHosts.txt"},  # duplicate doc
            {"source_uri": "/d/AWS/S3/Bucket.txt"},
        ]
        self.assertEqual(count_relevant_in_corpus(chunks, [["allocatehosts"]]), 1)
        self.assertEqual(count_relevant_in_corpus(chunks, [["ec2"], ["s3"]]), 2)


class TestRankingMetrics(unittest.TestCase):
    def test_no_groups_returns_none(self):
        m = ranking_metrics(["a", "b"], [], 0)
        self.assertIsNone(m["mrr"])
        self.assertIsNone(m["ndcg_at_k"])
        self.assertIsNone(m["recall_at_k"])

    def test_perfect_first_rank(self):
        uris = ["docs/gold.txt", "docs/other.txt"]
        m = ranking_metrics(uris, [["gold"]], total_relevant=1)
        self.assertEqual(m["mrr"], 1.0)
        self.assertEqual(m["ndcg_at_k"], 1.0)
        self.assertEqual(m["recall_at_k"], 1.0)

    def test_relevant_at_second_rank(self):
        uris = ["docs/other.txt", "docs/gold.txt"]
        m = ranking_metrics(uris, [["gold"]], total_relevant=1)
        self.assertEqual(m["mrr"], 0.5)
        # nDCG = (1/log2(3)) / 1 ~= 0.6309
        self.assertAlmostEqual(m["ndcg_at_k"], 0.6309, places=3)
        self.assertEqual(m["recall_at_k"], 1.0)

    def test_no_relevant_retrieved(self):
        uris = ["docs/a.txt", "docs/b.txt"]
        m = ranking_metrics(uris, [["gold"]], total_relevant=2)
        self.assertEqual(m["mrr"], 0.0)
        self.assertEqual(m["ndcg_at_k"], 0.0)
        self.assertEqual(m["recall_at_k"], 0.0)

    def test_partial_recall_with_dedupe(self):
        # gold universe of 4; retrieve 2 distinct relevant (one duplicated)
        uris = ["g/one.txt", "g/one.txt", "g/two.txt", "x/none.txt"]
        m = ranking_metrics(uris, [["g/"]], total_relevant=4)
        self.assertEqual(m["recall_at_k"], 0.5)
        self.assertEqual(m["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
