"""Unit tests for privacy profile PII masking (rag.profiles.query.redact_text)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.profiles.query import DETECTOR_LEVELS, redact_text, redact_with_counts


class RedactTextTest(unittest.TestCase):
    def test_email_is_redacted(self) -> None:
        out = redact_text("Contact me at user@example.com please")
        self.assertNotIn("user@example.com", out)
        self.assertIn("[REDACTED_EMAIL]", out)

    def test_aws_access_key_is_redacted(self) -> None:
        out = redact_text("key AKIAIOSFODNN7EXAMPLE leaked")
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", out)
        self.assertIn("[REDACTED_AWS_KEY]", out)

    def test_phone_number_is_redacted(self) -> None:
        out = redact_text("call 010-1234-5678 now")
        self.assertIn("[REDACTED_PHONE]", out)

    def test_rrn_is_redacted(self) -> None:
        out = redact_text("RRN 900101-1234567 on file")
        self.assertNotIn("900101-1234567", out)
        self.assertIn("[REDACTED_RRN]", out)

    def test_credit_card_is_redacted(self) -> None:
        out = redact_text("card 4111-1111-1111-1111 charged")
        self.assertNotIn("4111-1111-1111-1111", out)
        self.assertIn("[REDACTED_CREDIT_CARD]", out)

    def test_ip_is_redacted(self) -> None:
        out = redact_text("from 192.168.0.15 inbound")
        self.assertNotIn("192.168.0.15", out)
        self.assertIn("[REDACTED_IP]", out)

    def test_plain_text_is_untouched(self) -> None:
        text = "This document explains IAM roles for EC2 workloads."
        self.assertEqual(redact_text(text), text)

    def test_counts_by_type(self) -> None:
        text = (
            "mail a@b.com, c@d.org phone 010-1234-5678 rrn 900101-1234567 "
            "card 4111-1111-1111-1111 ip 10.0.0.1 key AKIAIOSFODNN7EXAMPLE"
        )
        _, counts = redact_with_counts(text)
        self.assertEqual(counts["email"], 2)
        self.assertEqual(counts["phone"], 1)
        self.assertEqual(counts["rrn"], 1)
        self.assertEqual(counts["credit_card"], 1)
        self.assertEqual(counts["ip"], 1)
        self.assertEqual(counts["aws_key"], 1)

    def test_clean_text_has_no_counts(self) -> None:
        _, counts = redact_with_counts("version 2.0 on port 8080 after 90 days")
        self.assertEqual(counts, {})

    def test_levels_are_defined(self) -> None:
        self.assertEqual(DETECTOR_LEVELS, ("conservative", "balanced", "aggressive"))

    def test_aggressive_catches_obfuscated_email_conservative_does_not(self) -> None:
        text = "reach me at jane [at] corp [dot] com today"
        _, conservative = redact_with_counts(text, level="conservative")
        _, aggressive = redact_with_counts(text, level="aggressive")
        self.assertEqual(conservative.get("email", 0), 0)
        self.assertGreaterEqual(aggressive.get("email", 0), 1)

    def test_aggressive_catches_bare_rrn(self) -> None:
        text = "id 9001011234567 on file"
        _, conservative = redact_with_counts(text, level="conservative")
        _, aggressive = redact_with_counts(text, level="aggressive")
        self.assertEqual(conservative.get("rrn", 0), 0)
        self.assertEqual(aggressive.get("rrn", 0), 1)

    def test_unknown_level_falls_back_to_balanced(self) -> None:
        a, ca = redact_with_counts("mail a@b.com", level="nonsense")
        b, cb = redact_with_counts("mail a@b.com", level="balanced")
        self.assertEqual(ca, cb)


if __name__ == "__main__":
    unittest.main()
