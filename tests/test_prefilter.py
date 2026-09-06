from __future__ import annotations

import json
from pathlib import Path
import unittest

from reddit_prefilter import Prefilter, PrefilterConfig, RawRecord, SourceLineage
from reddit_prefilter.adapter import AdapterError, load_fixture, records_from_fixture


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "fixtures" / "prefilter_records.json"


class PrefilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lineage = SourceLineage(
            provider="fixture",
            observed_at="2026-09-05T00:00:00Z",
            source_url="fixture://freelance/new",
            request_id="request-1",
            run_id="run-1",
            response_status=200,
            metadata={"page": 1},
        )
        self.records = load_fixture(FIXTURE)
        self.result = Prefilter(
            PrefilterConfig(
                minimum_text_length=12,
                subreddit_scope=("freelance",),
            )
        ).evaluate(self.records)
        self.by_id = {}
        for decision in self.result.decisions:
            if decision.record_id is not None:
                self.by_id.setdefault(decision.record_id, decision)

    def test_fixture_accepted_post_and_comment_preserve_relationship(self) -> None:
        self.assertEqual(self.by_id["post-accepted"].status, "accepted")
        self.assertEqual(self.by_id["post-accepted"].reason_codes, ("ACCEPTED",))
        comment = self.by_id["comment-accepted"]
        self.assertEqual(comment.status, "accepted")
        self.assertEqual(comment.metadata["relationship"]["post_id"], "post-accepted")
        self.assertEqual(comment.metadata["relationship"]["parent_id"], "t3_post-accepted")
        self.assertEqual(comment.metadata["subreddit_source"], "adapter_context")

    def test_deleted_and_removed_content_is_rejected_without_erasing_evidence(self) -> None:
        deleted = self.by_id["post-deleted"]
        removed = self.by_id["comment-removed"]
        self.assertEqual(deleted.status, "rejected")
        self.assertEqual(deleted.reason_codes, ("DELETED_CONTENT",))
        self.assertEqual(removed.status, "rejected")
        self.assertEqual(removed.reason_codes, ("REMOVED_CONTENT",))
        self.assertEqual(self.records[2].raw["selftext"], "[deleted]")
        self.assertEqual(self.records[3].raw["body"], "[removed]")

    def test_duplicate_is_rejected_and_points_to_first_evidence(self) -> None:
        decisions = [item for item in self.result.decisions if item.record_id == "post-accepted"]
        self.assertEqual([item.status for item in decisions], ["accepted", "rejected"])
        self.assertEqual(decisions[1].reason_codes, ("DUPLICATE_RECORD",))
        self.assertEqual(decisions[1].metadata["duplicate_of"], decisions[0].evidence_id)
        self.assertEqual(len(self.result.records), len(self.result.decisions))

    def test_minimum_text_length_and_spam_marker_are_rejected(self) -> None:
        self.assertEqual(self.by_id["post-short"].status, "rejected")
        self.assertEqual(self.by_id["post-short"].reason_codes, ("TEXT_TOO_SHORT",))
        spam = self.by_id["comment-spam"]
        self.assertEqual(spam.status, "rejected")
        self.assertEqual(spam.reason_codes, ("SPAM_MARKER",))
        self.assertEqual(spam.metadata["matched_spam_markers"], ["buy now", "click here", "free money"])

    def test_scope_is_rejected_but_unavailable_scope_is_incomplete(self) -> None:
        outside = self.by_id["post-outside"]
        no_scope = self.by_id["post-no-scope"]
        self.assertEqual(outside.status, "rejected")
        self.assertEqual(outside.reason_codes, ("OUT_OF_SCOPE",))
        self.assertEqual(no_scope.status, "incomplete")
        self.assertEqual(no_scope.reason_codes, ("SUBREDDIT_UNAVAILABLE",))

    def test_missing_invalid_and_malformed_records_are_incomplete(self) -> None:
        missing = next(
            item
            for item in self.result.decisions
            if item.record_id is None and "MISSING_IDENTIFIER" in item.reason_codes
        )
        invalid = next(item for item in self.result.decisions if "INVALID_IDENTIFIER" in item.reason_codes)
        no_link = self.by_id["comment-no-link"]
        malformed = next(item for item in self.result.decisions if "MALFORMED_RECORD" in item.reason_codes)
        self.assertEqual(missing.status, "incomplete")
        self.assertEqual(missing.reason_codes, ("MISSING_IDENTIFIER",))
        self.assertEqual(invalid.status, "incomplete")
        self.assertEqual(invalid.reason_codes, ("INVALID_IDENTIFIER",))
        self.assertEqual(no_link.status, "incomplete")
        self.assertEqual(no_link.reason_codes, ("MISSING_POST_RELATIONSHIP",))
        self.assertEqual(malformed.status, "incomplete")
        self.assertEqual(malformed.reason_codes, ("MALFORMED_RECORD",))

    def test_lineage_and_raw_evidence_are_in_every_audit_decision(self) -> None:
        decision = self.result.decisions[0]
        self.assertEqual(decision.lineage.provider, "fixture")
        self.assertEqual(decision.lineage.request_id, "fixture-1")
        self.assertEqual(decision.metadata["rule_version"], "reddit-prefilter-v1")
        self.assertEqual(decision.metadata["raw_sha256"], self.records[0].raw_sha256)
        serialized = self.result.to_dict()
        self.assertEqual(serialized["records"][0]["raw"]["title"], "Pricing a small project")
        self.assertEqual(serialized["decisions"][0]["evidence_id"], decision.evidence_id)

    def test_repeated_runs_are_idempotent_and_do_not_mutate_raw_input(self) -> None:
        before = json.dumps(self.result.to_dict(), sort_keys=True)
        rerun = Prefilter(
            PrefilterConfig(minimum_text_length=12, subreddit_scope=("freelance",))
        ).evaluate(self.records)
        self.assertEqual(rerun.to_dict(), self.result.to_dict())
        self.assertEqual(json.dumps(self.result.to_dict(), sort_keys=True), before)

    def test_conflicting_identifiers_and_subreddit_context_are_incomplete(self) -> None:
        conflict = RawRecord(
            "post",
            {
                "id": "post-one",
                "fullname": "t3_post-two",
                "subreddit": "freelance",
                "title": "A post with conflicting identity evidence",
            },
            self.lineage,
            "other",
        )
        invalid_subreddit = RawRecord(
            "post",
            {
                "id": "post-bad-subreddit",
                "subreddit": "not a subreddit",
                "title": "A post with an invalid subreddit value",
            },
            self.lineage,
        )
        decisions = Prefilter(PrefilterConfig()).evaluate((conflict, invalid_subreddit)).decisions
        self.assertEqual(decisions[0].status, "incomplete")
        self.assertEqual(
            decisions[0].reason_codes,
            ("IDENTIFIER_CONFLICT", "SUBREDDIT_CONFLICT"),
        )
        self.assertEqual(decisions[1].status, "incomplete")
        self.assertEqual(decisions[1].reason_codes, ("INVALID_SUBREDDIT",))

    def test_invalid_parent_and_conflicting_relationships_are_incomplete(self) -> None:
        invalid_parent = RawRecord(
            "comment",
            {
                "id": "comment-bad-parent",
                "link_id": "t3_post-accepted",
                "parent_id": "not a fullname",
                "body": "A comment with enough text to inspect.",
            },
            self.lineage,
            "freelance",
        )
        conflict = RawRecord(
            "comment",
            {
                "id": "comment-conflict",
                "post_id": "post-accepted",
                "link_id": "t3-other-post",
                "body": "A comment with a conflicting link relationship.",
            },
            self.lineage,
            "freelance",
        )
        decisions = Prefilter(PrefilterConfig(subreddit_scope=("freelance",))).evaluate(
            (invalid_parent, conflict)
        ).decisions
        self.assertEqual(decisions[0].status, "incomplete")
        self.assertEqual(decisions[0].reason_codes, ("INVALID_PARENT_ID",))
        self.assertEqual(decisions[1].status, "incomplete")
        self.assertEqual(decisions[1].reason_codes, ("RELATIONSHIP_CONFLICT",))

    def test_missing_and_invalid_lineage_are_incomplete_and_scope_is_optional(self) -> None:
        record = RawRecord(
            "post",
            {"id": "post-no-lineage", "title": "A post with enough text"},
        )
        missing_lineage = Prefilter(PrefilterConfig()).evaluate((record,)).decisions[0]
        self.assertEqual(missing_lineage.status, "incomplete")
        self.assertEqual(missing_lineage.reason_codes, ("MISSING_SOURCE_LINEAGE",))

        invalid_lineage = RawRecord(
            "post",
            {"id": "post-invalid-lineage", "title": "A post with enough text"},
            SourceLineage(provider="", observed_at=""),
        )
        invalid_decision = Prefilter(PrefilterConfig()).evaluate((invalid_lineage,)).decisions[0]
        self.assertEqual(invalid_decision.status, "incomplete")
        self.assertEqual(invalid_decision.reason_codes, ("INVALID_SOURCE_LINEAGE",))

        no_scope = Prefilter(PrefilterConfig(minimum_text_length=1)).evaluate(
            (RawRecord("post", {"id": "post-no-scope-check", "title": "Enough text"}, self.lineage),)
        ).decisions[0]
        self.assertEqual(no_scope.status, "accepted")

    def test_scope_and_marker_configuration_are_explicit(self) -> None:
        record = records_from_fixture(
            {
                "records": [
                    {
                        "record_type": "post",
                        "raw": {
                            "id": "p-custom",
                            "subreddit": "r/SAAS",
                            "title": "A long enough title for this rule",
                            "selftext": "ordinary body",
                        },
                        "lineage": {
                            "provider": "fixture",
                            "observed_at": "2026-09-05T00:00:00Z",
                        },
                    }
                ]
            }
        )
        decision = Prefilter(
            PrefilterConfig(
                minimum_text_length=1,
                subreddit_scope=("saas",),
                spam_markers=("ordinary",),
            )
        ).evaluate(record).decisions[0]
        self.assertEqual(decision.status, "rejected")
        self.assertEqual(decision.reason_codes, ("SPAM_MARKER",))
        self.assertTrue(decision.metadata["scope_match"])

    def test_fixture_adapter_rejects_an_ambiguous_top_level_shape(self) -> None:
        with self.assertRaises(AdapterError):
            records_from_fixture({"listings": []})

    def test_default_lineage_can_be_applied_by_the_adapter(self) -> None:
        records = records_from_fixture(
            {"records": [{"record_type": "post", "raw": {"id": "p", "title": "ok"}}]},
            default_lineage=self.lineage,
        )
        self.assertIs(records[0].lineage, self.lineage)


if __name__ == "__main__":
    unittest.main()
