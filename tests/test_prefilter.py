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

    def test_invalid_lineage_is_auditable_and_raw_evidence_is_snapshotted(self) -> None:
        raw = {
            "id": "post-snapshot",
            "title": "Original title",
            "details": {"source": "original"},
        }
        record = RawRecord("post", raw, {})
        result = Prefilter().evaluate((record,))
        decision = result.decisions[0]
        before = json.dumps(result.to_dict(), sort_keys=True)

        raw["title"] = "Changed title"
        raw["details"]["source"] = "changed"

        self.assertEqual(decision.status, "incomplete")
        self.assertEqual(decision.reason_codes, ("INVALID_SOURCE_LINEAGE",))
        self.assertEqual(json.dumps(result.to_dict(), sort_keys=True), before)
        self.assertEqual(result.records[0].raw["title"], "Original title")
        self.assertEqual(result.records[0].raw["details"]["source"], "original")
        with self.assertRaises(TypeError):
            result.records[0].raw["title"] = "not mutable"

    def test_lineage_metadata_is_snapshotted_with_raw_evidence(self) -> None:
        metadata = {"page": 1, "filters": ["new"]}
        lineage = SourceLineage(
            provider="fixture",
            observed_at="2026-09-05T00:00:00Z",
            metadata=metadata,
        )
        result = Prefilter().evaluate(
            (RawRecord("post", {"id": "post-lineage-snapshot", "title": "Enough"}, lineage),)
        )
        before = result.to_dict()

        metadata["page"] = 2
        metadata["filters"].append("hot")

        self.assertEqual(result.to_dict(), before)
        with self.assertRaises(TypeError):
            lineage.metadata["page"] = 3

    def test_unsupported_qualified_ids_and_relationships_are_incomplete(self) -> None:
        post = RawRecord(
            "post",
            {"id": "t2_comment", "title": "A sufficiently long post"},
            self.lineage,
        )
        comment_relationship = RawRecord(
            "comment",
            {
                "id": "comment-typed-link",
                "link_id": "t2_post",
                "body": "A sufficiently long comment body",
            },
            self.lineage,
            "freelance",
        )
        comment_parent = RawRecord(
            "comment",
            {
                "id": "comment-typed-parent",
                "link_id": "post-parent",
                "parent_id": "t2_parent",
                "body": "A sufficiently long comment body",
            },
            self.lineage,
            "freelance",
        )

        decisions = Prefilter().evaluate((post, comment_relationship, comment_parent)).decisions

        self.assertEqual(decisions[0].status, "incomplete")
        self.assertEqual(decisions[0].reason_codes, ("INVALID_IDENTIFIER",))
        self.assertEqual(decisions[1].status, "incomplete")
        self.assertEqual(decisions[1].reason_codes, ("INVALID_POST_RELATIONSHIP",))
        self.assertEqual(decisions[2].status, "incomplete")
        self.assertEqual(decisions[2].reason_codes, ("INVALID_PARENT_ID",))

    def test_content_aliases_use_the_first_non_null_value(self) -> None:
        post = RawRecord(
            "post",
            {
                "id": "post-body-fallback",
                "title": "A post title",
                "selftext": None,
                "body": "Buy now for a guaranteed result.",
            },
            self.lineage,
        )
        comment = RawRecord(
            "comment",
            {
                "id": "comment-bodytext-fallback",
                "link_id": "post-body-fallback",
                "body": None,
                "bodyText": "Click here for a guaranteed result.",
            },
            self.lineage,
        )

        decisions = Prefilter().evaluate((post, comment)).decisions

        self.assertEqual(decisions[0].status, "rejected")
        self.assertEqual(decisions[0].reason_codes, ("SPAM_MARKER",))
        self.assertEqual(decisions[1].status, "rejected")
        self.assertEqual(decisions[1].reason_codes, ("SPAM_MARKER",))

    def test_identifier_conflict_remains_incomplete_with_duplicate_evidence(self) -> None:
        first = RawRecord(
            "post",
            {"id": "post-duplicate-conflict", "title": "A canonical post"},
            self.lineage,
        )
        conflicting_duplicate = RawRecord(
            "post",
            {
                "id": "post-duplicate-conflict",
                "fullname": "t3-other-post",
                "title": "A conflicting duplicate post",
            },
            self.lineage,
        )

        decisions = Prefilter().evaluate((first, conflicting_duplicate)).decisions

        self.assertEqual(decisions[1].status, "incomplete")
        self.assertEqual(
            decisions[1].reason_codes,
            ("IDENTIFIER_CONFLICT", "DUPLICATE_RECORD"),
        )
        self.assertEqual(decisions[1].metadata["duplicate_of"], decisions[0].evidence_id)

    def test_removed_compatibility_aliases_and_community_fallback_are_not_public(self) -> None:
        config = PrefilterConfig()
        self.assertFalse(hasattr(config, "min_text_length"))
        self.assertFalse(hasattr(config, "subreddits"))

        record = RawRecord(
            "post",
            {
                "id": "post-community-only",
                "community": "freelance",
                "title": "A post with enough text",
            },
            self.lineage,
        )
        decision = Prefilter(PrefilterConfig(subreddit_scope=("freelance",))).evaluate(
            (record,)
        ).decisions[0]
        self.assertEqual(decision.status, "incomplete")
        self.assertEqual(decision.reason_codes, ("SUBREDDIT_UNAVAILABLE",))

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

    def test_fixture_rejects_null_required_lineage_fields(self) -> None:
        for field in ("provider", "observed_at"):
            lineage = {
                "provider": "fixture",
                "observed_at": "2026-09-05T00:00:00Z",
            }
            lineage[field] = None
            with self.subTest(field=field):
                with self.assertRaises(AdapterError):
                    records_from_fixture(
                        {
                            "records": [
                                {
                                    "record_type": "post",
                                    "lineage": lineage,
                                    "raw": {"id": "post-invalid-fixture-lineage"},
                                }
                            ]
                        }
                    )

    def test_text_content_detects_removed_markers_and_normalizes_whitespace(self) -> None:
        removed = RawRecord(
            "post",
            {"id": "post-removed-text", "text": "[removed]"},
            self.lineage,
        )
        marker = RawRecord(
            "comment",
            {
                "id": "comment-whitespace-marker",
                "link_id": "post-parent",
                "body": "Buy\nnow",
            },
            self.lineage,
        )
        short = RawRecord(
            "post",
            {"id": "post-whitespace-short", "title": "A\t\nB"},
            self.lineage,
        )

        removed_decision = Prefilter().evaluate((removed,)).decisions[0]
        marker_decision = Prefilter().evaluate((marker,)).decisions[0]
        short_decision = Prefilter(PrefilterConfig(minimum_text_length=4)).evaluate(
            (short,)
        ).decisions[0]

        self.assertEqual(removed_decision.status, "rejected")
        self.assertEqual(removed_decision.reason_codes, ("REMOVED_CONTENT",))
        self.assertEqual(marker_decision.status, "rejected")
        self.assertEqual(marker_decision.reason_codes, ("SPAM_MARKER",))
        self.assertEqual(short_decision.status, "rejected")
        self.assertEqual(short_decision.reason_codes, ("TEXT_TOO_SHORT",))
        self.assertEqual(short_decision.metadata["text_length"], 3)

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
