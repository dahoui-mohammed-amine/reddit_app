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
        self.assertEqual(spam.metadata["matched_spam_markers"], ("buy now", "click here", "free money"))

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

    def test_decision_metadata_is_immutable_and_serialized_as_a_copy(self) -> None:
        record = RawRecord(
            "comment",
            {
                "id": "comment-metadata-freeze",
                "link_id": "post-parent",
                "body": "Buy now for a guaranteed result.",
            },
            self.lineage,
        )
        decision = Prefilter().evaluate((record,)).decisions[0]
        before = decision.to_dict()

        with self.assertRaises(TypeError):
            decision.metadata["raw_sha256"] = "changed"
        with self.assertRaises(AttributeError):
            decision.metadata["matched_spam_markers"].append("changed")
        serialized = decision.to_dict()
        serialized["metadata"]["raw_sha256"] = "changed"

        self.assertEqual(decision.to_dict(), before)

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

    def test_empty_content_aliases_fall_back_to_populated_fields(self) -> None:
        post = RawRecord(
            "post",
            {
                "id": "post-empty-fallback",
                "title": "A post title",
                "selftext": "",
                "body": "Click here for a guaranteed result.",
            },
            self.lineage,
        )
        comment = RawRecord(
            "comment",
            {
                "id": "comment-empty-fallback",
                "link_id": "post-empty-fallback",
                "body": "",
                "bodyText": "Buy now for a guaranteed result.",
            },
            self.lineage,
        )

        decisions = Prefilter().evaluate((post, comment)).decisions

        self.assertEqual(decisions[0].reason_codes, ("SPAM_MARKER",))
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

    def test_empty_identifier_aliases_are_invalid_when_supplied(self) -> None:
        post = RawRecord(
            "post",
            {
                "id": "",
                "fullname": "t3_post-empty-id",
                "title": "Enough",
            },
            self.lineage,
        )
        link = RawRecord(
            "comment",
            {
                "id": "comment-empty-link",
                "post_id": "post-parent",
                "link_id": "",
                "body": "Enough",
            },
            self.lineage,
        )
        parent = RawRecord(
            "comment",
            {
                "id": "comment-empty-parent",
                "link_id": "post-parent",
                "parent_id": "",
                "body": "Enough",
            },
            self.lineage,
        )

        decisions = Prefilter().evaluate((post, link, parent)).decisions

        self.assertEqual(decisions[0].reason_codes, ("INVALID_IDENTIFIER",))
        self.assertEqual(decisions[1].reason_codes, ("INVALID_POST_RELATIONSHIP",))
        self.assertEqual(decisions[2].reason_codes, ("INVALID_PARENT_ID",))

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

    def test_fixture_preserves_malformed_required_lineage_fields(self) -> None:
        for field in ("provider", "observed_at"):
            lineage = {
                "provider": "fixture",
                "observed_at": "2026-09-05T00:00:00Z",
            }
            lineage[field] = None
            with self.subTest(field=field):
                records = records_from_fixture(
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
                decision = Prefilter().evaluate(records).decisions[0]
                self.assertEqual(decision.status, "incomplete")
                self.assertEqual(decision.reason_codes, ("INVALID_SOURCE_LINEAGE",))
                self.assertEqual(records[0].to_dict()["lineage"][field], None)

    def test_fixture_preserves_invalid_optional_lineage_fields(self) -> None:
        invalid_lineage = {
            "provider": "fixture",
            "observed_at": "2026-09-05T00:00:00Z",
            "source_url": [],
        }
        records = records_from_fixture(
            {
                "records": [
                    {
                        "record_type": "post",
                        "lineage": invalid_lineage,
                        "raw": {"id": "post-invalid-optional-lineage"},
                    }
                ]
            }
        )
        decision = Prefilter().evaluate(records).decisions[0]
        self.assertEqual(decision.status, "incomplete")
        self.assertEqual(decision.reason_codes, ("INVALID_SOURCE_LINEAGE",))
        self.assertEqual(records[0].to_dict()["lineage"], invalid_lineage)

    def test_invalid_direct_lineage_is_snapshotted_and_incomplete(self) -> None:
        source_url = ["fixture://source"]
        lineage = SourceLineage(
            provider="fixture",
            observed_at="2026-09-05T00:00:00Z",
            source_url=source_url,
        )
        result = Prefilter().evaluate(
            (RawRecord("post", {"id": "post-invalid-source-url", "title": "Enough"}, lineage),)
        )
        before = result.to_dict()

        source_url.append("mutated")

        self.assertEqual(result.decisions[0].status, "incomplete")
        self.assertEqual(result.decisions[0].reason_codes, ("INVALID_SOURCE_LINEAGE",))
        self.assertEqual(result.to_dict(), before)

    def test_malformed_lineage_is_preserved_in_evidence_identity(self) -> None:
        first = RawRecord(
            "post",
            {"id": "post-malformed-lineage", "title": "Enough"},
            {"provider": None},
        )
        second = RawRecord(
            "post",
            {"id": "post-malformed-lineage", "title": "Enough"},
            {"provider": 0},
        )

        result = Prefilter().evaluate((first, second))
        serialized = result.to_dict()

        self.assertNotEqual(result.decisions[0].evidence_id, result.decisions[1].evidence_id)
        self.assertEqual(serialized["records"][0]["lineage"], {"provider": None})
        self.assertEqual(serialized["records"][1]["lineage"], {"provider": 0})
        self.assertEqual(serialized["decisions"][0]["lineage"], {"provider": None})

    def test_non_string_mapping_keys_remain_distinct_evidence(self) -> None:
        mixed_raw = RawRecord(
            "post",
            {1: "numeric key", "1": "string key", "id": "post-mixed-keys", "title": "Enough"},
            self.lineage,
        )
        string_raw = RawRecord(
            "post",
            {"1": "string key", "id": "post-mixed-keys", "title": "Enough"},
            self.lineage,
        )
        mixed_lineage = SourceLineage(
            provider="fixture",
            observed_at="2026-09-05T00:00:00Z",
            metadata={1: "numeric key", "1": "string key"},
        )

        result = Prefilter().evaluate(
            (mixed_raw, string_raw, RawRecord("post", {"id": "post-mixed-lineage", "title": "Enough"}, mixed_lineage))
        )

        self.assertNotEqual(result.decisions[0].evidence_id, result.decisions[1].evidence_id)
        self.assertEqual(result.decisions[2].status, "accepted")
        self.assertEqual(len(result.decisions[2].evidence_id), 64)
        json.dumps(result.to_dict(), sort_keys=True)

    def test_mapping_tag_cannot_collide_with_literal_mapping_data(self) -> None:
        mixed = RawRecord("post", {1: "v"}, self.lineage)
        literal = RawRecord(
            "post",
            {
                "__mapping__": [
                    {"key_type": "builtins.int", "key": 1, "value": "v"}
                ]
            },
            self.lineage,
        )

        self.assertNotEqual(mixed.evidence_id, literal.evidence_id)

    def test_surrogate_content_keeps_evidence_hashes_nonempty(self) -> None:
        record = RawRecord(
            "post",
            {"id": "post-surrogate", "title": "Visible \ud800 content"},
            self.lineage,
        )

        result = Prefilter().evaluate((record,))
        decision = result.decisions[0]

        self.assertEqual(decision.status, "accepted")
        self.assertEqual(len(record.raw_sha256), 64)
        self.assertEqual(len(decision.raw_sha256), 64)
        self.assertEqual(len(record.evidence_id), 64)
        self.assertEqual(len(decision.evidence_id), 64)
        json.dumps(result.to_dict(), sort_keys=True)

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

    def test_post_titles_are_content_sentinels_but_authors_are_not(self) -> None:
        deleted_title = RawRecord(
            "post",
            {"id": "post-deleted-title", "title": "[deleted]"},
            self.lineage,
        )
        removed_title = RawRecord(
            "post",
            {"id": "post-removed-title", "title": "[removed]"},
            self.lineage,
        )
        visible_comment = RawRecord(
            "comment",
            {
                "id": "comment-deleted-author",
                "link_id": "post-parent",
                "author": "[deleted]",
                "body": "A visible comment with enough text",
            },
            self.lineage,
        )

        decisions = Prefilter().evaluate(
            (deleted_title, removed_title, visible_comment)
        ).decisions

        self.assertEqual(decisions[0].status, "rejected")
        self.assertEqual(decisions[0].reason_codes, ("DELETED_CONTENT",))
        self.assertEqual(decisions[1].status, "rejected")
        self.assertEqual(decisions[1].reason_codes, ("REMOVED_CONTENT",))
        self.assertEqual(decisions[2].status, "accepted")
        self.assertEqual(decisions[2].reason_codes, ("ACCEPTED",))

    def test_fixture_preserves_non_string_subreddit_context(self) -> None:
        records = records_from_fixture(
            {
                "records": [
                    {
                        "record_type": "post",
                        "subreddit_context": ["freelance"],
                        "raw": {"id": "post-invalid-context"},
                    }
                ]
            },
            default_lineage=self.lineage,
        )
        decision = Prefilter(PrefilterConfig(subreddit_scope=("freelance",))).evaluate(
            records
        ).decisions[0]
        self.assertEqual(decision.status, "incomplete")
        self.assertEqual(decision.reason_codes, ("INVALID_SUBREDDIT",))

    def test_direct_invalid_subreddit_context_is_incomplete(self) -> None:
        record = RawRecord(
            "post",
            {"id": "post-direct-invalid-context", "title": "Enough"},
            self.lineage,
            ["freelance"],
        )
        decision = Prefilter(PrefilterConfig(subreddit_scope=("freelance",))).evaluate(
            (record,)
        ).decisions[0]

        self.assertEqual(decision.status, "incomplete")
        self.assertEqual(decision.reason_codes, ("INVALID_SUBREDDIT",))

    def test_empty_raw_subreddit_does_not_fall_back_to_context(self) -> None:
        records = tuple(
            RawRecord(
                "post",
                {
                    "id": f"post-empty-subreddit-{index}",
                    "subreddit": value,
                    "title": "A visible post with enough text",
                },
                self.lineage,
                "freelance",
            )
            for index, value in enumerate(("", "   "))
        )

        for config in (PrefilterConfig(), PrefilterConfig(subreddit_scope=("freelance",))):
            with self.subTest(config=config):
                decisions = Prefilter(config).evaluate(records).decisions

                self.assertEqual(
                    [decision.status for decision in decisions],
                    ["incomplete", "incomplete"],
                )
                self.assertEqual(
                    [decision.reason_codes for decision in decisions],
                    [("INVALID_SUBREDDIT",), ("INVALID_SUBREDDIT",)],
                )

    def test_uppercase_empty_subreddit_prefix_is_invalid(self) -> None:
        records = tuple(
            RawRecord(
                "post",
                {"id": f"post-uppercase-empty-{index}", "subreddit": "R/", "title": "Enough"},
                self.lineage,
            )
            for index in range(2)
        )

        decisions = Prefilter(PrefilterConfig(subreddit_scope=("freelance",))).evaluate(
            records
        ).decisions

        self.assertEqual([decision.status for decision in decisions], ["incomplete", "incomplete"])
        self.assertEqual(
            [decision.reason_codes for decision in decisions],
            [("INVALID_SUBREDDIT",), ("INVALID_SUBREDDIT",)],
        )

    def test_blank_adapter_context_is_invalid(self) -> None:
        records = tuple(
            RawRecord(
                "post",
                {"id": f"post-blank-context-{index}", "title": "Enough"},
                self.lineage,
                value,
            )
            for index, value in enumerate(("", " "))
        )

        for config in (PrefilterConfig(), PrefilterConfig(subreddit_scope=("freelance",))):
            with self.subTest(config=config):
                decisions = Prefilter(config).evaluate(records).decisions

                self.assertEqual(
                    [decision.status for decision in decisions],
                    ["incomplete", "incomplete"],
                )
                self.assertEqual(
                    [decision.reason_codes for decision in decisions],
                    [("INVALID_SUBREDDIT",), ("INVALID_SUBREDDIT",)],
                )

    def test_malformed_content_state_is_incomplete(self) -> None:
        deleted = RawRecord(
            "post",
            {
                "id": "post-malformed-deleted-state",
                "deleted": [],
                "title": "A visible post with enough text",
            },
            self.lineage,
        )
        removed = RawRecord(
            "post",
            {
                "id": "post-malformed-removed-state",
                "removed": {},
                "title": "A visible post with enough text",
            },
            self.lineage,
        )

        corrupt_deleted = RawRecord(
            "post",
            {
                "id": "post-corrupt-deleted-state",
                "deleted": "corrupt",
                "title": "A visible post with enough text",
            },
            self.lineage,
        )
        corrupt_removed = RawRecord(
            "post",
            {
                "id": "post-corrupt-removed-state",
                "is_removed": "corrupt",
                "title": "A visible post with enough text",
            },
            self.lineage,
        )
        numeric_deleted = RawRecord(
            "post",
            {
                "id": "post-numeric-deleted-state",
                "deleted": 2,
                "title": "A visible post with enough text",
            },
            self.lineage,
        )
        numeric_category = RawRecord(
            "post",
            {
                "id": "post-numeric-category-state",
                "removed_by_category": 1,
                "title": "A visible post with enough text",
            },
            self.lineage,
        )

        decisions = Prefilter().evaluate(
            (deleted, removed, corrupt_deleted, corrupt_removed, numeric_deleted, numeric_category)
        ).decisions

        self.assertEqual(
            [decision.status for decision in decisions],
            ["incomplete", "incomplete", "incomplete", "incomplete", "incomplete", "incomplete"],
        )
        self.assertEqual(
            [decision.reason_codes for decision in decisions],
            [("INVALID_CONTENT_STATE",)] * 6,
        )

    def test_fixture_preserves_unmodeled_lineage_fields(self) -> None:
        records = records_from_fixture(
            {
                "records": [
                    {
                        "record_type": "post",
                        "lineage": {
                            "provider": "fixture",
                            "observed_at": "2026-09-05T00:00:00Z",
                            "source_batch": "b1",
                        },
                        "raw": {"id": "post-unmodeled-lineage", "title": "Enough"},
                    }
                ]
            }
        )

        result = Prefilter().evaluate(records)

        self.assertEqual(result.decisions[0].status, "incomplete")
        self.assertEqual(result.decisions[0].reason_codes, ("INVALID_SOURCE_LINEAGE",))
        self.assertEqual(result.to_dict()["records"][0]["lineage"]["source_batch"], "b1")

    def test_non_string_content_is_incomplete(self) -> None:
        record = RawRecord(
            "post",
            {"id": "post-invalid-content", "title": []},
            self.lineage,
        )
        decision = Prefilter().evaluate((record,)).decisions[0]

        self.assertEqual(decision.status, "incomplete")
        self.assertEqual(decision.reason_codes, ("INVALID_CONTENT",))

    def test_non_string_identifiers_and_relationships_are_incomplete(self) -> None:
        invalid_id = RawRecord(
            "post",
            {"id": 123, "title": "Enough"},
            self.lineage,
        )
        invalid_link = RawRecord(
            "comment",
            {"id": "comment-invalid-link", "link_id": 123, "body": "Enough"},
            self.lineage,
        )
        invalid_parent = RawRecord(
            "comment",
            {
                "id": "comment-invalid-parent",
                "link_id": "post-parent",
                "parent_id": 123,
                "body": "Enough",
            },
            self.lineage,
        )

        decisions = Prefilter().evaluate((invalid_id, invalid_link, invalid_parent)).decisions

        self.assertEqual(decisions[0].status, "incomplete")
        self.assertEqual(decisions[0].reason_codes, ("INVALID_IDENTIFIER",))
        self.assertEqual(decisions[1].status, "incomplete")
        self.assertEqual(decisions[1].reason_codes, ("INVALID_POST_RELATIONSHIP",))
        self.assertEqual(decisions[2].status, "incomplete")
        self.assertEqual(decisions[2].reason_codes, ("INVALID_PARENT_ID",))

    def test_malformed_record_type_is_incomplete(self) -> None:
        record = RawRecord(
            ["post"],
            {"id": "post-malformed-type", "title": "Enough"},
            self.lineage,
        )

        decision = Prefilter().evaluate((record,)).decisions[0]

        self.assertEqual(decision.status, "incomplete")
        self.assertEqual(decision.reason_codes, ("MALFORMED_RECORD",))

    def test_adapter_preserves_malformed_record_type_values(self) -> None:
        records = records_from_fixture(
            {
                "records": [
                    {
                        "record_type": 1,
                        "raw": {"id": "post-boundary-type", "title": "Enough"},
                    },
                    {
                        "record_type": "1",
                        "raw": {"id": "post-boundary-type", "title": "Enough"},
                    },
                ]
            },
            default_lineage=self.lineage,
        )

        result = Prefilter().evaluate(records)

        self.assertEqual(records[0].record_type, 1)
        self.assertEqual(records[1].record_type, "1")
        self.assertNotEqual(result.decisions[0].evidence_id, result.decisions[1].evidence_id)
        self.assertEqual(result.decisions[0].reason_codes, ("MALFORMED_RECORD",))
        self.assertEqual(result.decisions[1].reason_codes, ("MALFORMED_RECORD",))

    def test_malformed_record_type_mapping_serializes_as_audit_evidence(self) -> None:
        records = records_from_fixture(
            {
                "records": [
                    {
                        "record_type": {"kind": "post"},
                        "raw": {"id": "post-mapping-type", "title": "Enough"},
                    }
                ]
            },
            default_lineage=self.lineage,
        )

        result = Prefilter().evaluate(records)
        serialized = result.to_dict()

        self.assertEqual(serialized["records"][0]["record_type"], {"kind": "post"})
        self.assertEqual(serialized["decisions"][0]["record_type"], {"kind": "post"})
        self.assertEqual(serialized["decisions"][0]["reason_codes"], ["MALFORMED_RECORD"])
        json.dumps(serialized, sort_keys=True)

    def test_sequence_configuration_options_reject_bare_strings(self) -> None:
        with self.assertRaises(ValueError):
            PrefilterConfig(subreddit_scope="freelance")
        with self.assertRaises(ValueError):
            PrefilterConfig(spam_markers="spam")

    def test_sequence_configuration_options_are_snapshotted(self) -> None:
        marker_generator = iter(("buy now",))
        scope_values = ["freelance"]
        config = PrefilterConfig(
            subreddit_scope=iter(scope_values),
            spam_markers=marker_generator,
        )
        scope_values.clear()

        record = RawRecord(
            "post",
            {
                "id": "post-snapshotted-config",
                "subreddit": "freelance",
                "title": "Buy now for a guaranteed result.",
            },
            self.lineage,
        )
        decision = Prefilter(config).evaluate((record,)).decisions[0]

        self.assertEqual(config.subreddit_scope, ("freelance",))
        self.assertEqual(config.spam_markers, ("buy now",))
        self.assertEqual(decision.status, "rejected")
        self.assertEqual(decision.reason_codes, ("SPAM_MARKER",))

    def test_default_spam_rules_remain_conservative_and_configurable(self) -> None:
        record = RawRecord(
            "post",
            {"id": "post-mentions-spam", "title": "How do I report spam?"},
            self.lineage,
        )

        default_decision = Prefilter().evaluate((record,)).decisions[0]
        configured_decision = Prefilter(
            PrefilterConfig(spam_markers=("spam",))
        ).evaluate((record,)).decisions[0]

        self.assertEqual(default_decision.status, "accepted")
        self.assertEqual(default_decision.reason_codes, ("ACCEPTED",))
        self.assertEqual(configured_decision.status, "rejected")
        self.assertEqual(configured_decision.reason_codes, ("SPAM_MARKER",))

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
