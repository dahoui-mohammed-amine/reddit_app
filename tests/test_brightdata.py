from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reddit_ingestion.config import Config
from reddit_ingestion.db import Database
from reddit_ingestion.models import PostSnapshot
from reddit_ingestion.providers import BrightDataProvider, HttpResponse, JsonClient
from reddit_ingestion.runner import run_once

OBSERVED = "2026-09-06T00:00:00Z"


def config(root: Path, *, comments: str = "bounded", limit: int = 2, subreddits: tuple[str, ...] = ("smallbusiness", "freelance")) -> Config:
    return Config(
        subreddits=subreddits,
        database_path=root / "reddit.sqlite3",
        provider="brightdata",
        fixture_path=None,
        listing_limit=100,
        max_discovery_pages=3,
        refresh_interval_minutes=1,
        max_refresh_posts=100,
        comments_mode=comments,
        comment_depth=1,
        comment_limit=limit,
        request_timeout_seconds=1,
        max_retries=3,
    )


def post_record(subreddit: str, post_id: str = "p") -> dict[str, object]:
    return {
        "post_id": post_id,
        "url": f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/title/",
        "community_url": f"https://www.reddit.com/r/{subreddit}/",
        "title": "A post",
        "description": "A description",
        "user_posted": "author",
        "date_posted": OBSERVED,
        "num_upvotes": 7,
        "num_comments": 1,
    }


class BrightDataTests(unittest.TestCase):
    def test_combined_discovery_uses_documented_wrapper_bearer_and_new_sort(self) -> None:
        calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            calls.append((method, url, headers, body))
            return HttpResponse(200, {}, json.dumps([post_record("smallbusiness")]).encode())

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            root = Path(directory)
            cfg = config(root)
            client = JsonClient(timeout=1, retries=3, transport=transport, sleep=lambda _: None)
            provider = BrightDataProvider(cfg, client=client)
            first = provider.discover("smallbusiness", None, cfg)
            second = provider.discover("freelance", None, cfg)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][2]["Authorization"], "Bearer secret")
        query = dict(item.split("=", 1) for item in calls[0][1].split("?", 1)[1].split("&"))
        self.assertEqual(query["dataset_id"], BrightDataProvider.posts_dataset_id)
        self.assertEqual(query["type"], "discover_new")
        self.assertEqual(query["discover_by"], "subreddit_url")
        body = json.loads(calls[0][3].decode())
        self.assertEqual([item["sort_by"] for item in body["input"]], ["new", "new"])
        self.assertEqual([item["url"] for item in body["input"]], [
            "https://www.reddit.com/r/smallbusiness/",
            "https://www.reddit.com/r/freelance/",
        ])
        self.assertEqual(first.posts[0].post_id, "p")
        self.assertEqual(len(first.request_records), 1)
        self.assertEqual(len(second.request_records), 0)
        self.assertEqual(first.request_records[0].operation, "discover")
        self.assertEqual(first.request_records[0].metadata["accounting"]["requested_inputs"], 2)
        self.assertEqual(first.request_records[0].metadata["accounting"]["returned_records"], 1)
        self.assertEqual(client.retries, 0)  # Bright Data never replays collection requests

    def test_request_evidence_is_persisted_without_turning_records_into_request_units(self) -> None:
        raw = post_record("smallbusiness")

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [raw], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            root = Path(directory)
            cfg = config(root)
            db = Database(cfg.database_path)
            try:
                summary = run_once(db, BrightDataProvider(cfg, client=Client()), cfg, "discover", allow_paid=True)
                request = db.connection.execute("SELECT metadata_json FROM requests").fetchone()
                evidence = json.loads(request[0])
                self.assertEqual(summary.requests, 1)
                self.assertEqual(evidence["accounting"]["requested_inputs"], 2)
                self.assertEqual(evidence["accounting"]["returned_records"], 1)
                self.assertEqual(evidence["request_units"], 1)
                self.assertEqual(evidence["raw_payload"], [raw])
                self.assertIn("fetched_at", evidence)
            finally:
                db.close()

    def test_brightdata_fields_normalize_and_raw_payload_is_retained(self) -> None:
        raw = post_record("smallbusiness")
        raw["comments"] = [{
            "comment_id": "c",
            "post_id": "p",
            "user_posted": "commenter",
            "comment": "Useful reply",
            "date_posted": OBSERVED,
            "num_upvotes": 3,
            "parent_id": "t3_p",
        }]

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [raw], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            result = BrightDataProvider(config(Path(directory)), client=Client()).discover("smallbusiness", None, config(Path(directory)))

        post = result.posts[0]
        self.assertEqual((post.body, post.author, post.score, post.created_at), ("A description", "author", 7, OBSERVED))
        self.assertEqual(post.comments, [])
        self.assertTrue(any(gap.reason == "unsupported" and gap.entity_type == "comment" for gap in result.gaps))
        self.assertEqual(post.observed_at, result.metadata["fetched_at"])
        self.assertEqual(result.request_records[0].metadata["raw_payload"], [raw])
        self.assertEqual(result.request_records[0].metadata["response_shape"], "array")

    def test_sparse_records_are_valid_but_deleted_like_partial_errors_are_unavailable(self) -> None:
        sparse = {"post_id": "sparse", "community_url": "https://www.reddit.com/r/smallbusiness/", "num_comments": 0}
        deleted_error = {"url": "https://www.reddit.com/r/freelance/", "error": "deleted post"}

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [sparse, deleted_error], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            root = Path(directory)
            cfg = config(root)
            provider = BrightDataProvider(cfg, client=Client())
            sparse_page = provider.discover("smallbusiness", None, cfg)
            unavailable_page = provider.discover("freelance", None, cfg)

        self.assertEqual(sparse_page.posts[0].post_id, "sparse")
        self.assertIsNone(sparse_page.posts[0].title)
        self.assertFalse(any(gap.reason == "unavailable" for gap in sparse_page.gaps))
        self.assertTrue(any(gap.reason == "unavailable" for gap in unavailable_page.gaps))
        self.assertEqual(sparse_page.request_records[0].metadata["accounting"]["provider_error_count"], 1)

    def test_brightdata_200_object_data_and_records_are_malformed(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return self.payload, HttpResponse(200, {}, b""), "request"

        for payload in ({"data": [post_record("smallbusiness")]}, {"records": [post_record("smallbusiness")]}, {"unexpected": True}):
            with self.subTest(payload=payload):
                Client.payload = payload
                with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
                    root = Path(directory)
                    cfg = config(root)
                    result = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

                self.assertEqual(result.posts, [])
                self.assertTrue(any(gap.reason == "provider_error" and "malformed" in (gap.detail or "") for gap in result.gaps))
                self.assertTrue(result.metadata["request_failed"])
                self.assertEqual(result.request_records[0].metadata["raw_payload"], payload)

    def test_bounded_comments_are_explicitly_unsupported_without_comment_request(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                calls.append((url, payload or {}))
                record = post_record("smallbusiness")
                record["num_comments"] = 2
                record["comments"] = [
                    {"comment_id": "c1", "post_id": "p", "comment": "one", "parent_id": "t3_p"},
                    {"comment_id": "c2", "post_id": "p", "comment": "two", "parent_id": "t3_p"},
                ]
                return [record], HttpResponse(200, {}, b""), "post-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            root = Path(directory)
            cfg = config(root, limit=1)
            result = BrightDataProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual([record.operation for record in result.request_records], ["refresh"])
        self.assertEqual(result.posts[0].comments, [])
        self.assertTrue(any(gap.reason == "unsupported" and gap.entity_type == "comment" for gap in result.gaps))
        self.assertNotEqual(result.posts[0].observed_at, "")

    def test_202_snapshot_is_a_safe_unsupported_result_without_followup(self) -> None:
        calls: list[int] = []

        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            calls.append(1)
            return HttpResponse(202, {}, b'{"snapshot_id":"s_123"}')

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            root = Path(directory)
            cfg = config(root)
            provider = BrightDataProvider(cfg, client=JsonClient(timeout=1, retries=2, transport=transport, sleep=lambda _: None))
            result = provider.discover("smallbusiness", None, cfg)

        self.assertEqual(len(calls), 1)
        self.assertEqual(result.posts, [])
        self.assertEqual(len(result.gaps), 1)
        self.assertEqual(result.request_records[0].response_status, 202)
        self.assertEqual(result.metadata["async_snapshot_id"], "s_123")
        self.assertTrue(result.metadata["checkpoint_deferred"])
        self.assertTrue(any(gap.reason == "unsupported" for gap in result.gaps))

    def test_http_errors_are_not_retried_and_preserve_provider_error(self) -> None:
        for status, expected_reason in ((400, "provider_error"), (401, "provider_error"), (403, "provider_error"), (404, "unavailable"), (429, "provider_error"), (500, "provider_error")):
            calls: list[int] = []

            def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float, *, status: int = status, calls: list[int] = calls) -> HttpResponse:
                calls.append(1)
                return HttpResponse(status, {}, json.dumps({"error": f"error-{status}"}).encode())

            with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
                cfg = config(Path(directory))
                provider = BrightDataProvider(cfg, client=JsonClient(timeout=1, retries=4, transport=transport, sleep=lambda _: None))
                result = provider.discover("smallbusiness", None, cfg)

            self.assertEqual(len(calls), 1, status)
            self.assertEqual(len(result.gaps), 1, status)
            self.assertTrue(any(gap.reason == expected_reason for gap in result.gaps), status)
            metadata = result.request_records[0].metadata
            self.assertEqual(metadata["raw_payload"], {"error": f"error-{status}"})
            self.assertEqual(metadata["accounting"]["requested_inputs"], 2)
            self.assertEqual(metadata["accounting"]["returned_records"], 0)

    def test_cursor_and_full_comments_are_explicitly_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            root = Path(directory)
            cfg = config(root, comments="full")
            provider = BrightDataProvider(cfg, client=type("Client", (), {})())
            cursor_result = provider.discover("smallbusiness", "opaque", cfg)
            plan = provider.plan(cfg, 5, mode="refresh")

        self.assertEqual(cursor_result.gaps[0].reason, "unsupported")
        self.assertIn("Full comment expansion is unsupported", " ".join(plan.notes))
        self.assertNotIn("collect_comments_bounded", provider.status().capabilities)


if __name__ == "__main__":
    unittest.main()
