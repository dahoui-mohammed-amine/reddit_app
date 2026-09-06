from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reddit_ingestion.config import Config
from reddit_ingestion.db import Database
from reddit_ingestion.models import PageResult, PostSnapshot
from reddit_ingestion.normalize import parse_post
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

    def test_brightdata_deletion_like_fields_remain_report_only(self) -> None:
        raw = post_record("smallbusiness")
        raw.update({"description": "[deleted]", "user_posted": "[removed]"})
        raw["comments"] = [{
            "comment_id": "c",
            "post_id": "p",
            "user_posted": "[deleted]",
            "comment": "[removed]",
            "parent_id": "t3_p",
        }]
        post = parse_post(raw, default_subreddit="smallbusiness", observed_at=OBSERVED)

        self.assertFalse(post.deleted)
        self.assertFalse(post.removed)
        self.assertFalse(post.deletion_known)
        self.assertFalse(post.removal_known)
        self.assertFalse(post.comments[0].deleted)
        self.assertFalse(post.comments[0].removed)
        self.assertFalse(post.comments[0].deletion_known)
        self.assertFalse(post.comments[0].removal_known)

        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "reddit.sqlite3")
            try:
                run_id = db.start_run("brightdata", "discover", {}, OBSERVED)
                with db.transaction():
                    db.save_page(
                        run_id,
                        "brightdata",
                        "smallbusiness",
                        PageResult([post], None, None, OBSERVED, post.url, 200, "brightdata", metadata={"comments_expanded": True}),
                    )
                saved_post = db.connection.execute("SELECT title, body, author, deleted, removed FROM posts WHERE post_id='p'").fetchone()
                saved_comment = db.connection.execute("SELECT body, author, deleted, removed FROM comments WHERE comment_id='c'").fetchone()
            finally:
                db.close()

        self.assertEqual(tuple(saved_post), ("A post", "[deleted]", "[removed]", 0, 0))
        self.assertEqual(tuple(saved_comment), ("[removed]", "[deleted]", 0, 0))

    def test_brightdata_explicit_state_markers_are_report_only(self) -> None:
        raw = post_record("smallbusiness")
        raw.update({"deleted": True, "removed": True})

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [raw], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            root = Path(directory)
            cfg = config(root)
            page = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)
            db = Database(cfg.database_path)
            try:
                run_id = db.start_run("brightdata", "discover", {}, OBSERVED)
                with db.transaction():
                    db.save_page(run_id, "brightdata", "smallbusiness", page)
                saved = db.connection.execute("SELECT title, body, author, deleted, removed FROM posts WHERE post_id='p'").fetchone()
            finally:
                db.close()

        self.assertFalse(page.posts[0].deleted)
        self.assertFalse(page.posts[0].removed)
        self.assertTrue(any(gap.reason == "deleted" and gap.entity_id == "p" for gap in page.gaps))
        self.assertTrue(any(gap.reason == "removed" and gap.entity_id == "p" for gap in page.gaps))
        self.assertEqual(tuple(saved), ("A post", "A description", "author", 0, 0))

    def test_brightdata_content_markers_are_report_only(self) -> None:
        raw = post_record("smallbusiness")
        raw.update({"description": "[deleted]", "user_posted": "[removed]", "comment": "deleted"})

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [raw], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory))
            page = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

        self.assertFalse(page.posts[0].deleted)
        self.assertFalse(page.posts[0].removed)
        self.assertTrue(any(gap.reason == "deleted" and gap.entity_id == "p" for gap in page.gaps))
        self.assertTrue(any(gap.reason == "removed" and gap.entity_id == "p" for gap in page.gaps))
        self.assertTrue(page.metadata["checkpoint_deferred"])

    def test_sparse_records_are_valid_but_deleted_like_partial_errors_are_unavailable(self) -> None:
        sparse = {"post_id": "sparse", "url": "https://www.reddit.com/r/smallbusiness/comments/sparse/title/", "num_comments": 0}
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

        for payload in ({"data": [post_record("smallbusiness")]}, {"records": [post_record("smallbusiness")]}, {"errors": []}, {"unexpected": True}):
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

    def test_refresh_rejects_contradictory_post_url_identity(self) -> None:
        requested = "https://www.reddit.com/r/smallbusiness/comments/p/title/"

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [{"post_id": "p", "url": "https://www.reddit.com/r/smallbusiness/comments/q/other-title/"}], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory))
            result = BrightDataProvider(cfg, client=Client()).refresh_posts([PostSnapshot("p", "t3_p", "smallbusiness", permalink=requested)], cfg)

        self.assertEqual(result.posts, [])
        self.assertTrue(any(gap.reason == "provider_error" for gap in result.gaps))
        self.assertEqual(result.request_records[0].metadata["accounting"]["provider_error_count"], 1)

    def test_discovery_normalization_rejection_is_counted(self) -> None:
        raw = post_record("smallbusiness")
        raw["title"] = ["not scalar"]

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [raw], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory), subreddits=("smallbusiness",))
            result = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

        self.assertEqual(result.posts, [])
        self.assertTrue(any(gap.reason == "provider_error" for gap in result.gaps))
        self.assertEqual(result.request_records[0].metadata["accounting"]["provider_error_count"], 1)

    def test_auth_status_precedes_private_or_restricted_error_text(self) -> None:
        for status in (401, 403):
            class Client:
                def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                    return {"error": "private or restricted"}, HttpResponse(status, {}, b""), "request"

            with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
                cfg = config(Path(directory))
                result = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

            self.assertTrue(any(gap.reason == "provider_error" for gap in result.gaps), status)
            self.assertFalse(any(gap.reason == "unavailable" for gap in result.gaps), status)

    def test_refresh_error_with_deleted_url_slug_remains_report_only(self) -> None:
        requested = "https://www.reddit.com/r/smallbusiness/comments/p/deleted-project/"
        raw_error = {"url": requested, "error": "temporary provider failure"}

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [raw_error], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory))
            result = BrightDataProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink=requested)], cfg
            )

        self.assertEqual(result.posts, [])
        self.assertTrue(any(gap.reason == "provider_error" and gap.entity_id == "p" for gap in result.gaps))

    def test_non_reddit_urls_are_rejected_for_discovery_and_refresh(self) -> None:
        external_url = "https://evil.example/r/smallbusiness/comments/p/deleted-project/"

        class DiscoveryClient:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [{"post_id": "p", "url": external_url, "community_name": "smallbusiness"}], HttpResponse(200, {}, b""), "request"

        class RefreshClient:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [{"post_id": "p", "url": external_url}], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory), subreddits=("smallbusiness",))
            discovered = BrightDataProvider(cfg, client=DiscoveryClient()).discover("smallbusiness", None, cfg)
            refreshed = BrightDataProvider(cfg, client=RefreshClient()).refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
            )

        self.assertEqual(discovered.posts, [])
        self.assertTrue(any(gap.reason == "provider_error" for gap in discovered.gaps))
        self.assertEqual(refreshed.posts, [])
        self.assertTrue(any(gap.reason == "provider_error" for gap in refreshed.gaps))

    def test_bare_host_like_urls_are_rejected(self) -> None:
        bare_host_url = "evil.example/r/smallbusiness/comments/p/title/"

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [{"post_id": "p", "url": bare_host_url, "community_name": "smallbusiness"}], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory), subreddits=("smallbusiness",))
            result = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

        self.assertEqual(result.posts, [])
        self.assertTrue(any(gap.reason == "provider_error" for gap in result.gaps))

    def test_refresh_normalizes_provider_subreddit_to_requested_value(self) -> None:
        raw = post_record("smallbusiness")
        raw["community_name"] = "SmallBusiness"

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [raw], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory), subreddits=("smallbusiness",))
            result = BrightDataProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
            )

        self.assertEqual(result.posts[0].subreddit, "smallbusiness")

    def test_cross_subreddit_identity_is_rejected(self) -> None:
        discovery_record = {
            "post_id": "p",
            "url": "https://www.reddit.com/r/smallbusiness/comments/p/title/",
            "community_url": "https://www.reddit.com/r/freelance/",
        }

        class DiscoveryClient:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [discovery_record], HttpResponse(200, {}, b""), "request"

        class RefreshClient:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [{"post_id": "p", "url": "https://www.reddit.com/r/freelance/comments/p/title/"}], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory), subreddits=("smallbusiness", "freelance"))
            discovered = BrightDataProvider(cfg, client=DiscoveryClient()).discover("smallbusiness", None, cfg)
            refreshed = BrightDataProvider(cfg, client=RefreshClient()).refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
            )

        self.assertEqual(discovered.posts, [])
        self.assertTrue(any(gap.reason == "provider_error" for gap in discovered.gaps))
        self.assertEqual(refreshed.posts, [])
        self.assertTrue(any(gap.reason == "provider_error" for gap in refreshed.gaps))

    def test_null_provider_error_payload_is_retained(self) -> None:
        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            return HttpResponse(401, {}, b"null")

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory))
            provider = BrightDataProvider(cfg, client=JsonClient(timeout=1, retries=0, transport=transport, sleep=lambda _: None))
            result = provider.discover("smallbusiness", None, cfg)

        metadata = result.request_records[0].metadata
        self.assertIn("raw_payload", metadata)
        self.assertIsNone(metadata["raw_payload"])
        self.assertIn("provider_error", metadata)
        self.assertIsNone(metadata["provider_error"])
        self.assertTrue(metadata["provider_payload_present"])

    def test_unmatched_specific_discovery_error_defers_checkpoint(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [{"url": "https://www.reddit.com/r/other/", "error": "provider failure"}], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory), subreddits=("smallbusiness",))
            result = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

        self.assertTrue(any(gap.reason == "provider_error" for gap in result.gaps))
        self.assertTrue(result.metadata["checkpoint_deferred"])
        self.assertEqual(result.request_records[0].metadata["accounting"]["provider_error_count"], 1)

    def test_present_falsy_discovery_urls_are_rejected(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [{"post_id": "p", "url": self.bad_url, "community_url": "https://www.reddit.com/r/smallbusiness/"}], HttpResponse(200, {}, b""), "request"

        for bad_url in (False, None, ""):
            with self.subTest(bad_url=bad_url):
                Client.bad_url = bad_url
                with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
                    cfg = config(Path(directory), subreddits=("smallbusiness",))
                    result = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

                self.assertEqual(result.posts, [])
                self.assertTrue(any(gap.reason == "provider_error" for gap in result.gaps))
                self.assertTrue(result.metadata["checkpoint_deferred"])

    def test_discovery_counts_unusable_array_items_as_provider_errors(self) -> None:
        valid_url = post_record("smallbusiness")["url"]

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [None, {"url": valid_url}], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory), subreddits=("smallbusiness",))
            result = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

        self.assertEqual(result.posts, [])
        self.assertTrue(result.metadata["checkpoint_deferred"])
        self.assertEqual(result.request_records[0].metadata["accounting"]["returned_records"], 2)
        self.assertEqual(result.request_records[0].metadata["accounting"]["provider_error_count"], 2)

    def test_discovery_requires_a_usable_reddit_post_url(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [{"post_id": "p", "community_name": "smallbusiness"}], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory), subreddits=("smallbusiness",))
            result = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

        self.assertEqual(result.posts, [])
        self.assertTrue(any(gap.reason == "provider_error" for gap in result.gaps))
        self.assertTrue(result.metadata["checkpoint_deferred"])
        self.assertEqual(result.request_records[0].metadata["accounting"]["returned_records"], 1)
        self.assertEqual(result.request_records[0].metadata["accounting"]["provider_error_count"], 1)

    def test_refresh_retains_valid_records_and_reports_malformed_items(self) -> None:
        valid = post_record("smallbusiness")

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [valid, None, {}], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory))
            result = BrightDataProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
            )

        self.assertEqual([post.post_id for post in result.posts], ["p"])
        self.assertTrue(any(gap.reason == "provider_error" and "not an object" in (gap.detail or "") for gap in result.gaps))
        self.assertTrue(any(gap.reason == "provider_error" and "missing post identity" in (gap.detail or "") for gap in result.gaps))
        self.assertTrue(result.metadata["request_failed"])
        self.assertEqual(result.metadata["malformed_record_count"], 2)
        self.assertEqual(result.request_records[0].metadata["accounting"]["returned_records"], 1)
        self.assertEqual(result.request_records[0].metadata["accounting"]["provider_error_count"], 2)

    def test_refresh_reports_provider_errors_alongside_valid_records(self) -> None:
        valid = post_record("smallbusiness")
        provider_error = {"url": valid["url"], "error": "partial provider failure"}

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [valid, provider_error], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory))
            result = BrightDataProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
            )

        self.assertEqual([post.post_id for post in result.posts], ["p"])
        self.assertTrue(any(gap.reason == "provider_error" and gap.entity_id == "p" for gap in result.gaps))

    def test_refresh_reports_unmatched_provider_errors_at_batch_scope(self) -> None:
        unmatched = {"url": "https://www.reddit.com/r/other/comments/q/title/", "error": "provider failure"}

        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return [unmatched], HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory), subreddits=("smallbusiness",))
            result = BrightDataProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
            )

        self.assertTrue(any(gap.reason == "provider_error" and gap.entity_id is None for gap in result.gaps))

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

    def test_202_array_is_deferred_without_discovery_or_refresh_records(self) -> None:
        calls: list[int] = []

        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            calls.append(1)
            return HttpResponse(202, {}, json.dumps([post_record("smallbusiness")]).encode())

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            root = Path(directory)
            cfg = config(root)
            client = JsonClient(timeout=1, retries=2, transport=transport, sleep=lambda _: None)
            provider = BrightDataProvider(cfg, client=client)
            page = provider.discover("smallbusiness", None, cfg)
            result = provider.refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(page.posts, [])
        self.assertTrue(page.metadata["checkpoint_deferred"])
        self.assertTrue(any(gap.reason == "unsupported" for gap in page.gaps))
        self.assertEqual(page.request_records[0].metadata["raw_payload"], [post_record("smallbusiness")])
        self.assertEqual(page.request_records[0].metadata["accounting"]["returned_records"], 0)
        self.assertEqual(result.posts, [])
        self.assertTrue(any(gap.reason == "unsupported" for gap in result.gaps))
        self.assertEqual(result.request_records[0].metadata["raw_payload"], [post_record("smallbusiness")])
        self.assertEqual(result.request_records[0].metadata["accounting"]["returned_records"], 0)

    def test_202_object_envelopes_are_malformed(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return self.payload, HttpResponse(202, {}, b""), "request"

        for payload in ({"data": []}, {"records": []}, {"errors": []}, {"snapshot_id": True}):
            with self.subTest(payload=payload):
                Client.payload = payload
                with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
                    cfg = config(Path(directory))
                    page = BrightDataProvider(cfg, client=Client()).discover("smallbusiness", None, cfg)

                self.assertEqual(page.posts, [])
                self.assertTrue(any(gap.reason == "provider_error" and "malformed" in (gap.detail or "") for gap in page.gaps))
                self.assertFalse(any(gap.reason == "unsupported" for gap in page.gaps))
                self.assertTrue(page.metadata["checkpoint_deferred"])
                self.assertEqual(page.request_records[0].metadata["raw_payload"], payload)
                self.assertEqual(page.request_records[0].metadata["accounting"]["provider_error_count"], 1)

    def test_refresh_malformed_envelopes_account_provider_errors(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None):
                return {"data": []}, HttpResponse(self.status, {}, b""), "request"

        for status in (200, 202):
            with self.subTest(status=status):
                Client.status = status
                with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
                    cfg = config(Path(directory))
                    result = BrightDataProvider(cfg, client=Client()).refresh_posts(
                        [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
                    )

                self.assertTrue(any(gap.reason == "provider_error" for gap in result.gaps))
                self.assertTrue(result.metadata["request_failed"])
                self.assertEqual(result.metadata["provider_error_count"], 1)
                self.assertEqual(result.request_records[0].metadata["accounting"]["provider_error_count"], 1)

    def test_http_errors_are_not_retried_and_preserve_provider_error(self) -> None:
        for status, expected_reason in ((400, "provider_error"), (401, "provider_error"), (403, "provider_error"), (404, "provider_error"), (429, "provider_error"), (500, "provider_error")):
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
            self.assertEqual(result.metadata["provider_error_count"], 1, status)
            self.assertEqual(result.metadata["accounting"]["provider_error_count"], 1, status)

    def test_404_target_payload_is_unavailable(self) -> None:
        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            return HttpResponse(404, {}, b'{"error":"post not found"}')

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory))
            provider = BrightDataProvider(cfg, client=JsonClient(timeout=1, retries=0, transport=transport, sleep=lambda _: None))
            result = provider.discover("smallbusiness", None, cfg)

        self.assertTrue(any(gap.reason == "unavailable" for gap in result.gaps))

    def test_refresh_http_error_marks_result_failed(self) -> None:
        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            return HttpResponse(500, {}, b'{"error":"provider failure"}')

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory))
            provider = BrightDataProvider(cfg, client=JsonClient(timeout=1, retries=0, transport=transport, sleep=lambda _: None))
            result = provider.refresh_posts(
                [PostSnapshot("p", "t3_p", "smallbusiness", permalink="/r/smallbusiness/comments/p/title/")], cfg
            )

        self.assertTrue(result.metadata["request_failed"])

    def test_non_json_error_text_cannot_imply_unavailable(self) -> None:
        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            return HttpResponse(500, {}, b"/comments/p/deleted-project/")

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "secret"}):
            cfg = config(Path(directory))
            provider = BrightDataProvider(cfg, client=JsonClient(timeout=1, retries=0, transport=transport, sleep=lambda _: None))
            result = provider.discover("smallbusiness", None, cfg)

        self.assertTrue(any(gap.reason == "provider_error" for gap in result.gaps))
        self.assertFalse(any(gap.reason == "unavailable" for gap in result.gaps))

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
