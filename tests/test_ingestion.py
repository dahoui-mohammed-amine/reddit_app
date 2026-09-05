from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest import mock

from reddit_ingestion.cli import main as cli_main
from reddit_ingestion.comments import apply_comment_policy
from reddit_ingestion.config import Config
from reddit_ingestion.db import Database
from reddit_ingestion.models import CommentSnapshot, Gap, PageResult, PostSnapshot, ProviderStatus, RefreshResult, RequestRecord
from reddit_ingestion.normalize import parse_comment, parse_post
from reddit_ingestion.providers import FetchLayerProvider, FixtureProvider, JsonClient, HttpResponse, ProviderError, RedditApisProvider
from reddit_ingestion.runner import check_live_access, plan_for, run_once


FIXTURE = Path(__file__).parents[1] / "fixtures" / "sample.json"


def config(tmp: Path, *, comments: str = "bounded", limit: int = 20, provider: str = "fixture") -> Config:
    return Config(
        subreddits=("freelance", "smallbusiness", "SaaS"),
        database_path=tmp / "reddit.sqlite3",
        provider=provider,
        fixture_path=FIXTURE,
        listing_limit=100,
        max_discovery_pages=1,
        refresh_interval_minutes=1,
        max_refresh_posts=100,
        comments_mode=comments,
        comment_depth=1,
        comment_limit=limit,
        request_timeout_seconds=1,
        max_retries=2,
    )


class IngestionTests(unittest.TestCase):
    def test_normalization_preserves_ids_relationships_and_metrics(self) -> None:
        post = parse_post(
            {
                "name": "t3_abc",
                "subreddit": "SaaS",
                "permalink": "/r/SaaS/comments/abc/title/",
                "score": 7,
                "ups": 8,
                "upvote_ratio": 0.9,
                "num_comments": 2,
                "comments": [{"name": "t1_xyz", "parent_id": "t3_abc", "body": "hello", "score": 2}],
            },
            observed_at="2026-09-05T00:00:00Z",
        )
        self.assertEqual(post.post_id, "abc")
        self.assertEqual(post.fullname, "t3_abc")
        self.assertEqual(post.permalink, "https://www.reddit.com/r/SaaS/comments/abc/title/")
        self.assertEqual(post.comments[0].comment_id, "xyz")
        self.assertEqual(post.comments[0].parent_id, "t3_abc")
        self.assertEqual(post.score, 7)
        self.assertEqual(post.num_comments, 2)

    def test_normalization_recognizes_deleted_content_markers(self) -> None:
        post = parse_post({"id": "deleted", "author": None, "selftext": "[deleted]"})
        comment = parse_comment({"id": "comment", "author": None, "body": "[deleted]"}, post=post, observed_at=post.observed_at)
        self.assertTrue(post.deleted)
        self.assertTrue(comment.deleted)

    def test_fixture_discovery_collects_comments_and_preserves_request_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            run_once(db, FixtureProvider(FIXTURE), config(root), "discover")
            requests = db.connection.execute("SELECT request_id, billed FROM requests ORDER BY request_event_id").fetchall()
            self.assertEqual(len(requests), 3)
            self.assertEqual(db.counts()["comments"], 2)
            self.assertTrue(all(row["request_id"] == "fixture" and row["billed"] == 0 for row in requests))
            db.close()

    def test_discovery_resume_and_idempotent_current_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "db.sqlite3")
            provider = FixtureProvider(FIXTURE)
            cfg = config(Path(directory))
            first = run_once(db, provider, cfg, "discover")
            self.assertEqual(first.discovered, 3)
            checkpoint = db.checkpoint("freelance")
            self.assertEqual(checkpoint["cursor"], "t3_freelance_next")
            resumed_page = provider.discover("freelance", "t3_freelance_next", cfg)
            self.assertEqual([post.post_id for post in resumed_page.posts], ["freelance002"])
            second = run_once(db, provider, cfg, "discover")
            self.assertEqual(second.discovered, 4)
            self.assertEqual(db.counts()["posts"], 4)
            self.assertEqual(db.counts()["post_observations"], 7)
            db.close()

    def test_historical_refresh_and_bounded_comment_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            provider = FixtureProvider(FIXTURE)
            cfg = config(root, comments="bounded", limit=1)
            run_once(db, provider, cfg, "discover")
            db.connection.execute("UPDATE posts SET observed_at = '2020-01-01T00:00:00Z'")
            db.connection.commit()
            summary = run_once(db, provider, cfg, "refresh")
            self.assertEqual(summary.refreshed, 3)
            self.assertGreaterEqual(db.counts()["post_observations"], 6)
            self.assertEqual(db.counts()["comments"], 1)
            self.assertGreaterEqual(db.counts()["gaps"], 1)
            scores = [row[0] for row in db.connection.execute("SELECT score FROM post_observations WHERE post_id='freelance001' ORDER BY observation_id")]
            self.assertEqual(scores, [12, 15])
            db.close()

    def test_partial_comment_results_do_not_delete_existing_comments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            initial = PostSnapshot(
                "p",
                "t3_p",
                "freelance",
                title="original title",
                body="original body",
                author="original author",
                score=10,
                num_comments=2,
                observed_at=observed,
                comments=[
                    CommentSnapshot("c1", "t1_c1", "p", parent_id="t3_p", author="commenter", body="one", score=1, observed_at=observed),
                    CommentSnapshot("c2", "t1_c2", "p", body="two", observed_at=observed),
                ],
            )
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([initial], None, None, observed, "fixture://p", 200, "fixture"))
            partial = PostSnapshot(
                "p",
                "t3_p",
                "freelance",
                score=11,
                num_comments=2,
                observed_at=observed,
                comments=[CommentSnapshot("c1", "t1_c1", "p", score=2, observed_at=observed)],
            )
            with db.transaction():
                db.save_refresh(
                    run_id,
                    "fixture",
                    RefreshResult([partial], observed, "refresh", 200, metadata={"comments_expanded": True}),
                )
            comment_ids = [row[0] for row in db.connection.execute("SELECT comment_id FROM comments ORDER BY comment_id")]
            self.assertEqual(comment_ids, ["c1", "c2"])
            post_row = db.connection.execute("SELECT title, body, author, score FROM posts WHERE post_id='p'").fetchone()
            self.assertEqual(tuple(post_row), ("original title", "original body", "original author", 11))
            comment_row = db.connection.execute("SELECT parent_id, author, body, score FROM comments WHERE comment_id='c1'").fetchone()
            self.assertEqual(tuple(comment_row), ("t3_p", "commenter", "one", 2))
            db.close()

    def test_full_refresh_defers_omitted_comments_without_deletion_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            initial = PostSnapshot(
                "p",
                "t3_p",
                "freelance",
                num_comments=2,
                observed_at=observed,
                comments=[
                    CommentSnapshot("c1", "t1_c1", "p", body="one", observed_at=observed),
                    CommentSnapshot("c2", "t1_c2", "p", body="two", observed_at=observed),
                ],
            )
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([initial], None, None, observed, "fixture://p", 200, "fixture"))
            full = PostSnapshot(
                "p",
                "t3_p",
                "freelance",
                num_comments=1,
                observed_at=observed,
                comments=[CommentSnapshot("c1", "t1_c1", "p", body="updated", observed_at=observed)],
            )
            result = RefreshResult([full], observed, "refresh", 200, metadata={"comments_mode": "full", "comments_expanded": True})
            with db.transaction():
                db.save_refresh(run_id, "fixture", result)
            row = db.connection.execute("SELECT body FROM comments WHERE comment_id='c2'").fetchone()
            self.assertEqual(row[0], "two")
            self.assertTrue(any(gap.entity_type == "comment" and gap.reason == "unexpanded" for gap in result.gaps))
            metadata = json.loads(db.connection.execute("SELECT metadata_json FROM post_observations ORDER BY observation_id DESC LIMIT 1").fetchone()[0])
            self.assertFalse(metadata["comments_expanded"])
            db.close()

    def test_blocked_listing_records_gap_without_advancing_checkpoint(self) -> None:
        class BlockedProvider:
            name = "fixture"

            def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
                return PageResult(
                    [],
                    cursor,
                    "blocked-next",
                    "2026-09-05T00:00:00Z",
                    "fixture://blocked",
                    200,
                    "fixture-request",
                    gaps=[Gap("listing", "blocked", subreddit=subreddit)],
                    metadata={"blocked": True},
                )

        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "db.sqlite3")
            summary = run_once(db, BlockedProvider(), config(Path(directory)), "discover")
            self.assertEqual(summary.gaps, 3)
            self.assertIsNone(db.checkpoint("freelance"))
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM gaps WHERE reason='blocked'").fetchone()[0], 3)
            db.close()

    def test_provider_truncation_records_gap_without_advancing_checkpoint(self) -> None:
        observed = "2026-09-05T00:00:00Z"
        page = PageResult(
            [],
            "c1",
            "c2",
            observed,
            "fixture://truncated",
            200,
            "fixture-request",
            metadata={"listing_status": "truncated"},
        )

        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "db.sqlite3")
            run_id = db.start_run("fixture", "discover", {}, observed)
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", page)
            self.assertIsNone(db.checkpoint("freelance"))
            self.assertEqual(db.connection.execute("SELECT reason FROM gaps").fetchone()[0], "truncated")
            db.close()

    def test_provider_truncation_stops_followup_pages(self) -> None:
        class Provider:
            name = "fixture"

            def __init__(self) -> None:
                self.calls: list[tuple[str, str | None]] = []

            def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
                self.calls.append((subreddit, cursor))
                if cursor is not None:
                    raise AssertionError("truncated pages must stop pagination")
                return PageResult(
                    [],
                    cursor,
                    "next",
                    "2026-09-05T00:00:00Z",
                    "fixture://truncated",
                    200,
                    "request",
                    metadata={"listing_status": "truncated"},
                    request_records=[RequestRecord("request", "discover", 200)],
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            provider = Provider()
            cfg = replace(config(root), subreddits=("freelance",), max_discovery_pages=2)
            run_once(db, provider, cfg, "discover")
            self.assertEqual(provider.calls, [("freelance", None)])
            self.assertIsNone(db.checkpoint("freelance"))
            db.close()

    def test_partial_snapshot_preserves_deletion_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            deleted = parse_post({"id": "p", "title": "gone", "selftext": "[deleted]"}, default_subreddit="freelance", observed_at=observed)
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([deleted], None, None, observed, "fixture://p", 200, "fixture"))
            partial = parse_post({"id": "p", "score": 5, "num_comments": 0}, default_subreddit="freelance", observed_at=observed)
            with db.transaction():
                db.save_refresh(run_id, "fixture", RefreshResult([partial], observed, "refresh", 200))
            row = db.connection.execute("SELECT deleted, score FROM posts WHERE post_id='p'").fetchone()
            self.assertEqual(tuple(row), (1, 5))
            self.assertEqual(db.purge_deleted_content(), 1)
            cleared = parse_post({"id": "p", "deleted": False, "score": 6, "num_comments": 0}, default_subreddit="freelance", observed_at=observed)
            with db.transaction():
                db.save_refresh(run_id, "fixture", RefreshResult([cleared], observed, "refresh", 200))
            self.assertEqual(db.connection.execute("SELECT deleted, score FROM posts WHERE post_id='p'").fetchone()[0:2], (0, 6))
            self.assertEqual(db.purge_deleted_content(), 0)
            db.close()

    def test_cache_observation_timestamp_is_persisted_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("redditapis", "discover", {}, observed)
            record = RequestRecord("request", "discover", 200, "cached", "2026-09-05T00:00:01Z", False)
            page = PageResult([], None, None, observed, "https://provider.test", 200, "request", request_records=[record])
            with db.transaction():
                db.save_page(run_id, "redditapis", "freelance", page)
            timestamp = db.connection.execute("SELECT cache_observed_at FROM requests WHERE request_id='request'").fetchone()[0]
            self.assertEqual(timestamp, "2026-09-05T00:00:01Z")
            db.close()

    def test_bounded_policy_clears_stale_expansion_metadata(self) -> None:
        class Provider:
            name = "fixture"

            def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
                post = PostSnapshot(
                    "p",
                    "t3_p",
                    subreddit,
                    num_comments=1,
                    observed_at="2026-09-05T00:00:00Z",
                    comments=[CommentSnapshot("c", "t1_c", "p", depth=2, observed_at="2026-09-05T00:00:00Z")],
                )
                return PageResult(
                    [post],
                    cursor,
                    None,
                    post.observed_at,
                    "fixture://p",
                    200,
                    "request",
                    metadata={"comments_expanded": True},
                    request_records=[RequestRecord("request", "discover", 200)],
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            cfg = replace(config(root), subreddits=("freelance",))
            run_once(db, Provider(), cfg, "discover")
            metadata = json.loads(db.connection.execute("SELECT metadata_json FROM post_observations").fetchone()[0])
            self.assertFalse(metadata["comments_expanded"])
            db.close()

    def test_full_comment_cursor_cycle_stops_before_repeating_request(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                self.urls.append(url)
                if "/by_id/" in url:
                    return {"posts": [{"id": "p", "name": "t3_p", "num_comments": 3}]}, HttpResponse(200, {}, b""), "post-request"
                if "after=A" in url:
                    return {"comments": [{"id": "c2", "name": "t1_c2", "body": "two"}], "after": "B"}, HttpResponse(200, {}, b""), "comment-request-a"
                if "after=B" in url:
                    return {"comments": [{"id": "c3", "name": "t1_c3", "body": "three"}], "after": "A"}, HttpResponse(200, {}, b""), "comment-request-b"
                return {"comments": [{"id": "c1", "name": "t1_c1", "body": "one"}], "after": "A"}, HttpResponse(200, {}, b""), "comment-request-first"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis", comments="full")
            client = Client()
            result = RedditApisProvider(cfg, client=client).refresh_posts([PostSnapshot("p", "t3_p", "freelance")], cfg)
            self.assertEqual(len([url for url in client.urls if "/comments" in url]), 3)
            self.assertEqual(len(result.posts[0].comments), 3)
            self.assertTrue(any(gap.reason == "unexpanded" for gap in result.gaps))

    def test_duplicate_full_comment_pages_are_counted_once(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/by_id/" in url:
                    return {"posts": [{"id": "p", "name": "t3_p", "num_comments": 2}]}, HttpResponse(200, {}, b""), "post-request"
                comment = {"id": "c1", "name": "t1_c1", "body": "one"}
                if "after=A" in url:
                    return {"comments": [comment]}, HttpResponse(200, {}, b""), "comment-request-second"
                return {"comments": [comment], "after": "A"}, HttpResponse(200, {}, b""), "comment-request-first"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis", comments="full")
            result = RedditApisProvider(cfg, client=Client()).refresh_posts([PostSnapshot("p", "t3_p", "freelance")], cfg)
            self.assertEqual(len(result.posts[0].comments), 1)
            self.assertTrue(any(gap.reason == "unexpanded" for gap in result.gaps))

    def test_missing_comment_collection_is_an_unexpanded_gap(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/by_id/" in url:
                    return {"posts": [{"id": "p", "name": "t3_p"}]}, HttpResponse(200, {}, b""), "post-request"
                return {}, HttpResponse(200, {}, b""), "comment-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis")
            result = RedditApisProvider(cfg, client=Client()).refresh_posts([PostSnapshot("p", "t3_p", "freelance")], cfg)
            self.assertTrue(any(gap.reason == "unexpanded" for gap in result.gaps))

    def test_fetchlayer_missing_comment_collection_is_an_unexpanded_gap(self) -> None:
        class Client:
            def request(self, *args: object, **kwargs: object) -> tuple[dict[str, object], HttpResponse, str]:
                return {"id": "p", "name": "t3_p"}, HttpResponse(200, {}, b""), "refresh-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"FETCHLAYER_API_KEY": "test"}):
            cfg = config(Path(directory), provider="fetchlayer")
            result = FetchLayerProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "freelance", permalink="/r/freelance/comments/p/post/")], cfg
            )
            self.assertTrue(any(gap.reason == "unexpanded" for gap in result.gaps))

    def test_deletion_clears_mutable_content_but_retains_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "db.sqlite3")
            post = PostSnapshot("deleted", "t3_deleted", "freelance", "title", "body", "author", "/r/x", None, "2020", 1, 1, 1.0, 0, deleted=True, observed_at="2026-09-05T00:00:00Z")
            page = PageResult([post], None, None, post.observed_at, "fixture://x", 200, "fixture")
            run_id = db.start_run("fixture", "discover", {}, post.observed_at)
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", page)
            row = db.connection.execute("SELECT post_id, title, body, author, deleted FROM posts").fetchone()
            self.assertEqual(tuple(row), ("deleted", None, None, None, 1))
            db.close()

    def test_json_client_retries_retryable_errors(self) -> None:
        calls: list[int] = []

        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            calls.append(len(calls))
            if len(calls) < 3:
                return HttpResponse(503, {}, b'{"error":"temporary"}')
            return HttpResponse(200, {}, b'{"ok":true}')

        client = JsonClient(timeout=1, retries=2, transport=transport, sleep=lambda _: None)
        payload, response, _ = client.request("GET", "https://example.test", headers={})
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(response.status, 200)
        self.assertEqual(len(calls), 3)

    def test_retried_provider_request_persists_each_attempt_record(self) -> None:
        calls: list[int] = []

        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            calls.append(len(calls))
            if "/by_id/" in url and len(calls) == 1:
                return HttpResponse(503, {}, b'{"error":"temporary"}')
            if "/by_id/" in url:
                return HttpResponse(200, {}, b'{"posts":[{"id":"p","name":"t3_p","score":3,"num_comments":0}]}')
            return HttpResponse(200, {}, b'{"comments":[]}')

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis")
            client = JsonClient(timeout=1, retries=1, transport=transport, sleep=lambda _: None)
            result = RedditApisProvider(cfg, client=client).refresh_posts([PostSnapshot("p", "t3_p", "freelance")], cfg)
            refresh_records = [record for record in result.request_records if record.operation == "refresh"]
            self.assertEqual(len(refresh_records), 2)
            self.assertEqual([record.metadata["attempt"] for record in refresh_records], [1, 2])
            self.assertEqual([record.metadata["attempt_count"] for record in refresh_records], [2, 2])

    def test_failed_discovery_persists_each_retry_attempt(self) -> None:
        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            return HttpResponse(503, {}, b'{"error":"temporary"}')

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            cfg = replace(config(root, provider="redditapis"), subreddits=("freelance",))
            client = JsonClient(timeout=1, retries=2, transport=transport, sleep=lambda _: None)
            summary = run_once(db, RedditApisProvider(cfg, client=client), cfg, "discover")
            records = db.connection.execute("SELECT metadata_json FROM requests ORDER BY request_event_id").fetchall()
            self.assertEqual(summary.requests, 3)
            self.assertEqual(len(records), 3)
            self.assertEqual([json.loads(row[0])["attempt"] for row in records], [1, 2, 3])
            db.close()

    def test_resume_plan_includes_newest_page_poll(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            cfg = replace(config(root, provider="redditapis"), subreddits=("freelance",), max_discovery_pages=2)
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("redditapis", "discover", {}, observed)
            page = PageResult([], None, "saved-cursor", observed, "https://provider.test", 200, "request")
            with db.transaction():
                db.save_page(run_id, "redditapis", "freelance", page)
            plan = plan_for(db, RedditApisProvider(cfg), cfg)
            self.assertEqual(plan.discovery_requests, 3)
            self.assertEqual(plan.estimated_requests, 909)
            db.close()

    def test_full_redditapis_plan_does_not_claim_bounded_comment_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis", comments="full")
            plan = RedditApisProvider(cfg).plan(cfg, 100)
            self.assertIsNone(plan.estimated_requests)
            self.assertIsNone(plan.estimated_units)

    def test_json_client_paces_successive_requests(self) -> None:
        delays: list[float] = []
        client = JsonClient(timeout=1, retries=0, transport=lambda *args: HttpResponse(200, {}, b"{}"), sleep=delays.append, min_interval_seconds=1)
        client.request("GET", "https://example.test/one", headers={})
        client.request("GET", "https://example.test/two", headers={})
        self.assertTrue(any(delay > 0 for delay in delays))

    def test_json_client_reports_terminal_error_and_billing(self) -> None:
        client = JsonClient(timeout=1, retries=0, transport=lambda *args: HttpResponse(429, {}, b'{"error":"slow"}'), sleep=lambda _: None)
        with self.assertRaises(ProviderError) as context:
            client.request("GET", "https://example.test", headers={})
        self.assertTrue(context.exception.retryable)
        self.assertTrue(context.exception.billed)

    def test_live_provider_requires_access_and_explicit_cost_consent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {}, clear=True):
                provider = RedditApisProvider(config(Path(directory), provider="redditapis"))
                self.assertFalse(provider.status().available)
                with self.assertRaises(ProviderError):
                    check_live_access(provider, allow_paid=False)

    def test_run_once_requires_explicit_cost_consent(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            try:
                provider = RedditApisProvider(config(root, provider="redditapis"))
                with self.assertRaisesRegex(ProviderError, "paid provider calls are disabled"):
                    run_once(db, provider, config(root, provider="redditapis"), "discover")
                self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 0)
            finally:
                db.close()

    def test_redditapis_bounded_comments_are_requested_and_audited(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                self.urls.append(url)
                if "/by_id/" in url:
                    return {"posts": [{"id": "p", "name": "t3_p", "title": "Post", "num_comments": 1}]}, HttpResponse(200, {}, b""), "post-request"
                return {"comments": [{"id": "c", "name": "t1_c", "body": "Comment", "parent_id": "t3_p"}]}, HttpResponse(200, {}, b""), "comment-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis", limit=1)
            client = Client()
            result = RedditApisProvider(cfg, client=client).refresh_posts([PostSnapshot("p", "t3_p", "freelance", permalink="/r/freelance/comments/p/post/")], cfg)
            self.assertIn("depth=1", client.urls[1])
            self.assertIn("limit=1", client.urls[1])
            self.assertEqual([record.operation for record in result.request_records], ["refresh", "comments"])
            self.assertEqual(result.observation_requests["p"].request_id, "post-request")

    def test_fetchlayer_full_comments_report_unsupported_without_calling_provider(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls = 0

            def request(self, *args: object, **kwargs: object) -> tuple[dict[str, object], HttpResponse, str]:
                self.calls += 1
                return {"id": "p", "title": "Post", "score": 4, "num_comments": 5}, HttpResponse(200, {}, b""), "refresh-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"FETCHLAYER_API_KEY": "test"}):
            cfg = config(Path(directory), provider="fetchlayer", comments="full")
            client = Client()
            result = FetchLayerProvider(cfg, client=client).refresh_posts([PostSnapshot("p", "t3_p", "freelance", permalink="/r/freelance/comments/p/post/")], cfg)
            self.assertEqual(client.calls, 1)
            self.assertEqual(len(result.posts), 1)
            self.assertEqual(result.gaps[0].reason, "unexpanded")

    def test_fetchlayer_next_page_is_an_unexpanded_comment_gap(self) -> None:
        class Client:
            def request(self, *args: object, **kwargs: object) -> tuple[dict[str, object], HttpResponse, str]:
                return (
                    {
                        "id": "p",
                        "name": "t3_p",
                        "title": "Post",
                        "num_comments": 1,
                        "comments": [{"id": "c", "name": "t1_c", "body": "Comment"}],
                        "nextPageUrl": "https://provider.test/next",
                    },
                    HttpResponse(200, {}, b""),
                    "refresh-request",
                )

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"FETCHLAYER_API_KEY": "test"}):
            cfg = config(Path(directory), provider="fetchlayer")
            result = FetchLayerProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "freelance", permalink="/r/freelance/comments/p/post/")], cfg
            )
            self.assertEqual(len(result.posts[0].comments), 1)
            self.assertTrue(any(gap.reason == "unexpanded" for gap in result.gaps))
            self.assertFalse(result.metadata["comments_expanded"])

    def test_malformed_discovery_items_are_audited(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                return {"posts": [{"title": "missing id"}], "after": "next"}, HttpResponse(200, {}, b""), f"request-{url.rsplit('=', 1)[-1]}"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            cfg = config(root, provider="redditapis")
            summary = run_once(db, RedditApisProvider(cfg, client=Client()), cfg, "discover")
            self.assertEqual(summary.gaps, 3)
            self.assertEqual(summary.requests, 3)
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0], 3)
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM gaps WHERE reason='provider_error'").fetchone()[0], 3)
            self.assertIsNone(db.checkpoint("freelance"))
            db.close()

    def test_malformed_discovery_page_stops_later_checkpoint(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls = 0

            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                self.calls += 1
                return {"posts": [{"title": "missing id"}], "after": "next"}, HttpResponse(200, {}, b""), f"request-{self.calls}"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            cfg = replace(config(root, provider="redditapis"), subreddits=("freelance",), max_discovery_pages=2)
            client = Client()
            run_once(db, RedditApisProvider(cfg, client=client), cfg, "discover")
            self.assertEqual(client.calls, 1)
            self.assertIsNone(db.checkpoint("freelance"))
            db.close()

    def test_fetchlayer_comment_failure_is_incomplete_comment_evidence(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/community-posts" in url:
                    return {"items": [{"id": "p", "permalink": "/r/freelance/comments/p/post/", "num_comments": 1}]}, HttpResponse(200, {}, b""), "listing-request"
                raise ProviderError("comment service unavailable", status=503, request_id="comment-request")

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"FETCHLAYER_API_KEY": "test"}):
            cfg = config(Path(directory), provider="fetchlayer")
            result = FetchLayerProvider(cfg, client=Client()).discover("freelance", None, cfg)
            self.assertTrue(any(gap.entity_type == "comment" and gap.reason == "provider_error" for gap in result.gaps))
            self.assertFalse(result.metadata["comments_expanded"])

    def test_partial_refresh_uses_known_comment_count_for_completeness(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/by_id/" in url:
                    return {"posts": [{"id": "p", "name": "t3_p"}]}, HttpResponse(200, {}, b""), "post-request"
                return {"comments": []}, HttpResponse(200, {}, b""), "comment-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis")
            result = RedditApisProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "freelance", num_comments=3)], cfg
            )
            self.assertTrue(any(gap.entity_type == "comment" and gap.reason == "unexpanded" for gap in result.gaps))
            self.assertFalse(result.metadata["comments_expanded"])

    def test_refresh_rejects_mismatched_fetchlayer_post_id(self) -> None:
        class Client:
            def request(self, *args: object, **kwargs: object) -> tuple[dict[str, object], HttpResponse, str]:
                return {"id": "q", "name": "t3_q", "comments": []}, HttpResponse(200, {}, b""), "refresh-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"FETCHLAYER_API_KEY": "test"}):
            cfg = config(Path(directory), provider="fetchlayer")
            result = FetchLayerProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "freelance", permalink="/r/freelance/comments/p/post/")], cfg
            )
            self.assertEqual(result.posts, [])
            self.assertTrue(any(gap.entity_id == "p" and gap.reason == "provider_error" for gap in result.gaps))

    def test_fixture_inline_malformed_comment_preserves_post_and_records_gap(self) -> None:
        fixture = {
            "listings": {
                "freelance": [
                    {
                        "request_after": None,
                        "posts": [
                            {
                                "id": "p",
                                "num_comments": 2,
                                "comments": [{"id": "c", "body": "valid"}, {"body": "missing id"}],
                            }
                        ],
                    }
                ]
            },
            "refresh": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_path = root / "fixture.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            page = FixtureProvider(fixture_path).discover("freelance", None, config(root))
            self.assertEqual([post.post_id for post in page.posts], ["p"])
            self.assertEqual([comment.comment_id for comment in page.posts[0].comments], ["c"])
            self.assertTrue(any(gap.entity_type == "comment" and gap.reason == "provider_error" for gap in page.gaps))

    def test_missing_discovery_collection_is_audited_without_checkpoint(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                return {"after": "next"}, HttpResponse(200, {}, b""), "request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            cfg = replace(config(root, provider="redditapis"), subreddits=("freelance",))
            summary = run_once(db, RedditApisProvider(cfg, client=Client()), cfg, "discover")
            self.assertEqual(summary.gaps, 1)
            self.assertEqual(summary.requests, 1)
            self.assertIsNone(db.checkpoint("freelance"))
            gap = db.connection.execute("SELECT entity_type, reason FROM gaps").fetchone()
            self.assertEqual(tuple(gap), ("listing", "provider_error"))
            db.close()

    def test_malformed_comment_is_audited_and_expansion_metadata_stays_false(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/by_id/" in url:
                    return {"posts": [{"id": "p", "name": "t3_p", "num_comments": 1}]}, HttpResponse(200, {}, b""), "post-request"
                return {"comments": [{"body": "missing id"}]}, HttpResponse(200, {}, b""), "comment-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis", comments="full")
            result = RedditApisProvider(cfg, client=Client()).refresh_posts([PostSnapshot("p", "t3_p", "freelance")], cfg)
            self.assertEqual(len(result.request_records), 2)
            self.assertTrue(any(gap.entity_type == "comment" and gap.reason == "provider_error" for gap in result.gaps))
            self.assertFalse(result.metadata["comments_expanded"])

    def test_dry_run_plans_refreshes_from_existing_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "db.sqlite3"
            db = Database(database_path)
            observed = "2026-09-05T00:00:00Z"
            post = PostSnapshot("p", "t3_p", "freelance", observed_at=observed)
            run_id = db.start_run("fixture", "discover", {}, observed)
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([post], None, None, observed, "fixture://p", 200, "fixture"))
            db.close()
            config_path = root / "config.toml"
            config_path.write_text(
                f"""[ingestion]
subreddits = [\"freelance\"]
database_path = \"{database_path}\"

[provider]
name = \"fixture\"
fixture_path = \"{FIXTURE}\"
""",
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli_main(["plan", "--config", str(config_path), "--dry-run"]), 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "dry_run")
            self.assertEqual(payload["plan"]["refresh_events"], 1)
            check = Database(database_path)
            try:
                self.assertEqual(check.counts()["posts"], 1)
            finally:
                check.close()

    def test_discovery_provider_failure_persists_gap_without_checkpoint(self) -> None:
        class FailedProvider:
            name = "redditapis"

            def status(self) -> ProviderStatus:
                return ProviderStatus("redditapis", True, True, "test provider", True)

            def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
                raise ProviderError("service unavailable", status=503, retryable=True, billed=True, request_id="failed-request", url="https://provider.test/listing")

            def refresh_posts(self, posts: list[PostSnapshot], config: Config):
                raise AssertionError("refresh is not part of discover mode")

        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "db.sqlite3")
            summary = run_once(db, FailedProvider(), config(Path(directory), provider="redditapis"), "discover", allow_paid=True)
            self.assertEqual(summary.gaps, 3)
            self.assertIsNone(db.checkpoint("freelance"))
            request = db.connection.execute("SELECT request_id, response_status, billed FROM requests").fetchone()
            self.assertEqual(tuple(request), ("failed-request", 503, 1))
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM gaps WHERE reason='provider_error'").fetchone()[0], 3)
            db.close()

    def test_comment_policy_bounded_and_full(self) -> None:
        post = PostSnapshot("p", "t3_p", comments=[CommentSnapshot("c", "t1_c", "p", depth=2)])
        bounded = config(Path("/tmp"))
        self.assertEqual(apply_comment_policy([post], bounded)[0].reason, "bounded_depth")
        self.assertEqual(post.comments, [])
        post.comments = [CommentSnapshot("c", "t1_c", "p", depth=2)]
        full = config(Path("/tmp"), comments="full")
        self.assertEqual(apply_comment_policy([post], full)[0].reason, "unexpanded")
        self.assertEqual(len(post.comments), 1)

    def test_unknown_comment_count_is_not_complete(self) -> None:
        post = PostSnapshot("p", "t3_p", "freelance")
        gaps = apply_comment_policy([post], config(Path("/tmp"), comments="full"))
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].reason, "unexpanded")


if __name__ == "__main__":
    unittest.main()
