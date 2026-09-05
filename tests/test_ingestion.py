from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reddit_ingestion.comments import apply_comment_policy
from reddit_ingestion.config import Config
from reddit_ingestion.db import Database
from reddit_ingestion.models import CommentSnapshot, PageResult, PostSnapshot
from reddit_ingestion.normalize import parse_post
from reddit_ingestion.providers import FixtureProvider, JsonClient, HttpResponse, ProviderError, RedditApisProvider
from reddit_ingestion.runner import check_live_access, run_once


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
        allow_paid=False,
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

    def test_discovery_resume_and_idempotent_current_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "db.sqlite3")
            provider = FixtureProvider(FIXTURE)
            cfg = config(Path(directory), comments="off")
            first = run_once(db, provider, cfg, "discover")
            self.assertEqual(first.discovered, 3)
            checkpoint = db.checkpoint("freelance")
            self.assertEqual(checkpoint["cursor"], "t3_freelance_next")
            resumed_page = provider.discover("freelance", "t3_freelance_next", cfg)
            self.assertEqual([post.post_id for post in resumed_page.posts], ["freelance002"])
            second = run_once(db, provider, cfg, "discover")
            self.assertEqual(second.discovered, 3)
            self.assertEqual(db.counts()["posts"], 3)
            self.assertEqual(db.counts()["post_observations"], 6)
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

    def test_comment_policy_off_and_full(self) -> None:
        post = PostSnapshot("p", "t3_p", comments=[CommentSnapshot("c", "t1_c", "p", depth=2)])
        off = config(Path("/tmp"), comments="off")
        self.assertEqual(apply_comment_policy([post], off), [])
        self.assertEqual(post.comments, [])
        post.comments = [CommentSnapshot("c", "t1_c", "p", depth=2)]
        full = config(Path("/tmp"), comments="full")
        self.assertEqual(apply_comment_policy([post], full), [])
        self.assertEqual(len(post.comments), 1)


if __name__ == "__main__":
    unittest.main()
