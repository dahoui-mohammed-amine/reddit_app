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

    def test_normalization_rejects_empty_and_conflicting_ids(self) -> None:
        with self.assertRaises(ValueError):
            parse_post({"id": "t3_"})
        with self.assertRaises(ValueError):
            parse_post({"id": "p", "name": "t3_q"})
        post = parse_post({"id": "p"})
        with self.assertRaises(ValueError):
            parse_comment({"id": "t1_"}, post=post, observed_at=post.observed_at)
        with self.assertRaises(ValueError):
            parse_post({"id": {"bad": 1}})

    def test_comment_links_must_match_requested_post(self) -> None:
        post = parse_post({"id": "p1"}, observed_at="2026-09-05T00:00:00Z")
        linked = parse_comment({"id": "c", "link_id": "t3_p1"}, post=post, observed_at=post.observed_at)
        self.assertEqual(linked.post_id, "p1")
        with self.assertRaises(ValueError):
            parse_comment({"id": "c", "link_id": "t3_p2"}, post=post, observed_at=post.observed_at)

    def test_normalization_recognizes_deleted_content_markers(self) -> None:
        post = parse_post({"id": "deleted", "author": None, "selftext": "[deleted]"})
        comment = parse_comment({"id": "comment", "author": None, "body": "[deleted]"}, post=post, observed_at=post.observed_at)
        removed = parse_comment({"id": "removed", "text": "[removed]"}, post=post, observed_at=post.observed_at)
        self.assertTrue(post.deleted)
        self.assertTrue(comment.deleted)
        self.assertTrue(removed.removed)

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

    def test_fixture_missing_resume_page_preserves_checkpoint(self) -> None:
        fixture = {
            "listings": {"freelance": [{"request_after": None, "posts": []}]},
            "refresh": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_path = root / "fixture.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            db = Database(root / "db.sqlite3")
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([], None, "saved-cursor", observed, "fixture://freelance/new", 200, "fixture"))
            cfg = replace(config(root), subreddits=("freelance",))
            run_once(db, FixtureProvider(fixture_path), cfg, "discover")
            self.assertEqual(db.checkpoint("freelance")["cursor"], "saved-cursor")
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

    def test_refresh_scope_and_expiry_preserve_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            observed = "2026-09-01T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            posts = [
                PostSnapshot("active", "t3_active", "SaaS", observed_at=observed, refresh_until="2099-01-01T00:00:00Z"),
                PostSnapshot("removed-scope", "t3_removed-scope", "freelance", observed_at=observed, refresh_until="2099-01-01T00:00:00Z"),
                PostSnapshot("expired", "t3_expired", "SaaS", observed_at=observed, refresh_until="2020-01-01T00:00:00Z"),
                PostSnapshot("legacy", "t3_legacy", "SaaS", created_at="2020-01-01T00:00:00Z", observed_at=observed),
            ]
            with db.transaction():
                db.save_page(run_id, "fixture", "SaaS", PageResult(posts, None, None, observed, "fixture://posts", 200, "fixture"))
            cfg = replace(config(root), subreddits=("SaaS",))
            due = db.due_posts(cfg.refresh_interval_minutes, cfg.max_refresh_posts, cfg.subreddits)
            self.assertEqual([post.post_id for post in due], ["active"])
            plan = plan_for(db, FixtureProvider(FIXTURE), cfg, "refresh")
            self.assertEqual(plan.refresh_events, 1)
            class RecordingProvider:
                name = "fixture"

                def __init__(self) -> None:
                    self.post_ids: list[str] = []

                def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
                    self.post_ids = [post.post_id for post in posts]
                    return RefreshResult([], observed, None, 200)

            provider = RecordingProvider()
            run_once(db, provider, cfg, "refresh")
            self.assertEqual(provider.post_ids, ["active"])
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM posts").fetchone()[0], 4)
            db.close()

    def test_discovery_sets_configured_refresh_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            cfg = replace(config(root), subreddits=("SaaS",), refresh_expiry_days=2)
            run_once(db, FixtureProvider(FIXTURE), cfg, "discover")
            refresh_until = db.connection.execute("SELECT refresh_until FROM posts WHERE post_id='saas001'").fetchone()[0]
            self.assertEqual(refresh_until, "2026-09-06T13:00:00Z")
            db.close()

    def test_numeric_created_timestamp_sets_source_based_expiry(self) -> None:
        class Provider:
            name = "fixture"

            def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
                return PageResult(
                    [PostSnapshot("numeric", "t3_numeric", subreddit, created_at="0", observed_at="2026-09-06T00:00:00Z")],
                    cursor,
                    None,
                    "2026-09-06T00:00:00Z",
                    "fixture://numeric",
                    200,
                    "fixture",
                )

            def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
                raise AssertionError("refresh is not part of discovery mode")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            cfg = replace(config(root), subreddits=("freelance",), refresh_expiry_days=30)
            run_once(db, Provider(), cfg, "discover")
            refresh_until = db.connection.execute("SELECT refresh_until FROM posts WHERE post_id='numeric'").fetchone()[0]
            self.assertEqual(refresh_until, "1970-01-31T00:00:00Z")
            db.close()

    def test_refresh_expiry_preserves_stored_source_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            source_time = "2026-09-01T00:00:00Z"
            observed = "2026-09-06T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, source_time)
            with db.transaction():
                db.save_page(
                    run_id,
                    "fixture",
                    "freelance",
                    PageResult([PostSnapshot("p", "t3_p", "freelance", created_at=source_time, observed_at=source_time)], None, None, source_time, "fixture://p", 200, "fixture"),
                )

            class Provider:
                name = "fixture"

                def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
                    return RefreshResult([PostSnapshot("p", "t3_p", "freelance", observed_at=observed)], observed, "refresh", 200)

            cfg = replace(config(root), subreddits=("freelance",), refresh_expiry_days=30)
            run_once(db, Provider(), cfg, "refresh")
            refresh_until = db.connection.execute("SELECT refresh_until FROM posts WHERE post_id='p'").fetchone()[0]
            self.assertEqual(refresh_until, "2026-10-01T00:00:00Z")
            db.close()

    def test_later_source_timestamp_corrects_observation_based_expiry(self) -> None:
        class Provider:
            name = "fixture"

            def __init__(self) -> None:
                self.calls = 0

            def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
                self.calls += 1
                observed = "2026-09-05T00:00:00Z"
                created_at = None if self.calls == 1 else "2020-01-01T00:00:00Z"
                return PageResult(
                    [PostSnapshot("p", "t3_p", subreddit, created_at=created_at, num_comments=0, observed_at=observed)],
                    cursor,
                    None,
                    observed,
                    "fixture://p",
                    200,
                    "request",
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            cfg = replace(config(root), subreddits=("freelance",), refresh_expiry_days=30)
            provider = Provider()
            run_once(db, provider, cfg, "discover")
            run_once(db, provider, cfg, "discover")
            refresh_until = db.connection.execute("SELECT refresh_until FROM posts WHERE post_id='p'").fetchone()[0]
            self.assertEqual(refresh_until, "2020-01-31T00:00:00Z")
            db.close()

    def test_migration_seals_legacy_null_expiry_before_discovery(self) -> None:
        class Provider:
            name = "fixture"

            def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
                observed = "2026-09-06T00:00:00Z"
                return PageResult(
                    [PostSnapshot("p", "t3_p", subreddit, num_comments=0, observed_at=observed)],
                    cursor,
                    None,
                    observed,
                    "fixture://p",
                    200,
                    "request",
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "db.sqlite3"
            db = Database(database_path)
            source_time = "2026-09-01T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, source_time)
            with db.transaction():
                db.save_page(
                    run_id,
                    "fixture",
                    "freelance",
                    PageResult([PostSnapshot("p", "t3_p", "freelance", observed_at=source_time)], None, None, source_time, "fixture://p", 200, "request"),
                )
            db.close()
            db = Database(database_path, 30)
            migrated_expiry = db.connection.execute("SELECT refresh_until FROM posts WHERE post_id='p'").fetchone()[0]
            self.assertEqual(migrated_expiry, "2026-10-01T00:00:00Z")
            cfg = replace(config(root), subreddits=("freelance",), refresh_expiry_days=30)
            run_once(db, Provider(), cfg, "discover")
            refresh_until = db.connection.execute("SELECT refresh_until FROM posts WHERE post_id='p'").fetchone()[0]
            self.assertEqual(refresh_until, migrated_expiry)
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

    def test_full_refresh_tracks_comment_completeness_per_post(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            initial = PostSnapshot(
                "p1",
                "t3_p1",
                "freelance",
                observed_at=observed,
                comments=[CommentSnapshot("c1", "t1_c1", "p1", observed_at=observed)],
            )
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([initial], None, None, observed, "fixture://p1", 200, "fixture"))
            result = RefreshResult(
                [
                    PostSnapshot("p1", "t3_p1", "freelance", observed_at=observed, comments=[]),
                    PostSnapshot("p2", "t3_p2", "freelance", observed_at=observed, comments=[]),
                ],
                observed,
                "refresh",
                200,
                gaps=[Gap("comment", "malformed", entity_id="p2", subreddit="freelance")],
                metadata={"comments_mode": "full", "comments_expanded": False},
            )
            with db.transaction():
                db.save_refresh(run_id, "fixture", result)
            self.assertTrue(any(gap.entity_id == "p1" and gap.reason == "unexpanded" for gap in result.gaps))
            metadata = {
                row[0]: json.loads(row[1])
                for row in db.connection.execute(
                    "SELECT post_id, metadata_json FROM post_observations WHERE post_id IN ('p1', 'p2') ORDER BY observation_id"
                )
            }
            self.assertFalse(metadata["p1"]["comments_expanded"])
            self.assertFalse(metadata["p2"]["comments_expanded"])
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

    def test_discovery_cursor_cycle_defers_checkpoint_and_stops(self) -> None:
        class Provider:
            name = "fixture"

            def __init__(self) -> None:
                self.cursors: list[str | None] = []

            def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
                self.cursors.append(cursor)
                observed = "2026-09-05T00:00:00Z"
                return PageResult(
                    [PostSnapshot("p", "t3_p", subreddit, num_comments=0, observed_at=observed)],
                    cursor,
                    "cycle",
                    observed,
                    "fixture://cycle",
                    200,
                    "request",
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            provider = Provider()
            cfg = replace(config(root), subreddits=("freelance",), max_discovery_pages=3)
            run_once(db, provider, cfg, "discover")
            self.assertEqual(provider.cursors, [None, "cycle"])
            self.assertEqual(db.checkpoint("freelance")["cursor"], "cycle")
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM gaps WHERE reason='cursor_cycle'").fetchone()[0], 1)
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
            unknown = parse_post({"id": "p", "deleted": None, "score": 5, "num_comments": 0}, default_subreddit="freelance", observed_at=observed)
            with db.transaction():
                db.save_refresh(run_id, "fixture", RefreshResult([unknown], observed, "refresh", 200))
            self.assertEqual(db.connection.execute("SELECT deleted FROM posts WHERE post_id='p'").fetchone()[0], 1)
            cleared = parse_post({"id": "p", "deleted": False, "score": 6, "num_comments": 0}, default_subreddit="freelance", observed_at=observed)
            with db.transaction():
                db.save_refresh(run_id, "fixture", RefreshResult([cleared], observed, "refresh", 200))
            self.assertEqual(db.connection.execute("SELECT deleted, score FROM posts WHERE post_id='p'").fetchone()[0:2], (0, 6))
            self.assertEqual(db.purge_deleted_content(), 0)
            db.close()

    def test_partial_snapshot_preserves_archived_and_locked_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            initial = parse_post({"id": "p", "archived": True, "locked": True}, default_subreddit="freelance", observed_at=observed)
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([initial], None, None, observed, "fixture://p", 200, "fixture"))
            partial = parse_post({"id": "p", "score": 5}, default_subreddit="freelance", observed_at=observed)
            with db.transaction():
                db.save_refresh(run_id, "fixture", RefreshResult([partial], observed, "refresh", 200))
            row = db.connection.execute("SELECT archived, locked, score FROM posts WHERE post_id='p'").fetchone()
            self.assertEqual(tuple(row), (1, 1, 5))
            cleared = parse_post({"id": "p", "archived": False, "locked": False}, default_subreddit="freelance", observed_at=observed)
            with db.transaction():
                db.save_refresh(run_id, "fixture", RefreshResult([cleared], observed, "refresh", 200))
            self.assertEqual(tuple(db.connection.execute("SELECT archived, locked FROM posts WHERE post_id='p'").fetchone()), (0, 0))
            db.close()

    def test_comment_ownership_conflict_preserves_existing_post(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            first = PostSnapshot("p1", "t3_p1", "freelance", observed_at=observed, comments=[CommentSnapshot("c", "t1_c", "p1", observed_at=observed)])
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([first], None, None, observed, "fixture://p1", 200, "request"))
            second = PostSnapshot("p2", "t3_p2", "freelance", observed_at=observed, comments=[CommentSnapshot("c", "t1_c", "p2", observed_at=observed)])
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([second], None, None, observed, "fixture://p2", 200, "request", metadata={"comments_expanded": True}))
            self.assertEqual(db.connection.execute("SELECT post_id FROM comments WHERE comment_id='c'").fetchone()[0], "p1")
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM gaps WHERE entity_id='c' AND reason='provider_error'").fetchone()[0], 1)
            metadata = db.connection.execute("SELECT metadata_json FROM post_observations WHERE post_id='p2'").fetchone()[0]
            self.assertFalse(json.loads(metadata)["comments_expanded"])
            db.close()

    def test_partial_refresh_observation_preserves_source_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            source_time = "2026-01-01T00:00:00Z"
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            initial = PostSnapshot("p", "t3_p", "freelance", created_at=source_time, observed_at=observed)
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", PageResult([initial], None, None, observed, "fixture://p", 200, "request"))
            partial = PostSnapshot("p", "t3_p", "freelance", observed_at=observed, score=3)
            with db.transaction():
                db.save_refresh(run_id, "fixture", RefreshResult([partial], observed, "refresh", 200))
            source_times = [row[0] for row in db.connection.execute("SELECT source_created_at FROM post_observations WHERE post_id='p' ORDER BY observation_id")]
            self.assertEqual(source_times, [source_time, source_time])
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

    def test_discovery_tracks_comment_completeness_per_post(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "db.sqlite3")
            observed = "2026-09-05T00:00:00Z"
            run_id = db.start_run("fixture", "discover", {}, observed)
            page = PageResult(
                [
                    PostSnapshot("p1", "t3_p1", "freelance", num_comments=0, observed_at=observed),
                    PostSnapshot("p2", "t3_p2", "freelance", num_comments=0, observed_at=observed),
                ],
                None,
                None,
                observed,
                "fixture://posts",
                200,
                "request",
                gaps=[Gap("comment", "provider_error", entity_id="p2", subreddit="freelance")],
                metadata={"comments_expanded": False},
            )
            with db.transaction():
                db.save_page(run_id, "fixture", "freelance", page)
            metadata = {
                row[0]: json.loads(row[1])
                for row in db.connection.execute("SELECT post_id, metadata_json FROM post_observations ORDER BY observation_id")
            }
            self.assertTrue(metadata["p1"]["comments_expanded"])
            self.assertFalse(metadata["p2"]["comments_expanded"])
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

    def test_full_comments_stop_on_truncated_page(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.comment_calls = 0

            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/by_id/" in url:
                    return {"posts": [{"id": "p", "name": "t3_p", "num_comments": 1}]}, HttpResponse(200, {}, b""), "post-request"
                self.comment_calls += 1
                return {"comments": [{"id": "c", "name": "t1_c", "body": "one"}], "after": "next", "listing_status": "truncated"}, HttpResponse(200, {}, b""), "comment-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis", comments="full")
            client = Client()
            result = RedditApisProvider(cfg, client=client).refresh_posts([PostSnapshot("p", "t3_p", "freelance")], cfg)
            self.assertEqual(client.comment_calls, 1)
            self.assertEqual(len(result.posts[0].comments), 1)
            self.assertTrue(any(gap.reason == "unexpanded" for gap in result.gaps))

    def test_discovery_deduplicates_post_listing_items(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/posts?" in url:
                    item = {"id": "p", "name": "t3_p", "num_comments": 0}
                    return {"posts": [item, item], "after": None}, HttpResponse(200, {}, b""), "listing-request"
                return {"comments": []}, HttpResponse(200, {}, b""), "comment-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis")
            result = RedditApisProvider(cfg, client=Client()).discover("freelance", None, cfg)
            self.assertEqual([post.post_id for post in result.posts], ["p"])
            self.assertEqual(len(result.request_records), 2)

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

    def test_json_client_does_not_retry_nonretryable_transport_errors(self) -> None:
        calls: list[int] = []

        def transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> HttpResponse:
            calls.append(len(calls))
            raise ProviderError("terminal transport failure", status=401)

        client = JsonClient(timeout=1, retries=2, transport=transport, sleep=lambda _: None)
        with self.assertRaises(ProviderError) as context:
            client.request("GET", "https://example.test", headers={})
        self.assertEqual(len(calls), 1)
        self.assertFalse(context.exception.retryable)
        self.assertEqual(len(client.last_attempts), 1)

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

    def test_fixture_plan_respects_refresh_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = replace(config(root), subreddits=("freelance",), max_refresh_posts=100)
            plan = FixtureProvider(FIXTURE).plan(cfg, 101, mode="refresh")
            self.assertEqual(plan.refresh_events, 100)

    def test_json_client_paces_successive_requests(self) -> None:
        delays: list[float] = []
        client = JsonClient(timeout=1, retries=0, transport=lambda *args: HttpResponse(200, {}, b"{}"), sleep=delays.append, min_interval_seconds=1)
        client.request("GET", "https://example.test/one", headers={})
        client.request("GET", "https://example.test/two", headers={})
        self.assertTrue(any(delay > 0 for delay in delays))

    def test_json_client_reports_terminal_error_and_billing(self) -> None:
        client = JsonClient(timeout=1, retries=0, transport=lambda *args: HttpResponse(403, {}, b'{"error":"denied"}'), sleep=lambda _: None)
        with self.assertRaises(ProviderError) as context:
            client.request("GET", "https://example.test", headers={})
        self.assertFalse(context.exception.retryable)
        self.assertIsNone(context.exception.billed)
        self.assertIsNone(context.exception.attempts[-1].billed)

    def test_json_client_marks_malformed_success_billing_unknown(self) -> None:
        client = JsonClient(timeout=1, retries=0, transport=lambda *args: HttpResponse(200, {}, b"not-json"), sleep=lambda _: None)
        with self.assertRaises(ProviderError) as context:
            client.request("GET", "https://example.test", headers={})
        self.assertIsNone(context.exception.attempts[-1].billed)

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

    def test_fetchlayer_comment_ids_are_deduplicated_before_counting(self) -> None:
        class Client:
            def request(self, *args: object, **kwargs: object) -> tuple[dict[str, object], HttpResponse, str]:
                comment = {"id": "c", "name": "t1_c", "body": "Comment"}
                return {"id": "p", "name": "t3_p", "num_comments": 2, "comments": [comment, comment]}, HttpResponse(200, {}, b""), "refresh-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"FETCHLAYER_API_KEY": "test"}):
            cfg = config(Path(directory), provider="fetchlayer")
            result = FetchLayerProvider(cfg, client=Client()).refresh_posts(
                [PostSnapshot("p", "t3_p", "freelance", permalink="/r/freelance/comments/p/post/")], cfg
            )
            self.assertEqual(len(result.posts[0].comments), 1)
            self.assertTrue(any(gap.reason == "unexpanded" for gap in result.gaps))

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

    def test_fetchlayer_discovery_preserves_expansion_engagement_metrics(self) -> None:
        class Client:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/community-posts" in url:
                    return {"items": [{"id": "p", "permalink": "/r/freelance/comments/p/post/"}]}, HttpResponse(200, {}, b""), "listing-request"
                return {
                    "id": "p",
                    "name": "t3_p",
                    "score": 7,
                    "ups": 8,
                    "upvote_ratio": 0.9,
                    "num_comments": 2,
                    "comments": [{"id": "c1"}, {"id": "c2"}],
                }, HttpResponse(200, {}, b""), "expansion-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"FETCHLAYER_API_KEY": "test"}):
            cfg = config(Path(directory), provider="fetchlayer")
            result = FetchLayerProvider(cfg, client=Client()).discover("freelance", None, cfg)
            post = result.posts[0]
            self.assertEqual((post.score, post.ups, post.upvote_ratio, post.num_comments), (7, 8, 0.9, 2))
            self.assertTrue(result.metadata["comments_expanded"])
            self.assertFalse(any(gap.reason == "unexpanded" for gap in result.gaps))

    def test_http_403_comment_failures_remain_provider_errors(self) -> None:
        class RedditApisClient:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/by_id/" in url:
                    return {"posts": [{"id": "p", "name": "t3_p", "num_comments": 1}]}, HttpResponse(200, {}, b""), "post-request"
                raise ProviderError("comment access denied", status=403, request_id="comment-request")

        class FetchLayerClient:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/community-posts" in url:
                    return {"items": [{"id": "p", "permalink": "/r/freelance/comments/p/post/", "num_comments": 1}]}, HttpResponse(200, {}, b""), "listing-request"
                raise ProviderError("comment access denied", status=403, request_id="comment-request")

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test", "FETCHLAYER_API_KEY": "test"}):
            root = Path(directory)
            reddit_config = config(root, provider="redditapis")
            reddit_result = RedditApisProvider(reddit_config, client=RedditApisClient()).refresh_posts([PostSnapshot("p", "t3_p", "freelance")], reddit_config)
            self.assertTrue(any(gap.entity_type == "comment" and gap.reason == "provider_error" for gap in reddit_result.gaps))
            fetch_config = config(root, provider="fetchlayer")
            fetch_result = FetchLayerProvider(fetch_config, client=FetchLayerClient()).discover("freelance", None, fetch_config)
            self.assertTrue(any(gap.entity_type == "comment" and gap.reason == "provider_error" for gap in fetch_result.gaps))

    def test_redditapis_blocked_payloads_are_blocked_gaps(self) -> None:
        class BlockedRefreshClient:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                return {"blocked": True, "blockReason": "access denied"}, HttpResponse(200, {}, b""), "refresh-request"

        class BlockedCommentsClient:
            def request(self, method: str, url: str, *, headers: dict[str, str], payload: dict[str, object] | None = None) -> tuple[dict[str, object], HttpResponse, str]:
                if "/by_id/" in url:
                    return {"posts": [{"id": "p", "name": "t3_p", "num_comments": 1}]}, HttpResponse(200, {}, b""), "post-request"
                return {"blocked": True, "blockReason": "comments denied"}, HttpResponse(200, {}, b""), "comment-request"

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"REDDITAPIS_API_KEY": "test"}):
            cfg = config(Path(directory), provider="redditapis")
            post = PostSnapshot("p", "t3_p", "freelance")
            refresh = RedditApisProvider(cfg, client=BlockedRefreshClient()).refresh_posts([post], cfg)
            self.assertTrue(any(gap.entity_type == "post" and gap.reason == "blocked" for gap in refresh.gaps))
            comments = RedditApisProvider(cfg, client=BlockedCommentsClient()).refresh_posts([post], cfg)
            self.assertTrue(any(gap.entity_type == "comment" and gap.reason == "blocked" for gap in comments.gaps))

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
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli_main(["plan", "--config", str(config_path), "--dry-run", "--mode", "discover"]), 0)
            discover_payload = json.loads(output.getvalue())
            self.assertEqual(discover_payload["plan"]["refresh_events"], 0)
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli_main(["plan", "--config", str(config_path), "--dry-run", "--mode", "refresh"]), 0)
            refresh_payload = json.loads(output.getvalue())
            self.assertEqual(refresh_payload["plan"]["discovery_requests"], 0)
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

    def test_negative_comment_count_is_not_complete(self) -> None:
        post = PostSnapshot("p", "t3_p", "freelance", num_comments=-1)
        gaps = apply_comment_policy([post], config(Path("/tmp"), comments="full"))
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].reason, "unexpanded")

    def test_bounded_limit_does_not_create_false_depth_or_count_gap(self) -> None:
        post = PostSnapshot(
            "p",
            "t3_p",
            "freelance",
            num_comments=2,
            comments=[
                CommentSnapshot("c1", "t1_c1", "p", depth=0),
                CommentSnapshot("c2", "t1_c2", "p", depth=1),
            ],
        )
        gaps = apply_comment_policy([post], replace(config(Path("/tmp")), comment_limit=1, comment_depth=1))
        self.assertEqual([gap.reason for gap in gaps], ["bounded_limit"])
        self.assertEqual([comment.comment_id for comment in post.comments], ["c1"])

    def test_bounded_policy_excludes_unknown_depth(self) -> None:
        post = PostSnapshot("p", "t3_p", "freelance", comments=[CommentSnapshot("c", "t1_c", "p")])
        gaps = apply_comment_policy([post], config(Path("/tmp")))
        self.assertEqual(post.comments, [])
        self.assertEqual(gaps[0].reason, "bounded_depth")


if __name__ == "__main__":
    unittest.main()
