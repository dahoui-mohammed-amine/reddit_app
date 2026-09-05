from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import CommentSnapshot, Gap, PageResult, PostSnapshot, RefreshResult, RequestRecord


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    provider TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    fullname TEXT NOT NULL UNIQUE,
    subreddit TEXT,
    title TEXT,
    body TEXT,
    author TEXT,
    permalink TEXT,
    url TEXT,
    created_at TEXT,
    score INTEGER,
    ups INTEGER,
    upvote_ratio REAL,
    num_comments INTEGER,
    archived INTEGER NOT NULL DEFAULT 0,
    locked INTEGER NOT NULL DEFAULT 0,
    deleted INTEGER NOT NULL DEFAULT 0,
    removed INTEGER NOT NULL DEFAULT 0,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS post_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    source_created_at TEXT,
    score INTEGER,
    ups INTEGER,
    upvote_ratio REAL,
    num_comments INTEGER,
    response_status INTEGER,
    request_id TEXT,
    cache_status TEXT,
    cache_observed_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_post_observations_post_time ON post_observations(post_id, observed_at);
CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    fullname TEXT NOT NULL UNIQUE,
    post_id TEXT NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE,
    parent_id TEXT,
    author TEXT,
    body TEXT,
    permalink TEXT,
    created_at TEXT,
    score INTEGER,
    ups INTEGER,
    depth INTEGER,
    deleted INTEGER NOT NULL DEFAULT 0,
    removed INTEGER NOT NULL DEFAULT 0,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gaps (
    gap_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    subreddit TEXT,
    reason TEXT NOT NULL,
    detail TEXT,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    request_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    operation TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    response_status INTEGER,
    billed INTEGER,
    cache_status TEXT,
    cache_observed_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS checkpoints (
    subreddit TEXT PRIMARY KEY,
    cursor TEXT,
    page_count INTEGER NOT NULL DEFAULT 0,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    source_url TEXT
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self._migrate_requests()
        self.connection.commit()

    def _migrate_requests(self) -> None:
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(requests)")}
        if "request_event_id" not in columns:
            self.connection.execute("ALTER TABLE requests RENAME TO requests_legacy")
            self.connection.execute(
                """CREATE TABLE requests (
                request_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                operation TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                response_status INTEGER,
                billed INTEGER,
                cache_status TEXT,
                cache_observed_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
                )"""
            )
            cache_timestamp = "cache_observed_at" if "cache_observed_at" in columns else "NULL"
            self.connection.execute(
                f"""INSERT INTO requests(request_id, run_id, provider, operation, requested_at, response_status, billed, cache_status, cache_observed_at, metadata_json)
                SELECT request_id, run_id, provider, operation, requested_at, response_status, billed, cache_status, {cache_timestamp}, metadata_json
                FROM requests_legacy"""
            )
            self.connection.execute("DROP TABLE requests_legacy")
            columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(requests)")}
        if "cache_observed_at" not in columns:
            self.connection.execute("ALTER TABLE requests ADD COLUMN cache_observed_at TEXT")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def close(self) -> None:
        self.connection.close()

    def start_run(self, provider: str, mode: str, config: dict[str, Any], started_at: str) -> str:
        run_id = str(uuid.uuid4())
        self.connection.execute("INSERT INTO runs(run_id, started_at, provider, mode, status, config_json) VALUES (?, ?, ?, ?, 'running', ?)", (run_id, started_at, provider, mode, json.dumps(config, sort_keys=True)))
        self.connection.commit()
        return run_id

    def finish_run(self, run_id: str, finished_at: str, status: str) -> None:
        self.connection.execute("UPDATE runs SET finished_at = ?, status = ? WHERE run_id = ?", (finished_at, status, run_id))
        self.connection.commit()

    def checkpoint(self, subreddit: str) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM checkpoints WHERE subreddit = ?", (subreddit,)).fetchone()

    def save_page(self, run_id: str, provider: str, subreddit: str, page: PageResult) -> tuple[int, int]:
        for post in page.posts:
            self._save_post(post, provider)
            self._save_observation(post, provider, page)
            if post.deleted or post.removed:
                page.gaps.append(Gap("post", "deleted" if post.deleted else "removed", entity_id=post.post_id, subreddit=post.subreddit))
            for comment in post.comments:
                self._save_comment(comment, post.post_id, provider)
                if comment.deleted or comment.removed:
                    page.gaps.append(Gap("comment", "deleted" if comment.deleted else "removed", entity_id=comment.comment_id, subreddit=post.subreddit))
        listing_status = page.metadata.get("listing_status")
        if listing_status in {"truncated", "unknown"}:
            page.gaps.append(Gap("listing", "truncated", subreddit=subreddit, detail=f"listing_status={listing_status}"))
        for gap in page.gaps:
            self._save_gap(run_id, provider, gap, page.observed_at, subreddit=subreddit)
        blocked = page.metadata.get("blocked") or any(gap.entity_type == "listing" and gap.reason == "blocked" for gap in page.gaps)
        provider_incomplete = listing_status in {"truncated", "unknown"} or page.metadata.get("checkpoint_deferred")
        if not page.metadata.get("request_failed") and not page.metadata.get("checkpoint_deferred") and not blocked and not provider_incomplete:
            self.connection.execute(
                "INSERT INTO checkpoints(subreddit, cursor, page_count, observed_at, provider, source_url) VALUES (?, ?, 1, ?, ?, ?) ON CONFLICT(subreddit) DO UPDATE SET cursor=excluded.cursor, page_count=checkpoints.page_count+1, observed_at=excluded.observed_at, provider=excluded.provider, source_url=excluded.source_url",
                (subreddit, page.next_cursor, page.observed_at, provider, page.source_url),
            )
        self._save_request_records(run_id, provider, page.request_records, "discover", page.request_id, page.response_status, page.cache_status, page.cache_observed_at, page.metadata)
        return len(page.posts), sum(len(post.comments) for post in page.posts)

    def save_refresh(self, run_id: str, provider: str, result: RefreshResult) -> tuple[int, int]:
        self._defer_omitted_full_comments(result)
        for post in result.posts:
            self._save_post(post, provider)
            self._save_observation(post, provider, result, result.observation_requests.get(post.post_id))
            if post.deleted or post.removed:
                result.gaps.append(Gap("post", "deleted" if post.deleted else "removed", entity_id=post.post_id, subreddit=post.subreddit))
            for comment in post.comments:
                self._save_comment(comment, post.post_id, provider)
                if comment.deleted or comment.removed:
                    result.gaps.append(Gap("comment", "deleted" if comment.deleted else "removed", entity_id=comment.comment_id, subreddit=post.subreddit))
        for gap in result.gaps:
            self._save_gap(run_id, provider, gap, result.observed_at)
        self._save_request_records(run_id, provider, result.request_records, "refresh", result.request_id, result.response_status, result.cache_status, result.cache_observed_at, result.metadata)
        return len(result.posts), sum(len(post.comments) for post in result.posts)

    def _defer_omitted_full_comments(self, result: RefreshResult) -> None:
        if result.metadata.get("comments_mode") != "full" or result.metadata.get("comments_expanded") is False:
            return
        for post in result.posts:
            known_ids = {
                row[0]
                for row in self.connection.execute("SELECT comment_id FROM comments WHERE post_id = ?", (post.post_id,))
            }
            returned_ids = {comment.comment_id for comment in post.comments}
            if known_ids - returned_ids:
                result.gaps.append(
                    Gap(
                        "comment",
                        "unexpanded",
                        entity_id=post.post_id,
                        subreddit=post.subreddit,
                        detail="full refresh omitted previously known comments without deletion confirmation",
                    )
                )
                result.metadata["comments_expanded"] = False

    def _save_post(self, post: PostSnapshot, provider: str) -> None:
        deleted = int(post.deleted)
        removed = int(post.removed)
        title = None if deleted or removed else post.title
        body = None if deleted or removed else post.body
        author = None if deleted or removed else post.author
        deleted_update = "deleted=excluded.deleted" if post.deletion_known or post.deleted else "deleted=posts.deleted"
        removed_update = "removed=excluded.removed" if post.removal_known or post.removed else "removed=posts.removed"
        archived_update = "archived=excluded.archived" if post.archived_known or post.archived else "archived=posts.archived"
        locked_update = "locked=excluded.locked" if post.locked_known or post.locked else "locked=posts.locked"
        clear_post_content = ["excluded.deleted", "excluded.removed"]
        if not post.deletion_known:
            clear_post_content.append("posts.deleted")
        if not post.removal_known:
            clear_post_content.append("posts.removed")
        clear_post_content = " OR ".join(clear_post_content)
        self.connection.execute(
            f"""INSERT INTO posts(post_id, fullname, subreddit, title, body, author, permalink, url, created_at, score, ups, upvote_ratio, num_comments, archived, locked, deleted, removed, observed_at, provider, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(post_id) DO UPDATE SET fullname=excluded.fullname, subreddit=COALESCE(excluded.subreddit, posts.subreddit), title=CASE WHEN {clear_post_content} THEN NULL ELSE COALESCE(excluded.title, posts.title) END, body=CASE WHEN {clear_post_content} THEN NULL ELSE COALESCE(excluded.body, posts.body) END, author=CASE WHEN {clear_post_content} THEN NULL ELSE COALESCE(excluded.author, posts.author) END, permalink=COALESCE(excluded.permalink, posts.permalink), url=COALESCE(excluded.url, posts.url), created_at=COALESCE(excluded.created_at, posts.created_at), score=COALESCE(excluded.score, posts.score), ups=COALESCE(excluded.ups, posts.ups), upvote_ratio=COALESCE(excluded.upvote_ratio, posts.upvote_ratio), num_comments=COALESCE(excluded.num_comments, posts.num_comments), {archived_update}, {locked_update}, {deleted_update}, {removed_update}, observed_at=excluded.observed_at, provider=excluded.provider, updated_at=excluded.updated_at""",
            (post.post_id, post.fullname or f"t3_{post.post_id}", post.subreddit, title, body, author, post.permalink, post.url, post.created_at, post.score, post.ups, post.upvote_ratio, post.num_comments, int(post.archived), int(post.locked), deleted, removed, post.observed_at, provider, post.observed_at),
        )

    def _save_observation(self, post: PostSnapshot, provider: str, result: PageResult | RefreshResult, request: RequestRecord | None = None) -> None:
        response_status = request.response_status if request else result.response_status
        request_id = request.request_id if request else result.request_id
        cache_status = request.cache_status if request else result.cache_status
        cache_observed_at = request.cache_observed_at if request else result.cache_observed_at
        metadata = {**request.metadata, **result.metadata} if request else result.metadata
        self.connection.execute(
            "INSERT INTO post_observations(post_id, observed_at, provider, source_created_at, score, ups, upvote_ratio, num_comments, response_status, request_id, cache_status, cache_observed_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (post.post_id, post.observed_at, provider, post.created_at, post.score, post.ups, post.upvote_ratio, post.num_comments, response_status, request_id, cache_status, cache_observed_at, json.dumps(metadata, sort_keys=True)),
        )

    def _save_comment(self, comment: CommentSnapshot, post_id: str, provider: str) -> None:
        deleted = int(comment.deleted)
        removed = int(comment.removed)
        body = None if deleted or removed else comment.body
        author = None if deleted or removed else comment.author
        deleted_update = "deleted=excluded.deleted" if comment.deletion_known or comment.deleted else "deleted=comments.deleted"
        removed_update = "removed=excluded.removed" if comment.removal_known or comment.removed else "removed=comments.removed"
        clear_comment_content = ["excluded.deleted", "excluded.removed"]
        if not comment.deletion_known:
            clear_comment_content.append("comments.deleted")
        if not comment.removal_known:
            clear_comment_content.append("comments.removed")
        clear_comment_content = " OR ".join(clear_comment_content)
        self.connection.execute(
            f"""INSERT INTO comments(comment_id, fullname, post_id, parent_id, author, body, permalink, created_at, score, ups, depth, deleted, removed, observed_at, provider, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(comment_id) DO UPDATE SET fullname=excluded.fullname, post_id=excluded.post_id, parent_id=COALESCE(excluded.parent_id, comments.parent_id), author=CASE WHEN {clear_comment_content} THEN NULL ELSE COALESCE(excluded.author, comments.author) END, body=CASE WHEN {clear_comment_content} THEN NULL ELSE COALESCE(excluded.body, comments.body) END, permalink=COALESCE(excluded.permalink, comments.permalink), created_at=COALESCE(excluded.created_at, comments.created_at), score=COALESCE(excluded.score, comments.score), ups=COALESCE(excluded.ups, comments.ups), depth=COALESCE(excluded.depth, comments.depth), {deleted_update}, {removed_update}, observed_at=excluded.observed_at, provider=excluded.provider, updated_at=excluded.updated_at""",
            (comment.comment_id, comment.fullname or f"t1_{comment.comment_id}", post_id, comment.parent_id, author, body, comment.permalink, comment.created_at, comment.score, comment.ups, comment.depth, deleted, removed, comment.observed_at, provider, comment.observed_at),
        )

    def _save_gap(self, run_id: str, provider: str, gap: Gap, observed_at: str, *, subreddit: str | None = None) -> None:
        self.connection.execute("INSERT INTO gaps(run_id, entity_type, entity_id, subreddit, reason, detail, observed_at, provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (run_id, gap.entity_type, gap.entity_id, gap.subreddit or subreddit, gap.reason, gap.detail, observed_at, provider))

    def _save_request_records(
        self,
        run_id: str,
        provider: str,
        records: list[RequestRecord],
        operation: str,
        request_id: str | None,
        status: int | None,
        cache_status: str | None,
        cache_observed_at: str | None,
        metadata: dict[str, Any],
    ) -> None:
        if records:
            for record in records:
                self._save_request(run_id, provider, record.operation, record.request_id, record.response_status, record.cache_status, record.cache_observed_at, record.metadata, record.billed, record.request_units)
        elif request_id is not None:
            self._save_request(run_id, provider, operation, request_id, status, cache_status, cache_observed_at, metadata)

    def _save_request(
        self,
        run_id: str,
        provider: str,
        operation: str,
        request_id: str | None,
        status: int | None,
        cache_status: str | None,
        cache_observed_at: str | None,
        metadata: dict[str, Any],
        billed: bool | None = None,
        request_units: int = 1,
    ) -> None:
        request_id = request_id or str(uuid.uuid4())
        request_metadata = dict(metadata)
        request_metadata.setdefault("request_units", max(1, request_units))
        if billed is None:
            billed = False if provider == "fixture" else False if cache_status == "cached" else None
        self.connection.execute("INSERT INTO requests(request_id, run_id, provider, operation, requested_at, response_status, billed, cache_status, cache_observed_at, metadata_json) VALUES (?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?)", (request_id, run_id, provider, operation, status, None if billed is None else int(billed), cache_status, cache_observed_at, json.dumps(request_metadata, sort_keys=True)))

    def due_posts(self, interval_minutes: int, limit: int) -> list[PostSnapshot]:
        rows = self.connection.execute("SELECT * FROM posts WHERE datetime(observed_at) <= datetime('now', ?) ORDER BY observed_at ASC LIMIT ?", (f"-{interval_minutes} minutes", limit)).fetchall()
        return [PostSnapshot(post_id=row["post_id"], fullname=row["fullname"], subreddit=row["subreddit"], title=row["title"], body=row["body"], author=row["author"], permalink=row["permalink"], url=row["url"], created_at=row["created_at"], score=row["score"], ups=row["ups"], upvote_ratio=row["upvote_ratio"], num_comments=row["num_comments"], archived=bool(row["archived"]), locked=bool(row["locked"]), deleted=bool(row["deleted"]), removed=bool(row["removed"]), observed_at=row["observed_at"]) for row in rows]

    def purge_deleted_content(self) -> int:
        post_count = self.connection.execute("UPDATE posts SET title=NULL, body=NULL, author=NULL WHERE deleted=1 OR removed=1").rowcount
        self.connection.execute("UPDATE comments SET body=NULL, author=NULL WHERE deleted=1 OR removed=1")
        self.connection.commit()
        return post_count

    def counts(self) -> dict[str, int]:
        return {name: int(self.connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) for name in ("posts", "post_observations", "comments", "gaps", "requests", "checkpoints")}
