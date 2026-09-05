from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import CommentSnapshot, Gap, PageResult, PostSnapshot, RefreshResult


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
    request_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    operation TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    response_status INTEGER,
    billed INTEGER,
    cache_status TEXT,
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
        self.connection.commit()

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
        self.connection.execute(
            "INSERT INTO checkpoints(subreddit, cursor, page_count, observed_at, provider, source_url) VALUES (?, ?, 1, ?, ?, ?) ON CONFLICT(subreddit) DO UPDATE SET cursor=excluded.cursor, page_count=checkpoints.page_count+1, observed_at=excluded.observed_at, provider=excluded.provider, source_url=excluded.source_url",
            (subreddit, page.next_cursor, page.observed_at, provider, page.source_url),
        )
        self._save_request(run_id, provider, "discover", page.request_id, page.response_status, page.cache_status, page.metadata)
        return len(page.posts), sum(len(post.comments) for post in page.posts)

    def save_refresh(self, run_id: str, provider: str, result: RefreshResult) -> tuple[int, int]:
        for post in result.posts:
            self._save_post(post, provider)
            self._save_observation(post, provider, result)
            if post.deleted or post.removed:
                result.gaps.append(Gap("post", "deleted" if post.deleted else "removed", entity_id=post.post_id, subreddit=post.subreddit))
            for comment in post.comments:
                self._save_comment(comment, post.post_id, provider)
                if comment.deleted or comment.removed:
                    result.gaps.append(Gap("comment", "deleted" if comment.deleted else "removed", entity_id=comment.comment_id, subreddit=post.subreddit))
        for gap in result.gaps:
            self._save_gap(run_id, provider, gap, result.observed_at)
        self._save_request(run_id, provider, "refresh", result.request_id, result.response_status, result.cache_status, result.metadata)
        return len(result.posts), sum(len(post.comments) for post in result.posts)

    def _save_post(self, post: PostSnapshot, provider: str) -> None:
        deleted = int(post.deleted)
        removed = int(post.removed)
        title = None if deleted or removed else post.title
        body = None if deleted or removed else post.body
        author = None if deleted or removed else post.author
        self.connection.execute(
            """INSERT INTO posts(post_id, fullname, subreddit, title, body, author, permalink, url, created_at, score, ups, upvote_ratio, num_comments, archived, locked, deleted, removed, observed_at, provider, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(post_id) DO UPDATE SET fullname=excluded.fullname, subreddit=COALESCE(excluded.subreddit, posts.subreddit), title=excluded.title, body=excluded.body, author=excluded.author, permalink=COALESCE(excluded.permalink, posts.permalink), url=COALESCE(excluded.url, posts.url), created_at=COALESCE(excluded.created_at, posts.created_at), score=excluded.score, ups=excluded.ups, upvote_ratio=excluded.upvote_ratio, num_comments=excluded.num_comments, archived=excluded.archived, locked=excluded.locked, deleted=excluded.deleted, removed=excluded.removed, observed_at=excluded.observed_at, provider=excluded.provider, updated_at=excluded.updated_at""",
            (post.post_id, post.fullname or f"t3_{post.post_id}", post.subreddit, title, body, author, post.permalink, post.url, post.created_at, post.score, post.ups, post.upvote_ratio, post.num_comments, int(post.archived), int(post.locked), deleted, removed, post.observed_at, provider, post.observed_at),
        )

    def _save_observation(self, post: PostSnapshot, provider: str, result: PageResult | RefreshResult) -> None:
        self.connection.execute(
            "INSERT INTO post_observations(post_id, observed_at, provider, source_created_at, score, ups, upvote_ratio, num_comments, response_status, request_id, cache_status, cache_observed_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (post.post_id, post.observed_at, provider, post.created_at, post.score, post.ups, post.upvote_ratio, post.num_comments, result.response_status, result.request_id, result.cache_status, result.cache_observed_at, json.dumps(result.metadata, sort_keys=True)),
        )

    def _save_comment(self, comment: CommentSnapshot, post_id: str, provider: str) -> None:
        deleted = int(comment.deleted)
        removed = int(comment.removed)
        body = None if deleted or removed else comment.body
        author = None if deleted or removed else comment.author
        self.connection.execute(
            """INSERT INTO comments(comment_id, fullname, post_id, parent_id, author, body, permalink, created_at, score, ups, depth, deleted, removed, observed_at, provider, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(comment_id) DO UPDATE SET fullname=excluded.fullname, post_id=excluded.post_id, parent_id=excluded.parent_id, author=excluded.author, body=excluded.body, permalink=COALESCE(excluded.permalink, comments.permalink), created_at=COALESCE(excluded.created_at, comments.created_at), score=excluded.score, ups=excluded.ups, depth=excluded.depth, deleted=excluded.deleted, removed=excluded.removed, observed_at=excluded.observed_at, provider=excluded.provider, updated_at=excluded.updated_at""",
            (comment.comment_id, comment.fullname or f"t1_{comment.comment_id}", post_id, comment.parent_id, author, body, comment.permalink, comment.created_at, comment.score, comment.ups, comment.depth, deleted, removed, comment.observed_at, provider, comment.observed_at),
        )

    def _save_gap(self, run_id: str, provider: str, gap: Gap, observed_at: str, *, subreddit: str | None = None) -> None:
        self.connection.execute("INSERT INTO gaps(run_id, entity_type, entity_id, subreddit, reason, detail, observed_at, provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (run_id, gap.entity_type, gap.entity_id, gap.subreddit or subreddit, gap.reason, gap.detail, observed_at, provider))

    def _save_request(self, run_id: str, provider: str, operation: str, request_id: str | None, status: int | None, cache_status: str | None, metadata: dict[str, Any]) -> None:
        request_id = request_id or str(uuid.uuid4())
        self.connection.execute("INSERT OR REPLACE INTO requests(request_id, run_id, provider, operation, requested_at, response_status, billed, cache_status, metadata_json) VALUES (?, ?, ?, ?, datetime('now'), ?, ?, ?, ?)", (request_id, run_id, provider, operation, status, None if cache_status == "cached" else 1, cache_status, json.dumps(metadata, sort_keys=True)))

    def due_posts(self, interval_minutes: int, limit: int) -> list[PostSnapshot]:
        rows = self.connection.execute("SELECT * FROM posts WHERE datetime(observed_at) <= datetime('now', ?) ORDER BY observed_at ASC LIMIT ?", (f"-{interval_minutes} minutes", limit)).fetchall()
        return [PostSnapshot(post_id=row["post_id"], fullname=row["fullname"], subreddit=row["subreddit"], title=row["title"], body=row["body"], author=row["author"], permalink=row["permalink"], url=row["url"], created_at=row["created_at"], score=row["score"], ups=row["ups"], upvote_ratio=row["upvote_ratio"], num_comments=row["num_comments"], archived=bool(row["archived"]), locked=bool(row["locked"]), deleted=bool(row["deleted"]), removed=bool(row["removed"]), observed_at=row["observed_at"]) for row in rows]

    def purge_deleted_content(self) -> int:
        post_count = self.connection.execute("UPDATE posts SET title=NULL, body=NULL, author=NULL WHERE deleted=1 OR removed=1").rowcount
        self.connection.execute("UPDATE comments SET body=NULL, author=NULL WHERE deleted=1 OR removed=1")
        self.connection.commit()
        return post_count

    def counts(self) -> dict[str, int]:
        return {name: int(self.connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) for name in ("posts", "post_observations", "comments", "gaps", "checkpoints")}
