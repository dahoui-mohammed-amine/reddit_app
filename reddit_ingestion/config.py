from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True, slots=True)
class Config:
    subreddits: tuple[str, ...]
    database_path: Path
    provider: str
    fixture_path: Path | None
    listing_limit: int
    max_discovery_pages: int
    refresh_interval_minutes: int
    max_refresh_posts: int
    comments_mode: str
    comment_depth: int
    comment_limit: int
    request_timeout_seconds: float
    max_retries: int
    request_interval_seconds: float = 0.25


def _path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    ingestion = raw.get("ingestion", {})
    comments = raw.get("comments", {})
    provider = raw.get("provider", {})
    subreddits = tuple(str(item).strip().removeprefix("r/") for item in ingestion.get("subreddits", []))
    if not subreddits:
        raise ValueError("[ingestion].subreddits must contain at least one subreddit")
    if any(not item for item in subreddits):
        raise ValueError("[ingestion].subreddits cannot contain empty names")

    listing_limit = int(ingestion.get("listing_limit", 100))
    if not 1 <= listing_limit <= 100:
        raise ValueError("[ingestion].listing_limit must be between 1 and 100")
    mode = str(comments.get("mode", "bounded")).lower()
    if mode not in {"bounded", "full"}:
        raise ValueError("[comments].mode must be bounded or full")
    return Config(
        subreddits=subreddits,
        database_path=_path(str(ingestion.get("database_path", "data/reddit.sqlite3")), config_path.parent)
        or (config_path.parent / "data/reddit.sqlite3").resolve(),
        provider=str(provider.get("name", "fixture")).lower(),
        fixture_path=_path(provider.get("fixture_path"), config_path.parent),
        listing_limit=listing_limit,
        max_discovery_pages=max(1, int(ingestion.get("max_discovery_pages", 1))),
        refresh_interval_minutes=max(1, int(ingestion.get("refresh_interval_minutes", 360))),
        max_refresh_posts=max(1, int(ingestion.get("max_refresh_posts", 100))),
        comments_mode=mode,
        comment_depth=max(0, int(comments.get("depth", 1))),
        comment_limit=max(1, int(comments.get("limit", 20))),
        request_timeout_seconds=max(1.0, float(provider.get("request_timeout_seconds", 30))),
        max_retries=max(0, int(provider.get("max_retries", 3))),
        request_interval_seconds=max(0.0, float(provider.get("min_request_interval_seconds", 0.25))),
    )
