from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CommentSnapshot:
    comment_id: str
    fullname: str | None = None
    post_id: str | None = None
    parent_id: str | None = None
    author: str | None = None
    body: str | None = None
    permalink: str | None = None
    created_at: str | None = None
    score: int | None = None
    ups: int | None = None
    deleted: bool = False
    removed: bool = False
    depth: int | None = None
    observed_at: str = ""
    deletion_known: bool = False
    removal_known: bool = False


@dataclass(slots=True)
class PostSnapshot:
    post_id: str
    fullname: str | None = None
    subreddit: str | None = None
    title: str | None = None
    body: str | None = None
    author: str | None = None
    permalink: str | None = None
    url: str | None = None
    created_at: str | None = None
    score: int | None = None
    ups: int | None = None
    upvote_ratio: float | None = None
    num_comments: int | None = None
    archived: bool = False
    locked: bool = False
    deleted: bool = False
    removed: bool = False
    observed_at: str = ""
    comments: list[CommentSnapshot] = field(default_factory=list)
    deletion_known: bool = False
    removal_known: bool = False
    archived_known: bool = False
    locked_known: bool = False
    refresh_until: str | None = None


@dataclass(slots=True)
class Gap:
    entity_type: str
    reason: str
    entity_id: str | None = None
    subreddit: str | None = None
    detail: str | None = None


@dataclass(slots=True)
class RequestRecord:
    request_id: str | None
    operation: str
    response_status: int | None = None
    cache_status: str | None = None
    cache_observed_at: str | None = None
    billed: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    request_units: int = 1


@dataclass(slots=True)
class PageResult:
    posts: list[PostSnapshot]
    requested_cursor: str | None
    next_cursor: str | None
    observed_at: str
    source_url: str | None
    response_status: int | None
    request_id: str | None
    cache_status: str | None = None
    cache_observed_at: str | None = None
    gaps: list[Gap] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    request_records: list[RequestRecord] = field(default_factory=list)


@dataclass(slots=True)
class RefreshResult:
    posts: list[PostSnapshot]
    observed_at: str
    request_id: str | None
    response_status: int | None
    cache_status: str | None = None
    cache_observed_at: str | None = None
    gaps: list[Gap] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    request_records: list[RequestRecord] = field(default_factory=list)
    observation_requests: dict[str, RequestRecord] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderStatus:
    name: str
    available: bool
    access_required: bool
    message: str
    paid_calls_possible: bool
    supports_refresh_batch: bool = False
    refresh_batch_size: int = 1
    capabilities: tuple[str, ...] = ()


@dataclass(slots=True)
class Plan:
    provider: str
    discovery_requests: int
    refresh_events: int
    estimated_requests: int | None
    estimated_units: float | None
    unit_label: str
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunSummary:
    run_id: str
    discovered: int = 0
    refreshed: int = 0
    comments: int = 0
    gaps: int = 0
    requests: int = 0
    status: str = "completed"
