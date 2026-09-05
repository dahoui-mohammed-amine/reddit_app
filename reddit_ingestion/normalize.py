from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .models import CommentSnapshot, PostSnapshot


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _value(raw: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _relative_permalink(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://www.reddit.com{value}" if value.startswith("/") else value


def _deleted(raw: Mapping[str, Any]) -> bool:
    authors = [raw[key] for key in ("author", "author_username") if key in raw]
    markers = [raw[key] for key in ("deleted", "is_deleted") if key in raw]
    contents = [raw[key] for key in ("body", "bodyText", "selftext", "text") if key in raw]
    return bool(any(markers)) or any(value in {"[deleted]", "deleted"} for value in authors + contents)


def _removed(raw: Mapping[str, Any]) -> bool:
    markers = [raw[key] for key in ("removed", "is_removed", "removed_by_category") if key in raw]
    contents = [raw[key] for key in ("body", "bodyText", "selftext") if key in raw]
    return bool(any(markers)) or any(value in {"[removed]", "removed"} for value in contents)


def _state_known(raw: Mapping[str, Any], marker_keys: tuple[str, ...], content_keys: tuple[str, ...], values: set[str]) -> bool:
    return any(key in raw for key in marker_keys) or any(raw[key] in values for key in content_keys if key in raw)


def post_id(raw: Mapping[str, Any]) -> str:
    value = _value(raw, "id", "post_id")
    if value:
        return str(value).removeprefix("t3_")
    fullname = _text(_value(raw, "fullname", "name"))
    if fullname and fullname.startswith("t3_"):
        return fullname[3:]
    raise ValueError("post is missing id or t3_ fullname")


def comment_id(raw: Mapping[str, Any]) -> str:
    value = _value(raw, "id", "comment_id")
    if value:
        return str(value).removeprefix("t1_")
    fullname = _text(_value(raw, "fullname", "name"))
    if fullname and fullname.startswith("t1_"):
        return fullname[3:]
    raise ValueError("comment is missing id or t1_ fullname")


def parse_comment(raw: Mapping[str, Any], *, post: PostSnapshot | None, observed_at: str) -> CommentSnapshot:
    cid = comment_id(raw)
    fullname = _text(_value(raw, "fullname", "name")) or f"t1_{cid}"
    return CommentSnapshot(
        comment_id=cid,
        fullname=fullname,
        post_id=post.post_id if post else _text(_value(raw, "post_id", "link_id")),
        parent_id=_text(_value(raw, "parent_id", "parentFullname")),
        author=_text(_value(raw, "author", "author_username")),
        body=_text(_value(raw, "body", "bodyText", "text")),
        permalink=_relative_permalink(_text(_value(raw, "permalink", "url"))),
        created_at=_text(_value(raw, "created_at_iso", "createdAt", "created_utc", "created")),
        score=_int(_value(raw, "score", "points")),
        ups=_int(_value(raw, "ups", "upvotes")),
        deleted=_deleted(raw),
        removed=_removed(raw),
        depth=_int(_value(raw, "depth")),
        observed_at=observed_at,
        deletion_known=_state_known(raw, ("deleted", "is_deleted"), ("author", "author_username", "body", "bodyText", "text"), {"[deleted]", "deleted"}),
        removal_known=_state_known(raw, ("removed", "is_removed", "removed_by_category"), ("body", "bodyText", "text"), {"[removed]", "removed"}),
    )


def parse_post(
    raw: Mapping[str, Any],
    *,
    default_subreddit: str | None = None,
    observed_at: str | None = None,
    include_comments: bool = True,
) -> PostSnapshot:
    observed = observed_at or utc_now()
    pid = post_id(raw)
    fullname = _text(_value(raw, "fullname", "name")) or f"t3_{pid}"
    post = PostSnapshot(
        post_id=pid,
        fullname=fullname,
        subreddit=_text(_value(raw, "subreddit", "community")) or default_subreddit,
        title=_text(_value(raw, "title")),
        body=_text(_value(raw, "selftext", "body", "bodyText")),
        author=_text(_value(raw, "author", "author_username")),
        permalink=_relative_permalink(_text(_value(raw, "permalink"))),
        url=_text(_value(raw, "url", "post_url")),
        created_at=_text(_value(raw, "created_at_iso", "createdAt", "created_utc", "created")),
        score=_int(_value(raw, "score")),
        ups=_int(_value(raw, "ups", "upvotes")),
        upvote_ratio=_float(_value(raw, "upvote_ratio", "upvoteRatio")),
        num_comments=_int(_value(raw, "num_comments", "commentCount", "comment_count")),
        archived=_bool(_value(raw, "archived")),
        locked=_bool(_value(raw, "locked")),
        deleted=_deleted(raw),
        removed=_removed(raw),
        observed_at=observed,
        deletion_known=_state_known(raw, ("deleted", "is_deleted"), ("author", "author_username", "body", "bodyText", "selftext", "text"), {"[deleted]", "deleted"}),
        removal_known=_state_known(raw, ("removed", "is_removed", "removed_by_category"), ("body", "bodyText", "selftext"), {"[removed]", "removed"}),
    )
    if include_comments:
        comments = _value(raw, "comments")
        if isinstance(comments, list):
            post.comments = [parse_comment(item, post=post, observed_at=observed) for item in comments if isinstance(item, Mapping)]
    return post
