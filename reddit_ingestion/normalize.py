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
    contents = [raw[key] for key in ("body", "bodyText", "selftext", "text") if key in raw]
    return bool(any(markers)) or any(value in {"[removed]", "removed"} for value in contents)


def _state_known(raw: Mapping[str, Any], marker_keys: tuple[str, ...], content_keys: tuple[str, ...], values: set[str]) -> bool:
    return any(key in raw and raw[key] is not None for key in marker_keys) or any(raw[key] in values for key in content_keys if key in raw)


def _normalized_aliases(raw: Mapping[str, Any], keys: tuple[str, ...], prefix: str, label: str, *, require_prefix: bool = False) -> str | None:
    values: list[str] = []
    for key in keys:
        if key not in raw or raw[key] is None:
            continue
        if not isinstance(raw[key], (str, int, float, bool)):
            raise ValueError(f"{label} must be scalar")
        value = str(raw[key])
        if not value:
            raise ValueError(f"{label} is empty")
        if require_prefix and not value.startswith(prefix):
            raise ValueError(f"{label} must use {prefix} fullname")
        normalized = value.removeprefix(prefix)
        if not normalized:
            raise ValueError(f"{label} is empty")
        values.append(normalized)
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError(f"{label} aliases disagree")
    return values[0] if values else None


def post_id(raw: Mapping[str, Any]) -> str:
    value = _normalized_aliases(raw, ("id", "post_id"), "t3_", "post id")
    fullname = _normalized_aliases(raw, ("fullname", "name"), "t3_", "post fullname", require_prefix=True)
    if value and fullname and value != fullname:
        raise ValueError("post id and fullname disagree")
    if value is not None:
        return value
    if fullname is not None:
        return fullname
    raise ValueError("post is missing id or t3_ fullname")


def comment_id(raw: Mapping[str, Any]) -> str:
    value = _normalized_aliases(raw, ("id", "comment_id"), "t1_", "comment id")
    fullname = _normalized_aliases(raw, ("fullname", "name"), "t1_", "comment fullname", require_prefix=True)
    if value and fullname and value != fullname:
        raise ValueError("comment id and fullname disagree")
    if value is not None:
        return value
    if fullname is not None:
        return fullname
    raise ValueError("comment is missing id or t1_ fullname")


def _comment_link_id(raw: Mapping[str, Any]) -> str | None:
    links = [post_id({"post_id": raw[key]}) for key in ("post_id", "link_id") if key in raw and raw[key] is not None]
    if links and any(link != links[0] for link in links[1:]):
        raise ValueError("comment post link aliases disagree")
    return links[0] if links else None


def _parent_id(raw: Mapping[str, Any]) -> str | None:
    values: list[str] = []
    for key in ("parent_id", "parentFullname"):
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError("comment parent id must be scalar")
        normalized = str(value)
        if len(normalized) <= 3 or not normalized.startswith(("t1_", "t3_")):
            raise ValueError("comment parent id must use a t1_ or t3_ fullname")
        values.append(normalized)
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError("comment parent id aliases disagree")
    return values[0] if values else None


def parse_comment(raw: Mapping[str, Any], *, post: PostSnapshot | None, observed_at: str) -> CommentSnapshot:
    cid = comment_id(raw)
    linked_post_id = _comment_link_id(raw)
    if post is not None and linked_post_id is not None and linked_post_id != post.post_id:
        raise ValueError("comment post link disagrees with requested post")
    fullname = _text(_value(raw, "fullname", "name")) or f"t1_{cid}"
    return CommentSnapshot(
        comment_id=cid,
        fullname=fullname,
        post_id=post.post_id if post else linked_post_id,
        parent_id=_parent_id(raw),
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
    comment_errors: list[Exception] | None = None,
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
        archived_known=raw.get("archived") is not None,
        locked_known=raw.get("locked") is not None,
    )
    if include_comments:
        comments = _value(raw, "comments")
        if isinstance(comments, list):
            seen_comment_ids: set[str] = set()
            for item in comments:
                if not isinstance(item, Mapping):
                    if comment_errors is not None:
                        comment_errors.append(TypeError("comment item is not an object"))
                    continue
                try:
                    comment = parse_comment(item, post=post, observed_at=observed)
                    if comment.comment_id not in seen_comment_ids:
                        seen_comment_ids.add(comment.comment_id)
                        post.comments.append(comment)
                except (TypeError, ValueError) as exc:
                    if comment_errors is not None:
                        comment_errors.append(exc)
        elif comments is not None and comment_errors is not None:
            comment_errors.append(TypeError("comment collection is not a list"))
    return post
