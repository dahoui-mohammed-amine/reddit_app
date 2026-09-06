from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from .comments import comment_count_gap
from .config import Config
from .models import (
    Gap,
    PageResult,
    Plan,
    PostSnapshot,
    ProviderStatus,
    RefreshResult,
    RequestRecord,
)
from .normalize import parse_comment, parse_post, post_id, utc_now


@dataclass(slots=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


@dataclass(slots=True)
class RequestAttempt:
    status: int | None
    billed: bool | None
    error: str | None = None


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        billed: bool | None = None,
        request_id: str | None = None,
        url: str | None = None,
        provider_payload: Any = None,
        provider_payload_present: bool = False,
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.billed = billed
        self.request_id = request_id
        self.url = url
        self.provider_payload = provider_payload
        self.provider_payload_present = provider_payload_present or provider_payload is not None
        self.attempts: list[RequestAttempt] = []


class JsonTransport(Protocol):
    def __call__(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> HttpResponse: ...


def urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> HttpResponse:
    request = Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, {k.lower(): v for k, v in response.headers.items()}, response.read())
    except HTTPError as exc:
        return HttpResponse(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read())
    except URLError as exc:
        raise ProviderError(f"network error: {exc.reason}", retryable=True) from exc
    except OSError as exc:
        raise ProviderError(f"transport error: {exc}", retryable=True) from exc


class JsonClient:
    def __init__(
        self,
        *,
        timeout: float,
        retries: int,
        transport: JsonTransport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        min_interval_seconds: float = 0.0,
        allow_non_object: bool = False,
    ):
        self.timeout = timeout
        self.retries = retries
        self.transport = transport
        self.sleep = sleep
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.allow_non_object = allow_non_object
        self._last_request_at: float | None = None
        self.last_attempts: list[RequestAttempt] = []

    def request(self, method: str, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any] | None = None) -> tuple[Any, HttpResponse, str]:
        body = json.dumps(payload).encode() if payload is not None else None
        request_id = str(uuid.uuid4())
        self.last_attempts = []
        request_headers = {"Accept": "application/json", "User-Agent": "reddit-ingestion/0.1", **headers}
        if payload is not None:
            request_headers["Content-Type"] = "application/json"
        for attempt in range(self.retries + 1):
            if self._last_request_at is not None and self.min_interval_seconds:
                elapsed = time.monotonic() - self._last_request_at
                self.sleep(max(0.0, self.min_interval_seconds - elapsed))
            self._last_request_at = time.monotonic()
            try:
                response = self.transport(method, url, request_headers, body, self.timeout)
            except (OSError, ProviderError) as exc:
                error = exc if isinstance(exc, ProviderError) else ProviderError(f"transport error: {exc}", retryable=True)
                self.last_attempts.append(RequestAttempt(error.status, error.billed, str(error)))
                error.url = error.url or url
                if not error.retryable or attempt >= self.retries:
                    error.request_id = error.request_id or request_id
                    error.attempts = list(self.last_attempts)
                    raise error
                self.sleep(2**attempt)
                continue
            self.last_attempts.append(RequestAttempt(response.status, None))
            if response.status in {408, 425, 429, 500, 502, 503, 504} and attempt < self.retries:
                retry_after = response.headers.get("retry-after")
                try:
                    delay = min(60.0, max(0.0, float(retry_after))) if retry_after else float(2**attempt)
                except ValueError:
                    delay = float(2**attempt)
                self.sleep(delay)
                continue
            try:
                parsed = json.loads(response.body.decode("utf-8")) if response.body else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.last_attempts[-1].billed = None
                error = ProviderError(
                    f"provider returned non-JSON response with status {response.status}",
                    status=response.status,
                    retryable=response.status >= 500,
                    request_id=request_id,
                    url=url,
                    provider_payload=response.body.decode("utf-8", errors="replace"),
                )
                error.attempts = list(self.last_attempts)
                raise error from exc
            if response.status >= 400:
                error = ProviderError(
                    f"provider returned HTTP {response.status}: {parsed.get('error', parsed) if isinstance(parsed, dict) else parsed}",
                    status=response.status,
                    retryable=response.status in {408, 425, 429, 500, 502, 503, 504},
                    billed=None,
                    request_id=request_id,
                    url=url,
                    provider_payload=parsed,
                    provider_payload_present=True,
                )
                error.attempts = list(self.last_attempts)
                raise error
            if not isinstance(parsed, dict) and not self.allow_non_object:
                self.last_attempts[-1].billed = None
                error = ProviderError(
                    "provider response must be a JSON object",
                    status=response.status,
                    request_id=request_id,
                    url=url,
                    provider_payload=parsed,
                    provider_payload_present=True,
                )
                error.attempts = list(self.last_attempts)
                raise error
            return parsed, response, request_id
        raise AssertionError("unreachable")


class Provider(Protocol):
    name: str

    def status(self) -> ProviderStatus: ...
    def plan(self, config: Config, known_posts: int, *, resume_pages: int = 0, mode: str = "run") -> Plan: ...
    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult: ...
    def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult: ...


def _run_refresh_capacity(config: Config, discovery: int, known_posts: int, mode: str) -> int:
    if mode == "run":
        return known_posts + discovery * config.listing_limit
    return known_posts


def _env_key(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _cache_info(payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    cache_observed_at = next(
        (
            str(payload[key])
            for key in ("cached_at", "cachedAt")
            if key in payload
            and payload[key] not in (None, "")
            and isinstance(payload[key], (str, int, float))
            and not isinstance(payload[key], bool)
        ),
        None,
    )
    if "cache_status" in payload:
        status = str(payload["cache_status"]).lower()
        return (status if status in {"cached", "live", "unknown"} else "unknown", cache_observed_at)
    if "cacheStatus" in payload:
        status = str(payload["cacheStatus"]).lower()
        return (status if status in {"cached", "live", "unknown"} else "unknown", cache_observed_at)
    if "cached" not in payload:
        return "unknown", cache_observed_at
    cached = payload.get("cached")
    if isinstance(cached, bool):
        return ("cached" if cached else "live"), cache_observed_at
    if isinstance(cached, str) and cached.lower() in {"true", "false"}:
        return ("cached" if cached.lower() == "true" else "live"), cache_observed_at
    return "unknown", cache_observed_at


def _request_record(
    operation: str,
    request_id: str | None,
    status: int | None,
    cache_status: str | None,
    cache_observed_at: str | None,
    metadata: dict[str, Any],
    *,
    billed: bool | None,
    request_units: int = 1,
) -> RequestRecord:
    return RequestRecord(request_id, operation, status, cache_status, cache_observed_at, billed, metadata, max(1, request_units))


def _billing(cache_status: str | None) -> bool | None:
    if cache_status == "cached":
        return False
    if cache_status == "live":
        return True
    return None


def _client_attempts(client: Any) -> list[RequestAttempt]:
    attempts = getattr(client, "last_attempts", None)
    return attempts if isinstance(attempts, list) else []


def _request_records(
    client: Any,
    operation: str,
    request_id: str,
    response: HttpResponse,
    cache_status: str | None,
    cache_observed_at: str | None,
    metadata: dict[str, Any],
    *,
    billed: bool | None,
    request_units: int = 1,
) -> list[RequestRecord]:
    attempts = _client_attempts(client)
    if not attempts:
        return [_request_record(operation, request_id, response.status, cache_status, cache_observed_at, metadata, billed=billed, request_units=request_units)]
    records: list[RequestRecord] = []
    for index, attempt in enumerate(attempts, start=1):
        attempt_metadata = {**metadata, "attempt": index, "attempt_count": len(attempts)}
        if attempt.error:
            attempt_metadata["error"] = attempt.error
        final = index == len(attempts)
        records.append(
            _request_record(
                operation,
                request_id,
                response.status if final else attempt.status,
                cache_status if final else "unknown",
                cache_observed_at if final else None,
                attempt_metadata,
                billed=billed if final else attempt.billed,
                request_units=request_units,
            )
        )
    return records


def _failed_request_records(client: Any, operation: str, url: str, error: ProviderError, metadata: dict[str, Any] | None = None) -> list[RequestRecord]:
    details = {"url": url, "error": str(error), "provider_payload_present": error.provider_payload_present}
    if error.provider_payload_present:
        details["raw_payload"] = error.provider_payload
        details["provider_error"] = error.provider_payload
    if metadata:
        details.update(metadata)
    attempts = error.attempts or _client_attempts(client)
    if not attempts:
        return [_request_record(operation, error.request_id, error.status, "unknown", None, details, billed=error.billed)] if error.request_id else []
    records: list[RequestRecord] = []
    for index, attempt in enumerate(attempts, start=1):
        attempt_metadata = {**details, "attempt": index, "attempt_count": len(attempts)}
        if attempt.error:
            attempt_metadata["error"] = attempt.error
        records.append(_request_record(operation, error.request_id, attempt.status, "unknown", None, attempt_metadata, billed=attempt.billed))
    return records


def _malformed_gap(entity_type: str, *, entity_id: str | None = None, subreddit: str | None = None, detail: str) -> Gap:
    return Gap(entity_type, "provider_error", entity_id=entity_id, subreddit=subreddit, detail=f"malformed provider payload: {detail}")


def _report_only_state_gaps(raw: Mapping[str, Any], *, entity_id: str, subreddit: str | None) -> list[Gap]:
    gaps: list[Gap] = []
    content_keys = ("author", "author_username", "user_posted", "body", "bodyText", "selftext", "text", "description", "comment")
    content_markers = {
        "deleted": {"[deleted]", "deleted"},
        "removed": {"[removed]", "removed"},
    }
    for reason, keys in (("deleted", ("deleted", "is_deleted")), ("removed", ("removed", "is_removed", "removed_by_category"))):
        markers = [key for key in keys if raw.get(key) not in (None, "", False)]
        markers.extend(key for key in content_keys if isinstance(raw.get(key), str) and raw[key] in content_markers[reason])
        if markers:
            gaps.append(
                Gap(
                    "post",
                    reason,
                    entity_id=entity_id,
                    subreddit=subreddit,
                    detail=f"Bright Data {reason} markers are report-only: {', '.join(markers)}",
                )
            )
    return gaps


def _parse_post_items(
    items: Any,
    *,
    default_subreddit: str | None,
    observed_at: str,
    include_comments: bool,
    detect_deletion_state: bool = True,
) -> tuple[list[PostSnapshot], list[Gap]]:
    if not isinstance(items, list):
        return [], [_malformed_gap("listing", subreddit=default_subreddit, detail="post collection is not a list")]
    posts: list[PostSnapshot] = []
    gaps: list[Gap] = []
    seen_post_ids: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            gaps.append(_malformed_gap("post", subreddit=default_subreddit, detail="post item is not an object"))
            continue
        comment_errors: list[Exception] = []
        try:
            post = parse_post(
                item,
                default_subreddit=default_subreddit,
                observed_at=observed_at,
                include_comments=include_comments,
                comment_errors=comment_errors,
                detect_deletion_state=detect_deletion_state,
            )
        except (TypeError, ValueError) as exc:
            gaps.append(_malformed_gap("post", subreddit=default_subreddit, detail=str(exc)))
            continue
        if post.post_id in seen_post_ids:
            continue
        if default_subreddit and post.subreddit and post.subreddit.casefold() != default_subreddit.casefold():
            gaps.append(
                _malformed_gap(
                    "post",
                    entity_id=post.post_id,
                    subreddit=default_subreddit,
                    detail=f"provider returned subreddit {post.subreddit!r} for requested {default_subreddit!r}",
                )
            )
            continue
        if default_subreddit:
            post.subreddit = default_subreddit
        seen_post_ids.add(post.post_id)
        posts.append(post)
        if not detect_deletion_state:
            gaps.extend(_report_only_state_gaps(item, entity_id=post.post_id, subreddit=post.subreddit))
        gaps.extend(
            _malformed_gap("comment", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc))
            for exc in comment_errors
        )
    return posts, gaps


def _parse_comment_items(items: Any, *, post: PostSnapshot, observed_at: str) -> tuple[list[Any], list[Gap]]:
    if not isinstance(items, list):
        return [], [_malformed_gap("comment", entity_id=post.post_id, subreddit=post.subreddit, detail="comment collection is not a list")]
    comments: list[Any] = []
    seen_comment_ids: set[str] = set()
    gaps: list[Gap] = []
    for item in items:
        if not isinstance(item, Mapping):
            gaps.append(_malformed_gap("comment", entity_id=post.post_id, subreddit=post.subreddit, detail="comment item is not an object"))
            continue
        try:
            comment = parse_comment(item, post=post, observed_at=observed_at)
            if comment.comment_id not in seen_comment_ids:
                seen_comment_ids.add(comment.comment_id)
                comments.append(comment)
        except (TypeError, ValueError) as exc:
            gaps.append(_malformed_gap("comment", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)))
    return comments, gaps


def _comments_complete(gaps: list[Gap]) -> bool:
    return not any(gap.entity_type == "comment" for gap in gaps)


def _id_mismatch_gap(entity_type: str, expected: str, actual: str, subreddit: str | None) -> Gap:
    return Gap(
        entity_type,
        "provider_error",
        entity_id=expected,
        subreddit=subreddit,
        detail=f"provider returned post id {actual!r} for requested post {expected!r}",
    )


class FixtureProvider:
    name = "fixture"

    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path
        with fixture_path.open(encoding="utf-8") as handle:
            self.fixture = json.load(handle)

    def status(self) -> ProviderStatus:
        return ProviderStatus("fixture", True, False, f"fixture data: {self.fixture_path}", False)

    def plan(self, config: Config, known_posts: int, *, resume_pages: int = 0, mode: str = "run") -> Plan:
        discovery = len(config.subreddits) * config.max_discovery_pages + resume_pages if mode in {"run", "discover"} else 0
        refreshable = _run_refresh_capacity(config, discovery, known_posts, mode)
        refresh = 1 if refreshable and mode in {"run", "refresh"} else 0
        refresh_events = min(refreshable, config.max_refresh_posts) if mode in {"run", "refresh"} else 0
        return Plan("fixture", discovery, refresh_events, discovery + refresh, 0, "free fixture calls", ["No network calls or credentials are used."])

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        observed = utc_now()
        pages = self.fixture.get("listings", {}).get(subreddit, [])
        selected = next((page for page in pages if page.get("request_after") == cursor), None)
        record = _request_record("discover", "fixture", 200, None, None, {"subreddit": subreddit, "cursor": cursor}, billed=False)
        if selected is None:
            return PageResult([], cursor, None, observed, f"fixture://{subreddit}/new", 200, "fixture", gaps=[Gap("listing", "fixture_page_missing", subreddit=subreddit, detail=f"cursor={cursor!r}")], metadata={"request_failed": True, "checkpoint_deferred": True}, request_records=[record])
        raw_posts = selected.get("posts")
        posts, parse_gaps = _parse_post_items(raw_posts, default_subreddit=subreddit, observed_at=observed, include_comments=True)
        gaps = [Gap(**gap) for gap in selected.get("gaps", [])] + parse_gaps
        source = self.fixture.get("refresh", {})
        for post in posts:
            raw = source.get(post.post_id) or source.get(post.fullname or "")
            if raw is None:
                if post.num_comments:
                    gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=subreddit, detail="fixture has no comment expansion data"))
                else:
                    count_gap = comment_count_gap(post)
                    if count_gap:
                        gaps.append(count_gap)
                continue
            try:
                comment_errors: list[Exception] = []
                expanded = parse_post(raw, default_subreddit=subreddit, observed_at=observed, comment_errors=comment_errors)
            except (TypeError, ValueError) as exc:
                gaps.append(_malformed_gap("comment", entity_id=post.post_id, subreddit=subreddit, detail=str(exc)))
                continue
            if expanded.post_id != post.post_id:
                gaps.append(_id_mismatch_gap("comment", post.post_id, expanded.post_id, subreddit))
                continue
            gaps.extend(
                _malformed_gap("comment", entity_id=post.post_id, subreddit=subreddit, detail=str(exc))
                for exc in comment_errors
            )
            post.comments = expanded.comments
            if post.num_comments and not post.comments:
                gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=subreddit, detail="fixture expansion returned no comments"))
            count_gap = comment_count_gap(post)
            if count_gap and not any(gap.entity_id == post.post_id and gap.reason == "unexpanded" for gap in gaps):
                gaps.append(count_gap)
        return PageResult(
            posts,
            cursor,
            selected.get("after"),
            observed,
            f"fixture://{subreddit}/new",
            200,
            "fixture",
            gaps=gaps,
            metadata={
                "comments_expanded": _comments_complete(gaps),
                "request_failed": not isinstance(raw_posts, list) or any(gap.entity_type == "post" for gap in parse_gaps),
            },
            request_records=[record],
        )

    def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
        observed = utc_now()
        source = self.fixture.get("refresh", {})
        refreshed: list[PostSnapshot] = []
        gaps: list[Gap] = []
        record = _request_record("refresh", "fixture", 200, None, None, {"post_count": len(posts)}, billed=False)
        for post in posts:
            raw = source.get(post.post_id) or source.get(post.fullname or "")
            if raw is None:
                gaps.append(Gap("post", "fixture_post_missing", entity_id=post.post_id, subreddit=post.subreddit))
                continue
            try:
                comment_errors: list[Exception] = []
                refreshed_post = parse_post(raw, default_subreddit=post.subreddit, observed_at=observed, comment_errors=comment_errors)
            except (TypeError, ValueError) as exc:
                gaps.append(_malformed_gap("post", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)))
                continue
            if refreshed_post.post_id != post.post_id:
                gaps.append(_id_mismatch_gap("post", post.post_id, refreshed_post.post_id, post.subreddit))
                continue
            gaps.extend(
                _malformed_gap("comment", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc))
                for exc in comment_errors
            )
            count_gap = comment_count_gap(refreshed_post, expected_num_comments=post.num_comments)
            if count_gap:
                gaps.append(count_gap)
            refreshed.append(refreshed_post)
        observations = {post.post_id: record for post in refreshed}
        return RefreshResult(refreshed, observed, "fixture", 200, gaps=gaps, metadata={"comments_mode": config.comments_mode, "comments_expanded": _comments_complete(gaps)}, request_records=[record], observation_requests=observations)


class HttpProviderBase:
    name = "http"
    key_env = ""
    base_url = ""

    def __init__(self, config: Config, client: JsonClient | None = None):
        self.config = config
        self.client = client or JsonClient(timeout=config.request_timeout_seconds, retries=config.max_retries, min_interval_seconds=config.request_interval_seconds)
        self.key = _env_key(self.key_env)

    def status(self) -> ProviderStatus:
        if not self.key:
            return ProviderStatus(self.name, False, True, f"missing {self.key_env}; no network call will be made", True, self.name == "redditapis", 100 if self.name == "redditapis" else 1)
        return ProviderStatus(self.name, True, True, f"credential found in {self.key_env}; live access is provider-billed", True, self.name == "redditapis", 100 if self.name == "redditapis" else 1)

    def _require_key(self) -> str:
        if not self.key:
            raise ProviderError(f"missing {self.key_env}; configure provider access before running", status=401)
        return self.key


class RedditApisProvider(HttpProviderBase):
    name = "redditapis"
    key_env = "REDDITAPIS_API_KEY"
    base_url = "https://api.redditapis.com"

    def plan(self, config: Config, known_posts: int, *, resume_pages: int = 0, mode: str = "run") -> Plan:
        discovery = len(config.subreddits) * config.max_discovery_pages + resume_pages if mode in {"run", "discover"} else 0
        refreshable = _run_refresh_capacity(config, discovery, known_posts, mode)
        refresh = min(refreshable, config.max_refresh_posts) if mode in {"run", "refresh"} else 0
        refresh_batches = (refresh + 99) // 100
        calls = discovery + refresh_batches
        discovered_capacity = discovery * config.listing_limit
        notes = []
        if discovery:
            notes.append("Discovery estimate includes the configured maximum listing pages.")
        if refresh:
            notes.append("Refresh uses documented batches of up to 100 t3_ fullnames.")
        if resume_pages and discovery:
            notes.append("Estimate includes one newest-page poll per resumable subreddit.")
        if discovery:
            notes.append("Comment request estimate uses the listing-limit upper bound for newly discovered posts.")
        if mode == "run" and discovery:
            notes.append("Run estimate includes possible refreshes for newly discovered posts.")
        notes.append(f"Estimated requests include up to {config.max_retries + 1} transport attempts per call.")
        if config.comments_mode == "full":
            notes.append("Full comment expansion has unknown, unbounded pagination cost; review provider billing before live execution.")
            return Plan(self.name, discovery, refresh, None, None, "USD reads", notes)
        request_count = (calls + discovered_capacity + refresh) * (config.max_retries + 1)
        return Plan(self.name, discovery, refresh, request_count, request_count * 0.002, "USD reads", notes)

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        key = self._require_key()
        params = {"subreddit": subreddit, "sort": "new", "limit": str(config.listing_limit)}
        if cursor:
            params["after"] = cursor
        url = f"{self.base_url}/api/reddit/posts?{urlencode(params)}"
        payload, response, request_id = self.client.request("GET", url, headers={"Authorization": f"Bearer {key}"})
        observed = utc_now()
        blocked = bool(payload.get("blocked"))
        cache_status, cache_observed_at = _cache_info(payload)
        raw_posts = payload.get("posts")
        posts, parse_gaps = ([], []) if blocked else _parse_post_items(raw_posts, default_subreddit=subreddit, observed_at=observed, include_comments=False)
        gaps: list[Gap] = parse_gaps
        if blocked:
            gaps.append(Gap("listing", "blocked", subreddit=subreddit, detail=str(payload.get("blockReason"))))
        records = _request_records(self.client, "discover", request_id, response, cache_status, cache_observed_at, {"url": url, "subreddit": subreddit, "cursor": cursor}, billed=_billing(cache_status))
        for post in posts:
            comments, comment_gaps, comment_records = self._fetch_comments(post, config, key)
            existing = {comment.comment_id: comment for comment in post.comments}
            existing.update({comment.comment_id: comment for comment in comments})
            post.comments = list(existing.values())
            gaps.extend(comment_gaps)
            records.extend(comment_records)
        return PageResult(
            posts,
            cursor,
            cursor if blocked else payload.get("after"),
            observed,
            url,
            response.status,
            request_id,
            cache_status,
            cache_observed_at,
            gaps,
            {
                "listing_status": payload.get("listing_status"),
                "exhausted_reason": payload.get("exhausted_reason"),
                "comments_expanded": _comments_complete(gaps),
                "blocked": blocked,
                "request_failed": not blocked and (not isinstance(raw_posts, list) or any(gap.entity_type == "post" for gap in parse_gaps)),
            },
            records,
        )

    def _fetch_comments(
        self,
        post: PostSnapshot,
        config: Config,
        key: str,
        *,
        expected_num_comments: int | None = None,
    ) -> tuple[list[Any], list[Gap], list[RequestRecord]]:
        comments: list[Any] = []
        gaps: list[Gap] = []
        records: list[RequestRecord] = []
        seen_comment_ids: set[str] = set()
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            if cursor is not None:
                if cursor in seen_cursors:
                    gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider repeated a previously seen comment cursor"))
                    break
                seen_cursors.add(cursor)
            params: dict[str, str] = {}
            if config.comments_mode == "bounded":
                params.update({"depth": str(config.comment_depth), "limit": str(config.comment_limit)})
            if cursor:
                params["after"] = cursor
            query = f"?{urlencode(params)}" if params else ""
            url = f"{self.base_url}/api/reddit/post/{quote(post.post_id, safe='')}/comments{query}"
            try:
                payload, response, request_id = self.client.request("GET", url, headers={"Authorization": f"Bearer {key}"})
            except ProviderError as exc:
                records.extend(_failed_request_records(self.client, "comments", url, exc, {"post_id": post.post_id, "mode": config.comments_mode}))
                gaps.append(Gap("comment", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)))
                break
            observed = utc_now()
            cache_status, cache_observed_at = _cache_info(payload)
            records.extend(_request_records(self.client, "comments", request_id, response, cache_status, cache_observed_at, {"url": url, "post_id": post.post_id, "mode": config.comments_mode}, billed=_billing(cache_status)))
            if payload.get("blocked"):
                gaps.append(Gap("comment", "blocked", entity_id=post.post_id, subreddit=post.subreddit, detail=str(payload.get("blockReason"))))
                break
            if "comments" not in payload:
                gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider response is missing comment collection"))
                break
            raw_comments = payload["comments"]
            if not isinstance(raw_comments, list):
                gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider returned no comment list"))
                break
            parsed_comments, parse_gaps = _parse_comment_items(raw_comments, post=post, observed_at=observed)
            for comment in parsed_comments:
                if comment.comment_id not in seen_comment_ids:
                    seen_comment_ids.add(comment.comment_id)
                    comments.append(comment)
            gaps.extend(parse_gaps)
            next_cursor = payload.get("after")
            incomplete = payload.get("listing_status") in {"truncated", "unknown"} or payload.get("remainingMoreCommentsCount")
            if config.comments_mode == "bounded":
                if next_cursor or incomplete:
                    gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="bounded provider response has additional or incomplete comments"))
                break
            if incomplete:
                gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="full provider response is incomplete"))
                break
            if not next_cursor:
                break
            cursor = str(next_cursor)
        post.comments = comments
        count_gap = comment_count_gap(post, expected_num_comments=expected_num_comments)
        if count_gap and not any(gap.entity_id == post.post_id and gap.reason == "unexpanded" for gap in gaps):
            gaps.append(count_gap)
        expected = post.num_comments if post.num_comments is not None else expected_num_comments
        if expected and not comments and not any(gap.entity_id == post.post_id and gap.reason in {"provider_error", "unexpanded"} for gap in gaps):
            gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider returned no comment evidence"))
        return comments, gaps, records

    def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
        key = self._require_key()
        all_posts: list[PostSnapshot] = []
        gaps: list[Gap] = []
        records: list[RequestRecord] = []
        observation_requests: dict[str, RequestRecord] = {}
        for offset in range(0, len(posts), 100):
            batch = posts[offset : offset + 100]
            fullnames = ",".join(post.fullname or f"t3_{post.post_id}" for post in batch)
            url = f"{self.base_url}/api/reddit/by_id/{quote(fullnames, safe=',_')}"
            try:
                payload, response, request_id = self.client.request("GET", url, headers={"Authorization": f"Bearer {key}"})
            except ProviderError as exc:
                records.extend(_failed_request_records(self.client, "refresh", url, exc, {"post_count": len(batch)}))
                gaps.extend(Gap("post", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)) for post in batch)
                continue
            cache_status, cache_observed_at = _cache_info(payload)
            batch_records = _request_records(self.client, "refresh", request_id, response, cache_status, cache_observed_at, {"url": url, "post_count": len(batch)}, billed=_billing(cache_status))
            records.extend(batch_records)
            record = batch_records[-1]
            if payload.get("blocked"):
                gaps.extend(Gap("post", "blocked", entity_id=post.post_id, subreddit=post.subreddit, detail=str(payload.get("blockReason"))) for post in batch)
                continue
            returned: dict[str, Mapping[str, Any]] = {}
            raw_posts = payload.get("posts", [])
            if not isinstance(raw_posts, list):
                gaps.append(_malformed_gap("post", subreddit=None, detail="post collection is not a list"))
                continue
            for raw in raw_posts:
                if not isinstance(raw, Mapping):
                    gaps.append(_malformed_gap("post", detail="post item is missing an ID"))
                    continue
                try:
                    returned[post_id(raw)] = raw
                except (TypeError, ValueError) as exc:
                    gaps.append(_malformed_gap("post", detail=str(exc)))
            for post in batch:
                raw = returned.get(post.post_id)
                if raw is None:
                    gaps.append(Gap("post", "not_returned", entity_id=post.post_id, subreddit=post.subreddit))
                else:
                    try:
                        refreshed = parse_post(raw, default_subreddit=post.subreddit, observed_at=utc_now(), include_comments=False)
                    except (TypeError, ValueError) as exc:
                        gaps.append(_malformed_gap("post", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)))
                        continue
                    observation_requests[post.post_id] = record
                    comments, comment_gaps, comment_records = self._fetch_comments(
                        refreshed,
                        config,
                        key,
                        expected_num_comments=post.num_comments,
                    )
                    refreshed.comments = comments
                    gaps.extend(comment_gaps)
                    records.extend(comment_records)
                    all_posts.append(refreshed)
        primary = next((record for record in records if record.operation == "refresh"), None)
        return RefreshResult(all_posts, utc_now(), primary.request_id if primary else None, primary.response_status if primary else None, primary.cache_status if primary else None, primary.cache_observed_at if primary else None, gaps, {"batch_count": (len(posts) + 99) // 100, "comments_mode": config.comments_mode, "comments_expanded": _comments_complete(gaps)}, records, observation_requests)


class FetchLayerProvider(HttpProviderBase):
    name = "fetchlayer"
    key_env = "FETCHLAYER_API_KEY"
    base_url = "https://api.fetchlayer.dev/reddit"

    def plan(self, config: Config, known_posts: int, *, resume_pages: int = 0, mode: str = "run") -> Plan:
        discovery = len(config.subreddits) * config.max_discovery_pages + resume_pages if mode in {"run", "discover"} else 0
        refreshable = _run_refresh_capacity(config, discovery, known_posts, mode)
        refresh = min(refreshable, config.max_refresh_posts) if mode in {"run", "refresh"} else 0
        discovered_capacity = discovery * config.listing_limit
        comment_expansions = discovered_capacity if config.comments_mode == "bounded" else 0
        calls = discovery + refresh + comment_expansions
        notes = []
        if refresh:
            notes.append("No documented multi-post refresh batch; each post URL is one call.")
        if discovery:
            notes.append("Discovery estimate includes the configured maximum listing pages.")
        if mode == "run" and discovery:
            notes.append("Run estimate includes possible refreshes for newly discovered posts.")
        if resume_pages and discovery:
            notes.append("Estimate includes one newest-page poll per resumable subreddit.")
        notes.append(f"Estimated requests include up to {config.max_retries + 1} transport attempts per call.")
        if config.comments_mode == "full":
            notes.append("Full comment expansion is unsupported by this adapter; metric refresh still runs and each post receives an explicit unexpanded gap.")
        request_count = calls * (config.max_retries + 1)
        return Plan(self.name, discovery, refresh, request_count, request_count * 0.00199, "USD requests", notes)

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        key = self._require_key()
        request_payload: dict[str, Any] = {"subreddit": subreddit, "sort": "new", "limit": config.listing_limit, "pages": 1}
        if cursor:
            request_payload["pageUrl"] = cursor
        payload, response, request_id = self.client.request(
            "POST",
            f"{self.base_url}/community-posts",
            headers={"Authorization": f"Bearer {key}"},
            payload=request_payload,
        )
        observed = utc_now()
        blocked = bool(payload.get("blocked"))
        raw_items = payload.get("items")
        posts, parse_gaps = ([], []) if blocked else _parse_post_items(raw_items, default_subreddit=subreddit, observed_at=observed, include_comments=False)
        gaps: list[Gap] = parse_gaps
        if blocked:
            gaps.append(Gap("listing", "blocked", subreddit=subreddit, detail=str(payload.get("blockReason"))))
        cache_status, cache_observed_at = _cache_info(payload)
        pages = payload.get("pagesRequested") or payload.get("pagesScraped") or request_payload["pages"]
        try:
            request_units = max(1, int(pages))
        except (TypeError, ValueError):
            request_units = 1
        records = _request_records(self.client, "discover", request_id, response, cache_status, cache_observed_at, {"url": f"{self.base_url}/community-posts", **request_payload}, billed=_billing(cache_status), request_units=request_units)
        if config.comments_mode == "full":
            for post in posts:
                post.comments = []
            gaps.extend(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=subreddit, detail="FetchLayer full comment expansion is unsupported") for post in posts)
        else:
            for post in posts:
                expansion = self._fetch_post(post, config, "comment_expansion", key)
                records.extend(expansion.request_records)
                gaps.extend(expansion.gaps)
                if expansion.posts:
                    expanded_post = expansion.posts[0]
                    post.comments = expanded_post.comments
                    for field in ("score", "ups", "upvote_ratio", "num_comments"):
                        value = getattr(expanded_post, field)
                        if getattr(post, field) is None and value is not None:
                            setattr(post, field, value)
                    count_gap = comment_count_gap(post)
                    if count_gap and not any(gap.entity_id == post.post_id and gap.reason == "unexpanded" for gap in gaps):
                        gaps.append(count_gap)
        return PageResult(
            posts,
            cursor,
            cursor if blocked else payload.get("nextPageUrl"),
            observed,
            payload.get("requestedUrl"),
            response.status,
            request_id,
            cache_status,
            cache_observed_at,
            gaps,
            {
                "pagesScraped": payload.get("pagesScraped"),
                "pagesRequested": payload.get("pagesRequested"),
                "listing_status": payload.get("listing_status"),
                "comments_expanded": _comments_complete(gaps),
                "blocked": blocked,
                "request_failed": not blocked and (not isinstance(raw_items, list) or any(gap.entity_type == "post" for gap in parse_gaps)),
            },
            records,
        )

    def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
        refreshed: list[PostSnapshot] = []
        gaps: list[Gap] = []
        records: list[RequestRecord] = []
        observation_requests: dict[str, RequestRecord] = {}
        key = self._require_key()
        for post in posts:
            result = self._fetch_post(post, config, "refresh", key)
            refreshed.extend(result.posts)
            gaps.extend(result.gaps)
            records.extend(result.request_records)
            observation_requests.update(result.observation_requests)
        primary = next((record for record in records if record.operation == "refresh"), None)
        return RefreshResult(refreshed, utc_now(), primary.request_id if primary else None, primary.response_status if primary else None, primary.cache_status if primary else None, primary.cache_observed_at if primary else None, gaps, {"comments_mode": config.comments_mode, "comments_expanded": _comments_complete(gaps)}, records, observation_requests)

    def _fetch_post(self, post: PostSnapshot, config: Config, operation: str, key: str) -> RefreshResult:
        url = post.permalink or post.url
        if not url:
            entity_type = "comment" if operation == "comment_expansion" else "post"
            return RefreshResult([], utc_now(), None, None, None, None, gaps=[Gap(entity_type, "missing_permalink", entity_id=post.post_id, subreddit=post.subreddit)])
        payload: dict[str, Any] = {"url": url, "pages": 1, "depth": 0}
        if config.comments_mode == "bounded":
            payload.update({"depth": config.comment_depth, "commentLimit": config.comment_limit})
        request_url = f"{self.base_url}/post"
        try:
            response_payload, response, request_id = self.client.request(
                "POST",
                request_url,
                headers={"Authorization": f"Bearer {key}"},
                payload=payload,
            )
        except ProviderError as exc:
            records = _failed_request_records(self.client, operation, request_url, exc, {"post_id": post.post_id, "url": url, "payload": payload})
            entity_type = "comment" if operation == "comment_expansion" else "post"
            return RefreshResult([], utc_now(), exc.request_id, exc.status, "unknown", None, gaps=[Gap(entity_type, "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc))], request_records=records)
        cache_status, cache_observed_at = _cache_info(response_payload)
        pages = response_payload.get("pagesRequested") or response_payload.get("pagesScraped") or 1
        try:
            request_units = max(1, int(pages))
        except (TypeError, ValueError):
            request_units = 1
        records = _request_records(self.client, operation, request_id, response, cache_status, cache_observed_at, {"url": request_url, "post_id": post.post_id, "payload": payload}, billed=_billing(cache_status), request_units=request_units)
        record = records[-1]
        if response_payload.get("blocked"):
            entity_type = "comment" if operation == "comment_expansion" else "post"
            return RefreshResult([], utc_now(), request_id, response.status, cache_status, cache_observed_at, gaps=[Gap(entity_type, "blocked", entity_id=post.post_id, subreddit=post.subreddit, detail=str(response_payload.get("blockReason")))], request_records=records)
        try:
            refreshed_post = parse_post(response_payload, default_subreddit=post.subreddit, observed_at=utc_now(), include_comments=False)
        except (TypeError, ValueError) as exc:
            entity_type = "comment" if operation == "comment_expansion" else "post"
            return RefreshResult([], utc_now(), request_id, response.status, cache_status, cache_observed_at, gaps=[_malformed_gap(entity_type, entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc))], request_records=records)
        if refreshed_post.post_id != post.post_id:
            entity_type = "comment" if operation == "comment_expansion" else "post"
            return RefreshResult(
                [],
                utc_now(),
                request_id,
                response.status,
                cache_status,
                cache_observed_at,
                gaps=[_id_mismatch_gap(entity_type, post.post_id, refreshed_post.post_id, post.subreddit)],
                request_records=records,
            )
        result_gaps: list[Gap] = []
        if config.comments_mode == "full":
            result_gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="FetchLayer full comment expansion is unsupported"))
        elif "comments" not in response_payload or not isinstance(response_payload.get("comments"), list):
            result_gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider returned no comment collection"))
        else:
            refreshed_post.comments, comment_gaps = _parse_comment_items(response_payload["comments"], post=refreshed_post, observed_at=refreshed_post.observed_at)
            result_gaps.extend(comment_gaps)
            if response_payload.get("nextPageUrl") or response_payload.get("remainingMoreCommentsCount") or response_payload.get("listing_status") in {"truncated", "unknown"} or comment_count_gap(refreshed_post, expected_num_comments=post.num_comments):
                result_gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider returned additional or incomplete comments"))
        return RefreshResult([refreshed_post], utc_now(), request_id, response.status, cache_status, cache_observed_at, gaps=result_gaps, request_records=records, observation_requests={post.post_id: record})


class BrightDataProvider(HttpProviderBase):
    """Opt-in adapter for Bright Data's documented Reddit Scraper API.

    Bright Data's sync API has no documented cursor.  Discovery therefore makes
    one request for a batch of subreddit URLs and never turns a returned URL or
    snapshot into an invented follow-up page.  Comment expansion is unsupported
    because the documented comment records provide neither depth nor a count cap.
    """

    name = "brightdata"
    key_env = "BRIGHTDATA_API_KEY"
    base_url = "https://api.brightdata.com"
    posts_dataset_id = "gd_lvz8ah06191smkebj4"
    comments_dataset_id = "gd_lvzdpsdlw09j6t702"
    max_sync_inputs = 20
    reddit_hosts = frozenset({"www.reddit.com", "reddit.com", "old.reddit.com", "redd.it"})

    def __init__(self, config: Config, client: JsonClient | None = None):
        self.config = config
        # A timeout, ambiguous response, 429, or 500 must not be replayed:
        # collection may be billable even when no record is delivered.
        self.client = client or JsonClient(
            timeout=config.request_timeout_seconds,
            retries=0,
            min_interval_seconds=config.request_interval_seconds,
            allow_non_object=True,
        )
        if isinstance(self.client, JsonClient):
            self.client.retries = 0
            self.client.allow_non_object = True
        self.key = _env_key(self.key_env)
        self._discovery_pages: dict[str, PageResult] = {}
        self._discovery_served: set[str] = set()

    def status(self) -> ProviderStatus:
        if not self.key:
            message = f"missing {self.key_env}; no network call will be made"
            available = False
        else:
            message = f"credential found in {self.key_env}; live access is provider-billed"
            available = True
        return ProviderStatus(
            self.name,
            available,
            True,
            message,
            True,
            supports_refresh_batch=True,
            refresh_batch_size=self.max_sync_inputs,
            capabilities=("discover_subreddit", "collect_post"),
        )

    def plan(self, config: Config, known_posts: int, *, resume_pages: int = 0, mode: str = "run") -> Plan:
        if mode not in {"run", "discover", "refresh"}:
            raise ValueError("mode must be run, discover, or refresh")
        subreddit_inputs = len(config.subreddits) if mode in {"run", "discover"} else 0
        discovery = (subreddit_inputs + self.max_sync_inputs - 1) // self.max_sync_inputs
        refreshable = _run_refresh_capacity(config, subreddit_inputs, known_posts, mode)
        refresh_events = min(refreshable, config.max_refresh_posts) if mode in {"run", "refresh"} else 0
        post_requests = (refresh_events + self.max_sync_inputs - 1) // self.max_sync_inputs
        comment_requests = 0
        notes = [
            "Bright Data sync accepts at most 20 input URLs; each request is one request unit, not a record estimate.",
            "Bright Data record billing/credits are not inferred; inspect returned record and error counts.",
            "No documented Reddit cursor or pagination is used; max_discovery_pages beyond the first is unsupported.",
            "Bright Data requests are sent with the documented {input: [...]} JSON body and include_errors=true.",
            "No automatic retries are used, including for 429, 500, timeout, or ambiguous responses.",
        ]
        if resume_pages:
            notes.append("Existing cursors are not Bright Data pagination; the adapter performs a fresh discovery and does not resume them.")
        if config.comments_mode == "full":
            notes.append("Full comment expansion is unsupported: the comments dataset has no documented count or pagination bound.")
        else:
            notes.append("Bounded comments are unsupported: documented comment records expose no depth and days_back is not a count cap; no comments request is made.")
        estimated_requests: int | None = discovery + post_requests + comment_requests
        if mode == "refresh":
            estimated_requests = post_requests + comment_requests
        if mode == "discover":
            estimated_requests = discovery
        return Plan(
            self.name,
            discovery,
            refresh_events,
            estimated_requests,
            None,
            "provider requests; records unpriced",
            notes,
        )

    @staticmethod
    def _subreddit_key(subreddit: str) -> str:
        return subreddit.removeprefix("r/").casefold()

    @classmethod
    def _absolute_reddit_url(cls, value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip()
        if "://" in value and not value.startswith(("http://", "https://")):
            return None
        candidate = value if value.startswith(("http://", "https://")) else f"https://www.reddit.com{value if value.startswith('/') else '/' + value}"
        try:
            parsed = urlparse(candidate)
            hostname = parsed.hostname
        except ValueError:
            return None
        if parsed.scheme.casefold() not in {"http", "https"} or hostname is None or hostname.casefold() not in cls.reddit_hosts:
            return None
        return candidate

    @classmethod
    def _canonical_url(cls, value: Any) -> str | None:
        absolute = cls._absolute_reddit_url(value)
        return absolute.rstrip("/").casefold() if absolute else None

    @classmethod
    def _post_url_id(cls, value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        absolute = cls._absolute_reddit_url(value)
        if absolute is None:
            return None
        path = urlparse(absolute).path.strip("/").split("/")
        try:
            comments_index = next(index for index, item in enumerate(path) if item.casefold() == "comments")
        except StopIteration:
            return None
        if comments_index + 1 >= len(path) or not path[comments_index + 1]:
            return None
        return path[comments_index + 1].removeprefix("t3_").casefold()

    @classmethod
    def _subreddit_from_value(cls, value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip()
        if "://" in value or value.startswith("/"):
            absolute = cls._absolute_reddit_url(value)
            if absolute is None:
                return None
            path = urlparse(absolute).path.strip("/").split("/")
            try:
                index = next(index for index, item in enumerate(path) if item.casefold() == "r")
            except StopIteration:
                return None
            return path[index + 1] if index + 1 < len(path) and path[index + 1] else None
        value = value.strip("/")
        return value[2:] if value.casefold().startswith("r/") else value

    @classmethod
    def _record_urls_valid(cls, raw: Mapping[str, Any]) -> bool:
        values = [raw.get(key) for key in ("community_url", "subreddit_url", "url", "post_url", "permalink") if raw.get(key)]
        return all(cls._absolute_reddit_url(value) is not None for value in values)

    @classmethod
    def _record_subreddits(cls, raw: Mapping[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("community_url", "subreddit_url", "url", "post_url", "permalink", "community_name", "subreddit", "community"):
            subreddit = cls._subreddit_from_value(raw.get(key))
            if subreddit:
                values.append(subreddit)
        return values

    @classmethod
    def _record_subreddit(cls, raw: Mapping[str, Any]) -> str | None:
        values = cls._record_subreddits(raw)
        if values and len({cls._subreddit_key(value) for value in values}) == 1:
            return values[0]
        return None

    @staticmethod
    def _response_shape(payload: Any) -> str:
        if isinstance(payload, list):
            return "array"
        if isinstance(payload, dict):
            if payload.get("snapshot_id"):
                return "snapshot"
            return "object"
        return type(payload).__name__

    @classmethod
    def _is_deferred_payload(cls, payload: Any) -> bool:
        return isinstance(payload, list) or cls._response_shape(payload) == "snapshot"

    @classmethod
    def _records_and_errors(cls, payload: Any) -> tuple[list[Any] | None, list[Any], str]:
        if isinstance(payload, list):
            return payload, [], "array"
        if not isinstance(payload, dict):
            return None, [], cls._response_shape(payload)
        if payload.get("snapshot_id"):
            return None, [], "snapshot"
        return None, [], "object"

    @staticmethod
    def _error_text(error: Any) -> str:
        try:
            return json.dumps(error, sort_keys=True)
        except (TypeError, ValueError):
            return str(error)

    @classmethod
    def _error_matches(cls, error: Any, target_url: str | None) -> bool:
        if not isinstance(error, Mapping):
            return True
        candidates: list[Any] = []
        for key in ("url", "input", "requested_url", "original_url"):
            value = error.get(key)
            if isinstance(value, Mapping):
                candidates.extend(value.get(key_name) for key_name in ("url", "input") if value.get(key_name))
            elif value:
                candidates.append(value)
        if not candidates:
            return True
        target = cls._canonical_url(target_url)
        return target is not None and any(cls._canonical_url(candidate) == target for candidate in candidates)

    @classmethod
    def _unavailable_like(cls, error: Any, status: int | None = None) -> bool:
        if status == 404:
            return True
        if isinstance(error, Mapping):
            text = " ".join(
                cls._error_text(error[key])
                for key in ("error", "message", "reason", "code", "status", "status_code", "error_code")
                if key in error
            ).casefold()
        else:
            text = cls._error_text(error).casefold()
        return any(term in text for term in ("deleted", "removed", "not found", "unavailable", "private", "restricted", "404"))

    @classmethod
    def _provider_gap(cls, entity_type: str, *, entity_id: str | None, subreddit: str | None, error: Any, status: int | None = None) -> Gap:
        reason = "unavailable" if cls._unavailable_like(error, status) else "provider_error"
        return Gap(entity_type, reason, entity_id=entity_id, subreddit=subreddit, detail=f"Bright Data provider error: {cls._error_text(error)}")

    @classmethod
    def _request_metadata(
        cls,
        *,
        request_url: str,
        dataset_id: str,
        request_payload: Mapping[str, Any],
        fetched_at: str,
        raw_payload: Any,
        returned_records: int,
        provider_error_count: int,
    ) -> dict[str, Any]:
        return {
            "url": request_url,
            "dataset_id": dataset_id,
            "request_payload": request_payload,
            "fetched_at": fetched_at,
            "raw_payload": raw_payload,
            "response_shape": cls._response_shape(raw_payload),
            "requested_inputs": len(request_payload.get("input", [])),
            "returned_records": returned_records,
            "provider_error_count": provider_error_count,
            "accounting": {
                "requested_inputs": len(request_payload.get("input", [])),
                "returned_records": returned_records,
                "provider_error_count": provider_error_count,
                "request_units": 1,
                "billing_semantics": "undocumented",
            },
        }

    @staticmethod
    def _error_list_for_target(errors: list[Any], target_url: str | None) -> list[Any]:
        specific = [error for error in errors if isinstance(error, Mapping) and not BrightDataProvider._error_matches(error, None)]
        if specific:
            matched = [error for error in specific if BrightDataProvider._error_matches(error, target_url)]
            return matched
        return [error for error in errors if BrightDataProvider._error_matches(error, target_url)]

    def _request(
        self,
        *,
        dataset_id: str,
        query: Mapping[str, str],
        request_payload: Mapping[str, Any],
        operation: str,
    ) -> tuple[Any | None, HttpResponse | None, str | None, list[RequestRecord], str, ProviderError | None]:
        key = self._require_key()
        params = {"dataset_id": dataset_id, "include_errors": "true", **query}
        request_url = f"{self.base_url}/datasets/v3/scrape?{urlencode(params)}"
        fetched_at = utc_now()
        try:
            payload, response, request_id = self.client.request(
                "POST",
                request_url,
                headers={"Authorization": f"Bearer {key}"},
                payload=request_payload,
            )
        except ProviderError as exc:
            records = _failed_request_records(
                self.client,
                operation,
                request_url,
                exc,
                {
                    "dataset_id": dataset_id,
                    "request_payload": request_payload,
                    "fetched_at": fetched_at,
                    "requested_inputs": len(request_payload.get("input", [])),
                    "returned_records": 0,
                    "provider_error_count": 1,
                    "accounting": {
                        "requested_inputs": len(request_payload.get("input", [])),
                        "returned_records": 0,
                        "provider_error_count": 1,
                        "request_units": 1,
                        "billing_semantics": "undocumented",
                    },
                },
            )
            return None, None, exc.request_id, records, fetched_at, exc
        records = _request_records(
            self.client,
            operation,
            request_id,
            response,
            "unknown",
            None,
            self._request_metadata(
                request_url=request_url,
                dataset_id=dataset_id,
                request_payload=request_payload,
                fetched_at=fetched_at,
                raw_payload=payload,
                returned_records=len(payload) if isinstance(payload, list) else 0,
                provider_error_count=0,
            ),
            billed=None,
        )
        return payload, response, request_id, records, fetched_at, None

    @staticmethod
    def _update_accounting(records: list[RequestRecord], *, returned_records: int, provider_error_count: int) -> None:
        for record in records:
            record.metadata["returned_records"] = returned_records
            record.metadata["provider_error_count"] = provider_error_count
            accounting = record.metadata.setdefault("accounting", {})
            accounting["returned_records"] = returned_records
            accounting["provider_error_count"] = provider_error_count

    def _cursor_unsupported(self, subreddit: str, cursor: str) -> PageResult:
        observed = utc_now()
        return PageResult(
            [],
            cursor,
            cursor,
            observed,
            None,
            None,
            None,
            gaps=[Gap("listing", "unsupported", subreddit=subreddit, detail="Bright Data documents sort/date/limit controls, not a Reddit pagination cursor")],
            metadata={"request_failed": True, "checkpoint_deferred": True, "capability": "pagination_unsupported", "fetched_at": observed},
        )

    def _batch_for(self, subreddit: str, config: Config) -> list[str]:
        requested = self._subreddit_key(subreddit)
        configured = list(config.subreddits)
        for offset in range(0, len(configured), self.max_sync_inputs):
            batch = configured[offset : offset + self.max_sync_inputs]
            if any(self._subreddit_key(item) == requested for item in batch):
                return batch
        return [subreddit]

    def _fetch_discovery_batch(self, batch: list[str], config: Config) -> None:
        inputs = [{"url": f"https://www.reddit.com/r/{subreddit}/", "sort_by": "new"} for subreddit in batch]
        request_payload = {"input": inputs}
        query = {"type": "discover_new", "discover_by": "subreddit_url"}
        payload, response, request_id, request_records, fetched_at, error = self._request(
            dataset_id=self.posts_dataset_id,
            query=query,
            request_payload=request_payload,
            operation="discover",
        )
        by_subreddit: dict[str, list[Mapping[str, Any]]] = {self._subreddit_key(subreddit): [] for subreddit in batch}
        errors: list[Any] = []
        shared_error: Any = None
        snapshot_id: str | None = None
        malformed_shape: str | None = None
        deferred_202 = response is not None and response.status == 202 and self._is_deferred_payload(payload)
        unattributed_record = False
        returned_count = 0
        response_status = response.status if response else error.status if error else None
        if error is not None:
            shared_error = error.provider_payload if error.provider_payload is not None else str(error)
        elif deferred_202:
            self._update_accounting(request_records, returned_records=0, provider_error_count=0)
            if isinstance(payload, dict):
                snapshot_id = payload.get("snapshot_id")
        else:
            raw_records, errors, shape = self._records_and_errors(payload)
            returned_count = len(raw_records or [])
            self._update_accounting(request_records, returned_records=returned_count, provider_error_count=len(errors))
            if shape == "snapshot":
                snapshot_id = payload.get("snapshot_id") if isinstance(payload, dict) else None
            elif raw_records is None:
                malformed_shape = shape
            else:
                for raw in raw_records:
                    if not isinstance(raw, Mapping):
                        unattributed_record = True
                        continue
                    if "error" in raw and not any(key in raw for key in ("id", "post_id", "fullname", "name")):
                        errors.append(raw)
                        continue
                    if not self._record_urls_valid(raw):
                        unattributed_record = True
                        continue
                    subreddit = self._record_subreddit(raw)
                    key = self._subreddit_key(subreddit) if subreddit else None
                    if key not in by_subreddit:
                        unattributed_record = True
                        continue
                    by_subreddit[key].append(raw)
                self._update_accounting(request_records, returned_records=returned_count, provider_error_count=len(errors))
        batch_urls = [f"https://www.reddit.com/r/{subreddit}/" for subreddit in batch]
        unmatched_specific_errors = [
            item
            for item in errors
            if isinstance(item, Mapping)
            and not self._error_matches(item, None)
            and not any(self._error_matches(item, target_url) for target_url in batch_urls)
        ]
        for key, raw_items in by_subreddit.items():
            subreddit = next(item for item in batch if self._subreddit_key(item) == key)
            target_url = f"https://www.reddit.com/r/{subreddit}/"
            target_errors = self._error_list_for_target(errors, target_url) if error is None else []
            target_errors.extend(unmatched_specific_errors)
            posts, parse_gaps = _parse_post_items(
                [dict(raw, subreddit=subreddit) for raw in raw_items],
                default_subreddit=subreddit,
                observed_at=fetched_at,
                include_comments=False,
                detect_deletion_state=False,
            )
            if config.comments_mode in {"bounded", "full"}:
                parse_gaps.extend(self._comments_unsupported_gap(post, config.comments_mode) for post in posts)
            gaps = parse_gaps
            if shared_error is not None:
                gaps.append(self._provider_gap("listing", entity_id=None, subreddit=subreddit, error=shared_error, status=error.status if error else None))
            elif deferred_202:
                detail = "Bright Data returned deferred HTTP 202"
                if snapshot_id is not None:
                    detail += f" with async snapshot {snapshot_id!r}"
                detail += "; snapshot polling is not part of this ingestion seam"
                gaps.append(Gap("listing", "unsupported", subreddit=subreddit, detail=detail))
            elif snapshot_id is not None:
                gaps.append(Gap("listing", "unsupported", subreddit=subreddit, detail=f"Bright Data returned async snapshot {snapshot_id!r}; snapshot polling is not part of this ingestion seam"))
            elif malformed_shape is not None:
                gaps.append(Gap("listing", "provider_error", subreddit=subreddit, detail=f"malformed provider payload: expected a JSON array, got {malformed_shape}"))
            if unattributed_record:
                gaps.append(Gap("listing", "provider_error", subreddit=subreddit, detail="provider returned a discovery record that could not be attributed to a requested subreddit"))
            gaps.extend(self._provider_gap("listing", entity_id=None, subreddit=subreddit, error=item) for item in target_errors)
            request_failed = error is not None or bool(raw_items and any(gap.entity_type == "post" for gap in parse_gaps)) or bool(target_errors) or any(gap.entity_type == "listing" and gap.reason in {"provider_error", "unavailable", "unsupported"} for gap in gaps)
            metadata = {
                "dataset_id": self.posts_dataset_id,
                "fetched_at": fetched_at,
                "requested_inputs": len(batch),
                "returned_records": returned_count,
                "provider_error_count": len(errors),
                "accounting": {
                    "requested_inputs": len(batch),
                    "returned_records": returned_count,
                    "provider_error_count": len(errors),
                    "request_units": 1,
                    "billing_semantics": "undocumented",
                },
                "comments_expanded": not any(gap.entity_type == "comment" for gap in gaps),
                "request_failed": bool(request_failed),
                "checkpoint_deferred": bool(request_failed),
            }
            if payload is not None:
                metadata["raw_payload"] = payload
                metadata["response_shape"] = self._response_shape(payload)
            elif error is not None and error.provider_payload is not None:
                metadata["raw_payload"] = error.provider_payload
                metadata["provider_error"] = error.provider_payload
                metadata["response_shape"] = self._response_shape(error.provider_payload)
            if snapshot_id is not None:
                metadata["async_snapshot_id"] = snapshot_id
            first = self._subreddit_key(batch[0]) == key
            self._discovery_pages[key] = PageResult(
                posts,
                None,
                None,
                fetched_at,
                f"{self.base_url}/datasets/v3/scrape",
                response_status,
                request_id if first else None,
                "unknown",
                None,
                gaps,
                metadata,
                request_records if first else [],
            )

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        if cursor is not None:
            return self._cursor_unsupported(subreddit, cursor)
        key = self._subreddit_key(subreddit)
        if key not in self._discovery_pages or key in self._discovery_served:
            batch = self._batch_for(subreddit, config)
            for item in batch:
                self._discovery_pages.pop(self._subreddit_key(item), None)
                self._discovery_served.discard(self._subreddit_key(item))
            self._fetch_discovery_batch(batch, config)
        self._discovery_served.add(key)
        return self._discovery_pages[key]

    @staticmethod
    def _requested_post_url(post: PostSnapshot) -> str | None:
        value = post.permalink or post.url
        return BrightDataProvider._absolute_reddit_url(value) if value else None

    def _post_record_matches(self, raw: Mapping[str, Any], post: PostSnapshot) -> bool:
        requested_url = self._requested_post_url(post)
        requested_url_id = self._post_url_id(requested_url)
        raw_url_values = [raw.get(key) for key in ("url", "post_url", "permalink") if raw.get(key)]
        if not self._record_urls_valid(raw):
            return False
        raw_subreddits = self._record_subreddits(raw)
        if raw_subreddits and len({self._subreddit_key(value) for value in raw_subreddits}) != 1:
            return False
        if raw_subreddits and post.subreddit and self._subreddit_key(raw_subreddits[0]) != self._subreddit_key(post.subreddit):
            return False
        raw_url_ids = {url_id for value in raw_url_values if (url_id := self._post_url_id(value)) is not None}
        try:
            raw_post_id = post_id(raw).casefold()
        except (TypeError, ValueError):
            raw_post_id = None
        if len(raw_url_ids) > 1:
            return False
        if raw_url_ids:
            raw_url_id = next(iter(raw_url_ids))
            return (requested_url_id is None or raw_url_id == requested_url_id) and (raw_post_id is None or raw_post_id == raw_url_id)
        requested_canonical = self._canonical_url(requested_url)
        if requested_canonical is not None and any(self._canonical_url(value) == requested_canonical for value in raw_url_values):
            return raw_post_id is None or raw_post_id == post.post_id.casefold()
        return raw_post_id == post.post_id.casefold()

    @staticmethod
    def _comments_unsupported_gap(post: PostSnapshot, mode: str) -> Gap:
        detail = (
            "Bright Data full comment expansion is unsupported: the comments dataset has no documented count or pagination bound."
            if mode == "full"
            else "Bright Data bounded comments are unsupported: documented comment records expose no depth and days_back is not a count cap."
        )
        return Gap("comment", "unsupported", entity_id=post.post_id, subreddit=post.subreddit, detail=detail)

    def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
        all_posts: list[PostSnapshot] = []
        gaps: list[Gap] = []
        records: list[RequestRecord] = []
        observation_requests: dict[str, RequestRecord] = {}
        pending: list[tuple[PostSnapshot, str]] = []
        for post in posts:
            url = self._requested_post_url(post)
            if url is None:
                gaps.append(Gap("post", "missing_permalink", entity_id=post.post_id, subreddit=post.subreddit))
            else:
                pending.append((post, url))
        for offset in range(0, len(pending), self.max_sync_inputs):
            batch = pending[offset : offset + self.max_sync_inputs]
            request_payload = {"input": [{"url": url} for _, url in batch]}
            payload, response, request_id, request_records, fetched_at, error = self._request(
                dataset_id=self.posts_dataset_id,
                query={},
                request_payload=request_payload,
                operation="refresh",
            )
            records.extend(request_records)
            batch_record = next((record for record in request_records if record.operation == "refresh"), None)
            if error is not None:
                gaps.extend(self._provider_gap("post", entity_id=post.post_id, subreddit=post.subreddit, error=error.provider_payload if error.provider_payload is not None else str(error), status=error.status) for post, _ in batch)
                continue
            if response is not None and response.status == 202:
                self._update_accounting(request_records, returned_records=0, provider_error_count=0)
                if self._is_deferred_payload(payload):
                    snapshot_id = payload.get("snapshot_id") if isinstance(payload, dict) else None
                    detail = "Bright Data returned deferred HTTP 202"
                    if snapshot_id is not None:
                        detail += f" with async snapshot {snapshot_id!r}"
                    detail += "; snapshot polling is not enabled"
                    gaps.extend(Gap("post", "unsupported", entity_id=post.post_id, subreddit=post.subreddit, detail=detail) for post, _ in batch)
                else:
                    shape = self._response_shape(payload)
                    gaps.extend(Gap("post", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=f"malformed provider payload: expected a JSON array or snapshot, got {shape}") for post, _ in batch)
                continue
            raw_items, errors, shape = self._records_and_errors(payload)
            if shape == "snapshot":
                snapshot_id = payload.get("snapshot_id") if isinstance(payload, dict) else None
                gaps.extend(Gap("post", "unsupported", entity_id=post.post_id, subreddit=post.subreddit, detail=f"Bright Data returned async snapshot {snapshot_id!r}; snapshot polling is not enabled") for post, _ in batch)
                continue
            if raw_items is None:
                gaps.extend(Gap("post", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=f"malformed provider payload: expected a JSON array, got {shape}") for post, _ in batch)
                continue
            output_records = [item for item in raw_items if isinstance(item, Mapping) and not ("error" in item and not any(key in item for key in ("id", "post_id", "fullname", "name")))]
            provider_errors = errors + [item for item in raw_items if isinstance(item, Mapping) and "error" in item and not any(key in item for key in ("id", "post_id", "fullname", "name"))]
            self._update_accounting(request_records, returned_records=len(output_records), provider_error_count=len(provider_errors))
            returned_by_post: dict[str, Mapping[str, Any]] = {}
            for raw in output_records:
                matches = [post for post, _ in batch if self._post_record_matches(raw, post)]
                if len(matches) != 1:
                    gaps.append(Gap("post", "provider_error", detail="provider record could not be attributed to exactly one requested post"))
                    continue
                target = matches[0]
                if target.post_id in returned_by_post:
                    gaps.append(Gap("post", "provider_error", entity_id=target.post_id, subreddit=target.subreddit, detail="provider returned duplicate post records"))
                    continue
                returned_by_post[target.post_id] = raw
            for post, requested_url in batch:
                raw = returned_by_post.get(post.post_id)
                target_errors = [item for item in provider_errors if self._error_matches(item, requested_url)]
                if raw is None:
                    if target_errors:
                        gaps.extend(self._provider_gap("post", entity_id=post.post_id, subreddit=post.subreddit, error=item) for item in target_errors)
                    else:
                        gaps.append(Gap("post", "not_returned", entity_id=post.post_id, subreddit=post.subreddit, detail="Bright Data returned no record for this requested URL"))
                    continue
                try:
                    refreshed = parse_post(
                        raw,
                        default_subreddit=post.subreddit,
                        observed_at=fetched_at,
                        include_comments=False,
                        detect_deletion_state=False,
                    )
                except (TypeError, ValueError) as exc:
                    gaps.append(_malformed_gap("post", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)))
                    continue
                if refreshed.post_id != post.post_id:
                    gaps.append(_id_mismatch_gap("post", post.post_id, refreshed.post_id, post.subreddit))
                    continue
                gaps.extend(_report_only_state_gaps(raw, entity_id=post.post_id, subreddit=post.subreddit))
                if config.comments_mode in {"bounded", "full"}:
                    gaps.append(self._comments_unsupported_gap(refreshed, config.comments_mode))
                count_gap = comment_count_gap(refreshed, expected_num_comments=post.num_comments)
                if count_gap and not any(gap.entity_id == post.post_id and gap.reason == "unexpanded" for gap in gaps):
                    gaps.append(count_gap)
                all_posts.append(refreshed)
                if batch_record is not None:
                    observation_requests[post.post_id] = batch_record
        primary = next((record for record in records if record.operation == "refresh"), None)
        metadata = {
            "comments_mode": config.comments_mode,
            "comments_expanded": _comments_complete(gaps),
            "no_automatic_retries": True,
            "post_dataset_id": self.posts_dataset_id,
            "comment_dataset_id": self.comments_dataset_id,
            "request_count": len(records),
            "record_count": len(all_posts),
        }
        return RefreshResult(
            all_posts,
            utc_now(),
            primary.request_id if primary else request_id if 'request_id' in locals() else None,
            primary.response_status if primary else response.status if 'response' in locals() and response else None,
            "unknown" if primary else None,
            None,
            gaps,
            metadata,
            records,
            observation_requests,
        )


def make_provider(config: Config) -> Provider:
    if config.provider == "fixture":
        if config.fixture_path is None:
            raise ValueError("fixture provider requires [provider].fixture_path")
        return FixtureProvider(config.fixture_path)
    if config.provider == "redditapis":
        return RedditApisProvider(config)
    if config.provider == "fetchlayer":
        return FetchLayerProvider(config)
    if config.provider == "brightdata":
        return BrightDataProvider(config)
    raise ValueError(f"unknown provider {config.provider!r}; choose fixture, redditapis, fetchlayer, or brightdata")
