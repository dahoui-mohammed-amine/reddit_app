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
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .comments import comment_count_gap
from .config import Config
from .models import Gap, PageResult, Plan, PostSnapshot, ProviderStatus, RefreshResult, RequestRecord
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
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False, billed: bool | None = None, request_id: str | None = None, url: str | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.billed = billed
        self.request_id = request_id
        self.url = url
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
    ):
        self.timeout = timeout
        self.retries = retries
        self.transport = transport
        self.sleep = sleep
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._last_request_at: float | None = None
        self.last_attempts: list[RequestAttempt] = []

    def request(self, method: str, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], HttpResponse, str]:
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
                error = ProviderError(f"provider returned non-JSON response with status {response.status}", status=response.status, retryable=response.status >= 500, request_id=request_id, url=url)
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
                )
                error.attempts = list(self.last_attempts)
                raise error
            if not isinstance(parsed, dict):
                self.last_attempts[-1].billed = None
                error = ProviderError("provider response must be a JSON object", status=response.status, request_id=request_id, url=url)
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


def _env_key(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _cache_info(payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    if "cache_status" in payload:
        status = str(payload["cache_status"]).lower()
        return (status if status in {"cached", "live", "unknown"} else "unknown", payload.get("cached_at") or payload.get("cachedAt"))
    if "cacheStatus" in payload:
        status = str(payload["cacheStatus"]).lower()
        return (status if status in {"cached", "live", "unknown"} else "unknown", payload.get("cached_at") or payload.get("cachedAt"))
    if "cached" not in payload:
        return "unknown", payload.get("cached_at") or payload.get("cachedAt")
    cached = payload.get("cached")
    if isinstance(cached, bool):
        return ("cached" if cached else "live"), payload.get("cached_at") or payload.get("cachedAt")
    if isinstance(cached, str) and cached.lower() in {"true", "false"}:
        return ("cached" if cached.lower() == "true" else "live"), payload.get("cached_at") or payload.get("cachedAt")
    return "unknown", payload.get("cached_at") or payload.get("cachedAt")


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
    details = {"url": url, "error": str(error)}
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


def _parse_post_items(
    items: Any,
    *,
    default_subreddit: str | None,
    observed_at: str,
    include_comments: bool,
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
            )
        except (TypeError, ValueError) as exc:
            gaps.append(_malformed_gap("post", subreddit=default_subreddit, detail=str(exc)))
            continue
        if post.post_id in seen_post_ids:
            continue
        seen_post_ids.add(post.post_id)
        posts.append(post)
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
        refresh = 1 if known_posts and mode in {"run", "refresh"} else 0
        refresh_events = min(known_posts, config.max_refresh_posts) if mode in {"run", "refresh"} else 0
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
        refresh = min(known_posts, config.max_refresh_posts) if mode in {"run", "refresh"} else 0
        discovery = len(config.subreddits) * config.max_discovery_pages + resume_pages if mode in {"run", "discover"} else 0
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
        refresh = min(known_posts, config.max_refresh_posts) if mode in {"run", "refresh"} else 0
        discovery = len(config.subreddits) * config.max_discovery_pages + resume_pages if mode in {"run", "discover"} else 0
        discovered_capacity = discovery * config.listing_limit
        comment_expansions = discovered_capacity if config.comments_mode == "bounded" else 0
        calls = discovery + refresh + comment_expansions
        notes = []
        if refresh:
            notes.append("No documented multi-post refresh batch; each post URL is one call.")
        if discovery:
            notes.append("Discovery estimate includes the configured maximum listing pages.")
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
        pages = payload.get("pagesRequested") or payload.get("pagesScraped") or config.max_discovery_pages
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


def make_provider(config: Config) -> Provider:
    if config.provider == "fixture":
        if config.fixture_path is None:
            raise ValueError("fixture provider requires [provider].fixture_path")
        return FixtureProvider(config.fixture_path)
    if config.provider == "redditapis":
        return RedditApisProvider(config)
    if config.provider == "fetchlayer":
        return FetchLayerProvider(config)
    raise ValueError(f"unknown provider {config.provider!r}; choose fixture, redditapis, or fetchlayer")
