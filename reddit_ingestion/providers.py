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

from .config import Config
from .models import Gap, PageResult, Plan, PostSnapshot, ProviderStatus, RefreshResult, RequestRecord
from .normalize import parse_comment, parse_post, utc_now


@dataclass(slots=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False, billed: bool | None = None, request_id: str | None = None, url: str | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.billed = billed
        self.request_id = request_id
        self.url = url


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

    def request(self, method: str, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], HttpResponse, str]:
        body = json.dumps(payload).encode() if payload is not None else None
        request_id = str(uuid.uuid4())
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
            except ProviderError as exc:
                exc.url = exc.url or url
                if attempt >= self.retries:
                    exc.request_id = exc.request_id or request_id
                    raise
                self.sleep(2**attempt)
                continue
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
                raise ProviderError(f"provider returned non-JSON response with status {response.status}", status=response.status, retryable=response.status >= 500, request_id=request_id, url=url) from exc
            if response.status >= 400:
                raise ProviderError(
                    f"provider returned HTTP {response.status}: {parsed.get('error', parsed) if isinstance(parsed, dict) else parsed}",
                    status=response.status,
                    retryable=response.status in {408, 425, 429, 500, 502, 503, 504},
                    billed=response.status not in {401, 403, 404},
                    request_id=request_id,
                    url=url,
                )
            if not isinstance(parsed, dict):
                raise ProviderError("provider response must be a JSON object", status=response.status, request_id=request_id, url=url)
            return parsed, response, request_id
        raise AssertionError("unreachable")


class Provider(Protocol):
    name: str

    def status(self) -> ProviderStatus: ...
    def plan(self, config: Config, known_posts: int) -> Plan: ...
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


def _failed_request_record(operation: str, url: str, error: ProviderError, metadata: dict[str, Any] | None = None) -> RequestRecord | None:
    details = {"url": url, "error": str(error)}
    if metadata:
        details.update(metadata)
    return _request_record(operation, error.request_id, error.status, "unknown", None, details, billed=error.billed)


class FixtureProvider:
    name = "fixture"

    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path
        with fixture_path.open(encoding="utf-8") as handle:
            self.fixture = json.load(handle)

    def status(self) -> ProviderStatus:
        return ProviderStatus("fixture", True, False, f"fixture data: {self.fixture_path}", False)

    def plan(self, config: Config, known_posts: int) -> Plan:
        discovery = len(config.subreddits) * config.max_discovery_pages
        refresh = 1 if known_posts else 0
        return Plan("fixture", discovery, known_posts, discovery + refresh, 0, "free fixture calls", ["No network calls or credentials are used."])

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        observed = utc_now()
        pages = self.fixture.get("listings", {}).get(subreddit, [])
        selected = next((page for page in pages if page.get("request_after") == cursor), None)
        if selected is None:
            record = _request_record("discover", "fixture", 200, None, None, {"subreddit": subreddit, "cursor": cursor}, billed=False)
            return PageResult([], cursor, None, observed, f"fixture://{subreddit}/new", 200, "fixture", gaps=[Gap("listing", "fixture_page_missing", subreddit=subreddit, detail=f"cursor={cursor!r}")], request_records=[record])
        posts = [parse_post(item, default_subreddit=subreddit, observed_at=observed) for item in selected.get("posts", [])]
        gaps = [Gap(**gap) for gap in selected.get("gaps", [])]
        if config.comments_mode != "off":
            source = self.fixture.get("refresh", {})
            for post in posts:
                raw = source.get(post.post_id) or source.get(post.fullname or "")
                if raw is None:
                    if post.num_comments:
                        gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=subreddit, detail="fixture has no comment expansion data"))
                    continue
                expanded = parse_post(raw, default_subreddit=subreddit, observed_at=observed)
                post.comments = expanded.comments
                if post.num_comments and not post.comments:
                    gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=subreddit, detail="fixture expansion returned no comments"))
        record = _request_record("discover", "fixture", 200, None, None, {"subreddit": subreddit, "cursor": cursor}, billed=False)
        return PageResult(posts, cursor, selected.get("after"), observed, f"fixture://{subreddit}/new", 200, "fixture", gaps=gaps, metadata={"comments_expanded": config.comments_mode != "off"}, request_records=[record])

    def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
        observed = utc_now()
        source = self.fixture.get("refresh", {})
        refreshed: list[PostSnapshot] = []
        gaps: list[Gap] = []
        for post in posts:
            raw = source.get(post.post_id) or source.get(post.fullname or "")
            if raw is None:
                gaps.append(Gap("post", "fixture_post_missing", entity_id=post.post_id, subreddit=post.subreddit))
                continue
            refreshed.append(parse_post(raw, default_subreddit=post.subreddit, observed_at=observed))
        record = _request_record("refresh", "fixture", 200, None, None, {"post_count": len(posts)}, billed=False)
        observations = {post.post_id: record for post in refreshed}
        return RefreshResult(refreshed, observed, "fixture", 200, gaps=gaps, metadata={"comments_expanded": config.comments_mode != "off"}, request_records=[record], observation_requests=observations)


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

    def plan(self, config: Config, known_posts: int) -> Plan:
        refresh = min(known_posts, config.max_refresh_posts)
        discovery = len(config.subreddits) * config.max_discovery_pages
        refresh_batches = (refresh + 99) // 100
        calls = discovery + refresh_batches
        discovered_capacity = discovery * config.listing_limit
        notes = ["Discovery estimate includes the configured maximum listing pages.", "Refresh uses documented batches of up to 100 t3_ fullnames."]
        if config.comments_mode != "off":
            notes.append("Comment request estimate uses the listing-limit upper bound for newly discovered posts.")
        if config.comments_mode == "full":
            notes.append("Full comment expansion omits provider bounds and may add one request per returned comment page.")
        request_count = calls + (discovered_capacity + refresh if config.comments_mode != "off" else 0)
        return Plan(self.name, discovery, refresh, request_count, request_count * 0.002, "USD reads", notes)

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        key = self._require_key()
        params = {"subreddit": subreddit, "sort": "new", "limit": str(config.listing_limit)}
        if cursor:
            params["after"] = cursor
        url = f"{self.base_url}/api/reddit/posts?{urlencode(params)}"
        payload, response, request_id = self.client.request("GET", url, headers={"Authorization": f"Bearer {key}"})
        observed = utc_now()
        posts = [parse_post(item, default_subreddit=subreddit, observed_at=observed, include_comments=False) for item in payload.get("posts", []) if isinstance(item, Mapping)]
        cache_status, cache_observed_at = _cache_info(payload)
        gaps: list[Gap] = []
        records = [_request_record("discover", request_id, response.status, cache_status, cache_observed_at, {"url": url, "subreddit": subreddit, "cursor": cursor}, billed=cache_status != "cached")]
        if config.comments_mode != "off":
            for post in posts:
                comments, comment_gaps, comment_records = self._fetch_comments(post, config, key)
                existing = {comment.comment_id: comment for comment in post.comments}
                existing.update({comment.comment_id: comment for comment in comments})
                post.comments = list(existing.values())
                gaps.extend(comment_gaps)
                records.extend(comment_records)
        return PageResult(posts, cursor, payload.get("after"), observed, url, response.status, request_id, cache_status, cache_observed_at, gaps, {"listing_status": payload.get("listing_status"), "exhausted_reason": payload.get("exhausted_reason"), "comments_expanded": config.comments_mode != "off"}, records)

    def _fetch_comments(self, post: PostSnapshot, config: Config, key: str) -> tuple[list[Any], list[Gap], list[RequestRecord]]:
        comments: list[Any] = []
        gaps: list[Gap] = []
        records: list[RequestRecord] = []
        cursor: str | None = None
        while True:
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
                record = _failed_request_record("comments", url, exc, {"post_id": post.post_id, "mode": config.comments_mode})
                if record:
                    records.append(record)
                gaps.append(Gap("comment", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)))
                break
            observed = utc_now()
            cache_status, cache_observed_at = _cache_info(payload)
            records.append(_request_record("comments", request_id, response.status, cache_status, cache_observed_at, {"url": url, "post_id": post.post_id, "mode": config.comments_mode}, billed=cache_status != "cached"))
            raw_comments = payload.get("comments", [])
            if not isinstance(raw_comments, list):
                gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider returned no comment list"))
                break
            comments.extend(parse_comment(item, post=post, observed_at=observed) for item in raw_comments if isinstance(item, Mapping))
            next_cursor = payload.get("after")
            incomplete = payload.get("listing_status") in {"truncated", "unknown"} or payload.get("remainingMoreCommentsCount")
            if config.comments_mode == "bounded":
                if next_cursor or incomplete:
                    gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="bounded provider response has additional or incomplete comments"))
                break
            if incomplete and not next_cursor:
                gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="full provider response is incomplete"))
            if not next_cursor:
                break
            if next_cursor == cursor:
                gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider repeated the comment cursor"))
                break
            cursor = str(next_cursor)
        if post.num_comments and not comments and not any(gap.entity_id == post.post_id and gap.reason in {"provider_error", "unexpanded"} for gap in gaps):
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
                record = _failed_request_record("refresh", url, exc, {"post_count": len(batch)})
                if record:
                    records.append(record)
                gaps.extend(Gap("post", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)) for post in batch)
                continue
            cache_status, cache_observed_at = _cache_info(payload)
            record = _request_record("refresh", request_id, response.status, cache_status, cache_observed_at, {"url": url, "post_count": len(batch)}, billed=cache_status != "cached")
            records.append(record)
            returned = {post_id(raw): raw for raw in payload.get("posts", []) if isinstance(raw, Mapping) and _has_id(raw)}
            for post in batch:
                raw = returned.get(post.post_id) or returned.get(post.fullname or "")
                if raw is None:
                    gaps.append(Gap("post", "not_returned", entity_id=post.post_id, subreddit=post.subreddit))
                else:
                    refreshed = parse_post(raw, default_subreddit=post.subreddit, observed_at=utc_now(), include_comments=False)
                    observation_requests[post.post_id] = record
                    if config.comments_mode != "off":
                        comments, comment_gaps, comment_records = self._fetch_comments(refreshed, config, key)
                        refreshed.comments = comments
                        gaps.extend(comment_gaps)
                        records.extend(comment_records)
                    all_posts.append(refreshed)
        primary = next((record for record in records if record.operation == "refresh"), None)
        return RefreshResult(all_posts, utc_now(), primary.request_id if primary else None, primary.response_status if primary else None, primary.cache_status if primary else None, primary.cache_observed_at if primary else None, gaps, {"batch_count": (len(posts) + 99) // 100, "comments_expanded": config.comments_mode != "off"}, records, observation_requests)


def _has_id(raw: Mapping[str, Any]) -> bool:
    return bool(raw.get("id") or raw.get("post_id") or raw.get("fullname") or raw.get("name"))


def post_id(raw: Mapping[str, Any]) -> str:
    value = raw.get("id") or raw.get("post_id") or raw.get("fullname") or raw.get("name")
    return str(value).removeprefix("t3_")


class FetchLayerProvider(HttpProviderBase):
    name = "fetchlayer"
    key_env = "FETCHLAYER_API_KEY"
    base_url = "https://api.fetchlayer.dev/reddit"

    def plan(self, config: Config, known_posts: int) -> Plan:
        refresh = min(known_posts, config.max_refresh_posts)
        discovery = len(config.subreddits) * config.max_discovery_pages
        discovered_capacity = discovery * config.listing_limit
        comment_expansions = discovered_capacity if config.comments_mode == "bounded" else 0
        calls = discovery + (0 if config.comments_mode == "full" else refresh) + comment_expansions
        notes = ["No documented multi-post refresh batch; each post URL is one call.", "Discovery estimate includes the configured maximum listing pages."]
        if config.comments_mode == "full":
            notes.append("Full comment expansion is unsupported by this adapter; known-post refresh is skipped and an unexpanded gap is reported without a paid comment call.")
        return Plan(self.name, discovery, refresh, calls, calls * 0.00199, "USD requests", notes)

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        key = self._require_key()
        request_payload: dict[str, Any] = {"subreddit": subreddit, "sort": "new", "limit": config.listing_limit, "pages": config.max_discovery_pages}
        if cursor:
            request_payload["pageUrl"] = cursor
        payload, response, request_id = self.client.request(
            "POST",
            f"{self.base_url}/community-posts",
            headers={"Authorization": f"Bearer {key}"},
            payload=request_payload,
        )
        observed = utc_now()
        posts = [parse_post(item, default_subreddit=subreddit, observed_at=observed, include_comments=False) for item in payload.get("items", []) if isinstance(item, Mapping)]
        gaps: list[Gap] = []
        if payload.get("blocked"):
            gaps.append(Gap("listing", "blocked", subreddit=subreddit, detail=str(payload.get("blockReason"))))
        cache_status, cache_observed_at = _cache_info(payload)
        pages = payload.get("pagesRequested") or payload.get("pagesScraped") or config.max_discovery_pages
        try:
            request_units = max(1, int(pages))
        except (TypeError, ValueError):
            request_units = 1
        records = [_request_record("discover", request_id, response.status, cache_status, cache_observed_at, {"url": f"{self.base_url}/community-posts", **request_payload}, billed=cache_status != "cached", request_units=request_units)]
        if config.comments_mode == "full":
            for post in posts:
                post.comments = []
            gaps.extend(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=subreddit, detail="FetchLayer full comment expansion is unsupported") for post in posts)
        elif config.comments_mode == "bounded":
            for post in posts:
                expansion = self._fetch_post(post, config, "comment_expansion", key)
                records.extend(expansion.request_records)
                gaps.extend(expansion.gaps)
                if expansion.posts:
                    post.comments = expansion.posts[0].comments
        return PageResult(posts, cursor, payload.get("nextPageUrl"), observed, payload.get("requestedUrl"), response.status, request_id, cache_status, cache_observed_at, gaps, {"pagesScraped": payload.get("pagesScraped"), "pagesRequested": payload.get("pagesRequested"), "comments_expanded": config.comments_mode != "off" and config.comments_mode != "full"}, records)

    def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
        refreshed: list[PostSnapshot] = []
        gaps: list[Gap] = []
        records: list[RequestRecord] = []
        observation_requests: dict[str, RequestRecord] = {}
        if config.comments_mode == "full":
            gaps.extend(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="FetchLayer full comment expansion is unsupported") for post in posts)
            return RefreshResult([], utc_now(), None, None, "unknown", None, gaps=gaps, metadata={"full_expansion": "unsupported"})
        key = self._require_key()
        for post in posts:
            result = self._fetch_post(post, config, "refresh", key)
            refreshed.extend(result.posts)
            gaps.extend(result.gaps)
            records.extend(result.request_records)
            observation_requests.update(result.observation_requests)
        primary = next((record for record in records if record.operation == "refresh"), None)
        return RefreshResult(refreshed, utc_now(), primary.request_id if primary else None, primary.response_status if primary else None, primary.cache_status if primary else None, primary.cache_observed_at if primary else None, gaps, {"comments_expanded": config.comments_mode == "bounded"}, records, observation_requests)

    def _fetch_post(self, post: PostSnapshot, config: Config, operation: str, key: str) -> RefreshResult:
        url = post.permalink or post.url
        if not url:
            return RefreshResult([], utc_now(), None, None, None, None, gaps=[Gap("post", "missing_permalink", entity_id=post.post_id, subreddit=post.subreddit)])
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
            record = _failed_request_record(operation, request_url, exc, {"post_id": post.post_id, "url": url, "payload": payload})
            return RefreshResult([], utc_now(), exc.request_id, exc.status, "unknown", None, gaps=[Gap("post", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc))], request_records=[record] if record else [])
        cache_status, cache_observed_at = _cache_info(response_payload)
        pages = response_payload.get("pagesRequested") or response_payload.get("pagesScraped") or 1
        try:
            request_units = max(1, int(pages))
        except (TypeError, ValueError):
            request_units = 1
        record = _request_record(operation, request_id, response.status, cache_status, cache_observed_at, {"url": request_url, "post_id": post.post_id, "payload": payload}, billed=cache_status != "cached", request_units=request_units)
        if response_payload.get("blocked"):
            return RefreshResult([], utc_now(), request_id, response.status, cache_status, cache_observed_at, gaps=[Gap("post", "blocked", entity_id=post.post_id, subreddit=post.subreddit, detail=str(response_payload.get("blockReason")))], request_records=[record])
        refreshed_post = parse_post(response_payload, default_subreddit=post.subreddit, observed_at=utc_now(), include_comments=config.comments_mode != "off")
        if response_payload.get("remainingMoreCommentsCount") or response_payload.get("listing_status") in {"truncated", "unknown"} or (config.comments_mode == "bounded" and refreshed_post.num_comments and not refreshed_post.comments):
            gap = Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider returned additional or incomplete comments")
            return RefreshResult([refreshed_post], utc_now(), request_id, response.status, cache_status, cache_observed_at, gaps=[gap], request_records=[record], observation_requests={post.post_id: record})
        return RefreshResult([refreshed_post], utc_now(), request_id, response.status, cache_status, cache_observed_at, request_records=[record], observation_requests={post.post_id: record})


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
