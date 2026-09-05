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
from .models import Gap, PageResult, Plan, PostSnapshot, ProviderStatus, RefreshResult
from .normalize import parse_comment, parse_post, utc_now


@dataclass(slots=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False, billed: bool | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.billed = billed


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
    ):
        self.timeout = timeout
        self.retries = retries
        self.transport = transport
        self.sleep = sleep

    def request(self, method: str, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], HttpResponse, str]:
        body = json.dumps(payload).encode() if payload is not None else None
        request_id = str(uuid.uuid4())
        request_headers = {"Accept": "application/json", "User-Agent": "reddit-ingestion/0.1", **headers}
        if payload is not None:
            request_headers["Content-Type"] = "application/json"
        for attempt in range(self.retries + 1):
            try:
                response = self.transport(method, url, request_headers, body, self.timeout)
            except ProviderError:
                if attempt >= self.retries:
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
                raise ProviderError(f"provider returned non-JSON response with status {response.status}", status=response.status, retryable=response.status >= 500) from exc
            if response.status >= 400:
                raise ProviderError(
                    f"provider returned HTTP {response.status}: {parsed.get('error', parsed) if isinstance(parsed, dict) else parsed}",
                    status=response.status,
                    retryable=response.status in {408, 425, 429, 500, 502, 503, 504},
                    billed=response.status not in {401, 403, 404},
                )
            if not isinstance(parsed, dict):
                raise ProviderError("provider response must be a JSON object", status=response.status)
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


class FixtureProvider:
    name = "fixture"

    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path
        with fixture_path.open(encoding="utf-8") as handle:
            self.fixture = json.load(handle)

    def status(self) -> ProviderStatus:
        return ProviderStatus("fixture", True, False, f"fixture data: {self.fixture_path}", False)

    def plan(self, config: Config, known_posts: int) -> Plan:
        return Plan("fixture", len(config.subreddits), known_posts, len(config.subreddits) + known_posts, 0, "free fixture calls", ["No network calls or credentials are used."])

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        observed = utc_now()
        pages = self.fixture.get("listings", {}).get(subreddit, [])
        selected = next((page for page in pages if page.get("request_after") == cursor), None)
        if selected is None:
            return PageResult([], cursor, None, observed, f"fixture://{subreddit}/new", 200, "fixture", gaps=[Gap("listing", "fixture_page_missing", subreddit=subreddit, detail=f"cursor={cursor!r}")])
        posts = [parse_post(item, default_subreddit=subreddit, observed_at=observed) for item in selected.get("posts", [])]
        return PageResult(posts, cursor, selected.get("after"), observed, f"fixture://{subreddit}/new", 200, "fixture", gaps=[Gap(**gap) for gap in selected.get("gaps", [])])

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
        return RefreshResult(refreshed, observed, "fixture", 200, gaps=gaps)


class HttpProviderBase:
    name = "http"
    key_env = ""
    base_url = ""

    def __init__(self, config: Config, client: JsonClient | None = None):
        self.config = config
        self.client = client or JsonClient(timeout=config.request_timeout_seconds, retries=config.max_retries)
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
        calls = len(config.subreddits) + (refresh + 99) // 100
        notes = ["Refresh uses documented batches of up to 100 t3_ fullnames."]
        if config.comments_mode != "off":
            notes.append("Comments add at least one paid read per refreshed post; full expansion may expose additional cursors.")
        request_count = calls + (refresh if config.comments_mode != "off" else 0)
        return Plan(self.name, len(config.subreddits), refresh, request_count, request_count * 0.002, "USD reads", notes)

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        key = self._require_key()
        params = {"subreddit": subreddit, "sort": "new", "limit": str(config.listing_limit)}
        if cursor:
            params["after"] = cursor
        url = f"{self.base_url}/api/reddit/posts?{urlencode(params)}"
        payload, response, request_id = self.client.request("GET", url, headers={"Authorization": f"Bearer {key}"})
        observed = utc_now()
        posts = [parse_post(item, default_subreddit=subreddit, observed_at=observed) for item in payload.get("posts", []) if isinstance(item, Mapping)]
        return PageResult(posts, cursor, payload.get("after"), observed, url, response.status, request_id, metadata={"listing_status": payload.get("listing_status"), "exhausted_reason": payload.get("exhausted_reason")})

    def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
        key = self._require_key()
        all_posts: list[PostSnapshot] = []
        gaps: list[Gap] = []
        last_response: HttpResponse | None = None
        last_request_id: str | None = None
        for offset in range(0, len(posts), 100):
            batch = posts[offset : offset + 100]
            fullnames = ",".join(post.fullname or f"t3_{post.post_id}" for post in batch)
            url = f"{self.base_url}/api/reddit/by_id/{quote(fullnames, safe=',_')}"
            try:
                payload, response, request_id = self.client.request("GET", url, headers={"Authorization": f"Bearer {key}"})
            except ProviderError as exc:
                gaps.extend(Gap("post", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)) for post in batch)
                continue
            last_response, last_request_id = response, request_id
            returned = {post_id(raw): raw for raw in payload.get("posts", []) if isinstance(raw, Mapping) and _has_id(raw)}
            for post in batch:
                raw = returned.get(post.post_id) or returned.get(post.fullname or "")
                if raw is None:
                    gaps.append(Gap("post", "not_returned", entity_id=post.post_id, subreddit=post.subreddit))
                else:
                    refreshed = parse_post(raw, default_subreddit=post.subreddit, observed_at=utc_now(), include_comments=False)
                    if config.comments_mode != "off":
                        comments_url = f"{self.base_url}/api/reddit/post/{quote(post.post_id, safe='')}/comments"
                        try:
                            comment_payload, comment_response, comment_request_id = self.client.request("GET", comments_url, headers={"Authorization": f"Bearer {key}"})
                            comment_observed = utc_now()
                            raw_comments = comment_payload.get("comments", [])
                            if isinstance(raw_comments, list):
                                refreshed.comments = [parse_comment(item, post=refreshed, observed_at=comment_observed) for item in raw_comments if isinstance(item, Mapping)]
                            if comment_payload.get("after") or comment_payload.get("listing_status") in {"truncated", "unknown"}:
                                gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail="provider returned an additional comment cursor or incomplete tree"))
                            last_response, last_request_id = comment_response, comment_request_id
                        except ProviderError as exc:
                            gaps.append(Gap("comment", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)))
                    all_posts.append(refreshed)
        status = last_response.status if last_response else None
        return RefreshResult(all_posts, utc_now(), last_request_id, status, gaps=gaps)


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
        calls = len(config.subreddits) + refresh
        notes = ["No documented multi-post refresh batch; each post URL is one call.", "Extra listing pages add provider request units."]
        if config.comments_mode == "full":
            notes.append("Full comment expansion is explicitly requested; exact provider-side page cost is not guaranteed by public docs.")
        return Plan(self.name, len(config.subreddits), refresh, calls, calls * 0.00199, "USD requests", notes)

    def discover(self, subreddit: str, cursor: str | None, config: Config) -> PageResult:
        key = self._require_key()
        payload, response, request_id = self.client.request(
            "POST",
            f"{self.base_url}/community-posts",
            headers={"Authorization": f"Bearer {key}"},
            payload={"subreddit": subreddit, "sort": "new", "limit": config.listing_limit, "pages": config.max_discovery_pages},
        )
        observed = utc_now()
        posts = [parse_post(item, default_subreddit=subreddit, observed_at=observed) for item in payload.get("items", []) if isinstance(item, Mapping)]
        gaps: list[Gap] = []
        if payload.get("blocked"):
            gaps.append(Gap("listing", "blocked", subreddit=subreddit, detail=str(payload.get("blockReason"))))
        return PageResult(posts, cursor, payload.get("nextPageUrl"), observed, payload.get("requestedUrl"), response.status, request_id, "cached" if payload.get("cached") else "live", payload.get("cached_at") or payload.get("cachedAt"), gaps, {"pagesScraped": payload.get("pagesScraped"), "pagesRequested": payload.get("pagesRequested")})

    def refresh_posts(self, posts: list[PostSnapshot], config: Config) -> RefreshResult:
        key = self._require_key()
        refreshed: list[PostSnapshot] = []
        gaps: list[Gap] = []
        last_response: HttpResponse | None = None
        last_request_id: str | None = None
        last_cache_status: str | None = None
        last_cache_at: str | None = None
        for post in posts:
            url = post.permalink or post.url
            if not url:
                gaps.append(Gap("post", "missing_permalink", entity_id=post.post_id, subreddit=post.subreddit))
                continue
            try:
                payload, response, request_id = self.client.request(
                    "POST",
                    f"{self.base_url}/post",
                    headers={"Authorization": f"Bearer {key}"},
                    payload={"url": url, "pages": 1, "depth": config.comment_depth if config.comments_mode != "off" else 0, "commentLimit": config.comment_limit if config.comments_mode == "bounded" else None},
                )
            except ProviderError as exc:
                gaps.append(Gap("post", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)))
                continue
            last_response, last_request_id = response, request_id
            last_cache_status = "cached" if payload.get("cached") else "live"
            last_cache_at = payload.get("cached_at") or payload.get("cachedAt")
            if payload.get("blocked"):
                gaps.append(Gap("post", "blocked", entity_id=post.post_id, subreddit=post.subreddit, detail=str(payload.get("blockReason"))))
                continue
            refreshed_post = parse_post(payload, default_subreddit=post.subreddit, observed_at=utc_now(), include_comments=config.comments_mode != "off")
            if payload.get("remainingMoreCommentsCount"):
                gaps.append(Gap("comment", "unexpanded", entity_id=post.post_id, subreddit=post.subreddit, detail=str(payload.get("remainingMoreCommentsCount"))))
            refreshed.append(refreshed_post)
        return RefreshResult(refreshed, utc_now(), last_request_id, last_response.status if last_response else None, last_cache_status, last_cache_at, gaps=gaps)


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
