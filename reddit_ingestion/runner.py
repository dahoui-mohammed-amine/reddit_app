from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from .comments import apply_comment_policy
from .config import Config
from .db import Database
from .models import Gap, PageResult, Plan, PostSnapshot, RefreshResult, RunSummary, RequestRecord
from .normalize import utc_now
from .providers import Provider, ProviderError, _failed_request_records


def config_dict(config: Config) -> dict[str, Any]:
    data = asdict(config)
    data["database_path"] = str(config.database_path)
    data["fixture_path"] = str(config.fixture_path) if config.fixture_path else None
    data["subreddits"] = list(config.subreddits)
    return data


def _parse_refresh_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None


def _apply_refresh_expiry(
    posts: list[PostSnapshot],
    config: Config,
    source_created_at: dict[str, str | None] | None = None,
) -> None:
    source_created_at = source_created_at or {}
    for post in posts:
        if post.refresh_until:
            continue
        source_time = post.created_at or source_created_at.get(post.post_id)
        if source_time:
            observed = _parse_refresh_time(source_time)
            if observed is None and post.observed_at:
                observed = _parse_refresh_time(post.observed_at)
        else:
            observed = _parse_refresh_time(post.observed_at) if post.observed_at else None
        if observed is None:
            continue
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        post.refresh_until = (observed + timedelta(days=config.refresh_expiry_days)).isoformat().replace("+00:00", "Z")


def plan_for(db: Database, provider: Provider, config: Config, mode: str = "run") -> Plan:
    if mode not in {"run", "discover", "refresh"}:
        raise ValueError("mode must be run, discover, or refresh")
    known = db.refreshable_post_count(config.subreddits, config.refresh_expiry_days)
    resume_pages = sum(
        1
        for subreddit in config.subreddits
        if (checkpoint := db.checkpoint(subreddit))
        and checkpoint["provider"] == provider.name
        and checkpoint["cursor"]
    )
    return provider.plan(config, known, resume_pages=resume_pages, mode=mode)


def _request_units(records: list[RequestRecord], request_id: str | None) -> int:
    if records:
        return sum(max(1, record.request_units) for record in records)
    return 1 if request_id else 0


def run_once(db: Database, provider: Provider, config: Config, mode: str, *, allow_paid: bool = False) -> RunSummary:
    if mode not in {"run", "discover", "refresh"}:
        raise ValueError("mode must be run, discover, or refresh")
    if provider.name != "fixture":
        check_live_access(provider, allow_paid=allow_paid)
    started = utc_now()
    run_id = db.start_run(provider.name, mode, config_dict(config), started)
    summary = RunSummary(run_id)
    try:
        if mode in {"run", "discover"}:
            for subreddit in config.subreddits:
                checkpoint = db.checkpoint(subreddit)
                resume_cursor = checkpoint["cursor"] if checkpoint and checkpoint["provider"] == provider.name else None
                page_count = config.max_discovery_pages + (1 if resume_cursor else 0)
                cursor: str | None = None
                seen_cursors: set[str] = set()
                for page_number in range(page_count):
                    resumed_page_number = page_number - 1 if resume_cursor else page_number
                    if resume_cursor and page_number == 1:
                        cursor = resume_cursor
                    if cursor is not None:
                        cursor = str(cursor)
                        if cursor in seen_cursors:
                            page = PageResult(
                                posts=[],
                                requested_cursor=cursor,
                                next_cursor=cursor,
                                observed_at=utc_now(),
                                source_url=None,
                                response_status=None,
                                request_id=None,
                                gaps=[Gap("listing", "cursor_cycle", subreddit=subreddit, detail=f"repeated cursor={cursor!r}")],
                                metadata={"checkpoint_deferred": True, "cursor_cycle": True},
                            )
                            with db.transaction():
                                db.save_page(run_id, provider.name, subreddit, page)
                            summary.gaps += len(page.gaps)
                            break
                        seen_cursors.add(cursor)
                    try:
                        page = provider.discover(subreddit, cursor, config)
                    except ProviderError as exc:
                        page = PageResult(
                            posts=[],
                            requested_cursor=cursor,
                            next_cursor=cursor,
                            observed_at=utc_now(),
                            source_url=None,
                            response_status=exc.status,
                            request_id=exc.request_id,
                            cache_status="unknown" if exc.request_id else None,
                            gaps=[Gap("listing", "provider_error", subreddit=subreddit, detail=str(exc))],
                            metadata={"request_failed": True, "error": str(exc)},
                            request_records=_failed_request_records(None, "discover", exc.url or "", exc),
                        )
                        with db.transaction():
                            db.save_page(run_id, provider.name, subreddit, page)
                        summary.gaps += len(page.gaps)
                        summary.requests += _request_units(page.request_records, page.request_id)
                        break
                    page.gaps.extend(apply_comment_policy(page.posts, config))
                    if any(gap.entity_type == "comment" for gap in page.gaps):
                        page.metadata["comments_expanded"] = False
                    if resume_cursor and page_number == 0:
                        page.metadata["checkpoint_deferred"] = True
                    next_cursor = str(page.next_cursor) if page.next_cursor is not None else None
                    cursor_cycle = next_cursor is not None and (next_cursor == cursor or next_cursor in seen_cursors)
                    if cursor_cycle:
                        page.gaps.append(Gap("listing", "cursor_cycle", subreddit=subreddit, detail=f"repeated cursor={next_cursor!r}"))
                        page.metadata["checkpoint_deferred"] = True
                        page.metadata["cursor_cycle"] = True
                    blocked = page.metadata.get("blocked") or any(gap.entity_type == "listing" and gap.reason == "blocked" for gap in page.gaps)
                    provider_incomplete = page.metadata.get("listing_status") in {"truncated", "unknown"} or any(gap.entity_type == "listing" and gap.reason in {"provider_error", "truncated"} for gap in page.gaps) or cursor_cycle
                    if provider_incomplete:
                        page.metadata["checkpoint_deferred"] = True
                    at_page_cap = resumed_page_number >= config.max_discovery_pages - 1
                    if page.next_cursor and at_page_cap and not blocked and not page.metadata.get("request_failed") and not provider_incomplete:
                        page.gaps.append(Gap("listing", "truncated", subreddit=subreddit, detail=f"max_discovery_pages={config.max_discovery_pages}"))
                    _apply_refresh_expiry(page.posts, config)
                    with db.transaction():
                        discovered, comments = db.save_page(run_id, provider.name, subreddit, page)
                    summary.discovered += discovered
                    summary.comments += comments
                    summary.gaps += len(page.gaps)
                    summary.requests += _request_units(page.request_records, page.request_id)
                    if blocked or provider_incomplete or page.metadata.get("request_failed") or page.metadata.get("checkpoint_deferred") or page.metadata.get("cursor_cycle"):
                        break
                    if resume_cursor and page_number == 0:
                        cursor = resume_cursor
                        continue
                    if not page.next_cursor or at_page_cap:
                        break
                    cursor = page.next_cursor
        if mode in {"run", "refresh"}:
            due = db.due_posts(config.refresh_interval_minutes, config.max_refresh_posts, config.subreddits, config.refresh_expiry_days)
            if due:
                try:
                    result = provider.refresh_posts(due, config)
                except ProviderError as exc:
                    result = RefreshResult(
                        posts=[],
                        observed_at=utc_now(),
                        request_id=exc.request_id,
                        response_status=exc.status,
                        cache_status="unknown" if exc.request_id else None,
                        gaps=[Gap("post", "provider_error", entity_id=post.post_id, subreddit=post.subreddit, detail=str(exc)) for post in due],
                        metadata={"request_failed": True, "error": str(exc)},
                        request_records=_failed_request_records(None, "refresh", exc.url or "", exc),
                    )
                result.metadata["comments_mode"] = config.comments_mode
                result.gaps.extend(apply_comment_policy(result.posts, config))
                if any(gap.entity_type == "comment" for gap in result.gaps):
                    result.metadata["comments_expanded"] = False
                _apply_refresh_expiry(result.posts, config, {post.post_id: post.created_at for post in due})
                with db.transaction():
                    refreshed, comments = db.save_refresh(run_id, provider.name, result)
                summary.refreshed += refreshed
                summary.comments += comments
                summary.gaps += len(result.gaps)
                summary.requests += _request_units(result.request_records, result.request_id)
        db.finish_run(run_id, utc_now(), "completed")
        return summary
    except Exception:
        db.finish_run(run_id, utc_now(), "failed")
        raise


def render_plan(plan: Plan) -> str:
    return json.dumps(
        {
            "provider": plan.provider,
            "discovery_requests": plan.discovery_requests,
            "refresh_events": plan.refresh_events,
            "estimated_requests": plan.estimated_requests,
            "estimated_units": plan.estimated_units,
            "unit": plan.unit_label,
            "notes": plan.notes,
        },
        indent=2,
    )


def check_live_access(provider: Provider, *, allow_paid: bool) -> None:
    status = provider.status()
    if not status.available:
        raise ProviderError(status.message, status=401)
    if status.paid_calls_possible and not allow_paid:
        raise ProviderError("paid provider calls are disabled; review the plan and pass --allow-paid explicitly", status=402)
