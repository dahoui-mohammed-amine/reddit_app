from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .comments import apply_comment_policy
from .config import Config
from .db import Database
from .models import Gap, PageResult, Plan, RefreshResult, RunSummary, RequestRecord
from .normalize import utc_now
from .providers import Provider, ProviderError


def config_dict(config: Config) -> dict[str, Any]:
    data = asdict(config)
    data["database_path"] = str(config.database_path)
    data["fixture_path"] = str(config.fixture_path) if config.fixture_path else None
    data["subreddits"] = list(config.subreddits)
    return data


def plan_for(db: Database, provider: Provider, config: Config) -> Plan:
    known = db.counts()["posts"]
    return provider.plan(config, known)


def _request_units(records: list[RequestRecord], request_id: str | None) -> int:
    if records:
        return sum(max(1, record.request_units) for record in records)
    return 1 if request_id else 0


def _failure_record(operation: str, error: ProviderError) -> list[RequestRecord]:
    if error.request_id is None:
        return []
    metadata = {"error": str(error)}
    if error.url:
        metadata["url"] = error.url
    return [RequestRecord(error.request_id, operation, error.status, "unknown", None, error.billed, metadata)]


def run_once(db: Database, provider: Provider, config: Config, mode: str) -> RunSummary:
    if mode not in {"run", "discover", "refresh"}:
        raise ValueError("mode must be run, discover, or refresh")
    started = utc_now()
    run_id = db.start_run(provider.name, mode, config_dict(config), started)
    summary = RunSummary(run_id)
    try:
        if mode in {"run", "discover"}:
            for subreddit in config.subreddits:
                checkpoint = db.checkpoint(subreddit)
                cursor = checkpoint["cursor"] if checkpoint and checkpoint["provider"] == provider.name else None
                for _page_number in range(config.max_discovery_pages):
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
                            request_records=_failure_record("discover", exc),
                        )
                        with db.transaction():
                            db.save_page(run_id, provider.name, subreddit, page)
                        summary.gaps += len(page.gaps)
                        summary.requests += _request_units(page.request_records, page.request_id)
                        break
                    page.gaps.extend(apply_comment_policy(page.posts, config))
                    with db.transaction():
                        discovered, comments = db.save_page(run_id, provider.name, subreddit, page)
                    summary.discovered += discovered
                    summary.comments += comments
                    summary.gaps += len(page.gaps)
                    summary.requests += _request_units(page.request_records, page.request_id)
                    if not page.next_cursor or provider.name == "fetchlayer":
                        break
                    cursor = page.next_cursor
        if mode in {"run", "refresh"}:
            due = db.due_posts(config.refresh_interval_minutes, config.max_refresh_posts)
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
                        request_records=_failure_record("refresh", exc),
                    )
                result.gaps.extend(apply_comment_policy(result.posts, config))
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
