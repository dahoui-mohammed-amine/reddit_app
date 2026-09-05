from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .comments import apply_comment_policy
from .config import Config
from .db import Database
from .models import Plan, RunSummary
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


def run_once(db: Database, provider: Provider, config: Config, mode: str) -> RunSummary:
    if mode not in {"run", "discover", "refresh"}:
        raise ValueError("mode must be run, discover, or refresh")
    started = utc_now()
    run_id = db.start_run(provider.name, mode, config_dict(config), started)
    summary = RunSummary(run_id)
    try:
        if mode in {"run", "discover"}:
            for subreddit in config.subreddits:
                # Every scheduled poll starts at the newest listing. A cursor is
                # still checkpointed and followed within this run so bounded
                # pagination can resume safely without turning the next poll
                # into an old-page crawl.
                cursor = None
                for _page_number in range(config.max_discovery_pages):
                    page = provider.discover(subreddit, cursor, config)
                    page.gaps.extend(apply_comment_policy(page.posts, config))
                    with db.transaction():
                        discovered, comments = db.save_page(run_id, provider.name, subreddit, page)
                    summary.discovered += discovered
                    summary.comments += comments
                    summary.gaps += len(page.gaps)
                    summary.requests += 1
                    if not page.next_cursor or provider.name == "fetchlayer":
                        break
                    cursor = page.next_cursor
        if mode in {"run", "refresh"}:
            due = db.due_posts(config.refresh_interval_minutes, config.max_refresh_posts)
            if due:
                result = provider.refresh_posts(due, config)
                result.gaps.extend(apply_comment_policy(result.posts, config))
                with db.transaction():
                    refreshed, comments = db.save_refresh(run_id, provider.name, result)
                summary.refreshed += refreshed
                summary.comments += comments
                summary.gaps += len(result.gaps)
                summary.requests += max(1, len(due) if provider.name == "fetchlayer" else 1)
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
