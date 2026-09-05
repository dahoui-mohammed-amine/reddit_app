from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .db import Database
from .providers import ProviderError, make_provider
from .runner import check_live_access, plan_for, render_plan, run_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reddit-ingest", description="Run one local-first Reddit ingestion pass.")
    parser.add_argument("command", choices=("status", "plan", "run", "purge"))
    parser.add_argument("--config", default="config.toml", help="TOML configuration path")
    parser.add_argument("--mode", choices=("run", "discover", "refresh"), default="run", help="one-shot operation mode")
    parser.add_argument("--allow-paid", action="store_true", help="explicitly permit live provider calls that may consume paid units")
    parser.add_argument("--dry-run", action="store_true", help="show status and plan without network calls or ingestion writes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        provider = make_provider(config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "configuration_error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    db: Database | None = None
    try:
        status = provider.status()
        if args.dry_run:
            plan = provider.plan(config, 0)
            print(json.dumps({"status": "dry_run", "provider": asdict(status), "database": str(config.database_path), "plan": json.loads(render_plan(plan))}, indent=2))
            return 0
        db = Database(config.database_path)
        plan = plan_for(db, provider, config)
        if args.command == "status":
            print(json.dumps({"provider": asdict(status), "database": str(config.database_path), "counts": db.counts(), "subreddits": list(config.subreddits)}, indent=2))
            return 0 if status.available else 2
        if args.command == "plan":
            print(render_plan(plan))
            return 0
        if args.command == "purge":
            purged = db.purge_deleted_content()
            print(json.dumps({"status": "purged", "posts_cleared": purged}, indent=2))
            return 0
        # The estimate is emitted before a live call. Full comments remain an
        # explicit mode in config and require the same paid-call confirmation.
        print(json.dumps({"status": "plan", "provider": asdict(status), "plan": json.loads(render_plan(plan))}, indent=2))
        if provider.name != "fixture":
            check_live_access(provider, allow_paid=args.allow_paid)
        summary = run_once(db, provider, config, args.mode)
        print(json.dumps({"status": summary.status, **asdict(summary)}, indent=2))
        return 0
    except ProviderError as exc:
        print(json.dumps({"status": "provider_blocked", "error": str(exc), "http_status": exc.status, "retryable": exc.retryable, "billed": exc.billed}, indent=2), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "run_error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
