# Reddit ingestion node

A small local-first Reddit ingestion boundary. It discovers `new` posts, stores current normalized post/comment records, and appends post engagement observations so later analysis can compare observations. It does not calculate hotness, trends, opportunities, or any downstream result.

## Safe first run

The checked-in configuration uses the fixture provider. It performs no network calls, needs no credentials, and consumes no provider units.

```sh
uv sync
uv run reddit-ingest status --config config.toml
uv run reddit-ingest plan --config config.toml
uv run reddit-ingest run --config config.toml --mode discover
uv run reddit-ingest run --config config.toml --mode refresh
uv run reddit-ingest status --config config.toml
```

The SQLite database is `data/reddit.sqlite3` by default and is intentionally ignored by Git. The sample fixture covers `r/freelance`, `r/smallbusiness`, and `r/SaaS`, pagination, changing scores/comment counts, and comments.

## Configuration

Copy `config.example.toml` and edit the explicit `[ingestion]`, `[comments]`, and `[provider]` sections. The initial subreddits are configuration values, not hidden in application code.

- `listing_limit` is capped at Reddit-style 100.
- `refresh_interval_minutes` controls which stored posts are due for a later `refresh` pass.
- `comments.mode` is `bounded` or `full`. Bounded mode records explicit gaps when depth/limit truncates comments. Full mode is opt-in.
- `provider.min_request_interval_seconds` paces live requests before each attempt; retry backoff and provider retry-after values still apply.
- `provider.name` is `fixture`, `redditapis`, or `fetchlayer`.

## Provider access and cost gate

The app contains configuration seams for the two researched low-cost providers, but does not claim live access:

```sh
# RedditAPIs adapter; no call occurs unless the key exists and --allow-paid is supplied.
export REDDITAPIS_API_KEY='...'
uv run reddit-ingest plan --config config.toml
uv run reddit-ingest run --config config.toml --allow-paid

# FetchLayer adapter
export FETCHLAYER_API_KEY='...'
uv run reddit-ingest run --config config.toml --allow-paid
```

Change `provider.name` before using either adapter. Missing credentials are reported as a provider-access error. The application never automatically falls back from one provider to another, and it never silently expands every comment. Review the plan and provider terms first; no real secrets belong in this repository.

The RedditAPIs adapter uses the documented subreddit listing and up-to-100 `t3_` fullname batch refresh shape. The FetchLayer adapter uses subreddit-post and one-post-URL endpoints. Provider responses retain cache/status metadata where supplied. Score and comment-count changes are observations, not exact vote-arrival rates.

## Data and deletion posture

SQLite tables include:

- current `posts` and `comments`, keyed by stable IDs and preserving parent/link relationships;
- append-only `post_observations` for engagement history only;
- `checkpoints` for subreddit cursors;
- `requests` and `gaps` for source metadata, failures, blocked/deleted/removed/truncated/unexpanded content, and resumability;
- `runs` for one-shot orchestration boundaries.

Deleted or removed content clears mutable text/author fields while retaining stable identity and status. To clear those fields again after policy changes:

```sh
uv run reddit-ingest purge --config config.toml
```

This is not an immutable raw Reddit archive. Confirm applicable Reddit/provider retention and deletion obligations before storing live content.

## One-shot scheduling

There is no daemon or queue. Run the command once from a macOS launchd job, cron entry, or another scheduler after choosing cadence and budget. Use `--mode discover` for listings and `--mode refresh` for due known posts; `--mode run` performs both in one invocation.

## Tests

Tests are fixture-driven and require no credentials or network calls:

```sh
uv run python -m unittest discover -s tests -v
```

They cover normalization, cursor resume, idempotency, historical observations, bounded comment gaps, deletion/update clearing, retryable provider errors, and the live cost/access gate.
