# Reddit ingestion and pre-filter nodes

This repository contains the local-first Reddit ingestion node and a standalone deterministic pre-filter.
The ingestion node discovers new posts, stores current normalized post and comment records, and appends post engagement observations for later analysis.
The pre-filter classifies raw post and comment observations without calling Reddit, using an LLM, or inferring semantic relevance.
Neither node calculates hotness, trends, opportunities, or any downstream result.

## Safe first run

The checked-in configuration uses the fixture provider.
It performs no network calls, needs no credentials, and consumes no provider units.

```sh
uv sync
uv run reddit-ingest run --config config.toml --dry-run
uv run reddit-ingest status --config config.toml
uv run reddit-ingest plan --config config.toml
uv run reddit-ingest run --config config.toml --mode discover
uv run reddit-ingest run --config config.toml --mode refresh
uv run reddit-ingest status --config config.toml
```

`--dry-run` reports provider status and the estimated plan without network calls or ingestion writes.
The SQLite database is `data/reddit.sqlite3` by default and is intentionally ignored by Git.
The sample fixture covers `r/freelance`, `r/smallbusiness`, and `r/SaaS`, pagination, changing scores and comment counts, and comments.

## Configuration

Copy `config.example.toml` and edit the explicit `[ingestion]`, `[comments]`, and `[provider]` sections.
The initial subreddits are configuration values, not hidden in application code.

- `listing_limit` is capped at Reddit-style 100.
- `refresh_interval_minutes` controls which stored posts are due for a later `refresh` pass.
- `refresh_expiry_days` sets each post's refresh-until window from its source creation time when available, falling back to observation time.
- Expired posts remain stored with their history but are no longer refreshed.
- `comments.mode` is `bounded` or `full`.
- Bounded mode records explicit gaps when depth or limit truncates comments.
- Full mode is opt-in.
- FetchLayer reports full expansion as unsupported.
- `provider.min_request_interval_seconds` paces live requests before each attempt.
- Retry backoff and provider retry-after values still apply.
- `provider.name` is `fixture`, `redditapis`, or `fetchlayer`.

## Provider access and cost gate

The app contains configuration seams for the two researched low-cost providers.
It does not claim live access without a configured credential.

```sh
# Edit config.toml first: [provider] name = "redditapis".
export REDDITAPIS_API_KEY='...'
uv run reddit-ingest plan --config config.toml --mode discover
uv run reddit-ingest run --config config.toml --mode discover --allow-paid

# Edit config.toml first: [provider] name = "fetchlayer".
export FETCHLAYER_API_KEY='...'
uv run reddit-ingest run --config config.toml --mode discover --allow-paid
```

No call occurs unless the selected provider's key exists and `--allow-paid` is supplied.
Missing credentials are reported as a provider-access error.
The application never automatically falls back from one provider to another.
It never silently expands every comment.
Review the plan and provider terms first.
No real secrets belong in this repository.

The RedditAPIs adapter uses the documented subreddit listing and up-to-100 `t3_` fullname batch refresh shape. Its documented `upvotes` net-vote field maps to canonical `score`, `comments` maps to canonical `num_comments`, and the observed `ups` value remains separate; unsupported metric names stay unknown. Its comment adapter unwraps documented native `t1`/`data` nodes and nested `Listing` replies, preserving `id`/`name`, `link_id`, `parent_id`, timestamps, scores, and deletion markers. Documented `more` nodes remain explicit incomplete-comment gaps.
The FetchLayer adapter uses subreddit-post and one-post-URL endpoints.
Its full comment expansion is unsupported and records an explicit unexpanded gap while still refreshing post metrics.
Provider responses retain cache and status metadata where supplied.
Score and comment-count changes are observations, not exact vote-arrival rates.

## Ingestion data and deletion posture

SQLite tables include:

- current `posts` and `comments`, keyed by stable IDs and preserving parent and link relationships;
- append-only `post_observations` for engagement history only;
- `checkpoints` for subreddit cursors;
- `requests` and `gaps` for source metadata, failures, blocked, deleted, removed, truncated, and unexpanded content;
- `runs` for one-shot orchestration boundaries.

Deleted or removed content clears mutable text and author fields while retaining stable identity and status.
To clear those fields again after policy changes:

```sh
uv run reddit-ingest purge --config config.toml
```

The ingestion database is not an immutable raw Reddit archive.
Confirm applicable Reddit and provider retention and deletion obligations before storing live content.

## One-shot scheduling

There is no daemon or queue.
Run the command once from a macOS launchd job, cron entry, or another scheduler after choosing cadence and budget.
Use `--mode discover` for listings and `--mode refresh` for due known posts.
Use `--mode run` for both operations in one invocation.

## Pre-filter node

The pre-filter public input boundary is `reddit_prefilter.RawRecord`:

```python
from reddit_prefilter import Prefilter, PrefilterConfig, RawRecord, SourceLineage

records = [
    RawRecord(
        record_type="post",
        raw={"id": "abc", "subreddit": "freelance", "title": "...", "selftext": "..."},
        lineage=SourceLineage(
            provider="fixture",
            observed_at="2026-09-05T00:00:00Z",
            source_url="fixture://freelance/new",
            request_id="request-1",
        ),
    )
]
result = Prefilter(
    PrefilterConfig(minimum_text_length=20, subreddit_scope=("freelance",))
).evaluate(records)
```

`RawRecord.raw` is an immutable snapshot of the untouched provider mapping.
`SourceLineage` carries the provider, observation time, source URL, request and run identifiers, response metadata, and cache metadata.
A comment may carry `subreddit` on `RawRecord` as adapter context when that field is absent from the comment payload.
Its raw `link_id`, `post_id`, and `parent_id` remain preserved and checked.

A fixture can use the same envelope through `records_from_fixture` or `load_fixture`:

```json
{
  "records": [
    {
      "record_type": "post",
      "lineage": {"provider": "fixture", "observed_at": "2026-09-05T00:00:00Z"},
      "raw": {"id": "abc", "subreddit": "freelance", "title": "..."}
    }
  ]
}
```

The result contains all input records and one `Decision` per occurrence.
Each decision references a SHA-256 evidence ID and includes the raw hash, lineage, rule version, normalized text length, relationship IDs, scope result, matched markers, and stable reason codes.
Filtering therefore never deletes source evidence.
Duplicate identity is `(record_type, canonical ID)`.
The first occurrence in input order is canonical and later occurrences are rejected as `DUPLICATE_RECORD`.

## Pre-filter rules

A record is `accepted`, `rejected`, or `incomplete`.

- missing, invalid, or conflicting IDs are `incomplete`;
- malformed records, lineage, relationships, or content state are `incomplete`;
- missing comment-to-post relationships and bad parent IDs are `incomplete`;
- missing or unavailable configured subreddit context is `incomplete`;
- deleted or removed content is `rejected`;
- duplicate IDs are `rejected`;
- text shorter than `minimum_text_length` is `rejected`;
- literal case-insensitive configured spam markers are `rejected`;
- out-of-scope subreddits are `rejected`;
- all other valid records are `accepted` with `ACCEPTED`.

Post text is the title plus body.
Comment text is the body.
Whitespace is normalized before counting or matching.
`minimum_text_length=0` disables that rule.
An empty `subreddit_scope` disables scope checking.
The default spam list is intentionally short and literal: `buy now`, `free money`, `click here`, `promo code`, `crypto giveaway`, `telegram.me`, and `discord.gg/`.

## Ingestion integration boundary

The pre-filter intentionally accepts raw evidence through `RawRecord` rather than importing ingestion internals.
The current ingestion models expose normalized post and comment snapshots, relationships, and request metadata, but they do not retain every untouched provider payload.
A direct snapshot adapter would therefore lose evidence required by this node.

When these nodes are connected, the ingestion provider boundary must emit one `RawRecord` per raw post or comment with source lineage and adapter subreddit context where needed.
Normalized snapshots must not be substituted for raw source evidence.

## Tests

The test suite is fixture-driven and requires no credentials or network calls:

```sh
uv run python -m unittest discover -s tests -v
uvx ruff check .
uvx ruff format --check .
```

The tests cover ingestion normalization, cursor resume, idempotency, historical observations, bounded comment gaps, deletion and update clearing, provider access and cost gating, deterministic pre-filter rules, malformed evidence, relationship preservation, and repeat-run idempotence.
