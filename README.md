# Reddit pre-filter node

This repository contains a small, standalone pre-filter for Reddit post and
comment observations. It makes only deterministic structural and high-signal
checks; it does not call Reddit, use an LLM, or infer semantic relevance.

## Adapter boundary

The public input boundary is `reddit_prefilter.RawRecord`:

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
result = Prefilter(PrefilterConfig(minimum_text_length=20, subreddit_scope=("freelance",))).evaluate(records)
```

`RawRecord.raw` is an immutable snapshot of the untouched provider mapping.
`SourceLineage` carries the provider, observation time, source URL, request/run
identifiers, response and cache metadata. A comment may carry `subreddit` on
`RawRecord` as adapter context when that field is absent from the comment
payload; its raw `link_id`/`post_id` and `parent_id` are still preserved and
checked.

A fixture can use the same envelope through `records_from_fixture`; a
file-backed fixture can be loaded with `load_fixture` from
`reddit_prefilter.adapter`:

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

The result contains **all** input records and one `Decision` per occurrence.
Each decision references a SHA-256 evidence ID and includes the raw hash,
lineage, rule version, and stable reason codes. Rule-specific metadata records
normalized text length, relationship IDs, scope results, and matched markers
when those checks can be evaluated. Filtering therefore never deletes source
evidence. Duplicate identity is `(record_type, canonical ID)`; the first
occurrence in input order is canonical and later occurrences are rejected as
`DUPLICATE_RECORD`.

## Rules

A record is `accepted`, `rejected`, or `incomplete`:

- missing/invalid/conflicting IDs, malformed records, missing comment-to-post
  relationships, bad parent IDs, missing lineage, or unavailable configured
  subreddit are `incomplete`;
- deleted or removed content, duplicate IDs, text shorter than
  `minimum_text_length`, literal case-insensitive configured spam markers, and
  out-of-scope subreddits are `rejected`;
- otherwise it is `accepted` with `ACCEPTED`.

Post text is title plus body; comment text is body. Whitespace is normalized
before counting or matching. `minimum_text_length=0` disables that rule. An
empty `subreddit_scope` disables scope checking. The default spam list is
intentionally short and literal (`buy now`, `free money`, `click here`, `promo
code`, `crypto giveaway`, `telegram.me`, and `discord.gg/`).

## Integration decision

The active ingestion implementation is being validated in another worktree.
Its normalized `PostSnapshot`/`CommentSnapshot` models and `PageResult` expose
IDs, relationships, content state, and request metadata, but the current
contract/database does **not** retain the untouched provider payload. A direct
snapshot/SQLite adapter would therefore violate this node's raw-evidence and
lineage requirement.

Consequently this node intentionally stops at the documented `RawRecord`
adapter boundary and does not import or modify `reddit_ingestion`. Once
ingestion lands, its provider boundary must emit one `RawRecord` per raw post
or comment (including source lineage, and adapter subreddit context for
comments where needed). Do not substitute normalized snapshots as raw
source evidence.

## Tests

No credentials or network access are needed:

```sh
python -m unittest discover -s tests -v
```
