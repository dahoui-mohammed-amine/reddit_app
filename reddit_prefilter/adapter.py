from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import RawRecord, SourceLineage

_LINEAGE_FIELDS = frozenset(
    {
        "provider",
        "observed_at",
        "source_url",
        "request_id",
        "run_id",
        "response_status",
        "cache_status",
        "metadata",
    }
)


class AdapterError(ValueError):
    """The fixture does not satisfy the explicit pre-filter envelope."""


def _lineage(value: Any, default: SourceLineage | None) -> Any:
    if value is None:
        return default
    if isinstance(value, SourceLineage):
        return value
    if not isinstance(value, Mapping):
        return value
    if "provider" not in value or "observed_at" not in value:
        return value
    if any(key not in _LINEAGE_FIELDS for key in value):
        return value
    lineage = SourceLineage(
        provider=value["provider"],
        observed_at=value["observed_at"],
        source_url=value.get("source_url"),
        request_id=value.get("request_id"),
        run_id=value.get("run_id"),
        response_status=value.get("response_status"),
        cache_status=value.get("cache_status"),
        metadata=value.get("metadata", {}),
    )
    if not lineage.is_valid():
        return value
    return lineage


def records_from_fixture(
    payload: Mapping[str, Any],
    *,
    default_lineage: SourceLineage | None = None,
) -> tuple[RawRecord, ...]:
    """Convert the canonical fixture envelope into raw records.

    The only accepted top-level shape is ``{"records": [...]}``. Each record
    envelope has ``record_type`` and ``raw`` keys, with optional lineage and
    ``subreddit_context``. Invalid entries are retained as ``RawRecord`` values
    so the pre-filter can emit an auditable incomplete decision instead of
    silently dropping fixture evidence.
    """

    if not isinstance(payload, Mapping):
        raise AdapterError("fixture must be an object")
    entries = payload.get("records")
    if not isinstance(entries, list):
        raise AdapterError("fixture.records must be a list")

    records: list[RawRecord] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            records.append(RawRecord("unknown", entry, default_lineage))
            continue
        has_raw = "raw" in entry
        record_type = entry.get("record_type", "unknown") if has_raw else "unknown"
        raw = entry.get("raw")
        if not has_raw:
            # Retain the malformed envelope itself as evidence.
            raw = dict(entry)
        subreddit_context = entry.get("subreddit_context")
        records.append(
            RawRecord(
                record_type,
                raw,
                _lineage(entry.get("lineage"), default_lineage),
                subreddit_context,
            )
        )
    return tuple(records)


def load_fixture(
    path: str | Path,
    *,
    default_lineage: SourceLineage | None = None,
) -> tuple[RawRecord, ...]:
    """Load a canonical JSON fixture without making provider calls."""

    fixture_path = Path(path)
    try:
        with fixture_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"could not load fixture {fixture_path}: {exc}") from exc
    return records_from_fixture(payload, default_lineage=default_lineage)
