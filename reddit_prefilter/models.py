from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Literal, Mapping

RecordType = Literal["post", "comment"]
DecisionStatus = Literal["accepted", "rejected", "incomplete"]


def _json_default(value: Any) -> Any:
    """Keep evidence IDs deterministic for values outside the JSON contract."""
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return list(value)
    return {"type": type(value).__name__, "value": str(value)}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {deepcopy(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {deepcopy(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return [_thaw(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class SourceLineage:
    """Provider evidence needed to trace a record back to its observation."""

    provider: str
    observed_at: str
    source_url: str | None = None
    request_id: str | None = None
    run_id: str | None = None
    response_status: int | None = None
    cache_status: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lineage_to_dict(lineage: Any) -> dict[str, Any] | None:
    return lineage.to_dict() if isinstance(lineage, SourceLineage) else None


@dataclass(frozen=True, slots=True)
class RawRecord:
    """The adapter boundary: raw provider mapping plus observation lineage.

    ``raw`` is captured as an immutable snapshot and is never normalized by the
    pre-filter. ``subreddit`` is optional adapter context for records (usually
    comments) whose provider payload does not carry the parent post's subreddit.
    """

    record_type: str
    raw: Any
    lineage: SourceLineage | None = None
    subreddit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", _freeze(self.raw))

    @property
    def raw_sha256(self) -> str:
        return sha256_json(self.raw)

    @property
    def evidence_id(self) -> str:
        return sha256_json(
            {
                "record_type": self.record_type,
                "raw": self.raw,
                "lineage": _lineage_to_dict(self.lineage),
                "subreddit_context": self.subreddit,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "raw": _thaw(self.raw),
            "lineage": _lineage_to_dict(self.lineage),
            "subreddit_context": self.subreddit,
            "raw_sha256": self.raw_sha256,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """An auditable result that references, but does not replace, raw evidence."""

    evidence_id: str
    raw_sha256: str
    record_type: str
    record_id: str | None
    status: DecisionStatus
    reason_codes: tuple[str, ...]
    lineage: SourceLineage | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "raw_sha256": self.raw_sha256,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "lineage": _lineage_to_dict(self.lineage),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PrefilterResult:
    """All input evidence and one deterministic decision per input occurrence."""

    records: tuple[RawRecord, ...]
    decisions: tuple[Decision, ...]

    @property
    def accepted(self) -> tuple[Decision, ...]:
        return tuple(item for item in self.decisions if item.status == "accepted")

    @property
    def rejected(self) -> tuple[Decision, ...]:
        return tuple(item for item in self.decisions if item.status == "rejected")

    @property
    def incomplete(self) -> tuple[Decision, ...]:
        return tuple(item for item in self.decisions if item.status == "incomplete")

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "decisions": [decision.to_dict() for decision in self.decisions],
        }
