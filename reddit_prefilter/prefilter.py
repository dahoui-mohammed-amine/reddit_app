from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

from .models import Decision, PrefilterResult, RawRecord, SourceLineage

# These are deliberately literal, high-signal markers. This node does not try
# to infer relevance, sentiment, or spam semantics.
DEFAULT_SPAM_MARKERS = (
    "spam",
    "buy now",
    "free money",
    "click here",
    "promo code",
    "crypto giveaway",
    "telegram.me",
    "discord.gg/",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_QUALIFIED_ID_RE = re.compile(r"^t\d+_")
_REASON_ORDER = (
    "MALFORMED_RECORD",
    "MISSING_SOURCE_LINEAGE",
    "INVALID_SOURCE_LINEAGE",
    "MISSING_IDENTIFIER",
    "INVALID_IDENTIFIER",
    "IDENTIFIER_CONFLICT",
    "MISSING_POST_RELATIONSHIP",
    "INVALID_POST_RELATIONSHIP",
    "RELATIONSHIP_CONFLICT",
    "INVALID_PARENT_ID",
    "SUBREDDIT_UNAVAILABLE",
    "INVALID_SUBREDDIT",
    "SUBREDDIT_CONFLICT",
    "DUPLICATE_RECORD",
    "DELETED_CONTENT",
    "REMOVED_CONTENT",
    "OUT_OF_SCOPE",
    "TEXT_TOO_SHORT",
    "SPAM_MARKER",
    "ACCEPTED",
)
_REASON_INDEX = {reason: index for index, reason in enumerate(_REASON_ORDER)}
_INCOMPLETE_REASONS = {
    "MALFORMED_RECORD",
    "MISSING_SOURCE_LINEAGE",
    "INVALID_SOURCE_LINEAGE",
    "MISSING_IDENTIFIER",
    "INVALID_IDENTIFIER",
    "IDENTIFIER_CONFLICT",
    "MISSING_POST_RELATIONSHIP",
    "INVALID_POST_RELATIONSHIP",
    "RELATIONSHIP_CONFLICT",
    "INVALID_PARENT_ID",
    "SUBREDDIT_UNAVAILABLE",
    "INVALID_SUBREDDIT",
    "SUBREDDIT_CONFLICT",
}
_REJECTED_REASONS = {
    "DUPLICATE_RECORD",
    "DELETED_CONTENT",
    "REMOVED_CONTENT",
    "OUT_OF_SCOPE",
    "TEXT_TOO_SHORT",
    "SPAM_MARKER",
}


@dataclass(frozen=True, slots=True)
class PrefilterConfig:
    """Explicit, deterministic rules for the standalone pre-filter.

    An empty ``subreddit_scope`` disables scope checking. ``minimum_text_length``
    counts normalized characters in a post's title plus body, or a comment's
    body. Set it to zero to disable the text-length rule. Spam markers are
    case-insensitive literal substrings and can be replaced for a deployment.
    """

    minimum_text_length: int = 1
    subreddit_scope: tuple[str, ...] = ()
    spam_markers: tuple[str, ...] = DEFAULT_SPAM_MARKERS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.minimum_text_length, int)
            or isinstance(self.minimum_text_length, bool)
            or self.minimum_text_length < 0
        ):
            raise ValueError("minimum_text_length must be a non-negative integer")
        if any(
            not isinstance(item, str) or not item.strip() or not item.strip().casefold().removeprefix("r/")
            for item in self.subreddit_scope
        ):
            raise ValueError("subreddit_scope must contain non-empty subreddit names")
        if any(not isinstance(item, str) or not item.strip() for item in self.spam_markers):
            raise ValueError("spam_markers must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class _Identity:
    record_id: str | None
    source: str | None
    candidates: tuple[tuple[str, str | None], ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Relationship:
    post_id: str | None
    parent_id: str | None
    reasons: tuple[str, ...]


def _present(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _first_non_null(raw: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = raw.get(field)
        if value is not None:
            return value
    return None


def _canonical_id(value: Any, prefix: str) -> str | None:
    if not _present(value):
        return None
    if isinstance(value, bool):
        return None
    candidate = str(value).strip()
    if _QUALIFIED_ID_RE.match(candidate):
        if not candidate.startswith(prefix):
            return None
        candidate = candidate[3:]
    if not candidate or not _IDENTIFIER_RE.fullmatch(candidate):
        return None
    return candidate


def _identity(raw: Mapping[str, Any], record_type: str) -> _Identity:
    if record_type == "post":
        fields = (("id", "id"), ("post_id", "post_id"), ("fullname", "fullname"), ("name", "name"))
        prefix = "t3_"
    else:
        fields = (("id", "id"), ("comment_id", "comment_id"), ("fullname", "fullname"), ("name", "name"))
        prefix = "t1_"

    candidates: list[tuple[str, str | None]] = []
    invalid = False
    for field, source in fields:
        if field not in raw or not _present(raw[field]):
            continue
        canonical = _canonical_id(raw[field], prefix)
        candidates.append((source, canonical))
        invalid = invalid or canonical is None

    reasons: list[str] = []
    valid_ids = [candidate for _, candidate in candidates if candidate is not None]
    record_id = valid_ids[0] if valid_ids else None
    source = next((source for source, candidate in candidates if candidate is not None), None)
    if not candidates:
        reasons.append("MISSING_IDENTIFIER")
    if invalid:
        reasons.append("INVALID_IDENTIFIER")
    if len(set(valid_ids)) > 1:
        reasons.append("IDENTIFIER_CONFLICT")
    return _Identity(record_id, source, tuple(candidates), tuple(reasons))


def _relationship(raw: Mapping[str, Any], record_type: str) -> _Relationship:
    if record_type != "comment":
        return _Relationship(None, None, ())

    post_fields = ("post_id", "link_id")
    post_candidates: list[tuple[str, str | None]] = []
    invalid_post = False
    for field in post_fields:
        if field not in raw or not _present(raw[field]):
            continue
        canonical = _canonical_id(raw[field], "t3_")
        post_candidates.append((field, canonical))
        invalid_post = invalid_post or canonical is None

    reasons: list[str] = []
    post_ids = [candidate for _, candidate in post_candidates if candidate is not None]
    post_id = post_ids[0] if post_ids else None
    if not post_candidates:
        reasons.append("MISSING_POST_RELATIONSHIP")
    if invalid_post:
        reasons.append("INVALID_POST_RELATIONSHIP")
    if len(set(post_ids)) > 1:
        reasons.append("RELATIONSHIP_CONFLICT")

    parent_id: str | None = None
    if "parent_id" in raw and _present(raw["parent_id"]):
        parent_id = _canonical_parent_id(raw["parent_id"])
        if parent_id is None:
            reasons.append("INVALID_PARENT_ID")

    return _Relationship(post_id, parent_id, tuple(reasons))


def _canonical_parent_id(value: Any) -> str | None:
    if not _present(value) or isinstance(value, bool):
        return None
    candidate = str(value).strip()
    if _QUALIFIED_ID_RE.match(candidate):
        if not (candidate.startswith("t1_") or candidate.startswith("t3_")):
            return None
        suffix = candidate[3:]
        return candidate if suffix and _IDENTIFIER_RE.fullmatch(suffix) else None
    return candidate if _IDENTIFIER_RE.fullmatch(candidate) else None


def _subreddit(record: RawRecord, raw: Mapping[str, Any]) -> tuple[str | None, str, tuple[str, ...]]:
    raw_value = raw.get("subreddit")
    context_value = record.subreddit
    reasons: list[str] = []
    if _present(raw_value) and _present(context_value):
        raw_subreddit = str(raw_value).strip().casefold().removeprefix("r/")
        context_subreddit = str(context_value).strip().casefold().removeprefix("r/")
        if raw_subreddit != context_subreddit:
            reasons.append("SUBREDDIT_CONFLICT")
    value = raw_value if _present(raw_value) else context_value
    if not _present(value):
        return None, "unavailable", tuple(reasons)
    result = str(value).strip().removeprefix("r/")
    if not result or any(char.isspace() for char in result):
        reasons.append("INVALID_SUBREDDIT")
        return None, "invalid", tuple(reasons)
    return result, "raw" if _present(raw_value) else "adapter_context", tuple(reasons)


def _content_state(raw: Mapping[str, Any]) -> tuple[bool, bool]:
    authors = [raw[key] for key in ("author", "author_username") if key in raw]
    deleted_markers = [raw[key] for key in ("deleted", "is_deleted") if key in raw]
    removed_markers = [raw[key] for key in ("removed", "is_removed", "removed_by_category") if key in raw]
    deleted_content = [raw[key] for key in ("body", "bodyText", "selftext", "text") if key in raw]
    removed_content = [raw[key] for key in ("body", "bodyText", "selftext", "text") if key in raw]

    def flagged(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if value is None:
            return False
        return str(value).strip().casefold() not in {"", "false", "0", "no", "none", "null"}

    def marked(values: Sequence[Any], words: set[str]) -> bool:
        return any(
            str(value).strip().casefold() in words
            for value in values
            if value is not None
        )

    deleted = any(flagged(value) for value in deleted_markers) or marked(
        authors + deleted_content, {"[deleted]", "deleted"}
    )
    removed = any(flagged(value) for value in removed_markers) or marked(
        removed_content, {"[removed]", "removed"}
    )
    # A non-empty removed_by_category is evidence of removal even when the
    # provider uses a category string rather than a boolean marker.
    if any(flagged(raw.get(key)) for key in ("removed_by_category",)):
        removed = True
    return deleted, removed


def _normalized_content(raw: Mapping[str, Any], record_type: str) -> str:
    if record_type == "post":
        values = [raw.get("title"), _first_non_null(raw, ("selftext", "body", "bodyText", "text"))]
    else:
        values = [_first_non_null(raw, ("body", "bodyText", "text"))]
    return " ".join(
        " ".join(str(value).split())
        for value in values
        if value is not None
    ).strip()


def _marker_matches(text: str, markers: Sequence[str]) -> tuple[str, ...]:
    folded = " ".join(text.casefold().split())
    matches = {
        marker.strip()
        for marker in markers
        if " ".join(marker.casefold().split()) in folded
    }
    return tuple(sorted(matches, key=lambda value: (value.casefold(), value)))


def _ordered_reasons(reasons: set[str] | Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(reasons), key=lambda reason: (_REASON_INDEX.get(reason, len(_REASON_ORDER)), reason)))


class Prefilter:
    """Run only structural and high-signal deterministic pre-filter rules."""

    rule_version = "reddit-prefilter-v1"

    def __init__(self, config: PrefilterConfig | None = None):
        self.config = config or PrefilterConfig()
        self._scope = frozenset(self._normalize_subreddit(item) for item in self.config.subreddit_scope)

    @staticmethod
    def _normalize_subreddit(value: str) -> str:
        return value.strip().casefold().removeprefix("r/")

    def evaluate(self, records: Iterable[RawRecord]) -> PrefilterResult:
        """Return one decision per input while retaining every raw record.

        Duplicate identity is resolved by input order: the first occurrence is
        the canonical occurrence and each later occurrence is rejected with
        ``DUPLICATE_RECORD``. No record is removed from the result.
        """

        materialized = tuple(records)
        decisions: list[Decision] = []
        first_by_identity: dict[tuple[str, str], str] = {}
        for ordinal, record in enumerate(materialized):
            decision, identity_key = self._evaluate_one(record, ordinal, first_by_identity)
            if identity_key is not None and identity_key not in first_by_identity:
                first_by_identity[identity_key] = decision.evidence_id
            decisions.append(decision)
        return PrefilterResult(materialized, tuple(decisions))

    def _evaluate_one(
        self,
        record: RawRecord,
        ordinal: int,
        first_by_identity: dict[tuple[str, str], str],
    ) -> tuple[Decision, tuple[str, str] | None]:
        evidence_id = self._evidence_id(record)
        raw_sha256 = self._raw_sha256(record)
        lineage = getattr(record, "lineage", None)
        metadata: dict[str, Any] = {
            "rule_version": self.rule_version,
            "ordinal": ordinal,
            "raw_sha256": raw_sha256,
            "lineage": lineage.to_dict() if isinstance(lineage, SourceLineage) else None,
        }
        structural: set[str] = set()
        record_type = getattr(record, "record_type", "unknown")
        raw = getattr(record, "raw", None)

        if record_type not in {"post", "comment"} or not isinstance(raw, Mapping):
            structural.add("MALFORMED_RECORD")
            metadata["malformed"] = "record_type must be post/comment and raw must be an object"
            return self._decision(record, evidence_id, raw_sha256, None, structural, lineage, metadata), None
        if lineage is None:
            structural.add("MISSING_SOURCE_LINEAGE")
        elif not isinstance(lineage, SourceLineage) or (
            not isinstance(lineage.provider, str)
            or not lineage.provider.strip()
            or not isinstance(lineage.observed_at, str)
            or not lineage.observed_at.strip()
        ):
            structural.add("INVALID_SOURCE_LINEAGE")
        identity = _identity(raw, record_type)
        structural.update(identity.reasons)
        metadata["identifier_source"] = identity.source
        metadata["identifier_candidates"] = [
            {"source": source, "id": candidate}
            for source, candidate in identity.candidates
        ]
        metadata["canonical_id"] = identity.record_id

        relationship = _relationship(raw, record_type)
        structural.update(relationship.reasons)
        metadata["relationship"] = {
            "post_id": relationship.post_id,
            "parent_id": relationship.parent_id,
            "post_relationship_present": relationship.post_id is not None,
        }

        subreddit, subreddit_source, subreddit_reasons = _subreddit(record, raw)
        metadata["subreddit"] = subreddit
        metadata["subreddit_source"] = subreddit_source
        structural.update(subreddit_reasons)
        if self._scope:
            if subreddit is None:
                structural.add("SUBREDDIT_UNAVAILABLE")
            elif self._normalize_subreddit(subreddit) not in self._scope:
                metadata["configured_subreddits"] = sorted(self._scope)
                # Scope mismatch is a rejection, not an incomplete record.
                metadata["scope_match"] = False
            else:
                metadata["scope_match"] = True

        identity_key = (record_type, identity.record_id) if identity.record_id is not None else None
        duplicate_of = first_by_identity.get(identity_key) if identity_key is not None else None
        if duplicate_of is not None:
            metadata["duplicate_of"] = duplicate_of

        # A structurally incomplete record is not accepted, and its content is
        # not used to manufacture a confident rule decision. Scope and
        # duplicate evidence are still recorded when independently available.
        if structural:
            reasons = set(structural)
            if duplicate_of is not None:
                reasons.add("DUPLICATE_RECORD")
            return (
                self._decision(
                    record,
                    evidence_id,
                    raw_sha256,
                    identity.record_id,
                    reasons,
                    lineage,
                    metadata,
                ),
                identity_key,
            )

        reasons: set[str] = set()
        if duplicate_of is not None:
            reasons.add("DUPLICATE_RECORD")
        deleted, removed = _content_state(raw)
        metadata["content_state"] = {"deleted": deleted, "removed": removed}
        if deleted:
            reasons.add("DELETED_CONTENT")
        if removed:
            reasons.add("REMOVED_CONTENT")
        if self._scope and self._normalize_subreddit(subreddit or "") not in self._scope:
            reasons.add("OUT_OF_SCOPE")
            metadata["configured_subreddits"] = sorted(self._scope)
            metadata["scope_match"] = False

        content = _normalized_content(raw, record_type)
        text_length = len(content)
        metadata["text_length"] = text_length
        metadata["minimum_text_length"] = self.config.minimum_text_length
        if not deleted and not removed and text_length < self.config.minimum_text_length:
            reasons.add("TEXT_TOO_SHORT")

        matches = _marker_matches(content, self.config.spam_markers)
        metadata["matched_spam_markers"] = list(matches)
        if matches and not deleted and not removed:
            reasons.add("SPAM_MARKER")

        if not reasons:
            reasons.add("ACCEPTED")
        return (
            self._decision(
                record,
                evidence_id,
                raw_sha256,
                identity.record_id,
                reasons,
                lineage,
                metadata,
            ),
            identity_key,
        )

    def _decision(
        self,
        record: RawRecord,
        evidence_id: str,
        raw_sha256: str,
        record_id: str | None,
        reasons: set[str],
        lineage: Any,
        metadata: dict[str, Any],
    ) -> Decision:
        ordered = _ordered_reasons(reasons)
        if reasons.intersection(_INCOMPLETE_REASONS):
            status = "incomplete"
        elif "DUPLICATE_RECORD" in reasons or reasons.intersection(_REJECTED_REASONS):
            status = "rejected"
        else:
            status = "accepted"
        metadata["decision_reason_codes"] = list(ordered)
        return Decision(evidence_id, raw_sha256, record.record_type, record_id, status, ordered, lineage, metadata)

    @staticmethod
    def _raw_sha256(record: RawRecord) -> str:
        try:
            return record.raw_sha256
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _evidence_id(record: RawRecord) -> str:
        try:
            return record.evidence_id
        except (TypeError, ValueError):
            return ""
