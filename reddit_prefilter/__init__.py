"""Deterministic, evidence-preserving Reddit pre-filter."""

from .adapter import AdapterError, records_from_fixture
from .models import Decision, PrefilterResult, RawRecord, SourceLineage
from .prefilter import DEFAULT_SPAM_MARKERS, Prefilter, PrefilterConfig

__all__ = [
    "AdapterError",
    "DEFAULT_SPAM_MARKERS",
    "Decision",
    "Prefilter",
    "PrefilterConfig",
    "PrefilterResult",
    "RawRecord",
    "SourceLineage",
    "records_from_fixture",
]
