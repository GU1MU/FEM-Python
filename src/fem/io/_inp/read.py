from __future__ import annotations

from pathlib import Path

from ...core.model import FEMModel
from ..inp import (
    _InpSourceSpan,
    InpImportResult,
    InpSourceLocation,
    InpSourceOccurrence,
    InpSourceSummary,
)
from .builder import build_model_with_report
from .contracts import classify_keyword
from .parser import parse_file


def read(path: str | Path) -> FEMModel:
    """Read a model while intentionally discarding import notices.

    Use :func:`read_with_report` for user-facing import workflows.
    """

    return read_with_report(path).model


def read_with_report(path: str | Path) -> InpImportResult:
    """Read an Abaqus input file and retain source-level import notices."""

    deck = parse_file(path)
    built = build_model_with_report(deck)
    return InpImportResult(
        model=built.model,
        notices=built.notices,
        source_summary=_source_summary(deck),
    )


def _source_summary(deck: object) -> InpSourceSummary:
    """Copy parser evidence into an owned, immutable public value."""

    occurrences = tuple(
        _copy_source_occurrence(occurrence)
        for occurrence in tuple(getattr(deck, "keyword_occurrences", ()))
    )
    return InpSourceSummary(occurrences=occurrences)


def _copy_source_occurrence(occurrence: object) -> InpSourceOccurrence:
    source_span = occurrence.span
    physical_locations = tuple(
        _copy_source_location(location)
        for location in source_span.physical_locations
    )
    span = _InpSourceSpan(
        start=_copy_source_location(source_span.start),
        end=_copy_source_location(source_span.end),
        physical_locations=physical_locations,
    )
    return InpSourceOccurrence(
        name=str(occurrence.name),
        params=tuple(
            (str(key), str(value))
            for key, value in tuple(occurrence.params)
        ),
        flags=tuple(str(flag) for flag in tuple(occurrence.flags)),
        _span=span,
        raw_lines=tuple(str(line) for line in tuple(occurrence.raw_lines)),
        category=classify_keyword(str(occurrence.name)),
    )


def _copy_source_location(location: object) -> InpSourceLocation:
    return InpSourceLocation(
        getattr(location, "path", None),
        int(location.line),
        getattr(location, "keyword", None),
    )
