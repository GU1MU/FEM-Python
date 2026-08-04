from __future__ import annotations

from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Iterable as _Iterable

from ..core.model import FEMModel as _FEMModel


@_dataclass(frozen=True, slots=True)
class InpSourceLocation:
    """One physical source location in an INP input artifact."""

    path: _Path | None
    line: int
    keyword: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.line, bool) or int(self.line) < 1:
            raise ValueError("INP source line must be a positive integer")
        object.__setattr__(self, "line", int(self.line))
        if self.path is not None and not isinstance(self.path, _Path):
            object.__setattr__(self, "path", _Path(self.path))

    def format(self) -> str:
        prefix = str(self.path) if self.path is not None else "<input>"
        context = f"{prefix}:{self.line}"
        if self.keyword:
            context += f" [*{self.keyword.upper()}]"
        return context


@_dataclass(frozen=True, slots=True)
class _InpSourceSpan:
    """A logical keyword span and all of its physical source locations."""

    start: InpSourceLocation
    end: InpSourceLocation
    physical_locations: tuple[InpSourceLocation, ...]

    @property
    def location(self) -> InpSourceLocation:
        return self.start


class InpKeywordCategory(str, _Enum):
    """The single import classification used for source keyword evidence."""

    EXECUTED = "executed"
    POSTPROCESS_CANDIDATE = "postprocess candidate"
    PRESERVED = "preserved"
    HARMLESS_IGNORED = "harmless ignored"
    UNSUPPORTED_ENGINEERING_SEMANTICS = "unsupported engineering semantics"


@_dataclass(frozen=True, slots=True)
class InpSourceOccurrence:
    """Detached source evidence for one keyword occurrence."""

    name: str
    params: tuple[tuple[str, str], ...]
    flags: tuple[str, ...]
    _span: _InpSourceSpan
    raw_lines: tuple[str, ...] = ()
    category: InpKeywordCategory = (
        InpKeywordCategory.UNSUPPORTED_ENGINEERING_SEMANTICS
    )

    @property
    def location(self) -> InpSourceLocation:
        return self._span.start


@_dataclass(frozen=True, slots=True)
class InpSourceSummary:
    """Read-only source evidence retained by the complete-model facade."""

    occurrences: tuple[InpSourceOccurrence, ...] = ()


@_dataclass(frozen=True, slots=True)
class InpImportNotice:
    """One non-authoritative limitation reported by an INP import."""

    code: str
    message: str
    locations: tuple[InpSourceLocation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "locations", tuple(self.locations))


@_dataclass(frozen=True, slots=True)
class InpImportResult:
    """A detached model, notices, and optional read-only source evidence."""

    model: _FEMModel
    notices: tuple[InpImportNotice, ...] = ()
    source_summary: InpSourceSummary | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "notices", tuple(self.notices))


class InpInputError(ValueError):
    """Base input error retaining source and remediation evidence."""

    default_code = "abaqus.input.invalid"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        location: InpSourceLocation | None = None,
        record: _Any = None,
        remediation: str | None = None,
        locations: _Iterable[InpSourceLocation] = (),
        path: str | _Path | None = None,
        line: int | None = None,
        keyword: str | None = None,
    ) -> None:
        if location is None and (path is not None or line is not None):
            location = InpSourceLocation(
                None if path is None else _Path(path),
                1 if line is None else line,
                keyword,
            )

        ordered_locations: list[InpSourceLocation] = []
        if location is not None:
            ordered_locations.append(location)
        for item in locations:
            if item not in ordered_locations:
                ordered_locations.append(item)

        self.code = str(code or self.default_code)
        self.location = ordered_locations[0] if ordered_locations else None
        self.locations = tuple(ordered_locations)
        self.path = self.location.path if self.location is not None else None
        self.line = self.location.line if self.location is not None else None
        self.keyword = (
            self.location.keyword if self.location is not None else keyword
        )
        self.record = record
        self.remediation = remediation
        self.message = str(message)

        rendered = self.message
        if self.location is not None:
            rendered = f"{self.location.format()}: {rendered}"
        if remediation:
            rendered += f" Remediation: {remediation}"
        super().__init__(rendered)


class InpParseError(InpInputError):
    """A lexical or structural INP input error."""

    default_code = "abaqus.parse.invalid"


class InpBuildError(InpInputError):
    """A semantic construction error for a parsed INP source."""

    default_code = "abaqus.build.invalid"


class UnsupportedInpFeatureError(InpInputError):
    """A valid INP feature outside the currently implemented capability."""

    default_code = "abaqus.feature.unsupported"


def read(path: str | _Path) -> _FEMModel:
    """Read a complete INP model while discarding non-authoritative notices."""

    return read_with_report(path).model


def read_with_report(path: str | _Path) -> InpImportResult:
    """Read a complete INP model through the public facade."""

    from ._inp.read import read_with_report as _read_with_report

    return _read_with_report(path)


globals().pop("annotations", None)


__all__ = [
    "InpBuildError",
    "InpImportNotice",
    "InpImportResult",
    "InpInputError",
    "InpKeywordCategory",
    "InpParseError",
    "InpSourceLocation",
    "InpSourceOccurrence",
    "InpSourceSummary",
    "UnsupportedInpFeatureError",
    "read",
    "read_with_report",
]
