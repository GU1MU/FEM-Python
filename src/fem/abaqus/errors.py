from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class AbaqusSourceLocation:
    """One physical source location in an Abaqus input artifact."""

    path: Path | None
    line: int
    keyword: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.line, bool) or int(self.line) < 1:
            raise ValueError("Abaqus source line must be a positive integer")
        object.__setattr__(self, "line", int(self.line))
        if self.path is not None and not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))

    def format(self) -> str:
        prefix = str(self.path) if self.path is not None else "<input>"
        context = f"{prefix}:{self.line}"
        if self.keyword:
            context += f" [*{self.keyword.upper()}]"
        return context


class AbaqusInputError(ValueError):
    """Base input error retaining structured source and remediation evidence."""

    default_code = "abaqus.input.invalid"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        location: AbaqusSourceLocation | None = None,
        record: Any = None,
        remediation: str | None = None,
        locations: Iterable[AbaqusSourceLocation] = (),
        path: str | Path | None = None,
        line: int | None = None,
        keyword: str | None = None,
    ) -> None:
        if location is None and (path is not None or line is not None):
            location = AbaqusSourceLocation(
                None if path is None else Path(path),
                1 if line is None else line,
                keyword,
            )

        ordered_locations: list[AbaqusSourceLocation] = []
        if location is not None:
            ordered_locations.append(location)
        for item in locations:
            if item not in ordered_locations:
                ordered_locations.append(item)

        self.code = str(code or self.default_code)
        self.location = (
            ordered_locations[0] if ordered_locations else None
        )
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


class AbaqusParseError(AbaqusInputError):
    """A lexical or structural Abaqus input error."""

    default_code = "abaqus.parse.invalid"


class AbaqusBuildError(AbaqusInputError):
    """An Abaqus-to-domain semantic construction error."""

    default_code = "abaqus.build.invalid"


class UnsupportedAbaqusFeatureError(AbaqusInputError):
    """A valid Abaqus feature outside the explicitly supported subset."""

    default_code = "abaqus.feature.unsupported"


__all__ = [
    "AbaqusBuildError",
    "AbaqusInputError",
    "AbaqusParseError",
    "AbaqusSourceLocation",
    "UnsupportedAbaqusFeatureError",
]
