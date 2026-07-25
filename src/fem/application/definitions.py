"""Headless editable project definitions shared by the application and GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class NativePart:
    """Small serialisable representation of one editable native part."""

    name: str = "Part-1"
    body_name: str = "Body-1"


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """One item in the shallow native feature history."""

    name: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NamedRegion:
    """A logical native region mapped to mesh sets after regeneration."""

    name: str
    entity_kind: Literal["point", "edge", "face", "body"]
    entity_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SectionDefinition:
    """Editable section definition with a material linked by name."""

    name: str
    material: str
    section_type: str = "solid"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RegionAssignment:
    """Assign one named section to an existing element region."""

    section_name: str
    region_name: str


__all__ = [
    "FeatureRecord",
    "NamedRegion",
    "NativePart",
    "RegionAssignment",
    "SectionDefinition",
]
