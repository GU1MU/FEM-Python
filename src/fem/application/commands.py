"""Typed, detached command-boundary values for application edits."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from fem.core.model import AnalysisStep, MaterialDefinition

from .definitions import NamedRegion, RegionAssignment, SectionDefinition


class Unset:
    """Sentinel type for a command field that the caller did not edit."""

    __slots__ = ()
    _instance: Unset | None = None

    def __new__(cls) -> Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __copy__(self) -> Unset:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Unset:
        del memo
        return self

    def __repr__(self) -> str:
        return "UNSET"


UNSET = Unset()


@dataclass(frozen=True, slots=True)
class RenameIntent:
    """Explicitly identify one existing engineering name being renamed."""

    old_name: str
    new_name: str

    def __post_init__(self) -> None:
        old_name = _command_name(self.old_name, "rename old_name")
        new_name = _command_name(self.new_name, "rename new_name")
        if old_name == new_name:
            raise ValueError("rename old_name and new_name must differ")
        object.__setattr__(self, "old_name", old_name)
        object.__setattr__(self, "new_name", new_name)


@dataclass(frozen=True, slots=True)
class DeleteIntent:
    """Explicitly identify one existing engineering name being deleted."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _command_name(self.name, "delete name"),
        )


@dataclass(frozen=True, slots=True)
class DefinitionEditBatch:
    """One atomic post-state and identity ledger for model definitions."""

    base_session_revision: int
    materials: tuple[MaterialDefinition, ...]
    sections: tuple[SectionDefinition, ...]
    assignments: tuple[RegionAssignment, ...]
    steps: tuple[AnalysisStep, ...]
    material_renames: tuple[RenameIntent, ...] = ()
    section_renames: tuple[RenameIntent, ...] = ()
    material_deletes: tuple[DeleteIntent, ...] = ()
    section_deletes: tuple[DeleteIntent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_session_revision",
            _base_revision(self.base_session_revision),
        )
        for field_name, expected_type in (
            ("materials", MaterialDefinition),
            ("sections", SectionDefinition),
            ("assignments", RegionAssignment),
            ("steps", AnalysisStep),
        ):
            values = deepcopy(tuple(getattr(self, field_name)))
            if any(type(value) is not expected_type for value in values):
                raise TypeError(
                    f"{field_name} must contain only {expected_type.__name__} values"
                )
            object.__setattr__(
                self,
                field_name,
                values,
            )
        for field_name, expected_type in (
            ("material_renames", RenameIntent),
            ("section_renames", RenameIntent),
            ("material_deletes", DeleteIntent),
            ("section_deletes", DeleteIntent),
        ):
            values = deepcopy(tuple(getattr(self, field_name)))
            if any(type(value) is not expected_type for value in values):
                raise TypeError(
                    f"{field_name} must contain only {expected_type.__name__} values"
                )
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True)
class NamedRegionEditBatch:
    """One atomic post-state and identity ledger for named regions."""

    base_session_revision: int
    regions: tuple[NamedRegion, ...]
    renames: tuple[RenameIntent, ...] = ()
    deletes: tuple[DeleteIntent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_session_revision",
            _base_revision(self.base_session_revision),
        )
        regions = deepcopy(tuple(self.regions))
        if any(type(region) is not NamedRegion for region in regions):
            raise TypeError("regions must contain only NamedRegion values")
        object.__setattr__(self, "regions", regions)
        for field_name, expected_type in (
            ("renames", RenameIntent),
            ("deletes", DeleteIntent),
        ):
            values = deepcopy(tuple(getattr(self, field_name)))
            if any(type(value) is not expected_type for value in values):
                raise TypeError(
                    f"{field_name} must contain only {expected_type.__name__} values"
                )
            object.__setattr__(self, field_name, values)


def _command_name(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _base_revision(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("base_session_revision must be a non-negative integer")
    return int(value)


__all__ = [
    "DefinitionEditBatch",
    "DeleteIntent",
    "NamedRegionEditBatch",
    "RenameIntent",
    "UNSET",
    "Unset",
]
