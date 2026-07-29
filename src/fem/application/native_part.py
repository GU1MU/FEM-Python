"""Stable native-Part ownership contracts.

The application used to keep ``NativePart`` as display-only metadata while
geometry and mesh settings lived on the project.  This module owns the v7
aggregate without importing Session or persistence code, so both layers can
depend on it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from fem.geometry.part_namespace import (
    normalize_part_boolean_feature_id,
    normalize_part_id,
    part_boolean_feature_id_sort_key,
    part_id_sort_key,
)
from fem.geometry.recipes import (
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    geometry_dimension,
    is_single_solid_recipe,
)
from fem.mesh.settings import MeshSettings


def next_part_id(
    active_ids: Iterable[str],
    retired_ids: Iterable[str] = (),
) -> str:
    """Allocate monotonically after every active or retired Part identity."""

    numbers = {
        part_id_sort_key(value)
        for value in (*tuple(active_ids), *tuple(retired_ids))
    }
    return f"P{max(numbers, default=0) + 1}"


def next_part_boolean_feature_id(
    active_ids: Iterable[str],
    retired_ids: Iterable[str] = (),
) -> str:
    """Allocate monotonically after every active or retired PBF identity."""

    numbers = {
        part_boolean_feature_id_sort_key(value)
        for value in (*tuple(active_ids), *tuple(retired_ids))
    }
    return f"PBF{max(numbers, default=0) + 1}"


def _required_name(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _uses_part_id_syntax(value: str) -> bool:
    try:
        normalize_part_id(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class PartBooleanProvenance:
    """Read-only dependency edge from one Boolean result to its source Parts."""

    feature_id: str
    target_part_id: str
    tool_part_id: str
    operation: str

    def __post_init__(self) -> None:
        feature_id = normalize_part_boolean_feature_id(self.feature_id)
        target_id = normalize_part_id(self.target_part_id, "target_part_id")
        tool_id = normalize_part_id(self.tool_part_id, "tool_part_id")
        if target_id == tool_id:
            raise ValueError("target_part_id and tool_part_id must differ")
        if self.operation not in {"fuse", "cut"}:
            raise ValueError("Part Boolean operation must be fuse or cut")
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "target_part_id", target_id)
        object.__setattr__(self, "tool_part_id", tool_id)

    @property
    def source_part_ids(self) -> tuple[str, str]:
        return self.target_part_id, self.tool_part_id


@dataclass(frozen=True, slots=True, init=False)
class NativePart:
    """One independently owned native geometry, history, and mesh policy.

    ``NativePart("Part-1", "Body-1")`` remains accepted for v1-v6 codecs.
    New code should use keyword arguments and the v7 fields.
    """

    id: str
    name: str
    geometry_recipe: Any | None
    mesh_settings: MeshSettings | None
    suppressed: bool
    provenance: PartBooleanProvenance | None
    _legacy_body_name: str

    def __init__(
        self,
        *args: object,
        id: str | None = None,
        name: str | None = None,
        geometry_recipe: Any | None = None,
        mesh_settings: MeshSettings | None = None,
        suppressed: bool = False,
        provenance: PartBooleanProvenance | None = None,
        body_name: str | None = None,
        _legacy_body_name: str | None = None,
    ) -> None:
        if args:
            if id is not None or name is not None:
                raise TypeError(
                    "NativePart positional values cannot be mixed with id/name"
                )
            first = args[0]
            if (
                type(first) is str
                and _uses_part_id_syntax(first)
                and len(args) >= 2
            ):
                if len(args) > 6:
                    raise TypeError("NativePart accepts at most six v7 values")
                values = (*args, None, None, False, None)
                id = str(values[0])
                name = str(values[1])
                geometry_recipe = values[2]
                mesh_settings = values[3]
                suppressed = bool(values[4])
                provenance = values[5]
            else:
                if len(args) > 2:
                    raise TypeError(
                        "legacy NativePart accepts only name and body_name"
                    )
                name = str(first)
                if len(args) == 2:
                    body_name = str(args[1])

        normalized_id = normalize_part_id("P1" if id is None else id)
        normalized_name = _required_name(
            "Part-1" if name is None else name,
            "NativePart.name",
        )
        legacy_name = (
            _legacy_body_name
            if _legacy_body_name is not None
            else body_name
        )
        legacy_name = _required_name(
            "Body-1" if legacy_name is None else legacy_name,
            "NativePart.body_name",
        )
        owned_recipe = deepcopy(geometry_recipe)
        if owned_recipe is not None:
            if not isinstance(owned_recipe, NATIVE_GEOMETRY_TYPES):
                raise TypeError(
                    "NativePart.geometry_recipe must be native geometry or None"
                )
            if isinstance(owned_recipe, MultiBodyGeometry):
                raise ValueError(
                    "NativePart.geometry_recipe cannot be MultiBodyGeometry"
                )
            if (
                geometry_dimension(owned_recipe) == 3
                and not is_single_solid_recipe(owned_recipe)
            ):
                raise ValueError(
                    "3D NativePart geometry must be an exact single solid"
                )
        owned_settings = deepcopy(mesh_settings)
        if owned_settings is not None and type(owned_settings) is not MeshSettings:
            raise TypeError(
                "NativePart.mesh_settings must be MeshSettings or None"
            )
        if type(suppressed) is not bool:
            raise TypeError("NativePart.suppressed must be a bool")
        if provenance is not None and type(provenance) is not PartBooleanProvenance:
            raise TypeError(
                "NativePart.provenance must be PartBooleanProvenance or None"
            )
        if provenance is not None and owned_recipe is None:
            raise ValueError("Boolean result Part requires geometry")

        object.__setattr__(self, "id", normalized_id)
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "geometry_recipe", owned_recipe)
        object.__setattr__(self, "mesh_settings", owned_settings)
        object.__setattr__(self, "suppressed", suppressed)
        object.__setattr__(self, "provenance", deepcopy(provenance))
        object.__setattr__(self, "_legacy_body_name", legacy_name)

    @property
    def body_name(self) -> str:
        """Frozen compatibility label used only by v1-v6 project codecs."""

        return self._legacy_body_name

    @property
    def feature_history(self) -> tuple[Any, ...]:
        """Derive shallow history from the Part-owned recipe."""

        if self.geometry_recipe is None:
            return ()
        from .feature_history import derive_feature_history

        return derive_feature_history(self.geometry_recipe)

    @property
    def dimension(self) -> int | None:
        return (
            None
            if self.geometry_recipe is None
            else geometry_dimension(self.geometry_recipe)
        )


def validate_native_parts(
    parts: Iterable[NativePart],
    *,
    allow_empty_geometry: bool = False,
) -> tuple[NativePart, ...]:
    """Own and validate one canonical Part collection."""

    owned = deepcopy(tuple(parts))
    if any(type(part) is not NativePart for part in owned):
        raise TypeError("parts must contain only NativePart values")
    ids = tuple(part.id for part in owned)
    names = tuple(part.name for part in owned)
    if len(ids) != len(set(ids)):
        raise ValueError("Part IDs must be unique")
    if len(names) != len(set(names)):
        raise ValueError("Part names must be unique")
    if not allow_empty_geometry and any(
        part.geometry_recipe is None for part in owned
    ):
        raise ValueError("canonical Parts must own geometry")
    canonical = tuple(sorted(owned, key=lambda part: part_id_sort_key(part.id)))
    if owned != canonical:
        raise ValueError("Parts must use canonical Part ID order")
    return canonical


__all__ = [
    "NativePart",
    "PartBooleanProvenance",
    "next_part_boolean_feature_id",
    "next_part_id",
    "normalize_part_boolean_feature_id",
    "normalize_part_id",
    "part_boolean_feature_id_sort_key",
    "part_id_sort_key",
    "validate_native_parts",
]
