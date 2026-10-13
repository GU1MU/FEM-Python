"""Selection state shared by the viewport and model tree."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from fem.application import MeshEntityRef
from fem.application.definitions import mesh_entity_ref_sort_key
from fem.geometry import LogicalEntityRef, logical_ref_sort_key

SelectionMode = Literal["node", "element"]
SelectionSpace = Literal["geometry", "mesh"]
SelectionFilter = Literal["point", "element", "edge", "face", "body"]

_SELECTION_SPACES = frozenset({"geometry", "mesh"})
_SELECTION_FILTERS = frozenset({"point", "element", "edge", "face", "body"})


def _apply_selection_groups[Reference](
    selected: set[Reference],
    groups: Iterable[Iterable[Reference]],
    *,
    additive: bool,
    sort_key: Callable[[Reference], tuple],
) -> set[Reference]:
    previous = set(selected)
    unique_groups = {frozenset(group) for group in groups} - {frozenset()}
    if not additive:
        selected.clear()
        selected.update(reference for group in unique_groups for reference in group)
        return previous.symmetric_difference(selected)
    for group in sorted(
        unique_groups, key=lambda values: min(sort_key(value) for value in values),
    ):
        if additive and group.issubset(selected):
            selected.difference_update(group)
        else:
            selected.update(group)
    return previous.symmetric_difference(selected)


@dataclass(slots=True)
class SelectionReferences:
    """Own typed geometry and mesh selections with whole-group Ctrl toggles."""

    geometry: set[LogicalEntityRef] = field(default_factory=set)
    mesh: set[MeshEntityRef] = field(default_factory=set)

    def clear(self) -> None:
        self.geometry.clear()
        self.mesh.clear()

    def apply_geometry_groups(
        self, groups: Iterable[Iterable[LogicalEntityRef]], *, additive: bool,
    ) -> set[LogicalEntityRef]:
        return _apply_selection_groups(
            self.geometry, groups, additive=additive, sort_key=logical_ref_sort_key,
        )

    def apply_mesh_groups(
        self, groups: Iterable[Iterable[MeshEntityRef]], *, additive: bool,
    ) -> set[MeshEntityRef]:
        return _apply_selection_groups(
            self.mesh, groups, additive=additive, sort_key=mesh_entity_ref_sort_key,
        )


@dataclass(slots=True)
class SelectionContextState:
    """Remember the active semantic filter independently for each space."""

    space: SelectionSpace = "mesh"
    geometry_filter: SelectionFilter = "body"
    mesh_filter: SelectionFilter = "point"

    @property
    def active_filter(self) -> SelectionFilter:
        return (
            self.geometry_filter
            if self.space == "geometry"
            else self.mesh_filter
        )

    def set_space(self, space: SelectionSpace) -> SelectionFilter:
        if space not in _SELECTION_SPACES:
            raise ValueError("selection space must be geometry or mesh")
        self.space = space
        return self.active_filter

    def set_filter(self, selection_filter: SelectionFilter) -> None:
        if selection_filter not in _SELECTION_FILTERS:
            raise ValueError("unsupported semantic selection filter")
        if self.space == "geometry":
            if selection_filter == "element":
                raise ValueError("geometry selection does not support elements")
            self.geometry_filter = selection_filter
        else:
            self.mesh_filter = selection_filter


@dataclass(slots=True)
class SelectionState:
    """Store a single node or element selection."""

    mode: SelectionMode = "node"
    node_id: int | None = None
    element_id: int | None = None

    def clear(self) -> None:
        self.node_id = None
        self.element_id = None

    def select_node(self, node_id: int) -> None:
        self.mode = "node"
        self.node_id = int(node_id)
        self.element_id = None

    def select_element(self, element_id: int) -> None:
        self.mode = "element"
        self.element_id = int(element_id)
        self.node_id = None
