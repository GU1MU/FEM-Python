"""视口与模型树共享的选择状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SelectionMode = Literal["node", "element"]
SelectionSpace = Literal["geometry", "mesh"]
SelectionFilter = Literal["point", "element", "edge", "face", "body"]

_SELECTION_SPACES = frozenset({"geometry", "mesh"})
_SELECTION_FILTERS = frozenset({"point", "element", "edge", "face", "body"})


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
    """保存单一节点或单元选择。"""

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
