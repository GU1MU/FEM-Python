from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence, runtime_checkable

from .dof import DofMap


@dataclass
class Node2D:
    """2D node with id and coordinates."""
    id: int
    x: float
    y: float


@dataclass
class Element2D:
    """2D element with node list, type, and properties."""
    id: int
    node_ids: List[int]
    type: str
    props: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MeshProtocol(Protocol):
    """Shared protocol for FEM mesh containers."""
    dofs_per_node: int
    num_dofs: int
    num_nodes: int
    num_elements: int
    node_ids: Sequence[int]
    nodes: List
    elements: List
    dof_map: DofMap

    def global_dof(self, node_id: int, component: int) -> int: ...
    def node_dofs(self, node_id: int) -> Sequence[int]: ...
    def element_dofs(self, elem) -> Sequence[int]: ...
    def rebuild_dof_map(self) -> DofMap: ...
    def generate_global_dof_sequence(self) -> Sequence[tuple[int, int, int]]: ...


class _DofMappedMeshMixin:
    """Shared DOF access for mesh containers."""

    def __post_init__(self):
        self.rebuild_dof_map()

    def rebuild_dof_map(self) -> DofMap:
        """Rebuild cached node-to-DOF state after an explicit mesh edit."""
        self.dof_map = DofMap.from_nodes(self.nodes, self.dofs_per_node)
        return self.dof_map

    @property
    def node_ids(self):
        """Node ids in global DOF order."""
        return self.dof_map.node_ids

    @property
    def num_dofs(self) -> int:
        """Total number of DOFs."""
        return self.dof_map.num_dofs

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return self.dof_map.num_nodes

    @property
    def num_elements(self) -> int:
        """Number of elements."""
        return len(self.elements)

    def global_dof(self, node_id: int, component: int) -> int:
        """Return global DOF index for a node component."""
        return self.dof_map.global_dof(node_id, component)

    def node_dofs(self, node_id: int):
        """Return global DOF indices for a node."""
        return self.dof_map.node_dofs(node_id)

    def element_dofs(self, elem):
        """Return global DOF indices for an element."""
        return self.dof_map.element_dofs(elem.node_ids)

    def generate_global_dof_sequence(self):
        """Generate (node_id, component, dof_id) tuples."""
        return self.dof_map.generate_global_dof_sequence()


@dataclass
class Mesh2D(_DofMappedMeshMixin):
    """Topology-agnostic 2D mesh container."""
    nodes: List[Node2D]
    elements: List[Element2D]
    dofs_per_node: int = 2
    dof_map: DofMap = field(init=False)


@dataclass
class Node3D:
    """3D node with id and coordinates."""
    id: int
    x: float
    y: float
    z: float


@dataclass
class Element3D:
    """3D element with node list, type, and properties."""
    id: int
    node_ids: List[int]
    type: str
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Mesh3D(_DofMappedMeshMixin):
    """Topology-agnostic 3D mesh container."""
    nodes: List[Node3D]
    elements: List[Element3D]
    dofs_per_node: int = 3
    dof_map: DofMap = field(init=False)
