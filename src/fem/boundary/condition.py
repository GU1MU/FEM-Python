from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import math


@dataclass(frozen=True)
class ElementLoad:
    """Constant element load vector."""
    elem_id: int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class ElementGravityLoad:
    """Gravity acceleration resolved to one element."""
    elem_id: int
    acceleration: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "elem_id", int(self.elem_id))
        object.__setattr__(
            self,
            "acceleration",
            tuple(self.acceleration),
        )


@dataclass(frozen=True)
class SurfaceTraction:
    """Constant element boundary traction."""
    elem_id: int
    local_index: int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class EdgeTraction:
    """Constant element edge traction."""
    elem_id: int
    local_index: int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class LineElementLoad:
    """Resolved constant Beam2 line load."""
    elem_id: int
    vector: tuple[float, ...]
    coordinate_system: str


@dataclass
class BoundaryCondition:
    """Store Dirichlet conditions and loads by global DOF and element id."""
    prescribed_displacements: Dict[int, float] = field(default_factory=dict)
    nodal_forces: Dict[int, float] = field(default_factory=dict)
    body_forces: list[ElementLoad] = field(default_factory=list)
    surface_tractions: list[SurfaceTraction] = field(default_factory=list)
    gravity: tuple[float, ...] | None = None
    edge_tractions: list[EdgeTraction] = field(default_factory=list)
    line_loads: list[LineElementLoad] = field(default_factory=list)
    element_gravities: list[ElementGravityLoad] = field(default_factory=list)

    def add_displacement_dof(self, dof_id: int, value: float = 0.0) -> None:
        """Add prescribed displacement on a global DOF."""
        self.prescribed_displacements[int(dof_id)] = _finite_float(
            value, "prescribed displacement"
        )

    def add_displacement(self, node_id: int, component: int, value: float, mesh: Any) -> None:
        """Add prescribed displacement by node and component."""
        self.add_displacement_dof(mesh.global_dof(node_id, component), value)

    def add_fixed_support(
        self,
        node_id: int,
        components: Iterable[int] | None,
        mesh: Any,
    ) -> None:
        """Fix selected components on a node."""
        if components is None:
            components = range(mesh.dofs_per_node)
        for component in components:
            self.add_displacement(node_id, component, 0.0, mesh)

    def add_nodal_force_dof(self, dof_id: int, value: float) -> None:
        """Add nodal force on a global DOF."""
        _accumulate_dof_value(self.nodal_forces, dof_id, value)

    def add_nodal_force(self, node_id: int, component: int, value: float, mesh: Any) -> None:
        """Add nodal force by node and component."""
        self.add_nodal_force_dof(mesh.global_dof(node_id, component), value)

    def add_body_force_element(self, elem_id: int, *components: float) -> None:
        """Add constant body force on an element."""
        self.body_forces.append(ElementLoad(int(elem_id), _float_vector(components)))

    def add_surface_traction(
        self,
        elem_id: int,
        local_index: int,
        *components: float,
    ) -> None:
        """Add constant traction on an element face."""
        self.surface_tractions.append(
            SurfaceTraction(int(elem_id), int(local_index), _float_vector(components))
        )

    def add_edge_traction(
        self,
        elem_id: int,
        local_index: int,
        *components: float,
    ) -> None:
        """Add constant traction on an element edge."""
        self.edge_tractions.append(
            EdgeTraction(int(elem_id), int(local_index), _float_vector(components))
        )

    def set_gravity(self, *components: float) -> None:
        """Set global gravity acceleration."""
        self.gravity = _float_vector(components)

    def add_gravity_element(self, elem_id: int, *components: float) -> None:
        """Add gravity acceleration resolved to one element."""
        self.element_gravities.append(
            ElementGravityLoad(int(elem_id), _float_vector(components))
        )

    def add_line_load(
        self,
        elem_id: int,
        vector: Iterable[float],
        coordinate_system: str,
    ) -> None:
        """Add a resolved constant Beam2 line load."""
        self.line_loads.append(
            LineElementLoad(
                int(elem_id),
                _float_vector(tuple(vector)),
                str(coordinate_system),
            )
        )


def _accumulate_dof_value(values: Dict[int, float], dof_id: int, value: float) -> None:
    """Accumulate a scalar value on a DOF map."""
    dof_id = int(dof_id)
    value = _finite_float(value, "nodal force")
    total = values.get(dof_id, 0.0) + value
    if not math.isfinite(total):
        raise ValueError(f"accumulated nodal force at DOF {dof_id} must be finite")
    values[dof_id] = total


def _float_vector(components: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize a non-empty numeric vector."""
    if not components:
        raise ValueError("load vector must contain at least one component")
    return tuple(_finite_float(value, "load vector component") for value in components)


def _finite_float(value: float, name: str) -> float:
    """Return a finite scalar value."""
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return converted
