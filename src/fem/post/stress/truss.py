"""Pure typed Truss2 stress recovery."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np

from ...elements import canonical_element_type, get_element_kernel


@dataclass(frozen=True, slots=True)
class TrussStressRow:
    """One Truss2 centroid sample in mesh-element order."""

    element_id: int
    coordinates: tuple[float, float, float]
    displacement: tuple[float, float, float]
    LE11: float
    S11: float
    Mises: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "element_id",
            _validated_id(self.element_id, label="element_id"),
        )
        object.__setattr__(
            self,
            "coordinates",
            _owned_vector3(self.coordinates, label="coordinates"),
        )
        object.__setattr__(
            self,
            "displacement",
            _owned_vector3(self.displacement, label="displacement"),
        )
        object.__setattr__(self, "LE11", _finite_scalar(self.LE11, label="LE11"))
        object.__setattr__(self, "S11", _finite_scalar(self.S11, label="S11"))
        object.__setattr__(
            self,
            "Mises",
            _finite_scalar(self.Mises, label="Mises"),
        )
        if self.Mises < 0.0:
            raise ValueError("Mises must be non-negative")

    @property
    def le11(self) -> float:
        """Return the canonical axial-strain component."""

        return self.LE11

    @property
    def s11(self) -> float:
        """Return the canonical axial-stress component."""

        return self.S11

    @property
    def mises(self) -> float:
        """Return the current uniaxial equivalent stress."""

        return self.Mises

    def values(self) -> dict[str, float]:
        """Return a detached component mapping for compatibility adapters."""

        return {
            "LE11": self.LE11,
            "S11": self.S11,
            "Mises": self.Mises,
        }


@dataclass(frozen=True, slots=True)
class TrussStressField:
    """Immutable Truss2 centroid rows in exact mesh-element order."""

    rows: tuple[TrussStressRow, ...]

    position: ClassVar[str] = "centroid"
    component_names: ClassVar[tuple[str, ...]] = ("LE11", "S11", "Mises")

    def __post_init__(self) -> None:
        try:
            rows = tuple(self.rows)
        except TypeError as error:
            raise TypeError("rows must be an iterable of TrussStressRow") from error
        if any(type(row) is not TrussStressRow for row in rows):
            raise TypeError("rows must contain only TrussStressRow values")
        element_ids = [row.element_id for row in rows]
        if len(set(element_ids)) != len(element_ids):
            raise ValueError("TrussStressField element_id values must be unique")
        object.__setattr__(self, "rows", rows)


def recover(
    mesh: Any,
    U: Sequence[float],
    *,
    checkpoint: Callable[[], None] | None = None,
) -> TrussStressField:
    """Recover LE11, S11, and Mises for a homogeneous spatial Truss2 mesh."""

    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    elements, nodes = _validated_truss_mesh(mesh)
    displacement = _owned_displacement(mesh, U)
    kernel = get_element_kernel("Truss2")
    rows: list[TrussStressRow] = []

    for element in elements:
        if checkpoint is not None:
            checkpoint()
        try:
            raw_values = tuple(
                kernel.element_stress(
                    mesh,
                    element,
                    displacement,
                    nodes,
                )
            )
        except TypeError as error:
            raise ValueError(
                f"Truss2 kernel returned invalid stress values for element "
                f"{element.id!r}"
            ) from error
        if checkpoint is not None:
            checkpoint()
        if len(raw_values) != 3:
            raise ValueError(
                f"Truss2 kernel returned {len(raw_values)} stress values for "
                f"element {element.id!r}; expected LE11, S11, and Mises"
            )

        node_i_id, node_j_id = (int(value) for value in element.node_ids)
        node_i = nodes[node_i_id]
        node_j = nodes[node_j_id]
        coordinates_i = _node_coordinates(node_i)
        coordinates_j = _node_coordinates(node_j)
        dofs_i = tuple(mesh.node_dofs(node_i_id))
        dofs_j = tuple(mesh.node_dofs(node_j_id))
        if len(dofs_i) != 3 or len(dofs_j) != 3:
            raise ValueError(
                f"Truss2 element {element.id!r} requires three translational "
                "DOFs at each node"
            )
        displacement_i = displacement[np.asarray(dofs_i, dtype=int)]
        displacement_j = displacement[np.asarray(dofs_j, dtype=int)]

        rows.append(
            TrussStressRow(
                element_id=element.id,
                coordinates=tuple(
                    0.5 * (first + second)
                    for first, second in zip(
                        coordinates_i,
                        coordinates_j,
                        strict=True,
                    )
                ),
                displacement=tuple(
                    float(value)
                    for value in 0.5 * (displacement_i + displacement_j)
                ),
                LE11=raw_values[0],
                S11=raw_values[1],
                Mises=raw_values[2],
            )
        )

    return TrussStressField(tuple(rows))


def _validated_truss_mesh(
    mesh: Any,
) -> tuple[tuple[Any, ...], dict[int, Any]]:
    try:
        raw_dofs_per_node = mesh.dofs_per_node
        raw_nodes = mesh.nodes
        raw_elements = mesh.elements
        mesh.node_dofs
        mesh.num_dofs
    except AttributeError as error:
        raise TypeError(
            "mesh must provide nodes, elements, dofs_per_node, num_dofs, "
            "and node_dofs()"
        ) from error

    if (
        isinstance(raw_dofs_per_node, bool)
        or not isinstance(raw_dofs_per_node, Integral)
        or int(raw_dofs_per_node) != 3
    ):
        raise ValueError(
            "Truss2 stress recovery requires exactly three translational "
            "DOFs per node"
        )

    nodes: dict[int, Any] = {}
    for node in tuple(raw_nodes):
        try:
            node_id = _validated_id(node.id, label="node id")
        except AttributeError as error:
            raise TypeError("mesh nodes must expose an integer id") from error
        if node_id in nodes:
            raise ValueError(f"mesh node id {node_id} is duplicated")
        _node_coordinates(node)
        nodes[node_id] = node
    if not nodes:
        raise ValueError("Truss2 stress recovery requires at least one mesh node")

    elements = tuple(raw_elements)
    if not elements:
        raise ValueError("Truss2 stress recovery requires at least one element")
    seen_element_ids: set[int] = set()
    for element in elements:
        try:
            element_id = _validated_id(element.id, label="element id")
            element_type = element.type
            node_ids = tuple(element.node_ids)
        except AttributeError as error:
            raise TypeError(
                "mesh elements must expose id, type, and node_ids"
            ) from error
        if element_id in seen_element_ids:
            raise ValueError(f"mesh element id {element_id} is duplicated")
        seen_element_ids.add(element_id)
        try:
            canonical_type = canonical_element_type(element_type)
        except NotImplementedError as error:
            raise ValueError(
                "Truss2 stress recovery supports only homogeneous Truss2 "
                f"meshes; element {element_id} has unsupported type "
                f"{element_type!r}"
            ) from error
        if canonical_type != "Truss2":
            raise ValueError(
                "Truss2 stress recovery supports only homogeneous Truss2 "
                f"meshes; element {element_id} has type {element_type!r}"
            )
        if len(node_ids) != 2:
            raise ValueError(
                f"Truss2 element {element_id} requires exactly two nodes, "
                f"got {len(node_ids)}"
            )
        validated_node_ids = tuple(
            _validated_id(value, label=f"element {element_id} node id")
            for value in node_ids
        )
        missing = [node_id for node_id in validated_node_ids if node_id not in nodes]
        if missing:
            raise ValueError(
                f"Truss2 element {element_id} references missing mesh nodes "
                f"{missing}"
            )

    return elements, nodes


def _owned_displacement(mesh: Any, U: Sequence[float]) -> np.ndarray:
    try:
        raw = np.asarray(U)
    except (TypeError, ValueError) as error:
        raise TypeError("U must be a one-dimensional numeric sequence") from error
    if raw.dtype.kind == "b":
        raise TypeError("U must contain numeric displacement values, not booleans")
    try:
        values = np.array(U, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError("U must be a one-dimensional numeric sequence") from error
    if values.ndim != 1:
        raise ValueError(f"U must be one-dimensional, got shape {values.shape}")
    if values.shape != (int(mesh.num_dofs),):
        raise ValueError(
            f"U length {values.shape[0]} != mesh.num_dofs={mesh.num_dofs}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("U must contain only finite displacement values")
    return values


def _node_coordinates(node: Any) -> tuple[float, float, float]:
    try:
        coordinates = (node.x, node.y, node.z)
    except AttributeError as error:
        raise ValueError(
            f"Truss2 node {getattr(node, 'id', None)!r} must have x, y, z "
            "coordinates"
        ) from error
    return _owned_vector3(
        coordinates,
        label=f"node {getattr(node, 'id', None)!r} coordinates",
    )


def _validated_id(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _owned_vector3(
    values: Sequence[float],
    *,
    label: str,
) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must contain exactly three finite numbers")
    try:
        owned = tuple(
            _finite_scalar(value, label=f"{label}[{index}]")
            for index, value in enumerate(values)
        )
    except TypeError as error:
        raise TypeError(f"{label} must contain exactly three finite numbers") from error
    if len(owned) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return owned


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


__all__ = [
    "TrussStressField",
    "TrussStressRow",
    "recover",
]
