from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import operator
from typing import Any, ClassVar

import numpy as np

from ...boundary.step import boundary_for_step
from ...core.result import ModelResult
from ...elements import canonical_element_type, get_element_kernel
from ...elements.beam_section import axial_stress_extrema, parse_beam2_section
from .._paths import prepare_output_path


BEAM2_NODAL_STRESS_HEADER = (
    "node_id",
    "x",
    "y",
    "z",
    "axial_stress_max",
    "axial_stress_min",
    "axial_stress_abs_max",
)


@dataclass(frozen=True)
class Beam2NodalStress:
    """Axial-stress extrema enveloped at one mesh node."""

    node_id: int
    maximum: float
    minimum: float
    absolute_maximum: float


@dataclass(frozen=True, slots=True)
class BeamSectionEndStress:
    """Section stress and source actions at one Beam2 connectivity end."""

    element_id: int
    local_node: int
    node_id: int
    coordinates: tuple[float, float, float]
    displacement: tuple[float, float, float]
    axial_force: float
    moment_y: float
    moment_z: float
    s11_max: float
    s11_min: float
    s11_abs_max: float

    def __post_init__(self) -> None:
        for name in ("element_id", "local_node", "node_id"):
            object.__setattr__(
                self,
                name,
                _integer_id(name, getattr(self, name)),
            )
        if self.local_node not in {1, 2}:
            raise ValueError("local_node must be 1 or 2 for Beam2")
        object.__setattr__(
            self,
            "coordinates",
            _finite_triplet("coordinates", self.coordinates),
        )
        object.__setattr__(
            self,
            "displacement",
            _finite_triplet("displacement", self.displacement),
        )
        for name in (
            "axial_force",
            "moment_y",
            "moment_z",
            "s11_max",
            "s11_min",
            "s11_abs_max",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.s11_max < self.s11_min:
            raise ValueError("s11_max must be greater than or equal to s11_min")
        expected_absolute = max(abs(self.s11_max), abs(self.s11_min))
        if not math.isclose(
            self.s11_abs_max,
            expected_absolute,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "s11_abs_max must equal max(abs(s11_max), abs(s11_min))"
            )

    def values(self) -> dict[str, float]:
        """Return canonical section-stress components."""

        return {
            "S11Max": self.s11_max,
            "S11Min": self.s11_min,
            "S11AbsMax": self.s11_abs_max,
        }


@dataclass(frozen=True, slots=True)
class BeamEndStressField:
    """Beam2 section-end rows in mesh element and local-end order."""

    node_order: tuple[int, ...]
    rows: tuple[BeamSectionEndStress, ...]

    position: ClassVar[str] = "section_end"
    component_names: ClassVar[tuple[str, ...]] = (
        "S11Max",
        "S11Min",
        "S11AbsMax",
    )

    def __post_init__(self) -> None:
        node_order = _unique_integer_ids("node_order", self.node_order)
        rows = tuple(self.rows)
        if any(type(row) is not BeamSectionEndStress for row in rows):
            raise TypeError("rows must contain only BeamSectionEndStress values")
        node_ids = set(node_order)
        if any(row.node_id not in node_ids for row in rows):
            raise ValueError("every Beam section-end node must occur in node_order")
        identities = [
            (row.element_id, row.local_node)
            for row in rows
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("Beam section-end identities must be unique")
        object.__setattr__(self, "node_order", node_order)
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True, slots=True)
class BeamNodeEnvelopeStress:
    """Canonical Beam2 section-stress envelope at an incident mesh node."""

    node_id: int
    coordinates: tuple[float, float, float]
    displacement: tuple[float, float, float]
    s11_max: float
    s11_min: float
    s11_abs_max: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_id",
            _integer_id("node_id", self.node_id),
        )
        object.__setattr__(
            self,
            "coordinates",
            _finite_triplet("coordinates", self.coordinates),
        )
        object.__setattr__(
            self,
            "displacement",
            _finite_triplet("displacement", self.displacement),
        )
        for name in ("s11_max", "s11_min", "s11_abs_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.s11_max < self.s11_min:
            raise ValueError("s11_max must be greater than or equal to s11_min")
        expected_absolute = max(abs(self.s11_max), abs(self.s11_min))
        if not math.isclose(
            self.s11_abs_max,
            expected_absolute,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "s11_abs_max must equal max(abs(s11_max), abs(s11_min))"
            )

    def values(self) -> dict[str, float]:
        """Return canonical section-envelope components."""

        return {
            "S11Max": self.s11_max,
            "S11Min": self.s11_min,
            "S11AbsMax": self.s11_abs_max,
        }


@dataclass(frozen=True, slots=True)
class BeamNodeEnvelopeField:
    """Incident-node Beam2 envelopes in mesh node order."""

    rows: tuple[BeamNodeEnvelopeStress, ...]

    position: ClassVar[str] = "section_node_envelope"
    component_names: ClassVar[tuple[str, ...]] = (
        "S11Max",
        "S11Min",
        "S11AbsMax",
    )

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if any(type(row) is not BeamNodeEnvelopeStress for row in rows):
            raise TypeError("rows must contain only BeamNodeEnvelopeStress values")
        if len({row.node_id for row in rows}) != len(rows):
            raise ValueError("Beam node envelope rows must have unique node IDs")
        object.__setattr__(self, "rows", rows)


def recover_section_end_stress(result: ModelResult) -> BeamEndStressField:
    """Recover canonical Beam2 section-end rows from one complete result."""

    if not isinstance(result, ModelResult):
        raise TypeError("result must be ModelResult")
    mesh = result.model.mesh
    if not mesh.elements:
        raise ValueError("Beam2 section-end recovery requires at least one element")
    if _integer_id("dofs_per_node", mesh.dofs_per_node) != 6:
        raise ValueError(
            "Beam2 section-end recovery requires exactly six DOFs per node"
        )
    lookup = _validated_node_lookup(mesh.nodes)
    boundary = boundary_for_step(result.model, result.step)
    line_loads_by_element: dict[int, list[Any]] = {}
    for load in boundary.line_loads:
        load_element_id = _integer_id("line load element id", load.elem_id)
        line_loads_by_element.setdefault(load_element_id, []).append(load)

    rows: list[BeamSectionEndStress] = []
    for elem in mesh.elements:
        element_id = _integer_id("element id", elem.id)
        try:
            element_type = canonical_element_type(elem.type)
        except NotImplementedError as error:
            raise ValueError(
                "Beam2 section-end recovery requires only Beam2 elements, "
                f"got {elem.type!r}"
            ) from error
        if element_type != "Beam2":
            raise ValueError(
                "Beam2 section-end recovery requires only Beam2 elements, "
                f"got {elem.type!r}"
            )
        try:
            raw_node_ids = tuple(elem.node_ids)
        except TypeError as error:
            raise TypeError(
                f"Beam2 element {element_id} node_ids must be iterable"
            ) from error
        if len(raw_node_ids) != 2:
            raise ValueError(
                f"Beam2 element {element_id} requires exactly two nodes, "
                f"got {len(raw_node_ids)}"
            )
        node_ids = tuple(
            _integer_id(f"element {element_id} node id", node_id)
            for node_id in raw_node_ids
        )
        missing_node_ids = [
            node_id for node_id in node_ids if node_id not in lookup
        ]
        if missing_node_ids:
            raise ValueError(
                f"Beam2 element {element_id} references missing mesh nodes "
                f"{missing_node_ids}"
            )
        kernel = get_element_kernel(elem.type)
        local_load = np.zeros(12, dtype=float)
        for load in line_loads_by_element.get(element_id, ()):
            local_load += kernel.local_line_load(
                mesh,
                elem,
                load.vector,
                load.coordinate_system,
                lookup,
            )
        end_actions = np.asarray(
            kernel.local_end_actions(
                mesh,
                elem,
                result.U,
                local_load,
                lookup,
            ),
            dtype=float,
        )
        if end_actions.shape != (2, 3) or not np.all(np.isfinite(end_actions)):
            raise ValueError(
                f"Beam2 element {element_id} end actions must have shape "
                "(2, 3) with finite values"
            )
        section = parse_beam2_section(elem.props)
        for local_node, (
            node_id,
            (axial_force, moment_y, moment_z),
        ) in enumerate(zip(
            node_ids,
            end_actions,
            strict=True,
        ), start=1):
            maximum, minimum, absolute = axial_stress_extrema(
                section,
                axial_force,
                moment_y,
                moment_z,
            )
            node = lookup[int(node_id)]
            rows.append(
                BeamSectionEndStress(
                    element_id=element_id,
                    local_node=local_node,
                    node_id=node_id,
                    coordinates=_node_coordinates(node),
                    displacement=_node_displacement(mesh, result.U, node_id),
                    axial_force=float(axial_force),
                    moment_y=float(moment_y),
                    moment_z=float(moment_z),
                    s11_max=float(maximum),
                    s11_min=float(minimum),
                    s11_abs_max=float(absolute),
                )
            )
    return BeamEndStressField(
        node_order=tuple(
            _integer_id("mesh node id", node_id)
            for node_id in mesh.node_ids
        ),
        rows=tuple(rows),
    )


def section_node_envelope(
    field: BeamEndStressField,
) -> BeamNodeEnvelopeField:
    """Derive incident-node envelopes without inventing isolated-node rows."""

    if type(field) is not BeamEndStressField:
        raise TypeError("field must be BeamEndStressField")
    contributions: dict[int, list[BeamSectionEndStress]] = {}
    for row in field.rows:
        contributions.setdefault(row.node_id, []).append(row)

    rows: list[BeamNodeEnvelopeStress] = []
    for node_id in field.node_order:
        node_contributions = contributions.get(node_id, ())
        if not node_contributions:
            continue
        first = node_contributions[0]
        if any(
            row.coordinates != first.coordinates
            or row.displacement != first.displacement
            for row in node_contributions[1:]
        ):
            raise ValueError(
                f"Beam section-end rows disagree at shared node {node_id}"
            )
        maximum = max(row.s11_max for row in node_contributions)
        minimum = min(row.s11_min for row in node_contributions)
        rows.append(
            BeamNodeEnvelopeStress(
                node_id=node_id,
                coordinates=first.coordinates,
                displacement=first.displacement,
                s11_max=maximum,
                s11_min=minimum,
                s11_abs_max=max(abs(maximum), abs(minimum)),
            )
        )
    return BeamNodeEnvelopeField(tuple(rows))


def nodal_envelope(result: Any) -> tuple[Beam2NodalStress, ...]:
    """Compatibility view of Beam2 envelopes including isolated zero rows."""

    field = recover_section_end_stress(result)
    canonical = {
        row.node_id: row for row in section_node_envelope(field).rows
    }
    return tuple(
        Beam2NodalStress(
            node_id=node_id,
            maximum=(
                canonical[node_id].s11_max
                if node_id in canonical
                else 0.0
            ),
            minimum=(
                canonical[node_id].s11_min
                if node_id in canonical
                else 0.0
            ),
            absolute_maximum=(
                canonical[node_id].s11_abs_max
                if node_id in canonical
                else 0.0
            ),
        )
        for node_id in field.node_order
    )


def absolute_maximum(result: Any) -> float:
    """Return the maximum absolute stress from the Beam2 nodal envelope."""
    return max(
        (row.absolute_maximum for row in nodal_envelope(result)),
        default=0.0,
    )


def export_nodal(result: Any, path: str) -> None:
    """Write the Beam2 nodal axial-stress envelope CSV."""
    mesh = result.model.mesh
    lookup = {int(node.id): node for node in mesh.nodes}
    output_path = prepare_output_path(path)
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(BEAM2_NODAL_STRESS_HEADER)
        for row in nodal_envelope(result):
            node = lookup[row.node_id]
            writer.writerow(
                [
                    row.node_id,
                    node.x,
                    node.y,
                    node.z,
                    row.maximum,
                    row.minimum,
                    row.absolute_maximum,
                ]
            )


def _integer_id(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _unique_integer_ids(name: str, values: Any) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers") from error
    result: list[int] = []
    for value in raw_values:
        try:
            integer = _integer_id(name, value)
        except TypeError as error:
            raise TypeError(f"{name} must contain only integers") from error
        result.append(integer)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} values must be unique")
    return tuple(result)


def _validated_node_lookup(nodes: Any) -> dict[int, Any]:
    lookup: dict[int, Any] = {}
    for node in nodes:
        try:
            node_id = _integer_id("node id", node.id)
        except AttributeError as error:
            raise TypeError("mesh nodes must expose an integer id") from error
        if node_id in lookup:
            raise ValueError(f"mesh node id {node_id} is duplicated")
        _node_coordinates(node)
        lookup[node_id] = node
    return lookup


def _finite_triplet(name: str, values: Any) -> tuple[float, float, float]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain exactly three real values") from error
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _node_coordinates(node: Any) -> tuple[float, float, float]:
    return _finite_triplet(
        "node coordinates",
        (
            getattr(node, "x"),
            getattr(node, "y"),
            getattr(node, "z", 0.0),
        ),
    )


def _node_displacement(
    mesh: Any,
    displacement: np.ndarray,
    node_id: int,
) -> tuple[float, float, float]:
    return _finite_triplet(
        "node displacement",
        tuple(
            displacement[mesh.global_dof(node_id, component)]
            for component in range(3)
        ),
    )
