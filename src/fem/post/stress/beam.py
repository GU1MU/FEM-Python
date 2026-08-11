from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass
import math
import operator
from typing import Any, ClassVar

import numpy as np

from ...boundary.step import boundary_for_step
from ...core.result import ModelResult
from ...elements import canonical_element_type, get_element_kernel
from ...elements.beam_section import (
    BeamSectionPoint,
    BeamIntegrationPointForces,
    recover_integration_point_s11 as recover_point_s11,
    recover_section_point_stress,
    parse_beam2_section,
)
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
    torque: float
    s11_max: float
    s11_min: float
    s11_abs_max: float
    s12_abs_max: float
    shear_y: float = 0.0
    shear_z: float = 0.0

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
            "torque",
            "s11_max",
            "s11_min",
            "s11_abs_max",
            "s12_abs_max",
            "shear_y",
            "shear_z",
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

    @property
    def N(self) -> float:
        return self.axial_force

    @property
    def Vy(self) -> float:
        return self.shear_y

    @property
    def Vz(self) -> float:
        return self.shear_z

    @property
    def My(self) -> float:
        return self.moment_y

    @property
    def Mz(self) -> float:
        return self.moment_z

    @property
    def T(self) -> float:
        return self.torque

    def values(self) -> dict[str, float]:
        """Return the legacy longitudinal section-stress components."""

        return {
            "S11Max": self.s11_max,
            "S11Min": self.s11_min,
            "S11AbsMax": self.s11_abs_max,
        }

    def section_values(self) -> dict[str, float]:
        """Return the complete canonical section result components."""

        return {
            **self.values(),
            "S12AbsMax": self.s12_abs_max,
        }


@dataclass(frozen=True, slots=True)
class BeamSectionPointEndStress:
    """One Beam2 end stress row at an explicit section point."""

    element_id: int
    local_node: int
    node_id: int
    coordinates: tuple[float, float, float]
    displacement: tuple[float, float, float]
    section_point: BeamSectionPoint
    s11: float
    s12: float
    mises: float
    max_principal: float
    mid_principal: float
    min_principal: float

    def __post_init__(self) -> None:
        for name in ("element_id", "local_node", "node_id"):
            object.__setattr__(self, name, _integer_id(name, getattr(self, name)))
        if self.local_node not in {1, 2}:
            raise ValueError("local_node must be 1 or 2 for Beam2")
        object.__setattr__(
            self, "coordinates", _finite_triplet("coordinates", self.coordinates)
        )
        object.__setattr__(
            self,
            "displacement",
            _finite_triplet("displacement", self.displacement),
        )
        if type(self.section_point) is not BeamSectionPoint:
            raise TypeError("section_point must be BeamSectionPoint")
        for name in (
            "s11",
            "s12",
            "mises",
            "max_principal",
            "mid_principal",
            "min_principal",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    def values(self) -> dict[str, float]:
        return {
            "S11": self.s11,
            "S12": self.s12,
            "Mises": self.mises,
            "MaxPrincipal": self.max_principal,
            "MidPrincipal": self.mid_principal,
            "MinPrincipal": self.min_principal,
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
class BeamSectionPointField:
    """Rows for one canonical Beam2 section-point number."""

    point_number: int
    rows: tuple[BeamSectionPointEndStress, ...]

    position: ClassVar[str] = "section_point"
    component_names: ClassVar[tuple[str, ...]] = (
        "S11",
        "S12",
        "Mises",
        "MaxPrincipal",
        "MidPrincipal",
        "MinPrincipal",
    )

    def __post_init__(self) -> None:
        point_number = _integer_id("point_number", self.point_number)
        if point_number <= 0:
            raise ValueError("point_number must be positive")
        rows = tuple(self.rows)
        if any(type(row) is not BeamSectionPointEndStress for row in rows):
            raise TypeError(
                "rows must contain only BeamSectionPointEndStress values"
            )
        if any(row.section_point.number != point_number for row in rows):
            raise ValueError("every row must match point_number")
        identities = [(row.element_id, row.local_node) for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("section-point row identities must be unique")
        object.__setattr__(self, "point_number", point_number)
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True, slots=True)
class BeamIntegrationPointSectionResult:
    """Constitutive section result at the sole B31 longitudinal point."""

    element_id: int
    integration_point: int
    coordinates: tuple[float, float, float]
    displacement: tuple[float, float, float]
    forces: BeamIntegrationPointForces

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "element_id",
            _integer_id("element_id", self.element_id),
        )
        object.__setattr__(
            self,
            "integration_point",
            _integer_id("integration_point", self.integration_point),
        )
        if self.integration_point != 1:
            raise ValueError("Beam2 B31 has exactly one longitudinal point")
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
        if type(self.forces) is not BeamIntegrationPointForces:
            raise TypeError("forces must be BeamIntegrationPointForces")

    @property
    def N(self) -> float:
        return self.forces.N

    @property
    def My(self) -> float:
        return self.forces.My

    @property
    def Mz(self) -> float:
        return self.forces.Mz

    def values(self) -> dict[str, float]:
        return {"N": self.N, "My": self.My, "Mz": self.Mz}


@dataclass(frozen=True, slots=True)
class BeamIntegrationPointSectionField:
    """One constitutive section-result row per Beam2 element."""

    rows: tuple[BeamIntegrationPointSectionResult, ...]

    position: ClassVar[str] = "integration_point"
    component_names: ClassVar[tuple[str, ...]] = ("N", "My", "Mz")

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if any(type(row) is not BeamIntegrationPointSectionResult for row in rows):
            raise TypeError(
                "rows must contain only BeamIntegrationPointSectionResult values"
            )
        identities = [
            (row.element_id, row.integration_point) for row in rows
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("integration-point row identities must be unique")
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True, slots=True)
class BeamIntegrationPointS11:
    """One section-point S11 row at a B31 longitudinal integration point."""

    element_id: int
    integration_point: int
    coordinates: tuple[float, float, float]
    displacement: tuple[float, float, float]
    section_point: BeamSectionPoint
    s11: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "element_id",
            _integer_id("element_id", self.element_id),
        )
        object.__setattr__(
            self,
            "integration_point",
            _integer_id("integration_point", self.integration_point),
        )
        if self.integration_point != 1:
            raise ValueError("Beam2 B31 has exactly one longitudinal point")
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
        if type(self.section_point) is not BeamSectionPoint:
            raise TypeError("section_point must be BeamSectionPoint")
        value = float(self.s11)
        if not math.isfinite(value):
            raise ValueError("s11 must be finite")
        object.__setattr__(self, "s11", value)

    def values(self) -> dict[str, float]:
        return {"S11": self.s11}


@dataclass(frozen=True, slots=True)
class BeamIntegrationPointS11Field:
    """S11 rows for one section point, one longitudinal row per Beam2."""

    point_number: int
    rows: tuple[BeamIntegrationPointS11, ...]

    position: ClassVar[str] = "integration_point"
    component_names: ClassVar[tuple[str, ...]] = ("S11",)

    def __post_init__(self) -> None:
        point_number = _integer_id("point_number", self.point_number)
        rows = tuple(self.rows)
        if any(type(row) is not BeamIntegrationPointS11 for row in rows):
            raise TypeError(
                "rows must contain only BeamIntegrationPointS11 values"
            )
        if any(row.section_point.number != point_number for row in rows):
            raise ValueError("every row must match point_number")
        identities = [
            (row.element_id, row.integration_point) for row in rows
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("integration-point row identities must be unique")
        object.__setattr__(self, "point_number", point_number)
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True, slots=True)
class BeamIntegrationPointS11Recovery:
    """All section-point S11 fields at the sole B31 longitudinal point."""

    section_forces: BeamIntegrationPointSectionField
    section_points: tuple[BeamIntegrationPointS11Field, ...]

    def __post_init__(self) -> None:
        if type(self.section_forces) is not BeamIntegrationPointSectionField:
            raise TypeError(
                "section_forces must be BeamIntegrationPointSectionField"
            )
        fields = tuple(self.section_points)
        if any(type(field) is not BeamIntegrationPointS11Field for field in fields):
            raise TypeError(
                "section_points must contain BeamIntegrationPointS11Field values"
            )
        numbers = tuple(field.point_number for field in fields)
        if len(numbers) != len(set(numbers)):
            raise ValueError("section point field numbers must be unique")
        object.__setattr__(self, "section_points", fields)

    def point_field(self, number: int) -> BeamIntegrationPointS11Field:
        for field in self.section_points:
            if field.point_number == number:
                return field
        raise KeyError(number)


@dataclass(frozen=True, slots=True)
class BeamSectionStressRecovery:
    """Shared Beam2 recovery behind point and section result fields."""

    section_end: BeamEndStressField
    section_points: tuple[BeamSectionPointField, ...]

    def __post_init__(self) -> None:
        if type(self.section_end) is not BeamEndStressField:
            raise TypeError("section_end must be BeamEndStressField")
        if type(self.section_points) is not tuple or any(
            type(value) is not BeamSectionPointField
            for value in self.section_points
        ):
            raise TypeError(
                "section_points must contain BeamSectionPointField values"
            )
        numbers = tuple(value.point_number for value in self.section_points)
        if len(numbers) != len(set(numbers)):
            raise ValueError("section point field numbers must be unique")

    def point_field(self, number: int) -> BeamSectionPointField:
        for field in self.section_points:
            if field.point_number == number:
                return field
        raise KeyError(number)


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


def recover_integration_point_s11(
    result: ModelResult,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> BeamIntegrationPointS11Recovery:
    """Recover one B31 longitudinal integration-point S11 per element."""

    _validate_checkpoint(checkpoint)
    if not isinstance(result, ModelResult):
        raise TypeError("result must be ModelResult")
    mesh = result.model.mesh
    if not mesh.elements:
        raise ValueError("Beam2 integration-point recovery requires elements")
    if _integer_id("dofs_per_node", mesh.dofs_per_node) != 6:
        raise ValueError(
            "Beam2 integration-point recovery requires exactly six DOFs per node"
        )
    lookup = _validated_node_lookup(mesh.nodes)
    force_rows: list[BeamIntegrationPointSectionResult] = []
    point_rows: dict[int, list[BeamIntegrationPointS11]] = {}
    for elem in mesh.elements:
        _run_checkpoint(checkpoint)
        element_id = _integer_id("element id", elem.id)
        try:
            element_type = canonical_element_type(elem.type)
        except NotImplementedError as error:
            raise ValueError(
                "Beam2 integration-point recovery requires only Beam2 elements, "
                f"got {elem.type!r}"
            ) from error
        if element_type != "Beam2":
            raise ValueError(
                "Beam2 integration-point recovery requires only Beam2 elements, "
                f"got {elem.type!r}"
            )
        node_ids = tuple(
            _integer_id(f"element {element_id} node id", node_id)
            for node_id in elem.node_ids
        )
        if len(node_ids) != 2:
            raise ValueError(
                f"Beam2 element {element_id} requires exactly two nodes"
            )
        if any(node_id not in lookup for node_id in node_ids):
            raise ValueError(
                f"Beam2 element {element_id} references missing mesh nodes"
            )
        kernel = get_element_kernel(elem.type)
        forces = kernel.local_integration_point_forces(
            mesh,
            elem,
            result.U,
            lookup,
        )
        section = parse_beam2_section(elem.props)
        nodes = tuple(lookup[node_id] for node_id in node_ids)
        coordinates = tuple(
            (float(getattr(nodes[0], name)) + float(getattr(nodes[1], name)))
            / 2.0
            for name in ("x", "y", "z")
        )
        endpoint_displacements = tuple(
            _node_displacement(mesh, result.U, node_id) for node_id in node_ids
        )
        displacement = tuple(
            (endpoint_displacements[0][index] + endpoint_displacements[1][index])
            / 2.0
            for index in range(3)
        )
        force_rows.append(
            BeamIntegrationPointSectionResult(
                element_id=element_id,
                integration_point=1,
                coordinates=coordinates,
                displacement=displacement,
                forces=forces,
            )
        )
        for point_stress in recover_point_s11(section, forces):
            point_rows.setdefault(point_stress.point.number, []).append(
                BeamIntegrationPointS11(
                    element_id=element_id,
                    integration_point=1,
                    coordinates=coordinates,
                    displacement=displacement,
                    section_point=point_stress.point,
                    s11=point_stress.s11,
                )
            )
    return BeamIntegrationPointS11Recovery(
        section_forces=BeamIntegrationPointSectionField(tuple(force_rows)),
        section_points=tuple(
            BeamIntegrationPointS11Field(number, tuple(point_rows[number]))
            for number in sorted(point_rows)
        ),
    )


def recover_section_stress(
    result: ModelResult,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> BeamSectionStressRecovery:
    """Recover point and section Beam2 stresses in one numerical pass."""

    _validate_checkpoint(checkpoint)
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
    point_rows: dict[int, list[BeamSectionPointEndStress]] = {}
    for elem in mesh.elements:
        _run_checkpoint(checkpoint)
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
        end_forces = kernel.local_section_end_forces(
            mesh,
            elem,
            result.U,
            local_load,
            lookup,
        )
        _run_checkpoint(checkpoint)
        if len(end_forces) != 2:
            raise ValueError(
                f"Beam2 element {element_id} section forces require two ends"
            )
        section = parse_beam2_section(elem.props)
        for local_node, (node_id, forces) in enumerate(zip(
            node_ids,
            end_forces,
            strict=True,
        ), start=1):
            recovered = recover_section_point_stress(section, forces)
            node = lookup[int(node_id)]
            coordinates = _node_coordinates(node)
            displacement = _node_displacement(mesh, result.U, node_id)
            rows.append(
                BeamSectionEndStress(
                    element_id=element_id,
                    local_node=local_node,
                    node_id=node_id,
                    coordinates=coordinates,
                    displacement=displacement,
                    axial_force=forces.axial_force,
                    moment_y=forces.moment_y,
                    moment_z=forces.moment_z,
                    torque=forces.torque,
                    s11_max=recovered.s11_max,
                    s11_min=recovered.s11_min,
                    s11_abs_max=recovered.s11_abs_max,
                    s12_abs_max=recovered.s12_abs_max,
                    shear_y=forces.shear_y,
                    shear_z=forces.shear_z,
                )
            )
            for point_stress in recovered.point_stresses:
                point_rows.setdefault(point_stress.point.number, []).append(
                    BeamSectionPointEndStress(
                        element_id=element_id,
                        local_node=local_node,
                        node_id=node_id,
                        coordinates=coordinates,
                        displacement=displacement,
                        section_point=point_stress.point,
                        s11=point_stress.s11,
                        s12=point_stress.s12,
                        mises=point_stress.mises,
                        max_principal=point_stress.max_principal,
                        mid_principal=point_stress.mid_principal,
                        min_principal=point_stress.min_principal,
                    )
                )
    section_end = BeamEndStressField(
        node_order=tuple(
            _integer_id("mesh node id", node_id)
            for node_id in mesh.node_ids
        ),
        rows=tuple(rows),
    )
    return BeamSectionStressRecovery(
        section_end=section_end,
        section_points=tuple(
            BeamSectionPointField(number, tuple(point_rows[number]))
            for number in sorted(point_rows)
        ),
    )


def recover_section_end_stress(
    result: ModelResult,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> BeamEndStressField:
    """Compatibility wrapper for the legacy section-extrema entry point."""

    return recover_section_stress(result, checkpoint=checkpoint).section_end


def section_node_envelope(
    field: BeamEndStressField,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> BeamNodeEnvelopeField:
    """Derive incident-node envelopes without inventing isolated-node rows."""

    _validate_checkpoint(checkpoint)
    if type(field) is not BeamEndStressField:
        raise TypeError("field must be BeamEndStressField")
    contributions: dict[int, list[BeamSectionEndStress]] = {}
    for row in field.rows:
        _run_checkpoint(checkpoint)
        contributions.setdefault(row.node_id, []).append(row)

    rows: list[BeamNodeEnvelopeStress] = []
    for node_id in field.node_order:
        _run_checkpoint(checkpoint)
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


def nodal_envelope(
    result: Any,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[Beam2NodalStress, ...]:
    """Compatibility view of Beam2 envelopes including isolated zero rows."""

    _validate_checkpoint(checkpoint)
    field = recover_section_end_stress(
        result,
        checkpoint=checkpoint,
    )
    canonical = {
        row.node_id: row
        for row in section_node_envelope(
            field,
            checkpoint=checkpoint,
        ).rows
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


def _validate_checkpoint(
    checkpoint: Callable[[], None] | None,
) -> None:
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")


def _run_checkpoint(checkpoint: Callable[[], None] | None) -> None:
    if checkpoint is not None:
        checkpoint()


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
