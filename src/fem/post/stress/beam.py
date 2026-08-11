from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass
import math
import operator
from typing import Any, ClassVar

import numpy as np

from ...core.result import ModelResult
from ...elements import canonical_element_type, get_element_kernel
from ...elements.beam_section import (
    BeamSectionPoint,
    BeamIntegrationPointForces,
    recover_integration_point_stress as recover_point_stress,
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
class BeamIntegrationPointSectionResult:
    """Constitutive resultants at the sole B31 longitudinal point.

    The public project names map to Abaqus B31 resultants as
    ``SF1/SF2/SF3 = N/Vy/Vz`` and ``SM1/SM2/SM3 = T/My/Mz``.
    """

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
    def Vy(self) -> float:
        return self.forces.Vy

    @property
    def Vz(self) -> float:
        return self.forces.Vz

    @property
    def T(self) -> float:
        return self.forces.T

    @property
    def My(self) -> float:
        return self.forces.My

    @property
    def Mz(self) -> float:
        return self.forces.Mz

    def values(self) -> dict[str, float]:
        return {
            "N": self.N,
            "Vy": self.Vy,
            "Vz": self.Vz,
            "T": self.T,
            "My": self.My,
            "Mz": self.Mz,
        }


@dataclass(frozen=True, slots=True)
class BeamIntegrationPointSectionField:
    """One constitutive section-result row per Beam2 element."""

    rows: tuple[BeamIntegrationPointSectionResult, ...]

    position: ClassVar[str] = "integration_point"
    component_names: ClassVar[tuple[str, ...]] = (
        "N",
        "Vy",
        "Vz",
        "T",
        "My",
        "Mz",
    )

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
class BeamIntegrationPointStress:
    """One section-point stress row at a B31 longitudinal integration point."""

    element_id: int
    integration_point: int
    coordinates: tuple[float, float, float]
    displacement: tuple[float, float, float]
    section_point: BeamSectionPoint
    s11: float
    s22: float
    s12: float
    mises: float
    max_principal: float
    mid_principal: float
    min_principal: float

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
        for name in (
            "s11",
            "s22",
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
            "S22": self.s22,
            "S12": self.s12,
            "Mises": self.mises,
            "MaxPrincipal": self.max_principal,
            "MidPrincipal": self.mid_principal,
            "MinPrincipal": self.min_principal,
        }


@dataclass(frozen=True, slots=True)
class BeamIntegrationPointStressField:
    """Stress rows for one section point, one longitudinal row per Beam2."""

    point_number: int
    rows: tuple[BeamIntegrationPointStress, ...]

    position: ClassVar[str] = "integration_point"
    component_names: ClassVar[tuple[str, ...]] = (
        "S11",
        "S22",
        "S12",
        "Mises",
        "MaxPrincipal",
        "MidPrincipal",
        "MinPrincipal",
    )

    def __post_init__(self) -> None:
        point_number = _integer_id("point_number", self.point_number)
        rows = tuple(self.rows)
        if any(type(row) is not BeamIntegrationPointStress for row in rows):
            raise TypeError(
                "rows must contain only BeamIntegrationPointStress values"
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
class BeamIntegrationPointStressRecovery:
    """All point-stress fields at the sole B31 longitudinal point."""

    section_forces: BeamIntegrationPointSectionField
    section_points: tuple[BeamIntegrationPointStressField, ...]

    def __post_init__(self) -> None:
        if type(self.section_forces) is not BeamIntegrationPointSectionField:
            raise TypeError(
                "section_forces must be BeamIntegrationPointSectionField"
            )
        fields = tuple(self.section_points)
        if any(type(field) is not BeamIntegrationPointStressField for field in fields):
            raise TypeError(
                "section_points must contain BeamIntegrationPointStressField values"
            )
        numbers = tuple(field.point_number for field in fields)
        if len(numbers) != len(set(numbers)):
            raise ValueError("section point field numbers must be unique")
        object.__setattr__(self, "section_points", fields)

    def point_field(self, number: int) -> BeamIntegrationPointStressField:
        for field in self.section_points:
            if field.point_number == number:
                return field
        raise KeyError(number)


def recover_integration_point_stress(
    result: ModelResult,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> BeamIntegrationPointStressRecovery:
    """Recover one B31 longitudinal integration-point stress per element."""

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
    point_rows: dict[int, list[BeamIntegrationPointStress]] = {}
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
        for point_stress in recover_point_stress(section, forces):
            point_rows.setdefault(point_stress.point.number, []).append(
                BeamIntegrationPointStress(
                    element_id=element_id,
                    integration_point=1,
                    coordinates=coordinates,
                    displacement=displacement,
                    section_point=point_stress.point,
                    s11=point_stress.s11,
                    s22=point_stress.s22,
                    s12=point_stress.s12,
                    mises=point_stress.mises,
                    max_principal=point_stress.max_principal,
                    mid_principal=point_stress.mid_principal,
                    min_principal=point_stress.min_principal,
                )
            )
    return BeamIntegrationPointStressRecovery(
        section_forces=BeamIntegrationPointSectionField(tuple(force_rows)),
        section_points=tuple(
            BeamIntegrationPointStressField(number, tuple(point_rows[number]))
            for number in sorted(point_rows)
        ),
    )


def nodal_envelope(
    result: Any,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[Beam2NodalStress, ...]:
    """Project integration-point S11 extrema to each incident mesh node."""

    _validate_checkpoint(checkpoint)
    recovery = recover_integration_point_stress(result, checkpoint=checkpoint)
    element_nodes = {
        _integer_id("element id", element.id): tuple(
            _integer_id("element node id", node_id) for node_id in element.node_ids
        )
        for element in result.model.mesh.elements
    }
    contributions: dict[int, list[BeamIntegrationPointStress]] = {}
    for field in recovery.section_points:
        for row in field.rows:
            for node_id in element_nodes[row.element_id]:
                contributions.setdefault(node_id, []).append(row)
    return tuple(
        Beam2NodalStress(
            node_id=node_id,
            maximum=max(row.s11 for row in contributions[node_id]),
            minimum=min(row.s11 for row in contributions[node_id]),
            absolute_maximum=max(
                abs(row.s11) for row in contributions[node_id]
            ),
        )
        for node_id in result.model.mesh.node_ids
        if node_id in contributions
    )


def absolute_maximum(result: Any) -> float:
    """Return the maximum absolute stress from the Beam2 nodal envelope."""
    return max(
        (row.absolute_maximum for row in nodal_envelope(result)),
        default=0.0,
    )


def export_nodal(result: Any, path: str) -> None:
    """Write an incident-node projection of B31 integration-point S11."""
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
