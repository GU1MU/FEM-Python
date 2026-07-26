from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from ...elements import get_element_kernel
from ..averaging import NodalAveragingPolicy, resolve_nodal_stress
from ..fields import (
    MATERIAL_SIGNATURE_KEY as MATERIAL_SIGNATURE_KEY,
    ResultRegionKey,
    SECTION_SIGNATURE_KEY as SECTION_SIGNATURE_KEY,
    _result_region_key_from_compatible_signatures,
    result_region_key_for_element,
    result_region_sort_key,
)
from . import dispatch
from ._common import element_volume, node_lookup, validated_u
from .invariants import (
    StressInvariants,
    complete_plane_components,
    derive_stress_invariants,
)


PLANE_COMPONENT_NAMES = ("sig_x", "sig_y", "tau_xy")
SOLID_COMPONENT_NAMES = (
    "sig_x",
    "sig_y",
    "sig_z",
    "tau_xy",
    "tau_yz",
    "tau_zx",
)
CANONICAL_PLANE_COMPONENT_NAMES = ("S11", "S22", "S33", "S12")
CANONICAL_SOLID_COMPONENT_NAMES = ("S11", "S22", "S33", "S12", "S23", "S13")


class StressPosition(str, Enum):
    """Supported locations for continuum stress output."""

    INTEGRATION_POINT = "integration_point"
    CENTROID = "centroid"
    ELEMENT_NODAL = "element_nodal"
    NODAL = "nodal"


def StressRegionKey(
    material_signature: Any,
    section_signature: Any,
) -> ResultRegionKey:
    """Compatibility factory returning the sole result-region identity type."""

    return _result_region_key_from_compatible_signatures(
        material_signature,
        section_signature,
    )


@dataclass(frozen=True)
class StressRecord:
    """One complete stress tensor and its derived values at one result location."""

    position: StressPosition
    coordinates: tuple[float, ...]
    components: tuple[float, ...]
    invariants: StressInvariants
    elem_id: int | None = None
    integration_point: int | None = None
    natural_coordinates: tuple[float, ...] | None = None
    node_id: int | None = None
    local_node: int | None = None
    region_key: ResultRegionKey | None = None
    weight: float = 1.0
    displacement: tuple[float, ...] | None = None
    averaged: bool | None = None

    def values(self, component_names: Sequence[str]) -> dict[str, float]:
        """Return named components and invariants for exporters and GUI consumers."""
        values = {
            str(name): float(value)
            for name, value in zip(component_names, self.components)
        }
        values.update({
            "Mises": self.invariants.mises,
            "MaxPrincipal": self.invariants.max_principal,
            "MidPrincipal": self.invariants.mid_principal,
            "MinPrincipal": self.invariants.min_principal,
        })
        return values


@dataclass(frozen=True)
class StressField:
    """A deterministic collection of stress records at one result position."""

    position: StressPosition
    component_names: tuple[str, ...]
    records: tuple[StressRecord, ...]


@dataclass(frozen=True)
class _ElementIntegrationPointField:
    elem: Any
    type_key: str
    kernel: Any
    gauss_order: int | None
    natural_coordinates: np.ndarray
    components: np.ndarray
    region_key: ResultRegionKey
    weight: float


class StressRecovery:
    """Cache canonical integration-point components for repeated position recovery."""

    def __init__(
        self,
        mesh: Any,
        U: Sequence[float],
        element_type: str | None = None,
        gauss_order: int | None = None,
    ) -> None:
        type_keys = dispatch.resolve_type_keys(mesh, element_type)
        group = dispatch.stress_group_for_keys(type_keys)
        if group not in {"plane", "solid"}:
            raise ValueError(
                "StressPosition output currently supports plane and solid elements only"
            )
        self.mesh = mesh
        self.lookup = node_lookup(mesh)
        self.component_names = (
            CANONICAL_PLANE_COMPONENT_NAMES
            if group == "plane"
            else CANONICAL_SOLID_COMPONENT_NAMES
        )
        self._U = validated_u(mesh, U)
        self._ip_fields = _collect_element_integration_points(
            mesh,
            self._U,
            self.lookup,
            set(type_keys),
            group,
            gauss_order,
        )
        self._cache: dict[StressPosition, StressField] = {}

    def collect(
        self,
        position: StressPosition | str = StressPosition.INTEGRATION_POINT,
    ) -> StressField:
        """Recover and cache one requested stress position."""
        try:
            resolved_position = StressPosition(position)
        except ValueError as exc:
            choices = ", ".join(item.value for item in StressPosition)
            raise ValueError(
                f"Unsupported stress position {position!r}; expected one of {choices}"
            ) from exc
        cached = self._cache.get(resolved_position)
        if cached is not None:
            return cached

        if resolved_position is StressPosition.INTEGRATION_POINT:
            records = _integration_point_records(
                self.mesh, self.lookup, self._ip_fields, self.component_names
            )
        elif resolved_position is StressPosition.CENTROID:
            records = _centroid_records(
                self.mesh, self.lookup, self._ip_fields, self.component_names
            )
        elif resolved_position is StressPosition.ELEMENT_NODAL:
            records = _element_nodal_records(
                self.mesh,
                self.lookup,
                self._ip_fields,
                self.component_names,
                self._U,
            )
        else:
            element_nodal_field = self.collect(StressPosition.ELEMENT_NODAL)
            records = _average_nodal_records(
                self.mesh,
                self.lookup,
                element_nodal_field.records,
                self.component_names,
            )
        stress_field = StressField(
            resolved_position,
            self.component_names,
            tuple(records),
        )
        self._cache[resolved_position] = stress_field
        return stress_field


@dataclass(frozen=True)
class ElementNodalStressContribution:
    """One element-local stress tensor recovered at a mesh node."""

    node_id: int
    elem_id: int
    local_node: int
    components: tuple[float, ...]
    weight: float
    region_key: ResultRegionKey
    plane_type: str | None = None
    poisson_ratio: float | None = None


@dataclass(frozen=True)
class NodalStressField:
    """Raw element-nodal stress contributions grouped in mesh-node order."""

    component_names: tuple[str, ...]
    contributions_by_node: Mapping[int, tuple[ElementNodalStressContribution, ...]]
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedNodalStressRow:
    """One averaged or element-local nodal stress value for export."""

    node_id: int
    components: tuple[float, ...]
    elem_id: int | None
    local_node: int | None
    averaged: bool
    plane_type: str | None = None
    poisson_ratio: float | None = None


@dataclass(frozen=True)
class ResolvedNodalStressField:
    """Resolved nodal stress rows in deterministic mesh and element order."""

    component_names: tuple[str, ...]
    rows: tuple[ResolvedNodalStressRow, ...]


def collect_stress(
    mesh: Any,
    U: Sequence[float],
    position: StressPosition | str = StressPosition.INTEGRATION_POINT,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> StressField:
    """Collect one continuum stress field from the canonical integration-point data."""
    return StressRecovery(
        mesh,
        U,
        element_type=element_type,
        gauss_order=gauss_order,
    ).collect(position)


def _collect_element_integration_points(
    mesh: Any,
    U: np.ndarray,
    lookup: dict[int, Any],
    selected: set[str],
    group: str,
    gauss_order: int | None,
) -> list[_ElementIntegrationPointField]:
    fields: list[_ElementIntegrationPointField] = []
    for elem in mesh.elements:
        type_key = dispatch.type_key_from_name(elem.type)
        if type_key not in selected:
            continue
        kernel = get_element_kernel(elem.type)
        order = (
            gauss_order
            if gauss_order is not None
            else dispatch.default_gauss_order(type_key)
        )
        if order is None:
            natural, raw_components = kernel.integration_point_stress(
                mesh, elem, U, lookup
            )
        else:
            natural, raw_components = kernel.integration_point_stress(
                mesh, elem, U, lookup, order
            )
        raw_values = np.asarray(raw_components, dtype=float)
        if group == "plane":
            plane_type, nu = kernel._plane_data(elem)
            complete = np.asarray([
                complete_plane_components(row, plane_type, nu)
                for row in raw_values
            ], dtype=float)
        else:
            complete = raw_values
        expected = len(
            CANONICAL_PLANE_COMPONENT_NAMES
            if group == "plane"
            else CANONICAL_SOLID_COMPONENT_NAMES
        )
        if complete.ndim != 2 or complete.shape[1] != expected:
            raise ValueError(
                f"Element {elem.id} integration-point stress shape {complete.shape} "
                f"does not have {expected} components"
            )
        weight = (
            element_volume(mesh, elem, lookup)
            if type_key in {"tet4", "tet10"}
            else 1.0
        )
        fields.append(
            _ElementIntegrationPointField(
                elem=elem,
                type_key=type_key,
                kernel=kernel,
                gauss_order=order,
                natural_coordinates=np.asarray(natural, dtype=float),
                components=complete,
                region_key=result_region_key_for_element(elem),
                weight=float(weight),
            )
        )
    return fields


def _integration_point_records(
    mesh: Any,
    lookup: dict[int, Any],
    fields: Sequence[_ElementIntegrationPointField],
    component_names: tuple[str, ...],
) -> list[StressRecord]:
    records: list[StressRecord] = []
    for item in fields:
        for index, (natural, components) in enumerate(
            zip(item.natural_coordinates, item.components),
            start=1,
        ):
            records.append(
                _make_record(
                    StressPosition.INTEGRATION_POINT,
                    components,
                    component_names,
                    coordinates=_physical_coordinates(
                        item.type_key, item.elem, natural, lookup
                    ),
                    elem_id=int(item.elem.id),
                    integration_point=index,
                    natural_coordinates=tuple(float(value) for value in natural),
                    region_key=item.region_key,
                    weight=item.weight,
                )
            )
    return records


def _centroid_records(
    mesh: Any,
    lookup: dict[int, Any],
    fields: Sequence[_ElementIntegrationPointField],
    component_names: tuple[str, ...],
) -> list[StressRecord]:
    records: list[StressRecord] = []
    for item in fields:
        if item.gauss_order is None:
            components = item.kernel.interpolate_stress_to_centroid(item.components)
        else:
            components = item.kernel.interpolate_stress_to_centroid(
                item.components, item.gauss_order
            )
        natural = _centroid_natural_coordinates(item.type_key)
        records.append(
            _make_record(
                StressPosition.CENTROID,
                components,
                component_names,
                coordinates=_physical_coordinates(
                    item.type_key, item.elem, natural, lookup
                ),
                elem_id=int(item.elem.id),
                natural_coordinates=natural,
                region_key=item.region_key,
                weight=item.weight,
            )
        )
    return records


def _element_nodal_records(
    mesh: Any,
    lookup: dict[int, Any],
    fields: Sequence[_ElementIntegrationPointField],
    component_names: tuple[str, ...],
    U: np.ndarray,
) -> list[StressRecord]:
    records: list[StressRecord] = []
    for item in fields:
        if item.gauss_order is None:
            node_values = item.kernel.extrapolate_stress_to_nodes(item.components)
        else:
            node_values = item.kernel.extrapolate_stress_to_nodes(
                item.components, item.gauss_order
            )
        if node_values.shape != (len(item.elem.node_ids), len(component_names)):
            raise ValueError(
                f"Element {item.elem.id} element-nodal stress shape {node_values.shape} "
                f"does not match ({len(item.elem.node_ids)}, {len(component_names)})"
            )
        for local_node, (node_id, components) in enumerate(
            zip(item.elem.node_ids, node_values),
            start=1,
        ):
            node = lookup[int(node_id)]
            records.append(
                _make_record(
                    StressPosition.ELEMENT_NODAL,
                    components,
                    component_names,
                    coordinates=_node_coordinates(node),
                    elem_id=int(item.elem.id),
                    node_id=int(node_id),
                    local_node=local_node,
                    region_key=item.region_key,
                    weight=item.weight,
                    displacement=_node_displacement(mesh, int(node_id), U),
                )
            )
    return records


def _average_nodal_records(
    mesh: Any,
    lookup: dict[int, Any],
    element_nodal: Sequence[StressRecord],
    component_names: tuple[str, ...],
) -> list[StressRecord]:
    grouped: dict[tuple[int, ResultRegionKey], list[StressRecord]] = {}
    for record in element_nodal:
        if record.node_id is None or record.region_key is None:
            continue
        grouped.setdefault((record.node_id, record.region_key), []).append(record)

    node_order = {
        int(node_id): index for index, node_id in enumerate(mesh.node_ids)
    }
    records: list[StressRecord] = []
    for (node_id, region_key), contributions in sorted(
        grouped.items(),
        key=lambda item: (
            node_order.get(item[0][0], len(node_order)),
            result_region_sort_key(item[0][1]),
        ),
    ):
        weights = np.asarray(
            [record.weight for record in contributions],
            dtype=float,
        )
        components = np.average(
            np.asarray([record.components for record in contributions], dtype=float),
            axis=0,
            weights=weights,
        )
        records.append(
            _make_record(
                StressPosition.NODAL,
                components,
                component_names,
                coordinates=_node_coordinates(lookup[node_id]),
                node_id=node_id,
                region_key=region_key,
                weight=float(np.sum(weights)),
                displacement=contributions[0].displacement,
            )
        )
    return records


def _make_record(
    position: StressPosition,
    components,
    component_names: tuple[str, ...],
    *,
    coordinates: tuple[float, ...],
    elem_id: int | None = None,
    integration_point: int | None = None,
    natural_coordinates: tuple[float, ...] | None = None,
    node_id: int | None = None,
    local_node: int | None = None,
    region_key: ResultRegionKey | None = None,
    weight: float = 1.0,
    displacement: tuple[float, ...] | None = None,
    averaged: bool | None = None,
) -> StressRecord:
    values = tuple(float(value) for value in components)
    return StressRecord(
        position=position,
        coordinates=coordinates,
        components=values,
        invariants=derive_stress_invariants(values, component_names),
        elem_id=elem_id,
        integration_point=integration_point,
        natural_coordinates=natural_coordinates,
        node_id=node_id,
        local_node=local_node,
        region_key=region_key,
        weight=float(weight),
        displacement=displacement,
        averaged=averaged,
    )


def _node_coordinates(node: Any) -> tuple[float, ...]:
    coordinates = [float(node.x), float(node.y)]
    if hasattr(node, "z"):
        coordinates.append(float(node.z))
    return tuple(coordinates)


def _node_displacement(
    mesh: Any,
    node_id: int,
    U: np.ndarray,
) -> tuple[float, ...]:
    translation_count = (
        3
        if mesh.nodes and hasattr(mesh.nodes[0], "z")
        else 2
    )
    return tuple(
        float(U[mesh.global_dof(node_id, component)])
        for component in range(min(mesh.dofs_per_node, translation_count))
    )


def _physical_coordinates(
    type_key: str,
    elem: Any,
    natural_coordinates,
    lookup: dict[int, Any],
) -> tuple[float, ...]:
    natural = tuple(float(value) for value in natural_coordinates)
    shape_values = natural_shape_values(type_key, natural)
    node_coordinates = np.asarray(
        [_node_coordinates(lookup[int(node_id)]) for node_id in elem.node_ids],
        dtype=float,
    )
    coordinates = shape_values @ node_coordinates
    return tuple(float(value) for value in coordinates)


def natural_shape_values(
    type_key: str,
    natural_coordinates: Sequence[float],
) -> np.ndarray:
    """Return continuum shape-function values at one natural coordinate."""
    natural = tuple(float(value) for value in natural_coordinates)
    if type_key == "tri3":
        xi, eta = natural
        return np.asarray([1.0 - xi - eta, xi, eta], dtype=float)
    if type_key == "tri6":
        from ...elements.triangle import tri6_shape_funcs_grads

        return tri6_shape_funcs_grads(*natural)[0]
    if type_key == "quad4":
        xi, eta = natural
        return 0.25 * np.asarray([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ], dtype=float)
    if type_key == "quad8":
        from ...elements.quadrilateral import quad8_shape_funcs_grads

        return quad8_shape_funcs_grads(*natural)[0]
    if type_key == "tet4":
        from ...elements.tetrahedron import tet4_shape_funcs_grads

        return tet4_shape_funcs_grads(*natural)[0]
    if type_key == "tet10":
        from ...elements.tetrahedron import tet10_shape_funcs_grads

        return tet10_shape_funcs_grads(*natural)[0]
    if type_key == "hex8":
        from ...elements.hexahedron import hex8_shape_funcs_grads

        return hex8_shape_funcs_grads(*natural)[0]
    if type_key == "hex20":
        from ...elements.hexahedron import hex20_shape_funcs_grads

        return hex20_shape_funcs_grads(*natural)[0]
    raise ValueError(f"Unsupported continuum stress element type key: {type_key!r}")


def _centroid_natural_coordinates(type_key: str) -> tuple[float, ...]:
    if type_key in {"tri3", "tri6"}:
        return (1.0 / 3.0, 1.0 / 3.0)
    if type_key in {"quad4", "quad8"}:
        return (0.0, 0.0)
    if type_key in {"tet4", "tet10"}:
        return (0.25, 0.25, 0.25)
    if type_key in {"hex8", "hex20"}:
        return (0.0, 0.0, 0.0)
    raise ValueError(f"Unsupported continuum stress element type key: {type_key!r}")


def resolve(
    field: NodalStressField,
    threshold: float = 75.0,
) -> ResolvedNodalStressField:
    """Compatibility adapter over the canonical complete-tensor resolver.

    The historical component schema and isolated-node zero rows remain here
    only for legacy CSV/VTK callers.
    """

    try:
        policy = NodalAveragingPolicy(threshold_percent=threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "threshold must be a finite number from 0.0 through 100.0"
        ) from error

    canonical, metadata = _canonical_field_from_legacy(field)
    element_ids = tuple(
        dict.fromkeys(
            contribution.elem_id
            for node_id in field.node_ids
            for contribution in field.contributions_by_node.get(node_id, ())
        )
    )
    resolved = resolve_nodal_stress(
        canonical,
        policy,
        node_ids=field.node_ids,
        element_ids=element_ids,
    )
    canonical_plane = (
        canonical.component_names == CANONICAL_PLANE_COMPONENT_NAMES
    )
    rows_by_node_region: dict[
        tuple[int, ResultRegionKey],
        list[ResolvedNodalStressRow],
    ] = {}
    for record in resolved.records:
        if record.node_id is None or record.region_key is None:
            continue
        plane_type, poisson_ratio = metadata[
            (record.node_id, record.region_key)
        ]
        components = (
            (
                record.components[0],
                record.components[1],
                record.components[3],
            )
            if canonical_plane
            else record.components
        )
        rows_by_node_region.setdefault(
            (record.node_id, record.region_key),
            [],
        ).append(
            ResolvedNodalStressRow(
                node_id=record.node_id,
                components=tuple(float(value) for value in components),
                elem_id=record.elem_id,
                local_node=record.local_node,
                averaged=bool(record.averaged),
                plane_type=plane_type,
                poisson_ratio=poisson_ratio,
            )
        )

    rows: list[ResolvedNodalStressRow] = []
    for node_id in field.node_ids:
        contributions = tuple(field.contributions_by_node.get(node_id, ()))
        region_order = tuple(
            dict.fromkeys(
                contribution.region_key for contribution in contributions
            )
        )
        if region_order:
            # Historical field.resolve treated any multi-region node as fully
            # unaveraged. Keep that projection only in this legacy adapter.
            if len(region_order) > 1:
                rows.extend(
                    _legacy_raw_row(contribution)
                    for contribution in contributions
                )
                continue
            for region_key in region_order:
                rows.extend(
                    rows_by_node_region[(node_id, region_key)]
                )
            continue
        rows.append(
            ResolvedNodalStressRow(
                node_id=node_id,
                components=(0.0,) * len(field.component_names),
                elem_id=None,
                local_node=None,
                averaged=True,
            )
        )
    return ResolvedNodalStressField(field.component_names, tuple(rows))


def _legacy_raw_row(
    contribution: ElementNodalStressContribution,
) -> ResolvedNodalStressRow:
    return ResolvedNodalStressRow(
        node_id=contribution.node_id,
        components=contribution.components,
        elem_id=contribution.elem_id,
        local_node=contribution.local_node,
        averaged=False,
        plane_type=contribution.plane_type,
        poisson_ratio=contribution.poisson_ratio,
    )


def collect(
    mesh: Any,
    U: Sequence[float],
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> NodalStressField:
    """Compatibility view of canonical element-nodal stress records."""
    stress_field = collect_stress(
        mesh,
        U,
        position=StressPosition.ELEMENT_NODAL,
        element_type=element_type,
        gauss_order=gauss_order,
    )
    contributions: dict[int, list[ElementNodalStressContribution]] = {
        int(node_id): [] for node_id in mesh.node_ids
    }
    is_plane = stress_field.component_names == CANONICAL_PLANE_COMPONENT_NAMES
    element_lookup = {int(elem.id): elem for elem in mesh.elements}
    for record in stress_field.records:
        if (
            record.node_id is None
            or record.elem_id is None
            or record.local_node is None
            or record.region_key is None
        ):
            continue
        plane_type = None
        poisson_ratio = None
        if is_plane:
            plane_type, poisson_ratio = get_element_kernel(
                element_lookup[record.elem_id].type
            )._plane_data(element_lookup[record.elem_id])
        contributions[record.node_id].append(
            ElementNodalStressContribution(
                node_id=record.node_id,
                elem_id=record.elem_id,
                local_node=record.local_node,
                components=(
                    (
                        record.components[0],
                        record.components[1],
                        record.components[3],
                    )
                    if is_plane
                    else record.components
                ),
                weight=record.weight,
                region_key=record.region_key,
                plane_type=plane_type,
                poisson_ratio=poisson_ratio,
            )
        )

    return NodalStressField(
        component_names=(
            PLANE_COMPONENT_NAMES if is_plane else SOLID_COMPONENT_NAMES
        ),
        contributions_by_node={
            node_id: tuple(values) for node_id, values in contributions.items()
        },
        node_ids=tuple(int(node_id) for node_id in mesh.node_ids),
    )


def _canonical_field_from_legacy(
    legacy: NodalStressField,
) -> tuple[
    StressField,
    dict[tuple[int, ResultRegionKey], tuple[str | None, float | None]],
]:
    is_plane = legacy.component_names == PLANE_COMPONENT_NAMES
    if not is_plane and legacy.component_names != SOLID_COMPONENT_NAMES:
        raise ValueError("legacy nodal stress field has unsupported components")
    component_names = (
        CANONICAL_PLANE_COMPONENT_NAMES
        if is_plane
        else CANONICAL_SOLID_COMPONENT_NAMES
    )
    metadata: dict[
        tuple[int, ResultRegionKey],
        tuple[str | None, float | None],
    ] = {}
    records: list[StressRecord] = []
    for node_id in legacy.node_ids:
        for contribution in legacy.contributions_by_node.get(node_id, ()):
            if contribution.node_id != node_id:
                raise ValueError(
                    "legacy contribution node id does not match its node group"
                )
            if is_plane:
                plane_type = contribution.plane_type or "stress"
                poisson_ratio = (
                    0.0
                    if contribution.poisson_ratio is None
                    else contribution.poisson_ratio
                )
                components = complete_plane_components(
                    contribution.components,
                    plane_type,
                    poisson_ratio,
                )
            else:
                components = contribution.components
            metadata.setdefault(
                (node_id, contribution.region_key),
                (
                    contribution.plane_type,
                    contribution.poisson_ratio,
                ),
            )
            records.append(
                _make_record(
                    StressPosition.ELEMENT_NODAL,
                    components,
                    component_names,
                    coordinates=(),
                    elem_id=contribution.elem_id,
                    node_id=node_id,
                    local_node=contribution.local_node,
                    region_key=contribution.region_key,
                    weight=contribution.weight,
                )
            )
    return (
        StressField(
            StressPosition.ELEMENT_NODAL,
            component_names,
            tuple(records),
        ),
        metadata,
    )
