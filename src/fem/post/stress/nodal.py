from __future__ import annotations

import csv
from typing import Sequence

import numpy as np

from ...elements import get_element_kernel
from .._paths import prepare_output_path
from ..averaging import NodalAveragingPolicy
from . import dispatch
from ._common import (
    PLANE_NODAL_HEADER,
    SOLID_NODAL_HEADER,
    node_lookup,
)
from .field import (
    PlaneElementNodalField,
    ResolvedNodalStressField,
    ResolvedNodalStressRow,
    StressField,
    StressPosition,
    collect_plane_element_nodal,
    collect_stress,
)
from .invariants import von_mises_3d, von_mises_plane


def by_type(
    type_key: str,
    mesh,
    U: Sequence[float],
    path: str,
    gauss_order: int | None = None,
    threshold: float = 75.0,
) -> None:
    """Export resolved nodal stresses by normalized element type key."""
    if type_key == "truss2":
        raise ValueError("Nodal stress export is not available for Truss2 elements")
    if type_key not in dispatch.NODAL_STRESS_KEYS:
        raise ValueError(f"Unsupported stress element type key: {type_key!r}")
    _resolved(mesh, U, path, type_key, gauss_order, threshold)


def mixed(
    type_keys: Sequence[str],
    mesh,
    U: Sequence[float],
    path: str,
    gauss_order: int | None = None,
    threshold: float = 75.0,
) -> None:
    """Export resolved mixed nodal stresses for compatible stress groups."""
    if not dispatch.nodal_stress_supported(type_keys):
        raise ValueError(f"Nodal stress export is not available for {type_keys}")
    dispatch.stress_group_for_keys(type_keys)
    _resolved(mesh, U, path, None, gauss_order, threshold)


def _resolved(
    mesh,
    U: Sequence[float],
    path: str,
    element_type: str | None,
    gauss_order: int | None,
    threshold: float,
) -> None:
    """Collect, resolve, and write a nodal stress field."""
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if dispatch.stress_group_for_keys(type_keys) == "plane":
        recovered = collect_plane_element_nodal(
            mesh,
            U,
            element_type=element_type,
            gauss_order=gauss_order,
        )
        write_recovered(mesh, recovered, path, threshold=threshold)
        return
    recovered = collect_stress(
        mesh,
        U,
        position=StressPosition.ELEMENT_NODAL,
        element_type=element_type,
        gauss_order=gauss_order,
    )
    write_recovered(mesh, recovered, path, threshold=threshold)


def write_recovered(
    mesh,
    recovered: StressField | PlaneElementNodalField,
    path: str,
    *,
    threshold: float = 75.0,
) -> None:
    """Resolve and write one already-recovered element-nodal stress field."""

    resolved = _resolve_recovered(mesh, recovered, threshold)
    _write_resolved(mesh, resolved, path)


def _resolve_recovered(
    mesh,
    recovered: StressField | PlaneElementNodalField,
    threshold: float,
) -> ResolvedNodalStressField:
    """Resolve canonical records without legacy-to-canonical round trips."""

    try:
        policy = NodalAveragingPolicy(threshold_percent=threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "threshold must be a finite number from 0.0 through 100.0"
        ) from error
    if type(recovered) not in {StressField, PlaneElementNodalField}:
        raise TypeError(
            "recovered must be a StressField or PlaneElementNodalField"
        )
    if recovered.position is not StressPosition.ELEMENT_NODAL:
        raise ValueError("nodal export requires element-nodal stress")

    is_plane = recovered.component_names == ("S11", "S22", "S33", "S12")
    if not is_plane and recovered.component_names != (
        "S11",
        "S22",
        "S33",
        "S12",
        "S23",
        "S13",
    ):
        raise ValueError("nodal export requires canonical continuum stress")

    node_ids = tuple(int(node_id) for node_id in mesh.node_ids)
    node_order = {node_id: index for index, node_id in enumerate(node_ids)}
    records_by_node: list[list[int]] = [[] for _ in node_ids]
    record_count = len(recovered.records)
    component_count = len(recovered.component_names)
    values = np.empty((record_count, component_count), dtype=float)
    weights = np.empty(record_count, dtype=float)
    node_indices = np.empty(record_count, dtype=np.intp)
    region_indices = np.empty(record_count, dtype=np.intp)
    region_order: dict[object, int] = {}

    for index, record in enumerate(recovered.records):
        if (
            record.node_id is None
            or record.elem_id is None
            or record.local_node is None
            or record.region_key is None
        ):
            raise ValueError(
                "element-nodal stress is missing row provenance"
            )
        try:
            node_index = node_order[record.node_id]
        except KeyError as error:
            raise ValueError(
                f"element-nodal node {record.node_id} is absent from mesh"
            ) from error
        if len(record.components) != component_count:
            raise ValueError(
                "element-nodal stress component count does not match field"
            )
        records_by_node[node_index].append(index)
        node_indices[index] = node_index
        values[index] = record.components
        weights[index] = record.weight
        region_indices[index] = region_order.setdefault(
            record.region_key,
            len(region_order),
        )

    if not np.isfinite(values).all():
        raise ValueError("element-nodal stress components must be finite")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("element-nodal stress weights must be positive and finite")

    region_count = len(region_order)
    region_min = np.full((region_count, component_count), np.inf)
    region_max = np.full((region_count, component_count), -np.inf)
    np.minimum.at(region_min, region_indices, values)
    np.maximum.at(region_max, region_indices, values)
    region_ranges = region_max - region_min
    region_tolerances = (
        np.finfo(float).eps
        * np.maximum(1.0, np.maximum(np.abs(region_min), np.abs(region_max)))
        * 32.0
    )

    node_count = len(node_ids)
    node_min = np.full((node_count, component_count), np.inf)
    node_max = np.full((node_count, component_count), -np.inf)
    np.minimum.at(node_min, node_indices, values)
    np.maximum.at(node_max, node_indices, values)
    weighted_sums = np.zeros((node_count, component_count), dtype=float)
    np.add.at(weighted_sums, node_indices, values * weights[:, None])
    weight_sums = np.bincount(
        node_indices,
        weights=weights,
        minlength=node_count,
    )

    elements_by_id = {int(elem.id): elem for elem in mesh.elements}
    plane_metadata: dict[int, tuple[str, float]] = {}

    def metadata(record) -> tuple[str | None, float | None]:
        if not is_plane:
            return None, None
        elem_id = int(record.elem_id)
        cached = plane_metadata.get(elem_id)
        if cached is None:
            elem = elements_by_id[elem_id]
            cached = get_element_kernel(elem.type)._plane_data(elem)
            plane_metadata[elem_id] = cached
        return cached

    def raw_row(record) -> ResolvedNodalStressRow:
        plane_type, poisson_ratio = metadata(record)
        components = (
            (
                record.components[0],
                record.components[1],
                record.components[3],
            )
            if is_plane
            else record.components
        )
        return ResolvedNodalStressRow(
            node_id=int(record.node_id),
            components=tuple(float(value) for value in components),
            elem_id=int(record.elem_id),
            local_node=int(record.local_node),
            averaged=False,
            plane_type=plane_type,
            poisson_ratio=poisson_ratio,
        )

    rows: list[ResolvedNodalStressRow] = []
    for node_index, node_id in enumerate(node_ids):
        source_indices = records_by_node[node_index]
        if not source_indices:
            rows.append(
                ResolvedNodalStressRow(
                    node_id=node_id,
                    components=(0.0,) * (3 if is_plane else 6),
                    elem_id=None,
                    local_node=None,
                    averaged=True,
                )
            )
            continue

        source_regions = tuple(
            dict.fromkeys(int(region_indices[index]) for index in source_indices)
        )
        if (
            len(source_regions) > 1
            or len(source_indices) == 1
            or policy.threshold_percent == 0.0
        ):
            rows.extend(raw_row(recovered.records[index]) for index in source_indices)
            continue

        region_index = source_regions[0]
        region_range = region_ranges[region_index]
        outside_tolerance = region_range > region_tolerances[region_index]
        relative_variation = np.zeros(component_count, dtype=float)
        relative_variation[outside_tolerance] = (
            100.0
            * (node_max[node_index] - node_min[node_index])[outside_tolerance]
            / region_range[outside_tolerance]
        )
        if not np.all(relative_variation <= policy.threshold_percent):
            rows.extend(raw_row(recovered.records[index]) for index in source_indices)
            continue

        first = recovered.records[source_indices[0]]
        plane_type, poisson_ratio = metadata(first)
        averaged = weighted_sums[node_index] / weight_sums[node_index]
        components = (
            (averaged[0], averaged[1], averaged[3])
            if is_plane
            else averaged
        )
        rows.append(
            ResolvedNodalStressRow(
                node_id=node_id,
                components=tuple(float(value) for value in components),
                elem_id=None,
                local_node=None,
                averaged=True,
                plane_type=plane_type,
                poisson_ratio=poisson_ratio,
            )
        )

    return ResolvedNodalStressField(
        ("sig_x", "sig_y", "tau_xy")
        if is_plane
        else ("sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx"),
        tuple(rows),
    )


def _write_resolved(mesh, resolved: ResolvedNodalStressField, path: str) -> None:
    """Write resolved plane or solid rows with provenance metadata."""
    lookup = node_lookup(mesh)
    is_plane = resolved.component_names == ("sig_x", "sig_y", "tau_xy")
    output_path = prepare_output_path(path)
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(PLANE_NODAL_HEADER if is_plane else SOLID_NODAL_HEADER)
        for row in resolved.rows:
            node = lookup[row.node_id]
            provenance = [
                "" if row.elem_id is None else row.elem_id,
                "" if row.local_node is None else row.local_node,
                "true" if row.averaged else "false",
            ]
            if is_plane:
                sig_x, sig_y, tau_xy = row.components
                writer.writerow([
                    row.node_id,
                    node.x,
                    node.y,
                    *provenance,
                    sig_x,
                    sig_y,
                    tau_xy,
                    von_mises_plane(
                        sig_x,
                        sig_y,
                        tau_xy,
                        row.plane_type or "stress",
                        row.poisson_ratio or 0.0,
                    ),
                ])
                continue
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = row.components
            writer.writerow([
                row.node_id,
                node.x,
                node.y,
                node.z,
                *provenance,
                sig_x,
                sig_y,
                sig_z,
                tau_xy,
                tau_yz,
                tau_zx,
                von_mises_3d(sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx),
            ])
