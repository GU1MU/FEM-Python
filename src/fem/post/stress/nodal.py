from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from ...core.model import model_element_info
from ...core.mesh import HexMesh3D, Mesh3DProtocol, PlaneMesh2D
from . import dispatch
from ._common import (
    PLANE_NODAL_HEADER,
    matches,
    nodal_stress,
    node_lookup,
    validated_u,
)
from .averaging import (
    AveragedNodalStressRow,
    ElementNodalContribution,
    RegionKey,
    StressAveragingPolicy,
    average_solid_nodal_contributions,
)
from .invariants import von_mises_plane


SOLID_REGION_NODAL_HEADER = [
    "source_elem_id",
    "source_local_node",
    "original_node_id",
    "region_id",
    "cluster_id",
    "material_id",
    "section_id",
    "element_type_id",
    "x",
    "y",
    "z",
    "sig_x",
    "sig_y",
    "sig_z",
    "tau_xy",
    "tau_yz",
    "tau_zx",
    "mises",
]


def by_type(
    type_key: str,
    mesh,
    U: Sequence[float],
    path: str,
    gauss_order: int | None = None,
    model: Any | None = None,
    averaging_policy: StressAveragingPolicy | None = None,
) -> None:
    """Export nodal stresses by normalized element type key."""
    if type_key == "truss2d":
        raise ValueError("Nodal stress export is not available for Truss2D elements")
    if type_key == "tri3":
        tri3(mesh, U, path)
    elif type_key == "quad4":
        quad4(mesh, U, path, 2 if gauss_order is None else gauss_order)
    elif type_key == "quad8":
        quad8(mesh, U, path, 3 if gauss_order is None else gauss_order)
    elif type_key == "hex8":
        hex8(mesh, U, path, 2 if gauss_order is None else gauss_order, model, averaging_policy)
    elif type_key == "tet4":
        tet4(mesh, U, path, model, averaging_policy)
    elif type_key == "tet10":
        tet10(mesh, U, path, model, averaging_policy)
    else:
        raise ValueError(f"Unsupported stress element type key: {type_key!r}")


def mixed(
    type_keys: Sequence[str],
    mesh,
    U: Sequence[float],
    path: str,
    gauss_order: int | None = None,
    model: Any | None = None,
    averaging_policy: StressAveragingPolicy | None = None,
) -> None:
    """Export mixed nodal stresses for compatible stress groups."""
    if not dispatch.nodal_stress_supported(type_keys):
        raise ValueError(f"Nodal stress export is not available for {type_keys}")
    group = dispatch.stress_group_for_keys(type_keys)
    if group == "plane":
        _plane_multi(mesh, U, path, set(type_keys), gauss_order)
        return
    if group == "solid":
        _solid_multi(mesh, U, path, set(type_keys), gauss_order, model, averaging_policy)
        return
    raise ValueError(f"Mixed nodal stress export is not available for group {group!r}")


def tri3(mesh: PlaneMesh2D, U: Sequence[float], path: str) -> None:
    """Export Tri3 nodal stresses averaged from elements."""
    _plane(mesh, U, path, "tri3")


def quad4(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Quad4 nodal stresses averaged from elements."""
    _plane(mesh, U, path, "quad4", gauss_order)


def quad8(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Quad8 nodal stresses averaged from elements."""
    _plane(mesh, U, path, "quad8", gauss_order)


def hex8(
    mesh: HexMesh3D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
    model: Any | None = None,
    averaging_policy: StressAveragingPolicy | None = None,
) -> None:
    """Export Hex8 nodal stresses averaged from connected elements."""
    _solid(mesh, U, path, "hex8", gauss_order, model, averaging_policy)


def tet4(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    model: Any | None = None,
    averaging_policy: StressAveragingPolicy | None = None,
) -> None:
    """Export Tet4 nodal stresses averaged from connected elements."""
    _solid(mesh, U, path, "tet4", None, model, averaging_policy)


def tet10(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    model: Any | None = None,
    averaging_policy: StressAveragingPolicy | None = None,
) -> None:
    """Export Tet10 nodal stresses averaged from connected elements."""
    _solid(mesh, U, path, "tet10", None, model, averaging_policy)


def _plane(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    type_key: str,
    gauss_order: int | None = None,
) -> None:
    """Export plane nodal stresses averaged from element nodal stresses."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = "stress"
    nu_ref = 0.0

    for elem in mesh.elements:
        if not matches(elem, type_key):
            continue
        node_vals, plane_type, nu_ref = nodal_stress(mesh, elem, U, lookup, gauss_order)
        for i, nid in enumerate(elem.node_ids):
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + node_vals[i]
            counts[nid] = counts.get(nid, 0) + 1

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLANE_NODAL_HEADER)
        for nid in mesh.node_ids:
            node = lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                sig_x, sig_y, tau_xy = (sums[nid] / counts[nid]).tolist()
            writer.writerow([
                nid,
                node.x,
                node.y,
                sig_x,
                sig_y,
                tau_xy,
                von_mises_plane(sig_x, sig_y, tau_xy, plane_type, nu_ref),
            ])


def _solid(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    type_key: str,
    gauss_order: int | None = None,
    model: Any | None = None,
    averaging_policy: StressAveragingPolicy | None = None,
) -> None:
    """Export region-aware solid nodal stresses from connected element nodes."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)
    contributions = _solid_contributions(mesh, U, lookup, {type_key}, gauss_order, model)
    rows = average_solid_nodal_contributions(contributions, averaging_policy)

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SOLID_REGION_NODAL_HEADER)
        writer.writerows(_solid_region_row(row) for row in rows)


def _plane_multi(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    type_keys: set[str],
    gauss_order: int | None = None,
) -> None:
    """Export mixed plane nodal stresses averaged from connected elements."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = "stress"
    nu_ref = 0.0

    for elem in mesh.elements:
        type_key = dispatch.type_key_from_name(elem.type)
        if type_key not in type_keys:
            continue
        order = (
            gauss_order
            if gauss_order is not None
            else dispatch.default_gauss_order(type_key)
        )
        node_vals, plane_type, nu_ref = nodal_stress(mesh, elem, U, lookup, order)
        for i, nid in enumerate(elem.node_ids):
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + node_vals[i]
            counts[nid] = counts.get(nid, 0) + 1

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLANE_NODAL_HEADER)
        for nid in mesh.node_ids:
            node = lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                sig_x, sig_y, tau_xy = (sums[nid] / counts[nid]).tolist()
            writer.writerow([
                nid,
                node.x,
                node.y,
                sig_x,
                sig_y,
                tau_xy,
                von_mises_plane(sig_x, sig_y, tau_xy, plane_type, nu_ref),
            ])


def _solid_multi(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    type_keys: set[str],
    gauss_order: int | None = None,
    model: Any | None = None,
    averaging_policy: StressAveragingPolicy | None = None,
) -> None:
    """Export mixed region-aware solid nodal stresses from connected element nodes."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)
    contributions = _solid_contributions(mesh, U, lookup, type_keys, gauss_order, model)
    rows = average_solid_nodal_contributions(contributions, averaging_policy)

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SOLID_REGION_NODAL_HEADER)
        writer.writerows(_solid_region_row(row) for row in rows)


def _solid_contributions(
    mesh: Mesh3DProtocol,
    U: np.ndarray,
    lookup: dict[int, Any],
    type_keys: set[str],
    gauss_order: int | None,
    model: Any | None,
) -> list[ElementNodalContribution]:
    contributions: list[ElementNodalContribution] = []
    for elem in mesh.elements:
        type_key = dispatch.type_key_from_name(elem.type)
        if type_key not in type_keys:
            continue
        order = gauss_order if gauss_order is not None else dispatch.default_gauss_order(type_key)
        node_vals = nodal_stress(mesh, elem, U, lookup, order)
        region_key = _region_key(elem, type_key, model)
        for local_idx, nid in enumerate(elem.node_ids, start=1):
            node = lookup[nid]
            contributions.append(
                ElementNodalContribution(
                    source_elem_id=int(elem.id),
                    source_local_node=local_idx,
                    original_node_id=int(nid),
                    region_key=region_key,
                    x=float(node.x),
                    y=float(node.y),
                    z=float(node.z),
                    stress=np.asarray(node_vals[local_idx - 1], dtype=float),
                )
            )
    return contributions


def _region_key(elem: Any, type_key: str | None, model: Any | None) -> RegionKey:
    if model is not None:
        info = model_element_info(model, elem.id)
        return RegionKey(
            material=info.material or _property_signature(info.properties),
            section=_model_section_key(model, elem.id, info.section_type),
            element_type=dispatch.type_key_from_name(info.type) or str(info.type),
        )
    props = getattr(elem, "props", {})
    return RegionKey(
        material=props.get("material", _property_signature(props)),
        section=props.get("section", props.get("section_type", "")),
        element_type=type_key or dispatch.type_key_from_name(elem.type) or str(elem.type),
    )


def _model_section_key(model: Any, elem_id: int, section_type: str | None) -> Any:
    """Return a stable section boundary key for one model element."""
    all_sets = dict(getattr(model, "element_sets", {}))
    all_sets.update(getattr(model, "metadata", {}).get("_abaqus_internal_element_sets", {}))
    match = None
    for index, section in enumerate(getattr(model, "sections", ())):
        element_set = all_sets.get(section.element_set)
        if element_set is None or elem_id not in element_set.element_ids:
            continue
        match = (
            index + 1,
            section.element_set,
            section.section_type,
            section.material,
            _property_signature(section.properties),
        )
    return match if match is not None else (section_type or "")


def _property_signature(props: Any) -> Any:
    """Return a hashable property signature for region splitting."""
    if not props:
        return ""
    ignored = {"material", "section", "section_type"}
    return tuple(
        (str(key), repr(value))
        for key, value in sorted(dict(props).items(), key=lambda item: str(item[0]))
        if key not in ignored
    )


def _solid_region_row(row: AveragedNodalStressRow) -> list[float | int]:
    return [
        row.source_elem_id,
        row.source_local_node,
        row.original_node_id,
        row.region_id,
        row.cluster_id,
        row.material_id,
        row.section_id,
        row.element_type_id,
        row.x,
        row.y,
        row.z,
        *row.stress,
        row.mises,
    ]


def _prepare_path(path: str | Path) -> Path:
    """Create output parent directory and return a Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
