from __future__ import annotations

import csv
from typing import Sequence

from ...elements import get_element_kernel
from ...core.mesh import Mesh2D, Mesh3D
from .._paths import prepare_output_path
from . import dispatch, truss
from ._common import (
    PLANE_ELEMENT_HEADER,
    SOLID_HEADER,
    TET_CENTROID,
    node_lookup,
    validated_u,
)
from .field import (
    CANONICAL_PLANE_COMPONENT_NAMES,
    StressField,
    StressPosition,
    collect_stress,
)
from .invariants import von_mises_3d, von_mises_plane


def by_type(
    type_key: str,
    mesh,
    U: Sequence[float],
    path: str,
    gauss_order: int | None = None,
) -> None:
    """Export element stresses by normalized element type key."""
    if type_key == "truss2":
        truss2(mesh, U, path)
    elif type_key == "tri3":
        tri3(mesh, U, path)
    elif type_key == "tri6":
        tri6(mesh, U, path, 3 if gauss_order is None else gauss_order)
    elif type_key == "quad4":
        quad4(mesh, U, path, 2 if gauss_order is None else gauss_order)
    elif type_key == "quad8":
        quad8(mesh, U, path, 3 if gauss_order is None else gauss_order)
    elif type_key == "hex8":
        hex8(mesh, U, path)
    elif type_key == "hex20":
        hex20(mesh, U, path)
    elif type_key == "tet4":
        tet4(mesh, U, path)
    elif type_key == "tet10":
        tet10(mesh, U, path)
    else:
        raise ValueError(f"Unsupported stress element type key: {type_key!r}")


def mixed(
    type_keys: Sequence[str],
    mesh,
    U: Sequence[float],
    path: str,
    gauss_order: int | None = None,
) -> None:
    """Export element stresses for compatible mixed stress groups."""
    if not dispatch.element_stress_supported(type_keys):
        raise ValueError(f"Element stress export is not available for {type_keys}")
    group = dispatch.stress_group_for_keys(type_keys)
    if group == "plane":
        _plane_multi(mesh, U, path, set(type_keys), gauss_order)
        return
    if group == "solid":
        _solid_multi(mesh, U, path, set(type_keys))
        return
    raise ValueError(f"Mixed element stress export is not available for group {group!r}")


def truss2(mesh: Mesh3D, U: Sequence[float], path: str) -> None:
    """Export Truss2 element axial strain/stress and mises to CSV."""
    recovered = truss.recover(mesh, U)

    path = prepare_output_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elem_id",
            "node_i",
            "node_j",
            "axial_strain",
            "axial_stress",
            "mises",
        ])
        for elem, row in zip(mesh.elements, recovered.rows, strict=True):
            ni_id, nj_id = elem.node_ids
            writer.writerow([
                row.element_id,
                ni_id,
                nj_id,
                row.LE11,
                row.S11,
                row.Mises,
            ])


def tri3(mesh: Mesh2D, U: Sequence[float], path: str) -> None:
    """Export Tri3 element-nodal stresses without averaging."""
    _plane(mesh, U, path, "tri3")


def tri6(
    mesh: Mesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Tri6 element-nodal stresses without averaging."""
    _plane(mesh, U, path, "tri6", gauss_order)


def quad4(
    mesh: Mesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Quad4 element-nodal stresses without averaging."""
    _plane(mesh, U, path, "quad4", gauss_order)


def quad8(
    mesh: Mesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Quad8 element-nodal stresses without averaging."""
    _plane(mesh, U, path, "quad8", gauss_order)


def hex8(mesh: Mesh3D, U: Sequence[float], path: str) -> None:
    """Export Hex8 centroid stresses to CSV."""
    _solid(mesh, U, path, "hex8", (0.0, 0.0, 0.0))


def hex20(mesh: Mesh3D, U: Sequence[float], path: str) -> None:
    """Export Hex20 centroid stress to CSV."""
    _solid(mesh, U, path, "hex20", (0.0, 0.0, 0.0))


def tet4(mesh: Mesh3D, U: Sequence[float], path: str) -> None:
    """Export Tet4 centroid stresses to CSV."""
    _solid(mesh, U, path, "tet4", TET_CENTROID)


def tet10(mesh: Mesh3D, U: Sequence[float], path: str) -> None:
    """Export Tet10 centroid stresses to CSV."""
    _solid(mesh, U, path, "tet10", TET_CENTROID)


def _plane(
    mesh: Mesh2D,
    U: Sequence[float],
    path: str,
    type_key: str,
    gauss_order: int | None = None,
) -> None:
    """Export plane element-nodal stresses without averaging."""
    recovered = collect_stress(
        mesh,
        U,
        position=StressPosition.ELEMENT_NODAL,
        element_type=type_key,
        gauss_order=gauss_order,
    )

    path = prepare_output_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLANE_ELEMENT_HEADER)
        _write_plane_records(writer, recovered, mesh)


def _solid(
    mesh: Mesh3D,
    U: Sequence[float],
    path: str,
    type_key: str,
    natural_coords: tuple[float, float, float],
) -> None:
    """Export solid element stresses at one legacy representative point.

    This path intentionally remains direct: distorted solid elements can
    produce different values when integration-point stresses are interpolated
    to the canonical centroid.
    """
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)

    path = prepare_output_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SOLID_HEADER)
        for elem in mesh.elements:
            if dispatch.type_key_from_name(elem.type) != type_key:
                continue
            stress = get_element_kernel(elem.type).stress_at(
                mesh,
                elem,
                U,
                *natural_coords,
                lookup,
            )
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stress
            writer.writerow([
                elem.id,
                sig_x,
                sig_y,
                sig_z,
                tau_xy,
                tau_yz,
                tau_zx,
                von_mises_3d(sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx),
            ])


def _plane_multi(
    mesh: Mesh2D,
    U: Sequence[float],
    path: str,
    type_keys: set[str],
    gauss_order: int | None = None,
) -> None:
    """Export mixed plane element-nodal stresses without averaging."""
    recovered = collect_stress(
        mesh,
        U,
        position=StressPosition.ELEMENT_NODAL,
        gauss_order=gauss_order,
    )

    path = prepare_output_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLANE_ELEMENT_HEADER)
        selected_element_ids = {
            int(elem.id)
            for elem in mesh.elements
            if dispatch.type_key_from_name(elem.type) in type_keys
        }
        _write_plane_records(writer, recovered, mesh, selected_element_ids)


def _solid_multi(
    mesh: Mesh3D,
    U: Sequence[float],
    path: str,
    type_keys: set[str],
) -> None:
    """Export mixed solid element stresses at one representative point per element.

    This remains a controlled legacy exception.  Direct representative-point
    evaluation is observably different from canonical integration-point
    interpolation at ``StressPosition.CENTROID`` for distorted solids, so this
    compatibility schema cannot delegate to that field without changing values.
    """
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)

    path = prepare_output_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SOLID_HEADER)
        for elem in mesh.elements:
            type_key = dispatch.type_key_from_name(elem.type)
            if type_key not in type_keys:
                continue
            natural_coords = (
                (0.0, 0.0, 0.0)
                if type_key in {"hex8", "hex20"}
                else TET_CENTROID
            )
            stress = get_element_kernel(elem.type).stress_at(
                mesh,
                elem,
                U,
                *natural_coords,
                lookup,
            )
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stress
            writer.writerow([
                elem.id,
                sig_x,
                sig_y,
                sig_z,
                tau_xy,
                tau_yz,
                tau_zx,
                von_mises_3d(sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx),
            ])


def _write_plane_records(
    writer,
    recovered: StressField,
    mesh,
    selected_element_ids: set[int] | None = None,
) -> None:
    """Project canonical element-nodal plane rows to the legacy CSV schema."""
    if recovered.position is not StressPosition.ELEMENT_NODAL:
        raise ValueError("plane element export requires element-nodal stress")
    if recovered.component_names != CANONICAL_PLANE_COMPONENT_NAMES:
        raise ValueError("plane element export requires canonical plane stress")

    elements_by_id = {int(elem.id): elem for elem in mesh.elements}
    for record in recovered.records:
        if (
            record.elem_id is None
            or record.node_id is None
            or record.local_node is None
        ):
            raise ValueError(
                "canonical plane element-nodal stress is missing row provenance"
            )
        if (
            selected_element_ids is not None
            and record.elem_id not in selected_element_ids
        ):
            continue
        sig_x, sig_y, _sig_z, tau_xy = record.components
        elem = elements_by_id[record.elem_id]
        plane_type = elem.props.get("plane_type")
        if plane_type is None:
            plane_type = (
                "strain"
                if str(elem.type).upper().startswith("CPE")
                else "stress"
            )
        writer.writerow([
            record.elem_id,
            record.node_id,
            record.local_node,
            sig_x,
            sig_y,
            tau_xy,
            # Preserve the legacy byte contract for plane strain.  Computing
            # S33 after element-nodal extrapolation can differ by a final bit
            # from extrapolating the canonical complete tensor.
            von_mises_plane(
                sig_x,
                sig_y,
                tau_xy,
                str(plane_type),
                float(elem.props["nu"]),
            ),
        ])
