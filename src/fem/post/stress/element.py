from __future__ import annotations

import csv
from typing import Sequence

from ...elements import get_element_kernel
from ...core.mesh import Mesh2D, Mesh3D
from .._paths import prepare_output_path
from . import dispatch
from ._common import (
    PLANE_ELEMENT_HEADER,
    SOLID_HEADER,
    TET_CENTROID,
    matches,
    nodal_stress,
    node_lookup,
    validated_u,
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
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)

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
        for elem in mesh.elements:
            ni_id, nj_id = elem.node_ids
            axial_strain, axial_stress, mises = get_element_kernel(
                elem.type
            ).element_stress(mesh, elem, U, lookup)
            writer.writerow([
                elem.id,
                ni_id,
                nj_id,
                axial_strain,
                axial_stress,
                mises,
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
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)

    path = prepare_output_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLANE_ELEMENT_HEADER)
        for elem in mesh.elements:
            if not matches(elem, type_key):
                continue
            node_vals, plane_type, nu = nodal_stress(mesh, elem, U, lookup, gauss_order)
            for local_node, node_id in enumerate(elem.node_ids, start=1):
                sig_x, sig_y, tau_xy = node_vals[local_node - 1].tolist()
                writer.writerow([
                    elem.id,
                    node_id,
                    local_node,
                    sig_x,
                    sig_y,
                    tau_xy,
                    von_mises_plane(sig_x, sig_y, tau_xy, plane_type, nu),
                ])


def _solid(
    mesh: Mesh3D,
    U: Sequence[float],
    path: str,
    type_key: str,
    natural_coords: tuple[float, float, float],
) -> None:
    """Export solid element stresses at one natural coordinate point."""
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
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)

    path = prepare_output_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLANE_ELEMENT_HEADER)
        for elem in mesh.elements:
            type_key = dispatch.type_key_from_name(elem.type)
            if type_key not in type_keys:
                continue
            order = (
                gauss_order
                if gauss_order is not None
                else dispatch.default_gauss_order(type_key)
            )
            node_vals, plane_type, nu = nodal_stress(mesh, elem, U, lookup, order)
            for local_node, node_id in enumerate(elem.node_ids, start=1):
                sig_x, sig_y, tau_xy = node_vals[local_node - 1].tolist()
                writer.writerow([
                    elem.id,
                    node_id,
                    local_node,
                    sig_x,
                    sig_y,
                    tau_xy,
                    von_mises_plane(sig_x, sig_y, tau_xy, plane_type, nu),
                ])


def _solid_multi(
    mesh: Mesh3D,
    U: Sequence[float],
    path: str,
    type_keys: set[str],
) -> None:
    """Export mixed solid element stresses at one representative point per element."""
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
