from __future__ import annotations

import csv
from typing import Sequence

from .._paths import prepare_output_path
from . import dispatch
from ._common import (
    PLANE_NODAL_HEADER,
    SOLID_NODAL_HEADER,
    node_lookup,
)
from .field import ResolvedNodalStressField, collect, resolve
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
    if type_key == "truss2d":
        raise ValueError("Nodal stress export is not available for Truss2D elements")
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
    raw = collect(mesh, U, element_type=element_type, gauss_order=gauss_order)
    resolved = resolve(raw, threshold=threshold)
    _write_resolved(mesh, resolved, path)


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
