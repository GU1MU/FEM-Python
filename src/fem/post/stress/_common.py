from __future__ import annotations

import csv
from typing import Any, Sequence

import numpy as np

from ...elements import get_element_kernel
from ...core.mesh import Mesh3DProtocol, Node3D


TET_CENTROID = (0.25, 0.25, 0.25)
SOLID_HEADER = ["elem_id", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]
SOLID_NODAL_HEADER = [
    "node_id", "x", "y", "z",
    "elem_id", "local_node", "averaged",
    "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises",
]
PLANE_ELEMENT_HEADER = ["elem_id", "node_id", "local_node", "sig_x", "sig_y", "tau_xy", "mises"]
PLANE_NODAL_HEADER = [
    "node_id", "x", "y", "elem_id", "local_node", "averaged",
    "sig_x", "sig_y", "tau_xy", "mises",
]


def validated_u(mesh: Any, U: Sequence[float]) -> np.ndarray:
    """Validate and flatten a global displacement vector."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")
    return U


def node_lookup(mesh: Any) -> dict[int, Any]:
    """Return node lookup keyed by node id."""
    return {node.id: node for node in mesh.nodes}


def matches(elem: Any, type_key: str) -> bool:
    """Return whether an element type matches a stress exporter key."""
    return type_key in str(elem.type).lower()


def nodal_stress(
    mesh: Any,
    elem: Any,
    U: np.ndarray,
    node_lookup_: dict[int, Any],
    gauss_order: int | None,
):
    """Return element nodal stress through the element kernel."""
    kernel = get_element_kernel(elem.type)
    if gauss_order is None:
        return kernel.nodal_stress(mesh, elem, U, node_lookup_)
    return kernel.nodal_stress(mesh, elem, U, node_lookup_, gauss_order)


def element_volume(
    mesh: Mesh3DProtocol,
    elem: Any,
    node_lookup_: dict[int, Any],
) -> float:
    """Return element volume through the element kernel."""
    return float(get_element_kernel(elem.type).volume(mesh, elem, node_lookup_))


def write_zero_solid_node(writer: csv.writer, nid: int, node: Node3D) -> None:
    """Write a zero stress row for an unconnected solid node."""
    writer.writerow([nid, node.x, node.y, node.z, 0, 0, 0, 0, 0, 0, 0])
