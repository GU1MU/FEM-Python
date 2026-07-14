import csv
from typing import List, Optional, Sequence

import numpy as np

from ...core.mesh import Mesh2D, Mesh3D, Node2D, Node3D
from .._paths import prepare_output_path


def _export_nodal_displacement_2d(
    mesh: Mesh2D,
    U: Sequence[float],
    path: str,
    component_names: Optional[List[str]] = None,
) -> None:
    """Export nodal displacements to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    dofs_per_node = mesh.dofs_per_node

    if component_names is None:
        if dofs_per_node == 2:
            component_names = ["ux", "uy"]
        elif dofs_per_node == 3:
            component_names = ["ux", "uy", "rz"]
        else:
            component_names = [f"u{c}" for c in range(dofs_per_node)]
    else:
        if len(component_names) != dofs_per_node:
            raise ValueError(
                f"component_names length {len(component_names)} != dofs_per_node={dofs_per_node}"
            )

    node_lookup = {node.id: node for node in mesh.nodes}
    header = ["node_id", "x", "y"] + component_names

    path = prepare_output_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for node_id in mesh.node_ids:
            node: Node2D = node_lookup[node_id]
            dofs = mesh.node_dofs(node_id)
            disp_vals = [U[dof] for dof in dofs]
            writer.writerow([node_id, node.x, node.y] + disp_vals)


def _export_nodal_displacement_3d(
    mesh: Mesh3D,
    U: Sequence[float],
    path: str,
    component_names: Optional[List[str]] = None,
) -> None:
    """Export 3D nodal displacements to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    dofs_per_node = mesh.dofs_per_node

    if component_names is None:
        if dofs_per_node == 3:
            component_names = ["ux", "uy", "uz"]
        elif dofs_per_node == 6:
            component_names = ["ux", "uy", "uz", "rx", "ry", "rz"]
        else:
            component_names = [f"u{c}" for c in range(dofs_per_node)]
    else:
        if len(component_names) != dofs_per_node:
            raise ValueError(
                f"component_names length {len(component_names)} != dofs_per_node={dofs_per_node}"
            )

    node_lookup = {node.id: node for node in mesh.nodes}
    header = ["node_id", "x", "y", "z"] + component_names

    path = prepare_output_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for node_id in mesh.node_ids:
            node: Node3D = node_lookup[node_id]
            dofs = mesh.node_dofs(node_id)
            disp_vals = [U[dof] for dof in dofs]
            writer.writerow([node_id, node.x, node.y, node.z] + disp_vals)


def nodal(
    mesh,
    U: Sequence[float],
    path: str,
    component_names: Optional[List[str]] = None,
) -> None:
    """Export nodal displacements to CSV. Node coordinates define 2D or 3D output."""
    if mesh.nodes and hasattr(mesh.nodes[0], "z"):
        _export_nodal_displacement_3d(mesh, U, path, component_names)
    else:
        _export_nodal_displacement_2d(mesh, U, path, component_names)
