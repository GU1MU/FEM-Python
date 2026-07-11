from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..core.validation import validate_mesh
from ..elements import get_element_kernel


def assemble_global_stiffness(mesh: Any) -> np.ndarray:
    """Assemble a dense global stiffness matrix from a mesh."""
    _validate_mesh(mesh)
    node_lookup = {node.id: node for node in mesh.nodes}
    K = np.zeros((mesh.num_dofs, mesh.num_dofs), dtype=float)

    for elem in mesh.elements:
        Ke = get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup=node_lookup)
        dofs = list(mesh.element_dofs(elem))
        Ke = _validate_element_stiffness(Ke, dofs, mesh.num_dofs, elem)

        for local_i, global_i in enumerate(dofs):
            for local_j, global_j in enumerate(dofs):
                K[global_i, global_j] += Ke[local_i, local_j]

    return K


def assemble_global_stiffness_sparse(mesh: Any) -> csr_matrix:
    """Assemble a sparse global stiffness matrix from a mesh."""
    _validate_mesh(mesh)
    node_lookup = {node.id: node for node in mesh.nodes}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for elem in mesh.elements:
        Ke = get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup=node_lookup)
        dofs = list(mesh.element_dofs(elem))
        Ke = _validate_element_stiffness(Ke, dofs, mesh.num_dofs, elem)

        for local_i, global_i in enumerate(dofs):
            for local_j, global_j in enumerate(dofs):
                rows.append(global_i)
                cols.append(global_j)
                data.append(float(Ke[local_i, local_j]))

    return coo_matrix((data, (rows, cols)), shape=(mesh.num_dofs, mesh.num_dofs)).tocsr()


def _validate_mesh(mesh: Any) -> None:
    """Validate the mesh interface required for stiffness assembly."""
    validate_mesh(mesh)


def _validate_element_stiffness(
    Ke: np.ndarray,
    dofs: Sequence[int],
    num_dofs: int,
    elem_label: object,
) -> np.ndarray:
    """Validate element stiffness shape and DOF bounds."""
    Ke = np.asarray(Ke, dtype=float)
    nd = len(dofs)
    if Ke.shape != (nd, nd):
        raise ValueError(
            f"element {elem_label} stiffness shape {Ke.shape} does not match {nd} DOFs"
        )

    for dof in dofs:
        if dof < 0 or dof >= num_dofs:
            raise IndexError(
                f"element {elem_label} DOF index {dof} out of bounds [0, {num_dofs})"
            )

    if not np.all(np.isfinite(Ke)):
        raise ValueError(f"element {elem_label} stiffness contains non-finite values")
    if not np.allclose(Ke, Ke.T, rtol=1e-8, atol=1e-10):
        asymmetry = float(np.max(np.abs(Ke - Ke.T)))
        raise ValueError(
            f"element {elem_label} stiffness is not symmetric; "
            f"maximum asymmetry is {asymmetry:g}"
        )
    return Ke
