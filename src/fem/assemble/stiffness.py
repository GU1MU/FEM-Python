from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..core.validation import validate_mesh
from ..elements import get_element_kernel


@dataclass(frozen=True, slots=True)
class _AssemblyPlan:
    """Flat sparse-assembly storage shared by every element contribution."""

    dof_offsets: np.ndarray
    entry_offsets: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


def assemble_global_stiffness(
    mesh: Any,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Assemble a dense global stiffness matrix from a mesh."""
    _validate_mesh(mesh)
    _validate_strict(strict)
    node_lookup = {node.id: node for node in mesh.nodes}
    K = np.zeros((mesh.num_dofs, mesh.num_dofs), dtype=float)

    for elem in mesh.elements:
        Ke = get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup=node_lookup)
        dofs = _validated_element_dofs(mesh, elem)
        Ke = _validate_element_stiffness(
            Ke,
            len(dofs),
            elem,
            strict=strict,
        )

        for local_i, global_i in enumerate(dofs):
            for local_j, global_j in enumerate(dofs):
                K[global_i, global_j] += Ke[local_i, local_j]

    return K


def assemble_global_stiffness_sparse(
    mesh: Any,
    *,
    strict: bool = True,
) -> csr_matrix:
    """Assemble a sparse global stiffness matrix from a mesh."""
    _validate_mesh(mesh)
    _validate_strict(strict)
    node_lookup = {node.id: node for node in mesh.nodes}
    plan = _build_assembly_plan(mesh)
    data = np.empty(plan.rows.size, dtype=float)

    for element_index, elem in enumerate(mesh.elements):
        Ke = get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup=node_lookup)
        dof_count = int(
            plan.dof_offsets[element_index + 1]
            - plan.dof_offsets[element_index]
        )
        Ke = _validate_element_stiffness(
            Ke,
            dof_count,
            elem,
            strict=strict,
        )
        entry_start = int(plan.entry_offsets[element_index])
        entry_stop = int(plan.entry_offsets[element_index + 1])
        data[entry_start:entry_stop] = Ke.reshape(-1)

    return coo_matrix(
        (data, (plan.rows, plan.cols)),
        shape=(mesh.num_dofs, mesh.num_dofs),
    ).tocsr()


def _validate_mesh(mesh: Any) -> None:
    """Validate the mesh interface required for stiffness assembly."""
    validate_mesh(mesh)


def _validate_element_stiffness(
    Ke: np.ndarray,
    dof_count: int,
    elem_label: object,
    *,
    strict: bool,
) -> np.ndarray:
    """Validate one kernel result, with symmetry optional only by request."""
    Ke = np.asarray(Ke, dtype=float)
    if Ke.shape != (dof_count, dof_count):
        raise ValueError(
            f"element {elem_label} stiffness shape {Ke.shape} "
            f"does not match {dof_count} DOFs"
        )

    if not np.all(np.isfinite(Ke)):
        raise ValueError(f"element {elem_label} stiffness contains non-finite values")
    if strict and not np.allclose(Ke, Ke.T, rtol=1e-8, atol=1e-10):
        asymmetry = float(np.max(np.abs(Ke - Ke.T)))
        raise ValueError(
            f"element {elem_label} stiffness is not symmetric; "
            f"maximum asymmetry is {asymmetry:g}"
        )
    return Ke


def _build_assembly_plan(mesh: Any) -> _AssemblyPlan:
    """Precompute flat DOF and COO indices before the kernel hot loop."""

    element_count = len(mesh.elements)
    dof_offsets = np.empty(element_count + 1, dtype=np.int64)
    entry_offsets = np.empty(element_count + 1, dtype=np.int64)
    dof_offsets[0] = 0
    entry_offsets[0] = 0

    total_dofs = 0
    total_entries = 0
    for element_index, elem in enumerate(mesh.elements):
        dof_count = len(tuple(mesh.element_dofs(elem)))
        total_dofs += dof_count
        total_entries += dof_count * dof_count
        dof_offsets[element_index + 1] = total_dofs
        entry_offsets[element_index + 1] = total_entries

    element_dofs = np.empty(total_dofs, dtype=np.int64)
    rows = np.empty(total_entries, dtype=np.int64)
    cols = np.empty(total_entries, dtype=np.int64)

    for element_index, elem in enumerate(mesh.elements):
        raw_dofs = tuple(mesh.element_dofs(elem))
        dof_start = int(dof_offsets[element_index])
        dof_stop = int(dof_offsets[element_index + 1])
        dof_count = dof_stop - dof_start
        if len(raw_dofs) != dof_count:
            raise ValueError(
                f"element {elem} DOF mapping changed while building assembly plan"
            )
        dof_slice = element_dofs[dof_start:dof_stop]
        for local_index, raw_dof in enumerate(raw_dofs):
            dof_slice[local_index] = _validated_dof_index(
                raw_dof,
                mesh.num_dofs,
                elem,
            )

        entry_start = int(entry_offsets[element_index])
        entry_stop = int(entry_offsets[element_index + 1])
        rows[entry_start:entry_stop].reshape(dof_count, dof_count)[:] = (
            dof_slice[:, None]
        )
        cols[entry_start:entry_stop].reshape(dof_count, dof_count)[:] = (
            dof_slice[None, :]
        )

    return _AssemblyPlan(
        dof_offsets=dof_offsets,
        entry_offsets=entry_offsets,
        rows=rows,
        cols=cols,
    )


def _validated_element_dofs(mesh: Any, elem: Any) -> list[int]:
    """Return one checked DOF map for dense assembly."""

    return [
        _validated_dof_index(raw_dof, mesh.num_dofs, elem)
        for raw_dof in mesh.element_dofs(elem)
    ]


def _validated_dof_index(
    raw_dof: Any,
    num_dofs: int,
    elem_label: object,
) -> int:
    """Require one exact integer DOF index within global bounds."""

    if isinstance(raw_dof, bool):
        raise TypeError(
            f"element {elem_label} DOF index must be an integer, got {raw_dof!r}"
        )
    try:
        dof = int(operator.index(raw_dof))
    except TypeError as exc:
        raise TypeError(
            f"element {elem_label} DOF index must be an integer, got {raw_dof!r}"
        ) from exc
    if dof < 0 or dof >= num_dofs:
        raise IndexError(
            f"element {elem_label} DOF index {dof} "
            f"out of bounds [0, {num_dofs})"
        )
    return dof


def _validate_strict(strict: bool) -> None:
    """Require callers to opt out of symmetry checks explicitly."""

    if type(strict) is not bool:
        raise TypeError("strict must be bool")
