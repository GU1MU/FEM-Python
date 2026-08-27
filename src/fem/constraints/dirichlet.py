"""Dirichlet equation enforcement kept outside mathematical Problems."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


def apply_dirichlet_equations(
    tangent: Any,
    residual: np.ndarray,
    solution: np.ndarray,
    constraints: Any,
) -> csr_matrix:
    """Replace prescribed DOFs by identity equations for Newton."""

    constrained = getattr(constraints, "prescribed_values", None)
    if constrained is None:
        raise ValueError("problem constraints must provide prescribed_values")
    matrix = csr_matrix(tangent, copy=True)
    constrained_dofs = np.asarray(tuple(int(dof) for dof in constrained), dtype=int)
    if constrained_dofs.size == 0:
        return matrix
    if np.any(constrained_dofs < 0) or np.any(
        constrained_dofs >= residual.size
    ):
        invalid_mask = (constrained_dofs < 0) | (
            constrained_dofs >= residual.size
        )
        invalid = int(constrained_dofs[np.flatnonzero(invalid_mask)[0]])
        raise IndexError(f"constraint DOF {invalid} is out of bounds")

    # The compiled assembly owns a stable CSR pattern.  Zeroing constrained
    # columns directly in its data avoids the costly CSR -> LIL -> CSR
    # conversion that used to run for every Newton evaluation.  Keep the
    # general LIL implementation as a fallback for externally supplied
    # matrices that do not carry all constrained diagonal entries.
    matrix.sum_duplicates()
    if matrix.has_sorted_indices and _has_all_diagonal_entries(
        matrix,
        constrained_dofs,
    ):
        matrix.data[np.isin(matrix.indices, constrained_dofs)] = 0.0
        for dof in constrained_dofs:
            start = int(matrix.indptr[dof])
            stop = int(matrix.indptr[dof + 1])
            row_indices = matrix.indices[start:stop]
            diagonal = int(np.searchsorted(row_indices, dof))
            matrix.data[start:stop] = 0.0
            matrix.data[start + diagonal] = 1.0
        residual[constrained_dofs] = np.asarray(
            solution[constrained_dofs],
            dtype=float,
        ) - np.asarray(
            [float(constrained[dof]) for dof in constrained_dofs],
            dtype=float,
        )
        return matrix

    legacy = matrix.tolil(copy=False)
    for dof in constrained_dofs:
        matrix_column = int(dof)
        legacy[:, matrix_column] = 0.0
        legacy.rows[matrix_column] = [matrix_column]
        legacy.data[matrix_column] = [1.0]
        residual[matrix_column] = solution[matrix_column] - float(
            constrained[matrix_column]
        )
    return legacy.tocsr()


def _has_all_diagonal_entries(
    matrix: csr_matrix,
    constrained_dofs: np.ndarray,
) -> bool:
    """Return whether the fast CSR path can set every constrained diagonal."""

    for dof in constrained_dofs:
        start = int(matrix.indptr[dof])
        stop = int(matrix.indptr[dof + 1])
        row_indices = matrix.indices[start:stop]
        position = int(np.searchsorted(row_indices, dof))
        if position >= row_indices.size or int(row_indices[position]) != int(dof):
            return False
    return True


__all__ = ["apply_dirichlet_equations"]
