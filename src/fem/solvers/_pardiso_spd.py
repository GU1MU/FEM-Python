from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pymklpardiso import (
    MTYPE_REAL_SYM_POSDEF as _MTYPE_REAL_SYM_POSDEF,
)
from pymklpardiso import PardisoSolver as _NativePardisoSolver


class _PardisoSPDError(RuntimeError):
    """Stable internal error raised by the PARDISO SPD backend."""


class PardisoSPDFactor:
    """Owned wrapper around one factorized PARDISO SPD handle."""

    def __init__(self, solver: Any, dimension: int) -> None:
        self._solver = solver
        self._dimension = dimension
        self._closed = False

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self._closed:
            raise _PardisoSPDError("PARDISO SPD factor is closed")

        prepared_rhs = _prepare_rhs(rhs, self._dimension)
        try:
            result = np.asarray(self._solver.solve(prepared_rhs), dtype=np.float64)
        except Exception as exc:
            raise _PardisoSPDError("PARDISO SPD solve failed") from exc

        if result.shape != prepared_rhs.shape:
            raise _PardisoSPDError(
                "PARDISO SPD solve returned an invalid result shape "
                f"{result.shape}; expected {prepared_rhs.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise _PardisoSPDError(
                "PARDISO SPD solve returned non-finite values"
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        solver = self._solver
        self._solver = None
        try:
            solver.release()
        except Exception as exc:
            raise _PardisoSPDError("PARDISO SPD release failed") from exc

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def factorize_spd(matrix: sparse.spmatrix) -> PardisoSPDFactor:
    """Validate and factor an upper-triangular sparse SPD matrix."""

    prepared = _prepare_upper_csr(matrix)
    solver = _construct_solver(prepared)
    return PardisoSPDFactor(solver, prepared.shape[0])


def _construct_solver(matrix: sparse.spmatrix) -> Any:
    # Calling ``__new__`` separately retains the partly initialized wrapper when
    # native symbolic analysis or factorization raises from ``__init__``.
    solver = None
    try:
        solver = _NativePardisoSolver.__new__(_NativePardisoSolver)
        _NativePardisoSolver.__init__(
            solver,
            matrix,
            _MTYPE_REAL_SYM_POSDEF,
        )
    except Exception as exc:
        if solver is not None:
            try:
                solver.release()
            except Exception:
                pass
        raise _PardisoSPDError("PARDISO SPD factorization failed") from exc
    return solver


def _prepare_upper_csr(matrix: sparse.spmatrix) -> sparse.spmatrix:
    if not sparse.issparse(matrix) or getattr(matrix, "format", None) != "csr":
        raise ValueError("PARDISO SPD matrix must be in CSR format")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"PARDISO SPD matrix must be a square 2D matrix, got {matrix.shape}"
        )

    try:
        matrix.check_format(full_check=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("PARDISO SPD matrix has an invalid CSR structure") from exc

    data = np.asarray(matrix.data)
    if np.iscomplexobj(data) or not np.issubdtype(data.dtype, np.number):
        raise ValueError("PARDISO SPD matrix must contain real numeric values")
    if not np.all(np.isfinite(data)):
        raise ValueError("PARDISO SPD matrix must contain only finite values")

    prepared = matrix.astype(np.float64, copy=True)
    prepared.sum_duplicates()
    prepared.sort_indices()
    if not np.all(np.isfinite(prepared.data)):
        raise ValueError("PARDISO SPD matrix must contain only finite values")

    indptr = np.asarray(prepared.indptr)
    indices = np.asarray(prepared.indices)
    data = np.asarray(prepared.data)
    for row in range(prepared.shape[0]):
        row_indices = indices[indptr[row] : indptr[row + 1]]
        row_data = data[indptr[row] : indptr[row + 1]]
        if np.any((row_indices < row) & (row_data != 0.0)):
            raise ValueError(
                "PARDISO SPD matrix must contain only upper-triangular entries"
            )

        diagonal = row_data[row_indices == row]
        if diagonal.size != 1:
            raise ValueError(
                f"PARDISO SPD matrix must contain an explicit diagonal at row {row}"
            )
        if diagonal[0] <= 0.0:
            raise ValueError(
                f"PARDISO SPD matrix diagonal must be positive at row {row}"
            )

    prepared.eliminate_zeros()
    return prepared


def _prepare_rhs(rhs: np.ndarray, dimension: int) -> np.ndarray:
    raw = np.asarray(rhs)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("PARDISO SPD right-hand side must be real numeric data")
    try:
        prepared = np.array(rhs, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "PARDISO SPD right-hand side must be real numeric data"
        ) from exc

    if prepared.ndim not in (1, 2) or prepared.shape[0] != dimension:
        raise ValueError(
            "PARDISO SPD right-hand side must have shape "
            f"({dimension},) or ({dimension}, nrhs), got {prepared.shape}"
        )
    if not np.all(np.isfinite(prepared)):
        raise ValueError(
            "PARDISO SPD right-hand side must contain only finite values"
        )
    return prepared
