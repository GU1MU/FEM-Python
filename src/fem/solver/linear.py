from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, splu, spsolve


class SparseFactorization:
    """Reusable sparse LU factorization for repeated right-hand sides.

    ``spsolve`` is intentionally retained for the one-shot public ``solve``
    function.  Newton's modified strategy can hold this object across
    iterations, avoiding a second numeric factorization when the tangent
    matrix is intentionally unchanged.
    """

    __slots__ = ("_factor", "_matrix", "_closed")

    def __init__(self, factor: object, matrix: csr_matrix) -> None:
        self._factor = factor
        self._matrix = matrix
        self._closed = False

    def solve(self, F: np.ndarray) -> np.ndarray:
        """Solve one validated right-hand side and retain residual checks."""

        if self._closed:
            raise RuntimeError("sparse factorization is closed")
        rhs = _validated_rhs(self._matrix, F)
        try:
            U = np.asarray(self._factor.solve(rhs), dtype=float)  # type: ignore[union-attr]
            _validate_solution(self._matrix, rhs, U)
            return U
        except Exception as exc:
            raise RuntimeError(
                f"sparse linear solve failed: {exc}. "
                "The stiffness matrix may be singular or under-constrained."
            ) from exc

    def close(self) -> None:
        """Release the native factorization as soon as the Newton solve ends."""

        if self._closed:
            return
        self._closed = True
        self._factor = None
        self._matrix = None  # type: ignore[assignment]

    def __del__(self) -> None:
        self.close()


def factorize(K: csr_matrix) -> SparseFactorization:
    """Factor one CSR matrix for repeated solves without changing its values."""

    _validate_matrix(K)
    try:
        factor = splu(K.tocsc())
    except Exception as exc:
        raise RuntimeError(
            f"sparse linear factorization failed: {exc}. "
            "The stiffness matrix may be singular or under-constrained."
        ) from exc
    return SparseFactorization(factor, K)


def solve(K: csr_matrix, F: np.ndarray) -> np.ndarray:
    """Solve sparse linear system K @ U = F."""
    _validate_matrix(K)
    F = _validated_rhs(K, F)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            U = np.asarray(spsolve(K, F), dtype=float)
        _validate_solution(K, F, U)
        return U
    except MatrixRankWarning as exc:
        raise RuntimeError(
            "sparse linear solve failed: stiffness matrix is singular or under-constrained."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"sparse linear solve failed: {exc}. "
            "The stiffness matrix may be singular or under-constrained."
        ) from exc


def _validate_solution(K: csr_matrix, F: np.ndarray, U: np.ndarray) -> None:
    """Reject invalid sparse solver output."""
    if U.ndim != 1 or U.shape[0] != F.shape[0]:
        raise RuntimeError(f"sparse linear solve returned invalid shape {U.shape}")
    if not np.all(np.isfinite(U)):
        raise RuntimeError("sparse linear solve returned non-finite values")

    residual = K @ U - F
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))
    solution_scale = float(np.linalg.norm(K @ U, ord=np.inf))
    load_scale = float(np.linalg.norm(F, ord=np.inf))
    scale = max(solution_scale, load_scale, 1.0)
    if residual_norm > 1e-8 * scale:
        raise RuntimeError(
            f"sparse linear solve residual {residual_norm:g} exceeds tolerance"
        )


def _validate_matrix(K: csr_matrix) -> None:
    """Validate a CSR system before one-shot or reusable factorization."""

    if not isinstance(K, csr_matrix):
        raise TypeError(f"K must be csr_matrix, got {type(K)}")
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got {K.shape}")
    if not np.all(np.isfinite(K.data)):
        raise ValueError("K must contain only finite values")


def _validated_rhs(K: csr_matrix, F: np.ndarray) -> np.ndarray:
    """Normalize and validate one sparse-system right-hand side."""

    n = K.shape[0]
    F = np.asarray(F, dtype=float)
    if F.ndim == 2 and F.shape[1] == 1:
        F = F.ravel()
    if F.ndim != 1 or F.shape[0] != n:
        raise ValueError(f"F must have length {n}, got {F.shape}")
    if not np.all(np.isfinite(F)):
        raise ValueError("F must contain only finite values")
    return F
