from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve


def solve(K: csr_matrix, F: np.ndarray) -> np.ndarray:
    """Solve sparse linear system K @ U = F."""
    if not isinstance(K, csr_matrix):
        raise TypeError(f"K must be csr_matrix, got {type(K)}")
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got {K.shape}")
    if not np.all(np.isfinite(K.data)):
        raise ValueError("K must contain only finite values")

    n = K.shape[0]
    F = np.asarray(F, dtype=float)
    if F.ndim == 2 and F.shape[1] == 1:
        F = F.ravel()
    if F.ndim != 1 or F.shape[0] != n:
        raise ValueError(f"F must have length {n}, got {F.shape}")
    if not np.all(np.isfinite(F)):
        raise ValueError("F must contain only finite values")

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
