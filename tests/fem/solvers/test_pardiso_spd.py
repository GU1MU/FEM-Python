from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from fem.solvers import _pardiso_spd


@pytest.fixture
def fake_backend(monkeypatch):
    class FakePardisoSolver:
        instances = []
        construction_error = None
        solve_error = None
        solve_result = None
        release_error = None

        def __init__(self, matrix, mtype):
            self.matrix = matrix
            self.mtype = mtype
            self.release_calls = 0
            type(self).instances.append(self)
            if type(self).construction_error is not None:
                raise type(self).construction_error

        def solve(self, rhs):
            self.last_rhs = rhs
            if type(self).solve_error is not None:
                raise type(self).solve_error
            if type(self).solve_result is not None:
                return type(self).solve_result
            upper = self.matrix.toarray()
            dense = upper + upper.T - np.diag(np.diag(upper))
            return np.linalg.solve(dense, rhs)

        def release(self):
            self.release_calls += 1
            if type(self).release_error is not None:
                raise type(self).release_error

    monkeypatch.setattr(
        _pardiso_spd,
        "_NativePardisoSolver",
        FakePardisoSolver,
    )
    yield FakePardisoSolver

    assert all(backend.release_calls == 1 for backend in FakePardisoSolver.instances)


def _unsorted_spd_upper() -> csr_matrix:
    return csr_matrix(
        (
            np.array([1.0, 4.0, 1.0, 3.0, 2.0], dtype=np.float32),
            np.array([1, 0, 2, 1, 2], dtype=np.int32),
            np.array([0, 2, 4, 5], dtype=np.int32),
        ),
        shape=(3, 3),
    )


def _spd_upper() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [4.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
    )


def _dense_spd() -> np.ndarray:
    return np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )


def test_factorize_normalizes_an_owned_sorted_float64_csr_and_solves_1d_rhs(
    fake_backend,
):
    matrix = _unsorted_spd_upper()
    original_data = matrix.data.copy()
    original_indices = matrix.indices.copy()
    rhs = np.array([1.0, 2.0, 3.0])

    factor = _pardiso_spd.factorize_spd(matrix)
    backend = fake_backend.instances[0]
    try:
        result = factor.solve(rhs)
    finally:
        factor.close()

    assert backend.mtype == 2
    assert isinstance(backend.matrix, csr_matrix)
    assert backend.matrix.dtype == np.dtype(np.float64)
    assert backend.matrix.has_sorted_indices
    assert backend.matrix.has_canonical_format
    for buffer in ("data", "indices", "indptr"):
        assert not np.shares_memory(
            getattr(backend.matrix, buffer), getattr(matrix, buffer),
        )
    assert backend.last_rhs.dtype == np.float64
    assert not np.shares_memory(backend.last_rhs, rhs)
    np.testing.assert_array_equal(matrix.data, original_data)
    np.testing.assert_array_equal(matrix.indices, original_indices)
    assert matrix.dtype == np.dtype(np.float32)
    assert not matrix.has_sorted_indices
    np.testing.assert_allclose(
        result,
        np.linalg.solve(_dense_spd(), rhs),
        rtol=1e-12,
        atol=1e-12,
    )


def test_factor_solve_owns_float64_multiple_rhs_and_matches_dense_solution(fake_backend):
    rhs = np.array(
        [
            [1.0, 4.0],
            [2.0, -1.0],
            [3.0, 0.5],
        ],
        dtype=np.float32,
        order="F",
    )
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    try:
        result = factor.solve(rhs)
    finally:
        factor.close()

    backend = fake_backend.instances[0]
    assert backend.last_rhs.dtype == np.float64
    assert not np.shares_memory(backend.last_rhs, rhs)
    assert result.shape == rhs.shape
    np.testing.assert_allclose(
        result,
        np.linalg.solve(_dense_spd(), rhs),
        rtol=1e-12,
        atol=1e-12,
    )


def _csr_with_nonfinite(value: float) -> csr_matrix:
    matrix = _spd_upper()
    matrix.data[0] = value
    return matrix


def _csr_with_diagonal(value: float) -> csr_matrix:
    matrix = _spd_upper()
    matrix.data[matrix.indptr[1]] = value
    return matrix


def _csr_with_duplicate() -> csr_matrix:
    return csr_matrix(
        (
            np.array([2.0, 2.0, 1.0, 3.0], dtype=np.float64),
            np.array([0, 0, 1, 1], dtype=np.int32),
            np.array([0, 3, 4], dtype=np.int32),
        ),
        shape=(2, 2),
    )


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (csc_matrix(np.eye(2)), "CSR format"),
        (csr_matrix(np.ones((2, 3))), "square 2D"),
        (_csr_with_nonfinite(np.nan), "finite"),
        (
            csr_matrix(np.array([[2.0, 1.0], [0.0, 0.0]])),
            "explicit diagonal at row 1",
        ),
        (_csr_with_diagonal(0.0), "positive at row 1"),
        (_csr_with_diagonal(-1.0), "positive at row 1"),
        (
            csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]])),
            "upper-triangular",
        ),
        (csr_matrix(np.eye(2, dtype=np.complex128)), "real numeric"),
    ],
    ids=(
        "non-csr", "nonsquare", "nonfinite", "missing-diagonal",
        "zero-diagonal", "negative-diagonal", "lower-entry", "complex",
    ),
)
def test_factorize_rejects_invalid_matrix_before_backend_construction(
    matrix, message, fake_backend,
):
    with pytest.raises(ValueError, match=message):
        _pardiso_spd.factorize_spd(matrix)

    assert fake_backend.instances == []


def test_factorize_combines_duplicate_entries_without_modifying_input(
    fake_backend,
):
    matrix = _csr_with_duplicate()
    original_data = matrix.data.copy()
    original_indices = matrix.indices.copy()

    factor = _pardiso_spd.factorize_spd(matrix)
    backend = fake_backend.instances[0]
    try:
        result = factor.solve(np.array([1.0, 2.0]))
    finally:
        factor.close()

    assert backend.matrix.has_canonical_format
    np.testing.assert_array_equal(matrix.data, original_data)
    np.testing.assert_array_equal(matrix.indices, original_indices)
    np.testing.assert_allclose(
        result,
        np.linalg.solve(np.array([[4.0, 1.0], [1.0, 3.0]]), [1.0, 2.0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_factorize_discards_explicit_zero_below_the_diagonal(fake_backend):
    matrix = csr_matrix(
        (
            np.array([2.0, 1.0, 0.0, 2.0]),
            np.array([0, 1, 0, 1]),
            np.array([0, 2, 4]),
        ),
        shape=(2, 2),
    )

    factor = _pardiso_spd.factorize_spd(matrix)
    backend = fake_backend.instances[0]
    try:
        result = factor.solve(np.array([1.0, 2.0]))
    finally:
        factor.close()

    assert backend.matrix.nnz == 3
    np.testing.assert_allclose(
        result,
        np.linalg.solve(np.array([[2.0, 1.0], [1.0, 2.0]]), [1.0, 2.0]),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("rhs", "message"),
    [
        (np.ones(2), "must have shape"),
        (np.ones((3, 1, 1)), "must have shape"),
        (np.array([1.0, np.nan, 3.0]), "finite"),
        (np.ones(3, dtype=np.complex128), "real numeric"),
    ],
    ids=("wrong-length", "wrong-rank", "nonfinite", "complex"),
)
def test_factor_solve_rejects_invalid_rhs(rhs, message, fake_backend):
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    try:
        with pytest.raises(ValueError, match=message):
            factor.solve(rhs)
    finally:
        factor.close()


@pytest.mark.parametrize(
    ("stage", "native_error", "expected_type"),
    [
        (
            "factorization",
            MemoryError("native allocation failed"),
            _pardiso_spd._PardisoSPDMemoryError,
        ),
        (
            "factorization",
            RuntimeError("not enough memory for factorization"),
            _pardiso_spd._PardisoSPDMemoryError,
        ),
        (
            "factorization",
            RuntimeError("insufficient memory in PARDISO"),
            _pardiso_spd._PardisoSPDMemoryError,
        ),
        (
            "factorization",
            RuntimeError("matrix is not positive definite"),
            _pardiso_spd._PardisoSPDError,
        ),
        (
            "solve",
            MemoryError("native allocation failed"),
            _pardiso_spd._PardisoSPDMemoryError,
        ),
        (
            "solve",
            RuntimeError("zero pivot"),
            _pardiso_spd._PardisoSPDError,
        ),
    ],
    ids=(
        "factor-memory-error",
        "factor-not-enough-memory",
        "factor-insufficient-memory",
        "factor-generic-error",
        "solve-memory-error",
        "solve-generic-error",
    ),
)
def test_native_failure_preserves_error_category_cause_and_releases_backend(
    fake_backend, stage, native_error, expected_type,
):
    message = f"PARDISO SPD {stage} failed"
    if expected_type is _pardiso_spd._PardisoSPDMemoryError:
        message += ": insufficient memory"
    if stage == "factorization":
        fake_backend.construction_error = native_error
        with pytest.raises(expected_type, match=message) as caught:
            _pardiso_spd.factorize_spd(_spd_upper())
    else:
        fake_backend.solve_error = native_error
        factor = _pardiso_spd.factorize_spd(_spd_upper())
        try:
            with pytest.raises(expected_type, match=message) as caught:
                factor.solve(np.ones(3))
        finally:
            factor.close()

    assert type(caught.value) is expected_type
    assert caught.value.__cause__ is native_error
    assert len(fake_backend.instances) == 1
    assert fake_backend.instances[0].release_calls == 1


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (np.zeros(2), "invalid result shape"),
        (np.array([1.0, np.nan, 3.0]), "non-finite"),
    ],
    ids=("wrong-shape", "nonfinite"),
)
def test_factor_rejects_invalid_backend_result(result, message, fake_backend):
    fake_backend.solve_result = result
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    try:
        with pytest.raises(_pardiso_spd._PardisoSPDError, match=message):
            factor.solve(np.ones(3))
    finally:
        factor.close()


def test_close_is_idempotent_and_solve_after_close_fails(fake_backend):
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    backend = fake_backend.instances[0]

    factor.close()
    factor.close()

    assert backend.release_calls == 1
    with pytest.raises(
        _pardiso_spd._PardisoSPDError,
        match="PARDISO SPD factor is closed",
    ):
        factor.solve(np.ones(3))
    assert backend.release_calls == 1


def test_factor_finalizer_releases_an_open_backend_once(fake_backend):
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    backend = fake_backend.instances[0]
    factor_reference = weakref.ref(factor)

    del factor
    gc.collect()

    assert factor_reference() is None
    assert backend.release_calls == 1


def test_release_failure_is_chained_and_is_not_retried(fake_backend):
    native_error = RuntimeError("native release failed")
    fake_backend.release_error = native_error
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    backend = fake_backend.instances[0]

    with pytest.raises(
        _pardiso_spd._PardisoSPDError,
        match="PARDISO SPD release failed",
    ) as caught:
        factor.close()
    factor.close()

    assert caught.value.__cause__ is native_error
    assert backend.release_calls == 1


@pytest.mark.optional_runtime
def test_pardiso_native_runtime_solves_vector_and_multiple_rhs():
    matrix = _spd_upper()
    vector_rhs = np.array([1.0, 2.0, 3.0])
    matrix_rhs = np.column_stack((vector_rhs, np.array([4.0, -1.0, 0.5])))
    factor = _pardiso_spd.factorize_spd(matrix)
    try:
        vector_result = factor.solve(vector_rhs)
        matrix_result = factor.solve(matrix_rhs)
    finally:
        factor.close()

    np.testing.assert_allclose(
        vector_result,
        np.linalg.solve(_dense_spd(), vector_rhs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        matrix_result,
        np.linalg.solve(_dense_spd(), matrix_rhs),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.optional_runtime
def test_pardiso_native_runtime_rejects_an_indefinite_matrix_at_factorization():
    indefinite_upper = csr_matrix(
        np.array(
            [
                [1.0, 2.0],
                [0.0, 1.0],
            ]
        )
    )

    factor = None
    try:
        with pytest.raises(
            _pardiso_spd._PardisoSPDError,
            match="PARDISO SPD factorization failed",
        ) as caught:
            factor = _pardiso_spd.factorize_spd(indefinite_upper)
    finally:
        if factor is not None:
            factor.close()

    assert caught.value.__cause__ is not None
