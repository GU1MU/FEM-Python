from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from fem import solvers
from fem.solvers import _pardiso_spd


@pytest.fixture
def fake_backend(monkeypatch):
    class FakePardisoSolver:
        instances = []
        construction_error = None
        solve_error = None
        solve_result = None

        def __init__(self, matrix, mtype):
            self.matrix = matrix
            self.mtype = mtype
            self.release_calls = 0
            type(self).instances.append(self)
            if type(self).construction_error is not None:
                raise type(self).construction_error

        def solve(self, rhs):
            if type(self).solve_error is not None:
                raise type(self).solve_error
            if type(self).solve_result is not None:
                return type(self).solve_result
            upper = self.matrix.toarray()
            dense = upper + upper.T - np.diag(np.diag(upper))
            return np.linalg.solve(dense, rhs)

        def release(self):
            self.release_calls += 1

    monkeypatch.setattr(
        _pardiso_spd,
        "_NativePardisoSolver",
        FakePardisoSolver,
    )
    return FakePardisoSolver


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

    assert backend.mtype == _pardiso_spd._MTYPE_REAL_SYM_POSDEF == 2
    assert isinstance(backend.matrix, csr_matrix)
    assert backend.matrix.dtype == np.dtype(np.float64)
    assert backend.matrix.has_sorted_indices
    assert backend.matrix is not matrix
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


def test_factor_solve_accepts_multiple_rhs(fake_backend):
    rhs = np.array(
        [
            [1.0, 4.0],
            [2.0, -1.0],
            [3.0, 0.5],
        ]
    )
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    try:
        result = factor.solve(rhs)
    finally:
        factor.close()

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
        (_csr_with_nonfinite(np.inf), "finite"),
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
)
def test_factorize_rejects_invalid_matrix_contract(matrix, message, fake_backend):
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
        (np.array(1.0), "must have shape"),
        (np.ones(2), "must have shape"),
        (np.ones((3, 1, 1)), "must have shape"),
        (np.array([1.0, np.nan, 3.0]), "finite"),
        (np.array([1.0, np.inf, 3.0]), "finite"),
        (np.ones(3, dtype=np.complex128), "real numeric"),
    ],
)
def test_factor_solve_rejects_invalid_rhs(rhs, message, fake_backend):
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    try:
        with pytest.raises(ValueError, match=message):
            factor.solve(rhs)
    finally:
        factor.close()


def test_backend_solve_failure_has_a_stable_exception_chain(fake_backend):
    native_error = MemoryError("native allocation failed")
    fake_backend.solve_error = native_error
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    try:
        with pytest.raises(
            _pardiso_spd._PardisoSPDError,
            match="PARDISO SPD solve failed",
        ) as caught:
            factor.solve(np.ones(3))
    finally:
        factor.close()

    assert caught.value.__cause__ is native_error


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (np.zeros(2), "invalid result shape"),
        (np.array([1.0, np.nan, 3.0]), "non-finite"),
    ],
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


def test_release_failure_is_chained_and_is_not_retried(monkeypatch):
    native_error = RuntimeError("native release failed")

    class ReleaseFailingSolver:
        instances = []

        def __init__(self, _matrix, _mtype):
            self.release_calls = 0
            type(self).instances.append(self)

        def solve(self, rhs):
            return rhs

        def release(self):
            self.release_calls += 1
            raise native_error

    monkeypatch.setattr(
        _pardiso_spd,
        "_NativePardisoSolver",
        ReleaseFailingSolver,
    )
    factor = _pardiso_spd.factorize_spd(_spd_upper())
    backend = ReleaseFailingSolver.instances[0]

    with pytest.raises(
        _pardiso_spd._PardisoSPDError,
        match="PARDISO SPD release failed",
    ) as caught:
        factor.close()
    factor.close()

    assert caught.value.__cause__ is native_error
    assert backend.release_calls == 1


def test_factorization_failure_releases_the_partly_constructed_backend(
    fake_backend,
):
    native_error = RuntimeError("matrix is not positive definite")
    fake_backend.construction_error = native_error

    with pytest.raises(
        _pardiso_spd._PardisoSPDError,
        match="PARDISO SPD factorization failed",
    ) as caught:
        _pardiso_spd.factorize_spd(_spd_upper())

    assert caught.value.__cause__ is native_error
    assert len(fake_backend.instances) == 1
    assert fake_backend.instances[0].release_calls == 1


def test_private_adapter_is_not_in_the_solver_package_public_surface():
    assert "_pardiso_spd" not in solvers.__all__
    assert "factorize_spd" not in solvers.__all__
    assert "PardisoSPDFactor" not in solvers.__all__


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

    with pytest.raises(
        _pardiso_spd._PardisoSPDError,
        match="PARDISO SPD factorization failed",
    ) as caught:
        _pardiso_spd.factorize_spd(indefinite_upper)

    assert caught.value.__cause__ is not None
