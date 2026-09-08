import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.boundary.condition import BoundaryCondition
from fem.boundary.constraints import apply_dirichlet
from fem.solvers import linear


@pytest.mark.parametrize("invalid_target", ["K", "F", "displacement"])
def test_dirichlet_application_rejects_nonfinite_system_data(invalid_target):
    K = csr_matrix(np.eye(2))
    F = np.ones(2)
    boundary = BoundaryCondition(prescribed_displacements={0: 0.0})

    if invalid_target == "K":
        K.data[0] = np.nan
        expected = "K must contain only finite values"
    elif invalid_target == "F":
        F[0] = np.inf
        expected = "F must contain only finite values"
    else:
        boundary.prescribed_displacements[0] = np.nan
        expected = "prescribed displacement at DOF 0 must be finite"

    with pytest.raises(ValueError, match=expected):
        apply_dirichlet(K, F, boundary)


def test_dirichlet_application_rejects_non_vector_rhs():
    K = csr_matrix(np.eye(2))
    boundary = BoundaryCondition()

    with pytest.raises(
        ValueError,
        match="F must be one-dimensional or a column vector",
    ):
        apply_dirichlet(K, np.ones((2, 2)), boundary)


def test_dirichlet_application_accepts_column_vector_rhs():
    K = csr_matrix(np.eye(2))

    _, F_mod = apply_dirichlet(K, np.ones((2, 1)), BoundaryCondition())

    assert F_mod.shape == (2,)
    assert np.array_equal(F_mod, np.ones(2))


def test_nonzero_dirichlet_constraint_preserves_coupled_solution():
    K = csr_matrix([[2.0, -2.0], [-2.0, 2.0]])
    F = np.zeros(2, dtype=float)
    boundary = BoundaryCondition(prescribed_displacements={0: 0.25})

    K_mod, F_mod = apply_dirichlet(K, F, boundary)
    U = linear.solve(K_mod, F_mod)

    np.testing.assert_allclose(K_mod.toarray(), [[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(F_mod, [0.25, 0.5])
    assert U == pytest.approx([0.25, 0.25])
    assert K @ U - F == pytest.approx([0.0, 0.0])
    np.testing.assert_array_equal(K.toarray(), [[2.0, -2.0], [-2.0, 2.0]])
    np.testing.assert_array_equal(F, [0.0, 0.0])
    assert boundary.prescribed_displacements == {0: 0.25}
