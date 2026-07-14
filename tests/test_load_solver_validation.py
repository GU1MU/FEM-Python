import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.boundary.condition import BoundaryCondition
from fem.boundary.constraints import apply_dirichlet
from fem.boundary.loads import build_load_vector
from fem.solvers import linear
from tests.helpers.model_builders import make_simple_truss_mesh


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_boundary_condition_rejects_nonfinite_scalar_values(value):
    boundary = BoundaryCondition()

    with pytest.raises(ValueError, match="prescribed displacement must be finite"):
        boundary.add_displacement_dof(0, value)
    with pytest.raises(ValueError, match="nodal force must be finite"):
        boundary.add_nodal_force_dof(0, value)
    with pytest.raises(ValueError, match="load vector component must be finite"):
        boundary.add_body_force_element(1, value, 0.0)


def test_boundary_condition_rejects_overflow_during_force_accumulation():
    boundary = BoundaryCondition()
    boundary.add_nodal_force_dof(0, np.finfo(float).max)

    with pytest.raises(ValueError, match="accumulated nodal force at DOF 0 must be finite"):
        boundary.add_nodal_force_dof(0, np.finfo(float).max)


def test_load_assembly_rejects_nonfinite_direct_nodal_force_maps():
    mesh = make_simple_truss_mesh()
    boundary = BoundaryCondition(nodal_forces={0: np.nan})

    with pytest.raises(ValueError, match="nodal force at DOF 0 must be finite"):
        build_load_vector(mesh, boundary)


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


@pytest.mark.parametrize("invalid_target", ["K", "F"])
def test_sparse_solver_rejects_nonfinite_system_data(invalid_target):
    K = csr_matrix(np.eye(2))
    F = np.ones(2)

    if invalid_target == "K":
        K.data[0] = np.nan
        expected = "K must contain only finite values"
    else:
        F[0] = np.inf
        expected = "F must contain only finite values"

    with pytest.raises(ValueError, match=expected):
        linear.solve(K, F)
