import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem import solvers
from fem.core.result import ModelResult, ModelResults
from fem.solvers import static_linear
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
)


def test_static_linear_solver_builds_step_boundary_and_solves_case():
    model = make_static_pull_truss_model()
    mesh = model.mesh

    U = static_linear.solve(model, "pull").U

    assert U[mesh.global_dof(2, 0)] == pytest.approx(0.5)
    assert U[mesh.global_dof(2, 1)] == pytest.approx(0.0)


def test_static_linear_solver_public_surface():
    assert static_linear.__all__ == ["solve", "solve_all"]


def test_static_linear_solver_returns_result_with_displacements_and_reactions():
    model = make_static_pull_truss_model()
    mesh = model.mesh

    result = static_linear.solve(
        model,
        "pull",
        name="pull_case",
    )

    assert isinstance(result, ModelResult)
    assert result.step.name == "pull"
    assert result.name == "pull_case"
    assert result.U[mesh.global_dof(2, 0)] == pytest.approx(0.5)
    assert result.U[mesh.global_dof(2, 1)] == pytest.approx(0.0)
    assert result.reactions[mesh.global_dof(1, 0)] == pytest.approx(-100.0)
    assert result.reactions[mesh.global_dof(2, 0)] == pytest.approx(0.0)


def test_static_linear_solver_solve_all_returns_step_result_collection():
    model = make_two_step_static_pull_truss_model()
    mesh = model.mesh

    results = static_linear.solve_all(model)

    assert isinstance(results, ModelResults)
    assert len(results.results) == 2
    assert tuple(result.step.name for result in results.results) == ("pull1", "pull2")
    pull1, pull2 = results.results
    assert pull1.U[mesh.global_dof(2, 0)] == pytest.approx(0.5)
    assert pull2.U[mesh.global_dof(2, 0)] == pytest.approx(1.0)
    assert pull1.name == "bar_pull1"
    assert pull2.name == "bar_pull2"


def test_linear_solver_solves_sparse_system_and_rejects_dense_matrix():
    K = csr_matrix([[2.0, 0.0], [0.0, 4.0]])
    F = np.array([6.0, 8.0])

    U = solvers.linear.solve(K, F)

    assert np.allclose(U, [3.0, 2.0])
    with pytest.raises(TypeError):
        solvers.linear.solve(np.eye(2), F)


def test_linear_solver_rejects_singular_sparse_matrix():
    K = csr_matrix([[1.0, 0.0], [0.0, 0.0]])
    F = np.array([1.0, 1.0])

    with pytest.raises(RuntimeError):
        solvers.linear.solve(K, F)
