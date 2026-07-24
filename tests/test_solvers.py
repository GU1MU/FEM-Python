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

    bc = static_linear.boundary_for_step(model, "pull")
    assert len(bc.prescribed_displacements) == 5
    assert sum(bc.nodal_forces.values()) == pytest.approx(100.0)

    U = static_linear.solve(model, "pull").U

    assert U[mesh.global_dof(2, 0)] == pytest.approx(0.5)
    assert U[mesh.global_dof(2, 1)] == pytest.approx(0.0)


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


def test_static_linear_validate_problem_matches_solver_rules():
    model = make_static_pull_truss_model()
    assert static_linear.validate_problem(model, "pull").name == "pull"

    model.steps[0].procedure = "dynamic"
    with pytest.raises(ValueError, match="requires procedure 'static'"):
        static_linear.validate_problem(model, "pull")

    model.steps[0].procedure = "static"
    model.steps[0].metadata["nlgeom"] = "YES"
    with pytest.raises(ValueError, match="does not support nlgeom"):
        static_linear.validate_problem(model, "pull")

    model.steps[0].metadata.pop("nlgeom")
    model.steps[0].cloads = (type(model.steps[0].cloads[0])("缺失节点集", 1, 1.0),)
    with pytest.raises(KeyError, match="references missing node set 缺失节点集"):
        static_linear.validate_problem(model, "pull")


def test_static_linear_stiffness_preflight_detects_free_rigid_dofs():
    model = make_static_pull_truss_model()

    assert static_linear.validate_stiffness(model, "pull").name == "pull"

    model.steps[0].boundaries = model.steps[0].boundaries[:1]
    with pytest.raises(ValueError, match="约束不足或刚度矩阵奇异"):
        static_linear.validate_stiffness(model, "pull")


def test_solve_uses_validate_problem(monkeypatch):
    model = make_static_pull_truss_model()
    selected = model.steps[0]
    calls = []

    def validate(candidate, step):
        calls.append((candidate, step))
        return selected

    monkeypatch.setattr(static_linear, "validate_problem", validate)
    static_linear.solve(model, "pull")
    assert calls == [(model, "pull")]


def test_prevalidated_solve_skips_duplicate_validation_and_records_stages(monkeypatch):
    model = make_static_pull_truss_model()
    selected = static_linear.validate_problem(model, "pull")
    calls = []
    monkeypatch.setattr(
        static_linear,
        "validate_problem",
        lambda *_args, **_kwargs: calls.append(1),
    )
    timings: dict[str, float] = {}

    result = static_linear.solve(
        model,
        "pull",
        _validated_step=selected,
        timings=timings,
    )

    assert calls == []
    assert result.step is selected
    assert set(timings) == {
        "分析准备",
        "刚度矩阵装配",
        "载荷与边界条件",
        "线性方程求解",
        "反力与结果封装",
    }
    assert all(seconds >= 0.0 for seconds in timings.values())
