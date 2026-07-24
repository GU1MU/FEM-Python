import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem import solvers
from fem.core.model import AnalysisStep, DisplacementConstraint, NodalLoad
from fem.core.result import ModelResult, ModelResults
from fem.solvers import static_linear
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
)


def test_static_linear_solver_builds_step_boundary_and_solves_case():
    model = make_static_pull_truss_model()
    mesh = model.mesh

    displacement = static_linear.solve(model, "pull").U

    assert displacement[mesh.global_dof(2, 0)] == pytest.approx(0.5)
    assert displacement[mesh.global_dof(2, 1)] == pytest.approx(0.0)


def test_static_linear_solver_public_surface():
    assert static_linear.__all__ == ["solve"]


def test_static_linear_solver_returns_scalar_result_with_name_and_reactions():
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


@pytest.mark.parametrize(
    "selector_kind",
    ["default", "name", "index", "object"],
)
def test_scalar_step_selectors_always_return_model_result(selector_kind):
    model = make_two_step_static_pull_truss_model()

    if selector_kind == "default":
        result = static_linear.solve(model)
    elif selector_kind == "name":
        result = static_linear.solve(model, step="pull1")
    elif selector_kind == "index":
        result = static_linear.solve(model, step=1)
    else:
        result = static_linear.solve(model, step=model.steps[1])

    assert isinstance(result, ModelResult)
    assert result.step is model.steps[1]
    assert result.U[model.mesh.global_dof(2, 0)] == pytest.approx(0.5)


def test_plural_all_returns_ordered_independent_static_results():
    model = make_two_step_static_pull_truss_model()
    mesh = model.mesh

    results = static_linear.solve(model, steps="all")

    assert isinstance(results, ModelResults)
    assert tuple(result.step.name for result in results.results) == ("pull1", "pull2")
    pull1, pull2 = results.results
    assert pull1.U[mesh.global_dof(2, 0)] == pytest.approx(0.5)
    assert pull2.U[mesh.global_dof(2, 0)] == pytest.approx(1.0)
    assert pull1.name == "bar_pull1"
    assert pull2.name == "bar_pull2"


def test_model_results_supports_direct_sequence_access():
    model = make_two_step_static_pull_truss_model()

    results = static_linear.solve(model, steps="all")

    iterated = tuple(results)
    assert len(results) == 2
    assert all(
        actual is expected
        for actual, expected in zip(iterated, results.results, strict=True)
    )
    assert results[0] is results.results[0]
    assert results[:] == results.results


@pytest.mark.parametrize("selection_kind", ["tuple", "list", "generator"])
def test_plural_iterable_selection_preserves_caller_order(selection_kind):
    model = make_two_step_static_pull_truss_model()
    selectors = ("pull2", "pull1")
    if selection_kind == "list":
        selection = list(selectors)
    elif selection_kind == "generator":
        selection = (selector for selector in selectors)
    else:
        selection = selectors

    results = static_linear.solve(model, steps=selection)

    assert isinstance(results, ModelResults)
    assert tuple(result.step.name for result in results.results) == ("pull2", "pull1")
    assert tuple(
        result.U[model.mesh.global_dof(2, 0)] for result in results.results
    ) == pytest.approx((1.0, 0.5))


def test_one_item_plural_selection_keeps_collection_shape_and_step_suffix():
    model = make_two_step_static_pull_truss_model()

    results = static_linear.solve(model, name="cases", steps=["pull2"])

    assert isinstance(results, ModelResults)
    assert len(results.results) == 1
    assert results.results[0].step is model.steps[2]
    assert results.results[0].name == "cases_pull2"


def test_plural_selection_preserves_an_explicit_empty_name_prefix():
    model = make_two_step_static_pull_truss_model()

    results = static_linear.solve(model, name="", steps=("pull1",))

    assert results.results[0].name == "_pull1"


def test_scalar_and_plural_selections_accept_external_step_objects():
    model = make_two_step_static_pull_truss_model()
    external = AnalysisStep("external", cloads=[NodalLoad("TIP", 1, 50.0)])

    scalar = static_linear.solve(model, step=external, name="external_case")
    plural = static_linear.solve(model, name="batch", steps=(external,))

    assert scalar.step is external
    assert scalar.name == "external_case"
    assert scalar.U[model.mesh.global_dof(2, 0)] == pytest.approx(0.25)
    assert isinstance(plural, ModelResults)
    assert plural.results[0].step is external
    assert plural.results[0].name == "batch_external"
    assert plural.results[0].U[model.mesh.global_dof(2, 0)] == pytest.approx(0.25)


def test_plural_all_falls_back_to_initial_when_no_runnable_step_exists():
    model = make_two_step_static_pull_truss_model()
    model.steps[:] = model.steps[:1]

    results = static_linear.solve(model, steps="all")

    assert isinstance(results, ModelResults)
    assert len(results.results) == 1
    assert results.results[0].step is model.steps[0]
    assert results.results[0].name == "bar_Initial"
    assert np.allclose(results.results[0].U, 0.0)


def test_plural_all_falls_back_to_implicit_step_when_model_has_no_steps(monkeypatch):
    model = make_static_pull_truss_model()
    model.name = None
    model.steps.clear()
    monkeypatch.setattr(
        static_linear,
        "assemble_global_stiffness_sparse",
        lambda mesh: csr_matrix(np.eye(mesh.num_dofs)),
    )

    results = static_linear.solve(model, steps="all")

    assert isinstance(results, ModelResults)
    assert len(results.results) == 1
    assert results.results[0].step is None
    assert results.results[0].name == "result_step"
    assert np.allclose(results.results[0].U, 0.0)


@pytest.mark.parametrize("selection", [[], (), iter(())])
def test_plural_selection_rejects_empty_iterables(selection):
    model = make_two_step_static_pull_truss_model()

    with pytest.raises(ValueError, match="at least one selector"):
        static_linear.solve(model, steps=selection)


@pytest.mark.parametrize("selection", ["pull1", "ALL", ""])
def test_plural_selection_rejects_strings_other_than_exact_all(selection):
    model = make_two_step_static_pull_truss_model()

    with pytest.raises(TypeError, match="exact string 'all'"):
        static_linear.solve(model, steps=selection)


@pytest.mark.parametrize(
    "selection",
    [1, True, b"all", bytearray(b"all"), AnalysisStep("direct"), (None,), (object(),)],
)
def test_plural_selection_rejects_malformed_values(selection):
    model = make_two_step_static_pull_truss_model()

    with pytest.raises(TypeError, match="step"):
        static_linear.solve(model, steps=selection)


def test_scalar_and_plural_selections_reject_boolean_selectors():
    model = make_two_step_static_pull_truss_model()

    with pytest.raises(TypeError, match="step selector"):
        static_linear.solve(model, step=True)
    with pytest.raises(TypeError, match="step selector"):
        static_linear.solve(model, steps=(False,))


@pytest.mark.parametrize(
    "selection_factory",
    [
        lambda model: ("pull1", 1),
        lambda model: (model.steps[1], "pull1"),
    ],
)
def test_plural_selection_rejects_duplicate_resolved_steps(selection_factory):
    model = make_two_step_static_pull_truss_model()

    with pytest.raises(ValueError, match="selected more than once"):
        static_linear.solve(model, steps=selection_factory(model))


def test_plural_selection_rejects_external_result_name_collisions():
    model = make_two_step_static_pull_truss_model()
    first = AnalysisStep("external")
    second = AnalysisStep("EXTERNAL")

    with pytest.raises(ValueError, match="names must be unique ignoring case"):
        static_linear.solve(model, steps=(first, second))


def test_scalar_and_plural_selectors_are_mutually_exclusive():
    model = make_two_step_static_pull_truss_model()

    with pytest.raises(ValueError, match="mutually exclusive"):
        static_linear.solve(model, step="pull1", steps=("pull2",))


def test_plural_selection_preserves_canonical_resolver_errors():
    model = make_two_step_static_pull_truss_model()

    with pytest.raises(KeyError, match="analysis step missing is not defined"):
        static_linear.solve(model, steps=("pull1", "missing"))
    with pytest.raises(IndexError):
        static_linear.solve(model, steps=(99,))


@pytest.mark.parametrize(
    "invalid_step",
    [
        AnalysisStep("late_dynamic", procedure="dynamic"),
        AnalysisStep("late_invalid", cloads=[NodalLoad("MISSING", 1, 1.0)]),
    ],
    ids=["procedure", "reference"],
)
def test_all_plural_steps_validate_before_model_preparation(monkeypatch, invalid_step):
    model = make_two_step_static_pull_truss_model()
    preparation_calls = 0

    def track_preparation(model):
        nonlocal preparation_calls
        preparation_calls += 1

    monkeypatch.setattr(static_linear.materials, "apply_sections", track_preparation)

    with pytest.raises((KeyError, ValueError)):
        static_linear.solve(model, steps=("pull1", invalid_step))

    assert preparation_calls == 0


def test_plural_solve_prepares_once_and_runs_each_load_case_once(monkeypatch):
    model = make_two_step_static_pull_truss_model()
    calls = {
        "materials": 0,
        "boundary": 0,
        "stiffness": 0,
        "load": 0,
        "dirichlet": 0,
        "linear": 0,
    }
    prepared = {}

    original_apply_sections = static_linear.materials.apply_sections
    original_boundary_for_step = static_linear._boundary_step.boundary_for_step
    original_assemble = static_linear.assemble_global_stiffness_sparse
    original_build_load = static_linear.build_load_vector
    original_apply_dirichlet = static_linear.apply_dirichlet
    original_linear_solve = static_linear.linear.solve

    def apply_sections(model):
        calls["materials"] += 1
        return original_apply_sections(model)

    def boundary_for_step(model, step):
        calls["boundary"] += 1
        return original_boundary_for_step(model, step)

    def assemble(mesh):
        calls["stiffness"] += 1
        stiffness = original_assemble(mesh)
        prepared["stiffness"] = stiffness
        prepared["snapshot"] = stiffness.toarray().copy()
        return stiffness

    def build_load(mesh, boundary):
        calls["load"] += 1
        return original_build_load(mesh, boundary)

    def apply_constraints(stiffness, load, boundary):
        calls["dirichlet"] += 1
        assert stiffness is prepared["stiffness"]
        result = original_apply_dirichlet(stiffness, load, boundary)
        assert np.array_equal(stiffness.toarray(), prepared["snapshot"])
        return result

    def solve_linear(stiffness, load):
        calls["linear"] += 1
        return original_linear_solve(stiffness, load)

    monkeypatch.setattr(static_linear.materials, "apply_sections", apply_sections)
    monkeypatch.setattr(
        static_linear._boundary_step,
        "boundary_for_step",
        boundary_for_step,
    )
    monkeypatch.setattr(static_linear, "assemble_global_stiffness_sparse", assemble)
    monkeypatch.setattr(static_linear, "build_load_vector", build_load)
    monkeypatch.setattr(static_linear, "apply_dirichlet", apply_constraints)
    monkeypatch.setattr(static_linear.linear, "solve", solve_linear)

    results = static_linear.solve(model, steps="all")

    assert isinstance(results, ModelResults)
    assert calls == {
        "materials": 1,
        "boundary": 2,
        "stiffness": 1,
        "load": 2,
        "dirichlet": 2,
        "linear": 2,
    }
    assert np.array_equal(
        prepared["stiffness"].toarray(),
        prepared["snapshot"],
    )


def test_nonzero_prescribed_displacement_is_absolute_in_each_load_case():
    model = make_two_step_static_pull_truss_model()
    settlement = AnalysisStep(
        "settlement",
        boundaries=[DisplacementConstraint("TIP", 1, 1, 0.25)],
    )

    results = static_linear.solve(model, steps=("pull1", settlement))

    pull, settled = results.results
    loaded_dof = model.mesh.global_dof(2, 0)
    fixed_dof = model.mesh.global_dof(1, 0)
    assert pull.U[loaded_dof] == pytest.approx(0.5)
    assert settled.U[loaded_dof] == pytest.approx(0.25)
    assert settled.reactions[fixed_dof] == pytest.approx(-50.0)
    assert settled.reactions[loaded_dof] == pytest.approx(50.0)


def test_linear_solver_solves_sparse_system_and_rejects_dense_matrix():
    stiffness = csr_matrix([[2.0, 0.0], [0.0, 4.0]])
    load = np.array([6.0, 8.0])

    displacement = solvers.linear.solve(stiffness, load)

    assert np.allclose(displacement, [3.0, 2.0])
    with pytest.raises(TypeError):
        solvers.linear.solve(np.eye(2), load)


def test_linear_solver_rejects_singular_sparse_matrix():
    stiffness = csr_matrix([[1.0, 0.0], [0.0, 0.0]])
    load = np.array([1.0, 1.0])

    with pytest.raises(RuntimeError):
        solvers.linear.solve(stiffness, load)


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
