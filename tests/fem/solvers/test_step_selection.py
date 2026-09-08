import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.core.model import AnalysisStep, NodalLoad
from fem.core.result import ModelResult, ModelResults
from fem.solvers import static_linear
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
)


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


@pytest.mark.parametrize("selection_kind", ["list", "generator"])
def test_plural_iterable_selection_preserves_caller_order(selection_kind):
    model = make_two_step_static_pull_truss_model()
    selectors = ("pull2", "pull1")
    if selection_kind == "list":
        selection = list(selectors)
    else:
        selection = (selector for selector in selectors)

    results = static_linear.solve(model, steps=selection)

    assert isinstance(results, ModelResults)
    assert tuple(result.step.name for result in results.results) == ("pull2", "pull1")
    assert tuple(
        result.U[model.mesh.global_dof(2, 0)] for result in results.results
    ) == pytest.approx((1.0, 0.5))


@pytest.mark.parametrize(
    ("name", "expected_name"),
    [("cases", "cases_pull2"), ("", "_pull2")],
    ids=["named-prefix", "empty-prefix"],
)
def test_one_item_plural_selection_keeps_collection_shape_and_step_suffix(name, expected_name):
    model = make_two_step_static_pull_truss_model()

    results = static_linear.solve(model, name=name, steps=["pull2"])

    assert isinstance(results, ModelResults)
    assert len(results.results) == 1
    assert results.results[0].step is model.steps[2]
    assert results.results[0].name == expected_name
    assert results.results[0].U[model.mesh.global_dof(2, 0)] == pytest.approx(1.0)


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
    # Isolate step fallback from the unconstrained truss stiffness singularity.
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


def test_plural_selection_rejects_empty_iterables():
    model = make_two_step_static_pull_truss_model()

    with pytest.raises(ValueError, match="at least one selector"):
        static_linear.solve(model, steps=[])


def test_plural_selection_rejects_bare_step_name():
    model = make_two_step_static_pull_truss_model()

    with pytest.raises(TypeError, match="exact string 'all'"):
        static_linear.solve(model, steps="pull1")


@pytest.mark.parametrize(
    "selection",
    [1, (None,), (object(),)],
    ids=["non-iterable", "implicit-default-member", "invalid-member"],
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
    ids=["name-and-index", "object-and-name"],
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
