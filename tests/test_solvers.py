from concurrent.futures import ThreadPoolExecutor
from inspect import Parameter, signature
from time import perf_counter

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from fem import solvers
from fem.boundary.condition import BoundaryCondition
from fem.boundary.constraints import apply_dirichlet
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
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
    assert static_linear.__all__ == [
        "PreparedSystem",
        "prepare",
        "solve",
    ]


def test_static_linear_solver_public_call_shapes():
    def call_shape(callable_object):
        return tuple(
            (name, parameter.kind, parameter.default)
            for name, parameter in signature(callable_object).parameters.items()
        )

    assert call_shape(static_linear.prepare) == (
        ("model", Parameter.POSITIONAL_OR_KEYWORD, Parameter.empty),
        ("copy_model", Parameter.KEYWORD_ONLY, True),
        ("timings", Parameter.KEYWORD_ONLY, None),
    )
    assert call_shape(static_linear.solve) == (
        ("model", Parameter.POSITIONAL_OR_KEYWORD, Parameter.empty),
        ("step", Parameter.POSITIONAL_OR_KEYWORD, None),
        ("name", Parameter.POSITIONAL_OR_KEYWORD, None),
        ("steps", Parameter.KEYWORD_ONLY, None),
        ("_validated_step", Parameter.KEYWORD_ONLY, ...),
        ("_prepared_system", Parameter.KEYWORD_ONLY, None),
        ("timings", Parameter.KEYWORD_ONLY, None),
    )
    assert call_shape(static_linear.PreparedSystem.clone) == (
        ("self", Parameter.POSITIONAL_OR_KEYWORD, Parameter.empty),
    )
    assert call_shape(static_linear.PreparedSystem.solve) == (
        ("self", Parameter.POSITIONAL_OR_KEYWORD, Parameter.empty),
        ("step", Parameter.POSITIONAL_OR_KEYWORD, None),
        ("name", Parameter.POSITIONAL_OR_KEYWORD, None),
        ("steps", Parameter.KEYWORD_ONLY, None),
        ("_validated_step", Parameter.KEYWORD_ONLY, ...),
        ("timings", Parameter.KEYWORD_ONLY, None),
    )
    expected_validation_shape = (
        ("self", Parameter.POSITIONAL_OR_KEYWORD, Parameter.empty),
        ("step", Parameter.POSITIONAL_OR_KEYWORD, None),
    )
    assert (
        call_shape(static_linear.PreparedSystem.validate_step)
        == expected_validation_shape
    )
    assert (
        call_shape(static_linear.PreparedSystem.validate_stiffness)
        == expected_validation_shape
    )


def test_prepared_system_applies_sections_and_assembles_once_for_many_steps(
    monkeypatch,
):
    expected_model = make_two_step_static_pull_truss_model()
    expected = static_linear.solve(expected_model, steps="all")
    model = make_two_step_static_pull_truss_model()
    calls = {"materials": 0, "stiffness": 0, "factor": 0}
    original_apply = static_linear.materials.apply_sections
    original_assemble = static_linear.assemble_global_stiffness_sparse
    original_factor = static_linear.splu

    def apply_sections(candidate):
        calls["materials"] += 1
        return original_apply(candidate)

    def assemble(mesh):
        calls["stiffness"] += 1
        return original_assemble(mesh)

    def factor(stiffness):
        calls["factor"] += 1
        return original_factor(stiffness)

    monkeypatch.setattr(
        static_linear.materials,
        "apply_sections",
        apply_sections,
    )
    monkeypatch.setattr(
        static_linear,
        "assemble_global_stiffness_sparse",
        assemble,
    )
    monkeypatch.setattr(static_linear, "splu", factor)

    prepared = static_linear.prepare(model)
    results = prepared.solve(steps="all")
    pull1, pull2 = results.results

    assert calls == {"materials": 1, "stiffness": 1, "factor": 1}
    assert pull1.model is pull2.model
    assert pull1.model is not model
    np.testing.assert_allclose(pull1.U, expected.results[0].U)
    np.testing.assert_allclose(pull2.U, expected.results[1].U)
    pull1.model.mesh.nodes[1].x += 10.0
    selected = prepared.validate_step("pull1")
    stiffness_step = prepared.validate_stiffness("pull1")
    selected.name = "mutated-selected-step"
    stiffness_step.name = "mutated-stiffness-step"
    repeated = prepared.solve("pull1")
    assert repeated.step.name == "pull1"
    np.testing.assert_allclose(repeated.U, expected.results[0].U)
    with pytest.raises(ValueError):
        prepared._base_stiffness.data[0] = 0.0


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
        "factor": 0,
    }
    prepared = {}

    original_apply_sections = static_linear.materials.apply_sections
    original_boundary_for_step = static_linear._boundary_step.boundary_for_step
    original_assemble = static_linear.assemble_global_stiffness_sparse
    original_build_load = static_linear.build_load_vector
    original_factor = static_linear.splu

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

    def factor(stiffness):
        calls["factor"] += 1
        return original_factor(stiffness)

    monkeypatch.setattr(static_linear.materials, "apply_sections", apply_sections)
    monkeypatch.setattr(
        static_linear._boundary_step,
        "boundary_for_step",
        boundary_for_step,
    )
    monkeypatch.setattr(static_linear, "assemble_global_stiffness_sparse", assemble)
    monkeypatch.setattr(static_linear, "build_load_vector", build_load)
    monkeypatch.setattr(static_linear, "splu", factor)

    results = static_linear.solve(model, steps="all")

    assert isinstance(results, ModelResults)
    assert calls == {
        "materials": 1,
        "boundary": 2,
        "stiffness": 1,
        "load": 2,
        "factor": 1,
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


def test_reduced_solver_matches_full_dirichlet_oracle_for_multiple_cases():
    model = make_two_step_static_pull_truss_model()
    settlement = AnalysisStep(
        "settlement",
        boundaries=[DisplacementConstraint("TIP", 1, 1, 0.25)],
        cloads=[NodalLoad("TIP", 1, 30.0)],
    )
    prepared = static_linear.prepare(model)

    results = prepared._solve_owned(
        steps=("pull1", "pull2", settlement),
    )

    for result in results.results:
        boundary = boundary_for_step(result.model, result.step)
        load = build_load_vector(result.model.mesh, boundary)
        constrained_stiffness, constrained_load = apply_dirichlet(
            prepared._base_stiffness,
            load,
            boundary,
        )
        expected_displacement = solvers.linear.solve(
            constrained_stiffness,
            constrained_load,
        )
        expected_reactions = (
            prepared._base_stiffness @ expected_displacement - load
        )
        np.testing.assert_allclose(result.U, expected_displacement)
        np.testing.assert_allclose(
            result.reactions,
            expected_reactions,
        )


def test_factor_key_ignores_load_and_prescribed_values(monkeypatch):
    model = make_two_step_static_pull_truss_model()
    first = AnalysisStep(
        "first",
        boundaries=[DisplacementConstraint("FIXED", 1, 1, 0.1)],
        cloads=[NodalLoad("TIP", 1, 10.0)],
    )
    second = AnalysisStep(
        "second",
        boundaries=[DisplacementConstraint("FIXED", 1, 1, -0.2)],
        cloads=[NodalLoad("TIP", 1, 75.0)],
    )
    factor_calls = 0
    original_factor = static_linear.splu

    def factor(stiffness):
        nonlocal factor_calls
        factor_calls += 1
        return original_factor(stiffness)

    monkeypatch.setattr(static_linear, "splu", factor)
    prepared = static_linear.prepare(model)

    results = prepared._solve_owned(steps=(first, second))

    assert factor_calls == 1
    assert results.results[0].U[0] == pytest.approx(0.1)
    assert results.results[1].U[0] == pytest.approx(-0.2)


def test_factor_cache_lru_is_bounded_and_refactors_evicted_pattern(
    monkeypatch,
):
    stiffness = csr_matrix(np.diag(np.arange(1.0, 7.0)))
    cache = static_linear._FactorizationCache(stiffness)
    factor_calls = 0
    original_factor = static_linear.splu

    def factor(matrix):
        nonlocal factor_calls
        factor_calls += 1
        return original_factor(matrix)

    monkeypatch.setattr(static_linear, "splu", factor)
    patterns = tuple((dof,) for dof in range(5))
    for pattern in patterns:
        cache.factor_for(pattern)

    assert factor_calls == 5
    assert len(cache._entries) == static_linear._FACTOR_CACHE_MAX_ENTRIES
    assert tuple(cache._entries) == patterns[-4:]

    cache.factor_for(patterns[0])
    assert factor_calls == 6
    assert tuple(cache._entries) == patterns[-3:] + patterns[:1]


def test_factor_cache_serializes_concurrent_factor_creation(monkeypatch):
    stiffness = csr_matrix(np.diag(np.arange(1.0, 9.0)))
    cache = static_linear._FactorizationCache(stiffness)
    factor_calls = 0
    original_factor = static_linear.splu

    def factor(matrix):
        nonlocal factor_calls
        factor_calls += 1
        return original_factor(matrix)

    monkeypatch.setattr(static_linear, "splu", factor)

    def solve(scale):
        displacement, _ = cache.solve(
            scale * np.ones(8),
            (0,),
            np.array([0.0]),
        )
        return displacement

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(solve, range(1, 9)))

    assert factor_calls == 1
    for scale, displacement in enumerate(results, start=1):
        np.testing.assert_allclose(
            displacement[1:],
            scale / np.arange(2.0, 9.0),
        )


def test_factor_cache_handles_fully_constrained_and_unconstrained_systems(
    monkeypatch,
):
    stiffness = csr_matrix(np.diag([2.0, 3.0]))
    cache = static_linear._FactorizationCache(stiffness)
    factor_calls = 0
    original_factor = static_linear.splu

    def factor(matrix):
        nonlocal factor_calls
        factor_calls += 1
        return original_factor(matrix)

    monkeypatch.setattr(static_linear, "splu", factor)
    displacement, free_dofs = cache.solve(
        np.array([10.0, 20.0]),
        (0, 1),
        np.array([0.25, -0.5]),
    )

    np.testing.assert_allclose(displacement, [0.25, -0.5])
    assert free_dofs.size == 0
    assert factor_calls == 0

    displacement, free_dofs = cache.solve(
        np.array([4.0, 9.0]),
        (),
        np.empty(0),
    )
    np.testing.assert_allclose(displacement, [2.0, 3.0])
    np.testing.assert_array_equal(free_dofs, [0, 1])
    assert factor_calls == 1


def test_repeated_reduced_solve_matches_full_system_oracle_and_records_cost(
    record_property,
):
    dimension = 400
    stiffness = diags(
        (
            -np.ones(dimension - 1),
            2.0 * np.ones(dimension),
            -np.ones(dimension - 1),
        ),
        offsets=(-1, 0, 1),
        format="csr",
    )
    boundary = BoundaryCondition()
    boundary.add_displacement_dof(0, 0.125)
    boundary.add_displacement_dof(dimension - 1, -0.05)
    pattern, values = static_linear._validated_prescribed_displacements(
        boundary,
        dimension,
    )
    loads = tuple(
        np.linspace(0.0, float(scale), dimension)
        for scale in range(1, 7)
    )

    started = perf_counter()
    expected = []
    for load in loads:
        constrained_stiffness, constrained_load = apply_dirichlet(
            stiffness,
            load,
            boundary,
        )
        displacement = solvers.linear.solve(
            constrained_stiffness,
            constrained_load,
        )
        expected.append(
            (
                displacement,
                stiffness @ displacement - load,
            )
        )
    legacy_seconds = perf_counter() - started

    cache = static_linear._FactorizationCache(stiffness)
    started = perf_counter()
    actual = []
    for load in loads:
        displacement, free_dofs = cache.solve(
            load,
            pattern,
            values,
        )
        reactions = stiffness @ displacement - load
        static_linear._validate_free_dof_equilibrium(
            reactions,
            load,
            free_dofs,
        )
        actual.append((displacement, reactions))
    reduced_seconds = perf_counter() - started

    for (actual_u, actual_rf), (expected_u, expected_rf) in zip(
        actual,
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(actual_u, expected_u)
        np.testing.assert_allclose(actual_rf, expected_rf, atol=1e-10)
    record_property("full_matrix_dimension", dimension)
    record_property("free_matrix_dimension", dimension - len(pattern))
    record_property("repeated_solve_count", len(loads))
    record_property("legacy_full_solve_seconds", legacy_seconds)
    record_property("reduced_cached_solve_seconds", reduced_seconds)


def test_factor_cache_normalizes_singular_direct_solve_error():
    cache = static_linear._FactorizationCache(
        csr_matrix([[1.0, 0.0], [0.0, 0.0]])
    )

    with pytest.raises(
        RuntimeError,
        match="singular or under-constrained",
    ):
        cache.solve(np.ones(2), (), np.empty(0))

    constrained = static_linear._FactorizationCache(
        csr_matrix([[1.0, 0.0], [0.0, 0.0]])
    )
    displacement, _ = constrained.solve(
        np.array([1.0, 0.0]),
        (1,),
        np.array([0.0]),
    )
    np.testing.assert_allclose(displacement, [1.0, 0.0])


@pytest.mark.parametrize(
    ("prescribed", "expected_exception", "message"),
    [
        ({6: 0.0}, IndexError, r"out of bounds \[0, 6\)"),
        ({1.5: 0.0}, TypeError, "DOF index must be an integer"),
        ({True: 0.0}, TypeError, "DOF index must be an integer"),
        ({0: np.nan}, ValueError, "must be finite"),
    ],
)
def test_reduced_solver_validates_prescribed_mapping(
    prescribed,
    expected_exception,
    message,
):
    boundary = BoundaryCondition()
    boundary.prescribed_displacements = prescribed

    with pytest.raises(expected_exception, match=message):
        static_linear._validated_prescribed_displacements(boundary, 6)


def test_reduced_solver_rejects_duplicate_normalized_constraint_dofs():
    class DuplicateItems(dict):
        def items(self):
            return ((0, 0.0), (np.int64(0), 1.0))

    boundary = BoundaryCondition()
    boundary.prescribed_displacements = DuplicateItems()

    with pytest.raises(ValueError, match="repeats DOF index 0"):
        static_linear._validated_prescribed_displacements(boundary, 2)


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


def test_static_validation_only_checks_the_selected_step_references():
    model = make_two_step_static_pull_truss_model()
    model.steps[2].cloads = (NodalLoad("MISSING_IN_PULL2", 1, 1.0),)

    assert static_linear.validate_problem(model, "pull1") is model.steps[1]
    result = static_linear.solve(model, steps=("pull1",))
    assert result.results[0].step is model.steps[1]

    with pytest.raises(KeyError, match="MISSING_IN_PULL2"):
        static_linear.validate_problem(model, "pull2")


def test_static_linear_stiffness_preflight_detects_free_rigid_dofs():
    model = make_static_pull_truss_model()

    assert static_linear.validate_stiffness(model, "pull").name == "pull"

    model.steps[0].boundaries = model.steps[0].boundaries[:1]
    with pytest.raises(ValueError) as captured:
        static_linear.validate_stiffness(model, "pull")

    assert str(captured.value) == (
        "模型约束不足或刚度矩阵奇异；"
        "请检查刚体位移、材料、截面和单元连接"
    )
    assert captured.value.__cause__ is not None


def test_static_linear_solve_preserves_singular_error_summary_and_cause():
    model = make_static_pull_truss_model()
    model.steps[0].boundaries = model.steps[0].boundaries[:1]

    with pytest.raises(RuntimeError) as captured:
        static_linear.solve(model, "pull")

    assert str(captured.value) == (
        "sparse linear solve failed: stiffness matrix "
        "is singular or under-constrained."
    )
    assert captured.value.__cause__ is not None


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
