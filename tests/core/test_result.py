import numpy as np
import pytest

from fem.core.result import ModelResult, ModelResults
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
)


@pytest.fixture
def result():
    model = make_static_pull_truss_model()
    displacement = np.arange(model.mesh.num_dofs, dtype=float)
    return ModelResult(model, model.steps[0], displacement, -displacement)


def test_result_owns_displacement_and_reaction_vectors():
    model = make_static_pull_truss_model()
    num_dofs = model.mesh.num_dofs
    displacement = np.arange(num_dofs, dtype=float)
    reactions = -displacement

    result = ModelResult(model, model.steps[0], displacement, reactions)
    displacement[:] = 99.0
    reactions[:] = 88.0

    np.testing.assert_array_equal(result.U, np.arange(num_dofs, dtype=float))
    np.testing.assert_array_equal(result.reactions, -np.arange(num_dofs, dtype=float))


def test_result_queries_one_based_nodal_components(result):
    assert result.nodal_displacement(2, component=2) == 4.0
    assert result.nodal_reaction(2, component=2) == -4.0


@pytest.mark.parametrize(
    ("query", "component", "error", "message"),
    [
        ("nodal_displacement", True, TypeError, "component must be an integer"),
        ("nodal_displacement", 1.0, TypeError, "component must be an integer"),
        ("nodal_displacement", "1", TypeError, "component must be an integer"),
        ("nodal_reaction", 0, IndexError, "components are 1-based"),
        ("nodal_reaction", 4, IndexError, "components are 1-based"),
    ],
    ids=("boolean", "float", "string", "below-range", "above-range"),
)
def test_result_nodal_queries_reject_invalid_components(
    result, query, component, error, message,
):
    with pytest.raises(error, match=message):
        getattr(result, query)(2, component=component)


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("U", np.zeros((6, 1)), "U must be one-dimensional"),
        ("U", np.zeros(5), "U must have length 6"),
        ("reactions", np.full(6, np.nan), "reactions must contain only finite"),
    ],
    ids=("two-dimensional", "wrong-length", "nonfinite"),
)
def test_result_rejects_invalid_vectors(result, field, values, message):
    vectors = {"U": result.U, "reactions": result.reactions}
    vectors[field] = values

    with pytest.raises(ValueError, match=message):
        ModelResult(result.model, result.step, **vectors)


def test_results_sequence_preserves_result_identity_and_order():
    model = make_two_step_static_pull_truss_model()
    first, second = (
        ModelResult(
            model,
            step,
            np.full(model.mesh.num_dofs, value),
            np.zeros(model.mesh.num_dofs),
        )
        for step, value in zip(model.steps[1:], (0.5, 1.0), strict=True)
    )
    results = ModelResults(model, (first, second))

    assert len(results) == 2
    assert all(
        actual is expected
        for actual, expected in zip(results, (first, second), strict=True)
    )
    assert results[0] is first
    assert results[1] is second
    assert results[:] == (first, second)
    assert tuple(item.step.name for item in results) == ("pull1", "pull2")
    assert tuple(item.U[3] for item in results) == (0.5, 1.0)
