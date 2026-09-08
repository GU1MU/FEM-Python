import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.boundary import step as boundary_step
from fem.boundary.condition import BoundaryCondition
from fem.solvers import linear, static_linear
from tests.helpers.model_builders import make_static_pull_truss_model


def test_linear_solver_solves_sparse_system_and_rejects_dense_input():
    stiffness = csr_matrix([[2.0, -1.0], [-1.0, 3.0]])
    load = np.array([4.0, 3.0])

    displacement = linear.solve(stiffness, load)

    np.testing.assert_allclose(displacement, [3.0, 2.0])
    with pytest.raises(TypeError, match="csr_matrix"):
        linear.solve(stiffness.toarray(), load)


@pytest.mark.parametrize("invalid_target", ["K", "F"])
def test_linear_solver_rejects_nonfinite_system_data(invalid_target):
    stiffness = csr_matrix(np.eye(2))
    load = np.ones(2)
    if invalid_target == "K":
        stiffness.data[0] = np.nan
    else:
        load[0] = np.inf

    with pytest.raises(ValueError, match=rf"{invalid_target}.*finite"):
        linear.solve(stiffness, load)


def test_linear_solver_reports_singular_system_with_cause():
    stiffness = csr_matrix([[1.0, 0.0], [0.0, 0.0]])

    with pytest.raises(RuntimeError, match="singular") as caught:
        linear.solve(stiffness, np.ones(2))

    assert caught.value.__cause__ is not None


@pytest.mark.parametrize(
    ("operation", "error", "message"),
    [
        (static_linear.solve, RuntimeError, "singular or under-constrained"),
        (static_linear.validate_stiffness, ValueError, "约束不足|奇异"),
    ],
    ids=["solve", "preflight"],
)
def test_static_solver_reports_free_rigid_dofs(operation, error, message):
    model = make_static_pull_truss_model()
    model.steps[0].boundaries = model.steps[0].boundaries[:1]

    with pytest.raises(error, match=message) as caught:
        operation(model, "pull")

    assert caught.value.__cause__ is not None


def test_stiffness_preflight_distinguishes_a_numerically_null_mode(monkeypatch):
    model = make_static_pull_truss_model()
    model.steps[0].boundaries = model.steps[0].boundaries[:1]
    stiffness = np.eye(6)
    stiffness[3:5, 3:5] = (
        (1.0, 1.0 - 1.0e-15),
        (1.0 - 1.0e-15, 1.0),
    )
    monkeypatch.setattr(
        static_linear, "assemble_global_stiffness_sparse",
        lambda _mesh: csr_matrix(stiffness),
    )

    with pytest.raises(ValueError, match="约束不足|奇异") as caught:
        static_linear.validate_stiffness(model, "pull")

    assert "numerically null free mode" in str(caught.value.__cause__)


@pytest.mark.parametrize(
    ("operation", "stage"),
    [
        (static_linear.solve, "factorization"),
        (static_linear.solve, "solve"),
        (static_linear.validate_stiffness, "factorization"),
    ],
    ids=["solve-factorization", "solve-substitution", "preflight-factorization"],
)
def test_static_solver_preserves_memory_failure_classification_and_cause(
    monkeypatch, operation, stage,
):
    native_error = MemoryError("native allocation failed")

    def fail():
        raise static_linear._PardisoSPDMemoryError("insufficient memory") from native_error

    class MemoryFailingFactor:
        def solve(self, _rhs):
            fail()

        def close(self):
            pass

    def factorize(_matrix):
        if stage == "factorization":
            fail()
        return MemoryFailingFactor()

    monkeypatch.setattr(static_linear, "factorize_spd", factorize)

    with pytest.raises(RuntimeError, match="insufficient memory") as caught:
        operation(make_static_pull_truss_model(), "pull")

    assert "singular" not in str(caught.value)
    assert isinstance(caught.value.__cause__, static_linear._PardisoSPDMemoryError)
    assert caught.value.__cause__.__cause__ is native_error


@pytest.mark.parametrize(
    ("prescribed", "error", "message"),
    [
        ({6: 0.0}, IndexError, "out of bounds"),
        ({True: 0.0}, TypeError, "DOF index.*integer"),
        ({1.5: 0.0}, TypeError, "DOF index.*integer"),
        ({0: np.nan}, ValueError, "finite"),
    ],
    ids=["out-of-range", "boolean-index", "fractional-index", "nonfinite-value"],
)
def test_static_solve_rejects_invalid_resolved_displacements(
    monkeypatch, prescribed, error, message,
):
    boundary = BoundaryCondition()
    boundary.prescribed_displacements = prescribed
    # Inject malformed boundary output at the integration boundary, then exercise solve.
    monkeypatch.setattr(boundary_step, "boundary_for_step", lambda *_args: boundary)

    with pytest.raises(error, match=message):
        static_linear.solve(make_static_pull_truss_model(), "pull")
