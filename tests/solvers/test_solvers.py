from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import perf_counter, sleep

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from fem import solvers
from fem.boundary.condition import BoundaryCondition
from fem.boundary.constraints import apply_dirichlet
from fem.core.model import AnalysisStep, DisplacementConstraint, NodalLoad
from fem.solvers import static_linear
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
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
    original_factor = static_linear.factorize_spd

    def factor(stiffness):
        nonlocal factor_calls
        factor_calls += 1
        return original_factor(stiffness)

    monkeypatch.setattr(static_linear, "factorize_spd", factor)
    prepared = static_linear.prepare(model)

    results = prepared._solve_owned(steps=(first, second))

    assert factor_calls == 1
    assert results.results[0].U[0] == pytest.approx(0.1)
    assert results.results[1].U[0] == pytest.approx(-0.2)


def test_factor_cache_lru_is_bounded_and_closes_evicted_pattern_once(
    monkeypatch,
):
    stiffness = csr_matrix(np.diag(np.arange(1.0, 7.0)))
    cache = static_linear._FactorizationCache(stiffness)
    factors = []
    events = []
    live_factors = 0
    maximum_live_factors = 0
    original_factor = static_linear.factorize_spd

    class TrackedFactor:
        def __init__(self, factor, factor_id):
            self.factor = factor
            self.factor_id = factor_id
            self.close_calls = 0

        def solve(self, rhs):
            return self.factor.solve(rhs)

        def close(self):
            nonlocal live_factors
            self.close_calls += 1
            self.factor.close()
            live_factors -= 1
            events.append(("close", self.factor_id))

    def factor(matrix):
        nonlocal live_factors, maximum_live_factors
        factor_id = len(factors)
        events.append(("create", factor_id))
        assert live_factors == 0
        tracked = TrackedFactor(original_factor(matrix), factor_id)
        factors.append(tracked)
        live_factors += 1
        maximum_live_factors = max(maximum_live_factors, live_factors)
        return tracked

    monkeypatch.setattr(static_linear, "factorize_spd", factor)
    patterns = tuple((dof,) for dof in range(5))
    for pattern in patterns:
        cache.factor_for(pattern)

    assert len(factors) == 5
    assert static_linear._FACTOR_CACHE_MAX_ENTRIES == 1
    assert len(cache._entries) == static_linear._FACTOR_CACHE_MAX_ENTRIES
    assert tuple(cache._entries) == patterns[-1:]
    assert [factor.close_calls for factor in factors] == [1, 1, 1, 1, 0]
    assert events == [
        ("create", 0),
        ("close", 0),
        ("create", 1),
        ("close", 1),
        ("create", 2),
        ("close", 2),
        ("create", 3),
        ("close", 3),
        ("create", 4),
    ]
    assert maximum_live_factors == 1

    def failing_factor(_matrix):
        events.append(("fail", len(factors)))
        assert live_factors == 0
        raise ValueError("replacement factorization failed")

    monkeypatch.setattr(static_linear, "factorize_spd", failing_factor)
    with pytest.raises(ValueError, match="replacement factorization failed"):
        cache.factor_for(patterns[0])
    assert tuple(cache._entries) == ()
    assert [factor.close_calls for factor in factors] == [1, 1, 1, 1, 1]

    monkeypatch.setattr(static_linear, "factorize_spd", factor)
    cache.factor_for(patterns[0])
    assert len(factors) == 6
    assert tuple(cache._entries) == patterns[:1]
    assert [factor.close_calls for factor in factors] == [1, 1, 1, 1, 1, 0]

    cache.close()
    cache.close()
    assert [factor.close_calls for factor in factors] == [1, 1, 1, 1, 1, 1]
    assert live_factors == 0


def test_factor_cache_serializes_concurrent_factor_creation(monkeypatch):
    stiffness = csr_matrix(np.diag(np.arange(1.0, 9.0)))
    cache = static_linear._FactorizationCache(stiffness)
    factor_calls = 0
    active_solves = 0
    maximum_active_solves = 0
    close_calls = 0
    counter_lock = Lock()

    class IdentityFactor:
        def solve(self, rhs):
            nonlocal active_solves, maximum_active_solves
            with counter_lock:
                active_solves += 1
                maximum_active_solves = max(
                    maximum_active_solves,
                    active_solves,
                )
            try:
                sleep(0.01)
                return np.asarray(rhs)
            finally:
                with counter_lock:
                    active_solves -= 1

        def close(self):
            nonlocal close_calls
            close_calls += 1

    def factor(_matrix):
        nonlocal factor_calls
        factor_calls += 1
        return IdentityFactor()

    monkeypatch.setattr(static_linear, "factorize_spd", factor)

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
    assert maximum_active_solves == 1
    for scale, displacement in enumerate(results, start=1):
        np.testing.assert_allclose(
            displacement[1:],
            scale / np.arange(2.0, 9.0),
        )
    cache.close()
    assert close_calls == 1


def test_factor_cache_handles_fully_constrained_and_unconstrained_systems(
    monkeypatch,
):
    stiffness = csr_matrix(np.diag([2.0, 3.0]))
    factor_calls = 0
    close_calls = 0

    class IdentityFactor:
        def solve(self, rhs):
            return np.asarray(rhs)

        def close(self):
            nonlocal close_calls
            close_calls += 1

    def factor(_matrix):
        nonlocal factor_calls
        factor_calls += 1
        return IdentityFactor()

    monkeypatch.setattr(static_linear, "factorize_spd", factor)
    fully_constrained = static_linear._FactorizationCache(stiffness)
    displacement, free_dofs = fully_constrained.solve(
        np.array([10.0, 20.0]),
        (0, 1),
        np.array([0.25, -0.5]),
    )

    np.testing.assert_allclose(displacement, [0.25, -0.5])
    assert free_dofs.size == 0
    assert factor_calls == 0
    fully_constrained.close()
    fully_constrained.close()
    assert close_calls == 0

    unconstrained = static_linear._FactorizationCache(stiffness)
    displacement, free_dofs = unconstrained.solve(
        np.array([4.0, 9.0]),
        (),
        np.empty(0),
    )
    np.testing.assert_allclose(displacement, [2.0, 3.0])
    np.testing.assert_array_equal(free_dofs, [0, 1])
    assert factor_calls == 1
    unconstrained.close()
    unconstrained.close()
    assert close_calls == 1


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


@pytest.mark.parametrize("stage", ["factorization", "solve"])
def test_factor_cache_preserves_pardiso_memory_failure(
    monkeypatch,
    stage,
):
    native_error = MemoryError("native allocation failed")

    def raise_memory_error(message):
        raise static_linear._PardisoSPDMemoryError(message) from native_error

    class MemoryFailingFactor:
        def solve(self, _rhs):
            raise_memory_error("PARDISO SPD solve failed: insufficient memory")

    def factorize(_matrix):
        if stage == "factorization":
            raise_memory_error(
                "PARDISO SPD factorization failed: insufficient memory"
            )
        return MemoryFailingFactor()

    monkeypatch.setattr(static_linear, "factorize_spd", factorize)
    cache = static_linear._FactorizationCache(csr_matrix(np.eye(2)))

    with pytest.raises(
        RuntimeError,
        match="PARDISO SPD solver failed: insufficient memory",
    ) as caught:
        cache.solve(np.ones(2), (), np.empty(0))

    assert "singular" not in str(caught.value)
    assert isinstance(
        caught.value.__cause__,
        static_linear._PardisoSPDMemoryError,
    )
    assert caught.value.__cause__.__cause__ is native_error


def test_stiffness_preflight_preserves_pardiso_memory_failure(monkeypatch):
    native_error = MemoryError("native allocation failed")

    def factorize(_matrix):
        raise static_linear._PardisoSPDMemoryError(
            "PARDISO SPD factorization failed: insufficient memory"
        ) from native_error

    monkeypatch.setattr(static_linear, "factorize_spd", factorize)

    with pytest.raises(
        RuntimeError,
        match="PARDISO SPD solver failed: insufficient memory",
    ) as caught:
        static_linear.validate_stiffness(
            make_static_pull_truss_model(),
            "pull",
        )

    assert caught.value.__cause__.__cause__ is native_error


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


def test_static_linear_stiffness_preflight_rejects_near_null_mode(
    monkeypatch,
):
    model = make_static_pull_truss_model()
    model.steps[0].boundaries = model.steps[0].boundaries[:1]
    stiffness = np.eye(6)
    stiffness[3:5, 3:5] = (
        (1.0, 1.0 - 1.0e-15),
        (1.0 - 1.0e-15, 1.0),
    )
    monkeypatch.setattr(
        static_linear,
        "assemble_global_stiffness_sparse",
        lambda _mesh: csr_matrix(stiffness),
    )

    with pytest.raises(ValueError) as captured:
        static_linear.validate_stiffness(model, "pull")

    assert str(captured.value) == (
        "模型约束不足或刚度矩阵奇异；"
        "请检查刚体位移、材料、截面和单元连接"
    )
    assert "numerically null free mode" in str(captured.value.__cause__)


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
        solvers.linear.solve(K, F)
