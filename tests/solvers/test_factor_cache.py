import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.core.model import AnalysisStep, DisplacementConstraint, NodalLoad
from fem.solvers import static_linear
from tests.helpers.model_builders import make_two_step_static_pull_truss_model


@pytest.fixture
def tracked_factors(monkeypatch):
    factors = []
    events = []
    original_factor = static_linear.factorize_spd

    class TrackedFactor:
        def __init__(self, factor):
            self.factor = factor
            self.index = len(factors)
            self.close_calls = 0

        def solve(self, rhs):
            return self.factor.solve(rhs)

        def close(self):
            self.close_calls += 1
            self.factor.close()
            events.append(("close", self.index))

    def factor(matrix):
        tracked = TrackedFactor(original_factor(matrix))
        factors.append(tracked)
        events.append(("create", tracked.index))
        return tracked

    monkeypatch.setattr(static_linear, "factorize_spd", factor)
    try:
        yield factors, events
    finally:
        for tracked in factors:
            if tracked.close_calls == 0:
                tracked.close()


def test_prepared_factor_reuses_constraint_pattern_across_loads_and_values(tracked_factors):
    factors, _ = tracked_factors
    first = AnalysisStep(
        "first", boundaries=[DisplacementConstraint("FIXED", 1, 1, 0.1)],
        cloads=[NodalLoad("TIP", 1, 10.0)],
    )
    second = AnalysisStep(
        "second", boundaries=[DisplacementConstraint("FIXED", 1, 1, -0.2)],
        cloads=[NodalLoad("TIP", 1, 75.0)],
    )
    prepared = static_linear.prepare(make_two_step_static_pull_truss_model())
    try:
        results = prepared.solve(steps=(first, second))
        assert len(factors) == 1
        for result, expected in zip(results, [(0.1, 0.15), (-0.2, 0.175)]):
            mesh = result.model.mesh
            np.testing.assert_allclose(
                result.U[[mesh.global_dof(1, 0), mesh.global_dof(2, 0)]], expected
            )
    finally:
        del prepared
        gc.collect()
    assert factors[0].close_calls == 1


def test_factor_cache_evicts_least_recently_used_before_replacement_and_retries(
    monkeypatch, tracked_factors
):
    factors, events = tracked_factors
    cache = static_linear._FactorizationCache(csr_matrix(np.diag([2., 3., 4., 5.])), max_entries=2)
    factor = static_linear.factorize_spd
    try:
        cache.factor_for((0,))
        cache.factor_for((1,))
        cache.factor_for((0,))  # Refresh the first pattern, so the second is oldest.
        cache.factor_for((2,))
        assert events == [("create", 0), ("create", 1), ("close", 1), ("create", 2)]

        def fail(_matrix):
            events.append(("fail", 3))
            raise ValueError("replacement factorization failed")

        monkeypatch.setattr(static_linear, "factorize_spd", fail)
        with pytest.raises(ValueError, match="replacement factorization failed"):
            cache.factor_for((1,))
        assert events[-2:] == [("close", 0), ("fail", 3)]
        monkeypatch.setattr(static_linear, "factorize_spd", factor)
        cache.factor_for((1,))
        cache.factor_for((2,))
        assert len(factors) == 4
        assert [factor.close_calls for factor in factors] == [1, 1, 0, 0]
    finally:
        cache.close()
        cache.close()
    assert [factor.close_calls for factor in factors] == [1, 1, 1, 1]


def test_factor_cache_serializes_concurrent_creation_and_solves(monkeypatch):
    cache = static_linear._FactorizationCache(csr_matrix(np.diag([2., 3., 4.])))
    start = Barrier(4, timeout=2)
    lock = Lock()
    factor_calls = active_solves = maximum_active_solves = close_calls = 0

    class IdentityFactor:
        def solve(self, rhs):
            nonlocal active_solves, maximum_active_solves
            with lock:
                active_solves += 1
                maximum_active_solves = max(maximum_active_solves, active_solves)
            try:
                # Give competing callers a chance to overlap if the cache loses its lock.
                sleep(0.01)
                return np.asarray(rhs)
            finally:
                with lock:
                    active_solves -= 1

        def close(self):
            nonlocal close_calls
            close_calls += 1

    def factor(_matrix):
        nonlocal factor_calls
        with lock:
            factor_calls += 1
        return IdentityFactor()

    def solve(scale):
        start.wait(timeout=2)
        return cache.solve(scale * np.ones(3), (0,), np.array([0.0]))[0]

    monkeypatch.setattr(static_linear, "factorize_spd", factor)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = tuple(executor.map(solve, range(1, 5), timeout=10))
        assert factor_calls == 1
        assert maximum_active_solves == 1
        for scale, displacement in enumerate(results, start=1):
            np.testing.assert_allclose(displacement, [0., scale / 3., scale / 4.])
    finally:
        start.abort()
        cache.close()
    assert close_calls == 1


def test_factor_cache_handles_fully_constrained_and_unconstrained_systems(tracked_factors):
    factors, _ = tracked_factors
    stiffness = csr_matrix([[2., 0.], [0., 3.]])
    for pattern, values, load, expected, expected_free in [
        ((0, 1), [0.25, -0.5], [10., 20.], [0.25, -0.5], []),
        ((), [], [4., 9.], [2., 3.], [0, 1]),
    ]:
        cache = static_linear._FactorizationCache(stiffness)
        try:
            displacement, free_dofs = cache.solve(np.array(load), pattern, np.array(values))
            np.testing.assert_allclose(displacement, expected)
            np.testing.assert_array_equal(free_dofs, expected_free)
            assert len(factors) == (1 if expected_free else 0)
        finally:
            cache.close()
            cache.close()
    assert factors[0].close_calls == 1


def test_repeated_reduced_solve_matches_independent_coupled_system_oracle():
    stiffness = csr_matrix([[2., -1., 0., 0.], [-1., 2., -1., 0.],
                            [0., -1., 2., -1.], [0., 0., -1., 2.]])
    cache = static_linear._FactorizationCache(stiffness)
    try:
        # Fixed endpoints (1, -1) leave a coupled two-DOF system.
        for load, expected_u, expected_reactions in [
            ([0., 0., 0., 0.], [1., 1/3, -1/3, -1.], [5/3, 0., 0., -5/3]),
            ([0., 1., 2., 0.], [1., 5/3, 4/3, -1.], [1/3, 0., 0., -10/3]),
        ]:
            displacement, free = cache.solve(np.array(load), (0, 3), np.array([1., -1.]))
            np.testing.assert_allclose(displacement, expected_u)
            np.testing.assert_array_equal(free, [1, 2])
            np.testing.assert_allclose(stiffness @ displacement - load, expected_reactions, atol=1e-14)
    finally:
        cache.close()


def test_prepared_factor_is_reused_across_preflight_solves_and_clones(tracked_factors):
    factors, _ = tracked_factors
    prepared = static_linear.prepare(make_two_step_static_pull_truss_model())
    cloned = None
    try:
        prepared.validate_stiffness("pull1")
        first = prepared.solve("pull1")
        second = prepared.solve("pull2")
        cloned = prepared.clone()
        third = cloned.solve("pull1")
        assert len(factors) == 1
        for result, expected in [(first, 0.5), (second, 1.0), (third, 0.5)]:
            assert result.U[result.model.mesh.global_dof(2, 0)] == pytest.approx(expected)
    finally:
        del prepared, cloned
        gc.collect()
    assert factors[0].close_calls == 1


def test_prepared_clone_keeps_factor_alive_until_last_owner(tracked_factors):
    factors, _ = tracked_factors
    prepared = static_linear.prepare(make_two_step_static_pull_truss_model())
    cloned = None
    try:
        prepared.validate_stiffness("pull1")
        cloned = prepared.clone()
        prepared = None
        gc.collect()
        assert len(factors) == 1
        assert factors[0].close_calls == 0
        result = cloned.solve("pull2")
        assert result.U[result.model.mesh.global_dof(2, 0)] == pytest.approx(1.)
        assert factors[0].close_calls == 0
    finally:
        del prepared, cloned
        gc.collect()
    assert factors[0].close_calls == 1
