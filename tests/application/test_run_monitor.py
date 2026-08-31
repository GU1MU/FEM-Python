from __future__ import annotations

import numpy as np

from fem.application import AttemptStatus, RunDiagnostics
from fem.model import DofSpace
from fem.problem import ProblemContext, ProblemEvaluation
from fem.analysis import incremental
from fem.state import SolutionState


class _OneDProblem:
    @property
    def dof_space(self):
        return DofSpace.single_field("U", 1)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        if context.solution is None:
            raise ValueError("one-dimensional problem requires a solution")
        displacement = float(context.solution.values[0])
        target = float(context.load_factor)
        return ProblemEvaluation(
            residual=np.array([displacement - target]),
            tangent=np.array([[1.0]]),
        )


def test_run_diagnostics_keeps_cutback_and_accepted_attempts_separate():
    monitor = RunDiagnostics(
        "run-1",
        "Job-1",
        "Step-1",
        analysis_type="完整分析",
        procedure="Static, General",
        nlgeom=True,
    )
    monitor.start()
    monitor.increment_started(1, 1, 0.5, 0.0)
    monitor.newton_iteration(1, 1, 1, 1.0, None)
    monitor.increment_failed(
        1,
        1,
        1.0,
        "det(F) <= 0",
        retryable=True,
    )
    monitor.cutback(1, 1, 0.5, 0.25)
    monitor.increment_started(1, 2, 0.25, 0.0)
    monitor.newton_iteration(1, 2, 1, 0.1, None)
    monitor.newton_iteration(1, 2, 2, 1.0e-10, 1.0e-8, trial_scale=0.5)
    monitor.increment_converged(1, 2, 0.25, 2, 1.0e-10)

    snapshot = monitor.snapshot()

    assert len(snapshot.attempts) == 2
    assert snapshot.attempts[0].status is AttemptStatus.CUTBACK
    assert snapshot.attempts[0].cutback_to == 0.25
    assert snapshot.attempts[0].result_frame is None
    assert snapshot.attempts[1].status is AttemptStatus.CONVERGED
    assert snapshot.attempts[1].result_frame == 1
    assert len(snapshot.attempts[1].newton_history) == 2
    assert snapshot.attempts[1].newton_history[-1].trial_scale == 0.5
    assert any("试探步缩放" in message.text for message in snapshot.messages)
    assert snapshot.completed_increments == 1
    assert snapshot.last_converged_load_factor == 0.25


def test_incremental_solver_emits_one_monitor_row_per_converged_increment():
    monitor = RunDiagnostics("run-1", "Job-1", "Step-1")
    monitor.start()

    result = incremental.solve(
        _OneDProblem(),
        (0.5, 1.0),
        initial_state=SolutionState.zeros(_OneDProblem().dof_space),
        monitor=monitor,
    )

    snapshot = monitor.snapshot()
    assert len(result.increments) == 2
    assert len(snapshot.attempts) == 2
    assert [row.status for row in snapshot.attempts] == [
        AttemptStatus.CONVERGED,
        AttemptStatus.CONVERGED,
    ]
    assert [row.result_frame for row in snapshot.attempts] == [1, 2]
    assert all(row.newton_history for row in snapshot.attempts)


def test_run_diagnostics_keeps_dynamic_time_and_energy_history():
    monitor = RunDiagnostics(
        "run-dynamic",
        "Dynamic-1",
        "Dynamic-Step",
        analysis_type="瞬态动力学",
        procedure="Dynamic, Implicit",
    )
    monitor.start()
    monitor.dynamic_increment_started(
        1,
        0.01,
        0.01,
        solver_kind="dynamic_implicit",
        stable_time_increment=0.02,
    )
    monitor.dynamic_increment_converged(
        1,
        0.01,
        0.01,
        1.0e-9,
        solver_kind="dynamic_implicit",
        iterations=4,
        stable_time_increment=0.02,
        kinetic_energy=1.0,
        internal_energy=2.0,
        external_work=3.0,
        total_energy=3.0,
        energy_balance_error=0.0,
    )

    snapshot = monitor.snapshot()

    assert len(snapshot.dynamic_increments) == 1
    record = snapshot.dynamic_increments[0]
    assert record.solver_kind == "dynamic_implicit"
    assert record.step_time == 0.01
    assert record.time_increment == 0.01
    assert record.stable_time_increment == 0.02
    assert record.iterations == 4
    assert record.total_energy == 3.0
    assert snapshot.current_step_time == 0.01
    assert snapshot.current_total_energy == 3.0


def test_run_diagnostics_marks_an_active_dynamic_increment_failed():
    monitor = RunDiagnostics("run-dynamic", "Dynamic-1", "Dynamic-Step")
    monitor.start()
    monitor.dynamic_increment_started(
        1,
        0.01,
        0.01,
        solver_kind="dynamic_explicit",
    )
    monitor.failed("稳定时间步小于最小时间增量")

    record = monitor.snapshot().dynamic_increments[-1]

    assert record.status == "failed"
    assert record.error == "稳定时间步小于最小时间增量"
