"""Structured solve diagnostics consumed by the Job Manager UI.

The numerical solvers report through the small :class:`SolveMonitor` protocol
instead of importing any GUI type.  ``RunDiagnostics`` keeps a thread-safe
snapshot so the GUI can refresh it from its normal Qt timer while a worker
thread is solving.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import math
from threading import RLock
from time import monotonic
from typing import Protocol


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AttemptStatus(str, Enum):
    """Status shown for one increment attempt in the monitor table."""

    RUNNING = "running"
    CONVERGED = "converged"
    CUTBACK = "cutback"
    FAILED = "failed"


class SolveMonitor(Protocol):
    """Optional numerical progress sink used by Newton/incremental solvers."""

    def increment_started(
        self,
        increment: int,
        attempt: int,
        target_load_factor: float,
        previous_load_factor: float,
    ) -> None: ...

    def newton_iteration(
        self,
        increment: int,
        attempt: int,
        iteration: int,
        residual_norm: float,
        increment_norm: float | None,
        trial_scale: float | None = None,
        force_norm: float | None = None,
        relative_force_norm: float | None = None,
        constraint_norm: float | None = None,
        displacement_norm: float | None = None,
        energy_norm: float | None = None,
    ) -> None: ...

    def increment_failed(
        self,
        increment: int,
        attempt: int,
        residual_norm: float | None,
        error: str,
        *,
        retryable: bool,
        record: object | None = None,
    ) -> None: ...

    def cutback(
        self,
        increment: int,
        attempt: int,
        from_load_factor: float,
        to_load_factor: float,
    ) -> None: ...

    def increment_converged(
        self,
        increment: int,
        attempt: int,
        load_factor: float,
        iterations: int,
        residual_norm: float,
        *,
        record: object | None = None,
    ) -> None: ...

    def dynamic_increment_started(
        self,
        increment: int,
        step_time: float,
        time_increment: float,
        *,
        solver_kind: str,
        attempt: int = 1,
        stable_time_increment: float | None = None,
        critical_element: int | None = None,
    ) -> None: ...

    def dynamic_increment_converged(
        self,
        increment: int,
        step_time: float,
        time_increment: float,
        residual_norm: float,
        *,
        solver_kind: str | None = None,
        iterations: int | None = None,
        stable_time_increment: float | None = None,
        critical_element: int | None = None,
        kinetic_energy: float | None = None,
        internal_energy: float | None = None,
        external_work: float | None = None,
        damping_dissipation: float | None = None,
        total_energy: float | None = None,
        energy_balance_error: float | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class NewtonIterationRecord:
    """One Newton iteration as shown by the Job Monitor."""

    iteration: int
    residual_norm: float
    increment_norm: float | None = None
    trial_scale: float | None = None
    force_norm: float | None = None
    relative_force_norm: float | None = None
    constraint_norm: float | None = None
    displacement_norm: float | None = None
    energy_norm: float | None = None


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One accepted increment attempt or one failed/cutback attempt."""

    increment: int
    attempt: int
    target_load_factor: float
    previous_load_factor: float
    status: AttemptStatus = AttemptStatus.RUNNING
    load_factor: float | None = None
    iterations: int = 0
    residual_norm: float | None = None
    increment_norm: float | None = None
    cutback_to: float | None = None
    result_frame: int | None = None
    error: str | None = None
    duration_seconds: float | None = None
    newton_history: tuple[NewtonIterationRecord, ...] = ()
    force_norm: float | None = None
    relative_force_norm: float | None = None
    constraint_norm: float | None = None
    displacement_norm: float | None = None
    energy_norm: float | None = None
    failure_code: str | None = None
    tangent_strategy: str | None = None
    tangent_evaluations: int | None = None
    line_search_backtracks: int | None = None
    predictor_used: bool = False


@dataclass(frozen=True, slots=True)
class DynamicIncrementRecord:
    """One accepted implicit or explicit dynamic increment."""

    solver_kind: str
    increment: int
    step_time: float
    time_increment: float
    attempt: int = 1
    iterations: int | None = None
    residual_norm: float | None = None
    stable_time_increment: float | None = None
    critical_element: int | None = None
    kinetic_energy: float | None = None
    internal_energy: float | None = None
    external_work: float | None = None
    damping_dissipation: float | None = None
    total_energy: float | None = None
    energy_balance_error: float | None = None
    status: str = "running"
    duration_seconds: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.solver_kind, str) or not self.solver_kind.strip():
            raise ValueError("solver_kind must be a non-empty string")
        if (
            isinstance(self.increment, bool)
            or not isinstance(self.increment, int)
            or self.increment < 1
        ):
            raise ValueError("increment must be an integer >= 1")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("attempt must be an integer >= 1")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must be a non-empty string")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        for name in ("step_time", "time_increment"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        for name in (
            "residual_norm",
            "stable_time_increment",
            "kinetic_energy",
            "internal_energy",
            "external_work",
            "damping_dissipation",
            "total_energy",
            "energy_balance_error",
            "duration_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"{name} must be finite or None")
                if name == "stable_time_increment" and numeric <= 0.0:
                    raise ValueError("stable_time_increment must be > 0")
                if name == "duration_seconds" and numeric < 0.0:
                    raise ValueError("duration_seconds must be >= 0")
                object.__setattr__(self, name, numeric)


@dataclass(frozen=True, slots=True)
class RunMessage:
    """One categorized message shown in the monitor's lower tabs."""

    level: str
    text: str
    increment: int | None = None
    attempt: int | None = None
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class RunDiagnosticsSnapshot:
    """Immutable view of one running or completed job."""

    run_id: str
    name: str
    step_name: str
    analysis_type: str
    model_name: str = "—"
    procedure: str = "Static, General"
    nlgeom: bool = False
    status: str = "pending"
    stage: str = "等待提交"
    current_increment: int | None = None
    current_attempt: int | None = None
    current_iteration: int | None = None
    current_load_factor: float | None = None
    current_residual: float | None = None
    current_step_time: float | None = None
    current_time_increment: float | None = None
    current_stable_time_increment: float | None = None
    current_kinetic_energy: float | None = None
    current_internal_energy: float | None = None
    current_total_energy: float | None = None
    elapsed_seconds: float | None = None
    attempts: tuple[AttemptRecord, ...] = ()
    dynamic_increments: tuple[DynamicIncrementRecord, ...] = ()
    messages: tuple[RunMessage, ...] = ()
    error: str | None = None

    @property
    def completed_increments(self) -> int:
        return sum(
            attempt.status is AttemptStatus.CONVERGED
            for attempt in self.attempts
        )

    @property
    def last_converged_load_factor(self) -> float | None:
        for attempt in reversed(self.attempts):
            if attempt.status is AttemptStatus.CONVERGED:
                return attempt.load_factor
        return None


class RunDiagnostics(SolveMonitor):
    """Thread-safe mutable monitor for one :class:`AnalysisRun`."""

    def __init__(
        self,
        run_id: str,
        name: str,
        step_name: str,
        *,
        analysis_type: str = "—",
        model_name: str = "—",
        procedure: str = "Static, General",
        nlgeom: bool = False,
    ) -> None:
        self._run_id = str(run_id)
        self._name = str(name)
        self._step_name = str(step_name)
        self._analysis_type = str(analysis_type)
        self._model_name = str(model_name)
        self._procedure = str(procedure)
        self._nlgeom = bool(nlgeom)
        self._lock = RLock()
        self._status = "pending"
        self._stage = "等待提交"
        self._started = monotonic()
        self._finished: float | None = None
        self._current_increment: int | None = None
        self._current_attempt: int | None = None
        self._current_iteration: int | None = None
        self._current_load_factor: float | None = None
        self._current_residual: float | None = None
        self._current_step_time: float | None = None
        self._current_time_increment: float | None = None
        self._current_stable_time_increment: float | None = None
        self._current_kinetic_energy: float | None = None
        self._current_internal_energy: float | None = None
        self._current_total_energy: float | None = None
        self._attempts: list[AttemptRecord] = []
        self._dynamic_increments: list[DynamicIncrementRecord] = []
        self._attempt_started: dict[tuple[int, int], float] = {}
        self._messages: list[RunMessage] = []
        self._error: str | None = None

    def start(self) -> None:
        with self._lock:
            self._status = "running"
            self._stage = "准备求解"
            self._message("info", "开始求解")

    def stage_changed(self, text: str) -> None:
        with self._lock:
            self._stage = str(text).strip() or "求解中"
            self._message("info", self._stage)

    def dynamic_increment_started(
        self,
        increment: int,
        step_time: float,
        time_increment: float,
        *,
        solver_kind: str,
        attempt: int = 1,
        stable_time_increment: float | None = None,
        critical_element: int | None = None,
    ) -> None:
        """Start one dynamic increment and expose it to the common monitor."""

        with self._lock:
            previous_time = (
                self._dynamic_increments[-1].step_time
                if self._dynamic_increments
                else 0.0
            )
            # Newton iterations already use the common AttemptRecord protocol;
            # seed one row so implicit dynamics can reuse that history.
            self.increment_started(
                increment,
                attempt,
                float(step_time),
                float(previous_time),
            )
            self._current_step_time = float(step_time)
            self._current_time_increment = float(time_increment)
            self._current_stable_time_increment = (
                None if stable_time_increment is None else float(stable_time_increment)
            )
            self._dynamic_increments.append(
                DynamicIncrementRecord(
                    solver_kind=str(solver_kind),
                    increment=int(increment),
                    step_time=float(step_time),
                    time_increment=float(time_increment),
                    attempt=int(attempt),
                    stable_time_increment=stable_time_increment,
                    critical_element=critical_element,
                )
            )
            self._stage = f"动力学增量 {int(increment)}"
            self._message(
                "info",
                f"开始动力学增量 {int(increment)}，步时间 {float(step_time):g}",
                increment,
                attempt,
            )

    def dynamic_increment_converged(
        self,
        increment: int,
        step_time: float,
        time_increment: float,
        residual_norm: float,
        *,
        solver_kind: str | None = None,
        iterations: int | None = None,
        stable_time_increment: float | None = None,
        critical_element: int | None = None,
        kinetic_energy: float | None = None,
        internal_energy: float | None = None,
        external_work: float | None = None,
        damping_dissipation: float | None = None,
        total_energy: float | None = None,
        energy_balance_error: float | None = None,
    ) -> None:
        """Publish one accepted dynamic frame to the live monitor."""

        with self._lock:
            index = next(
                (
                    index
                    for index in range(len(self._dynamic_increments) - 1, -1, -1)
                    if self._dynamic_increments[index].increment == int(increment)
                ),
                None,
            )
            if index is None:
                self.dynamic_increment_started(
                    increment,
                    step_time,
                    time_increment,
                    solver_kind=solver_kind or "dynamic_implicit",
                )
                index = len(self._dynamic_increments) - 1
            current = self._dynamic_increments[index]
            accepted_iterations = current.iterations if iterations is None else int(iterations)
            self._dynamic_increments[index] = replace(
                current,
                step_time=float(step_time),
                time_increment=float(time_increment),
                iterations=accepted_iterations,
                residual_norm=float(residual_norm),
                stable_time_increment=stable_time_increment,
                critical_element=critical_element,
                kinetic_energy=kinetic_energy,
                internal_energy=internal_energy,
                external_work=external_work,
                damping_dissipation=damping_dissipation,
                total_energy=total_energy,
                energy_balance_error=energy_balance_error,
                status="converged",
            )
            self._current_step_time = float(step_time)
            self._current_time_increment = float(time_increment)
            self._current_stable_time_increment = stable_time_increment
            self._current_kinetic_energy = kinetic_energy
            self._current_internal_energy = internal_energy
            self._current_total_energy = total_energy
            self.increment_converged(
                increment,
                current.attempt,
                float(step_time),
                0 if iterations is None else int(iterations),
                float(residual_norm),
            )
            attempt_index = self._attempt_index(increment, current.attempt)
            duration = (
                None
                if attempt_index is None
                else self._attempts[attempt_index].duration_seconds
            )
            self._dynamic_increments[index] = replace(
                self._dynamic_increments[index],
                duration_seconds=duration,
            )
            self._stage = f"动力学增量 {int(increment)} 已完成"

    def increment_started(
        self,
        increment: int,
        attempt: int,
        target_load_factor: float,
        previous_load_factor: float,
    ) -> None:
        with self._lock:
            self._current_increment = int(increment)
            self._current_attempt = int(attempt)
            self._current_iteration = None
            self._current_load_factor = float(target_load_factor)
            self._current_residual = None
            self._attempts.append(
                AttemptRecord(
                    increment=int(increment),
                    attempt=int(attempt),
                    target_load_factor=float(target_load_factor),
                    previous_load_factor=float(previous_load_factor),
                )
            )
            self._attempt_started[(int(increment), int(attempt))] = monotonic()
            self._stage = f"增量 {int(increment)} / 尝试 {int(attempt)}"
            self._message(
                "info",
                f"开始增量 {int(increment)}，目标载荷因子 "
                f"{float(target_load_factor):g}",
                increment,
                attempt,
            )

    def newton_iteration(
        self,
        increment: int,
        attempt: int,
        iteration: int,
        residual_norm: float,
        increment_norm: float | None,
        trial_scale: float | None = None,
        force_norm: float | None = None,
        relative_force_norm: float | None = None,
        constraint_norm: float | None = None,
        displacement_norm: float | None = None,
        energy_norm: float | None = None,
    ) -> None:
        with self._lock:
            index = self._attempt_index(increment, attempt)
            if index is None:
                return
            record = self._attempts[index]
            history = record.newton_history + (
                NewtonIterationRecord(
                    iteration=int(iteration),
                    residual_norm=float(residual_norm),
                    increment_norm=(
                        None
                        if increment_norm is None
                        else float(increment_norm)
                    ),
                    trial_scale=(
                        None
                        if trial_scale is None
                        else float(trial_scale)
                    ),
                    force_norm=_optional_float(force_norm),
                    relative_force_norm=_optional_float(relative_force_norm),
                    constraint_norm=_optional_float(constraint_norm),
                    displacement_norm=_optional_float(displacement_norm),
                    energy_norm=_optional_float(energy_norm),
                ),
            )
            self._attempts[index] = replace(
                record,
                iterations=max(record.iterations, int(iteration)),
                residual_norm=float(residual_norm),
                increment_norm=(
                    None
                    if increment_norm is None
                    else float(increment_norm)
                ),
                force_norm=_optional_float(force_norm),
                relative_force_norm=_optional_float(relative_force_norm),
                constraint_norm=_optional_float(constraint_norm),
                displacement_norm=_optional_float(displacement_norm),
                energy_norm=_optional_float(energy_norm),
                newton_history=history,
            )
            if trial_scale is not None and float(trial_scale) < 1.0:
                self._message(
                    "warning",
                    f"Newton 试探步缩放为 {float(trial_scale):.3g}",
                    increment,
                    attempt,
                )
            self._current_increment = int(increment)
            self._current_attempt = int(attempt)
            self._current_iteration = int(iteration)
            self._current_residual = float(residual_norm)

    def increment_failed(
        self,
        increment: int,
        attempt: int,
        residual_norm: float | None,
        error: str,
        *,
        retryable: bool,
        record: object | None = None,
    ) -> None:
        with self._lock:
            index = self._attempt_index(increment, attempt)
            if index is None:
                return
            current_record = self._attempts[index]
            text = str(error).strip() or "增量未收敛"
            reported_residual = _record_float(record, "residual_norm")
            reported_duration = _record_float(record, "duration_seconds")
            self._attempts[index] = replace(
                current_record,
                status=(
                    AttemptStatus.CUTBACK
                    if retryable
                    else AttemptStatus.FAILED
                ),
                residual_norm=(
                    current_record.residual_norm
                    if residual_norm is None and reported_residual is None
                    else (
                        reported_residual
                        if reported_residual is not None
                        else float(residual_norm)
                    )
                ),
                error=text,
                duration_seconds=(
                    reported_duration
                    if reported_duration is not None
                    else self._attempt_duration(increment, attempt)
                ),
                force_norm=_record_float(record, "force_norm"),
                relative_force_norm=_record_float(
                    record,
                    "relative_force_norm",
                ),
                constraint_norm=_record_float(record, "constraint_norm"),
                displacement_norm=_record_float(
                    record,
                    "displacement_norm",
                ),
                energy_norm=_record_float(record, "energy_norm"),
                failure_code=_record_str(record, "failure_code"),
                tangent_strategy=_record_str(record, "tangent_strategy"),
                tangent_evaluations=_record_int(
                    record,
                    "tangent_evaluations",
                ),
                line_search_backtracks=_record_int(
                    record,
                    "line_search_backtracks",
                ),
                predictor_used=_record_bool(record, "predictor_used"),
            )
            self._current_residual = (
                current_record.residual_norm
                if residual_norm is None and reported_residual is None
                else (
                    reported_residual
                    if reported_residual is not None
                    else float(residual_norm)
                )
            )
            self._message(
                "warning" if retryable else "error",
                text,
                increment,
                attempt,
            )

    def cutback(
        self,
        increment: int,
        attempt: int,
        from_load_factor: float,
        to_load_factor: float,
    ) -> None:
        with self._lock:
            index = self._attempt_index(increment, attempt)
            if index is not None:
                self._attempts[index] = replace(
                    self._attempts[index],
                    cutback_to=float(to_load_factor),
                )
            self._stage = "自动切步"
            self._message(
                "warning",
                f"自动切步：{float(from_load_factor):g} → "
                f"{float(to_load_factor):g}",
                increment,
                attempt,
            )

    def increment_converged(
        self,
        increment: int,
        attempt: int,
        load_factor: float,
        iterations: int,
        residual_norm: float,
        *,
        record: object | None = None,
    ) -> None:
        with self._lock:
            index = self._attempt_index(increment, attempt)
            if index is None:
                return
            frame = self._completed_count() + 1
            current_record = self._attempts[index]
            reported_load_factor = _record_float(record, "load_factor")
            reported_iterations = _record_int(record, "iterations")
            reported_residual = _record_float(record, "residual_norm")
            reported_duration = _record_float(record, "duration_seconds")
            self._attempts[index] = replace(
                current_record,
                status=AttemptStatus.CONVERGED,
                load_factor=(
                    float(load_factor)
                    if reported_load_factor is None
                    else reported_load_factor
                ),
                iterations=max(
                    current_record.iterations,
                    int(iterations)
                    if reported_iterations is None
                    else reported_iterations,
                ),
                residual_norm=(
                    float(residual_norm)
                    if reported_residual is None
                    else reported_residual
                ),
                result_frame=frame,
                duration_seconds=(
                    reported_duration
                    if reported_duration is not None
                    else self._attempt_duration(increment, attempt)
                ),
                force_norm=_record_float(record, "force_norm"),
                relative_force_norm=_record_float(
                    record,
                    "relative_force_norm",
                ),
                constraint_norm=_record_float(record, "constraint_norm"),
                displacement_norm=_record_float(
                    record,
                    "displacement_norm",
                ),
                energy_norm=_record_float(record, "energy_norm"),
                tangent_strategy=_record_str(record, "tangent_strategy"),
                tangent_evaluations=_record_int(
                    record,
                    "tangent_evaluations",
                ),
                line_search_backtracks=_record_int(
                    record,
                    "line_search_backtracks",
                ),
                predictor_used=_record_bool(record, "predictor_used"),
            )
            self._current_load_factor = (
                float(load_factor)
                if reported_load_factor is None
                else reported_load_factor
            )
            self._current_residual = (
                float(residual_norm)
                if reported_residual is None
                else reported_residual
            )
            self._stage = f"增量 {int(increment)} 已收敛"
            self._message(
                "info",
                f"增量 {int(increment)} 已收敛，Newton {int(iterations)} 次",
                increment,
                attempt,
            )

    def succeeded(self) -> None:
        with self._lock:
            self._status = "succeeded"
            self._stage = "分析完成"
            self._finished = monotonic()
            self._message("info", "分析完成")

    def failed(self, error: str) -> None:
        with self._lock:
            self._status = "failed"
            self._stage = "分析失败"
            self._error = str(error).strip() or "分析失败"
            if self._dynamic_increments:
                latest = self._dynamic_increments[-1]
                if latest.status == "running":
                    self._dynamic_increments[-1] = replace(
                        latest,
                        status="failed",
                        residual_norm=self._current_residual,
                        error=self._error,
                    )
            self._finished = monotonic()
            self._message("error", self._error)

    def cancelled(self) -> None:
        with self._lock:
            self._status = "cancelled"
            self._stage = "已取消"
            self._finished = monotonic()
            self._message("warning", "分析已取消")

    def snapshot(self) -> RunDiagnosticsSnapshot:
        with self._lock:
            end = self._finished or monotonic()
            return RunDiagnosticsSnapshot(
                run_id=self._run_id,
                name=self._name,
                step_name=self._step_name,
                analysis_type=self._analysis_type,
                model_name=self._model_name,
                procedure=self._procedure,
                nlgeom=self._nlgeom,
                status=self._status,
                stage=self._stage,
                current_increment=self._current_increment,
                current_attempt=self._current_attempt,
                current_iteration=self._current_iteration,
                current_load_factor=self._current_load_factor,
                current_residual=self._current_residual,
                current_step_time=self._current_step_time,
                current_time_increment=self._current_time_increment,
                current_stable_time_increment=self._current_stable_time_increment,
                current_kinetic_energy=self._current_kinetic_energy,
                current_internal_energy=self._current_internal_energy,
                current_total_energy=self._current_total_energy,
                elapsed_seconds=max(0.0, end - self._started),
                attempts=tuple(self._attempts),
                dynamic_increments=tuple(self._dynamic_increments),
                messages=tuple(self._messages),
                error=self._error,
            )

    def _attempt_index(self, increment: int, attempt: int) -> int | None:
        for index in range(len(self._attempts) - 1, -1, -1):
            record = self._attempts[index]
            if record.increment == int(increment) and record.attempt == int(attempt):
                return index
        return None

    def _completed_count(self) -> int:
        return sum(
            record.status is AttemptStatus.CONVERGED
            for record in self._attempts
        )

    def _attempt_duration(self, increment: int, attempt: int) -> float:
        started = self._attempt_started.get((int(increment), int(attempt)))
        if started is None:
            return 0.0
        return max(0.0, monotonic() - started)

    def _message(
        self,
        level: str,
        text: str,
        increment: int | None = None,
        attempt: int | None = None,
    ) -> None:
        self._messages.append(
            RunMessage(
                level=str(level),
                text=str(text),
                increment=None if increment is None else int(increment),
                attempt=None if attempt is None else int(attempt),
            )
        )


def snapshot_map(
    monitors: dict[str, RunDiagnostics],
) -> dict[str, RunDiagnosticsSnapshot]:
    """Return detached snapshots for a GUI refresh."""

    return {run_id: monitor.snapshot() for run_id, monitor in monitors.items()}


def _record_float(record: object | None, name: str) -> float | None:
    """Read an optional numeric fact without importing the solver package."""

    if record is None:
        return None
    value = getattr(record, name, None)
    if value is None:
        return None
    return float(value)


def _record_int(record: object | None, name: str) -> int | None:
    """Read an optional integer fact without coupling monitor to solver types."""

    if record is None:
        return None
    value = getattr(record, name, None)
    if value is None:
        return None
    return int(value)


def _record_str(record: object | None, name: str) -> str | None:
    if record is None:
        return None
    value = getattr(record, name, None)
    if value is None:
        return None
    return str(value)


def _record_bool(record: object | None, name: str) -> bool:
    if record is None:
        return False
    value = getattr(record, name, False)
    return bool(value)


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "AttemptRecord",
    "AttemptStatus",
    "DynamicIncrementRecord",
    "NewtonIterationRecord",
    "RunDiagnostics",
    "RunDiagnosticsSnapshot",
    "RunMessage",
    "SolveMonitor",
    "snapshot_map",
]
