"""Prepare and accept analysis commands for their originating documents."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Callable

from fem.application import (
    AnalysisRun,
    ModelSession,
    PreparedPreflight,
    PreparedSolveTask,
    PreflightReport,
    SessionDelta,
    SolvePreparationRequest,
    ValidationTaskSnapshot,
    safe_prepare_static_preflight,
    safe_static_preflight,
)
from fem.application.results import SolveResultBundle, build_solve_result_bundle
from fem.solvers import static_linear
from fem.solvers.static_linear import PreparedSystem

from .commands import GuiCommandCompletion
from .document_tasks import DocumentTaskRouter, session_task_outcome
from .task_controller import BackgroundTaskState, TaskApplyOutcome, TaskCompletion
from .workers import TaskContext
from .workspace import WorkspaceDocument


_NUMERICAL_CHECK_DOF_LIMIT = 50_000
_NUMERICAL_CHECK_ELEMENT_LIMIT = 100_000


@dataclass(frozen=True, slots=True)
class DocumentAnalysisPresentation:
    """Visible analysis operations; task routing decides when they are shown."""

    project_delta: Callable[[WorkspaceDocument, SessionDelta], bool]
    checked: Callable[[PreflightReport], None]
    run_started: Callable[[AnalysisRun], None]
    run_completed: Callable[[AnalysisRun, dict[str, float]], None]
    run_failed: Callable[[AnalysisRun, str], None]
    run_cancelled: Callable[[AnalysisRun], None]


@dataclass(frozen=True, slots=True)
class PreparedSolveResult:
    """Detached output of one solve, including the reusable prepared system."""

    bundle: SolveResultBundle
    timings: dict[str, float]
    prepared_system: PreparedSystem
    cache_candidate: PreparedSystem | None
    prepared_task: PreparedSolveTask


def should_run_numerical_model_check(model: object) -> bool:
    """Keep large-model checks within the existing stiffness-check budget."""

    mesh = getattr(model, "mesh", None)
    if mesh is None:
        return True
    try:
        element_count = len(mesh.elements)
        dof_count = int(mesh.num_dofs)
    except (AttributeError, TypeError, ValueError):
        return True
    return (
        element_count <= _NUMERICAL_CHECK_ELEMENT_LIMIT
        and dof_count <= _NUMERICAL_CHECK_DOF_LIMIT
    )


def prepare_model_check(
    task: ValidationTaskSnapshot, context: TaskContext,
) -> PreparedPreflight:
    full_check = should_run_numerical_model_check(task.model)
    context.report(
        "Checking model..."
        if full_check
        else "Running a quick check for the large model..."
    )
    options = {
        "token": task.token,
        "check_numerical_stability": full_check,
        "copy_model": full_check,
        "quick_check": not full_check,
    }
    result = (
        safe_prepare_static_preflight(task.model, task.step_name, **options)
        if full_check
        else PreparedPreflight(
            safe_static_preflight(task.model, task.step_name, **options)
        )
    )
    context.checkpoint()
    return result


def prepare_solve_result(
    request: SolvePreparationRequest, context: TaskContext,
) -> PreparedSolveResult:
    timings: dict[str, float] = {}
    context.report("Preparing analysis...")
    started = perf_counter()
    prepared_task = ModelSession.prepare_solve_task(request)
    timings["Solve model snapshot"] = perf_counter() - started
    context.checkpoint()
    task = prepared_task.snapshot
    context.report("Assembling and solving...")
    prepared = task.prepared_system
    if prepared is None:
        prepared = static_linear.prepare(
            task.model, copy_model=False, timings=timings,
        )
    result = static_linear.solve(
        task.model,
        task.step_name,
        name=task.run_name,
        _prepared_system=prepared,
        timings=timings,
    )
    context.report("Executing output requests...")
    started = perf_counter()
    bundle = build_solve_result_bundle(task, result, cancellation=context)
    timings["Output requests and initial results"] = perf_counter() - started
    cache_candidate = prepared.clone() if task.prepared_system is None else None
    context.checkpoint()
    return PreparedSolveResult(bundle, timings, prepared, cache_candidate, prepared_task)


class DocumentAnalysisCommands:
    """Own analysis preparation, token acceptance and terminal cleanup."""

    def __init__(
        self, tasks: DocumentTaskRouter, presentation: DocumentAnalysisPresentation,
    ) -> None:
        self.tasks = tasks
        self.presentation = presentation

    def start_check(
        self,
        document: WorkspaceDocument,
        step_name: str,
        *,
        completion: GuiCommandCompletion | None = None,
        expected_session_revision: int | None = None,
    ) -> bool:
        self._require_idle_document(document)
        session = document.session
        if not session.can_check(step_name):
            raise RuntimeError(f"step cannot be checked: {step_name}")
        task = session.prepare_validation(
            step_name,
            detach_model=False,
            **(
                {} if expected_session_revision is None
                else {"expected_session_revision": expected_session_revision}
            ),
        )

        def apply(value: object) -> TaskApplyOutcome:
            if type(value) is not PreparedPreflight:
                raise TypeError("model check must return PreparedPreflight")
            delta = session.accept_validation_with_prepared_system(
                task.token, value.report, value.prepared_system,
            )
            return session_task_outcome(delta, value.report)

        def project(value: object) -> None:
            delta, report = value
            if not self.presentation.project_delta(document, delta):
                raise RuntimeError("Cannot project the accepted model check")
            self.presentation.checked(report)

        def failed(message: str, *, present: bool = True) -> None:
            delta = session.accept_task_failed(task.token, message)
            if not delta.accepted:
                return
            self.presentation.project_delta(document, delta)
            if present and self.tasks.is_active(document):
                self.tasks.presentation.show_error("Model check failed", message)

        def cancelled() -> None:
            delta = session.accept_task_cancelled(task.token)
            if delta.accepted:
                self.presentation.project_delta(document, delta)

        try:
            started = self.tasks.start(
                lambda context: prepare_model_check(task, context),
                project,
                "Model check failed",
                failed,
                context=document,
                controller=document.task_controller,
                task_name="Model Check",
                on_cancelled=cancelled,
                on_inactive_failure=failed,
                on_inactive_cancelled=cancelled,
                apply_result=apply,
                completion=completion,
            )
        except Exception as error:
            failed(str(error).strip() or type(error).__name__, present=False)
            raise
        if not started:
            failed("the model-check task could not be started", present=False)
        return started

    def create_run(
        self, document: WorkspaceDocument, name: str, step_name: str,
    ) -> SessionDelta:
        self._require_idle_document(document)
        name, step_name = self._run_inputs(document, name, step_name, existing=False)
        delta = document.session.create_run(step_name, name)
        self.tasks.workspace.remember_job_name(name)
        return delta

    def submit_run(
        self,
        document: WorkspaceDocument,
        name: str,
        step_name: str,
        *,
        completion: GuiCommandCompletion | None = None,
        expected_session_revision: int | None = None,
        existing_run: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> AnalysisRun | None:
        self._require_idle_document(document)
        session = document.session
        name, step_name = self._run_inputs(
            document, name, step_name, existing=existing_run,
        )
        task = (
            session.prepare_run_solve_request(
                name, expected_session_revision=expected_session_revision,
            )
            if existing_run
            else session.prepare_solve_request(
                step_name, name, expected_session_revision=expected_session_revision,
            )
        )

        def failed(message: str, *, present: bool = True) -> None:
            delta = session.accept_run_failed(task.token, message)
            if not delta.accepted:
                return
            if not self.tasks.is_live(document) or document.session is not session:
                return
            self.presentation.project_delta(document, delta)
            current = session.find_run(task.run_id)
            if present and current is not None and self.tasks.is_active(document):
                self.presentation.run_failed(current, message)

        def cancelled() -> None:
            delta = session.accept_run_cancelled(task.token)
            if not delta.accepted or not self.tasks.is_live(document) or document.session is not session:
                return
            self.presentation.project_delta(document, delta)
            current = session.find_run(task.run_id)
            if current is not None and self.tasks.is_active(document):
                self.presentation.run_cancelled(current)

        def apply(value: object) -> TaskApplyOutcome:
            if type(value) is not PreparedSolveResult:
                raise TypeError("analysis run must return PreparedSolveResult")
            delta = session.accept_prepared_run_succeeded_with_prepared_system(
                task.token,
                value.prepared_task,
                value.bundle,
                value.prepared_system,
                cache_candidate=value.cache_candidate,
                timings=value.timings,
            )
            return session_task_outcome(delta, value.timings)

        def project(value: object) -> None:
            delta, timings = value
            if not self.presentation.project_delta(document, delta):
                raise RuntimeError("Cannot project the accepted solver result")
            completed = session.find_run(task.run_id)
            if completed is None:
                raise RuntimeError("The accepted analysis job does not exist")
            self.presentation.run_completed(completed, timings)

        def terminal(record: TaskCompletion) -> None:
            # Routing can discard a closed/rebound document before its apply
            # callback. Always finish the original reservation in that case.
            if record.state is BackgroundTaskState.SUCCEEDED:
                return
            delta = (
                session.accept_run_cancelled(task.token)
                if record.state in {BackgroundTaskState.CANCELLED, BackgroundTaskState.DISCARDED}
                else session.accept_run_failed(task.token, record.message or record.state.value)
            )
            if delta.accepted and self.tasks.is_live(document) and document.session is session:
                try:
                    self.presentation.project_delta(document, delta)
                except Exception:
                    logging.exception("Cannot project analysis task cleanup")

        try:
            if not existing_run:
                self.tasks.workspace.remember_job_name(name)
            if task.delta is not None:
                self.presentation.project_delta(document, task.delta)
            self.presentation.project_delta(document, session.begin_run(task.token))
            run = session.find_run(task.run_id)
            if run is None:
                raise RuntimeError("The prepared analysis job does not exist")
            if self.tasks.is_active(document):
                self.presentation.run_started(run)
            started = self.tasks.start(
                lambda context: prepare_solve_result(task, context),
                project,
                "Analysis run failed",
                failed,
                context=document,
                controller=document.task_controller,
                task_name=f"Job {run.name}",
                on_cancelled=cancelled,
                on_inactive_failure=failed,
                on_inactive_cancelled=cancelled,
                apply_result=apply,
                completion=completion,
                on_progress=on_progress,
                on_terminal=terminal,
            )
        except Exception as error:
            failed(str(error).strip() or type(error).__name__, present=False)
            raise
        if not started:
            failed("analysis task could not be started", present=False)
            return None
        return run

    def _require_idle_document(self, document: WorkspaceDocument) -> None:
        if not self.tasks.is_live(document):
            raise RuntimeError("No document is available for this task")
        if document.task_controller.busy:
            raise RuntimeError("a background task is already running")

    def _run_inputs(
        self, document: WorkspaceDocument, name: str, step_name: str,
        *, existing: bool,
    ) -> tuple[str, str]:
        if document.projection.model is None:
            raise RuntimeError("a current model is required")
        if type(name) is not str or type(step_name) is not str:
            raise TypeError("name and step_name must be strings")
        name, step_name = name.strip(), step_name.strip()
        if not name:
            raise ValueError("Job name cannot be empty.")
        if len(name) > 64:
            raise ValueError("Job name cannot exceed 64 characters.")
        if not existing and self.tasks.workspace.job_name_exists(name):
            raise ValueError(f"Job name already exists: {name}")
        if step_name not in document.session.runnable_step_names():
            raise ValueError(f"Step does not exist: {step_name}")
        return name, step_name
