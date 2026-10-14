"""Generate and replace document models through their background commands."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from time import perf_counter
from typing import Callable

from fem.application import MeshTaskSnapshot, ModelSession, SessionDelta, TaskToken
from fem.io.inp import InpImportNotice, read_with_report as read_inp_with_report
from fem_agent.authoring import ProposalState

from .commands import GuiCommandCompletion
from .document_tasks import DocumentTaskRouter, session_task_outcome
from .mesh_tasks import PreparedMeshDisplay, prepare_mesh_display
from .task_controller import BackgroundTaskState, TaskApplyOutcome, TaskCompletion
from .visualization.model_adapter import ModelGeometry, build_model_geometry
from .workers import TaskContext
from .workspace import WorkspaceDocument


@dataclass(frozen=True, slots=True)
class PreparedImportedModel:
    """Owned INP replacement and its display adapter before acceptance."""

    model_transfer: object
    geometry: ModelGeometry
    timings: dict[str, float]
    notices: tuple[InpImportNotice, ...]


@dataclass(frozen=True, slots=True)
class AgentMeshLifecycle:
    """Bound callbacks for one already authorized Agent proposal."""

    accept_model: Callable[[object], SessionDelta]
    terminate: Callable[[ProposalState, str], SessionDelta]


@dataclass(frozen=True, slots=True)
class DocumentModelPresentation:
    """Concrete model presentation stays outside command ownership."""

    project_model: Callable[
        [WorkspaceDocument, SessionDelta, PreparedMeshDisplay | PreparedImportedModel, str],
        bool,
    ]
    mesh_completed: Callable[[bool], None]
    import_completed: Callable[[WorkspaceDocument], None]
    agent_terminal: Callable[[WorkspaceDocument, ProposalState, str], None]


class DocumentModelCommands:
    """Own background model generation/replacement, excluding definition edits.

    Manual remeshing clears the old model at its public confirmation boundary.
    Agent preparation and proposal authority remain in the authoring port, and
    its old model stays installed until atomic Session acceptance.
    """

    def __init__(
        self, tasks: DocumentTaskRouter, presentation: DocumentModelPresentation,
    ) -> None:
        self.tasks = tasks
        self.presentation = presentation

    def start_mesh(
        self, document: WorkspaceDocument, *,
        completion: GuiCommandCompletion | None = None,
    ) -> bool:
        self._require_idle_document(document)
        session = document.session
        task = session.prepare_mesh_generation()

        def apply(value: object) -> TaskApplyOutcome:
            if type(value) is not PreparedMeshDisplay:
                raise TypeError("mesh generation must return PreparedMeshDisplay")
            delta = session.accept_generated_model(task.token, value.model)
            value = self._retain_accepted_display(document, session, delta, value)
            return session_task_outcome(delta, value)

        def project(value: object) -> None:
            delta, prepared = value
            self._project_model(document, delta, prepared, self._mesh_label(prepared))
            if self.tasks.is_active(document):
                self.presentation.mesh_completed(True)

        return self._start_session_task(
            document, session, task.token,
            lambda context: prepare_mesh_display(task, context),
            apply, project, "Mesh Generation", "Mesh generation failed", completion,
        )

    def start_reload(
        self, document: WorkspaceDocument, path: Path, *,
        completion: GuiCommandCompletion | None = None,
    ) -> bool:
        self._require_idle_document(document)
        session = document.session
        task = session.prepare_import(path)

        def prepare(context: TaskContext) -> PreparedImportedModel:
            timings: dict[str, float] = {}
            context.report("Parsing and building INP...")
            started = perf_counter()
            imported = read_inp_with_report(task.source_path)
            timings["INP parsing and building"] = perf_counter() - started
            context.checkpoint()
            context.report("Generating display mesh...")
            started = perf_counter()
            geometry = build_model_geometry(imported.model)
            timings["VTK display geometry construction"] = perf_counter() - started
            context.checkpoint()
            context.report("Preparing session model...")
            started = perf_counter()
            transfer = session.prepare_owned_imported_model_transfer(imported.model)
            timings["Session model ownership preparation"] = perf_counter() - started
            context.checkpoint()
            return PreparedImportedModel(transfer, geometry, timings, tuple(imported.notices))

        def apply(value: object) -> TaskApplyOutcome:
            if type(value) is not PreparedImportedModel:
                raise TypeError("INP reload must return PreparedImportedModel")
            delta = session.accept_imported_model_transfer(task.token, value.model_transfer)
            value = self._retain_accepted_display(document, session, delta, value)
            return session_task_outcome(delta, value)

        def project(value: object) -> None:
            delta, prepared = value
            self._project_model(document, delta, prepared, task.source_path.name)
            if self.tasks.is_active(document):
                self.presentation.import_completed(document)

        return self._start_session_task(
            document, session, task.token, prepare, apply, project,
            "INP Import", "Failed to load model", completion,
        )

    def start_agent_mesh(
        self, document: WorkspaceDocument, task: MeshTaskSnapshot,
        lifecycle: AgentMeshLifecycle,
    ) -> bool:
        self._require_idle_document(document)
        session = document.session
        if task.token.session_id != session.session_id:
            raise ValueError("Agent mesh task must belong to its target Session")

        def apply(value: object) -> TaskApplyOutcome:
            if type(value) is not PreparedMeshDisplay:
                raise TypeError("Agent mesh generation must return PreparedMeshDisplay")
            delta = lifecycle.accept_model(value.model)
            value = self._retain_accepted_display(document, session, delta, value)
            return session_task_outcome(delta, value)

        def project(value: object) -> None:
            delta, prepared = value
            self._project_model(document, delta, prepared, self._mesh_label(prepared))
            if self.tasks.is_active(document):
                self.presentation.mesh_completed(False)

        def terminal(record: TaskCompletion) -> None:
            state = {
                BackgroundTaskState.SUCCEEDED: ProposalState.SUCCEEDED,
                BackgroundTaskState.CANCELLED: ProposalState.CANCELLED,
                BackgroundTaskState.DISCARDED: ProposalState.STALE,
            }.get(record.state, ProposalState.FAILED)
            message = (
                "Agent mesh installed atomically" if state is ProposalState.SUCCEEDED
                else record.message or record.state.value
            )
            if state is not ProposalState.SUCCEEDED:
                lifecycle.terminate(state, message)
            if self.tasks.is_active(document) and document.session is session:
                self.presentation.agent_terminal(document, state, message)

        # The authoring port owns cleanup when its authorized starter refuses
        # or raises; routed terminal callbacks own cleanup once a task starts.
        return self.tasks.start(
            lambda context: prepare_mesh_display(task, context), project,
            "Agent mesh generation failed", lambda _message: None,
            context=document, controller=document.task_controller,
            task_name="Agent Mesh Generation", apply_result=apply, on_terminal=terminal,
            on_inactive_success=project,
        )

    def _start_session_task(
        self, document: WorkspaceDocument, session: ModelSession, token: TaskToken,
        prepare: Callable[[TaskContext], object],
        apply: Callable[[object], TaskApplyOutcome],
        project: Callable[[object], None], task_name: str, error_title: str,
        completion: GuiCommandCompletion | None,
    ) -> bool:
        def finish(state: BackgroundTaskState, message: str, *, present: bool) -> None:
            delta = (
                session.accept_task_cancelled(token)
                if state in {BackgroundTaskState.CANCELLED, BackgroundTaskState.DISCARDED}
                else session.accept_task_failed(token, message)
            )
            if not delta.accepted:
                return
            if self.tasks.is_live(document) and document.session is session:
                try:
                    self.tasks.presentation.project_delta(document, delta)
                except Exception:
                    # Token cleanup already succeeded. Preserve the initiating
                    # failure and its receipt even if UI cleanup fails too.
                    logging.exception("Cannot project document model task cleanup")
                if present and self.tasks.is_active(document) and state is BackgroundTaskState.FAILED:
                    self.tasks.presentation.show_error(error_title, message)

        def terminal(record: TaskCompletion) -> None:
            if record.state is not BackgroundTaskState.SUCCEEDED:
                finish(record.state, record.message or record.state.value, present=True)

        try:
            started = self.tasks.start(
                prepare, project, error_title, lambda _message: None,
                context=document, controller=document.task_controller,
                task_name=task_name, apply_result=apply, completion=completion,
                on_terminal=terminal,
                on_inactive_success=project,
            )
        except Exception as error:
            finish(BackgroundTaskState.FAILED, str(error).strip() or type(error).__name__, present=False)
            raise
        if not started:
            finish(BackgroundTaskState.FAILED, f"the {task_name} task could not be started", present=False)
        return started

    def _retain_accepted_display(
        self, document: WorkspaceDocument, session: ModelSession, delta: SessionDelta,
        prepared: PreparedMeshDisplay | PreparedImportedModel,
    ) -> PreparedMeshDisplay | PreparedImportedModel:
        """Keep shallow adapters before UI work, only for this accepted model."""

        if not delta.accepted or not self.tasks.is_live(document) or document.session is not session:
            return prepared
        snapshot = session.projection_snapshot()
        artifact = snapshot.artifact
        if (
            snapshot.session_revision != delta.session_revision
            or artifact is None
            or artifact.session_id != snapshot.session_id
            or artifact.model_revision != snapshot.model_revision
        ):
            return prepared
        geometry = prepared.geometry
        if geometry.artifact_id != artifact.artifact_id:
            geometry = replace(geometry, artifact_id=artifact.artifact_id)
            prepared = replace(prepared, geometry=geometry)
        cache = document.presentation_cache
        if cache.artifact_id != artifact.artifact_id:
            cache.invalidate_model()
        cache.artifact_id = artifact.artifact_id
        cache.model_geometry = geometry
        document.presentation_state.import_notices = (
            prepared.notices if type(prepared) is PreparedImportedModel else ()
        )
        return prepared

    def _project_model(
        self, document: WorkspaceDocument, delta: SessionDelta,
        prepared: PreparedMeshDisplay | PreparedImportedModel, source_label: str,
    ) -> None:
        if not self.presentation.project_model(document, delta, prepared, source_label):
            raise RuntimeError("Cannot project the accepted document model")

    @staticmethod
    def _mesh_label(prepared: PreparedMeshDisplay) -> str:
        return str(getattr(prepared.model, "name", None) or "Untitled Model")

    def _require_idle_document(self, document: WorkspaceDocument) -> None:
        if not self.tasks.is_live(document):
            raise RuntimeError("No document is available for this task")
        if document.task_controller.busy:
            raise RuntimeError("a background task is already running")
