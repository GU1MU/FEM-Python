"""Own guided scope intent and document-bound topology preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fem.application import MeshEntityRef, ModelSession, SessionDelta
from fem.core.model import AnalysisStep

from .analysis_definition_dialogs import DisplacementDialogState, LoadDialogState
from .document_tasks import DocumentTaskRouter
from .scope_selection import (
    MeshSelectionTopology, ScopeSelectionTopology,
    build_mesh_selection_topology, build_scope_selection_topology,
)
from .task_controller import BackgroundTaskState, TaskApplyOutcome, TaskCompletion
from .workers import TaskContext
from .workspace import WorkspaceDocument


AnalysisScopeEdit = tuple[tuple[str, int, int | None], str, tuple[AnalysisStep, ...]]
AnalysisScopeDialogState = DisplacementDialogState | LoadDialogState | None


@dataclass(frozen=True, slots=True)
class ScopeModelVersion:
    model_revision: int
    artifact_id: str | None

    @classmethod
    def capture(cls, session: ModelSession) -> ScopeModelVersion:
        snapshot = session.projection_snapshot()
        artifact = snapshot.artifact
        return cls(snapshot.model_revision, None if artifact is None else artifact.artifact_id)


@dataclass(slots=True)
class CommittedScopeForm:
    document: WorkspaceDocument
    session: ModelSession
    generation: int
    command_id: int
    callback: Callable[[], None]
    version: ScopeModelVersion | None = None


@dataclass(slots=True)
class GuidedScopeSession:
    document: WorkspaceDocument
    session: ModelSession
    model_revision: int
    artifact_id: str | None
    generation: int
    operation: str
    requested_kind: str | None
    semantic_kind: str
    dialog_state: AnalysisScopeDialogState = None
    resume_edit: AnalysisScopeEdit | None = None
    overlay_active: bool = False
    committing: bool = False
    commit_command_id: int | None = None
    accepted_version: ScopeModelVersion | None = None

    @classmethod
    def capture(
        cls, document: WorkspaceDocument, *, generation: int, operation: str,
        requested_kind: str | None, semantic_kind: str,
        dialog_state: AnalysisScopeDialogState = None,
        resume_edit: AnalysisScopeEdit | None = None,
    ) -> GuidedScopeSession:
        artifact = document.projection.artifact
        return cls(
            document, document.session, document.session.model_revision,
            None if artifact is None else artifact.artifact_id,
            generation, operation, requested_kind, semantic_kind,
            dialog_state, resume_edit,
        )

    @property
    def analysis_kind(self) -> str:
        return (
            self.resume_edit[1] if self.resume_edit is not None else
            self.requested_kind or
            (None if self.dialog_state is None else self.dialog_state.scope_kind) or
            {"node": "node", "edge": "edge", "face": "surface",
             "body": "body", "element": "line"}[self.semantic_kind]
        )


@dataclass(slots=True)
class MeshTopologyPreparation:
    document: WorkspaceDocument
    session: ModelSession
    artifact_id: str
    selection_filter: str | None
    on_ready: Callable[[], None] | None
    guided_state: GuidedScopeSession | None = None


@dataclass(frozen=True, slots=True)
class GuidedScopePresentation:
    begin_selection: Callable[[GuidedScopeSession, ScopeSelectionTopology | None], None]
    finish_selection: Callable[[GuidedScopeSession, bool], None]
    resume_form: Callable[[GuidedScopeSession, str], None]
    commit_scope: Callable[[GuidedScopeSession, str, tuple[MeshEntityRef, ...]], None]
    install_topology: Callable[[ScopeSelectionTopology, MeshSelectionTopology], None]
    topology_ready: Callable[[str | None], None]
    defer: Callable[[Callable[[], None]], None]
    shutting_down: Callable[[], bool]


class GuidedScopeCoordinator:
    """Keep one scope intent valid through preparation, commit and form resume."""

    def __init__(
        self, tasks: DocumentTaskRouter, presentation: GuidedScopePresentation,
    ) -> None:
        self.tasks = tasks
        self.presentation = presentation
        self.session: GuidedScopeSession | None = None
        self._resume: GuidedScopeSession | None = None
        self._committed_form: CommittedScopeForm | None = None
        self.generation = 0
        self._topology_requests: dict[int, MeshTopologyPreparation] = {}

    @property
    def active(self) -> bool:
        return self.session is not None

    @property
    def pending(self) -> bool:
        return self.session is not None or self._resume is not None or self._committed_form is not None

    @property
    def semantic_kind(self) -> str | None:
        return None if self.session is None else self.session.semantic_kind

    @property
    def overlay_active(self) -> bool:
        return self.session is not None and self.session.overlay_active

    def _owned(self, state: GuidedScopeSession) -> bool:
        return (
            state.generation == self.generation
            and (self.session is state or self._resume is state)
            and self.context_current(state.document, state.session)
        )

    def context_current(
        self, document: WorkspaceDocument, session: ModelSession,
    ) -> bool:
        return (
            not self.presentation.shutting_down()
            and self.tasks.is_active(document)
            and document.session is session
            and document.projection.is_open
        )

    def current(self, state: GuidedScopeSession) -> bool:
        artifact = state.document.projection.artifact
        return (
            self._owned(state)
            and state.session.model_revision == state.model_revision
            and (None if artifact is None else artifact.artifact_id) == state.artifact_id
        )

    def begin(
        self, document: WorkspaceDocument, operation: str,
        requested_kind: str | None, semantic_kind: str, *,
        dialog_state: AnalysisScopeDialogState = None,
        resume_edit: AnalysisScopeEdit | None = None,
        topology: ScopeSelectionTopology | None = None,
        prepare_topology: bool = False,
    ) -> GuidedScopeSession | None:
        self.cancel()
        if not self.context_current(document, document.session):
            return None
        state = GuidedScopeSession.capture(
            document, generation=self.generation, operation=operation,
            requested_kind=requested_kind, semantic_kind=semantic_kind,
            dialog_state=dialog_state, resume_edit=resume_edit,
        )
        self.session = state
        if prepare_topology:
            self.prepare_topology(
                document, document.projection.model, document.projection.geometry_recipe,
                on_ready=lambda: self._begin_prepared_selection(state),
                guided_state=state,
            )
        else:
            self._show_selection(state, topology)
        return state

    def _show_selection(
        self, state: GuidedScopeSession, topology: ScopeSelectionTopology | None,
    ) -> None:
        if self.current(state):
            state.overlay_active = topology is not None
            self.presentation.begin_selection(state, topology)

    def _begin_prepared_selection(self, state: GuidedScopeSession) -> None:
        # The topology port has installed the accepted cache before continuing.
        if self.current(state):
            self.presentation.begin_selection(state, None)

    def confirm(self, name: str, references: tuple[MeshEntityRef, ...]) -> None:
        state = self.session
        if state is None or state.committing or not self.current(state):
            return
        self.presentation.commit_scope(state, name, references)

    def commit_started(self, state: GuidedScopeSession, command_id: int) -> None:
        if self._owned(state):
            state.committing = True
            state.commit_command_id = command_id

    def commit_failed(self, state: GuidedScopeSession) -> None:
        if self._owned(state):
            state.committing = False
            state.commit_command_id = None
            state.accepted_version = None

    def scope_accepted(
        self, document: WorkspaceDocument, command_id: int, delta: SessionDelta,
    ) -> None:
        state = self.session
        if (
            state is not None and state.document is document
            and state.commit_command_id == command_id
            and state.session.session_revision == delta.session_revision
        ):
            state.accepted_version = ScopeModelVersion.capture(state.session)
        form = self._committed_form
        if (
            form is not None and form.document is document
            and form.command_id == command_id
            and form.session.session_revision == delta.session_revision
        ):
            form.version = ScopeModelVersion.capture(form.session)

    def committed(
        self, state: GuidedScopeSession, name: str, delta: SessionDelta | None = None,
    ) -> None:
        """Advance to this accepted scope version before deferring its form."""
        if not self._owned(state) or self.session is not state:
            return
        version = state.accepted_version
        if delta is not None:
            if not delta.accepted or state.session.session_revision != delta.session_revision:
                self.cancel()
                return
            version = ScopeModelVersion.capture(state.session)
        if version is None:
            if not self.current(state):
                self.cancel()
                return
        else:
            state.model_revision = version.model_revision
            state.artifact_id = version.artifact_id
            if not self.current(state):
                self.cancel()
                return
        state.committing = False
        self.session = None
        self._resume = state
        self.presentation.finish_selection(state, True)
        state.overlay_active = False
        if state.operation == "scope":
            self._resume = None
            return

        def resumed() -> None:
            if self.current(state):
                self._resume = None
                self.presentation.resume_form(state, name)
            elif self._resume is state:
                self._resume = None

        self.presentation.defer(resumed)

    def cancel(self, *, restore_context: bool = True) -> bool:
        self.generation += 1
        state = self.session
        self.session = None
        self._resume = None
        self._committed_form = None
        if state is not None:
            self.presentation.finish_selection(state, restore_context)
            state.overlay_active = False
        # A running topology still owns its task, but no longer owns a UI intent.
        for request in self._topology_requests.values():
            request.on_ready = None
            request.selection_filter = None
        return state is not None

    def dispose_document(self, document: WorkspaceDocument) -> None:
        if (
            (self.session is not None and self.session.document is document)
            or (self._resume is not None and self._resume.document is document)
            or (self._committed_form is not None and self._committed_form.document is document)
        ):
            self.cancel(restore_context=self.tasks.is_active(document))
        request = self._topology_requests.pop(document.document_id, None)
        if request is not None:
            request.on_ready = None
            request.selection_filter = None
            request.guided_state = None

    def watch_committed_form(
        self, document: WorkspaceDocument, session: ModelSession,
        generation: int, command_id: int, callback: Callable[[], None],
    ) -> None:
        if generation == self.generation and self.context_current(document, session):
            self._committed_form = CommittedScopeForm(
                document, session, generation, command_id, callback,
            )

    def finish_committed_form(self, command_id: int, *, accepted: bool) -> None:
        form = self._committed_form
        if form is None or form.command_id != command_id:
            return
        if not accepted or form.version is None:
            self._committed_form = None
            return

        def resumed() -> None:
            current_artifact = form.document.projection.artifact
            if (
                self._committed_form is form
                and form.generation == self.generation
                and self.context_current(form.document, form.session)
                and form.session.model_revision == form.version.model_revision
                and (None if current_artifact is None else current_artifact.artifact_id) == form.version.artifact_id
            ):
                self._committed_form = None
                form.callback()
            elif self._committed_form is form:
                self._committed_form = None

        self.presentation.defer(resumed)

    def prepare_topology(
        self, document: WorkspaceDocument, model: object, recipe: object,
        selection_filter: str | None = None, *,
        on_ready: Callable[[], None] | None = None,
        guided_state: GuidedScopeSession | None = None,
    ) -> None:
        artifact = document.projection.artifact
        if model is None or artifact is None or not self.context_current(document, document.session):
            return
        existing = self._topology_requests.get(document.document_id)
        if (
            existing is not None and existing.session is document.session
            and existing.artifact_id == artifact.artifact_id
            and document.task_controller.busy
        ):
            if existing.guided_state is self.session and self.session is not None and guided_state is None:
                self.cancel()
            existing.selection_filter = selection_filter
            existing.on_ready = on_ready
            existing.guided_state = guided_state
            return
        request = MeshTopologyPreparation(
            document, document.session, artifact.artifact_id, selection_filter, on_ready, guided_state,
        )
        self._topology_requests[document.document_id] = request

        def workload(context: TaskContext) -> object:
            context.report("Analyzing mesh scopes...")
            scope = build_scope_selection_topology(model, recipe)
            context.checkpoint()
            context.report("Preparing edge, face, and body selection...")
            mesh = build_mesh_selection_topology(model, scope_topology=scope)
            context.checkpoint()
            return scope, mesh

        def apply(value: object) -> TaskApplyOutcome:
            current_artifact = document.session.projection_snapshot().artifact
            if current_artifact is None or current_artifact.artifact_id != request.artifact_id:
                return TaskApplyOutcome.stale("Model changed; selection topology discarded")
            return TaskApplyOutcome.accepted(value)

        def succeeded(value: object) -> None:
            if self._topology_requests.get(document.document_id) is not request:
                return
            scope, mesh = value
            self.presentation.install_topology(scope, mesh)
            callback = request.on_ready
            requested = request.selection_filter
            request.on_ready = None
            request.selection_filter = None
            if callback is not None:
                callback()
            else:
                self.presentation.topology_ready(requested)

        def terminal(record: TaskCompletion | None) -> None:
            if self._topology_requests.get(document.document_id) is request:
                self._topology_requests.pop(document.document_id)
            if (
                request.guided_state is self.session and self.session is not None
                and (record is None or record.state is not BackgroundTaskState.SUCCEEDED)
            ):
                self.cancel()
            request.on_ready = None
            request.selection_filter = None

        try:
            started = self.tasks.start(
                workload, succeeded, "Failed to prepare mesh selection topology",
                context=document, controller=document.task_controller,
                task_name="Prepare Mesh Selection Topology", apply_result=apply,
                on_terminal=terminal,
            )
        except Exception:
            terminal(None)
            raise
        if not started:
            terminal(None)
