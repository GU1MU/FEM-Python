"""Route background work to its owning document and visible presentation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable
from weakref import WeakSet

from fem.application import SessionDelta, TokenStatus

from .commands import GuiCommandCompletion
from .task_controller import (
    BackgroundTaskController,
    BackgroundTaskState,
    TaskApplyOutcome,
    TaskCompletion,
)
from .workers import TaskContext
from .workspace import FEMWorkspace, WorkspaceDocument


@dataclass(frozen=True, slots=True)
class DocumentTaskPresentation:
    """The shared view operations required by document task routing."""

    set_status: Callable[[str, int], None]
    show_error: Callable[[str, str], None]
    project_delta: Callable[[WorkspaceDocument, SessionDelta], bool]
    rebuild: Callable[[], None]
    projection_failed: Callable[[str], None]
    busy_changed: Callable[[bool], None]
    cancelling_changed: Callable[[bool], None]


class DocumentTaskRouter:
    """Keep task ownership fixed while the shared view changes documents."""

    def __init__(
        self,
        workspace: FEMWorkspace,
        presentation: DocumentTaskPresentation,
    ) -> None:
        self.workspace = workspace
        self.presentation = presentation
        self._bound_controllers: WeakSet[BackgroundTaskController] = WeakSet()

    def is_live(self, document: WorkspaceDocument) -> bool:
        try:
            return self.workspace.document(document.document_id) is document
        except (KeyError, TypeError, ValueError):
            return False

    def is_active(self, document: WorkspaceDocument) -> bool:
        return self.is_live(document) and self.workspace.active_document() is document

    def bind(self, document: WorkspaceDocument) -> None:
        controller = document.task_controller
        if controller in self._bound_controllers:
            return
        self._bound_controllers.add(controller)

        def busy_changed(busy: bool) -> None:
            if self.is_active(document):
                self.presentation.busy_changed(bool(busy))

        def cancelling_changed(cancelling: bool) -> None:
            if self.is_active(document):
                self.presentation.cancelling_changed(bool(cancelling))

        controller.busy_changed.connect(busy_changed)
        controller.cancelling_changed.connect(cancelling_changed)

    @staticmethod
    def after_idle(
        controller: BackgroundTaskController, callback: Callable[[], None],
    ) -> None:
        """Observe actual cleanup, including a task already at terminal state."""

        if not controller.busy:
            callback()
            return

        def busy_changed(busy: bool) -> None:
            if not busy:
                controller.busy_changed.disconnect(busy_changed)
                callback()

        controller.busy_changed.connect(busy_changed)

    def start(
        self,
        workload: Callable[[TaskContext], object],
        on_success: Callable[[object], None],
        error_title: str,
        on_failure: Callable[[str], None] | None = None,
        *,
        context: WorkspaceDocument | None,
        controller: BackgroundTaskController,
        task_name: str = "Background Task",
        on_cancelled: Callable[[], None] | None = None,
        on_inactive_failure: Callable[[str], None] | None = None,
        on_inactive_cancelled: Callable[[], None] | None = None,
        on_inactive_success: Callable[[object], None] | None = None,
        apply_result: Callable[[object], TaskApplyOutcome] | None = None,
        completion: GuiCommandCompletion | None = None,
        on_progress: Callable[[str], None] | None = None,
        on_terminal: Callable[[TaskCompletion], None] | None = None,
        rebuild_projection: Callable[[], None] | None = None,
    ) -> bool:
        if context is not None:
            if controller is not context.task_controller:
                raise ValueError("document task must use its owning controller")
            if not self.is_live(context):
                return False
            self.bind(context)

        def visible() -> bool:
            return context is None or self.is_active(context)

        if controller.busy:
            if visible():
                self.presentation.set_status(
                    f"Task running: {controller.current_task_name or 'Background Task'}",
                    4000,
                )
            return False
        result_applier = apply_result or TaskApplyOutcome.accepted
        target_session = None if context is None else context.session

        def live() -> bool:
            return context is None or (
                self.is_live(context) and context.session is target_session
            )

        def apply(value: object) -> TaskApplyOutcome:
            if not live():
                return TaskApplyOutcome.stale("Target document closed or replaced")
            outcome = result_applier(value)
            if type(outcome) is not TaskApplyOutcome:
                raise TypeError("apply_result must return a TaskApplyOutcome")
            return outcome

        def project(value: object) -> None:
            if not live():
                return
            if visible():
                on_success(value)
            elif on_inactive_success is not None:
                on_inactive_success(value)
            else:
                delta = session_delta_from_task_value(value)
                if delta is not None:
                    self.presentation.project_delta(context, delta)

        def terminal(record: TaskCompletion) -> None:
            try:
                if live():
                    active = visible()
                    if record.state is BackgroundTaskState.FAILED:
                        message = record.message or "Background task failed"
                        callback = on_failure if active else on_inactive_failure
                        if callback is not None:
                            callback(message)
                        elif active:
                            self.presentation.show_error(error_title, message)
                    elif record.state is BackgroundTaskState.CANCELLED:
                        callback = on_cancelled if active else on_inactive_cancelled
                        if callback is not None:
                            callback()
                        if active:
                            self.presentation.set_status(f"Canceled: {record.task_name}", 4000)
                    elif record.state is BackgroundTaskState.DISCARDED and active:
                        self.presentation.set_status(
                            record.message or "Task result is stale and was not applied",
                            5000,
                        )
            except Exception as error:
                logging.exception("GUI background task terminal callback failed")
                if visible():
                    self.presentation.show_error(
                        error_title, str(error).strip() or type(error).__name__,
                    )
            finally:
                try:
                    if on_terminal is not None:
                        on_terminal(record)
                finally:
                    if completion is not None:
                        completion.complete(record)

        def progress(message: str) -> None:
            if live() and visible():
                self.presentation.set_status(message, 0)
                if on_progress is not None:
                    on_progress(message)

        def rebuild() -> None:
            if live() and visible():
                if rebuild_projection is None:
                    self.presentation.rebuild()
                else:
                    rebuild_projection()

        def projection_failed(message: str) -> None:
            if live() and visible():
                self.presentation.projection_failed(message)

        task_id = controller.start(
            workload,
            task_name=task_name,
            apply_result=apply,
            project_result=project,
            rebuild_projection=rebuild,
            on_terminal=terminal,
            on_progress=progress,
            on_projection_error=projection_failed,
        )
        if task_id is not None and completion is not None:
            completion.bind_task_id(task_id)
        return task_id is not None


def session_delta_from_task_value(value: object) -> SessionDelta | None:
    if type(value) is SessionDelta:
        return value
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        return None
    for item in values:
        delta = session_delta_from_task_value(item)
        if delta is not None:
            return delta
    return None


def session_task_outcome(delta: SessionDelta, value: object) -> TaskApplyOutcome:
    if type(delta) is not SessionDelta:
        raise TypeError("Session task acceptance must return SessionDelta")
    if delta.accepted:
        return TaskApplyOutcome.accepted((delta, value))
    status = delta.token_status
    message = delta.reason or (
        status.value if status is not None else "task result was rejected"
    )
    if status in {TokenStatus.WRONG_KIND, TokenStatus.INVALID_STATE}:
        return TaskApplyOutcome.rejected(message)
    return TaskApplyOutcome.stale(message)
