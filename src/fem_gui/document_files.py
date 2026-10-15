"""Execute document saves against captured Sessions and reserved paths."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter, sleep
from typing import Literal

from PySide6.QtWidgets import QApplication

from fem.application import ProjectSaveSnapshot, ResultArchiveSaveSnapshot, SessionDelta
from fem.io.project import MODEL_FILE_SUFFIX, save_project
from fem.io.result_archive import save_result_archive

from .commands import (
    GuiCommandCompletion,
    GuiCommandDiagnostic,
    GuiCommandReceipt,
    ResultArchiveSaveSpec,
)
from .document_tasks import DocumentTaskRouter, session_task_outcome
from .task_controller import BackgroundTaskState, TaskApplyOutcome, TaskCompletion
from .workers import TaskContext
from .workspace import WorkspaceDocument


class DocumentFileCommands:
    """Own save preparation, worker execution, token cleanup and path claims."""

    def __init__(self, tasks: DocumentTaskRouter) -> None:
        self.tasks = tasks

    def save_project(
        self, command_id: int, document: WorkspaceDocument, path: str | Path,
    ) -> GuiCommandReceipt:
        if not document.projection.can_save:
            return self._rejected(
                command_id, "project.save.unavailable",
                "current session cannot be saved as a native project",
            )
        target = Path(path)
        if target.suffix.casefold() != MODEL_FILE_SUFFIX:
            target = target.with_suffix(MODEL_FILE_SUFFIX)
        return self._save(command_id, document, target, "project")

    def save_result(
        self, command_id: int, document: WorkspaceDocument, path: str | Path,
    ) -> GuiCommandReceipt:
        return self._save(
            command_id, document, ResultArchiveSaveSpec(Path(path)).path, "result",
        )

    def _save(
        self,
        command_id: int,
        document: WorkspaceDocument,
        path: Path,
        kind: Literal["project", "result"],
    ) -> GuiCommandReceipt:
        controller = document.task_controller
        if controller.busy:
            return self._rejected(command_id, "task.busy", "a background task is already running")
        session = document.session
        reservation = None
        snapshot: ProjectSaveSnapshot | ResultArchiveSaveSnapshot | None = None
        completion = GuiCommandCompletion(command_id)
        error_title = (
            "Failed to save native project" if kind == "project"
            else "Failed to save analysis results"
        )

        def project_receipt(delta: SessionDelta) -> None:
            if (
                delta.accepted and (delta.changed or delta.invalidated)
                and self.tasks.is_live(document) and document.session is session
            ):
                self.tasks.presentation.project_delta(document, delta)

        def cancelled() -> None:
            project_receipt(session.accept_task_cancelled(snapshot.token))

        def failed(message: str) -> None:
            delta = session.accept_task_failed(snapshot.token, message)
            project_receipt(delta)
            if delta.accepted and document.session is session and self.tasks.is_active(document):
                self.tasks.presentation.show_error(error_title, message)

        def terminal(record: TaskCompletion) -> None:
            if record.state is BackgroundTaskState.SUCCEEDED:
                return
            delta = (
                session.accept_task_cancelled(snapshot.token)
                if record.state in {BackgroundTaskState.CANCELLED, BackgroundTaskState.DISCARDED}
                else session.accept_task_failed(snapshot.token, record.message or record.state.value)
            )
            try:
                project_receipt(delta)
            except Exception:
                logging.exception("Cannot project save task cleanup")

        try:
            reservation = self.tasks.workspace.reserve_path(
                path, controller=controller, document_id=document.document_id,
            )
            if kind == "project":
                snapshot = session.prepare_project_save()
            else:
                run_id = document.projection.displayed_result_run_id
                if run_id is None:
                    raise RuntimeError("No successful results are available to save")
                snapshot = session.prepare_result_archive_save(run_id)

            def workload(context: TaskContext) -> Path:
                if kind == "project":
                    context.report("Validating and saving native project...")
                    return save_project(path, snapshot, checkpoint=context.checkpoint)
                context.report("Encoding and saving analysis results...")
                return save_result_archive(path, snapshot.archive, checkpoint=context.checkpoint)

            def apply_result(payload: object) -> TaskApplyOutcome:
                if not isinstance(payload, Path):
                    raise TypeError("save worker must return pathlib.Path")
                delta = (
                    session.accept_project_saved(snapshot.token, payload)
                    if kind == "project"
                    else session.accept_result_archive_saved(snapshot.token, payload)
                )
                return session_task_outcome(delta, payload)

            def succeeded(value: object) -> None:
                delta, saved_path = value
                project_receipt(delta)
                if kind == "result":
                    self.tasks.presentation.set_status(
                        f"Analysis results saved: {saved_path.name}", 5000,
                    )

            started = self.tasks.start(
                workload, succeeded, error_title, failed,
                context=document, controller=controller,
                task_name="Save Native Project" if kind == "project" else "Save Analysis Results",
                on_cancelled=cancelled,
                on_inactive_failure=failed,
                on_inactive_cancelled=cancelled,
                apply_result=apply_result,
                completion=completion,
                on_terminal=terminal,
            )
        except Exception as error:
            if snapshot is not None and not controller.busy:
                try:
                    cancelled()
                except Exception:
                    logging.exception("Cannot project save startup cleanup")
            return self._rejected(command_id, f"{kind}.save.rejected", error)
        finally:
            if reservation is not None and not controller.busy:
                self.tasks.workspace.release_path(reservation)
        if not started:
            cancelled()
            return self._rejected(
                command_id, "task.start.rejected", f"the {kind} save task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def wait_for_save(
        self,
        document: WorkspaceDocument,
        completion: GuiCommandCompletion,
        timeout: float,
    ) -> TaskCompletion | None:
        """Keep the legacy synchronous adapter bound across nested Qt events."""

        controller = document.task_controller
        try:
            terminal = completion.result(timeout)
        except TimeoutError:
            if self.tasks.is_live(document) and controller.current_task_id == completion.task_id:
                controller.request_cancel()
            return None
        deadline = perf_counter() + 5.0
        while self.tasks.is_live(document) and controller.busy and perf_counter() < deadline:
            QApplication.processEvents()
            sleep(0.001)
        return terminal

    @staticmethod
    def _rejected(command_id: int, code: str, error: object) -> GuiCommandReceipt:
        return GuiCommandReceipt.rejected(
            command_id,
            GuiCommandDiagnostic(code, str(error).strip() or type(error).__name__),
        )
