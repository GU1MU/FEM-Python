"""Accept detached file reads before presenting their workspace documents."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal

from fem.application import ImportTaskSnapshot, ModelSession, SessionDelta
from fem.io.inp import InpImportNotice, read_with_report as read_inp_with_report
from fem.io.project import LoadedProject, ProjectMigrationNotice, load_project
from fem.io.result_archive import LoadedResultArchive, load_result_archive

from .commands import (
    GuiCommandCompletion,
    GuiCommandDiagnostic,
    GuiCommandOutcome,
    GuiCommandReceipt,
    ResultArchiveOpenSpec,
)
from .document_tasks import DocumentTaskRouter
from .task_controller import TaskApplyOutcome
from .visualization.model_adapter import (
    ArchiveModelView,
    ModelGeometry,
    build_model_geometry,
    build_result_archive_geometry,
    build_result_archive_model_view,
)
from .workers import TaskContext
from .workspace import (
    DocumentPresentationCache,
    DocumentPresentationState,
    WorkspaceDocument,
    WorkspacePathReservation,
    canonical_path,
)


OpenKind = Literal["project", "inp", "result"]
OpenNotice = InpImportNotice | ProjectMigrationNotice


@dataclass(frozen=True, slots=True)
class PreparedProjectOpen:
    loaded: LoadedProject
    geometry: ModelGeometry | None


@dataclass(frozen=True, slots=True)
class PreparedInpOpen:
    model_transfer: object
    geometry: ModelGeometry
    timings: dict[str, float]
    notices: tuple[InpImportNotice, ...]


@dataclass(frozen=True, slots=True)
class PreparedResultOpen:
    loaded: LoadedResultArchive
    geometry: ModelGeometry
    model_view: ArchiveModelView
    timings: dict[str, float]


@dataclass(frozen=True, slots=True)
class AcceptedDocumentOpen:
    document: WorkspaceDocument
    delta: SessionDelta
    timings: dict[str, float]
    notices: tuple[OpenNotice, ...]


@dataclass(frozen=True, slots=True)
class DocumentOpenPresentation:
    """UI operations used after a complete document has been registered."""

    refresh_document: Callable[[WorkspaceDocument], None]
    activate_document: Callable[[WorkspaceDocument], bool]
    result_opened: Callable[[Path], None]
    shutting_down: Callable[[], bool]


class DocumentOpenCommands:
    """Own path claims, preparation, acceptance and file-open task cleanup."""

    def __init__(
        self, tasks: DocumentTaskRouter, presentation: DocumentOpenPresentation,
    ) -> None:
        self.tasks = tasks
        self.presentation = presentation

    def open_project(self, command_id: int, path: str | Path) -> GuiCommandReceipt:
        return self._open(command_id, path, "project")

    def open_inp(self, command_id: int, path: str | Path) -> GuiCommandReceipt:
        return self._open(command_id, path, "inp")

    def open_result(self, command_id: int, path: str | Path) -> GuiCommandReceipt:
        return self._open(command_id, path, "result")

    def _open(
        self, command_id: int, path: str | Path, kind: OpenKind,
    ) -> GuiCommandReceipt:
        rejection_code = {
            "project": "project.open.rejected",
            "inp": "import.open.rejected",
            "result": "result.open.rejected",
        }[kind]
        reservation = None
        controller = None
        session = None
        import_task = None
        started = False
        try:
            if self.presentation.shutting_down():
                raise RuntimeError("the workspace is closing")
            target = ResultArchiveOpenSpec(Path(path)).path if kind == "result" else Path(path)
            existing_receipt = self._activate_existing(command_id, target, kind)
            if existing_receipt is not None:
                return existing_receipt
            controller = self.tasks.workspace.ensure_open_controller()
            if controller.busy:
                message = {
                    "project": "a project open task is already running",
                    "inp": "a model open task is already running",
                    "result": "a result archive is already being opened",
                }[kind]
                return self._rejected(command_id, "task.busy", message)
            reservation = self.tasks.workspace.reserve_path(target, controller=controller)
            session = ModelSession()
            import_task = session.prepare_import(target) if kind == "inp" else None
            completion = GuiCommandCompletion(command_id)
            accepted: AcceptedDocumentOpen | None = None

            def workload(context: TaskContext) -> PreparedProjectOpen | PreparedInpOpen | PreparedResultOpen:
                return self._prepare(target, kind, context)

            def apply_result(payload: object) -> TaskApplyOutcome:
                nonlocal accepted
                if self.presentation.shutting_down():
                    if import_task is not None:
                        session.accept_task_cancelled(import_task.token)
                    return TaskApplyOutcome.stale("The workspace closed before the file was accepted")
                try:
                    accepted = self._accept(target, kind, payload, session, import_task, reservation)
                except Exception as error:
                    return TaskApplyOutcome.rejected(str(error).strip() or type(error).__name__)
                return TaskApplyOutcome.accepted(accepted)

            def present(value: object) -> None:
                if type(value) is not AcceptedDocumentOpen:
                    raise TypeError("file open acceptance must return AcceptedDocumentOpen")
                self.presentation.refresh_document(value.document)
                started = perf_counter()
                if not self.presentation.activate_document(value.document):
                    raise RuntimeError(f"opened {kind} document could not be activated")
                value.timings["Workspace document activation"] = perf_counter() - started
                if value.timings:
                    logging.info(
                        "File open performance %s: %s (total %.3fs)", target.name,
                        ", ".join(f"{name}={seconds:.3f}s" for name, seconds in value.timings.items()),
                        sum(value.timings.values()),
                    )
                if kind == "result":
                    self.presentation.result_opened(target)

            def rebuild_registration() -> None:
                if accepted is not None and self.tasks.is_live(accepted.document):
                    self.presentation.refresh_document(accepted.document)

            def failed(message: str) -> None:
                if import_task is not None:
                    session.accept_task_failed(import_task.token, message)
                self.tasks.presentation.show_error(error_title, message)

            def cancelled() -> None:
                if import_task is not None:
                    session.accept_task_cancelled(import_task.token)

            if kind == "inp":
                self.tasks.presentation.set_status("Importing new model...", 0)
            error_title = {
                "project": "Failed to open native project",
                "inp": "Failed to load model",
                "result": "Failed to open analysis results",
            }[kind]
            started = self.tasks.start(
                workload, present, error_title, failed,
                context=None, controller=controller,
                task_name={
                    "project": "Open Native Project",
                    "inp": "INP Import",
                    "result": "Open Analysis Results",
                }[kind],
                apply_result=apply_result, completion=completion,
                rebuild_projection=rebuild_registration,
                on_cancelled=cancelled,
            )
        except Exception as error:
            remediation = "Check the project file version and contents." if kind == "project" else ""
            return self._rejected(command_id, rejection_code, error, remediation)
        finally:
            if reservation is not None and not controller.busy:
                try:
                    if not started and import_task is not None:
                        session.accept_task_cancelled(import_task.token)
                finally:
                    self.tasks.workspace.release_path(reservation)
        if not started:
            message = {
                "project": "the project open task could not be started",
                "inp": "the import task could not be started",
                "result": "the result open task could not be started",
            }[kind]
            return self._rejected(command_id, "task.start.rejected", message)
        return GuiCommandReceipt.pending(command_id, completion)

    def _activate_existing(
        self, command_id: int, target: Path, kind: OpenKind,
    ) -> GuiCommandReceipt | None:
        workspace = self.tasks.workspace
        paths = workspace.result_paths if kind == "result" else workspace.model_paths
        existing_id = paths.get(canonical_path(target))
        if existing_id is None:
            return None
        document = workspace.document(existing_id)
        self.presentation.refresh_document(document)
        if not self.presentation.activate_document(document):
            code = "result.activate.rejected" if kind == "result" else "workspace.activate.rejected"
            message = {
                "project": "the already-open project could not be activated",
                "inp": "the already-open INP model could not be activated",
                "result": "the existing result document could not be activated",
            }[kind]
            return self._rejected(command_id, code, message)
        summary = {
            "project": "Activated the already open model",
            "inp": "Activated the already open INP model",
            "result": "",
        }[kind]
        if kind == "result":
            self.presentation.result_opened(target)
        return GuiCommandReceipt.accepted(
            command_id, outcome=GuiCommandOutcome(output_path=target, diagnostic_summary=summary),
        )

    @staticmethod
    def _prepare(
        target: Path, kind: OpenKind, context: TaskContext,
    ) -> PreparedProjectOpen | PreparedInpOpen | PreparedResultOpen:
        context.checkpoint()
        if kind == "project":
            context.report("Reading and validating native project...")
            loaded = load_project(target)
            context.checkpoint()
            model = loaded.snapshot.model
            geometry = None if model is None else build_model_geometry(model)
            context.checkpoint()
            return PreparedProjectOpen(loaded, geometry)
        timings: dict[str, float] = {}
        if kind == "inp":
            context.report("Parsing and building INP...")
            started = perf_counter()
            imported = read_inp_with_report(target)
            timings["INP parsing and building"] = perf_counter() - started
            context.checkpoint()
            context.report("Generating display mesh...")
            started = perf_counter()
            geometry = build_model_geometry(imported.model)
            timings["VTK display geometry construction"] = perf_counter() - started
            context.checkpoint()
            context.report("Preparing an independent model document...")
            started = perf_counter()
            transfer = ModelSession.prepare_owned_imported_model_transfer(imported.model)
            timings["Session model ownership preparation"] = perf_counter() - started
            context.checkpoint()
            return PreparedInpOpen(transfer, geometry, timings, tuple(imported.notices))
        context.report("reading and validating result archive")
        loaded = load_result_archive(target)
        context.checkpoint()
        started = perf_counter()
        model_view = build_result_archive_model_view(
            loaded.snapshot.model_projection, loaded.snapshot.profile,
            name=str(loaded.snapshot.origin.model_name or "Result"),
        )
        geometry = build_result_archive_geometry(loaded.snapshot.model_projection)
        timings["result display adapters"] = perf_counter() - started
        context.checkpoint()
        return PreparedResultOpen(loaded, geometry, model_view, timings)

    def _accept(
        self, target: Path, kind: OpenKind, payload: object, session: ModelSession,
        import_task: ImportTaskSnapshot | None, reservation: WorkspacePathReservation,
    ) -> AcceptedDocumentOpen:
        model_view = None
        notices = ()
        timings = {}
        if kind == "project":
            if type(payload) is not PreparedProjectOpen:
                raise TypeError("project loader must return PreparedProjectOpen")
            delta = session.replace_from_snapshot(payload.loaded.snapshot)
            source_path = payload.loaded.path or target
            display_name = str(payload.loaded.snapshot.model_name or "").strip() or target.stem
            geometry = payload.geometry
            notices = payload.loaded.notices
        elif kind == "inp":
            if type(payload) is not PreparedInpOpen or import_task is None:
                raise TypeError("INP loader must return PreparedInpOpen with its import task")
            delta = session.accept_imported_model_transfer(import_task.token, payload.model_transfer)
            source_path = target
            display_name = target.stem
            geometry, timings, notices = payload.geometry, payload.timings, payload.notices
        else:
            if type(payload) is not PreparedResultOpen:
                raise TypeError("result loader must return PreparedResultOpen")
            delta = session.replace_from_result_archive(payload.loaded, path=target)
            source_path = target
            display_name = target.stem
            geometry, model_view, timings = payload.geometry, payload.model_view, payload.timings
        if not delta.accepted:
            raise RuntimeError(delta.reason or "file contents were rejected")
        projection = session.projection_snapshot()
        artifact = projection.artifact
        if kind == "inp" and artifact is None:
            raise RuntimeError("The accepted INP model is missing its model artifact")
        if geometry is not None and artifact is not None and geometry.artifact_id != artifact.artifact_id:
            geometry = replace(geometry, artifact_id=artifact.artifact_id)
        cache = DocumentPresentationCache(
            artifact_id=None if artifact is None else artifact.artifact_id,
            model_geometry=geometry, result_model_view=model_view,
        )
        identity = session.current_result_identity()
        if identity is not None:
            cache.result_source, cache.result_generation = identity
        state = DocumentPresentationState(
            module_name="Results" if kind == "result" else None,
            import_notices=tuple(notices),
        )
        register = self.tasks.workspace.add_result if kind == "result" else self.tasks.workspace.add_model
        document = register(
            session=session, projection=projection, display_name=display_name,
            source_path=source_path, path_reservation=reservation, activate=False,
            presentation_cache=cache, presentation_state=state,
        )
        return AcceptedDocumentOpen(document, delta, timings, tuple(notices))

    @staticmethod
    def _rejected(
        command_id: int, code: str, error: object, remediation: str = "",
    ) -> GuiCommandReceipt:
        return GuiCommandReceipt.rejected(
            command_id, GuiCommandDiagnostic(
                code, str(error).strip() or type(error).__name__, remediation,
            ),
        )
