"""Execute typed result commands against their owning documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fem.application import SessionDelta
from fem.application.results import (
    FieldMaterializationKey,
    FieldState,
    ResultExportSnapshot,
    ResultMaterializationPatch,
    ResultProvider,
    ResultQuery,
    ResultQueryResult,
    ResultQueryValidationError,
    ResultSourceKey,
    ScalarFieldSelection,
    prepare_result_export_snapshot,
)
from fem.io.result_csv import write_result_table_csv
from fem.io.result_vtk import write_result_vtk
from fem_agent.export_storage import MAX_EXPORT_CSV_BYTES, verify_export_file_size

from .commands import (
    GuiCommandCompletion,
    GuiCommandDiagnostic,
    GuiCommandOutcome,
    GuiCommandReceipt,
    ResultCsvExportSpec,
    ResultVtkExportSpec,
)
from .document_tasks import DocumentTaskRouter, session_task_outcome
from .result_command_state import PendingResultMaterialization
from .result_presentation import result_field_is_beam_section, visible_result_fields
from .task_controller import TaskApplyOutcome, TaskCompletion
from .workers import TaskContext
from .workspace import WorkspaceDocument


@dataclass(frozen=True, slots=True)
class DocumentResultPresentation:
    """The visible result operations needed by document commands."""

    current_provider: Callable[[WorkspaceDocument], ResultProvider | None]
    current_selection: Callable[[WorkspaceDocument], ScalarFieldSelection | None]
    install_selection: Callable[[WorkspaceDocument, ResultProvider, ScalarFieldSelection], None]
    query_completed: Callable[[WorkspaceDocument, ResultQueryResult], None]
    state_changed: Callable[[WorkspaceDocument], None]
    field_finished: Callable[[WorkspaceDocument, PendingResultMaterialization, TaskCompletion], None]


class DocumentResultCommands:
    """Own validation, materialization, exports and pending request cleanup."""

    def __init__(
        self, tasks: DocumentTaskRouter, presentation: DocumentResultPresentation,
    ) -> None:
        self.tasks = tasks
        self.presentation = presentation

    def select_field(
        self, command_id: int, document: WorkspaceDocument,
        provider: ResultProvider | None, selection: ScalarFieldSelection,
    ) -> GuiCommandReceipt:
        if type(selection) is not ScalarFieldSelection:
            return self._rejected(command_id, "command.type.invalid", "selection must be a ScalarFieldSelection")
        if document.task_controller.busy:
            return self._rejected(command_id, "task.busy", "a background task is already running")
        if not self._provider_is_current(document, provider):
            return self._rejected(command_id, "result.current.unavailable", "there is no current accepted result provider")
        try:
            availability = provider.field_status(selection.field_key)
            if selection.component not in availability.descriptor.columns:
                raise ValueError("selection component is outside the field descriptor")
        except (KeyError, TypeError, ValueError) as error:
            return self._rejected(command_id, "result.field.selection.invalid", error)
        if availability.state is FieldState.LAZY:
            return self._materialize(
                command_id, document, provider, selection,
                self.materialization_keys(provider, selection),
            )
        if availability.state is FieldState.UNAVAILABLE:
            diagnostic = next((item for item in availability.diagnostics if item.message.strip()), None)
            return self._rejected(
                command_id, "result.field.unavailable",
                "the selected field is unavailable" if diagnostic is None else diagnostic.message,
                "" if diagnostic is None else diagnostic.remediation,
            )
        outcome = self._selection_outcome(provider, selection)
        if self.presentation.current_selection(document) != selection:
            try:
                self.presentation.install_selection(document, provider, selection)
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                return self._rejected(command_id, "result.field.projection.failed", error)
        return GuiCommandReceipt.accepted(command_id, outcome=outcome)

    def query(
        self, command_id: int, document: WorkspaceDocument,
        provider: ResultProvider | None, query: ResultQuery,
        *, expected_source: ResultSourceKey | None = None,
    ) -> GuiCommandReceipt:
        if type(query) is not ResultQuery:
            return self._rejected(command_id, "command.type.invalid", "query must be a ResultQuery")
        if expected_source is not None and type(expected_source) is not ResultSourceKey:
            return self._rejected(command_id, "command.type.invalid", "expected_source must be a ResultSourceKey or None")
        if not self._provider_is_current(document, provider):
            return self._rejected(command_id, "result.current.unavailable", "there is no current accepted result provider")
        if expected_source is not None and provider.source != expected_source:
            return self._rejected(command_id, "result.query.source.stale", "the query dialog source is no longer current")
        if document.task_controller.busy:
            return self._rejected(command_id, "task.busy", "a background task is already running")
        try:
            availability = provider.validate_query(query)
        except ResultQueryValidationError as error:
            return self._rejected(command_id, error.code, error)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected(command_id, "result.query.validation.rejected", error)
        if availability.state is FieldState.LAZY:
            return self._materialize(command_id, document, provider, query, (query.field_key,))
        if availability.state is not FieldState.READY:
            return self._rejected(command_id, "result.query.field.unavailable", "the selected query field is unavailable")
        try:
            result = provider.query(query)
            self._validate_query_result(provider, query, result)
        except ResultQueryValidationError as error:
            return self._rejected(command_id, error.code, error)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected(command_id, "result.query.rejected", error)
        self.presentation.query_completed(document, result)
        return GuiCommandReceipt.accepted(command_id, outcome=self._query_outcome(result))

    def _materialize(
        self, command_id: int, document: WorkspaceDocument,
        provider: ResultProvider, request: ScalarFieldSelection | ResultQuery,
        field_keys: tuple[FieldMaterializationKey, ...],
    ) -> GuiCommandReceipt:
        """Share the token and task lifetime for field display and queries."""

        session = document.session
        is_query = type(request) is ResultQuery
        title = "Result query failed" if is_query else "Failed to load result fields on demand"
        task_name = "Result Query" if is_query else "Load Result Fields on Demand"
        task = None
        pending = None
        accepted_delta: SessionDelta | None = None
        accepted_query: ResultQueryResult | None = None

        def clear_pending() -> None:
            if document.result_command_state.materialization is pending:
                document.result_command_state.materialization = None
                self.presentation.state_changed(document)

        def finished(terminal: TaskCompletion) -> None:
            if document.result_command_state.materialization is not pending:
                return
            document.result_command_state.materialization = None
            try:
                if not is_query and self.tasks.is_active(document):
                    self.presentation.field_finished(document, pending, terminal)
            finally:
                self.presentation.state_changed(document)

        def failed(message: str) -> None:
            if accepted_delta is None:
                self._failed(document, task.token, title, message)
            elif self.tasks.is_active(document):
                # Materialization consumed the token before the query failed.
                self.tasks.presentation.show_error(title, message)

        try:
            task = session.prepare_result_materialization(provider.source.run_id, field_keys)
            completion = GuiCommandCompletion(command_id)
            pending = PendingResultMaterialization(
                task.materialization.source, task.materialization.generation, request,
            )
            document.result_command_state.materialization = pending
            self.presentation.state_changed(document)

            def workload(context: TaskContext) -> ResultMaterializationPatch:
                context.report(
                    "Loading query fields on demand..." if is_query
                    else "Loading result fields on demand..."
                )
                return provider.materialize(task.field_keys, cancellation=context)

            def apply_result(value: object) -> TaskApplyOutcome:
                nonlocal accepted_delta, accepted_query
                if type(value) is not ResultMaterializationPatch:
                    raise TypeError("result materialization worker must return ResultMaterializationPatch")
                delta = session.accept_result_materialization(task.token, value)
                acceptance = session_task_outcome(delta, value)
                if not delta.accepted:
                    return acceptance
                accepted_delta = delta
                if is_query:
                    try:
                        accepted_provider = provider.advance(value)
                        accepted_query = accepted_provider.query(request)
                        self._validate_query_result(accepted_provider, request, accepted_query)
                    except Exception:
                        # The accepted patch remains committed even if querying fails.
                        self._project_delta(document, delta)
                        raise
                    outcome = self._query_outcome(accepted_query)
                else:
                    materialized = next(
                        (field for field in value.fields if field.key == request.field_key), None,
                    )
                    outcome = GuiCommandOutcome(
                        source=task.materialization.source,
                        materialization_generation=task.materialization.generation + (1 if value.fields else 0),
                        selection=request,
                        record_count=None if materialized is None else len(materialized.locations),
                    )
                return TaskApplyOutcome.accepted(outcome)

            def succeeded(value: object) -> None:
                if type(value) is not GuiCommandOutcome or accepted_delta is None:
                    raise RuntimeError("accepted result materialization has no typed outcome")
                if not self._project_delta(document, accepted_delta):
                    raise RuntimeError("accepted result materialization could not be projected")
                current = self.presentation.current_provider(document)
                if (
                    current is not None and current.source == value.source
                    and current.snapshot.generation == value.materialization_generation
                ):
                    if is_query:
                        self.presentation.query_completed(document, accepted_query)
                    else:
                        self.presentation.install_selection(document, current, request)
                self.presentation.state_changed(document)
                self.tasks.presentation.set_status(
                    "Result query complete" if is_query else "Result fields loaded on demand", 4000,
                )

            started = self.tasks.start(
                workload, succeeded, title,
                failed,
                context=document, controller=document.task_controller,
                task_name=task_name,
                on_cancelled=lambda: self._cancelled(document, task.token),
                on_inactive_failure=failed,
                on_inactive_cancelled=lambda: self._cancelled(document, task.token),
                on_inactive_success=lambda _value: self._project_delta(document, accepted_delta),
                apply_result=apply_result, completion=completion, on_terminal=finished,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            if task is not None:
                self._project_delta(document, session.accept_task_failed(task.token, error))
            clear_pending()
            return self._rejected(
                command_id,
                "result.query.materialization.rejected" if is_query else "result.field.materialization.rejected",
                error,
            )
        if not started:
            self._cancelled(document, task.token)
            clear_pending()
            return self._rejected(
                command_id, "task.start.rejected",
                "the result query task could not be started" if is_query
                else "the result materialization task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    @staticmethod
    def materialization_keys(
        provider: ResultProvider, selection: ScalarFieldSelection,
    ) -> tuple[FieldMaterializationKey, ...]:
        """Batch Beam section display fields behind one recovery pass."""

        if not result_field_is_beam_section(selection.field_key.request.field_id):
            return (selection.field_key,)
        keys = tuple(
            availability.key for availability in visible_result_fields(provider.catalog().fields)
            if availability.state is FieldState.LAZY
            and result_field_is_beam_section(availability.descriptor.field_id)
        )
        return keys or (selection.field_key,)

    def export_csv(
        self, command_id: int, document: WorkspaceDocument,
        provider: ResultProvider | None, path: str | Path, spec: ResultCsvExportSpec,
    ) -> GuiCommandReceipt:
        return self._export(command_id, document, provider, path, spec, "csv")

    def export_vtk(
        self, command_id: int, document: WorkspaceDocument,
        provider: ResultProvider | None, path: str | Path, spec: ResultVtkExportSpec,
    ) -> GuiCommandReceipt:
        return self._export(command_id, document, provider, path, spec, "vtk")

    def _export(
        self, command_id: int, document: WorkspaceDocument,
        provider: ResultProvider | None, path: str | Path,
        spec: ResultCsvExportSpec | ResultVtkExportSpec, kind: str,
    ) -> GuiCommandReceipt:
        if document.task_controller.busy:
            return self._rejected(command_id, "task.busy", "a background task is already running")
        expected_type = ResultCsvExportSpec if kind == "csv" else ResultVtkExportSpec
        if type(spec) is not expected_type:
            return self._rejected(command_id, "command.type.invalid", f"spec must be {expected_type.__name__}")
        label = kind.upper()
        try:
            target = self._export_path(path, kind)
            exports = self._prepare_exports(document, provider, spec)
            export = exports[0]
            completion = GuiCommandCompletion(command_id)

            def workload(context: TaskContext) -> GuiCommandOutcome:
                if kind == "csv":
                    installed = write_result_table_csv(target, exports, checkpoint=context.checkpoint)
                else:
                    installed = write_result_vtk(
                        target, export, spec.deformation_scale, checkpoint=context.checkpoint,
                    )
                return GuiCommandOutcome(
                    output_path=installed, source=export.source,
                    materialization_generation=export.materialization_generation,
                    selection=export.selection, record_count=len(export.field.locations),
                )

            started = self.tasks.start(
                workload,
                lambda _value: self.tasks.presentation.set_status(f"{label} export complete", 5000),
                f"Failed to export {label}", context=document, controller=document.task_controller,
                task_name=f"{label} Export", completion=completion,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected(command_id, f"result.{kind}_export.rejected", error)
        if not started:
            return self._rejected(command_id, "task.start.rejected", f"the {label} export task could not be started")
        return GuiCommandReceipt.pending(command_id, completion)

    def export_csv_to(
        self, document: WorkspaceDocument, provider: ResultProvider | None,
        path: str | Path, spec: ResultCsvExportSpec,
    ) -> tuple[int, str]:
        """Keep the Agent's synchronous ready-field export contract."""

        if document.task_controller.busy:
            raise RuntimeError("a background task is already running")
        if type(spec) is not ResultCsvExportSpec:
            raise TypeError("spec must be ResultCsvExportSpec")
        target = self._export_path(path, "csv")
        installed = write_result_table_csv(target, self._prepare_exports(document, provider, spec))
        return verify_export_file_size(installed, maximum_bytes=MAX_EXPORT_CSV_BYTES)

    def _prepare_exports(
        self, document: WorkspaceDocument, provider: ResultProvider | None,
        spec: ResultCsvExportSpec | ResultVtkExportSpec,
    ) -> tuple[ResultExportSnapshot, ...]:
        if not self._provider_is_current(document, provider):
            raise RuntimeError("there is no current accepted result")
        materialization = provider.snapshot
        if materialization.source != spec.source:
            raise RuntimeError("export source does not match the current accepted result")
        if materialization.generation != spec.materialization_generation:
            raise RuntimeError("export generation does not match the current materialization")
        selections = spec.selections if type(spec) is ResultCsvExportSpec else (spec.selection,)
        return tuple(prepare_result_export_snapshot(materialization, selection) for selection in selections)

    def _provider_is_current(
        self, document: WorkspaceDocument, provider: ResultProvider | None,
    ) -> bool:
        return (
            self.tasks.is_live(document) and type(provider) is ResultProvider
            and document.session.current_result_identity() == (provider.source, provider.snapshot.generation)
        )

    def _project_delta(self, document: WorkspaceDocument, delta: SessionDelta) -> bool:
        if not delta.accepted:
            return False
        if delta.changed or delta.invalidated:
            return self.tasks.presentation.project_delta(document, delta)
        return True

    def _failed(self, document: WorkspaceDocument, token: object, title: str, message: str) -> None:
        delta = document.session.accept_task_failed(token, message)
        if self._project_delta(document, delta) and self.tasks.is_active(document):
            self.tasks.presentation.show_error(title, message)

    def _cancelled(self, document: WorkspaceDocument, token: object) -> None:
        self._project_delta(document, document.session.accept_task_cancelled(token))

    @staticmethod
    def _export_path(path: str | Path, kind: str) -> Path:
        target = Path(path)
        if target.suffix.casefold() != f".{kind}":
            label = "result CSV" if kind == "csv" else "canonical result VTK"
            raise ValueError(f"{label} target must use the .{kind} extension")
        return target

    @staticmethod
    def _selection_outcome(provider: ResultProvider, selection: ScalarFieldSelection) -> GuiCommandOutcome:
        return GuiCommandOutcome(
            source=provider.source, materialization_generation=provider.snapshot.generation,
            selection=selection, record_count=len(provider.field(selection.field_key).locations),
        )

    @staticmethod
    def _validate_query_result(provider: ResultProvider, query: ResultQuery, result: ResultQueryResult) -> None:
        if type(result) is not ResultQueryResult:
            raise TypeError("provider query must return ResultQueryResult")
        if (
            result.source != provider.source or result.materialization_generation != provider.snapshot.generation
            or result.query != query
        ):
            raise ValueError("query result must match the provider source, generation, and exact query")

    @staticmethod
    def _query_outcome(result: ResultQueryResult) -> GuiCommandOutcome:
        return GuiCommandOutcome(
            source=result.source, materialization_generation=result.materialization_generation,
            selection=ScalarFieldSelection(result.query.field_key, result.query.component),
            record_count=len(result.records),
        )

    @staticmethod
    def _rejected(command_id: int, code: str, error: object, remediation: str = "") -> GuiCommandReceipt:
        return GuiCommandReceipt.rejected(
            command_id, GuiCommandDiagnostic(code, str(error).strip() or type(error).__name__, remediation),
        )
