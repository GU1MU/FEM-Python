"""GUI adapter for CSV export receipts and workspace export ledger records."""

from __future__ import annotations

from datetime import datetime, timezone

from fem.application.results import ResultSourceKey
from fem_agent.export_authoring import (
    EXPORT_CSV_TOOL_NAME,
    ExportCsvRequest,
    ExportCsvResponse,
    ExportFileReceipt,
    ResultDisplayContextResponse,
)
from fem_agent.export_storage import (
    ExportStorageError,
    allocate_export_path,
    ensure_export_directory,
    sanitize_export_filename,
)

from .agent_workspace import ExportLedgerRecord, append_export_ledger_record
from .commands import ResultCsvExportSpec


class SessionExportPort:
    """Agent export port: write CSV to the user workspace and record it in the export ledger.

    The port depends only on the export facade for fake injection in tests. Without a workspace, return
    a fixed diagnostic; if the workspace becomes unavailable, report it without cleanup or retries.
    """

    def __init__(self, facade: object) -> None:
        for attribute in (
            "current_workspace",
            "agent_data_root",
            "document_id",
            "result_display_context",
            "resolve_result_selection",
            "export_result_csv_to",
        ):
            if not callable(getattr(facade, attribute, None)):
                raise TypeError("export facade does not implement the protocol")
        self._facade = facade

    def read_result_display_context(self) -> ResultDisplayContextResponse:
        try:
            context = self._facade.result_display_context()
        except (RuntimeError, ValueError) as error:
            return ResultDisplayContextResponse.failure(
                "export.context.unavailable",
                f"No READY results available for export: {str(error)[:256]}",
                retryable=False,
                clarification_required=True,
            )
        return ResultDisplayContextResponse.success(context)

    def export_accepted_result_csv(
        self,
        request: ExportCsvRequest,
    ) -> ExportCsvResponse:
        workspace = self._facade.current_workspace()
        if workspace is None:
            return ExportCsvResponse.no_workspace()
        try:
            context = self._facade.result_display_context()
        except (RuntimeError, ValueError) as error:
            return ExportCsvResponse.failure(
                "export.result.unavailable",
                f"No READY results available for export: {str(error)[:256]}",
                retryable=False,
                clarification_required=True,
            )
        field = next(
            (
                item
                for item in context.fields
                if item.field_ref == request.field_ref
            ),
            None,
        )
        if field is None:
            ready_refs = ", ".join(
                item.field_ref for item in context.fields[:8]
            )
            return ExportCsvResponse.failure(
                "export.field.unknown",
                (
                    "field_ref is not in the current READY field catalog; "
                    f"available fields: {ready_refs}"
                ),
                retryable=False,
                clarification_required=True,
            )
        if request.component not in field.components:
            return ExportCsvResponse.failure(
                "export.component.unknown",
                (
                    "component does not belong to the selected field; "
                    f"available components: {', '.join(field.components)}"
                ),
                retryable=False,
                clarification_required=True,
            )
        try:
            selection = self._facade.resolve_result_selection(
                request.field_ref,
                request.component,
            )
        except (RuntimeError, ValueError) as error:
            return ExportCsvResponse.failure(
                "export.field.unknown",
                str(error)[:512],
                retryable=False,
                clarification_required=True,
            )
        stem_source = (
            request.name
            if request.name is not None
            else f"{field.display_name}_{request.expected_source.run_id}"
        )
        filename = sanitize_export_filename(stem_source, ".csv")
        try:
            exports_root = ensure_export_directory(workspace.root)
            target = allocate_export_path(exports_root, filename)
        except ExportStorageError as error:
            return ExportCsvResponse.failure(
                "export.storage.rejected",
                str(error)[:512] or "Export directory unavailable",
                retryable=False,
                clarification_required=True,
            )
        source = request.expected_source
        spec = ResultCsvExportSpec(
            ResultSourceKey(
                result_id=source.result_id,
                session_id=source.session_id,
                artifact_id=source.artifact_id,
                model_revision=source.model_revision,
                step_name=source.step_name,
                run_id=source.run_id,
            ),
            request.expected_materialization_generation,
            selection,
        )
        try:
            size_bytes, digest = self._facade.export_result_csv_to(target, spec)
        except OSError:
            return ExportCsvResponse.failure(
                "export.workspace.unavailable",
                "Workspace directory unavailable; export incomplete. Check the workspace and have the user retry",
                retryable=False,
                clarification_required=True,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            message = str(error)
            if "background task" in message:
                code = "export.busy"
            elif "does not match" in message:
                code = "export.result.stale"
            else:
                code = "export.rejected"
            return ExportCsvResponse.failure(
                code,
                message[:512],
                retryable=False,
                clarification_required=code != "export.busy",
            )
        relative_path = target.relative_to(workspace.root).as_posix()
        document_id = self._facade.document_id()
        try:
            append_export_ledger_record(
                self._facade.agent_data_root(),
                workspace.workspace_id,
                ExportLedgerRecord(
                    display_path=relative_path,
                    kind="csv",
                    sha256=digest,
                    size_bytes=size_bytes,
                    document_id=(
                        str(document_id)
                        if document_id
                        else f"document:{source.session_id}"
                    ),
                    session_id=source.session_id,
                    run_id=source.run_id,
                    materialization_generation=(
                        request.expected_materialization_generation
                    ),
                    overrides_summary=(
                        request.name if request.name is not None else "none"
                    ),
                    exported_at=datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    tool=EXPORT_CSV_TOOL_NAME,
                ),
            )
        except (OSError, ValueError):
            # The ledger is a best-effort audit record; ledger failure does not roll back a successful write.
            pass
        return ExportCsvResponse.success(
            ExportFileReceipt(
                workspace_relative_path=relative_path,
                filename=target.name,
                sha256=digest,
                size_bytes=size_bytes,
                kind="csv",
            )
        )


__all__ = ["SessionExportPort"]
