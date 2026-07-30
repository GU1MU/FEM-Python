"""Strict model-callable tool registry and deterministic dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from ..artifacts import ArtifactStore
from ..capabilities import show_capabilities
from ..confirmation import ConfirmationStore
from ..diagnostics import (
    DiagnosticCode,
    exception_diagnostic,
    has_errors,
    make_diagnostic,
)
from ..providers.base import ToolDefinition
from ..schemas import (
    ExportFormat,
    ResultQuery,
    ResultQueryKind,
    RunStatus,
    ToolResult,
    UnitContext,
)
from ..state import RevisionRecord, RevisionStore
from ..worker import (
    InspectionWorkerError,
    IsolatedFEMInspector,
    IsolatedFEMResultQuerier,
    ResultQueryWorkerError,
)

if TYPE_CHECKING:
    from ..worker import InspectionResponse, WorkerResponse


class DynamicToolRegistry(Protocol):
    @property
    def definitions(self) -> tuple[ToolDefinition, ...]: ...

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: "ToolExecutionContext",
    ) -> ToolResult: ...


@dataclass(frozen=True)
class ToolExecutionContext:
    session_id: str
    expected_revision: int
    idempotency_key: str
    completed_run: WorkerResponse | None = None


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    mutates_revision: bool = False
    requires_confirmation: bool = False


class AgentToolRegistry:
    """Publish a small schema catalog and reject malformed calls locally."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        cancel_event: threading.Event | None = None,
        dynamic_tools: DynamicToolRegistry | None = None,
    ):
        self.artifacts = ArtifactStore(workspace)
        self.revisions = RevisionStore(self.artifacts.root)
        self.confirmations = ConfirmationStore(
            self.artifacts.root,
            self.revisions,
        )
        self.inspector = IsolatedFEMInspector(self.artifacts.root)
        self.result_querier = IsolatedFEMResultQuerier(self.artifacts.root)
        self._cancel_event = cancel_event
        self._inspection_cache: dict[
            tuple[str, bool],
            "InspectionResponse",
        ] = {}
        self._inspection_lock = threading.RLock()
        self._tools = _registered_tools()
        self._dynamic_tools = dynamic_tools

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        static = tuple(item.definition for item in self._tools.values())
        if self._dynamic_tools is None:
            return static
        dynamic = tuple(self._dynamic_tools.definitions)
        static_names = {item.name for item in static}
        duplicates = static_names.intersection(
            item.name for item in dynamic
        )
        if duplicates:
            raise ValueError(
                "dynamic Agent tools conflict with V0 tools: "
                + ", ".join(sorted(duplicates))
            )
        return static + dynamic

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        tool = self._tools.get(str(name))
        if tool is None:
            if self._dynamic_tools is not None and name in {
                item.name for item in self._dynamic_tools.definitions
            }:
                return self._dynamic_tools.dispatch(
                    name,
                    arguments,
                    context,
                )
            return self._failure(
                context,
                DiagnosticCode.UNKNOWN_TOOL,
                f"Unknown Agent tool {name!r}.",
            )
        if not isinstance(arguments, Mapping):
            return self._failure(
                context,
                DiagnosticCode.INVALID_TOOL_ARGUMENTS,
                "Tool arguments must be a JSON object.",
            )
        try:
            if name == "show_capabilities":
                _require_fields(arguments, required=set())
                return self._success(
                    context,
                    "V0 capabilities are available.",
                    data=show_capabilities(),
                    allow_empty=True,
                )
            record = self.revisions.require_current(
                context.session_id,
                expected_revision=context.expected_revision,
            )
            if name == "inspect_abaqus":
                _require_fields(arguments, required=set())
                return self._inspect(record, context)
            if name == "set_unit_context":
                units = UnitContext.from_dict(arguments)
                updated = self.revisions.mutate(
                    record.session_id,
                    expected_revision=record.revision,
                    idempotency_key=context.idempotency_key,
                    changes={"unit_context": units},
                    operation="set_units",
                )
                return self._success(
                    context,
                    "Unit context recorded; no conversion was performed.",
                    output_revision=updated.revision,
                    data={"unit_context": units.to_dict()},
                )
            if name == "set_result_requests":
                return self._set_result_requests(record, arguments, context)
            if name == "get_analysis_summary":
                _require_fields(arguments, required=set())
                summary = self.analysis_summary(record)
                return self._success(
                    context,
                    "Deterministic analysis summary generated locally.",
                    data={"analysis_summary": summary.to_dict()},
                    diagnostics=summary.diagnostics,
                )
            if name == "validate_analysis":
                _require_fields(arguments, required=set())
                inspection = self.inspect_record(record, validate=True)
                diagnostics = inspection.diagnostics
                return self._success(
                    context,
                    (
                        "Validation produced blocking diagnostics."
                        if has_errors(diagnostics)
                        else "Deterministic model validation passed."
                    ),
                    data={
                        "validated": not has_errors(diagnostics),
                        "analysis_step": record.spec.analysis_step,
                    },
                    diagnostics=tuple(diagnostics),
                    ok=not has_errors(diagnostics),
                )
            if name == "solve_confirmed_analysis":
                _require_fields(arguments, required=set())
                return self._failure(
                    context,
                    DiagnosticCode.CONFIRMATION_REQUIRED,
                    (
                        "The cloud model cannot authorize a solve. "
                        "The user must enter /confirm for the current revision."
                    ),
                )
            if name == "query_results":
                data = _require_fields(
                    arguments,
                    required=set(),
                    optional={"queries"},
                )
                raw_queries = data.get("queries")
                queries = (
                    None
                    if raw_queries is None
                    else tuple(
                        ResultQuery.from_dict(_mapping(item, "query"))
                        for item in _array(raw_queries, "queries")
                    )
                )
                if queries == ():
                    raise ValueError("at least one result query is required")
                return self._completed_results(context, queries)
            if name == "export_results":
                _require_fields(arguments, required=set())
                return self._completed_exports(context)
            if name == "list_artifacts":
                _require_fields(arguments, required=set())
                records = self.artifacts.list_artifacts(context.session_id)
                return self._success(
                    context,
                    f"{len(records)} local artifacts are registered.",
                    data={"artifacts": [item.to_dict() for item in records]},
                    artifacts=records,
                )
        except Exception as error:
            if isinstance(error, InspectionWorkerError):
                message = str(error).casefold()
                if (
                    self._cancel_event is not None
                    and self._cancel_event.is_set()
                ) or "cancelled" in message:
                    code = DiagnosticCode.OPERATION_CANCELLED
                    summary = "The model inspection was cancelled."
                elif "time limit" in message:
                    code = DiagnosticCode.WORKER_TIMEOUT
                    summary = "The isolated model inspection exceeded its time limit."
                else:
                    code = DiagnosticCode.WORKER_CRASH
                    summary = "The isolated model inspection process failed."
                diagnostic = make_diagnostic(
                    code,
                    summary,
                    source=f"agent.tool.{name}",
                    remediation=(
                        "Retry the inspection in a new user turn. "
                        "If it recurs, inspect the local worker logs."
                    ),
                )
            else:
                diagnostic = exception_diagnostic(
                    (
                        DiagnosticCode.INVALID_TOOL_ARGUMENTS
                        if isinstance(error, (TypeError, ValueError, KeyError))
                        else DiagnosticCode.INVALID_MODEL
                    ),
                    error,
                    source=f"agent.tool.{name}",
                )
            return ToolResult(
                ok=False,
                session_id=context.session_id,
                input_revision=max(context.expected_revision, 0),
                idempotency_key=context.idempotency_key,
                summary=diagnostic.message,
                diagnostics=(diagnostic,),
            )
        return self._failure(
            context,
            DiagnosticCode.UNKNOWN_TOOL,
            f"Tool {name!r} has no dispatcher.",
        )

    def analysis_summary(self, record: RevisionRecord):
        return self.inspect_record(record).summary

    def inspect_record(
        self,
        record: RevisionRecord,
        *,
        validate: bool = False,
    ) -> "InspectionResponse":
        key = (record.revision_hash, bool(validate))
        with self._inspection_lock:
            cached = self._inspection_cache.get(key)
            if cached is not None:
                return cached
            inspected = self.inspector.inspect(
                record.spec,
                record.revision_hash,
                validate=validate,
                cancel_event=self._cancel_event,
            )
            if len(self._inspection_cache) >= 32:
                oldest = next(iter(self._inspection_cache))
                self._inspection_cache.pop(oldest)
            self._inspection_cache[key] = inspected
            return inspected

    def _inspect(
        self,
        record: RevisionRecord,
        context: ToolExecutionContext,
    ) -> ToolResult:
        inspection = self.inspect_record(record)
        summary = inspection.summary
        updated = record
        runnable_name = (
            None
            if summary.analysis_step is None
            else str(summary.analysis_step.get("name") or "") or None
        )
        if (
            runnable_name is not None
            and runnable_name != record.spec.analysis_step
        ):
            updated = self.revisions.mutate(
                record.session_id,
                expected_revision=record.revision,
                idempotency_key=context.idempotency_key,
                changes={"analysis_step": runnable_name},
                operation="inspect_abaqus",
            )
            summary = self.inspect_record(updated).summary
        return self._success(
            context,
            (
                "Abaqus input inspected with blocking diagnostics."
                if summary.has_blocking_diagnostics
                else "Abaqus input inspected."
            ),
            output_revision=(
                updated.revision if updated.revision != record.revision else None
            ),
            data={
                "analysis_summary": summary.to_dict(),
            },
            diagnostics=summary.diagnostics,
            ok=not summary.has_blocking_diagnostics,
        )

    def _set_result_requests(
        self,
        record: RevisionRecord,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        active_run = context.completed_run
        if (
            active_run is not None
            and active_run.session_id == record.session_id
            and active_run.revision == record.revision
            and active_run.status == RunStatus.SUCCEEDED
        ):
            return self._failure(
                context,
                DiagnosticCode.INVALID_TOOL_ARGUMENTS,
                (
                    "The analysis is already solved. Use query_results for "
                    "post-solve questions instead of changing the revision."
                ),
            )
        data = _require_fields(
            arguments,
            required={"queries", "export_formats"},
        )
        queries_raw = _array(data["queries"], "queries")
        formats_raw = _array(data["export_formats"], "export_formats")
        queries = tuple(
            ResultQuery.from_dict(_mapping(item, "query"))
            for item in queries_raw
        )
        formats = tuple(ExportFormat(item) for item in formats_raw)
        updated = self.revisions.mutate(
            record.session_id,
            expected_revision=record.revision,
            idempotency_key=context.idempotency_key,
            changes={
                "requested_queries": queries,
                "export_formats": formats,
            },
            operation="set_results",
        )
        return self._success(
            context,
            "Optional precomputed result requests and export formats recorded.",
            output_revision=updated.revision,
            data={
                "requested_queries": [item.to_dict() for item in queries],
                "export_formats": [item.value for item in formats],
            },
        )

    def _completed_results(
        self,
        context: ToolExecutionContext,
        queries: tuple[ResultQuery, ...] | None,
    ) -> ToolResult:
        response = context.completed_run
        if (
            response is None
            or response.session_id != context.session_id
            or response.revision != context.expected_revision
            or response.status != RunStatus.SUCCEEDED
        ):
            return self._failure(
                context,
                DiagnosticCode.RESULT_QUERY_FAILED,
                "No completed run is available for result queries.",
            )
        if queries is None:
            if response.result_summary is None:
                return self._failure(
                    context,
                    DiagnosticCode.RESULT_QUERY_FAILED,
                    (
                        "No precomputed result summary is available. Supply "
                        "one or more queries to query_results."
                    ),
                )
            summary = response.result_summary
        else:
            solution_available = any(
                item.kind == "solution" for item in response.artifacts
            )
            if not solution_available:
                record = self.revisions.require_current(
                    context.session_id,
                    expected_revision=context.expected_revision,
                )
                if (
                    response.result_summary is not None
                    and queries == record.spec.requested_queries
                ):
                    summary = response.result_summary
                else:
                    return self._failure(
                        context,
                        DiagnosticCode.RESULT_QUERY_FAILED,
                        (
                            "This completed run predates reusable "
                            "solution storage. Start a new analysis run before "
                            "requesting additional result quantities."
                        ),
                    )
            else:
                try:
                    summary = self.result_querier.query(
                        response,
                        queries,
                        cancel_event=self._cancel_event,
                    )
                except ResultQueryWorkerError:
                    return self._failure(
                        context,
                        DiagnosticCode.RESULT_QUERY_FAILED,
                        "The isolated post-solve result query failed.",
                    )
        return self._success(
            context,
            "Bounded results queried from the completed solution.",
            data={"result_summary": summary.to_dict()},
            diagnostics=summary.diagnostics,
            ok=not has_errors(summary.diagnostics),
        )

    def _completed_exports(
        self,
        context: ToolExecutionContext,
    ) -> ToolResult:
        response = context.completed_run
        if (
            response is None
            or response.session_id != context.session_id
            or response.revision != context.expected_revision
            or response.status != RunStatus.SUCCEEDED
        ):
            return self._failure(
                context,
                DiagnosticCode.EXPORT_FAILED,
                "No completed run is available for exports.",
            )
        exports = tuple(
            item
            for item in response.artifacts
            if item.kind in {ExportFormat.CSV.value, ExportFormat.VTK.value}
        )
        return self._success(
            context,
            f"{len(exports)} result export artifacts are available.",
            data={"artifacts": [item.to_dict() for item in exports]},
            artifacts=exports,
        )

    def _success(
        self,
        context: ToolExecutionContext,
        summary: str,
        *,
        output_revision: int | None = None,
        data: Mapping[str, Any] | None = None,
        diagnostics=(),
        artifacts=(),
        ok: bool = True,
        allow_empty: bool = False,
    ) -> ToolResult:
        input_revision = max(context.expected_revision, 0)
        if not allow_empty and input_revision == 0:
            raise ValueError("an attached analysis revision is required")
        return ToolResult(
            ok=ok,
            session_id=context.session_id,
            input_revision=input_revision,
            output_revision=output_revision,
            idempotency_key=context.idempotency_key,
            summary=summary,
            data=data,
            diagnostics=tuple(diagnostics),
            artifacts=tuple(artifacts),
        )

    def _failure(
        self,
        context: ToolExecutionContext,
        code: DiagnosticCode,
        message: str,
    ) -> ToolResult:
        diagnostic = make_diagnostic(
            code,
            message,
            source="agent.tool_registry",
        )
        return ToolResult(
            ok=False,
            session_id=context.session_id,
            input_revision=max(context.expected_revision, 0),
            idempotency_key=context.idempotency_key,
            summary=message,
            diagnostics=(diagnostic,),
        )


def _registered_tools() -> dict[str, RegisteredTool]:
    no_arguments = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    query_schema = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [item.value for item in ResultQueryKind],
            },
            "component": {"type": ["integer", "null"]},
            "node_id": {"type": ["integer", "null"]},
            "node_set": {
                "type": ["string", "null"],
                "description": (
                    "A named node set listed under node_sets in the model "
                    "summary."
                ),
            },
            "edge": {
                "type": ["string", "null"],
                "description": (
                    "A named 2D edge listed under edges in the model summary. "
                    "Abaqus 2D element surfaces are represented as edges."
                ),
            },
            "surface": {
                "type": ["string", "null"],
                "description": (
                    "A named 3D face region listed under surfaces in the "
                    "model summary."
                ),
            },
            "element_set": {"type": ["string", "null"]},
            "measure": {"type": ["string", "null"]},
        },
        "required": ["kind"],
        "additionalProperties": False,
    }
    definitions = (
        RegisteredTool(
            ToolDefinition(
                "show_capabilities",
                "Report the supported local FEM Agent V0 capabilities.",
                no_arguments,
            )
        ),
        RegisteredTool(
            ToolDefinition(
                "inspect_abaqus",
                "Inspect the already attached Abaqus artifact locally.",
                no_arguments,
            ),
            mutates_revision=True,
        ),
        RegisteredTool(
            ToolDefinition(
                "set_unit_context",
                "Record the user's complete unit declaration without conversion.",
                {
                    "type": "object",
                    "properties": {
                        name: {"type": "string"}
                        for name in (
                            "length",
                            "force",
                            "stress",
                            "density",
                            "acceleration",
                        )
                    }
                    | {
                        "convention": {"type": ["string", "null"]},
                    },
                    "required": [
                        "length",
                        "force",
                        "stress",
                        "density",
                        "acceleration",
                    ],
                    "additionalProperties": False,
                },
            ),
            mutates_revision=True,
        ),
        RegisteredTool(
            ToolDefinition(
                "set_result_requests",
                (
                    "Optionally record precomputed result queries and local "
                    "export formats before solving. This is not required for "
                    "confirmation. "
                    "For aggregate nodal queries, use node_set, edge, or "
                    "surface according to the entity type in the model summary."
                ),
                {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": query_schema,
                        },
                        "export_formats": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [item.value for item in ExportFormat],
                            },
                        },
                    },
                    "required": ["queries", "export_formats"],
                    "additionalProperties": False,
                },
            ),
            mutates_revision=True,
        ),
        RegisteredTool(
            ToolDefinition(
                "get_analysis_summary",
                "Return the deterministic, bounded confirmation summary.",
                no_arguments,
            )
        ),
        RegisteredTool(
            ToolDefinition(
                "validate_analysis",
                "Run deterministic import and FEM model validation locally.",
                no_arguments,
            )
        ),
        RegisteredTool(
            ToolDefinition(
                "solve_confirmed_analysis",
                "Solve only after the user confirms the current revision with /confirm.",
                no_arguments,
            ),
            requires_confirmation=True,
        ),
        RegisteredTool(
            ToolDefinition(
                "query_results",
                (
                    "After a successful solve, evaluate bounded queries from "
                    "the saved solution without changing the revision or "
                    "repeating the solve."
                ),
                {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": query_schema,
                            "minItems": 1,
                            "maxItems": 64,
                        },
                    },
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            ),
            requires_confirmation=True,
        ),
        RegisteredTool(
            ToolDefinition(
                "export_results",
                "List result exports already produced inside the run directory.",
                no_arguments,
            ),
            requires_confirmation=True,
        ),
        RegisteredTool(
            ToolDefinition(
                "list_artifacts",
                "List opaque local artifact records without exposing absolute paths.",
                no_arguments,
            )
        ),
    )
    return {item.definition.name: item for item in definitions}


def _require_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    data = dict(value)
    optional = optional or set()
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise ValueError(f"missing tool fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unknown tool fields: {', '.join(sorted(unknown))}")
    return data


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return value


__all__ = [
    "AgentToolRegistry",
    "DynamicToolRegistry",
    "RegisteredTool",
    "ToolExecutionContext",
]
