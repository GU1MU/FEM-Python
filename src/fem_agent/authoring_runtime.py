"""Dynamic authoring workflow and state ownership over local tool handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import re
import threading
from typing import TYPE_CHECKING, Any

from .authoring import (
    AuthoringContext,
    AuthoringContractError,
    ProposalState,
    RequirementLedger,
    RequirementStatus,
)
from .diagnostics import DiagnosticCode, make_diagnostic
from .providers.base import ToolDefinition
from .result_authoring import (
    ANALYSIS_RUN_CATALOG_TOOL_NAME,
    RESULT_CATALOG_TOOL_NAME,
    RESULT_COMPARISON_TOOL_NAME,
    RESULT_QUERY_TOOL_NAME,
)
from .export_authoring import (
    EXPORT_CSV_TOOL_NAME,
    RESULT_DISPLAY_CONTEXT_TOOL_NAME,
)
from .schemas import ToolResult
from .authoring_values import (
    AUTHORING_TURN_SNAPSHOT_MAX_BYTES,
    AUTHORING_TURN_SNAPSHOT_MAX_ITEMS,
    AUTHORING_TURN_SNAPSHOT_SCHEMA_VERSION,
    AuthoringTerminalRecord,
    AuthoringTurnSnapshot,
    AuthoringToolOutcome,
    AuthoringWorkflowStage,
    PlanarConstructionAuditRecord,
    ProjectSaveProposalRecord,
)
from .authoring_payload import (
    _provider_safe_error_detail,
    provider_safe_authoring_payload,
)
from .authoring_validation import (
    _AuthoringArgumentSchemaError,
    _nonblank_string,
    _require_exact_fields,
    _validate_geometry_edit_arguments,
    _validate_requirement_value,
)
from .authoring_schemas.session import (
    _CREATE_NATIVE_MODEL_DOCUMENT,
    _READ_CONTEXT,
    _REQUEST_PROJECT_SAVE,
    _REQUIREMENT_SPECS,
    _SET_REQUIREMENTS,
)
from .authoring_schemas.geometry import (
    _PREPARE_GEOMETRY,
    _PREPARE_GEOMETRY_EDIT,
    _PREPARE_PLANAR_CONSTRUCTION,
    _PREPARE_PROFILE_EXTRUSION,
    _PREPARE_PROFILE_PATH_SWEEP,
    _PREPARE_PROFILE_REVOLUTION,
    _READ_GEOMETRY_EDIT_CONTEXT,
    _planar_construction_context_capability,
)
from .authoring_schemas.definitions import (
    _APPLY_DEFINITION,
    _EDIT_MODEL_OBJECT,
    _PREPARE_DELETE,
)
from .authoring_schemas.analysis import (
    _PREPARE_MESH,
    _PREPARE_SOLVE,
    _RUN_PREFLIGHT,
)
from .authoring_schemas.catalogs import _GEOMETRY_FEATURE_CATALOG
from .authoring_tool_policy import (
    PROFILE_TRANSFORM_TOOL_NAMES as _PROFILE_TRANSFORM_TOOLS,
    AuthoringToolPolicyState,
    GeometryEditSchemaScope as _GeometryEditSchemaScope,
    capability_enabled as _capability_enabled,
    current_requirement_group,
    project_save_available,
    publication_needs_live_context,
    publication_needs_workspace_export,
    publish_authoring_tools,
    requirement_keys,
    requirement_spec,
    requirement_stage as _requirement_stage,
)

if TYPE_CHECKING:
    from .tools.registry import ToolExecutionContext


@dataclass(frozen=True, slots=True)
class _PlanarConstructionRetryState:
    turn_id: str
    construction_digest: str
    recovery_digest: str
    failure_signature_digest: str | None
    diagnostic_code: str
    node_id: str | None
    allowed_fields: tuple[str, ...]
    attempt: int


AuthoringToolHandler = Callable[
    [Mapping[str, object], "AuthoringWorkflowController"],
    AuthoringToolOutcome,
]
ContextReader = Callable[[], AuthoringContext | Mapping[str, object]]


_FORBIDDEN_CONFIRMATION_TOOL_NAMES = frozenset(
    {
        "accept_proposal",
        "confirm_mesh",
        "confirm_solve",
        "reject_proposal",
        "cancel_operation",
    }
)
_DEFAULT_GEOMETRY_REQUIREMENTS: dict[str, str] = {
    "length_unit": "mm",
    "force_unit": "N",
    "stress_unit": "MPa",
}
_DEFAULT_REQUIREMENT_SOURCE_TURN_ID = "agent-default"


class AuthoringWorkflowController:
    """Strict A8 state machine over injected A1-A7 local handlers."""

    def __init__(
        self,
        context_reader: ContextReader,
        handlers: Mapping[str, AuthoringToolHandler],
        *,
        workspace_result_inventory: Callable[[], tuple[int, int]] | None = None,
        workspace_export_available: Callable[[], bool] | None = None,
    ) -> None:
        if not callable(context_reader):
            raise TypeError("context_reader must be callable")
        normalized = dict(handlers)
        if any(
            not isinstance(name, str) or not callable(handler)
            for name, handler in normalized.items()
        ):
            raise TypeError("authoring handlers must be named callables")
        forbidden = [
            name
            for name in normalized
            if name.casefold() in _FORBIDDEN_CONFIRMATION_TOOL_NAMES
        ]
        if forbidden:
            raise ValueError("confirmation handlers cannot be model-callable")
        self._context_reader = context_reader
        self._handlers = normalized
        if workspace_result_inventory is not None and not callable(
            workspace_result_inventory
        ):
            raise TypeError("workspace_result_inventory must be callable or None")
        if workspace_export_available is not None and not callable(
            workspace_export_available
        ):
            raise TypeError("workspace_export_available must be callable or None")
        self._workspace_result_inventory = workspace_result_inventory
        self._workspace_export_available = workspace_export_available
        self._ledger = RequirementLedger()
        self._stage = AuthoringWorkflowStage.REQUIREMENTS
        self._pending_operation: str | None = None
        self._project_save_record: ProjectSaveProposalRecord | None = None
        self._geometry_resume_stage: AuthoringWorkflowStage | None = None
        self._mesh_resume_stage: AuthoringWorkflowStage | None = None
        self._destructive_resume_stage: AuthoringWorkflowStage | None = None
        self._pending_destructive_object_type: str | None = None
        self._terminals: list[AuthoringTerminalRecord] = []
        self._planar_retry_state: _PlanarConstructionRetryState | None = None
        self._geometry_edit_schema_scope: _GeometryEditSchemaScope | None = None
        self._planar_audit: list[PlanarConstructionAuditRecord] = []
        self._pending_planar_proposal_id: str | None = None
        self._active_tool_context: ToolExecutionContext | None = None
        self._binding_identity: tuple[str, str, int] | None = None
        self._observed_context: AuthoringContext | None = None
        self._snapshot_generation = 0
        self._turn_snapshot = AuthoringTurnSnapshot.unavailable()
        self._lock = threading.RLock()
        # Set before cancel_turn takes the workflow lock: dispatch holds that
        # lock for the whole handler call, so long-running handlers need one
        # lock-free signal that a cancellation was requested.
        self._cancel_requested = threading.Event()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    @property
    def stage(self) -> AuthoringWorkflowStage:
        with self._lock:
            return self._stage

    @property
    def planar_construction_audit(self) -> tuple[PlanarConstructionAuditRecord, ...]:
        with self._lock:
            return tuple(self._planar_audit)

    @property
    def geometry_edit_supported_operations(self) -> tuple[str, ...]:
        with self._lock:
            scope = self._geometry_edit_schema_scope
            if scope is None or scope.binding_identity != self._binding_identity:
                return ()
            return scope.supported_operations

    def assess_planar_construction_failure(
        self,
        request: Mapping[str, object],
        *,
        code: str,
        node_id: str | None,
        retryable: bool,
        allowed_fields: Sequence[str],
        failure_evidence: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Bound retries and require changed payload and topology evidence."""

        raw_construction = request.get("construction")
        construction = (
            raw_construction if isinstance(raw_construction, Mapping) else request
        )
        payload = json.dumps(
            construction,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        fields = tuple(str(field) for field in allowed_fields)
        node_payload = next(
            (
                item
                for item in construction.get("nodes", [])
                if isinstance(item, Mapping) and item.get("id") == node_id
            ),
            None,
        )
        recovery_payload = {
            field: (
                node_payload
                if field == "nodes" and node_payload is not None
                else construction.get("nodes")
                if field == "nodes"
                else node_payload.get(field)
                if node_payload is not None and field in node_payload
                else construction
                if field == "construction"
                else request.get(field)
                if field in request
                else construction.get(field)
            )
            for field in fields
        }
        recovery_digest = hashlib.sha256(
            json.dumps(
                recovery_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            ).encode("utf-8")
        ).hexdigest()
        evidence_payload = dict(failure_evidence or {})
        failure_signature_digest = (
            hashlib.sha256(
                json.dumps(
                    {
                        "code": str(code),
                        "node_id": node_id,
                        "evidence": evidence_payload,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if evidence_payload
            else None
        )
        active = self._active_tool_context
        turn_id = (
            "local" if active is None else str(active.turn_id or active.idempotency_key)
        )
        with self._lock:
            prior = self._planar_retry_state
            if prior is not None and prior.turn_id != turn_id:
                prior = None
            attempt = 1 if prior is None else prior.attempt + 1
            same_failure_signature = bool(
                prior is not None
                and failure_signature_digest is not None
                and failure_signature_digest == prior.failure_signature_digest
            )
            changed = bool(
                prior is None
                or (
                    recovery_digest != prior.recovery_digest
                    and not same_failure_signature
                )
            )
            may_retry = bool(retryable and changed and attempt < 3)
            blocker = None
            if same_failure_signature:
                blocker = (
                    "Retry changed the construction syntax but preserved the "
                    "same topology failure evidence. Revise the failed node's "
                    "bounds or spatial relation."
                )
            elif prior is not None and not changed:
                blocker = "Retry must modify the failed node or allowed field."
            elif retryable and attempt >= 3:
                blocker = (
                    "Planar construction retry limit reached after three attempts."
                )
            self._planar_retry_state = _PlanarConstructionRetryState(
                turn_id,
                digest,
                recovery_digest,
                failure_signature_digest,
                str(code),
                node_id,
                fields,
                attempt,
            )
            self._append_planar_audit(
                PlanarConstructionAuditRecord(
                    digest,
                    "validation"
                    if code
                    in {
                        "planar-ir.schema-invalid",
                        "planar-ir.budget-exceeded",
                        "planar-ir.duplicate-node-id",
                        "planar-ir.reference-missing",
                        "planar-ir.cycle-detected",
                        "planar-ir.unreachable-node",
                        "planar-ir.invalid-primitive",
                        "planar-ir.invalid-path-stroke",
                    }
                    else "compile",
                    str(code),
                    None,
                    "failed",
                )
            )
            return {
                "construction_digest": digest,
                "attempt": attempt,
                "limit": 3,
                "retryable": may_retry,
                "blocker": blocker,
            }

    def planar_construction_retry_blocker(self) -> dict[str, object] | None:
        """Reject further planar submissions after this provider turn is exhausted."""

        active = self._active_tool_context
        if active is None:
            return None
        turn_id = str(active.turn_id or active.idempotency_key)
        with self._lock:
            prior = self._planar_retry_state
            if (
                prior is None
                or prior.turn_id != turn_id
                or prior.attempt < 3
            ):
                return None
            return {
                "construction_digest": prior.construction_digest,
                "attempt": prior.attempt,
                "limit": 3,
                "retryable": False,
                "blocker": (
                    "Planar construction retry limit reached after three attempts."
                ),
            }

    def record_planar_construction_proposal(
        self,
        construction_digest: str,
        proposal_id: str,
    ) -> None:
        with self._lock:
            self._planar_retry_state = None
            self._pending_planar_proposal_id = str(proposal_id)
            self._append_planar_audit(
                PlanarConstructionAuditRecord(
                    str(construction_digest),
                    "proposal",
                    None,
                    str(proposal_id),
                    ProposalState.PENDING_CONFIRMATION.value,
                )
            )

    def _append_planar_audit(self, record: PlanarConstructionAuditRecord) -> None:
        self._planar_audit.append(record)
        del self._planar_audit[:-64]

    @property
    def ledger(self) -> RequirementLedger:
        return self._ledger

    @property
    def terminal_records(self) -> tuple[AuthoringTerminalRecord, ...]:
        with self._lock:
            return tuple(self._terminals)

    @property
    def binding_identity(self) -> tuple[str, str, int] | None:
        """Return the last owner-thread-observed model identity."""

        with self._lock:
            return self._binding_identity

    @property
    def turn_snapshot(self) -> AuthoringTurnSnapshot:
        """Return the last owner-thread projected immutable turn snapshot."""

        with self._lock:
            return self._turn_snapshot

    @property
    def authoring_turn_snapshot(self) -> AuthoringTurnSnapshot:
        return self.turn_snapshot

    @property
    def provider_snapshot(self) -> AuthoringTurnSnapshot:
        snapshot = self.turn_snapshot
        # A controller can be used directly by headless baseline tests.  Until
        # its owner adapter binds the actual published tool catalog, exposing
        # document details would claim a provider turn that never received an
        # advertisement.  The Qt adapter calls ``set_published_tool_names``
        # before each turn.
        if not snapshot.published_tool_names:
            return AuthoringTurnSnapshot.unavailable(
                generation=snapshot.snapshot_generation,
            )
        return snapshot

    def refresh_turn_snapshot(
        self,
        published_tool_names: Sequence[str] = (),
    ) -> AuthoringTurnSnapshot:
        """Observe current typed context and update the owner-thread cache.

        The Qt adapter calls this method on its owner thread immediately before
        a provider turn and after local tool/proposal transitions.  Provider
        threads only consume :attr:`turn_snapshot` and never call the context
        reader through this method.
        """

        with self._lock:
            raw = self._context_reader()
            if type(raw) is AuthoringContext:
                self.observe_binding(raw)
            else:
                # A malformed/unavailable reader result must not keep using a
                # previously observed document as if it were current.
                self._observed_context = None
                self._binding_identity = None
            self._snapshot_generation += 1
            self._turn_snapshot = AuthoringTurnSnapshot.from_context(
                self._observed_context,
                workflow_stage=self._stage if self._observed_context else None,
                published_tool_names=published_tool_names,
                generation=self._snapshot_generation,
            )
            return self._turn_snapshot

    def set_published_tool_names(
        self,
        published_tool_names: Sequence[str],
    ) -> AuthoringTurnSnapshot:
        """Bind the owner-thread tool advertisement to the latest snapshot."""

        with self._lock:
            self._snapshot_generation += 1
            self._turn_snapshot = AuthoringTurnSnapshot.from_context(
                self._observed_context,
                workflow_stage=(
                    self._stage if self._observed_context is not None else None
                ),
                published_tool_names=published_tool_names,
                generation=self._snapshot_generation,
            )
            return self._turn_snapshot

    def invalidate_turn_snapshot(self) -> AuthoringTurnSnapshot:
        """Invalidate provider projection without discarding state-machine context."""

        with self._lock:
            self._snapshot_generation += 1
            self._turn_snapshot = AuthoringTurnSnapshot.unavailable(
                generation=self._snapshot_generation,
            )
            return self._turn_snapshot

    def _refresh_turn_snapshot_locked(self) -> None:
        self._snapshot_generation += 1
        self._turn_snapshot = AuthoringTurnSnapshot.from_context(
            self._observed_context,
            workflow_stage=self._stage if self._observed_context else None,
            published_tool_names=self._turn_snapshot.published_tool_names,
            generation=self._snapshot_generation,
        )

    @property
    def project_save_record(self) -> ProjectSaveProposalRecord | None:
        with self._lock:
            return self._project_save_record

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        with self._lock:
            inventory = self._result_inventory()
            policy_state = AuthoringToolPolicyState(
                stage=self._stage,
                registered_tool_names=frozenset(self._handlers),
                observed_context=self._observed_context,
                proposed_requirement_keys=frozenset(
                    item.key
                    for item in self._ledger.entries
                    if item.status is RequirementStatus.PROPOSED
                ),
                binding_identity=self._binding_identity,
                geometry_edit_scope=self._geometry_edit_schema_scope,
                workspace_result_inventory=inventory,
            )
            export_selected = (
                self._workspace_export_selected()
                if publication_needs_workspace_export(policy_state)
                else False
            )
            live_context = None
            if publication_needs_live_context(policy_state):
                try:
                    raw = self._context_reader()
                except Exception:
                    raw = None
                if type(raw) is AuthoringContext:
                    live_context = raw
            return publish_authoring_tools(
                policy_state,
                live_context=live_context,
                workspace_export_selected=export_selected,
            )

    def _record_geometry_edit_schema_scope(
        self,
        outcome: AuthoringToolOutcome,
    ) -> None:
        data = outcome.data
        part_id = data.get("part_id")
        recipe_kind = data.get("kind") or data.get("recipe_kind")
        raw_operations = data.get("supported_edits")
        if (
            not outcome.ok
            or type(part_id) is not str
            or type(recipe_kind) is not str
            or not isinstance(raw_operations, list)
            or self._binding_identity is None
        ):
            return
        operations = tuple(
            str(item)
            for item in raw_operations
            if isinstance(item, str) and item
        )
        if not operations:
            return
        self._geometry_edit_schema_scope = _GeometryEditSchemaScope(
            self._binding_identity,
            part_id,
            recipe_kind,
            operations,
        )

    def _result_inventory(self) -> tuple[int, int] | None:
        reader = self._workspace_result_inventory
        if reader is None:
            return None
        run_count, result_count = reader()
        if (
            type(run_count) is not int
            or type(result_count) is not int
            or run_count < 0
            or result_count < 0
            or result_count > run_count
        ):
            raise ValueError("workspace result inventory is invalid")
        return run_count, result_count

    def _workspace_export_selected(self) -> bool:
        reader = self._workspace_export_available
        if reader is None:
            return False
        try:
            return bool(reader())
        except Exception:
            return False

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        with self._lock:
            definitions = {item.name: item for item in self.definitions}
            available = set(definitions)
            if (
                name not in available
                and self._stage
                in {
                    AuthoringWorkflowStage.STALE,
                    AuthoringWorkflowStage.CANCELLED,
                }
                and name.startswith("read_")
                and (name == _READ_CONTEXT.name or name in self._handlers)
            ):
                try:
                    self._read_context()
                except Exception as error:
                    return self._failure(
                        context,
                        DiagnosticCode.INVALID_MODEL,
                        f"Authoring context resynchronization failed: {error}",
                    )
                definitions = {item.name: item for item in self.definitions}
                available = set(definitions)
            if name not in available:
                return self._failure(
                    context,
                    DiagnosticCode.UNKNOWN_TOOL,
                    f"Authoring tool {name!r} is unavailable in stage {self._stage.value}.",
                )
            try:
                if name == _PREPARE_GEOMETRY_EDIT.name:
                    _validate_geometry_edit_arguments(
                        arguments,
                        definitions[name],
                    )
                if name == _READ_CONTEXT.name:
                    _require_exact_fields(arguments, set())
                    outcome = self._read_context()
                elif name == _SET_REQUIREMENTS.name:
                    outcome = self._set_requirements(arguments)
                else:
                    _require_exact_fields(arguments, set()) if name not in {
                        RESULT_QUERY_TOOL_NAME,
                        RESULT_CATALOG_TOOL_NAME,
                        RESULT_COMPARISON_TOOL_NAME,
                        ANALYSIS_RUN_CATALOG_TOOL_NAME,
                        EXPORT_CSV_TOOL_NAME,
                        RESULT_DISPLAY_CONTEXT_TOOL_NAME,
                        _RUN_PREFLIGHT.name,
                        _PREPARE_SOLVE.name,
                        _PREPARE_DELETE.name,
                        _CREATE_NATIVE_MODEL_DOCUMENT.name,
                        _PREPARE_GEOMETRY.name,
                        _PREPARE_PLANAR_CONSTRUCTION.name,
                        _GEOMETRY_FEATURE_CATALOG.name,
                        _READ_GEOMETRY_EDIT_CONTEXT.name,
                        _PREPARE_GEOMETRY_EDIT.name,
                        *_PROFILE_TRANSFORM_TOOLS,
                        _PREPARE_MESH.name,
                        _EDIT_MODEL_OBJECT.name,
                        _APPLY_DEFINITION.name,
                    } else None
                    self._active_tool_context = context
                    try:
                        outcome = self._handlers[name](dict(arguments), self)
                    finally:
                        self._active_tool_context = None
                    if type(outcome) is not AuthoringToolOutcome:
                        raise TypeError(
                            "authoring handler must return AuthoringToolOutcome"
                        )
                    if name == _READ_GEOMETRY_EDIT_CONTEXT.name:
                        self._record_geometry_edit_schema_scope(outcome)
                    self._advance_after_success(name, outcome)
                safe_data = provider_safe_authoring_payload(outcome.data)
                return ToolResult(
                    ok=outcome.ok,
                    session_id=context.session_id,
                    input_revision=max(context.expected_revision, 0),
                    idempotency_key=context.idempotency_key,
                    summary=outcome.summary,
                    data=safe_data,
                    diagnostics=(
                        ()
                        if outcome.ok
                        else (
                            make_diagnostic(
                                DiagnosticCode.INVALID_INPUT,
                                outcome.summary,
                                source="agent.authoring_runtime",
                            ),
                        )
                    ),
                )
            except _AuthoringArgumentSchemaError as error:
                allowed_values = list(error.allowed_values)
                summary = f"Invalid geometry edit argument at {error.path}: {error.reason}."
                return ToolResult(
                    ok=False,
                    session_id=context.session_id,
                    input_revision=max(context.expected_revision, 0),
                    idempotency_key=context.idempotency_key,
                    summary=summary,
                    data={
                        "error": {
                            "code": "geometry-edit.schema-invalid",
                            "path": error.path,
                            "reason": error.reason,
                            "allowed_values": allowed_values,
                            "remediation": {
                                "action": "revise_prepare_geometry_edit_arguments",
                                "read_context_again": False,
                            },
                        }
                    },
                    diagnostics=(
                        make_diagnostic(
                            DiagnosticCode.INVALID_TOOL_ARGUMENTS,
                            summary,
                            source="agent.authoring_runtime",
                        ),
                    ),
                )
            except Exception as error:
                error_type = (
                    re.sub(
                        r"[^A-Za-z0-9_]",
                        "",
                        type(error).__name__,
                    )[:64]
                    or "LocalAuthoringError"
                )
                # Curated contract/validation errors carry provider-safe,
                # actionable detail; surface them instead of the masked
                # placeholder so failures can be diagnosed in one turn.
                # Unexpected exception types stay masked, and curated detail
                # that carries local-only content (filesystem paths or
                # credential markers) falls back to the masked form too.
                detail = ""
                if isinstance(
                    error,
                    (AuthoringContractError, ValueError, TypeError, KeyError),
                ):
                    detail = _provider_safe_error_detail(str(error))
                rejection_message = (
                    f"{error_type}: {detail}"
                    if detail
                    else f"{error_type}: the local authoring request was rejected."
                )
                return ToolResult(
                    ok=False,
                    session_id=context.session_id,
                    input_revision=max(context.expected_revision, 0),
                    idempotency_key=context.idempotency_key,
                    summary="The local authoring tool rejected the request.",
                    diagnostics=(
                        make_diagnostic(
                            DiagnosticCode.INVALID_TOOL_ARGUMENTS,
                            rejection_message,
                            source="agent.authoring_runtime",
                        ),
                    ),
                )

    def observe_binding(
        self,
        context: AuthoringContext,
        *,
        proposal_staled: bool = False,
        saved_state_transition: bool = False,
        project_save_terminal_transition: bool = False,
    ) -> bool:
        """Track accepted GUI identity and reject cross-binding stage reuse."""

        if type(context) is not AuthoringContext:
            raise TypeError("context must be AuthoringContext")
        current = (
            context.binding.document_id,
            context.binding.session_id,
            context.binding.session_revision,
        )
        with self._lock:
            previous_context = self._observed_context
            self._observed_context = context
            prior = self._binding_identity
            if prior is None:
                self._binding_identity = current
                if self._stage is not AuthoringWorkflowStage.STALE:
                    self._stage = _restored_stage_for_context(context)
                self._seed_default_geometry_requirements(context)
                self._refresh_turn_snapshot_locked()
                return True
            if prior == current:
                self._seed_default_geometry_requirements(context)
                if self._pending_operation is None:
                    was_active_job = (
                        previous_context is not None
                        and _job_status_is_active(previous_context.job_status)
                    )
                    is_active_job = _job_status_is_active(context.job_status)
                    if is_active_job or (
                        self._stage is AuthoringWorkflowStage.SOLVE_PENDING
                        and was_active_job
                    ):
                        self._stage = _restored_stage_for_context(context)
                self._refresh_turn_snapshot_locked()
                return True
            same_session = prior[:2] == current[:2]
            revision_increased = current[2] > prior[2]
            self._geometry_edit_schema_scope = None
            first_native_project_transition = (
                self._pending_operation == "geometry"
                and prior[2] == 0
                and current[2] == 1
            )
            expected_local_transition = (
                not proposal_staled
                and revision_increased
                and (
                    (
                        same_session
                        and (
                            self._active_tool_context is not None
                            or self._pending_operation
                            in {
                                "geometry",
                                "mesh",
                                "preflight",
                                "solve",
                                "destructive_edit",
                            }
                        )
                    )
                    or first_native_project_transition
                    or (same_session and saved_state_transition)
                    or (
                        same_session
                        and project_save_terminal_transition
                        and self._pending_operation == "project_save"
                    )
                )
            )
            self._binding_identity = current
            if expected_local_transition:
                self._refresh_turn_snapshot_locked()
                return True
            operation = self._pending_operation or "binding"
            if self._pending_operation == "project_save":
                self._mark_project_save_terminal(
                    ProposalState.STALE,
                    (
                        "a pending project save was staled by the binding change"
                        if proposal_staled
                        else "document, session, or revision changed"
                    ),
                )
            self._record_terminal(
                operation,
                "stale",
                (
                    "a pending proposal was staled by the binding change"
                    if proposal_staled
                    else (
                        "document, session, or revision changed outside "
                        "the active workflow"
                    )
                ),
            )
            self._pending_operation = None
            self._clear_destructive_pending()
            self._ledger = RequirementLedger()
            self._stage = AuthoringWorkflowStage.STALE
            self._refresh_turn_snapshot_locked()
            return False

    def record_proposal_state(
        self,
        operation: str,
        state: ProposalState | str,
        message: str = "",
    ) -> None:
        """Consume a state returned by the existing GUI-controlled bridge."""

        normalized_operation = str(operation)
        normalized_state = ProposalState(state)
        with self._lock:
            if self._pending_operation != normalized_operation:
                raise ValueError("proposal state does not match the pending operation")
            if normalized_state in {
                ProposalState.PENDING_CONFIRMATION,
                ProposalState.ACCEPTED,
                ProposalState.RUNNING,
            }:
                return
            if normalized_operation == "geometry":
                self._stage = (
                    AuthoringWorkflowStage.MESH_READY
                    if normalized_state is ProposalState.SUCCEEDED
                    else (
                        self._geometry_resume_stage
                        or AuthoringWorkflowStage.GEOMETRY_READY
                    )
                )
                self._geometry_resume_stage = None
            elif normalized_operation == "mesh":
                self._stage = (
                    AuthoringWorkflowStage.DEFINITIONS_READY
                    if normalized_state is ProposalState.SUCCEEDED
                    else (self._mesh_resume_stage or AuthoringWorkflowStage.MESH_READY)
                )
                self._mesh_resume_stage = None
            elif normalized_operation == "solve":
                self._stage = (
                    AuthoringWorkflowStage.RESULTS_READY
                    if normalized_state is ProposalState.SUCCEEDED
                    else AuthoringWorkflowStage.SOLVE_READY
                )
            elif normalized_operation == "destructive_edit":
                resume_stage = self._destructive_resume_stage
                object_type = self._pending_destructive_object_type
                if resume_stage is None or object_type is None:
                    raise ValueError("destructive edit has no pending target")
                if normalized_state is ProposalState.SUCCEEDED:
                    if (
                        object_type in {"part", "feature", "generated_mesh"}
                        and self._observed_context is not None
                    ):
                        self._stage = _restored_stage_for_context(
                            self._observed_context
                        )
                        self._seed_default_geometry_requirements(self._observed_context)
                    else:
                        self._stage = {
                            "part": AuthoringWorkflowStage.GEOMETRY_READY,
                            "feature": AuthoringWorkflowStage.GEOMETRY_READY,
                            "generated_mesh": AuthoringWorkflowStage.MESH_READY,
                            "named_region": AuthoringWorkflowStage.DEFINITIONS_READY,
                            "analysis_step": (
                                AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                            ),
                            "boundary_condition": (
                                AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                            ),
                            "load": (AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY),
                            "result_request": (
                                AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                            ),
                        }[object_type]
                else:
                    self._stage = resume_stage
                self._clear_destructive_pending()
            else:
                raise ValueError("unknown pending proposal operation")
            if (
                normalized_operation == "geometry"
                and self._pending_planar_proposal_id is not None
            ):
                pending_id = self._pending_planar_proposal_id
                pending = next(
                    (
                        item
                        for item in reversed(self._planar_audit)
                        if item.proposal_id == pending_id
                    ),
                    None,
                )
                if pending is not None:
                    self._append_planar_audit(
                        replace(pending, terminal_state=normalized_state.value)
                    )
                self._pending_planar_proposal_id = None
            self._pending_operation = None
            self._record_terminal(
                normalized_operation,
                normalized_state.value,
                message,
            )
            self._refresh_turn_snapshot_locked()

    def invalidate_binding(self, reason: str) -> None:
        normalized = str(reason).strip()
        if not normalized:
            raise ValueError("stale reason must be non-blank")
        with self._lock:
            operation = self._pending_operation or "binding"
            if self._pending_operation == "project_save":
                self._mark_project_save_terminal(
                    ProposalState.STALE,
                    normalized,
                )
            self._record_terminal(operation, ProposalState.STALE.value, normalized)
            self._pending_operation = None
            self._clear_destructive_pending()
            self._ledger = RequirementLedger()
            # A document/session transition invalidates the last typed GUI
            # context.  Keep the workflow terminal record, but do not let
            # provider-facing projections or dynamic definitions reuse that
            # context until the owner thread observes a fresh one.
            self._observed_context = None
            self._binding_identity = None
            self._stage = AuthoringWorkflowStage.STALE
            self.invalidate_turn_snapshot()

    def record_preflight_state(
        self,
        state: str,
        message: str = "",
    ) -> None:
        normalized = str(state).strip().casefold()
        if normalized not in {
            "passed",
            "blocked",
            "failed",
            "cancelled",
            "stale",
        }:
            raise ValueError("unknown preflight terminal state")
        with self._lock:
            if (
                self._stage is not AuthoringWorkflowStage.PREFLIGHT_PENDING
                or self._pending_operation != "preflight"
            ):
                raise ValueError("there is no pending preflight")
            self._stage = (
                AuthoringWorkflowStage.SOLVE_READY
                if normalized == "passed"
                else AuthoringWorkflowStage.PREFLIGHT_READY
            )
            self._pending_operation = None
            self._record_terminal("preflight", normalized, message)
            self._refresh_turn_snapshot_locked()

    def cancel_turn(self, reason: str = "provider turn cancelled") -> None:
        normalized = str(reason).strip()
        self._cancel_requested.set()
        with self._lock:
            # The workflow lock is only acquired after any in-flight dispatch
            # released it, so the in-flight handler already observed the flag.
            self._cancel_requested.clear()
            self._record_terminal("provider_turn", "cancelled", normalized)
            if self._stage not in {
                AuthoringWorkflowStage.GEOMETRY_PENDING,
                AuthoringWorkflowStage.MESH_PENDING,
                AuthoringWorkflowStage.SOLVE_PENDING,
                AuthoringWorkflowStage.PROJECT_SAVE_PENDING,
                AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING,
            }:
                self._stage = AuthoringWorkflowStage.CANCELLED
            self._refresh_turn_snapshot_locked()

    def reset_for_binding(self) -> None:
        with self._lock:
            self._ledger = RequirementLedger()
            self._pending_operation = None
            self._project_save_record = None
            self._geometry_resume_stage = None
            self._mesh_resume_stage = None
            self._planar_retry_state = None
            self._geometry_edit_schema_scope = None
            self._pending_planar_proposal_id = None
            self._clear_destructive_pending()
            self._binding_identity = None
            self._observed_context = None
            self._stage = AuthoringWorkflowStage.REQUIREMENTS
            self._refresh_turn_snapshot_locked()

    def register_project_save_proposal(
        self,
        proposal_id: str,
        context: AuthoringContext,
    ) -> ProjectSaveProposalRecord:
        """Register one path-free save card during a model tool dispatch."""

        record = self.preview_project_save_proposal(proposal_id, context)
        with self._lock:
            self._project_save_record = record
            return record

    def preview_project_save_proposal(
        self,
        proposal_id: str,
        context: AuthoringContext,
    ) -> ProjectSaveProposalRecord:
        """Build the path-free save card record without registering it."""

        normalized_id = _nonblank_string(proposal_id, "proposal_id")
        if type(context) is not AuthoringContext:
            raise TypeError("context must be AuthoringContext")
        with self._lock:
            if not project_save_available(
                self._stage, frozenset(self._handlers), context
            ):
                raise ValueError("current native project is not ready to save")
            active = self._project_save_record
            if active is not None and active.state in {
                ProposalState.PENDING_CONFIRMATION,
                ProposalState.RUNNING,
            }:
                raise ValueError("a project save proposal is already active")
            binding = context.binding
            identity = {
                "proposal_id": normalized_id,
                "target_document_id": binding.document_id,
                "target_session_id": binding.session_id,
                "base_session_revision": binding.session_revision,
                "operation": "project_save",
            }
            proposal_hash = hashlib.sha256(
                json.dumps(
                    identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            return ProjectSaveProposalRecord(
                normalized_id,
                proposal_hash,
                binding.document_id,
                binding.session_id,
                binding.session_revision,
                self._stage,
            )

    def can_accept_project_save_from_gui(
        self,
        proposal_id: str,
        proposal_hash: str,
        context: AuthoringContext,
    ) -> bool:
        if type(context) is not AuthoringContext:
            return False
        with self._lock:
            record = self._project_save_record
            binding = context.binding
            return bool(
                self._stage is AuthoringWorkflowStage.PROJECT_SAVE_PENDING
                and self._pending_operation == "project_save"
                and record is not None
                and record.state is ProposalState.PENDING_CONFIRMATION
                and record.proposal_id == proposal_id
                and record.proposal_hash == proposal_hash
                and binding.supported
                and binding.source_kind == "native"
                and binding.document_id == record.target_document_id
                and binding.session_id == record.target_session_id
                and binding.session_revision == record.base_session_revision
                and _capability_enabled(context, "request_project_save")
            )

    def begin_project_save_from_gui(
        self,
        proposal_id: str,
        proposal_hash: str,
        context: AuthoringContext,
    ) -> None:
        """Consume the sole GUI acceptance edge for a save proposal."""

        with self._lock:
            if not self.can_accept_project_save_from_gui(
                proposal_id,
                proposal_hash,
                context,
            ):
                raise ValueError("project save proposal is stale or already consumed")
            assert self._project_save_record is not None
            self._project_save_record = replace(
                self._project_save_record,
                state=ProposalState.RUNNING,
            )

    def record_project_save_state(
        self,
        proposal_id: str,
        proposal_hash: str,
        state: ProposalState | str,
        message: str = "",
    ) -> None:
        """Record exactly one local save terminal returned by MainWindow."""

        normalized_state = ProposalState(state)
        if normalized_state not in {
            ProposalState.SUCCEEDED,
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
            ProposalState.REJECTED,
        }:
            raise ValueError("project save state must be terminal")
        with self._lock:
            record = self._project_save_record
            if (
                record is None
                or record.proposal_id != proposal_id
                or record.proposal_hash != proposal_hash
                or record.state
                not in {
                    ProposalState.PENDING_CONFIRMATION,
                    ProposalState.RUNNING,
                }
            ):
                raise ValueError("project save state does not match an active proposal")
            self._mark_project_save_terminal(normalized_state, message)
            self._pending_operation = None
            self._stage = record.resume_stage
            self._record_terminal(
                "project_save",
                normalized_state.value,
                message,
            )
            self._refresh_turn_snapshot_locked()

    def collected_requirements(
        self,
        group: str,
    ) -> dict[str, object]:
        """Return a complete explicit geometry or mesh requirement group."""

        keys = self._requirement_keys(group)
        values = {
            item.key: item.value
            for item in self._ledger.entries
            if item.status is RequirementStatus.PROPOSED
        }
        missing = [key for key in keys if key not in values]
        if missing:
            raise ValueError("clarification_required: " + ", ".join(missing))
        return {key: values[key] for key in keys}

    def defaulted_requirement_keys(
        self,
        group: str,
    ) -> tuple[str, ...]:
        """Return active requirement keys supplied by local defaults."""

        keys = self._requirement_keys(group)
        entries = {
            item.key: item
            for item in self._ledger.entries
            if item.status is RequirementStatus.PROPOSED
        }
        return tuple(
            key
            for key in keys
            if key in entries
            and entries[key].source_turn_id == _DEFAULT_REQUIREMENT_SOURCE_TURN_ID
        )

    def _current_requirement_group(self) -> str | None:
        return current_requirement_group(self._stage)

    def _requirement_keys(self, group: str) -> tuple[str, ...]:
        return requirement_keys(group, self._observed_context)

    def _requirement_spec(self, key: str) -> Mapping[str, object]:
        return requirement_spec(key, self._observed_context)

    def _seed_default_geometry_requirements(
        self,
        context: AuthoringContext,
    ) -> None:
        if context.binding.source_kind not in {"blank", "native"} or context.parts:
            return
        recorded = {
            item.key
            for item in self._ledger.entries
            if item.status is RequirementStatus.PROPOSED
        }
        accepted_units = (
            {}
            if context.unit_context is None
            else {
                "length_unit": context.unit_context.length,
                "force_unit": context.unit_context.force,
                "stress_unit": context.unit_context.stress,
            }
        )
        values = accepted_units or _DEFAULT_GEOMETRY_REQUIREMENTS
        source_turn_id = (
            "accepted-unit-context"
            if accepted_units
            else _DEFAULT_REQUIREMENT_SOURCE_TURN_ID
        )
        for key, value in values.items():
            if value is None:
                continue
            if key in recorded:
                continue
            self._ledger.record(
                key,
                field_type=str(_REQUIREMENT_SPECS[key]["type"]),
                stage=_requirement_stage(key),
                value=value,
                source_turn_id=source_turn_id,
                status=RequirementStatus.PROPOSED,
            )

    def invocation_metadata(self, prefix: str) -> dict[str, object]:
        """Return bounded envelope identities for the active local handler."""

        normalized = re.sub(r"[^A-Za-z0-9_-]", "", str(prefix))[:24]
        if not normalized:
            raise ValueError("invocation prefix must contain an identifier")
        with self._lock:
            context = self._active_tool_context
            if context is None:
                raise RuntimeError(
                    "invocation metadata is available only during dispatch"
                )
            suffix = context.idempotency_key[:24]
            return {
                "agent_session_id": context.session_id,
                "turn_id": f"turn-{suffix}",
                "source_tool_call_ids": (f"call-{suffix}",),
                "draft_revision": self._ledger.revision,
                "identity_suffix": f"{normalized}-{suffix}",
            }

    def _read_context(self) -> AuthoringToolOutcome:
        raw = self._context_reader()
        if self._stage in {
            AuthoringWorkflowStage.STALE,
            AuthoringWorkflowStage.CANCELLED,
        }:
            self.reset_for_binding()
        if type(raw) is AuthoringContext:
            self.observe_binding(raw)
        data = (
            raw.to_provider_dict()
            if type(raw) is AuthoringContext
            else dict(raw)
            if isinstance(raw, Mapping)
            else None
        )
        if data is None:
            raise TypeError("context_reader must return AuthoringContext or an object")
        published_authoring_tools = tuple(item.name for item in self.definitions)
        published_set = set(published_authoring_tools)
        raw_capabilities = data.get("capabilities")
        if isinstance(raw_capabilities, list):
            data["capabilities"] = [
                dict(item)
                for item in raw_capabilities
                if isinstance(item, Mapping) and item.get("operation") in published_set
            ]
        data["published_authoring_tool_names"] = list(published_authoring_tools)
        if _PREPARE_PLANAR_CONSTRUCTION.name in published_set:
            data["planar_construction_ir"] = _planar_construction_context_capability()
        requirement_group = self._current_requirement_group()
        required = (
            ()
            if requirement_group is None
            else self._requirement_keys(requirement_group)
        )
        recorded = {
            item.key
            for item in self._ledger.entries
            if item.status is RequirementStatus.PROPOSED
        }
        return AuthoringToolOutcome(
            "Bounded authoring context read locally.",
            {
                "workflow_stage": self._stage.value,
                "requirement_stage": requirement_group,
                "context": data,
                "missing_requirements": [
                    key for key in required if key not in recorded
                ],
                "defaulted_requirements": list(
                    self.defaulted_requirement_keys(requirement_group)
                    if requirement_group is not None
                    else ()
                ),
            },
        )

    def _set_requirements(
        self,
        arguments: Mapping[str, Any],
    ) -> AuthoringToolOutcome:
        data = _require_exact_fields(
            arguments,
            {"turn_id", "requirements"},
        )
        turn_id = _nonblank_string(data["turn_id"], "turn_id")
        raw_requirements = data["requirements"]
        if not isinstance(raw_requirements, Mapping) or not raw_requirements:
            raise ValueError("requirements must be a non-empty object")
        requirement_group = self._current_requirement_group()
        if requirement_group is None:
            raise ValueError("there is no active requirement stage")
        allowed = set(self._requirement_keys(requirement_group))
        unknown = set(raw_requirements) - set(_REQUIREMENT_SPECS)
        if unknown:
            raise ValueError(
                f"unknown requirement fields: {', '.join(sorted(unknown))}"
            )
        out_of_stage = set(raw_requirements) - allowed
        if out_of_stage:
            raise ValueError(
                "out-of-stage requirement fields: " + ", ".join(sorted(out_of_stage))
            )
        for key, value in raw_requirements.items():
            _validate_requirement_value(
                key,
                value,
                self._requirement_spec(key),
            )
        for key, value in raw_requirements.items():
            self._ledger.record(
                key,
                field_type=str(self._requirement_spec(key)["type"]),
                stage=_requirement_stage(key),
                value=value,
                source_turn_id=turn_id,
                status=RequirementStatus.PROPOSED,
            )
        recorded = {
            item.key
            for item in self._ledger.entries
            if item.status is RequirementStatus.PROPOSED
        }
        missing = [
            key
            for key in self._requirement_keys(requirement_group)
            if key not in recorded
        ]
        data = {
            "ledger_revision": self._ledger.revision,
            "requirement_stage": requirement_group,
            "recorded": sorted(raw_requirements),
            "missing_requirements": missing,
            "operation_confirmation_required": not missing,
        }
        summary = "Explicit current-operation requirements recorded."
        return AuthoringToolOutcome(summary, data)

    def _advance_after_success(
        self,
        name: str,
        outcome: AuthoringToolOutcome,
    ) -> None:
        if not outcome.ok:
            return
        if name == _PREPARE_GEOMETRY.name:
            self._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
            self._pending_operation = "geometry"
        elif name == _PREPARE_PLANAR_CONSTRUCTION.name:
            self._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
            self._pending_operation = "geometry"
        elif name == _PREPARE_GEOMETRY_EDIT.name:
            self._geometry_resume_stage = self._stage
            self._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
            self._pending_operation = "geometry"
        elif name in {
            _PREPARE_PROFILE_EXTRUSION.name,
            _PREPARE_PROFILE_REVOLUTION.name,
            _PREPARE_PROFILE_PATH_SWEEP.name,
        }:
            self._geometry_resume_stage = self._stage
            self._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
            self._pending_operation = "geometry"
        elif name == _PREPARE_MESH.name:
            self._mesh_resume_stage = self._stage
            self._stage = AuthoringWorkflowStage.MESH_PENDING
            self._pending_operation = "mesh"
        elif name == _APPLY_DEFINITION.name:
            object_type = outcome.data.get("definition_object_type")
            if object_type not in {"named_region", "analysis_step"}:
                raise ValueError(
                    "definition handler registered no exact resume object type"
                )
            if "proposal_id" in outcome.data:
                self._destructive_resume_stage = self._stage
                self._pending_destructive_object_type = object_type
                self._stage = AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
                self._pending_operation = "destructive_edit"
            else:
                self._stage = (
                    _restored_stage_for_context(self._observed_context)
                    if self._observed_context is not None
                    else (
                        AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                        if object_type == "analysis_step"
                        else AuthoringWorkflowStage.DEFINITIONS_READY
                    )
                )
                self._record_terminal(
                    "model_definition",
                    "succeeded",
                    outcome.summary,
                )
        elif name == _EDIT_MODEL_OBJECT.name:
            object_type = outcome.data.get("edit_object_type")
            if object_type not in {
                "named_region",
                "material",
                "section",
                "section_assignment",
                "analysis_step",
                "boundary_condition",
                "load",
                "result_request",
            }:
                raise ValueError("edit handler registered no exact edited object type")
            if "proposal_id" in outcome.data:
                self._destructive_resume_stage = self._stage
                self._pending_destructive_object_type = object_type
                self._stage = AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
                self._pending_operation = "destructive_edit"
            else:
                self._stage = (
                    _restored_stage_for_context(self._observed_context)
                    if self._observed_context is not None
                    else AuthoringWorkflowStage.DEFINITIONS_READY
                )
                self._record_terminal(
                    "model_edit",
                    "succeeded",
                    outcome.summary,
                )
        elif name == _RUN_PREFLIGHT.name:
            if outcome.data.get("passed") is True:
                self._stage = AuthoringWorkflowStage.SOLVE_READY
                self._record_terminal("preflight", "passed", outcome.summary)
            elif outcome.data.get("state") == "running":
                self._stage = AuthoringWorkflowStage.PREFLIGHT_PENDING
                self._pending_operation = "preflight"
            else:
                raise ValueError("preflight did not pass or start")
        elif name == _PREPARE_SOLVE.name:
            self._stage = AuthoringWorkflowStage.SOLVE_PENDING
            self._pending_operation = "solve"
        elif name == _REQUEST_PROJECT_SAVE.name:
            if self._project_save_record is None:
                raise ValueError("project save handler registered no proposal")
            self._stage = AuthoringWorkflowStage.PROJECT_SAVE_PENDING
            self._pending_operation = "project_save"
        elif name == _PREPARE_DELETE.name:
            object_type = outcome.data.get("delete_object_type")
            if type(object_type) is not str or object_type not in {
                "part",
                "feature",
                "generated_mesh",
                "named_region",
                "analysis_step",
                "boundary_condition",
                "load",
                "result_request",
            }:
                raise ValueError("destructive handler registered no exact target type")
            self._destructive_resume_stage = self._stage
            self._pending_destructive_object_type = object_type
            self._stage = AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
            self._pending_operation = "destructive_edit"
        self._refresh_turn_snapshot_locked()

    def _clear_destructive_pending(self) -> None:
        self._destructive_resume_stage = None
        self._pending_destructive_object_type = None

    def _mark_project_save_terminal(
        self,
        state: ProposalState,
        message: str,
    ) -> None:
        record = self._project_save_record
        if record is not None:
            self._project_save_record = replace(
                record,
                state=state,
                message=str(message)[:512],
            )

    def _record_terminal(
        self,
        operation: str,
        state: str,
        message: str,
    ) -> None:
        self._terminals.append(
            AuthoringTerminalRecord(
                str(operation),
                str(state),
                str(message)[:512],
            )
        )
        self._terminals = self._terminals[-64:]

    @staticmethod
    def _failure(
        context: ToolExecutionContext,
        code: DiagnosticCode,
        message: str,
    ) -> ToolResult:
        return ToolResult(
            ok=False,
            session_id=context.session_id,
            input_revision=max(context.expected_revision, 0),
            idempotency_key=context.idempotency_key,
            summary=message,
            diagnostics=(
                make_diagnostic(
                    code,
                    message,
                    source="agent.authoring_runtime",
                ),
            ),
        )


def _restored_stage_for_context(
    context: AuthoringContext,
) -> AuthoringWorkflowStage:
    """Resume one accepted native project from its authoritative local state."""

    binding = context.binding
    if not binding.supported:
        return AuthoringWorkflowStage.STALE
    if binding.source_kind != "native" or not context.parts:
        return AuthoringWorkflowStage.REQUIREMENTS
    if _job_status_is_active(context.job_status):
        return AuthoringWorkflowStage.SOLVE_PENDING
    if context.result_available:
        return AuthoringWorkflowStage.RESULTS_READY
    if (
        context.mesh.current
        and context.validation_status == "passed"
        and context.definitions.analysis_step_count > 0
    ):
        return AuthoringWorkflowStage.SOLVE_READY
    if context.mesh.current and context.definitions.analysis_step_count > 0:
        return AuthoringWorkflowStage.PREFLIGHT_READY
    if context.mesh.current:
        return AuthoringWorkflowStage.DEFINITIONS_READY
    return AuthoringWorkflowStage.MESH_READY


def _job_status_is_active(value: str) -> bool:
    return value.casefold() in {
        "running",
        "queued",
        "pending",
        "cancelling",
        "canceling",
    }


__all__ = [
    "AUTHORING_TURN_SNAPSHOT_MAX_BYTES",
    "AUTHORING_TURN_SNAPSHOT_SCHEMA_VERSION",
    "AuthoringTerminalRecord",
    "AuthoringTurnSnapshot",
    "AuthoringToolHandler",
    "AuthoringToolOutcome",
    "AuthoringWorkflowController",
    "AuthoringWorkflowStage",
    "ProjectSaveProposalRecord",
    "provider_safe_authoring_payload",
]
