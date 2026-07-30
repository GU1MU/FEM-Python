"""A8 dynamic authoring workflow and provider-safe tool boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import json
import math
import re
import threading
from typing import TYPE_CHECKING, Any
import uuid

from .authoring import (
    AuthoringContext,
    ProposalState,
    RequirementLedger,
    RequirementReview,
    RequirementReviewStatus,
    RequirementStatus,
)
from .diagnostics import DiagnosticCode, make_diagnostic
from .providers.base import ToolDefinition
from .result_authoring import (
    RESULT_QUERY_TOOL_NAME,
    result_catalog_tool_schema,
    result_query_tool_schema,
)
from .schemas import ToolResult

if TYPE_CHECKING:
    from .tools.registry import ToolExecutionContext


class AuthoringWorkflowStage(str, Enum):
    REQUIREMENTS = "requirements"
    REVIEW_PENDING = "review_pending"
    GEOMETRY_READY = "geometry_ready"
    GEOMETRY_PENDING = "geometry_pending"
    MESH_READY = "mesh_ready"
    MESH_PENDING = "mesh_pending"
    DEFINITIONS_READY = "definitions_ready"
    ANALYSIS_DEFINITIONS_READY = "analysis_definitions_ready"
    PREFLIGHT_READY = "preflight_ready"
    PREFLIGHT_PENDING = "preflight_pending"
    SOLVE_READY = "solve_ready"
    SOLVE_PENDING = "solve_pending"
    RESULTS_READY = "results_ready"
    STALE = "stale"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AuthoringToolOutcome:
    summary: str
    data: Mapping[str, object]
    ok: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("authoring tool summary must be non-blank")
        if not isinstance(self.data, Mapping):
            raise TypeError("authoring tool data must be an object")
        if not isinstance(self.ok, bool):
            raise TypeError("authoring tool ok must be boolean")
        object.__setattr__(
            self,
            "summary",
            _provider_safe_summary(self.summary),
        )
        object.__setattr__(self, "data", dict(self.data))


@dataclass(frozen=True, slots=True)
class AuthoringTerminalRecord:
    operation: str
    state: str
    message: str


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
        "confirm_requirement_review",
        "reject_proposal",
        "cancel_operation",
    }
)
_NO_ARGUMENTS = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}
_ABSOLUTE_PATH = re.compile(
    r"""(?ix)
    (?:
        [a-z]:[\\/]
        |
        \\\\[^\\/\s]+[\\/][^\\/\s]+
        |
        (?<![a-z0-9])/(?:[^/\s]+/)+[^/\s]*
    )
    """
)
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "nodes",
        "node_ids",
        "node_coordinates",
        "coordinates",
        "elements",
        "element_ids",
        "connectivity",
        "records",
        "result_arrays",
        "result_provider",
        "model_session",
        "model_patch",
        "qt_object",
        "vtk",
        "gmsh_model",
        "raw_patch",
        "patch_before",
        "patch_after",
        "source_path",
        "absolute_path",
        "api_key",
        "credential",
    }
)
_MAX_PROVIDER_PAYLOAD_BYTES = 32_768
_MAX_PROVIDER_COLLECTION_ITEMS = 128
_MAX_PROVIDER_DEPTH = 12


_REQUIREMENT_SPECS: dict[str, dict[str, object]] = {
    "modeling_assumption": {
        "type": "string",
        "enum": ["plane_stress", "plane_strain"],
    },
    "length_unit": {"type": "string"},
    "force_unit": {"type": "string"},
    "stress_unit": {"type": "string"},
    "plate_width": {"type": "number", "exclusiveMinimum": 0},
    "plate_height": {"type": "number", "exclusiveMinimum": 0},
    "plate_thickness": {"type": "number", "exclusiveMinimum": 0},
    "hole_radius": {"type": "number", "exclusiveMinimum": 0},
    "hole_center_x": {"type": "number"},
    "hole_center_y": {"type": "number"},
    "young_modulus": {"type": "number", "exclusiveMinimum": 0},
    "poisson_ratio": {
        "type": "number",
        "exclusiveMinimum": -1,
        "exclusiveMaximum": 0.5,
    },
    "mesh_cell_shape": {
        "type": "string",
        "enum": ["triangle", "quadrilateral"],
    },
    "mesh_order": {"type": "integer", "enum": [1, 2]},
    "mesh_global_size": {"type": "number", "exclusiveMinimum": 0},
    "hole_mesh_size": {"type": "number", "exclusiveMinimum": 0},
    "fixed_dofs": {
        "type": "array",
        "items": {"type": "integer", "minimum": 1, "maximum": 2},
        "minItems": 1,
        "maxItems": 2,
        "uniqueItems": True,
    },
    "load_type": {
        "type": "string",
        "enum": ["edge_traction", "edge_pressure"],
    },
    "load_direction": {
        "type": "string",
        "enum": ["x", "y", "inward_normal", "outward_normal"],
    },
    "load_magnitude": {"type": "number"},
    "load_unit": {"type": "string"},
    "load_distribution": {"type": "string", "enum": ["uniform"]},
    "analysis_procedure": {"type": "string", "enum": ["static"]},
    "nlgeom": {"type": "boolean", "enum": [False]},
    "result_requests": {
        "type": "array",
        "items": {"type": "string", "enum": ["U", "S", "RF"]},
        "minItems": 1,
        "maxItems": 3,
        "uniqueItems": True,
    },
}
_REQUIRED_REQUIREMENTS = tuple(_REQUIREMENT_SPECS)


def _tool(
    name: str,
    description: str,
    parameters: Mapping[str, object],
) -> ToolDefinition:
    return ToolDefinition(name, description, parameters)


def _result_tool_definition(schema: Mapping[str, object]) -> ToolDefinition:
    return ToolDefinition(
        str(schema["name"]),
        str(schema["description"]),
        schema["input_schema"],  # type: ignore[arg-type]
    )


_READ_CONTEXT = _tool(
    "read_authoring_context",
    "Read the bounded current native authoring context and workflow stage.",
    _NO_ARGUMENTS,
)
_SET_REQUIREMENTS = _tool(
    "set_authoring_requirements",
    (
        "Record only engineering values explicitly supplied by the user. "
        "Values remain proposed until a GUI RequirementReview is confirmed."
    ),
    {
        "type": "object",
        "properties": {
            "turn_id": {"type": "string"},
            "requirements": {
                "type": "object",
                "properties": _REQUIREMENT_SPECS,
                "additionalProperties": False,
                "minProperties": 1,
            },
        },
        "required": ["turn_id", "requirements"],
        "additionalProperties": False,
    },
)
_REQUEST_REVIEW = _tool(
    "request_requirement_review",
    (
        "Create a local RequirementReview after every required engineering "
        "value has been explicitly supplied. This tool cannot confirm it."
    ),
    _NO_ARGUMENTS,
)
_PREPARE_GEOMETRY = _tool(
    "prepare_geometry_proposal",
    (
        "Build and present a revision-bound geometry proposal. The geometry "
        "is not added until the GUI control is clicked."
    ),
    _NO_ARGUMENTS,
)
_PREPARE_MESH = _tool(
    "prepare_mesh_proposal",
    (
        "Build and present a mesh proposal. Gmsh is not called until the GUI "
        "control is clicked."
    ),
    _NO_ARGUMENTS,
)
_APPLY_SCOPES = _tool(
    "apply_scopes_and_materials",
    (
        "Apply confirmed additive A4 scopes, material, section and assignment "
        "through one reversible local patch."
    ),
    _NO_ARGUMENTS,
)
_APPLY_ANALYSIS = _tool(
    "apply_analysis_definitions",
    (
        "Apply the confirmed additive A5 step, boundary, load and result "
        "requests through one reversible local patch."
    ),
    _NO_ARGUMENTS,
)
_RUN_PREFLIGHT = _tool(
    "run_native_preflight",
    "Run the existing deterministic native preflight without solving.",
    _NO_ARGUMENTS,
)
_PREPARE_SOLVE = _tool(
    "prepare_solve_proposal",
    (
        "Build and present a revision/stamp-bound solve proposal. The solver "
        "is not started until the GUI control is clicked."
    ),
    _NO_ARGUMENTS,
)
_RESULT_CATALOG = _result_tool_definition(result_catalog_tool_schema())
_RESULT_QUERY = _result_tool_definition(result_query_tool_schema())


_STAGE_TOOLS: dict[AuthoringWorkflowStage, tuple[ToolDefinition, ...]] = {
    AuthoringWorkflowStage.REQUIREMENTS: (
        _READ_CONTEXT,
        _SET_REQUIREMENTS,
        _REQUEST_REVIEW,
    ),
    AuthoringWorkflowStage.REVIEW_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.GEOMETRY_READY: (
        _READ_CONTEXT,
        _PREPARE_GEOMETRY,
    ),
    AuthoringWorkflowStage.GEOMETRY_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.MESH_READY: (
        _READ_CONTEXT,
        _PREPARE_MESH,
    ),
    AuthoringWorkflowStage.MESH_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.DEFINITIONS_READY: (
        _READ_CONTEXT,
        _APPLY_SCOPES,
    ),
    AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY: (
        _READ_CONTEXT,
        _APPLY_ANALYSIS,
    ),
    AuthoringWorkflowStage.PREFLIGHT_READY: (
        _READ_CONTEXT,
        _RUN_PREFLIGHT,
    ),
    AuthoringWorkflowStage.PREFLIGHT_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.SOLVE_READY: (
        _READ_CONTEXT,
        _PREPARE_SOLVE,
    ),
    AuthoringWorkflowStage.SOLVE_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.RESULTS_READY: (
        _READ_CONTEXT,
        _RESULT_CATALOG,
        _RESULT_QUERY,
    ),
    AuthoringWorkflowStage.STALE: (_READ_CONTEXT,),
    AuthoringWorkflowStage.CANCELLED: (_READ_CONTEXT,),
}


class AuthoringWorkflowController:
    """Strict A8 state machine over injected A1-A7 local handlers."""

    def __init__(
        self,
        context_reader: ContextReader,
        handlers: Mapping[str, AuthoringToolHandler],
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
        self._ledger = RequirementLedger()
        self._stage = AuthoringWorkflowStage.REQUIREMENTS
        self._pending_review: RequirementReview | None = None
        self._review_binding: tuple[str, str, int] | None = None
        self._pending_operation: str | None = None
        self._terminals: list[AuthoringTerminalRecord] = []
        self._active_tool_context: ToolExecutionContext | None = None
        self._binding_identity: tuple[str, str, int] | None = None
        self._lock = threading.RLock()

    @property
    def stage(self) -> AuthoringWorkflowStage:
        with self._lock:
            return self._stage

    @property
    def ledger(self) -> RequirementLedger:
        return self._ledger

    @property
    def pending_review(self) -> RequirementReview | None:
        with self._lock:
            return self._pending_review

    @property
    def terminal_records(self) -> tuple[AuthoringTerminalRecord, ...]:
        with self._lock:
            return tuple(self._terminals)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        with self._lock:
            definitions = _STAGE_TOOLS[self._stage]
            return tuple(
                item
                for item in definitions
                if (
                    item.name
                    in {
                        _READ_CONTEXT.name,
                        _SET_REQUIREMENTS.name,
                        _REQUEST_REVIEW.name,
                    }
                    or item.name in self._handlers
                )
            )

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        with self._lock:
            available = {item.name for item in self.definitions}
            if name not in available:
                return self._failure(
                    context,
                    DiagnosticCode.UNKNOWN_TOOL,
                    f"Authoring tool {name!r} is unavailable in stage {self._stage.value}.",
                )
            try:
                if name == _READ_CONTEXT.name:
                    _require_exact_fields(arguments, set())
                    outcome = self._read_context()
                elif name == _SET_REQUIREMENTS.name:
                    outcome = self._set_requirements(arguments)
                elif name == _REQUEST_REVIEW.name:
                    _require_exact_fields(arguments, set())
                    outcome = self._request_review()
                else:
                    _require_exact_fields(arguments, set()) if name not in {
                        RESULT_QUERY_TOOL_NAME
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
            except Exception as error:
                error_type = re.sub(
                    r"[^A-Za-z0-9_]",
                    "",
                    type(error).__name__,
                )[:64] or "LocalAuthoringError"
                return ToolResult(
                    ok=False,
                    session_id=context.session_id,
                    input_revision=max(context.expected_revision, 0),
                    idempotency_key=context.idempotency_key,
                    summary="The local authoring tool rejected the request.",
                    diagnostics=(
                        make_diagnostic(
                            DiagnosticCode.INVALID_TOOL_ARGUMENTS,
                            (
                                f"{error_type}: the local authoring request "
                                "was rejected."
                            ),
                            source="agent.authoring_runtime",
                        ),
                    ),
                )

    def resolve_requirement_review(
        self,
        review: RequirementReview,
    ) -> None:
        """Advance only after the existing GUI bridge returned a review state."""

        if type(review) is not RequirementReview:
            raise TypeError("review must be RequirementReview")
        with self._lock:
            pending = self._pending_review
            if (
                pending is None
                or pending.review_id != review.review_id
                or pending.review_hash != review.review_hash
            ):
                raise ValueError("RequirementReview does not match the pending review")
            if review.status is RequirementReviewStatus.CONFIRMED:
                self._stage = AuthoringWorkflowStage.GEOMETRY_READY
            elif review.status in {
                RequirementReviewStatus.REJECTED,
                RequirementReviewStatus.STALE,
            }:
                self._stage = AuthoringWorkflowStage.REQUIREMENTS
            else:
                raise ValueError("RequirementReview is not terminal")
            self._pending_review = None
            self._review_binding = None
            self._record_terminal("requirement_review", review.status.value, "")

    def stale_review_for_binding(
        self,
        context: AuthoringContext,
    ) -> bool:
        if type(context) is not AuthoringContext:
            raise TypeError("context must be AuthoringContext")
        current = (
            context.binding.document_id,
            context.binding.session_id,
            context.binding.session_revision,
        )
        with self._lock:
            if (
                self._stage is not AuthoringWorkflowStage.REVIEW_PENDING
                or self._review_binding is None
                or self._review_binding == current
            ):
                return False
            self._record_terminal(
                "requirement_review",
                "stale",
                "document, session, or revision changed",
            )
            self._pending_review = None
            self._review_binding = None
            self._binding_identity = current
            self._ledger = RequirementLedger()
            self._stage = AuthoringWorkflowStage.STALE
            return True

    def observe_binding(
        self,
        context: AuthoringContext,
        *,
        proposal_staled: bool = False,
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
            prior = self._binding_identity
            if prior is None:
                self._binding_identity = current
                return True
            if prior == current:
                return True
            same_session = prior[:2] == current[:2]
            revision_increased = current[2] > prior[2]
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
                            in {"geometry", "mesh", "preflight", "solve"}
                        )
                    )
                    or first_native_project_transition
                )
            )
            self._binding_identity = current
            if expected_local_transition:
                return True
            operation = self._pending_operation or "binding"
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
            self._pending_review = None
            self._review_binding = None
            self._ledger = RequirementLedger()
            self._stage = AuthoringWorkflowStage.STALE
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
                    else AuthoringWorkflowStage.GEOMETRY_READY
                )
            elif normalized_operation == "mesh":
                self._stage = (
                    AuthoringWorkflowStage.DEFINITIONS_READY
                    if normalized_state is ProposalState.SUCCEEDED
                    else AuthoringWorkflowStage.MESH_READY
                )
            elif normalized_operation == "solve":
                self._stage = (
                    AuthoringWorkflowStage.RESULTS_READY
                    if normalized_state is ProposalState.SUCCEEDED
                    else AuthoringWorkflowStage.SOLVE_READY
                )
            else:
                raise ValueError("unknown pending proposal operation")
            self._pending_operation = None
            self._record_terminal(
                normalized_operation,
                normalized_state.value,
                message,
            )

    def invalidate_binding(self, reason: str) -> None:
        normalized = str(reason).strip()
        if not normalized:
            raise ValueError("stale reason must be non-blank")
        with self._lock:
            operation = self._pending_operation or "binding"
            self._record_terminal(operation, ProposalState.STALE.value, normalized)
            self._pending_operation = None
            self._pending_review = None
            self._review_binding = None
            self._ledger = RequirementLedger()
            self._stage = AuthoringWorkflowStage.STALE

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

    def cancel_turn(self, reason: str = "provider turn cancelled") -> None:
        normalized = str(reason).strip()
        with self._lock:
            self._record_terminal("provider_turn", "cancelled", normalized)
            if self._stage not in {
                AuthoringWorkflowStage.GEOMETRY_PENDING,
                AuthoringWorkflowStage.MESH_PENDING,
                AuthoringWorkflowStage.SOLVE_PENDING,
            }:
                self._stage = AuthoringWorkflowStage.CANCELLED

    def reset_for_binding(self) -> None:
        with self._lock:
            self._ledger = RequirementLedger()
            self._pending_review = None
            self._review_binding = None
            self._pending_operation = None
            self._binding_identity = None
            self._stage = AuthoringWorkflowStage.REQUIREMENTS

    def confirmed_requirements(self) -> dict[str, object]:
        return {
            item.key: item.value
            for item in self._ledger.require_confirmed(
                "authoring",
                _REQUIRED_REQUIREMENTS,
            )
        }

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
            raise TypeError(
                "context_reader must return AuthoringContext or an object"
            )
        return AuthoringToolOutcome(
            "Bounded authoring context read locally.",
            {
                "workflow_stage": self._stage.value,
                "context": data,
                "missing_requirements": [
                    key
                    for key in _REQUIRED_REQUIREMENTS
                    if key
                    not in {
                        item.key
                        for item in self._ledger.entries
                        if item.status
                        in {
                            RequirementStatus.PROPOSED,
                            RequirementStatus.CONFIRMED,
                        }
                    }
                ],
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
        unknown = set(raw_requirements) - set(_REQUIREMENT_SPECS)
        if unknown:
            raise ValueError(
                f"unknown requirement fields: {', '.join(sorted(unknown))}"
            )
        for key, value in raw_requirements.items():
            _validate_requirement_value(
                key,
                value,
                _REQUIREMENT_SPECS[key],
            )
        for key, value in raw_requirements.items():
            self._ledger.record(
                key,
                field_type=str(_REQUIREMENT_SPECS[key]["type"]),
                stage=_requirement_stage(key),
                value=value,
                source_turn_id=turn_id,
                status=RequirementStatus.PROPOSED,
            )
        missing = [
            key
            for key in _REQUIRED_REQUIREMENTS
            if key not in {item.key for item in self._ledger.entries}
        ]
        return AuthoringToolOutcome(
            "Explicit authoring requirements recorded as proposed.",
            {
                "ledger_revision": self._ledger.revision,
                "recorded": sorted(raw_requirements),
                "missing_requirements": missing,
                "review_required": True,
            },
        )

    def _request_review(self) -> AuthoringToolOutcome:
        missing = [
            key
            for key in _REQUIRED_REQUIREMENTS
            if key not in {item.key for item in self._ledger.entries}
        ]
        if missing:
            raise ValueError(
                "clarification_required: " + ", ".join(missing)
            )
        values = {item.key: item.value for item in self._ledger.entries}
        _validate_supported_requirement_combination(values)
        review = self._ledger.create_review(
            f"review-{uuid.uuid4().hex}",
            _REQUIRED_REQUIREMENTS,
        )
        raw_context = self._context_reader()
        context_data = (
            raw_context.to_provider_dict()
            if type(raw_context) is AuthoringContext
            else dict(raw_context)
            if isinstance(raw_context, Mapping)
            else None
        )
        if context_data is None or not isinstance(
            context_data.get("binding"),
            Mapping,
        ):
            raise TypeError("context_reader returned no binding")
        binding = dict(context_data["binding"])
        self._pending_review = review
        self._review_binding = (
            str(binding["document_id"]),
            str(binding["session_id"]),
            int(binding["session_revision"]),
        )
        self._stage = AuthoringWorkflowStage.REVIEW_PENDING
        return AuthoringToolOutcome(
            "RequirementReview is waiting for the local GUI control.",
            {
                "review_id": review.review_id,
                "review_hash": review.review_hash,
                "ledger_revision": review.ledger_revision,
                "status": review.status.value,
                "fields": [item.to_dict() for item in review.fields],
                "proposal_view": {
                    "proposal_id": review.review_id,
                    "proposal_hash": review.review_hash,
                    "proposal_kind": "requirement_review",
                    "title": "确认完整工程需求",
                    "summary": (
                        f"请审阅 {len(review.fields)} 项几何、材料、"
                        "网格、分析和结果参数"
                    ),
                    "impact": "确认后这些工程值才可用于建模工具",
                    "confirm_label": "确认需求",
                    "target_document_id": binding["document_id"],
                    "target_session_id": binding["session_id"],
                    "base_session_revision": binding["session_revision"],
                },
            },
        )

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
        elif name == _PREPARE_MESH.name:
            self._stage = AuthoringWorkflowStage.MESH_PENDING
            self._pending_operation = "mesh"
        elif name == _APPLY_SCOPES.name:
            self._stage = AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
            self._record_terminal("scopes_and_materials", "succeeded", outcome.summary)
        elif name == _APPLY_ANALYSIS.name:
            self._stage = AuthoringWorkflowStage.PREFLIGHT_READY
            self._record_terminal("analysis_definitions", "succeeded", outcome.summary)
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


def provider_safe_authoring_payload(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate and copy one bounded JSON object before Provider exposure."""

    if not isinstance(value, Mapping):
        raise TypeError("authoring payload must be an object")
    normalized = _safe_json_value(dict(value), depth=0)
    assert isinstance(normalized, dict)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_PROVIDER_PAYLOAD_BYTES:
        raise ValueError("authoring payload exceeds the provider-safe byte budget")
    return normalized


def _provider_safe_summary(value: str) -> str:
    normalized = value.strip()
    encoded = normalized.encode("utf-8")
    if len(encoded) > 2048:
        raise ValueError("authoring summary exceeds the provider-safe byte budget")
    if _ABSOLUTE_PATH.search(normalized):
        raise ValueError("absolute paths are local-only")
    lowered = normalized.casefold()
    if any(
        marker in lowered
        for marker in ("api_key", "apikey", "password", "secret", "credential")
    ):
        raise ValueError("credential material is local-only")
    return normalized


def _safe_json_value(value: object, *, depth: int) -> object:
    if depth > _MAX_PROVIDER_DEPTH:
        raise ValueError("authoring payload exceeds the nesting budget")
    if isinstance(value, Mapping):
        if len(value) > _MAX_PROVIDER_COLLECTION_ITEMS:
            raise ValueError("authoring payload object exceeds the item budget")
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("authoring payload keys must be strings")
            if key.casefold() in _FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"authoring payload field {key!r} is local-only")
            result[key] = _safe_json_value(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_PROVIDER_COLLECTION_ITEMS:
            raise ValueError("authoring payload array exceeds the item budget")
        return [_safe_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError("authoring payload text exceeds the item budget")
        if _ABSOLUTE_PATH.search(value):
            raise ValueError("absolute paths are local-only")
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("authoring payload must contain finite JSON values only")


def _require_exact_fields(
    value: Mapping[str, Any],
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("tool arguments must be an object")
    data = dict(value)
    if set(data) != required:
        raise ValueError("tool argument fields do not match the strict schema")
    return data


def _nonblank_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-blank string")
    return value.strip()


def _validate_requirement_value(
    key: str,
    value: object,
    spec: Mapping[str, object],
) -> None:
    expected = spec["type"]
    if expected == "string":
        normalized = _nonblank_string(value, key)
        if "enum" in spec and normalized not in spec["enum"]:
            raise ValueError(f"{key} is outside the supported values")
    elif expected == "number":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TypeError(f"{key} must be a finite number")
        numeric = float(value)
        if "exclusiveMinimum" in spec and numeric <= float(spec["exclusiveMinimum"]):
            raise ValueError(f"{key} is below its strict lower bound")
        if "exclusiveMaximum" in spec and numeric >= float(spec["exclusiveMaximum"]):
            raise ValueError(f"{key} is above its strict upper bound")
    elif expected == "integer":
        if isinstance(value, bool) or type(value) is not int:
            raise TypeError(f"{key} must be an integer")
        if "minimum" in spec and value < int(spec["minimum"]):
            raise ValueError(f"{key} is below its lower bound")
        if "maximum" in spec and value > int(spec["maximum"]):
            raise ValueError(f"{key} is above its upper bound")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{key} is outside the supported values")
    elif expected == "boolean":
        if type(value) is not bool:
            raise TypeError(f"{key} must be a boolean")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{key} is outside the supported values")
    elif expected == "array":
        if not isinstance(value, list):
            raise TypeError(f"{key} must be an array")
        if not int(spec["minItems"]) <= len(value) <= int(spec["maxItems"]):
            raise ValueError(f"{key} has an unsupported number of values")
        if len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            raise ValueError(f"{key} must contain unique values")
        item_spec = spec["items"]
        assert isinstance(item_spec, Mapping)
        for item in value:
            _validate_requirement_value(key, item, item_spec)
    else:
        raise RuntimeError("unsupported requirement schema")


def _validate_supported_requirement_combination(
    values: Mapping[str, object],
) -> None:
    fixed_dofs = tuple(values["fixed_dofs"])
    if fixed_dofs != tuple(range(min(fixed_dofs), max(fixed_dofs) + 1)):
        raise ValueError(
            "clarification_required: fixed_dofs must be one contiguous 2D range"
        )
    load_type = str(values["load_type"])
    direction = str(values["load_direction"])
    magnitude = float(values["load_magnitude"])
    if load_type == "edge_traction":
        if direction not in {"x", "y"}:
            raise ValueError(
                "clarification_required: edge traction direction must be x or y"
            )
        return
    if direction not in {"inward_normal", "outward_normal"}:
        raise ValueError(
            "clarification_required: edge pressure requires a normal direction"
        )
    if (
        direction == "inward_normal"
        and magnitude <= 0.0
    ) or (
        direction == "outward_normal"
        and magnitude >= 0.0
    ):
        raise ValueError(
            "clarification_required: pressure sign must match its normal direction"
        )


def _requirement_stage(key: str) -> str:
    if key.startswith(("plate_", "hole_", "modeling_", "length_")):
        return "geometry"
    if key.startswith("mesh_"):
        return "mesh"
    if key in {"young_modulus", "poisson_ratio", "stress_unit"}:
        return "definitions"
    if key.startswith(("fixed_", "load_", "analysis_", "nlgeom")):
        return "analysis"
    return "results"


__all__ = [
    "AuthoringTerminalRecord",
    "AuthoringToolHandler",
    "AuthoringToolOutcome",
    "AuthoringWorkflowController",
    "AuthoringWorkflowStage",
    "provider_safe_authoring_payload",
]
