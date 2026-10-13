"""A1 GUI boundary for detached FEM Agent authoring.

The adapter reads a detached session snapshot into bounded DTOs.  The bridge
owns only those DTOs and an ``AuthoringPort``; it never stores or mutates a
``ModelSession``.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

from fem.application import MeshTaskSnapshot, ModelSession, UnitContext
from fem.application.feature_history import (
    derive_feature_history,
    remove_terminal_feature,
)

from fem_agent.authoring import (
    AgentProposal,
    AuthoringAuthorizationError,
    AuthoringContext,
    AuthoringContractError,
    CapabilitySummary,
    DefinitionSummary,
    LocalModelBinding,
    MeshSummary,
    ModelPatch,
    OperationKind,
    PartSummary,
    ProposalKind,
    ProposalPortRecord,
    ProposalState,
    UnitContextSummary,
)
from fem_agent.boolean_authoring import BODY_BOOLEAN_TOOL_HANDLING, PART_BOOLEAN_TOOL_HANDLING
from fem_agent.authoring_payload import provider_safe_authoring_payload
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.authoring_values import AuthoringWorkflowStage
from fem_agent.definition_authoring import (
    inverse_operations_for_snapshot,
    scoped_definition_batch_from_operations,
)
from fem_agent.definition_action_authoring import require_strict_definition_batch
from fem_agent.deletion_authoring import apply_delete_operation, deletable_object_catalog
from fem_agent.editing_authoring import apply_edit_operation, editable_object_catalog
from fem_agent.geometry_authoring import _as_strict_planar_sketch
from fem_agent.geometry_payload import geometry_recipe_from_payload
from fem_agent.mesh_authoring import MeshIntent
from fem_agent.incremental_authoring import (
    require_incremental_definition_batch,
)
from fem_agent.naming import NameAllocator
from fem_agent.solve_authoring import solve_operation_identity, validation_stamp_for_snapshot
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.workspace_catalog import WorkspaceCatalogBridge
from fem_agent.export_authoring import (
    EXPORT_CSV_TOOL_NAME,
    RESULT_DISPLAY_CONTEXT_TOOL_NAME,
    AgentExportBridge,
)
from .agent_export import SessionExportPort
from .agent_result_query import SessionResultQueryPort
from fem.geometry import (
    BooleanGeometry,
    ExtrudedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    PathSweptGeometry,
    RevolvedGeometry,
    WireGeometry,
    geometry_dimension,
    resolve_extrusion_source_faces,
)

from .agent_authoring_values import AgentPreflightState, AgentProposalPreview
from .agent_handlers.binding import AuthoringHandlerBinding
from .agent_handlers.geometry_creation import GeometryCreationHandlers
from .agent_handlers.geometry_editing import GeometryEditingHandlers
from .agent_handlers.geometry_support import _provider_recipe_kind
from .agent_handlers.mesh_solve import MeshSolveHandlers
from .agent_handlers.model_definitions import ModelDefinitionHandlers
from .agent_handlers.profile_transforms import ProfileTransformHandlers
from .agent_handlers.workspace_results import WorkspaceResultHandlers


@dataclass(frozen=True, slots=True)
class _PendingMeshTask:
    session: ModelSession
    snapshot: MeshTaskSnapshot


class _SessionSnapshot(Protocol):
    session_id: str
    session_revision: int
    model_revision: int
    source_kind: str | None
    can_save: bool
    model_name: str | None
    active_part_id: str | None
    parts: object
    named_regions: object
    materials: object
    sections: object
    assignments: object
    steps: object
    artifact: object | None
    validations: object
    runs: object
    selected_run_id: str | None
    displayed_result_run_id: str | None
    displayed_result: object | None
    mesh_current: bool
    unit_context: object | None


def _canonical_profile_source_recipe(recipe: object) -> object:
    """Project current planar primitives onto the sketch graph for replay checks."""

    try:
        return _as_strict_planar_sketch(recipe)
    except (TypeError, ValueError):
        return recipe


def _bounded_count(value: object) -> int:
    try:
        return max(0, int(len(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _recipe_dimension(recipe: object | None) -> int | None:
    if recipe is None:
        return None
    dimension = getattr(recipe, "dimension", None)
    if dimension in {1, 2, 3}:
        return int(dimension)
    if isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        return int(geometry_dimension(recipe))
    name = type(recipe).__name__.casefold()
    if any(
        token in name for token in ("rectangle", "disk", "plate", "sketch", "planar")
    ):
        return 2
    if any(token in name for token in ("box", "cylinder", "extruded")):
        return 3
    if "wire" in name or "line" in name:
        return 1
    return None


def _model_counts(artifact: object | None) -> tuple[int | None, int | None]:
    model = None if artifact is None else getattr(artifact, "model", None)
    if model is None:
        return None, None
    mesh = getattr(model, "mesh", None)
    count_source = model if mesh is None else mesh
    nodes = getattr(count_source, "nodes", None)
    elements = getattr(count_source, "elements", None)
    return _bounded_count(nodes), _bounded_count(elements)


def authoring_context_from_snapshot(
    snapshot: _SessionSnapshot,
    *,
    document_id: str | int | None = None,
    workspace_selected: bool = False,
) -> AuthoringContext:
    """Copy a session projection into a bounded, provider-safe DTO."""

    session_id = str(snapshot.session_id)
    source_kind = "blank" if snapshot.source_kind is None else str(snapshot.source_kind)
    supported = source_kind in {"blank", "native"}
    binding = LocalModelBinding(
        document_id=(
            f"document:{session_id}" if document_id is None else str(document_id)
        ),
        session_id=session_id,
        session_revision=int(snapshot.session_revision),
        source_kind=source_kind,
        supported=supported,
    )

    parts: list[PartSummary] = []
    if source_kind == "native":
        for part in tuple(snapshot.parts)[:128]:  # type: ignore[arg-type]
            recipe = getattr(part, "geometry_recipe", None)
            parts.append(
                PartSummary(
                    part_id=str(getattr(part, "id")),
                    name=str(getattr(part, "name")),
                    recipe_kind=_provider_recipe_kind(recipe),
                    dimension=_recipe_dimension(recipe),
                    suppressed=bool(getattr(part, "suppressed", False)),
                )
            )

    node_count, element_count = _model_counts(snapshot.artifact)
    mesh_present = bool(
        snapshot.artifact is not None
        and element_count is not None
        and element_count > 0
    )
    mesh_current = bool(mesh_present and getattr(snapshot, "mesh_current", False))
    validation_status = "not_run"
    validations = getattr(snapshot, "validations", {})
    if _bounded_count(validations):
        records = tuple(validations.values())  # type: ignore[union-attr]
        validation_status = (
            "passed"
            if records and any(bool(getattr(item, "passed", False)) for item in records)
            else "blocked"
        )
    runs = tuple(getattr(snapshot, "runs", ()))
    result_count = sum(
        1
        for item in runs
        if getattr(item, "result_id", None) is not None
        and str(getattr(item, "status", "")).casefold().endswith("succeeded")
    )
    job_status = "idle"
    if any(
        str(getattr(item, "status", "")).casefold().endswith("running") for item in runs
    ):
        job_status = "running"
    elif runs:
        job_status = (
            str(getattr(runs[-1], "status", "completed")).split(".")[-1].casefold()
        )

    blocked_reason = None if supported else "V1 only binds blank or native documents"
    deletable_objects_available = bool(deletable_object_catalog(snapshot))
    editable_objects_available = bool(editable_object_catalog(snapshot))
    editable_geometry_available = bool(
        supported
        and source_kind == "native"
        and any(
            getattr(part, "geometry_recipe", None) is not None
            and not bool(getattr(part, "suppressed", False))
            for part in tuple(snapshot.parts)[:128]  # type: ignore[arg-type]
        )
    )
    capabilities = (
        CapabilitySummary("read_authoring_context", supported, blocked_reason),
        CapabilitySummary(
            "read_geometry_feature_catalog",
            editable_geometry_available,
            (
                None
                if editable_geometry_available
                else "No readable part geometry in the current native project"
            ),
        ),
        CapabilitySummary("review_requirements", supported, blocked_reason),
        CapabilitySummary("build_agent_draft", supported, blocked_reason),
        CapabilitySummary("present_static_proposal", supported, blocked_reason),
        CapabilitySummary("draft_native_geometry", supported, blocked_reason),
        CapabilitySummary("commit_native_geometry", supported, blocked_reason),
        CapabilitySummary(
            "edit_native_geometry",
            editable_geometry_available,
            (
                None
                if editable_geometry_available
                else "No editable part geometry in the current native project"
            ),
        ),
        CapabilitySummary(
            "draft_mesh_intent",
            supported and source_kind == "native",
            (
                None
                if supported and source_kind == "native"
                else "Mesh intent requires a native project"
            ),
        ),
        CapabilitySummary(
            "request_mesh_proposal",
            supported and source_kind == "native",
            (
                None
                if supported and source_kind == "native"
                else "Mesh proposals require a native project"
            ),
        ),
        CapabilitySummary(
            "run_model_preflight",
            (
                supported
                and source_kind == "native"
                and bool(getattr(snapshot, "mesh_current", False))
            ),
            (
                None
                if (
                    supported
                    and source_kind == "native"
                    and bool(getattr(snapshot, "mesh_current", False))
                )
                else "Model preflight requires a current native mesh"
            ),
        ),
        CapabilitySummary(
            "request_solve_proposal",
            (
                supported
                and source_kind == "native"
                and bool(getattr(snapshot, "mesh_current", False))
                and validation_status == "passed"
            ),
            (
                None
                if (
                    supported
                    and source_kind == "native"
                    and bool(getattr(snapshot, "mesh_current", False))
                    and validation_status == "passed"
                )
                else "Solve proposals require a current native mesh and passed preflight"
            ),
        ),
        CapabilitySummary(
            "request_project_save",
            bool(getattr(snapshot, "can_save", False)),
            (
                None
                if bool(getattr(snapshot, "can_save", False))
                else "Saving requires an open native project"
            ),
        ),
        CapabilitySummary(
            "delete_model_objects",
            deletable_objects_available,
            (None if deletable_objects_available else "No deletable objects in the current native project"),
        ),
        CapabilitySummary(
            "edit_model_objects",
            editable_objects_available,
            (
                None
                if editable_objects_available
                else "No editable definition objects in the current native project"
            ),
        ),
        CapabilitySummary(
            "query_accepted_result",
            (supported and source_kind == "native" and result_count > 0),
            (
                None
                if (supported and source_kind == "native" and result_count > 0)
                else "Result queries require current accepted native results"
            ),
        ),
        CapabilitySummary(
            "compare_accepted_results",
            supported and source_kind == "native" and result_count >= 2,
            (
                None
                if supported and source_kind == "native" and result_count >= 2
                else "Result comparison requires at least two accepted native results in the current session"
            ),
        ),
        CapabilitySummary(
            "read_result_display_context",
            supported and source_kind == "native" and result_count > 0,
            (
                None
                if supported and source_kind == "native" and result_count > 0
                else "Result display context requires current accepted native results"
            ),
        ),
        CapabilitySummary(
            "export_accepted_result_csv",
            (
                supported
                and source_kind == "native"
                and result_count > 0
                and bool(workspace_selected)
            ),
            (
                None
                if (
                    supported
                    and source_kind == "native"
                    and result_count > 0
                    and bool(workspace_selected)
                )
                else (
                    "CSV result export requires a user workspace selected through /workspace "
                    "and current accepted native results"
                )
            ),
        ),
    )
    return AuthoringContext(
        binding=binding,
        model_name=(
            str(snapshot.model_name) if snapshot.model_name is not None else None
        ),
        active_part_id=(
            str(snapshot.active_part_id)
            if snapshot.active_part_id is not None
            else None
        ),
        parts=tuple(parts),
        mesh=MeshSummary(
            present=mesh_present,
            current=mesh_current,
            node_count=node_count,
            element_count=element_count,
        ),
        definitions=DefinitionSummary(
            named_region_count=_bounded_count(snapshot.named_regions),
            material_count=_bounded_count(snapshot.materials),
            section_count=_bounded_count(snapshot.sections),
            assignment_count=_bounded_count(snapshot.assignments),
            analysis_step_count=_bounded_count(snapshot.steps),
        ),
        validation_status=validation_status,
        job_status=job_status,
        result_available=snapshot.displayed_result is not None,
        run_count=len(runs),
        result_count=result_count,
        selected_run_id=getattr(snapshot, "selected_run_id", None),
        displayed_result_run_id=getattr(
            snapshot,
            "displayed_result_run_id",
            None,
        ),
        capabilities=capabilities,
        unit_context=(
            None
            if getattr(snapshot, "unit_context", None) is None
            else UnitContextSummary(
                length=str(snapshot.unit_context.length),
                force=str(snapshot.unit_context.force),
                stress=str(snapshot.unit_context.stress),
                density=(
                    None
                    if snapshot.unit_context.density is None
                    else str(snapshot.unit_context.density)
                ),
                acceleration=(
                    None
                    if snapshot.unit_context.acceleration is None
                    else str(snapshot.unit_context.acceleration)
                ),
                convention=(
                    None
                    if snapshot.unit_context.convention is None
                    else str(snapshot.unit_context.convention)
                ),
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class BridgeReceipt:
    proposal_id: str
    state: ProposalState
    message: str = ""
    replayed: bool = False


class AppliedPatchState(str, Enum):
    APPLIED = "applied"
    UNDONE = "undone"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class AgentPreflightRecord:
    request_id: str
    step_name: str
    base_session_revision: int
    state: AgentPreflightState
    message: str = ""
    validation_stamp: SolveValidationStamp | None = None


@dataclass(frozen=True, slots=True)
class AppliedPatchRecord:
    """One local automatic patch and its exact one-shot inverse."""

    patch: ModelPatch
    inverse_patch: ModelPatch
    session_revision: int
    delta: SessionDelta
    state: AppliedPatchState = AppliedPatchState.APPLIED
    message: str = ""
    replayed: bool = False

    @property
    def undo_available(self) -> bool:
        return self.state is AppliedPatchState.APPLIED

    @property
    def display_summary(self) -> object:
        return self.patch.display_summary


@dataclass(frozen=True, slots=True)
class FeatureProposalUndoRecord:
    """One accepted geometry proposal that appended one native feature."""

    proposal_id: str
    proposal_hash: str
    target_document_id: str
    target_session_id: str
    part_id: str
    feature_name: str
    before_recipe: object
    after_recipe: object
    session_revision: int
    state: AppliedPatchState = AppliedPatchState.APPLIED
    message: str = ""

    @property
    def undo_available(self) -> bool:
        return self.state is AppliedPatchState.APPLIED


@dataclass(frozen=True, slots=True)
class _GuiControlAuthorization:
    proposal_id: str
    action: str
    nonce: int


@dataclass(frozen=True, slots=True)
class AgentMeshTaskRequest:
    proposal_id: str
    proposal_hash: str
    task: object


@dataclass(frozen=True, slots=True)
class AgentSolveTaskRequest:
    """Metadata-only request passed to the existing GUI job entry."""

    proposal_id: str
    proposal_hash: str
    step_name: str
    job_name: str
    base_session_revision: int
    artifact_id: str
    model_revision: int
    validation_stamp: SolveValidationStamp


@dataclass(frozen=True, slots=True)
class AgentGeometryMutation:
    """One validated synchronous geometry mutation for in-place or child commit."""

    operation_kind: OperationKind
    affected_part_ids: tuple[str, ...]
    apply: Callable[[ModelSession, int], None]

    def __post_init__(self) -> None:
        if type(self.operation_kind) is not OperationKind:
            raise TypeError("operation_kind must be OperationKind")
        if not self.affected_part_ids or any(
            type(item) is not str or not item.strip() for item in self.affected_part_ids
        ):
            raise ValueError("affected_part_ids must contain nonblank strings")
        if len(set(self.affected_part_ids)) != len(self.affected_part_ids):
            raise ValueError("affected_part_ids must be unique")
        if not callable(self.apply):
            raise TypeError("apply must be callable")


@dataclass(frozen=True, slots=True)
class AgentPreflightTaskRequest:
    """Exact metadata used to run an automatic local preflight."""

    request_id: str
    step_name: str
    base_session_revision: int
    session_id: str
    artifact_id: str
    model_revision: int


@dataclass(frozen=True, slots=True)
class _PendingPreflightTask:
    session: ModelSession
    request: AgentPreflightTaskRequest


def _ignore_projection_refresh() -> None:
    pass


def _snapshot_part_recipe(snapshot: object, part_id: str) -> object | None:
    for part in tuple(getattr(snapshot, "parts", ())):
        if str(getattr(part, "id", "")) == part_id:
            return getattr(part, "geometry_recipe", None)
    return None


class SessionGeometryAuthoringPort:
    """A2/A3 port for atomic geometry writes and detached mesh tasks."""

    def __init__(
        self,
        session: ModelSession,
        refresh_projection: Callable[[], None],
        start_mesh_task: Callable[[AgentMeshTaskRequest], bool] | None = None,
        apply_definition_delta: Callable[[SessionDelta], None] | None = None,
        start_solve_task: Callable[[AgentSolveTaskRequest], bool] | None = None,
        start_preflight_task: (
            Callable[[AgentPreflightTaskRequest], bool] | None
        ) = None,
        geometry_edit_mode: Callable[[], str] | None = None,
        commit_geometry_edit: (
            Callable[[AgentGeometryMutation, int], Mapping[str, object]] | None
        ) = None,
    ) -> None:
        if type(session) is not ModelSession:
            raise TypeError("session must be ModelSession")
        if not callable(refresh_projection):
            raise TypeError("refresh_projection must be callable")
        self._session = session
        self._refresh_callback = refresh_projection
        self._start_mesh_task = start_mesh_task
        self._start_solve_task = start_solve_task
        self._start_preflight_task = start_preflight_task
        if geometry_edit_mode is not None and not callable(geometry_edit_mode):
            raise TypeError("geometry_edit_mode must be callable or None")
        if commit_geometry_edit is not None and not callable(commit_geometry_edit):
            raise TypeError("commit_geometry_edit must be callable or None")
        self._geometry_edit_mode_callback = geometry_edit_mode
        self._commit_geometry_edit_callback = commit_geometry_edit
        self._latest_geometry_iteration_report: dict[str, object] | None = None
        self._geometry_iteration_report_binding: tuple[str, str] | None = None
        if apply_definition_delta is not None and not callable(apply_definition_delta):
            raise TypeError("apply_definition_delta must be callable or None")
        self._apply_definition_delta = apply_definition_delta
        self._context: AuthoringContext | None = None
        self._records: dict[str, ProposalPortRecord] = {}
        self._mesh_tasks: dict[str, _PendingMeshTask] = {}
        self._patch_records: dict[str, AppliedPatchRecord] = {}
        self._feature_proposal_records: dict[str, FeatureProposalUndoRecord] = {}
        self._preflight_records: dict[str, AgentPreflightRecord] = {}
        self._preflight_tasks: dict[str, _PendingPreflightTask] = {}
        self._preflight_counter = 0
        self._record_listener: Callable[[ProposalPortRecord], None] | None = None

    @property
    def session(self) -> ModelSession:
        """Return the currently bound model Session."""

        return self._session

    def bind_session(
        self,
        session: ModelSession,
        *,
        idle: Callable[[], bool] | None = None,
    ) -> None:
        """Rebind all authoring writes to one idle document Session."""

        if type(session) is not ModelSession:
            raise TypeError("session must be ModelSession")
        if idle is not None and not idle():
            raise RuntimeError("Agent runtime must be idle before rebinding")
        if session.session_id != self._session.session_id:
            self._clear_geometry_iteration_report()
        self._session = session

    rebind_session = bind_session

    def set_record_listener(
        self,
        callback: Callable[[ProposalPortRecord], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("record listener must be callable")
        self._record_listener = callback

    def detach_gui_callbacks(self) -> None:
        """Release callbacks owned by a closing GUI window."""

        self._refresh_callback = _ignore_projection_refresh
        self._start_mesh_task = None
        self._start_solve_task = None
        self._start_preflight_task = None
        self._apply_definition_delta = None
        self._record_listener = None
        self._geometry_edit_mode_callback = None
        self._commit_geometry_edit_callback = None

    def geometry_edit_mode(self) -> str:
        callback = self._geometry_edit_mode_callback
        mode = "in_place" if callback is None else str(callback())
        if mode not in {"in_place", "branch"}:
            raise AuthoringContractError("invalid geometry edit mode")
        return mode

    def latest_geometry_iteration_report(self) -> Mapping[str, object] | None:
        report = self._latest_geometry_iteration_report
        context = self._context
        binding = (
            None
            if context is None
            else (
                context.binding.document_id,
                context.binding.session_id,
            )
        )
        if report is None or binding != self._geometry_iteration_report_binding:
            return None
        return deepcopy(report)

    def _clear_geometry_iteration_report(self) -> None:
        self._latest_geometry_iteration_report = None
        self._geometry_iteration_report_binding = None

    def set_context(self, context: AuthoringContext) -> None:
        if type(context) is not AuthoringContext:
            raise AuthoringContractError("context must be AuthoringContext")
        previous = self._context
        previous_binding = (
            None
            if previous is None
            else (
                previous.binding.document_id,
                previous.binding.session_id,
            )
        )
        next_binding = (
            context.binding.document_id,
            context.binding.session_id,
        )
        if previous_binding is not None and previous_binding != next_binding:
            self._clear_geometry_iteration_report()
        self._context = context

    def clear_context(self) -> None:
        """Detach the port from any document that is no longer editable."""

        self._context = None
        self._clear_geometry_iteration_report()

    def present(self, proposal: AgentProposal) -> ProposalPortRecord:
        if proposal.proposal_kind not in {
            ProposalKind.GEOMETRY,
            ProposalKind.MESH,
            ProposalKind.SOLVE,
            ProposalKind.DESTRUCTIVE_EDIT,
        }:
            raise AuthoringContractError(
                "session authoring port does not accept this proposal kind"
            )
        geometry_valid = (
            proposal.proposal_kind is ProposalKind.GEOMETRY
            and len(proposal.operations) == 1
            and proposal.operations[0].kind
            in {
                OperationKind.CREATE_NATIVE_PROJECT,
                OperationKind.ADD_NATIVE_PART,
                OperationKind.REPLACE_PART_GEOMETRY,
                OperationKind.EXTRUDE_PART_PROFILES,
                OperationKind.REVOLVE_PART_PROFILE,
                OperationKind.SWEEP_PART_PROFILE,
                OperationKind.APPLY_PART_BOOLEAN,
                OperationKind.APPLY_BODY_BOOLEAN,
            }
        )
        mesh_valid = proposal.proposal_kind is ProposalKind.MESH and tuple(
            item.kind for item in proposal.operations
        ) == (
            OperationKind.SET_PART_MESH_INTENT,
            OperationKind.REQUEST_MESH,
        )
        destructive_valid = (
            proposal.proposal_kind is ProposalKind.DESTRUCTIVE_EDIT
            and (
                tuple(item.kind for item in proposal.operations)
                == (
                    OperationKind.UPSERT_NAMED_REGIONS,
                    OperationKind.UPSERT_MODEL_DEFINITIONS,
                )
                or (
                    len(proposal.operations) == 1
                    and proposal.operations[0].kind
                    in {
                        OperationKind.DELETE_MODEL_OBJECT,
                        OperationKind.EDIT_MODEL_OBJECT,
                    }
                )
            )
        )
        solve_valid = (
            proposal.proposal_kind is ProposalKind.SOLVE
            and len(proposal.operations) == 1
            and proposal.operations[0].kind is OperationKind.REQUEST_SOLVE
            and isinstance(proposal.preconditions, Mapping)
            and proposal.preconditions.get("authoring_phase") == "A6"
        )
        if solve_valid:
            solve_operation_identity(proposal.operations[0])
        if (
            not geometry_valid
            and not mesh_valid
            and not destructive_valid
            and not solve_valid
        ):
            raise AuthoringContractError(
                "proposal operations do not match its authoring kind"
            )
        if proposal.proposal_id in self._records:
            raise AuthoringContractError("proposal_id is already registered")
        record = ProposalPortRecord(
            proposal,
            ProposalState.PENDING_CONFIRMATION,
        )
        self._records[proposal.proposal_id] = record
        return record

    def accept(self, proposal_id: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        proposal = record.proposal
        current_id = self._session.session_id
        current_revision = self._session.session_revision
        bound_context = self._context
        current_document_id = (
            bound_context.binding.document_id
            if bound_context is not None
            else f"document:{current_id}"
        )
        if (
            proposal.target_session_id != current_id
            or proposal.target_document_id != current_document_id
            or proposal.base_session_revision != current_revision
        ):
            raise AuthoringContractError("authoring proposal target is stale")
        if proposal.proposal_kind is ProposalKind.MESH:
            return self._accept_mesh(record)
        if proposal.proposal_kind is ProposalKind.SOLVE:
            return self._accept_solve(record)
        if proposal.proposal_kind is ProposalKind.DESTRUCTIVE_EDIT:
            return self._accept_destructive_edit(record)
        operation = proposal.operations[0]
        parameters = operation.parameters
        if operation.kind is OperationKind.APPLY_PART_BOOLEAN:
            return self._accept_part_boolean(record)
        if operation.kind is OperationKind.APPLY_BODY_BOOLEAN:
            return self._accept_body_boolean(record)
        if operation.kind is OperationKind.EXTRUDE_PART_PROFILES:
            return self._accept_profile_extrusion(record)
        if operation.kind in {
            OperationKind.REVOLVE_PART_PROFILE,
            OperationKind.SWEEP_PART_PROFILE,
        }:
            return self._accept_derived_profile_feature(record)
        recipe = geometry_recipe_from_payload(parameters["recipe"])
        snapshot = self._session.snapshot()
        if operation.kind is OperationKind.REPLACE_PART_GEOMETRY:
            return self._accept_geometry_replacement(record, recipe)
        return self._accept_geometry_creation(record, recipe, snapshot)

    def _accept_destructive_edit(
        self,
        record: ProposalPortRecord,
    ) -> ProposalPortRecord:
        proposal = record.proposal
        if len(proposal.operations) == 1:
            operation = proposal.operations[0]
            if operation.kind is OperationKind.DELETE_MODEL_OBJECT:
                delta = apply_delete_operation(
                    self._session,
                    operation,
                    base_session_revision=proposal.base_session_revision,
                )
                if operation.parameters["object_type"] not in {
                    "part",
                    "feature",
                    "generated_mesh",
                }:
                    self._project_definition_delta(delta)
            elif operation.kind is OperationKind.EDIT_MODEL_OBJECT:
                delta = apply_edit_operation(
                    self._session,
                    operation,
                    base_session_revision=proposal.base_session_revision,
                )
                self._project_definition_delta(delta)
            else:
                raise AuthoringContractError(
                    "unsupported destructive edit operation"
                )
        else:
            batch = scoped_definition_batch_from_operations(
                proposal.operations,
                base_session_revision=proposal.base_session_revision,
            )
            authoring_mode = proposal.preconditions.get("authoring_mode")
            if authoring_mode == "strict_incremental":
                require_strict_definition_batch(
                    self._session.snapshot(),
                    batch,
                    proposal.preconditions.get("direct_action"),
                )
            elif authoring_mode == "direct_incremental":
                require_incremental_definition_batch(
                    self._session.snapshot(),
                    batch,
                    proposal.preconditions.get("direct_action"),
                )
            delta = self._session.apply_scoped_definition_batch(batch)
            self._project_definition_delta(delta)
        succeeded = replace(record, state=ProposalState.SUCCEEDED)
        self._records[proposal.proposal_id] = succeeded
        return succeeded

    def _accept_part_boolean(
        self,
        record: ProposalPortRecord,
    ) -> ProposalPortRecord:
        operation = record.proposal.operations[0]
        parameters = operation.parameters
        if parameters["tool_handling"] != PART_BOOLEAN_TOOL_HANDLING:
            raise AuthoringContractError(
                "Part Boolean tool handling does not match canonical policy"
            )
        recipe = geometry_recipe_from_payload(
            json.loads(str(parameters["recipe_json"]))
        )
        if not isinstance(recipe, BooleanGeometry):
            raise AuthoringContractError(
                "Part Boolean proposal recipe is not BooleanGeometry"
            )

        def commit_part_boolean(
            session: ModelSession,
            expected_revision: int,
        ) -> None:
            session.apply_part_boolean(
                str(parameters["target_part_id"]),
                str(parameters["tool_part_id"]),
                str(parameters["operation"]),
                str(parameters["result_name"]),
                result_recipe=recipe,
                expected_session_revision=expected_revision,
            )

        return self._accept_geometry_mutation(
            record,
            AgentGeometryMutation(
                operation.kind,
                (
                    str(parameters["target_part_id"]),
                    str(parameters["tool_part_id"]),
                    str(parameters["result_part_id"]),
                ),
                commit_part_boolean,
            ),
        )

    def _accept_body_boolean(
        self,
        record: ProposalPortRecord,
    ) -> ProposalPortRecord:
        operation = record.proposal.operations[0]
        parameters = operation.parameters
        if parameters["tool_handling"] != BODY_BOOLEAN_TOOL_HANDLING:
            raise AuthoringContractError(
                "Body Boolean tool handling does not match canonical policy"
            )
        geometry = geometry_recipe_from_payload(
            json.loads(str(parameters["recipe_json"]))
        )
        if type(geometry) is not MultiBodyGeometry:
            raise AuthoringContractError(
                "Body Boolean proposal recipe is not MultiBodyGeometry"
            )

        def commit_body_boolean(
            session: ModelSession,
            expected_revision: int,
        ) -> None:
            session.apply_body_boolean(
                str(parameters["part_id"]),
                str(parameters["target_body_id"]),
                str(parameters["tool_body_id"]),
                str(parameters["operation"]),
                str(parameters["result_name"]),
                result=geometry,
                expected_session_revision=expected_revision,
            )

        return self._accept_geometry_mutation(
            record,
            AgentGeometryMutation(
                operation.kind,
                (str(parameters["part_id"]),),
                commit_body_boolean,
            ),
        )

    def _accept_profile_extrusion(
        self,
        record: ProposalPortRecord,
    ) -> ProposalPortRecord:
        operation = record.proposal.operations[0]
        parameters = operation.parameters
        base_recipe = geometry_recipe_from_payload(parameters["base_recipe"])
        raw_source_ids = parameters["source_face_ids"]
        if not isinstance(raw_source_ids, list) or not raw_source_ids:
            raise AuthoringContractError(
                "Profile extrusion requires explicit source_face_ids"
            )
        source_face_ids = tuple(str(item) for item in raw_source_ids)
        selection = resolve_extrusion_source_faces(
            base_recipe,
            source_face_ids,
        )
        if selection.face_ids != source_face_ids:
            raise AuthoringContractError(
                "Profile extrusion sources are not canonical"
            )
        part_id = str(parameters["part_id"])
        self._require_current_profile_source(
            part_id,
            base_recipe,
            mismatch_message=(
                "Profile extrusion source Part no longer matches its proposal"
            ),
        )
        recipes = tuple(
            ExtrudedGeometry(
                base_recipe,
                parameters["height"],
                (face_id,),
            )
            for face_id in source_face_ids
        )

        def commit_profile_extrusion(
            session: ModelSession,
            expected_revision: int,
        ) -> None:
            session.replace_part_with_extruded_siblings(
                part_id,
                recipes,
                expected_session_revision=expected_revision,
            )

        return self._accept_geometry_mutation(
            record,
            AgentGeometryMutation(
                operation.kind,
                (part_id,),
                commit_profile_extrusion,
            ),
        )

    def _accept_derived_profile_feature(
        self,
        record: ProposalPortRecord,
    ) -> ProposalPortRecord:
        operation = record.proposal.operations[0]
        parameters = operation.parameters
        base_recipe = geometry_recipe_from_payload(parameters["base_recipe"])
        source_face_id = str(parameters["source_face_id"])
        selection = resolve_extrusion_source_faces(
            base_recipe,
            (source_face_id,),
        )
        if selection.face_ids != (source_face_id,):
            raise AuthoringContractError("derived Profile source is not canonical")
        part_id = str(parameters["part_id"])
        self._require_current_profile_source(
            part_id,
            base_recipe,
            mismatch_message=(
                "derived Profile source Part no longer matches its proposal"
            ),
        )
        if operation.kind is OperationKind.REVOLVE_PART_PROFILE:
            recipe = RevolvedGeometry(
                base_recipe,
                str(parameters["axis"]),
                parameters["angle_degrees"],
                (source_face_id,),
            )
        else:
            path = geometry_recipe_from_payload(parameters["ordered_wire"])
            if type(path) is not WireGeometry:
                raise AuthoringContractError(
                    "path sweep proposal path is not an explicit WireGeometry"
                )
            recipe = PathSweptGeometry(
                base_recipe,
                path,
                (source_face_id,),
                str(parameters["frame_strategy"]),
            )

        def commit_derived_feature(
            session: ModelSession,
            expected_revision: int,
        ) -> None:
            session.replace_part_geometry(
                part_id,
                recipe,
                expected_session_revision=expected_revision,
            )

        return self._accept_geometry_mutation(
            record,
            AgentGeometryMutation(
                operation.kind,
                (part_id,),
                commit_derived_feature,
            ),
        )

    def _require_current_profile_source(
        self,
        part_id: str,
        base_recipe: object,
        *,
        mismatch_message: str,
    ) -> None:
        snapshot = self._session.snapshot()
        source_part = next(
            (
                part
                for part in snapshot.parts
                if str(part.id) == part_id and not part.suppressed
            ),
            None,
        )
        if source_part is None:
            raise AuthoringContractError(mismatch_message)
        source_recipe = _canonical_profile_source_recipe(
            source_part.geometry_recipe
        )
        if source_recipe != base_recipe:
            raise AuthoringContractError(mismatch_message)

    def _accept_geometry_replacement(
        self,
        record: ProposalPortRecord,
        recipe: object,
    ) -> ProposalPortRecord:
        operation = record.proposal.operations[0]
        parameters = operation.parameters
        part_id = str(parameters["part_id"])

        def commit_replacement(
            session: ModelSession,
            expected_revision: int,
        ) -> None:
            session.replace_part_geometry(
                part_id,
                recipe,
                expected_session_revision=expected_revision,
            )

        return self._accept_geometry_mutation(
            record,
            AgentGeometryMutation(
                operation.kind,
                (part_id,),
                commit_replacement,
            ),
        )

    def _accept_geometry_creation(
        self,
        record: ProposalPortRecord,
        recipe: object,
        snapshot: _SessionSnapshot,
    ) -> ProposalPortRecord:
        proposal = record.proposal
        operation = proposal.operations[0]
        parameters = operation.parameters
        part_name = NameAllocator(
            {"parts": (part.name for part in snapshot.parts)}
        ).require_next(
            "parts",
            "Part",
            str(parameters["part_name"]),
        )
        raw_units = parameters.get("unit_context")
        if not isinstance(raw_units, dict):
            raise AuthoringContractError(
                "geometry proposal requires confirmed unit_context"
            )
        units = UnitContext.from_dict(raw_units)
        if operation.kind is OperationKind.CREATE_NATIVE_PROJECT:
            if snapshot.source_kind is not None:
                raise AuthoringContractError(
                    "create_native_project requires a blank session"
                )
            self._session.create_native_project_with_first_part(
                NameAllocator().require_next(
                    "models",
                    "Model",
                    str(parameters["project_name"]),
                ),
                units,
                recipe,
                part_name=part_name,
                expected_session_revision=proposal.base_session_revision,
            )
        else:
            if snapshot.source_kind != "native":
                raise AuthoringContractError(
                    "add_native_part requires a native project"
                )
            self._session.add_native_part(
                recipe,
                name=part_name,
                mesh_settings=None,
                expected_session_revision=proposal.base_session_revision,
                unit_context=units,
            )
        succeeded = replace(record, state=ProposalState.SUCCEEDED)
        self._records[proposal.proposal_id] = succeeded
        return succeeded

    def _accept_geometry_mutation(
        self,
        record: ProposalPortRecord,
        mutation: AgentGeometryMutation,
    ) -> ProposalPortRecord:
        proposal = record.proposal
        planned_mode = str(proposal.preconditions.get("geometry_edit_mode", "in_place"))
        if planned_mode not in {"in_place", "branch"}:
            raise AuthoringContractError(
                "geometry proposal omitted a valid frozen edit mode"
            )
        if planned_mode != self.geometry_edit_mode():
            raise AuthoringContractError(
                "geometry edit policy changed after proposal creation"
            )
        before_snapshot = self._session.snapshot()
        before_recipe = _snapshot_part_recipe(
            before_snapshot,
            mutation.affected_part_ids[0],
        )
        commit = self._commit_geometry_edit_callback
        if commit is None:
            if planned_mode != "in_place":
                raise AuthoringContractError(
                    "branch geometry edit requires a workspace-aware commit seam"
                )
            mutation.apply(self._session, proposal.base_session_revision)
            context = self._context
            document_id = (
                f"document:{self._session.session_id}"
                if context is None
                else context.binding.document_id
            )
            report = {
                "mode": "in_place",
                "source": {
                    "document_id": document_id,
                    "session_id": self._session.session_id,
                },
                "target": {
                    "document_id": document_id,
                    "session_id": self._session.session_id,
                },
                "part_id": mutation.affected_part_ids[0],
                "affected_part_ids": list(mutation.affected_part_ids),
                "requires_remesh": True,
                "validations": "reset",
                "runs": "not_migrated",
                "results": "not_migrated",
            }
        else:
            report = dict(commit(mutation, proposal.base_session_revision))
        report.setdefault("mode", planned_mode)
        if report.get("mode") != planned_mode:
            raise AuthoringContractError(
                "geometry edit commit mode does not match its proposal"
            )
        provider_safe_authoring_payload(report)
        self._record_feature_proposal_undo(
            proposal,
            mutation,
            before_recipe,
            report,
        )
        self._latest_geometry_iteration_report = deepcopy(report)
        context = self._context
        self._geometry_iteration_report_binding = (
            None
            if context is None
            else (
                context.binding.document_id,
                context.binding.session_id,
            )
        )
        succeeded = replace(record, state=ProposalState.SUCCEEDED)
        self._records[proposal.proposal_id] = succeeded
        return succeeded

    def _record_feature_proposal_undo(
        self,
        proposal: AgentProposal,
        mutation: AgentGeometryMutation,
        before_recipe: object | None,
        report: Mapping[str, object],
    ) -> None:
        if mutation.operation_kind is OperationKind.EXTRUDE_PART_PROFILES:
            source_face_ids = proposal.operations[0].parameters.get(
                "source_face_ids"
            )
            if not isinstance(source_face_ids, list) or len(source_face_ids) != 1:
                return
        if (
            before_recipe is None
            or len(mutation.affected_part_ids) != 1
            or mutation.operation_kind
            not in {
                OperationKind.REPLACE_PART_GEOMETRY,
                OperationKind.EXTRUDE_PART_PROFILES,
                OperationKind.REVOLVE_PART_PROFILE,
                OperationKind.SWEEP_PART_PROFILE,
            }
        ):
            return
        target = report.get("target")
        if not isinstance(target, Mapping):
            return
        target_session_id = str(target.get("session_id", "")).strip()
        target_document_id = str(target.get("document_id", "")).strip()
        if (
            not target_session_id
            or not target_document_id
            or target_session_id != self._session.session_id
        ):
            return
        part_id = mutation.affected_part_ids[0]
        after_recipe = _snapshot_part_recipe(self._session.snapshot(), part_id)
        if after_recipe is None:
            return
        try:
            predecessor = remove_terminal_feature(after_recipe)
            feature_name = derive_feature_history(after_recipe)[-1].name
        except (IndexError, TypeError, ValueError):
            return
        if predecessor != before_recipe:
            return
        self._feature_proposal_records[proposal.proposal_id] = (
            FeatureProposalUndoRecord(
                proposal.proposal_id,
                proposal.proposal_hash,
                target_document_id,
                target_session_id,
                part_id,
                feature_name,
                deepcopy(before_recipe),
                deepcopy(after_recipe),
                self._session.session_revision,
            )
        )

    def feature_proposal_record(
        self,
        proposal_id: str,
    ) -> FeatureProposalUndoRecord:
        try:
            return self._feature_proposal_records[str(proposal_id)]
        except KeyError as error:
            raise AuthoringContractError(
                "proposal did not append a reversible native feature"
            ) from error

    def can_undo_proposal(self, proposal_id: str) -> bool:
        record = self._feature_proposal_records.get(str(proposal_id))
        if (
            record is None
            or record.state is not AppliedPatchState.APPLIED
            or self._session.session_id != record.target_session_id
            or self._session.session_revision != record.session_revision
        ):
            return False
        context = self._context
        if (
            context is not None
            and context.binding.document_id != record.target_document_id
        ):
            return False
        current = _snapshot_part_recipe(self._session.snapshot(), record.part_id)
        if current != record.after_recipe:
            return False
        try:
            return remove_terminal_feature(current) == record.before_recipe
        except (TypeError, ValueError):
            return False

    def undo_proposal(self, proposal_id: str) -> FeatureProposalUndoRecord:
        try:
            record = self._feature_proposal_records[str(proposal_id)]
        except KeyError as error:
            raise AuthoringContractError(
                "proposal did not append a reversible native feature"
            ) from error
        if record.state is not AppliedPatchState.APPLIED:
            raise AuthoringAuthorizationError(
                f"proposal feature is already {record.state.value}"
            )
        if not self.can_undo_proposal(proposal_id):
            stale = replace(
                record,
                state=AppliedPatchState.STALE,
                message="session revision or terminal feature changed after proposal",
            )
            self._feature_proposal_records[str(proposal_id)] = stale
            raise AuthoringAuthorizationError(stale.message)
        snapshot = self._session.snapshot()
        delta = self._session.replace_part_geometry(
            record.part_id,
            deepcopy(record.before_recipe),
            expected_part_revision=snapshot.part_revision(record.part_id),
            expected_session_revision=record.session_revision,
        )
        undone = replace(
            record,
            session_revision=delta.session_revision,
            state=AppliedPatchState.UNDONE,
            message=f"removed native feature {record.feature_name}",
        )
        self._feature_proposal_records[str(proposal_id)] = undone
        return undone

    def can_accept(self, proposal_id: str) -> bool:
        """Return whether the current Session still satisfies the proposal."""

        try:
            record = self._pending(proposal_id)
            proposal = record.proposal
            if proposal.proposal_kind is not ProposalKind.SOLVE:
                return True
            self._require_current_solve_identity(proposal)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def request_preflight(self, step_name: str) -> AgentPreflightRecord:
        """Start one confirmation-free preflight through the GUI task owner."""

        if self._start_preflight_task is None:
            raise AuthoringContractError(
                "automatic preflight execution is not configured"
            )
        if any(
            item.state is AgentPreflightState.RUNNING
            for item in self._preflight_records.values()
        ):
            raise AuthoringContractError("an Agent preflight is already running")
        target_session = self._session
        snapshot = target_session.snapshot()
        clean_step = str(step_name)
        artifact = snapshot.artifact
        if (
            snapshot.source_kind != "native"
            or not snapshot.model_current
            or not snapshot.mesh_current
            or artifact is None
            or clean_step not in snapshot.runnable_step_names()
        ):
            raise AuthoringContractError(
                "automatic preflight requires a current native model and step"
            )
        self._preflight_counter += 1
        request_id = f"preflight-{self._preflight_counter}"
        record = AgentPreflightRecord(
            request_id=request_id,
            step_name=clean_step,
            base_session_revision=snapshot.session_revision,
            state=AgentPreflightState.RUNNING,
            message="Running deterministic model preflight in the background",
        )
        self._preflight_records[request_id] = record
        request = AgentPreflightTaskRequest(
            request_id=request_id,
            step_name=clean_step,
            base_session_revision=snapshot.session_revision,
            session_id=snapshot.session_id,
            artifact_id=artifact.artifact_id,
            model_revision=snapshot.model_revision,
        )
        self._preflight_tasks[request_id] = _PendingPreflightTask(
            target_session, request,
        )
        try:
            started = self._start_preflight_task(request)
        except Exception:
            self._preflight_tasks.pop(request_id, None)
            current = self._preflight_records[request_id]
            if current.state is AgentPreflightState.RUNNING:
                self._preflight_records[request_id] = replace(
                    current,
                    state=AgentPreflightState.FAILED,
                    message="GUI preflight task failed to start",
                )
            raise
        if (
            not started
            and self._preflight_records[request_id].state is AgentPreflightState.RUNNING
        ):
            self._preflight_tasks.pop(request_id, None)
            failed = replace(
                record,
                state=AgentPreflightState.FAILED,
                message="GUI background task controller is busy or rejected startup",
            )
            self._preflight_records[request_id] = failed
            return failed
        return self._preflight_records[request_id]

    def complete_preflight(
        self,
        request_id: str,
        state: AgentPreflightState,
        message: str = "",
    ) -> AgentPreflightRecord:
        if state is AgentPreflightState.RUNNING:
            raise ValueError("preflight completion requires a terminal state")
        record = self._preflight_records[str(request_id)]
        if record.state is not AgentPreflightState.RUNNING:
            raise AuthoringAuthorizationError(
                f"preflight is already {record.state.value}"
            )
        pending = self._preflight_tasks.pop(str(request_id), None)
        stamp = None
        if state is AgentPreflightState.PASSED:
            try:
                if pending is None:
                    raise AuthoringContractError("preflight has no originating Session")
                snapshot = pending.session.snapshot()
                request = pending.request
                artifact = snapshot.artifact
                if (
                    not snapshot.is_open
                    or snapshot.session_id != request.session_id
                    or artifact is None
                    or artifact.artifact_id != request.artifact_id
                    or snapshot.model_revision != request.model_revision
                ):
                    state = AgentPreflightState.STALE
                    message = "The originating model changed before preflight completion"
                else:
                    stamp = validation_stamp_for_snapshot(snapshot, request.step_name)
            except Exception as error:
                state = AgentPreflightState.FAILED
                message = (
                    "Cannot bind the preflight result to its originating model: "
                    + (str(error).strip() or type(error).__name__)
                )
        completed = replace(
            record,
            state=state,
            message=str(message).strip(),
            validation_stamp=stamp,
        )
        self._preflight_records[str(request_id)] = completed
        return completed

    def preflight_record(self, request_id: str) -> AgentPreflightRecord:
        try:
            return self._preflight_records[str(request_id)]
        except KeyError as error:
            raise AuthoringContractError(
                "automatic preflight request is not registered"
            ) from error

    def apply_patch(self, patch: ModelPatch) -> AppliedPatchRecord:
        """Apply one non-destructive A4/A5 patch and retain its exact inverse."""

        if type(patch) is not ModelPatch:
            raise TypeError("patch must be ModelPatch")
        current = self._patch_records.get(patch.patch_id)
        if current is not None:
            if current.patch.patch_hash != patch.patch_hash:
                raise AuthoringContractError(
                    "patch_id was reused with different content"
                )
            return replace(current, replayed=True)
        snapshot = self._session.snapshot()
        current_document_id = (
            self._context.binding.document_id
            if self._context is not None
            else f"document:{snapshot.session_id}"
        )
        if (
            patch.target_session_id != snapshot.session_id
            or patch.target_document_id != current_document_id
            or patch.base_session_revision != snapshot.session_revision
        ):
            raise AuthoringContractError("automatic patch target is stale")
        preconditions = patch.preconditions
        authoring_mode = (
            preconditions.get("authoring_mode")
            if isinstance(preconditions, Mapping)
            else None
        )
        has_results = any(
            bool(getattr(run, "has_result", False)) for run in snapshot.runs
        )
        if has_results and patch.invalidation_impact.get("results") is True:
            raise AuthoringAuthorizationError(
                "a result-invalidating edit requires GUI confirmation"
            )
        inverse_operations = inverse_operations_for_snapshot(snapshot)
        if authoring_mode == "direct_edit":
            if (
                len(patch.operations) != 1
                or patch.operations[0].kind is not OperationKind.EDIT_MODEL_OBJECT
            ):
                raise AuthoringContractError(
                    "direct edit patch requires one edit operation"
                )
            delta = apply_edit_operation(
                self._session,
                patch.operations[0],
                base_session_revision=patch.base_session_revision,
            )
        else:
            batch = scoped_definition_batch_from_operations(
                patch.operations,
                base_session_revision=patch.base_session_revision,
            )
            if authoring_mode == "direct_incremental":
                require_incremental_definition_batch(
                    snapshot,
                    batch,
                    preconditions.get("direct_action"),
                )
            elif authoring_mode == "strict_incremental":
                require_strict_definition_batch(
                    snapshot,
                    batch,
                    preconditions.get("direct_action"),
                )
            else:
                raise AuthoringContractError("unsupported definition authoring mode")
            delta = self._session.apply_scoped_definition_batch(batch)
        inverse = ModelPatch.create(
            patch_id=f"inverse-{patch.patch_hash[:24]}",
            agent_session_id=patch.agent_session_id,
            turn_id=patch.turn_id,
            source_tool_call_ids=patch.source_tool_call_ids,
            target_document_id=patch.target_document_id,
            target_session_id=patch.target_session_id,
            base_session_revision=delta.session_revision,
            draft_revision=patch.draft_revision,
            operations=inverse_operations,
            preconditions={
                "forward_patch_hash": patch.patch_hash,
                "expected_session_revision": delta.session_revision,
                "one_shot": True,
            },
            expected_changes={"restore_exact_pre_state": True},
            invalidation_impact={
                "model": True,
                "validation": True,
                "results": False,
                "historical_results_retained": True,
                "current_validation_reset": True,
                "current_result_display_reset": True,
            },
            display_summary={
                "title": "Undo these Agent changes",
                "forward_patch_id": patch.patch_id,
            },
        )
        applied = AppliedPatchRecord(
            patch,
            inverse,
            delta.session_revision,
            delta,
        )
        self._patch_records[patch.patch_id] = applied
        self._project_definition_delta(delta)
        return applied

    def can_undo_patch(self, patch_id: str) -> bool:
        record = self._patch_records.get(str(patch_id))
        return (
            record is not None
            and record.state is AppliedPatchState.APPLIED
            and self._session.session_id == record.inverse_patch.target_session_id
            and self._session.session_revision
            == record.inverse_patch.base_session_revision
        )

    def undo_patch(self, patch_id: str) -> AppliedPatchRecord:
        """Apply an inverse exactly once while its post revision is current."""

        try:
            record = self._patch_records[str(patch_id)]
        except KeyError as error:
            raise AuthoringContractError("automatic patch is not registered") from error
        if record.state is not AppliedPatchState.APPLIED:
            raise AuthoringAuthorizationError(
                f"automatic patch is already {record.state.value}"
            )
        if not self.can_undo_patch(patch_id):
            stale = replace(
                record,
                state=AppliedPatchState.STALE,
                message="session revision changed after the Agent patch",
            )
            self._patch_records[str(patch_id)] = stale
            raise AuthoringAuthorizationError(stale.message)
        inverse = record.inverse_patch
        batch = scoped_definition_batch_from_operations(
            inverse.operations,
            base_session_revision=inverse.base_session_revision,
        )
        delta = self._session.apply_scoped_definition_batch(batch)
        undone = replace(
            record,
            session_revision=delta.session_revision,
            delta=delta,
            state=AppliedPatchState.UNDONE,
            message="Agent patch inverse applied",
        )
        self._patch_records[str(patch_id)] = undone
        self._project_definition_delta(delta)
        return undone

    def patch_record(self, patch_id: str) -> AppliedPatchRecord:
        try:
            return self._patch_records[str(patch_id)]
        except KeyError as error:
            raise AuthoringContractError("automatic patch is not registered") from error

    def _project_definition_delta(self, delta: SessionDelta) -> None:
        if self._apply_definition_delta is not None:
            self._apply_definition_delta(delta)

    def _accept_mesh(self, record: ProposalPortRecord) -> ProposalPortRecord:
        if self._start_mesh_task is None:
            raise AuthoringContractError("mesh proposal execution is not configured")
        proposal = record.proposal
        intent_operation, request_operation = proposal.operations
        intent = MeshIntent.from_dict(
            intent_operation.parameters["mesh_intent"]  # type: ignore[arg-type]
        )
        part_id = str(intent_operation.parameters["part_id"])
        if (
            str(request_operation.parameters["part_id"]) != part_id
            or str(request_operation.parameters["mesh_intent_hash"])
            != intent.intent_hash
        ):
            raise AuthoringContractError(
                "mesh request does not match its retained MeshIntent"
            )
        snapshot = self._session.snapshot()
        part = next(
            (item for item in snapshot.parts if item.id == part_id),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError("mesh proposal Part is unavailable")
        settings = intent.to_mesh_settings(part.geometry_recipe)
        task = self._session.prepare_agent_mesh_generation(
            part_id,
            settings,
            intent.intent_hash,
            expected_session_revision=proposal.base_session_revision,
        )
        target_session = self._session
        self._mesh_tasks[proposal.proposal_id] = _PendingMeshTask(target_session, task)
        running = replace(record, state=ProposalState.RUNNING)
        self._records[proposal.proposal_id] = running
        try:
            started = self._start_mesh_task(
                AgentMeshTaskRequest(
                    proposal.proposal_id,
                    proposal.proposal_hash,
                    task,
                )
            )
        except Exception as error:
            target_session.terminate_agent_mesh_task(task.token, str(error))
            self._mesh_tasks.pop(proposal.proposal_id, None)
            raise
        if not started:
            target_session.terminate_agent_mesh_task(
                task.token,
                "GUI background task controller is busy",
            )
            self._mesh_tasks.pop(proposal.proposal_id, None)
            failed = replace(
                running,
                state=ProposalState.FAILED,
                message="GUI background task controller is busy",
            )
            self._records[proposal.proposal_id] = failed
            raise AuthoringContractError(failed.message)
        return self._records[proposal.proposal_id]

    def _accept_solve(
        self,
        record: ProposalPortRecord,
    ) -> ProposalPortRecord:
        if self._start_solve_task is None:
            raise AuthoringContractError("solve proposal execution is not configured")
        proposal = record.proposal
        (
            step_name,
            job_name,
            artifact_id,
            model_revision,
            stamp,
        ) = self._require_current_solve_identity(proposal)
        running = replace(
            record,
            state=ProposalState.RUNNING,
            message="GUI authorized; submitting background job",
        )
        self._records[proposal.proposal_id] = running
        if self._record_listener is not None:
            self._record_listener(running)
        request = AgentSolveTaskRequest(
            proposal_id=proposal.proposal_id,
            proposal_hash=proposal.proposal_hash,
            step_name=step_name,
            job_name=job_name,
            base_session_revision=proposal.base_session_revision,
            artifact_id=artifact_id,
            model_revision=model_revision,
            validation_stamp=stamp,
        )
        try:
            started = self._start_solve_task(request)
        except Exception:
            self._records[proposal.proposal_id] = replace(
                running,
                state=ProposalState.FAILED,
                message="GUI background job failed to start",
            )
            raise
        if not started:
            failed = replace(
                running,
                state=ProposalState.FAILED,
                message="GUI background task controller is busy or rejected startup",
            )
            self._records[proposal.proposal_id] = failed
            raise AuthoringContractError(failed.message)
        return self._records[proposal.proposal_id]

    def _require_current_solve_identity(
        self,
        proposal: AgentProposal,
    ) -> tuple[str, str, str, int, SolveValidationStamp]:
        identity = solve_operation_identity(proposal.operations[0])
        step_name, job_name, artifact_id, model_revision, stamp = identity
        snapshot = self._session.snapshot()
        current_stamp = validation_stamp_for_snapshot(snapshot, step_name)
        current_document_id = (
            self._context.binding.document_id
            if self._context is not None
            else f"document:{snapshot.session_id}"
        )
        if (
            proposal.target_session_id != snapshot.session_id
            or proposal.target_document_id != current_document_id
            or proposal.base_session_revision != snapshot.session_revision
            or artifact_id != getattr(snapshot.artifact, "artifact_id", None)
            or model_revision != snapshot.model_revision
            or stamp != current_stamp
        ):
            raise AuthoringContractError(
                "solve proposal revision or validation stamp is stale"
            )
        if self._session.find_run(job_name) is not None:
            raise AuthoringContractError("solve proposal job name is already used")
        return identity

    def progress_solve(
        self,
        proposal_id: str,
        message: str,
    ) -> ProposalPortRecord:
        record = self._records[str(proposal_id)]
        if record.state is not ProposalState.RUNNING:
            raise AuthoringAuthorizationError(
                "solve progress requires a running proposal"
            )
        updated = replace(record, message=str(message).strip())
        self._records[str(proposal_id)] = updated
        if self._record_listener is not None:
            self._record_listener(updated)
        return updated

    def complete_solve(
        self,
        proposal_id: str,
        state: ProposalState,
        message: str = "",
    ) -> ProposalPortRecord:
        if state not in {
            ProposalState.SUCCEEDED,
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
        }:
            raise ValueError("solve completion state must be terminal")
        record = self._records[str(proposal_id)]
        if record.state is not ProposalState.RUNNING:
            raise AuthoringAuthorizationError(
                f"solve proposal is already {record.state.value}"
            )
        completed = replace(
            record,
            state=state,
            message=str(message).strip(),
        )
        self._records[str(proposal_id)] = completed
        if self._record_listener is not None:
            self._record_listener(completed)
        return completed

    def accept_mesh_result(
        self,
        proposal_id: str,
        model: object,
    ) -> SessionDelta:
        record = self._records.get(proposal_id)
        if record is None or record.state is not ProposalState.RUNNING:
            raise AuthoringAuthorizationError("mesh result requires a running proposal")
        pending = self._mesh_tasks[proposal_id]
        delta = pending.session.accept_agent_generated_model(pending.snapshot.token, model)
        if delta.accepted:
            self.complete_mesh(
                proposal_id,
                ProposalState.SUCCEEDED,
                "Mesh intent and generated model committed atomically",
            )
        return delta

    def terminate_mesh(
        self,
        proposal_id: str,
        state: ProposalState,
        message: str,
    ) -> SessionDelta:
        if state not in {
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
        }:
            raise ValueError("terminated mesh proposal requires a failure state")
        pending = self._mesh_tasks[proposal_id]
        delta = pending.session.terminate_agent_mesh_task(pending.snapshot.token, message)
        self.complete_mesh(proposal_id, state, message)
        return delta

    def refresh_projection(self) -> None:
        self._refresh_callback()

    def reject(self, proposal_id: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        rejected = replace(record, state=ProposalState.REJECTED)
        self._records[proposal_id] = rejected
        return rejected

    def cancel(self, proposal_id: str, reason: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        cancelled = replace(
            record,
            state=ProposalState.CANCELLED,
            message=str(reason).strip(),
        )
        self._records[proposal_id] = cancelled
        return cancelled

    def stale(self, proposal_id: str, reason: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        stale = replace(
            record,
            state=ProposalState.STALE,
            message=str(reason).strip(),
        )
        self._records[proposal_id] = stale
        return stale

    def mark_failed(self, proposal_id: str, message: str) -> ProposalPortRecord:
        record = self._records[proposal_id]
        failed = replace(
            record,
            state=ProposalState.FAILED,
            message=str(message).strip(),
        )
        self._records[proposal_id] = failed
        return failed

    def complete_mesh(
        self,
        proposal_id: str,
        state: ProposalState,
        message: str = "",
    ) -> ProposalPortRecord:
        if state not in {
            ProposalState.SUCCEEDED,
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
        }:
            raise ValueError("mesh completion state must be terminal")
        try:
            record = self._records[proposal_id]
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc
        if record.state is not ProposalState.RUNNING:
            raise AuthoringAuthorizationError(
                f"mesh proposal is already {record.state.value}"
            )
        completed = replace(
            record,
            state=state,
            message=str(message).strip(),
        )
        self._records[proposal_id] = completed
        self._mesh_tasks.pop(proposal_id, None)
        if self._record_listener is not None:
            self._record_listener(completed)
        return completed

    def _pending(self, proposal_id: str) -> ProposalPortRecord:
        try:
            record = self._records[proposal_id]
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc
        if record.state is not ProposalState.PENDING_CONFIRMATION:
            raise AuthoringAuthorizationError(
                f"proposal is already {record.state.value}"
            )
        return record


class AgentAuthoringBridge:
    """A1 single write boundary with a Fake Port and no model writes."""

    def __init__(self, port: AuthoringPort) -> None:
        self._port = port
        self._context: AuthoringContext | None = None
        self._records: dict[str, ProposalPortRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._patch_idempotency: dict[str, str] = {}
        self._authorization_nonce = 0
        self._unused_authorizations: set[_GuiControlAuthorization] = set()
        self._accepting_proposal_id: str | None = None
        self._gui_thread_id = threading.get_ident()
        self._patch_listener: Callable[[AppliedPatchRecord], None] | None = None
        self._result_invalidation_confirmation: Callable[[], bool] | None = None
        self._lifecycle_listener: (
            Callable[[AgentProposal, ProposalState, str], None] | None
        ) = None
        self._preview_listener: (
            Callable[[str, AgentProposalPreview | None], None] | None
        ) = None
        self._proposal_previews: dict[str, AgentProposalPreview] = {}
        self._last_lifecycle_notice: dict[str, tuple[ProposalState, str]] = {}
        listener = getattr(port, "set_record_listener", None)
        if callable(listener):
            listener(self._receive_port_record)

    @property
    def context(self) -> AuthoringContext | None:
        return self._context

    @property
    def port(self) -> AuthoringPort:
        return self._port

    def set_patch_listener(
        self,
        callback: Callable[[AppliedPatchRecord], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("patch listener must be callable")
        self._patch_listener = callback

    def set_lifecycle_listener(
        self,
        callback: Callable[[AgentProposal, ProposalState, str], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("lifecycle listener must be callable")
        self._lifecycle_listener = callback

    def set_preview_listener(
        self,
        callback: Callable[[str, AgentProposalPreview | None], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("preview listener must be callable")
        self._preview_listener = callback

    def set_result_invalidation_confirmation(
        self,
        callback: Callable[[], bool],
    ) -> None:
        """Install the GUI's canonical unsaved-result confirmation gate."""

        if not callable(callback):
            raise TypeError("result invalidation confirmation must be callable")
        self._result_invalidation_confirmation = callback

    def detach_gui_callbacks(self) -> None:
        """Break GUI-owned callback cycles when the window closes."""

        self.clear_proposal_previews()
        self._patch_listener = None
        self._lifecycle_listener = None
        self._preview_listener = None
        self._result_invalidation_confirmation = None
        detach_port = getattr(self._port, "detach_gui_callbacks", None)
        if callable(detach_port):
            detach_port()

    def bind_snapshot(
        self,
        snapshot: _SessionSnapshot,
        *,
        document_id: str | int | None = None,
        workspace_selected: bool = False,
    ) -> tuple[str, ...]:
        return self.bind_context(
            authoring_context_from_snapshot(
                snapshot,
                document_id=document_id,
                workspace_selected=workspace_selected,
            )
        )

    def bind_context(self, context: AuthoringContext) -> tuple[str, ...]:
        if type(context) is not AuthoringContext:
            raise AuthoringContractError("context must be AuthoringContext")
        prior_binding = None if self._context is None else self._context.binding
        self._context = context
        self._port.set_context(context)
        if prior_binding is None or prior_binding == context.binding:
            return ()

        stale_ids: list[str] = []
        for proposal_id, record in tuple(self._records.items()):
            if (
                record.state is ProposalState.PENDING_CONFIRMATION
                and proposal_id != self._accepting_proposal_id
            ):
                stale = self._port.stale(
                    proposal_id,
                    "Bound document, session, or revision changed",
                )
                self._records[proposal_id] = stale
                self._notify_lifecycle(stale)
                stale_ids.append(proposal_id)
        return tuple(stale_ids)

    def unbind_context(self, reason: str) -> tuple[str, ...]:
        """Invalidate Agent authoring state when no editable model is bound."""

        self._require_gui_thread()
        message = str(reason).strip()
        if not message:
            raise ValueError("unbind reason must be non-blank")
        stale_ids = self.stale_pending_proposals_from_gui(message)
        clear_context = getattr(self._port, "clear_context", None)
        if callable(clear_context):
            clear_context()
        self._context = None
        return stale_ids

    def stale_pending_proposals_from_gui(
        self,
        reason: str,
    ) -> tuple[str, ...]:
        """Make every old Agent-session proposal terminal on the GUI owner."""

        self._require_gui_thread()
        message = str(reason).strip()
        if not message:
            raise ValueError("stale reason must be non-blank")
        stale_ids: list[str] = []
        for proposal_id, record in tuple(self._records.items()):
            if record.state is not ProposalState.PENDING_CONFIRMATION:
                continue
            stale = self._port.stale(proposal_id, message)
            self._records[proposal_id] = stale
            self._notify_lifecycle(stale)
            stale_ids.append(proposal_id)
        return tuple(stale_ids)

    def cancel_pending_proposals_from_gui(
        self,
        reason: str,
    ) -> tuple[str, ...]:
        self._require_gui_thread()
        message = str(reason).strip()
        if not message:
            raise ValueError("cancel reason must be non-blank")
        cancelled_ids: list[str] = []
        for proposal_id, record in tuple(self._records.items()):
            if record.state is not ProposalState.PENDING_CONFIRMATION:
                continue
            cancelled = self._port.cancel(proposal_id, message)
            self._records[proposal_id] = cancelled
            self._notify_lifecycle(cancelled)
            cancelled_ids.append(proposal_id)
        return tuple(cancelled_ids)

    def register_proposal(
        self,
        proposal: AgentProposal,
        detached_preview: AgentProposalPreview | None = None,
    ) -> BridgeReceipt:
        if type(proposal) is not AgentProposal:
            raise AuthoringContractError("proposal must be AgentProposal")
        current = self._records.get(proposal.proposal_id)
        if current is not None:
            if current.proposal.proposal_hash != proposal.proposal_hash:
                raise AuthoringContractError(
                    "proposal_id was reused with different content"
                )
            return self._receipt(current, replayed=True)
        replay_id = self._idempotency.get(proposal.idempotency_key)
        if replay_id is not None:
            return self._receipt(self._records[replay_id], replayed=True)
        if detached_preview is not None:
            evidence = proposal.preconditions.get("local_evidence")
            if not isinstance(evidence, Mapping) or (
                evidence.get("output_recipe_digest") != detached_preview.recipe_digest
                or evidence.get("output_proof_digest") != detached_preview.proof_digest
            ):
                raise AuthoringContractError(
                    "detached preview does not match the proposal Recipe proof"
                )
        self._require_live_target(proposal)
        record = self._port.present(proposal)
        self._records[proposal.proposal_id] = record
        self._idempotency[proposal.idempotency_key] = proposal.proposal_id
        if detached_preview is not None:
            self._proposal_previews[proposal.proposal_id] = detached_preview
            if self._preview_listener is not None:
                self._preview_listener(proposal.proposal_id, detached_preview)
        return self._receipt(record)

    def clear_proposal_previews(self) -> None:
        for proposal_id in tuple(self._proposal_previews):
            self._clear_proposal_preview(proposal_id)

    def _clear_proposal_preview(self, proposal_id: str) -> None:
        if self._proposal_previews.pop(proposal_id, None) is not None:
            if self._preview_listener is not None:
                self._preview_listener(proposal_id, None)

    def apply_automatic_patch(
        self,
        patch: ModelPatch,
    ) -> AppliedPatchRecord:
        """Apply a revision-bound reversible edit without a confirmation click."""

        if type(patch) is not ModelPatch:
            raise TypeError("patch must be ModelPatch")
        replay_id = self._patch_idempotency.get(patch.idempotency_key)
        if replay_id is not None:
            record_getter = getattr(self._port, "patch_record")
            record = record_getter(replay_id)
            if record.patch.patch_hash != patch.patch_hash:
                raise AuthoringContractError(
                    "patch idempotency key was reused with different content"
                )
            return replace(record, replayed=True)
        self._require_live_patch_target(patch)
        port_apply = getattr(self._port, "apply_patch", None)
        if not callable(port_apply):
            raise AuthoringContractError(
                "authoring port does not support automatic patches"
            )
        record = port_apply(patch)
        self._patch_idempotency[patch.idempotency_key] = patch.patch_id
        if self._patch_listener is not None:
            self._patch_listener(record)
        return record

    def can_undo_patch(self, patch_id: str) -> bool:
        check = getattr(self._port, "can_undo_patch", None)
        return bool(callable(check) and check(patch_id))

    def undo_patch_from_gui_control(
        self,
        patch_id: str,
    ) -> AppliedPatchRecord:
        self._require_gui_thread()
        undo = getattr(self._port, "undo_patch", None)
        if not callable(undo):
            raise AuthoringContractError(
                "authoring port does not support patch inverse"
            )
        record = undo(patch_id)
        if self._patch_listener is not None:
            self._patch_listener(record)
        return record

    def feature_proposal_record(
        self,
        proposal_id: str,
    ) -> FeatureProposalUndoRecord | None:
        getter = getattr(self._port, "feature_proposal_record", None)
        if not callable(getter):
            return None
        try:
            return getter(proposal_id)
        except AuthoringContractError:
            return None

    def can_undo_proposal(self, proposal_id: str) -> bool:
        check = getattr(self._port, "can_undo_proposal", None)
        return bool(callable(check) and check(proposal_id))

    def undo_proposal_from_gui_control(
        self,
        proposal_id: str,
    ) -> FeatureProposalUndoRecord:
        self._require_gui_thread()
        undo = getattr(self._port, "undo_proposal", None)
        if not callable(undo):
            raise AuthoringContractError(
                "authoring port does not support proposal feature removal"
            )
        record = undo(proposal_id)
        refresh = getattr(self._port, "refresh_projection", None)
        if callable(refresh):
            refresh()
        return record

    def accept_proposal(
        self,
        proposal_id: str,
        authorization: object | None = None,
    ) -> BridgeReceipt:
        token = self._consume_gui_authorization(
            proposal_id,
            "accept",
            authorization,
        )
        del token
        record = self._pending_record(proposal_id)
        self._require_live_target(record.proposal)
        self._accepting_proposal_id = proposal_id
        try:
            accepted = self._port.accept(proposal_id)
        except Exception as exc:
            failed = replace(
                record,
                state=ProposalState.FAILED,
                message=str(exc).strip() or type(exc).__name__,
            )
            marker = getattr(self._port, "mark_failed", None)
            if callable(marker):
                failed = marker(proposal_id, failed.message)
            self._records[proposal_id] = failed
            self._notify_lifecycle(failed)
            return self._receipt(failed)
        finally:
            self._accepting_proposal_id = None
        self._records[proposal_id] = accepted
        if accepted.state is ProposalState.SUCCEEDED and (
            accepted.proposal.proposal_kind is ProposalKind.GEOMETRY
            or (
                accepted.proposal.proposal_kind is ProposalKind.DESTRUCTIVE_EDIT
                and len(accepted.proposal.operations) == 1
                and accepted.proposal.operations[0].kind
                in {
                    OperationKind.DELETE_MODEL_OBJECT,
                    OperationKind.EDIT_MODEL_OBJECT,
                }
            )
        ):
            refresh = getattr(self._port, "refresh_projection", None)
            if callable(refresh):
                refresh()
        self._notify_lifecycle(accepted)
        return self._receipt(accepted)

    def reject_proposal(
        self,
        proposal_id: str,
        authorization: object | None = None,
    ) -> BridgeReceipt:
        token = self._consume_gui_authorization(
            proposal_id,
            "reject",
            authorization,
        )
        del token
        self._pending_record(proposal_id)
        rejected = self._port.reject(proposal_id)
        self._records[proposal_id] = rejected
        self._notify_lifecycle(rejected)
        return self._receipt(rejected)

    def accept_from_gui_control(self, proposal_id: str) -> BridgeReceipt:
        self._require_gui_thread()
        record = self._pending_record(proposal_id)
        if record.proposal.invalidation_impact.get("results") is True:
            confirmation = self._result_invalidation_confirmation
            if confirmation is not None and not confirmation():
                raise AuthoringAuthorizationError(
                    "result-invalidating proposal was cancelled by the user"
                )
        authorization = self._issue_gui_authorization(
            proposal_id,
            "accept",
        )
        return self.accept_proposal(proposal_id, authorization)

    def can_accept_from_gui_control(self, proposal_id: str) -> bool:
        """Return the exact local gate used to enable a GUI accept button."""

        try:
            record = self._pending_record(proposal_id)
            self._require_live_target(record.proposal)
            port_check = getattr(self._port, "can_accept", None)
            return not callable(port_check) or bool(port_check(proposal_id))
        except (
            AuthoringAuthorizationError,
            AuthoringContractError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return False

    def request_preflight(
        self,
        step_name: str,
    ) -> AgentPreflightRecord:
        """Start a local read-only preflight without a confirmation token."""

        self._require_gui_thread()
        request = getattr(self._port, "request_preflight", None)
        if not callable(request):
            raise AuthoringContractError(
                "authoring port does not support automatic preflight"
            )
        return request(step_name)

    def reject_from_gui_control(self, proposal_id: str) -> BridgeReceipt:
        self._require_gui_thread()
        authorization = self._issue_gui_authorization(
            proposal_id,
            "reject",
        )
        return self.reject_proposal(proposal_id, authorization)

    def state(self, proposal_id: str) -> ProposalState:
        try:
            return self._records[proposal_id].state
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc

    def ensure_display_identity_from_gui(
        self,
        proposal_id: str,
        proposal_hash: str,
        agent_session_id: str,
        turn_id: str,
    ) -> bool:
        """Stale a pending proposal if its rendered identity was substituted."""

        self._require_gui_thread()
        record = self._records.get(str(proposal_id))
        if record is None:
            return False
        proposal = record.proposal
        matches = (
            proposal.proposal_hash == str(proposal_hash)
            and proposal.agent_session_id == str(agent_session_id)
            and proposal.turn_id == str(turn_id)
        )
        if matches:
            return record.state is ProposalState.PENDING_CONFIRMATION
        if record.state is ProposalState.PENDING_CONFIRMATION:
            stale = self._port.stale(
                proposal.proposal_id,
                "Proposal hash, Agent session, or turn identity mismatch",
            )
            self._records[proposal.proposal_id] = stale
            self._notify_lifecycle(stale)
        return False

    def _require_live_target(self, proposal: AgentProposal) -> None:
        context = self._context
        if context is None:
            raise AuthoringContractError("there is no local model binding")
        binding = context.binding
        if not binding.supported:
            raise AuthoringContractError(
                "the current document cannot be bound for V1 authoring"
            )
        if (
            proposal.target_document_id != binding.document_id
            or proposal.target_session_id != binding.session_id
            or proposal.base_session_revision != binding.session_revision
        ):
            raise AuthoringContractError("proposal target is stale")

    def _require_live_patch_target(self, patch: ModelPatch) -> None:
        context = self._context
        if context is None:
            raise AuthoringContractError("there is no local model binding")
        binding = context.binding
        if (
            not binding.supported
            or patch.target_document_id != binding.document_id
            or patch.target_session_id != binding.session_id
            or patch.base_session_revision != binding.session_revision
        ):
            raise AuthoringContractError("patch target is stale")

    def _pending_record(self, proposal_id: str) -> ProposalPortRecord:
        try:
            record = self._records[proposal_id]
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc
        if record.state is not ProposalState.PENDING_CONFIRMATION:
            raise AuthoringAuthorizationError(
                f"proposal is already {record.state.value}"
            )
        return record

    def _issue_gui_authorization(
        self,
        proposal_id: str,
        action: str,
    ) -> _GuiControlAuthorization:
        self._authorization_nonce += 1
        authorization = _GuiControlAuthorization(
            proposal_id,
            action,
            self._authorization_nonce,
        )
        self._unused_authorizations.add(authorization)
        return authorization

    def _consume_gui_authorization(
        self,
        proposal_id: str,
        action: str,
        authorization: object | None,
    ) -> _GuiControlAuthorization:
        if (
            type(authorization) is not _GuiControlAuthorization
            or authorization not in self._unused_authorizations
            or authorization.proposal_id != proposal_id
            or authorization.action != action
        ):
            raise AuthoringAuthorizationError(
                "a live GUI control authorization is required"
            )
        self._unused_authorizations.remove(authorization)
        return authorization

    def _require_gui_thread(self) -> None:
        if threading.get_ident() != self._gui_thread_id:
            raise AuthoringAuthorizationError(
                "GUI authorization must run on the bridge owner thread"
            )

    def _receive_port_record(self, record: ProposalPortRecord) -> None:
        current = self._records.get(record.proposal.proposal_id)
        if (
            current is None
            or current.proposal.proposal_hash != record.proposal.proposal_hash
        ):
            raise AuthoringContractError(
                "port lifecycle update does not match a registered proposal"
            )
        self._records[record.proposal.proposal_id] = record
        self._notify_lifecycle(record)

    def _notify_lifecycle(self, record: ProposalPortRecord) -> None:
        notice = (record.state, record.message)
        proposal_id = record.proposal.proposal_id
        if self._last_lifecycle_notice.get(proposal_id) == notice:
            return
        self._last_lifecycle_notice[proposal_id] = notice
        if record.state in {
            ProposalState.SUCCEEDED,
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
            ProposalState.REJECTED,
        }:
            self._clear_proposal_preview(proposal_id)
        if self._lifecycle_listener is not None:
            self._lifecycle_listener(
                record.proposal,
                record.state,
                record.message,
            )

    @staticmethod
    def _receipt(
        record: ProposalPortRecord,
        *,
        replayed: bool = False,
    ) -> BridgeReceipt:
        return BridgeReceipt(
            proposal_id=record.proposal.proposal_id,
            state=record.state,
            message=record.message,
            replayed=replayed,
        )


def create_session_authoring_workflow_controller(
    session: ModelSession,
    authoring_bridge: AgentAuthoringBridge,
    result_bridge: AgentResultQueryBridge,
    *,
    next_job_name: Callable[[], str] | None = None,
    workspace_catalog_bridge: WorkspaceCatalogBridge | None = None,
    create_model_document: Callable[[str | None], str] | None = None,
    export_facade: object | None = None,
) -> AuthoringWorkflowController:
    """Wire A1-A7 handlers to one GUI-owner A8 controller."""

    if type(session) is not ModelSession:
        raise TypeError("session must be exactly ModelSession")
    if type(authoring_bridge) is not AgentAuthoringBridge:
        raise TypeError("authoring_bridge must be AgentAuthoringBridge")
    if type(result_bridge) is not AgentResultQueryBridge:
        raise TypeError("result_bridge must be AgentResultQueryBridge")
    if next_job_name is not None and not callable(next_job_name):
        raise TypeError("next_job_name must be callable or None")
    if create_model_document is not None and not callable(create_model_document):
        raise TypeError("create_model_document must be callable or None")
    if (
        workspace_catalog_bridge is not None
        and type(workspace_catalog_bridge) is not WorkspaceCatalogBridge
    ):
        raise TypeError(
            "workspace_catalog_bridge must be WorkspaceCatalogBridge or None"
        )
    if export_facade is not None and not callable(
        getattr(export_facade, "current_workspace", None)
    ):
        raise TypeError("export_facade must expose the export facade protocol")
    export_bridge: AgentExportBridge | None = (
        None if export_facade is None else AgentExportBridge(SessionExportPort(export_facade))
    )

    binding = AuthoringHandlerBinding(authoring_bridge)
    geometry_creation = GeometryCreationHandlers(binding)
    geometry_editing = GeometryEditingHandlers(binding)
    profile_transforms = ProfileTransformHandlers(
        binding,
        geometry_editing.prepare_geometry_edit,
    )
    mesh_solve = MeshSolveHandlers(binding, next_job_name)
    model_definitions = ModelDefinitionHandlers(binding)
    workspace_results = WorkspaceResultHandlers(
        binding,
        result_bridge,
        workspace_catalog_bridge=workspace_catalog_bridge,
        create_model_document=create_model_document,
        export_bridge=export_bridge,
        export_facade=export_facade,
    )

    controller = AuthoringWorkflowController(
        binding.current_context,
        {
            "prepare_geometry_proposal": geometry_creation.prepare_geometry,
            "prepare_planar_construction_proposal": geometry_creation.prepare_planar_construction,
            "read_profile_transform_context": profile_transforms.read_profile_transform_context,
            "prepare_profile_extrusion": profile_transforms.prepare_profile_extrusion,
            "prepare_profile_revolution": profile_transforms.prepare_profile_revolution,
            "prepare_profile_path_sweep": profile_transforms.prepare_profile_path_sweep,
            "read_geometry_edit_context": geometry_editing.read_geometry_edit_context,
            "prepare_geometry_edit": geometry_editing.prepare_geometry_edit_with_diagnostics,
            "read_mesh_refinement_context": mesh_solve.read_mesh_refinement_context,
            "read_model_topology_context": mesh_solve.read_model_topology_context,
            "prepare_mesh_proposal": mesh_solve.prepare_mesh,
            "apply_model_definition": model_definitions.apply_model_definition,
            "run_native_preflight": mesh_solve.run_preflight,
            "prepare_solve_proposal": mesh_solve.prepare_solve,
            "request_project_save": workspace_results.request_project_save,
            **(
                {}
                if create_model_document is None
                else {"create_native_model_document": workspace_results.create_native_model_document}
            ),
            "read_deletable_objects": model_definitions.read_deletable_objects,
            "prepare_delete_proposal": model_definitions.prepare_delete,
            "read_editable_model_objects": model_definitions.read_editable_objects,
            "edit_model_object": model_definitions.edit_model_object,
            "read_accepted_result_catalog": workspace_results.read_catalog,
            "read_analysis_run_catalog": workspace_results.read_analysis_run_catalog,
            "read_geometry_feature_catalog": geometry_editing.read_geometry_feature_catalog,
            "query_accepted_result": workspace_results.query_result,
            "compare_accepted_results": workspace_results.compare_results,
            **(
                {}
                if workspace_catalog_bridge is None
                else {"read_workspace_documents": workspace_results.read_workspace_documents}
            ),
            **(
                {}
                if export_bridge is None
                else {
                    EXPORT_CSV_TOOL_NAME: workspace_results.export_result_csv,
                    RESULT_DISPLAY_CONTEXT_TOOL_NAME: (
                        workspace_results.read_result_display_context
                    ),
                }
            ),
        },
        workspace_result_inventory=(
            None if workspace_catalog_bridge is None else workspace_results.workspace_result_inventory
        ),
        workspace_export_available=(
            None if export_bridge is None else workspace_results.workspace_export_available
        ),
    )
    if authoring_bridge.context is not None:
        controller.observe_binding(binding.current_context())
    return controller


__all__ = [
    "AppliedPatchRecord",
    "AppliedPatchState",
    "FeatureProposalUndoRecord",
    "AgentPreflightRecord",
    "AgentPreflightState",
    "AgentPreflightTaskRequest",
    "AgentGeometryMutation",
    "AgentMeshTaskRequest",
    "AgentProposalPreview",
    "AgentSolveTaskRequest",
    "AgentAuthoringBridge",
    "AgentProposal",
    "AgentResultQueryBridge",
    "AuthoringWorkflowController",
    "AuthoringWorkflowStage",
    "BridgeReceipt",
    "ProposalState",
    "SessionResultQueryPort",
    "SessionGeometryAuthoringPort",
    "SessionExportPort",
    "authoring_context_from_snapshot",
    "create_session_authoring_workflow_controller",
]
