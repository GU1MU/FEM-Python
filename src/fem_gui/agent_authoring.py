"""A1 GUI boundary for detached FEM Agent authoring.

The adapter reads a detached session snapshot into bounded DTOs.  The bridge
owns only those DTOs and an ``AuthoringPort``; it never stores or mutates a
``ModelSession``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Protocol

from fem.application import ModelSession, UnitContext
from fem.application.changes import SessionDelta

from fem_agent.authoring import (
    AgentProposal,
    AuthoringAuthorizationError,
    AuthoringContext,
    AuthoringContractError,
    AuthoringPort,
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
    RequirementLedger,
    RequirementReview,
    UnitContextSummary,
)
from fem_agent.definition_authoring import (
    inverse_operations_for_snapshot,
    require_non_destructive_a4_batch,
    scoped_definition_batch_from_operations,
)
from fem_agent.geometry_authoring import geometry_recipe_from_payload
from fem_agent.mesh_authoring import MeshIntent
from fem_agent.naming import NameAllocator


class _SessionSnapshot(Protocol):
    session_id: str
    session_revision: int
    source_kind: str | None
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
    displayed_result: object | None
    mesh_current: bool
    unit_context: object | None


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
    nodes = getattr(model, "nodes", None)
    elements = getattr(model, "elements", None)
    return _bounded_count(nodes), _bounded_count(elements)


def authoring_context_from_snapshot(
    snapshot: _SessionSnapshot,
) -> AuthoringContext:
    """Copy a session projection into a bounded, provider-safe DTO."""

    session_id = str(snapshot.session_id)
    source_kind = "blank" if snapshot.source_kind is None else str(snapshot.source_kind)
    supported = source_kind in {"blank", "native"}
    binding = LocalModelBinding(
        document_id=f"document:{session_id}",
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
                    recipe_kind=(None if recipe is None else type(recipe).__name__),
                    dimension=_recipe_dimension(recipe),
                    suppressed=bool(getattr(part, "suppressed", False)),
                )
            )

    node_count, element_count = _model_counts(snapshot.artifact)
    mesh_present = snapshot.artifact is not None
    validation_status = "not_run"
    validations = getattr(snapshot, "validations", {})
    if _bounded_count(validations):
        records = tuple(validations.values())  # type: ignore[union-attr]
        validation_status = (
            "passed"
            if records and all(bool(getattr(item, "passed", False)) for item in records)
            else "blocked"
        )
    runs = tuple(getattr(snapshot, "runs", ()))
    job_status = "idle"
    if any(
        str(getattr(item, "status", "")).casefold().endswith("running") for item in runs
    ):
        job_status = "running"
    elif runs:
        job_status = (
            str(getattr(runs[-1], "status", "completed")).split(".")[-1].casefold()
        )

    blocked_reason = None if supported else "V1 只绑定空白或 native 文档"
    capabilities = (
        CapabilitySummary("read_authoring_context", supported, blocked_reason),
        CapabilitySummary("review_requirements", supported, blocked_reason),
        CapabilitySummary("build_agent_draft", supported, blocked_reason),
        CapabilitySummary("present_static_proposal", supported, blocked_reason),
        CapabilitySummary("draft_native_geometry", supported, blocked_reason),
        CapabilitySummary("commit_native_geometry", supported, blocked_reason),
        CapabilitySummary(
            "draft_mesh_intent",
            supported and source_kind == "native",
            (
                None
                if supported and source_kind == "native"
                else "网格意图需要 native 项目"
            ),
        ),
        CapabilitySummary(
            "request_mesh_proposal",
            supported and source_kind == "native",
            (
                None
                if supported and source_kind == "native"
                else "网格提案需要 native 项目"
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
            current=bool(getattr(snapshot, "mesh_current", False)),
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
class _GuiControlAuthorization:
    proposal_id: str
    action: str
    nonce: int


@dataclass(frozen=True, slots=True)
class AgentMeshTaskRequest:
    proposal_id: str
    proposal_hash: str
    task: object


class SessionGeometryAuthoringPort:
    """A2/A3 port for atomic geometry writes and detached mesh tasks."""

    def __init__(
        self,
        session: ModelSession,
        refresh_projection: Callable[[], None],
        start_mesh_task: Callable[[AgentMeshTaskRequest], bool] | None = None,
        apply_definition_delta: Callable[[SessionDelta], None] | None = None,
    ) -> None:
        if type(session) is not ModelSession:
            raise TypeError("session must be ModelSession")
        if not callable(refresh_projection):
            raise TypeError("refresh_projection must be callable")
        self._session = session
        self._refresh_callback = refresh_projection
        self._start_mesh_task = start_mesh_task
        if apply_definition_delta is not None and not callable(
            apply_definition_delta
        ):
            raise TypeError("apply_definition_delta must be callable or None")
        self._apply_definition_delta = apply_definition_delta
        self._context: AuthoringContext | None = None
        self._records: dict[str, ProposalPortRecord] = {}
        self._mesh_tasks: dict[str, object] = {}
        self._patch_records: dict[str, AppliedPatchRecord] = {}
        self._record_listener: Callable[[ProposalPortRecord], None] | None = None

    def set_record_listener(
        self,
        callback: Callable[[ProposalPortRecord], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("record listener must be callable")
        self._record_listener = callback

    def set_context(self, context: AuthoringContext) -> None:
        if type(context) is not AuthoringContext:
            raise AuthoringContractError("context must be AuthoringContext")
        self._context = context

    def present(self, proposal: AgentProposal) -> ProposalPortRecord:
        if proposal.proposal_kind not in {
            ProposalKind.GEOMETRY,
            ProposalKind.MESH,
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
            }
        )
        mesh_valid = (
            proposal.proposal_kind is ProposalKind.MESH
            and tuple(item.kind for item in proposal.operations)
            == (
                OperationKind.SET_PART_MESH_INTENT,
                OperationKind.REQUEST_MESH,
            )
        )
        destructive_valid = (
            proposal.proposal_kind is ProposalKind.DESTRUCTIVE_EDIT
            and tuple(item.kind for item in proposal.operations)
            == (
                OperationKind.UPSERT_NAMED_REGIONS,
                OperationKind.UPSERT_MODEL_DEFINITIONS,
            )
        )
        if not geometry_valid and not mesh_valid and not destructive_valid:
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
        if (
            proposal.target_session_id != current_id
            or proposal.target_document_id != f"document:{current_id}"
            or proposal.base_session_revision != current_revision
        ):
            raise AuthoringContractError("authoring proposal target is stale")
        if proposal.proposal_kind is ProposalKind.MESH:
            return self._accept_mesh(record)
        if proposal.proposal_kind is ProposalKind.DESTRUCTIVE_EDIT:
            batch = scoped_definition_batch_from_operations(
                proposal.operations,
                self._session.snapshot(),
                base_session_revision=proposal.base_session_revision,
            )
            delta = self._session.apply_scoped_definition_batch(batch)
            self._project_definition_delta(delta)
            succeeded = replace(record, state=ProposalState.SUCCEEDED)
            self._records[proposal_id] = succeeded
            return succeeded
        operation = proposal.operations[0]
        parameters = operation.parameters
        recipe = geometry_recipe_from_payload(parameters["recipe"])
        snapshot = self._session.snapshot()
        part_name = NameAllocator(
            {"parts": (part.name for part in snapshot.parts)}
        ).require_next(
            "parts",
            "部件",
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
                    "模型",
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
        self._records[proposal_id] = succeeded
        return succeeded

    def apply_patch(self, patch: ModelPatch) -> AppliedPatchRecord:
        """Apply one non-destructive A4 patch and retain its exact inverse."""

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
        if (
            patch.target_session_id != snapshot.session_id
            or patch.target_document_id
            != f"document:{snapshot.session_id}"
            or patch.base_session_revision != snapshot.session_revision
        ):
            raise AuthoringContractError("automatic patch target is stale")
        if any(bool(getattr(run, "has_result", False)) for run in snapshot.runs):
            raise AuthoringAuthorizationError(
                "a result-invalidating edit requires GUI confirmation"
            )
        inverse_operations = inverse_operations_for_snapshot(snapshot)
        batch = scoped_definition_batch_from_operations(
            patch.operations,
            snapshot,
            base_session_revision=patch.base_session_revision,
        )
        require_non_destructive_a4_batch(snapshot, batch)
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
            },
            display_summary={
                "title": "撤销本次 Agent 修改",
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
            and self._session.session_id
            == record.inverse_patch.target_session_id
            and self._session.session_revision
            == record.inverse_patch.base_session_revision
        )

    def undo_patch(self, patch_id: str) -> AppliedPatchRecord:
        """Apply an inverse exactly once while its post revision is current."""

        try:
            record = self._patch_records[str(patch_id)]
        except KeyError as error:
            raise AuthoringContractError(
                "automatic patch is not registered"
            ) from error
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
        snapshot = self._session.snapshot()
        inverse = record.inverse_patch
        batch = scoped_definition_batch_from_operations(
            inverse.operations,
            snapshot,
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
            raise AuthoringContractError(
                "automatic patch is not registered"
            ) from error

    def _project_definition_delta(self, delta: SessionDelta) -> None:
        if self._apply_definition_delta is not None:
            self._apply_definition_delta(delta)

    def _accept_mesh(self, record: ProposalPortRecord) -> ProposalPortRecord:
        if self._start_mesh_task is None:
            raise AuthoringContractError(
                "mesh proposal execution is not configured"
            )
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
        self._mesh_tasks[proposal.proposal_id] = task
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
            self._session.terminate_agent_mesh_task(task.token, str(error))
            self._mesh_tasks.pop(proposal.proposal_id, None)
            raise
        if not started:
            self._session.terminate_agent_mesh_task(
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

    def accept_mesh_result(
        self,
        proposal_id: str,
        model: object,
    ) -> SessionDelta:
        record = self._records.get(proposal_id)
        if record is None or record.state is not ProposalState.RUNNING:
            raise AuthoringAuthorizationError(
                "mesh result requires a running proposal"
            )
        task = self._mesh_tasks[proposal_id]
        delta = self._session.accept_agent_generated_model(task.token, model)
        if delta.accepted:
            self.complete_mesh(
                proposal_id,
                ProposalState.SUCCEEDED,
                "网格意图和生成模型已原子提交",
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
        task = self._mesh_tasks[proposal_id]
        delta = self._session.terminate_agent_mesh_task(task.token, message)
        self.complete_mesh(proposal_id, state, message)
        return delta

    def refresh_projection(self) -> None:
        self._refresh_callback()

    def reject(self, proposal_id: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        rejected = replace(record, state=ProposalState.REJECTED)
        self._records[proposal_id] = rejected
        return rejected

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
        self._gui_thread_id = threading.get_ident()
        self._patch_listener: Callable[[AppliedPatchRecord], None] | None = None
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

    def bind_snapshot(
        self,
        snapshot: _SessionSnapshot,
    ) -> tuple[str, ...]:
        return self.bind_context(authoring_context_from_snapshot(snapshot))

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
            if record.state is ProposalState.PENDING_CONFIRMATION:
                stale = self._port.stale(
                    proposal_id,
                    "绑定文档、session 或 revision 已改变",
                )
                self._records[proposal_id] = stale
                stale_ids.append(proposal_id)
        return tuple(stale_ids)

    def register_proposal(self, proposal: AgentProposal) -> BridgeReceipt:
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
        self._require_live_target(proposal)
        record = self._port.present(proposal)
        self._records[proposal.proposal_id] = record
        self._idempotency[proposal.idempotency_key] = proposal.proposal_id
        return self._receipt(record)

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
            return self._receipt(failed)
        self._records[proposal_id] = accepted
        if (
            accepted.state is ProposalState.SUCCEEDED
            and accepted.proposal.proposal_kind
            is ProposalKind.GEOMETRY
        ):
            refresh = getattr(self._port, "refresh_projection", None)
            if callable(refresh):
                refresh()
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
        return self._receipt(rejected)

    def accept_from_gui_control(self, proposal_id: str) -> BridgeReceipt:
        self._require_gui_thread()
        authorization = self._issue_gui_authorization(
            proposal_id,
            "accept",
        )
        return self.accept_proposal(proposal_id, authorization)

    def reject_from_gui_control(self, proposal_id: str) -> BridgeReceipt:
        self._require_gui_thread()
        authorization = self._issue_gui_authorization(
            proposal_id,
            "reject",
        )
        return self.reject_proposal(proposal_id, authorization)

    def confirm_requirement_review_from_gui(
        self,
        ledger: RequirementLedger,
        review: RequirementReview,
    ) -> RequirementReview:
        self._require_gui_thread()
        return ledger._confirm_review_from_gui(review)

    def reject_requirement_review_from_gui(
        self,
        ledger: RequirementLedger,
        review: RequirementReview,
    ) -> RequirementReview:
        self._require_gui_thread()
        return ledger._reject_review_from_gui(review)

    def state(self, proposal_id: str) -> ProposalState:
        try:
            return self._records[proposal_id].state
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc

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
        if current is None or current.proposal.proposal_hash != record.proposal.proposal_hash:
            raise AuthoringContractError(
                "port lifecycle update does not match a registered proposal"
            )
        self._records[record.proposal.proposal_id] = record

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


__all__ = [
    "AppliedPatchRecord",
    "AppliedPatchState",
    "AgentMeshTaskRequest",
    "AgentAuthoringBridge",
    "BridgeReceipt",
    "SessionGeometryAuthoringPort",
    "authoring_context_from_snapshot",
]
