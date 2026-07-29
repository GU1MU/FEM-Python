"""A1 GUI boundary for detached FEM Agent authoring.

The adapter reads a detached session snapshot into bounded DTOs.  The bridge
owns only those DTOs and an ``AuthoringPort``; it never stores or mutates a
``ModelSession``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Callable, Protocol

from fem.application import ModelSession, UnitContext

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
    OperationKind,
    PartSummary,
    ProposalKind,
    ProposalPortRecord,
    ProposalState,
    RequirementLedger,
    RequirementReview,
    UnitContextSummary,
)
from fem_agent.geometry_authoring import geometry_recipe_from_payload
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


@dataclass(frozen=True, slots=True)
class _GuiControlAuthorization:
    proposal_id: str
    action: str
    nonce: int


class SessionGeometryAuthoringPort:
    """A2 port that applies one geometry operation to the authoritative session."""

    def __init__(
        self,
        session: ModelSession,
        refresh_projection: Callable[[], None],
    ) -> None:
        if type(session) is not ModelSession:
            raise TypeError("session must be ModelSession")
        if not callable(refresh_projection):
            raise TypeError("refresh_projection must be callable")
        self._session = session
        self._refresh_callback = refresh_projection
        self._context: AuthoringContext | None = None
        self._records: dict[str, ProposalPortRecord] = {}

    def set_context(self, context: AuthoringContext) -> None:
        if type(context) is not AuthoringContext:
            raise AuthoringContractError("context must be AuthoringContext")
        self._context = context

    def present(self, proposal: AgentProposal) -> ProposalPortRecord:
        if proposal.proposal_kind is not ProposalKind.GEOMETRY:
            raise AuthoringContractError(
                "A2 session port only accepts geometry proposals"
            )
        if len(proposal.operations) != 1 or proposal.operations[0].kind not in {
            OperationKind.CREATE_NATIVE_PROJECT,
            OperationKind.ADD_NATIVE_PART,
        }:
            raise AuthoringContractError(
                "A2 geometry proposal must contain one create/add operation"
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
            raise AuthoringContractError("geometry proposal target is stale")
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
        self._authorization_nonce = 0
        self._unused_authorizations: set[_GuiControlAuthorization] = set()
        self._gui_thread_id = threading.get_ident()

    @property
    def context(self) -> AuthoringContext | None:
        return self._context

    @property
    def port(self) -> AuthoringPort:
        return self._port

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
    "AgentAuthoringBridge",
    "BridgeReceipt",
    "SessionGeometryAuthoringPort",
    "authoring_context_from_snapshot",
]
