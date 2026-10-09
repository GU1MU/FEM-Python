"""Live GUI binding and proposal envelopes shared by tool handler groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from fem.application import ModelSession
from fem_agent.authoring import (
    AgentProposal,
    AuthoringContext,
    AuthoringContractError,
    AuthoringPort,
    ModelPatch,
    ProposalState,
)
from fem_agent.authoring_payload import provider_safe_authoring_payload
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.authoring_values import AuthoringToolOutcome
from ..agent_authoring_values import AgentPreflightState, AgentProposalPreview


class ProposalReceipt(Protocol):
    @property
    def state(self) -> ProposalState: ...


class AppliedPatchReceipt(Protocol):
    @property
    def patch(self) -> ModelPatch: ...

    @property
    def undo_available(self) -> bool: ...


class PreflightReceipt(Protocol):
    @property
    def request_id(self) -> str: ...

    @property
    def state(self) -> AgentPreflightState: ...

    @property
    def message(self) -> str: ...


class AuthoringHandlerBridge(Protocol):
    @property
    def context(self) -> AuthoringContext | None: ...

    @property
    def port(self) -> AuthoringPort: ...

    def register_proposal(
        self,
        proposal: AgentProposal,
        *,
        detached_preview: AgentProposalPreview | None = None,
    ) -> ProposalReceipt: ...

    def apply_automatic_patch(self, patch: ModelPatch) -> AppliedPatchReceipt: ...

    def request_preflight(self, step_name: str) -> PreflightReceipt: ...


def envelope(
    controller: AuthoringWorkflowController,
    prefix: str,
) -> dict[str, object]:
    return controller.invocation_metadata(prefix)


class AuthoringHandlerBinding:
    """Resolve each invocation against the bridge's current GUI binding."""

    def __init__(self, bridge: AuthoringHandlerBridge) -> None:
        self.bridge = bridge

    def current_context(self) -> AuthoringContext:
        context = self.bridge.context
        if context is None:
            raise AuthoringContractError("there is no current authoring binding")
        return context

    def current_session(self) -> ModelSession:
        """Resolve the Session through the rebindable authoring port."""

        port = self.bridge.port
        session_reader = getattr(port, "session", None)
        if type(session_reader) is not ModelSession:
            raise AuthoringContractError("there is no current model Session")
        return session_reader

    def planned_geometry_edit_mode(self) -> str:
        reader = getattr(self.bridge.port, "geometry_edit_mode", None)
        return "in_place" if not callable(reader) else str(reader())

    def proposal_outcome(
        self,
        proposal: AgentProposal,
        *,
        summary: str,
        impact: str,
        confirm_label: str,
        extra_data: Mapping[str, object] | None = None,
        detached_preview: AgentProposalPreview | None = None,
    ) -> AuthoringToolOutcome:
        data = {
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.proposal_hash,
            "state": ProposalState.PENDING_CONFIRMATION.value,
            "proposal_view": {
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.proposal_hash,
                "proposal_kind": proposal.proposal_kind.value,
                "title": str(proposal.display_summary.get("title", summary)),
                "summary": summary,
                "impact": impact,
                "confirm_label": confirm_label,
                "target_document_id": proposal.target_document_id,
                "target_session_id": proposal.target_session_id,
                "base_session_revision": proposal.base_session_revision,
            },
            "continuation_checkpoint": {
                "session_id": proposal.agent_session_id,
                "source_turn_id": proposal.turn_id,
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.proposal_hash,
                "model_revision": proposal.base_session_revision,
                "proposal_kind": proposal.proposal_kind.value,
            },
        }
        if extra_data is not None:
            data.update(dict(extra_data))
        provisional = AuthoringToolOutcome(summary, data)
        provider_safe_authoring_payload(provisional.data)
        receipt = self.bridge.register_proposal(
            proposal,
            detached_preview=detached_preview,
        )
        if receipt.state is ProposalState.PENDING_CONFIRMATION:
            return provisional
        final = AuthoringToolOutcome(
            summary,
            {**data, "state": receipt.state.value},
        )
        provider_safe_authoring_payload(final.data)
        return final
