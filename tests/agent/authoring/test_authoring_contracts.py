from __future__ import annotations

from dataclasses import replace

import pytest

from fem_agent.authoring import (
    AgentDraft,
    AgentProposal,
    AuthoringAuthorizationError,
    AuthoringContext,
    AuthoringContractError,
    CapabilitySummary,
    ClarificationRequiredError,
    DefinitionSummary,
    FakeAuthoringPort,
    LocalModelBinding,
    MeshSummary,
    ModelOperation,
    ModelPatch,
    OperationKind,
    PartSummary,
    ProposalKind,
    ProposalState,
    RequirementLedger,
    RequirementStatus,
)


def _binding(*, revision: int = 4) -> LocalModelBinding:
    return LocalModelBinding(
        document_id="document-1",
        session_id="session-1",
        session_revision=revision,
        source_kind="native",
        supported=True,
    )


def _operation() -> ModelOperation:
    return ModelOperation(
        OperationKind.ADD_NATIVE_PART,
        {
            "part_name": "部件-偏心孔板",
            "recipe": {
                "kind": "PlateWithHoleGeometry",
                "dimension": 2,
            },
        },
    )


def _envelope_values(envelope_id: str) -> dict[str, object]:
    return {
        "proposal_id": envelope_id,
        "agent_session_id": "agent-session-1",
        "turn_id": "turn-1",
        "source_tool_call_ids": ("call-1",),
        "target_document_id": "document-1",
        "target_session_id": "session-1",
        "base_session_revision": 4,
        "draft_revision": 2,
        "operations": (_operation(),),
        "preconditions": {"source_kind": "native"},
        "expected_changes": {"part_count_delta": 1},
        "invalidation_impact": {"mesh": False, "results": False},
        "display_summary": {
            "title": "加入偏心孔板",
            "summary": "A1 静态提案，不修改当前模型",
        },
    }


def _proposal(proposal_id: str = "proposal-1") -> AgentProposal:
    return AgentProposal.create(
        proposal_kind=ProposalKind.GEOMETRY,
        **_envelope_values(proposal_id),
    )


def test_authoring_context_is_bounded_and_provider_safe() -> None:
    context = AuthoringContext(
        binding=_binding(),
        model_name="模型-孔板",
        active_part_id="P1",
        parts=(
            PartSummary(
                "P1",
                "部件-偏心孔板",
                "PlateWithHoleGeometry",
                2,
                False,
            ),
        ),
        mesh=MeshSummary(True, True, 120, 210),
        definitions=DefinitionSummary(3, 1, 1, 1, 2),
        validation_status="passed",
        job_status="idle",
        result_available=False,
        capabilities=(CapabilitySummary("read_context", True),),
    )

    payload = context.to_provider_dict()
    flattened = repr(payload)

    assert payload["binding"]["session_revision"] == 4
    assert payload["mesh"] == {
        "present": True,
        "current": True,
        "node_count": 120,
        "element_count": 210,
    }
    assert "nodes" not in payload
    assert "elements" not in payload
    assert "ModelSession" not in flattened
    assert "C:\\" not in flattened


def test_requirement_review_is_the_only_confirmation_gate() -> None:
    ledger = RequirementLedger()
    ledger.record(
        "geometry.dimension",
        field_type="integer",
        stage="geometry",
        value=2,
        source_turn_id="turn-1",
    )
    ledger.record(
        "geometry.width",
        field_type="number",
        stage="geometry",
        value=200.0,
        source_turn_id="turn-1",
        dependencies=("geometry.dimension",),
    )

    with pytest.raises(ClarificationRequiredError) as missing:
        ledger.require_confirmed(
            "geometry",
            ("geometry.dimension", "geometry.width"),
        )
    assert missing.value.code == "clarification_required"

    with pytest.raises(AuthoringAuthorizationError):
        ledger.record(
            "geometry.height",
            field_type="number",
            stage="geometry",
            value=100.0,
            source_turn_id="turn-1",
            status=RequirementStatus.CONFIRMED,
        )

    review = ledger.create_review(
        "review-1",
        ("geometry.dimension", "geometry.width"),
    )
    confirmed = ledger._confirm_review_from_gui(review)

    assert confirmed.status.value == "confirmed"
    assert [item.key for item in ledger.require_confirmed(
        "geometry",
        ("geometry.dimension", "geometry.width"),
    )] == ["geometry.dimension", "geometry.width"]

    ledger.record(
        "geometry.dimension",
        field_type="integer",
        stage="geometry",
        value=3,
        source_turn_id="turn-2",
    )
    width = next(
        item for item in ledger.entries if item.key == "geometry.width"
    )
    assert width.status is RequirementStatus.INVALIDATED


def test_agent_draft_has_binding_identity_and_monotonic_revision() -> None:
    draft = AgentDraft.create(
        draft_id="draft-1",
        agent_session_id="agent-session-1",
        binding=_binding(),
        confirmed_requirements={"dimension": 2},
        candidate_summary={"recipe_kind": "PlateWithHoleGeometry"},
    )
    revised = draft.revise(
        confirmed_requirements={"dimension": 2, "width": 200.0},
        candidate_summary={
            "recipe_kind": "PlateWithHoleGeometry",
            "feature_count": 1,
        },
        pending_proposal_ids=("proposal-1",),
    )

    assert draft.draft_revision == 0
    assert revised.draft_revision == 1
    assert revised.base_document_id == draft.base_document_id
    assert revised.confirmed_requirements_hash != (
        draft.confirmed_requirements_hash
    )
    assert revised.candidate_model_hash != draft.candidate_model_hash


def test_patch_and_proposal_hashes_are_strict_and_idempotent() -> None:
    proposal = _proposal()
    replay_with_new_id = _proposal("proposal-2")
    values = _envelope_values("unused")
    values["patch_id"] = "patch-1"
    values.pop("proposal_id")
    patch = ModelPatch.create(**values)

    assert AgentProposal.from_dict(proposal.to_dict()) == proposal
    assert ModelPatch.from_dict(patch.to_dict()) == patch
    assert proposal.idempotency_key == replay_with_new_id.idempotency_key
    assert proposal.proposal_hash != replay_with_new_id.proposal_hash
    assert len(patch.patch_hash) == 64

    tampered = proposal.to_dict()
    tampered["base_session_revision"] = 5
    with pytest.raises(AuthoringContractError, match="idempotency"):
        AgentProposal.from_dict(tampered)

    unknown = proposal.to_dict()
    unknown["callback"] = "run"
    with pytest.raises(AuthoringContractError, match="unknown"):
        AgentProposal.from_dict(unknown)

    with pytest.raises(AuthoringContractError, match="not allowed"):
        ModelOperation(
            OperationKind.ADD_NATIVE_PART,
            {
                "part_name": "部件-孔板",
                "recipe": {},
                "script": "do_work()",
            },
        )
    with pytest.raises(AuthoringContractError, match="unknown parameter"):
        ModelOperation(
            OperationKind.ADD_NATIVE_PART,
            {
                "part_name": "部件-孔板",
                "recipe": {},
                "arbitrary": True,
            },
        )


def test_fake_authoring_port_has_no_side_effect_and_one_terminal_decision() -> None:
    port = FakeAuthoringPort()
    context = AuthoringContext(
        binding=_binding(),
        model_name="模型-孔板",
        active_part_id=None,
    )
    port.set_context(context)
    proposal = _proposal()

    pending = port.present(proposal)
    accepted = port.accept(proposal.proposal_id)

    assert pending.state is ProposalState.PENDING_CONFIRMATION
    assert accepted.state is ProposalState.ACCEPTED
    assert port.context is context
    with pytest.raises(AuthoringAuthorizationError):
        port.accept(proposal.proposal_id)

    rejected_proposal = _proposal("proposal-reject")
    port.present(rejected_proposal)
    assert port.reject(rejected_proposal.proposal_id).state is (
        ProposalState.REJECTED
    )

    stale_proposal = _proposal("proposal-stale")
    port.present(stale_proposal)
    assert port.stale(
        stale_proposal.proposal_id,
        "binding changed",
    ).state is ProposalState.STALE


def test_hash_rejects_nonfinite_and_local_or_executable_payloads() -> None:
    with pytest.raises(AuthoringContractError, match="finite JSON"):
        ModelOperation(
            OperationKind.REQUEST_RESULT_QUERY,
            {"query": {"value": float("nan")}},
        )
    with pytest.raises(AuthoringContractError, match="absolute paths"):
        ModelOperation(
            OperationKind.REQUEST_RESULT_QUERY,
            {"query": {"output": "D:\\private\\result.vtk"}},
        )
    with pytest.raises(AuthoringContractError, match="paths"):
        ModelOperation(
            OperationKind.REQUEST_RESULT_QUERY,
            {"query": {"source_path": "relative.txt"}},
        )

    proposal = _proposal()
    with pytest.raises(AuthoringContractError, match="hash"):
        replace(proposal, proposal_hash="0" * 64)
