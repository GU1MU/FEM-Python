from __future__ import annotations

from dataclasses import replace

import pytest

from fem_agent.naming import NameAllocator, NamePolicy, NamePolicyError
from fem_agent.authoring import (
    AgentProposal,
    AuthoringAuthorizationError,
    AuthoringContext,
    AuthoringContractError,
    CapabilitySummary,
    DefinitionSummary,
    LocalModelBinding,
    MeshSummary,
    ModelOperation,
    ModelPatch,
    OperationKind,
    PartSummary,
    ProposalKind,
    RequirementLedger,
    RequirementStatus,
    UnitContextSummary,
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


def test_requirement_changes_invalidate_dependents() -> None:
    ledger = RequirementLedger()
    ledger.record(
        "dimension", field_type="integer", stage="geometry",
        value=2, source_turn_id="t1",
    )
    ledger.record(
        "width", field_type="number", stage="geometry", value=200.0,
        source_turn_id="t1", dependencies=("dimension",),
    )
    ledger.record(
        "dimension", field_type="integer", stage="geometry",
        value=3, source_turn_id="t2",
    )
    width = next(entry for entry in ledger.entries if entry.key == "width")
    assert width.status is RequirementStatus.INVALIDATED
    with pytest.raises(AuthoringAuthorizationError):
        ledger.record(
            "height", field_type="number", stage="geometry", value=100.0,
            source_turn_id="t2", status=RequirementStatus.CONFIRMED,
        )


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


def test_name_policy_allocates_normalized_unique_stable_names() -> None:
    policy = NamePolicy()
    allocator = NameAllocator(
        {
            "parts": (
                "Part-偏心孔板",
                "Part-偏心孔板-2",
                "Part-Ａ板",
            ),
            "models": ("Model-偏心孔板",),
        },
        policy=policy,
    )

    assert allocator.allocate("parts", "Part", "偏心孔板") == "Part-偏心孔板-3"
    assert allocator.allocate("models", "Part", "偏心孔板") == "Part-偏心孔板"
    assert allocator.allocate("parts", "Part", "A板") == "Part-A板-2"
    assert policy.compose("Edge", "固定端") == "Edge-固定端"

    assert policy.validate("部件-偏心孔板") == "部件-偏心孔板"
    legacy_limit_name = "位移-" + "长" * 93
    assert policy.validate(legacy_limit_name) == legacy_limit_name
    assert policy.compose("部件", "偏心孔板") == "Part-偏心孔板"
    assert allocator.require_next("legacy", "Part", "部件-偏心孔板") == "部件-偏心孔板"

    with pytest.raises(NamePolicyError):
        policy.compose("Part", " 偏心孔板")
    with pytest.raises(NamePolicyError):
        policy.compose("未知", "偏心孔板")
    with pytest.raises(NamePolicyError):
        policy.compose("Part", "Part-1")
    with pytest.raises(NamePolicyError):
        policy.validate("Part-Ａ板")


def test_unit_summary_keeps_explicit_not_applicable_fields() -> None:
    units = UnitContextSummary(
        length="mm",
        force="N",
        stress="MPa",
        density=None,
        acceleration=None,
        convention="N-mm-MPa",
    )

    assert units.to_dict() == {
        "length": "mm",
        "force": "N",
        "stress": "MPa",
        "density": None,
        "acceleration": None,
        "convention": "N-mm-MPa",
    }
