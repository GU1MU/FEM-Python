"""Regression: numeric workspace document identities bind cleanly.

Multi-document workspaces bind stable integer document ids such as "2"
instead of the legacy synthesized ``document:<session_id>`` format.  The
stale-context guards, the stage recovery path, and the solve proposal
acceptance must all treat the numeric format as a first-class binding.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fem.application import ModelSession, UnitContext, run_static_preflight
from fem.geometry import PlateWithHoleGeometry
from fem.io.inp import read
from fem.mesh.settings import MeshSettings
from fem_agent.analysis_authoring import _require_live_context as _analysis_guard
from fem_agent.authoring import (
    AgentProposal,
    AuthoringContext,
    AuthoringContractError,
    LocalModelBinding,
)
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.definition_action_authoring import (
    _require_live_native_context as _definition_action_guard,
)
from fem_agent.definition_authoring import (
    _require_context_matches_snapshot as _definition_guard,
)
from fem_agent.incremental_authoring import (
    _require_live_native_context as _incremental_guard,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.solve_authoring import create_solve_proposal
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from tests.helpers.agent_session_fixtures import _a5_session


_FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
    / "truss2_tension.inp"
)
STEP_NAME = "Tension"

_GUARDS = (
    _definition_action_guard,
    _incremental_guard,
    _definition_guard,
    _analysis_guard,
)


def _bound_context(snapshot, *, document_id: str = "2", revision: int | None = None):
    binding = LocalModelBinding(
        document_id=document_id,
        session_id=snapshot.session_id,
        session_revision=(
            snapshot.session_revision if revision is None else revision
        ),
        source_kind="native",
        supported=True,
    )
    return AuthoringContext(
        binding=binding,
        model_name=None,
        active_part_id=None,
    )


@pytest.mark.parametrize("guard", _GUARDS)
def test_numeric_document_binding_passes_stale_context_guards(guard) -> None:
    session = _a5_session()
    snapshot = session.snapshot()
    context = _bound_context(snapshot, document_id="2")

    guard(context, snapshot)


@pytest.mark.parametrize("guard", _GUARDS)
def test_numeric_binding_still_rejects_a_mismatched_revision(guard) -> None:
    session = _a5_session()
    snapshot = session.snapshot()
    stale = _bound_context(
        snapshot,
        document_id="2",
        revision=snapshot.session_revision + 1,
    )

    with pytest.raises(Exception):
        guard(stale, snapshot)


def test_stale_stage_recovers_after_context_read_under_numeric_document() -> None:
    session = _a5_session()
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot(), document_id=2)
    assert bridge.context.binding.document_id == "2"
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    controller.invalidate_binding("document is read-only")
    assert controller.stage is AuthoringWorkflowStage.STALE

    outcome = controller.dispatch(
        "read_authoring_context",
        {},
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "rebind",
        ),
    )

    assert outcome.ok
    assert controller.stage is not AuthoringWorkflowStage.STALE


def _validated_solve_session() -> ModelSession:
    model = read(_FIXTURE)
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-桁架",
        UnitContext("mm", "N", "MPa"),
        PlateWithHoleGeometry("实体-偏心孔板", 10.0, 6.0, 6.5, 2.0, 1.0),
        part_name="部件-桁架",
    )
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(1.0, cell_shape="triangle"),
    )
    mesh = session.prepare_mesh_generation()
    assert session.accept_generated_model(mesh.token, model).accepted
    task = session.prepare_validation(STEP_NAME)
    report = run_static_preflight(task.model, task.step_name, token=task.token)
    assert report.passed
    assert session.accept_validation(task.token, report).accepted
    return session


def _solve_proposal(session: ModelSession, *, target_document_id: str | None):
    return create_solve_proposal(
        proposal_id="proposal-binding-solve",
        agent_session_id="agent-binding",
        turn_id="turn-binding-solve",
        source_tool_call_ids=("call-binding-solve",),
        snapshot=session.snapshot(),
        draft_revision=6,
        step_name=STEP_NAME,
        job_name="作业-静力1",
        target_document_id=target_document_id,
    )


def test_solve_proposal_targets_bound_numeric_document() -> None:
    session = _validated_solve_session()
    snapshot = session.snapshot()

    bound = _solve_proposal(session, target_document_id="2")
    assert bound.target_document_id == "2"

    fallback = create_solve_proposal(
        proposal_id="proposal-binding-solve-fallback",
        agent_session_id="agent-binding",
        turn_id="turn-binding-solve-fallback",
        source_tool_call_ids=("call-binding-solve-fallback",),
        snapshot=snapshot,
        draft_revision=6,
        step_name=STEP_NAME,
        job_name="作业-静力2",
    )
    assert fallback.target_document_id == f"document:{snapshot.session_id}"


def test_solve_acceptance_rejects_mismatched_document_target() -> None:
    session = _validated_solve_session()
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot(), document_id=2)
    matching = _solve_proposal(session, target_document_id="2")
    mismatched = AgentProposal.create(
        proposal_kind=matching.proposal_kind,
        proposal_id="proposal-binding-mismatch",
        agent_session_id=matching.agent_session_id,
        turn_id=matching.turn_id,
        source_tool_call_ids=matching.source_tool_call_ids,
        target_document_id="999",
        target_session_id=matching.target_session_id,
        base_session_revision=matching.base_session_revision,
        draft_revision=matching.draft_revision,
        operations=matching.operations,
        preconditions=matching.preconditions,
        expected_changes=matching.expected_changes,
        invalidation_impact=matching.invalidation_impact,
        display_summary=matching.display_summary,
    )

    receipt = bridge.register_proposal(matching)
    assert receipt.state.name in {
        "PENDING_CONFIRMATION",
        "RUNNING",
        "SUCCEEDED",
    }
    with pytest.raises(AuthoringContractError, match="target is stale"):
        bridge.register_proposal(mismatched)
