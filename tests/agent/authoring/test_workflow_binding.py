from __future__ import annotations

from dataclasses import replace

import pytest

from fem_agent.authoring import (
    AuthoringContext,
    DefinitionSummary,
    LocalModelBinding,
    MeshSummary,
    PartSummary,
    ProposalState,
)
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
)

from tests.helpers.agent_authoring_workflow_fixtures import (
    _context,
    _requirements_for,
    _dispatch,
)


@pytest.mark.parametrize("job_status", ["running", "queued", "cancelling"])
def test_restore_active_job_is_read_only_and_never_exposes_solve(
    job_status: str,
) -> None:
    context = AuthoringContext(
        binding=LocalModelBinding(
            "document:active-job",
            "native-active-job",
            7,
            "native",
            True,
        ),
        model_name="模型-活动作业",
        active_part_id="part-active",
        parts=(
            PartSummary(
                "part-active",
                "部件-活动作业",
                "planar_sketch",
                2,
                False,
            ),
        ),
        mesh=MeshSummary(True, True, 20, 10),
        definitions=DefinitionSummary(analysis_step_count=1),
        validation_status="passed",
        job_status=job_status,
    )
    controller = AuthoringWorkflowController(
        lambda: context,
        {
            "prepare_solve_proposal": lambda _arguments, _controller: (
                AuthoringToolOutcome("Prepared.", {"state": "pending_confirmation"})
            ),
        },
    )

    controller.observe_binding(context)

    assert controller.stage is AuthoringWorkflowStage.SOLVE_PENDING
    assert {tool.name for tool in controller.definitions} == {
        "read_authoring_context"
    }


@pytest.mark.parametrize(
    ("terminal_status", "result_available", "expected_stage"),
    [
        ("completed", True, AuthoringWorkflowStage.RESULTS_READY),
        ("failed", False, AuthoringWorkflowStage.SOLVE_READY),
    ],
)
def test_restored_job_terminal_refreshes_without_revision_change(
    terminal_status: str,
    result_available: bool,
    expected_stage: AuthoringWorkflowStage,
) -> None:
    active = AuthoringContext(
        binding=LocalModelBinding(
            "document:active-job",
            "native-active-job",
            7,
            "native",
            True,
        ),
        model_name="模型-活动作业",
        active_part_id="part-active",
        parts=(
            PartSummary(
                "part-active",
                "部件-活动作业",
                "planar_sketch",
                2,
                False,
            ),
        ),
        mesh=MeshSummary(True, True, 20, 10),
        definitions=DefinitionSummary(analysis_step_count=1),
        validation_status="passed",
        job_status="running",
    )
    current = [active]
    controller = AuthoringWorkflowController(lambda: current[0], {})
    controller.observe_binding(active)
    assert controller.stage is AuthoringWorkflowStage.SOLVE_PENDING

    terminal = replace(
        active,
        job_status=terminal_status,
        result_available=result_available,
    )
    current[0] = terminal

    assert controller.observe_binding(terminal)
    assert controller.stage is expected_stage


def test_binding_change_clears_collected_requirements() -> None:
    initial = _context()
    switched = AuthoringContext(
        binding=LocalModelBinding(
            "document:switched",
            "native-switched",
            0,
            "native",
            True,
        ),
        model_name="模型-新文档",
        active_part_id=None,
    )
    current = [initial]
    controller = AuthoringWorkflowController(lambda: current[0], {})
    controller.observe_binding(initial)
    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-before-switch",
            "requirements": _requirements_for("geometry"),
        },
        8,
    ).ok
    assert controller.collected_requirements("geometry")["length_unit"] == "mm"

    current[0] = switched
    assert controller.observe_binding(switched) is False
    assert controller.stage is AuthoringWorkflowStage.STALE
    assert controller.ledger.entries == ()
    with pytest.raises(ValueError, match="clarification_required"):
        controller.collected_requirements("geometry")


def test_binding_allows_expected_pending_revision_only() -> None:
    initial = _context()
    next_revision = AuthoringContext(
        binding=LocalModelBinding(
            initial.binding.document_id,
            initial.binding.session_id,
            1,
            initial.binding.source_kind,
            initial.binding.supported,
        ),
        model_name=initial.model_name,
        active_part_id="part-a8",
    )
    controller = AuthoringWorkflowController(lambda: initial, {})
    controller.observe_binding(initial)
    controller._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
    controller._pending_operation = "geometry"

    assert controller.observe_binding(next_revision) is True
    controller.record_proposal_state("geometry", ProposalState.SUCCEEDED)
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
