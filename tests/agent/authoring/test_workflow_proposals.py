from __future__ import annotations

import pytest

from fem_agent.authoring import ProposalState
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


@pytest.mark.parametrize(
    ("operation", "terminal", "expected"),
    [
        ("geometry", ProposalState.FAILED, AuthoringWorkflowStage.GEOMETRY_READY),
        ("mesh", ProposalState.CANCELLED, AuthoringWorkflowStage.MESH_READY),
        ("solve", ProposalState.STALE, AuthoringWorkflowStage.SOLVE_READY),
    ],
)
def test_proposal_terminal_returns_to_the_operation_boundary(
    operation,
    terminal,
    expected,
) -> None:
    controller = AuthoringWorkflowController(lambda: _context(), {})
    controller._pending_operation = operation
    controller._stage = {
        "geometry": AuthoringWorkflowStage.GEOMETRY_PENDING,
        "mesh": AuthoringWorkflowStage.MESH_PENDING,
        "solve": AuthoringWorkflowStage.SOLVE_PENDING,
    }[operation]

    controller.record_proposal_state(operation, terminal, "terminal")
    assert controller.stage is expected
    assert controller.terminal_records[-1].state == terminal.value


@pytest.mark.parametrize(
    "resume_stage",
    [
        AuthoringWorkflowStage.DEFINITIONS_READY,
        AuthoringWorkflowStage.PREFLIGHT_READY,
        AuthoringWorkflowStage.SOLVE_READY,
        AuthoringWorkflowStage.RESULTS_READY,
    ],
)
def test_remesh_terminal_returns_to_the_existing_mesh_stage(
    resume_stage: AuthoringWorkflowStage,
) -> None:
    controller = AuthoringWorkflowController(
        lambda: _context(),
        {
            "prepare_mesh_proposal": lambda _arguments, _controller: (
                AuthoringToolOutcome(
                    "Prepared.",
                    {"state": "pending_confirmation"},
                )
            ),
        },
    )
    controller._stage = resume_stage
    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-remesh",
            "requirements": _requirements_for("mesh"),
        },
        90,
    ).ok
    prepared = _dispatch(
        controller,
        "prepare_mesh_proposal",
        {},
        91,
    )

    assert prepared.ok
    assert controller.stage is AuthoringWorkflowStage.MESH_PENDING

    controller.record_proposal_state("mesh", ProposalState.REJECTED)

    assert controller.stage is resume_stage
