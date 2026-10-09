from __future__ import annotations

import subprocess
import sys

from fem.application import ModelSession, UnitContext
from fem.geometry import MovedGeometry, RectangleGeometry
from fem_agent.authoring import ProposalState
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)


def test_handler_modules_import_without_loading_authoring_factory() -> None:
    script = """
import importlib
import sys

for name in (
    'binding',
    'geometry_creation',
    'geometry_editing',
    'geometry_support',
    'mesh_solve',
    'model_definitions',
    'profile_transforms',
    'workspace_results',
):
    importlib.import_module('fem_gui.agent_handlers.' + name)
assert 'fem_gui.agent_authoring' not in sys.modules

from fem_gui import agent_authoring, agent_authoring_values
assert agent_authoring.AgentProposalPreview is agent_authoring_values.AgentProposalPreview
assert agent_authoring.AgentPreflightState is agent_authoring_values.AgentPreflightState
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_geometry_edit_after_rebinding_targets_and_mutates_current_session() -> None:
    first = ModelSession()
    first.create_native_project_with_first_part(
        "First model",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("First geometry", 1.0, 1.0),
        part_name="First part",
    )
    second = ModelSession()
    second.create_native_project_with_first_part(
        "Second model",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("Second geometry", 3.0, 2.0),
        part_name="Second part",
    )
    second.add_native_part(RectangleGeometry("Additional geometry", 1.0, 1.0))
    first_before = first.snapshot()
    second_before = second.snapshot()

    port = SessionGeometryAuthoringPort(first, lambda: None)
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(first_before, document_id="first-document")
    result_port = SessionResultQueryPort(first)
    controller = create_session_authoring_workflow_controller(
        first,
        bridge,
        AgentResultQueryBridge(result_port),
    )
    port.bind_session(second, idle=lambda: True)
    result_port.bind_session(second, idle=lambda: True)
    bridge.bind_snapshot(second_before, document_id="second-document")
    controller.reset_for_binding()
    controller.observe_binding(bridge.context)

    outcome = controller.dispatch(
        "prepare_geometry_edit",
        {"part_id": "P1", "edit": {"operation": "translate", "dx": 4.0, "dy": 5.0}},
        ToolExecutionContext("agent-rebinding", 0, "rebound-geometry-edit"),
    )

    assert outcome.ok, outcome.to_json()
    view = outcome.data["proposal_view"]
    assert view["target_document_id"] == "second-document"
    assert view["target_session_id"] == second.session_id
    assert view["base_session_revision"] == second_before.session_revision
    assert outcome.summary == "Translate part Second part"
    assert first.snapshot() == first_before
    assert second.snapshot() == second_before

    receipt = bridge.accept_from_gui_control(outcome.data["proposal_id"])

    assert receipt.state is ProposalState.SUCCEEDED
    assert first.snapshot() == first_before
    after = second.snapshot()
    assert after.session_revision == second_before.session_revision + 1
    recipe = after.parts[0].geometry_recipe
    assert type(recipe) is MovedGeometry
    assert recipe.base == second_before.parts[0].geometry_recipe
    assert (recipe.dx, recipe.dy, recipe.dz) == (4.0, 5.0, 0.0)
    assert after.parts[1] == second_before.parts[1]
