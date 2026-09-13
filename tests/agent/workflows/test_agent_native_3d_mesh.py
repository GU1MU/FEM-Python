from __future__ import annotations

import pytest

from fem.application import ModelSession, UnitContext
from fem.application.preprocessing import generate_fem_model
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
)
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import ProposalState
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    authoring_context_from_snapshot,
    create_session_authoring_workflow_controller,
)

from tests.helpers.agent_native_3d_workflows import (
    _path_sweep,
    _dispatch,
    _record_mesh_requirements,
    _production_controller,
)


def _cut_solid() -> BooleanGeometry:
    return BooleanGeometry(
        "Boolean cut",
        "cut",
        BoxGeometry("Target", 2.0, 2.0, 1.0),
        MovedGeometry(CylinderGeometry("Tool", 0.35, 1.0), 1.0, 1.0, 0.0),
    )


def test_mesh_intent_schema_is_strict_and_hex_is_capability_gated() -> None:
    automatic = MeshIntent("tetrahedron", 2, auto_level=4)
    payload = automatic.to_dict()

    assert MeshIntent.from_dict(payload) == automatic
    assert automatic.to_auto_mesh_spec().cell_shape == "tet"
    assert MeshIntent("hexahedron", 1, global_size=0.25).to_mesh_settings(
        BoxGeometry("Structured", 1.0, 1.0, 1.0)
    ).cell_shape == "hexahedron"
    with pytest.raises(ValueError, match="mesh.hex.unsupported-shape"):
        MeshIntent("hexahedron", 1, global_size=0.25).to_mesh_settings(
            _path_sweep()
        )
    with pytest.raises(ValueError, match="fields do not match"):
        MeshIntent.from_dict({**payload, "fallback": "tetrahedron"})
    with pytest.raises(ValueError, match="schema 1.1"):
        MeshIntent(
            "tetrahedron",
            1,
            global_size=0.25,
            schema_version="1.1",
        )


@pytest.mark.usefixtures("real_gmsh")
def test_runtime_exposes_solid_mesh_requirements_for_derived_geometry() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent solid requirements",
        UnitContext("mm", "N", "MPa"),
        _path_sweep(),
        part_name="Swept member",
    )
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    requirement_tool = next(
        definition
        for definition in controller.definitions
        if definition.name == "set_authoring_requirements"
    )
    properties = requirement_tool.parameters["properties"]["requirements"][
        "properties"
    ]

    assert authoring_context_from_snapshot(session.snapshot()).parts[0].dimension == 3
    assert properties["mesh_cell_shape"]["enum"] == [
        "tetrahedron",
        "hexahedron",
    ]
    assert properties["mesh_order"]["enum"] == [1, 2]


@pytest.mark.parametrize(
    "recipe",
    (
        ExtrudedGeometry(RectangleGeometry("Block", 1.0, 0.8), 0.6),
        ExtrudedGeometry(
            PlateWithHoleGeometry("Perforated", 2.0, 1.5, 1.0, 0.75, 0.25),
            0.5,
        ),
        _path_sweep(),
        _cut_solid(),
    ),
    ids=("extruded-block", "extruded-hole", "path-sweep", "boolean-cut"),
)
def test_feature_results_generate_pure_tet4(real_gmsh, recipe) -> None:
    del real_gmsh

    model = generate_fem_model(
        recipe,
        MeshIntent("tetrahedron", 1, global_size=0.35).to_mesh_settings(recipe),
    )

    assert model.mesh.nodes
    assert model.mesh.elements
    assert {element.type for element in model.mesh.elements} == {"Tet4"}


@pytest.mark.parametrize(("order", "expected"), ((1, "Tet4"), (2, "Tet10")))
def test_tet_intent_round_trip_generates_requested_order(
    real_gmsh,
    order: int,
    expected: str,
) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(RectangleGeometry("Ordered Tet", 1.0, 0.8), 0.6)
    intent = MeshIntent("tetrahedron", order, global_size=0.35)

    restored = MeshIntent.from_dict(intent.to_dict())
    model = generate_fem_model(recipe, restored.to_mesh_settings(recipe))

    assert restored == intent
    assert intent.to_dict()["schema_version"] == "1.2"
    assert intent.to_auto_mesh_spec() is None
    assert {element.type for element in model.mesh.elements} == {expected}


def test_gui_bridge_commits_tet_intent_and_model_atomically(real_gmsh) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(RectangleGeometry("Agent block", 1.0, 0.8), 0.6)
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent Tet model",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Extruded part",
    )
    before = session.snapshot()
    proposal = create_mesh_proposal(
        proposal_id="proposal-solid-mesh-tet",
        agent_session_id="agent-solid-mesh",
        turn_id="turn-solid-mesh",
        source_tool_call_ids=("call-solid-mesh",),
        context=authoring_context_from_snapshot(before),
        draft_revision=5,
        part_id="P1",
        mesh_intent=MeshIntent("tetrahedron", 1, global_size=0.35),
    )
    requests = []
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(
            session,
            lambda: None,
            lambda request: requests.append(request) is None,
        )
    )
    bridge.bind_snapshot(before)
    bridge.register_proposal(proposal)

    running = bridge.accept_from_gui_control(proposal.proposal_id)
    assert running.state.value == "running"
    assert session.snapshot() == before
    task = requests[0].task
    candidate = generate_fem_model(task)
    delta = bridge.port.accept_mesh_result(proposal.proposal_id, candidate)

    assert delta.accepted
    assert bridge.state(proposal.proposal_id).value == "succeeded"
    assert session.snapshot().parts[0].mesh_settings == MeshSettings(
        0.35,
        cell_shape="tetrahedron",
        strict_cell_shape=True,
    )
    assert {element.type for element in session.snapshot().model.mesh.elements} == {
        "Tet4"
    }


def test_unsupported_hex_is_diagnostic_and_session_atomic(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = _path_sweep()
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Unsupported Hex",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Swept member",
    )
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.35, cell_shape="tetrahedron", strict_cell_shape=True),
    )
    old_mesh_task = session.prepare_mesh_generation()
    assert session.accept_generated_model(
        old_mesh_task.token,
        generate_fem_model(old_mesh_task),
    ).accepted
    before = session.snapshot()
    started = []
    controller, bridge = _production_controller(
        session,
        start_mesh_task=lambda request: started.append(request) or True,
    )
    _record_mesh_requirements(
        controller,
        session,
        cell_shape="hexahedron",
        order=1,
        global_size=0.3,
    )

    failed = _dispatch(
        controller,
        session,
        "prepare_mesh_proposal",
        {},
        "unsupported-hex-proposal",
    )

    assert not failed.ok
    assert "mesh.hex.unsupported-shape" in failed.diagnostics[0].message
    assert failed.data["diagnostic_code"] == "mesh.hex.unsupported-shape"
    assert "proposal_id" not in failed.data
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    assert bridge._records == {}
    assert started == []
    assert session.snapshot() == before

    # The GUI acceptance boundary repeats the same check for a tampered or
    # otherwise externally retained proposal.
    proposal = create_mesh_proposal(
        proposal_id="proposal-solid-mesh-tampered-hex",
        agent_session_id="agent-solid-mesh",
        turn_id="turn-solid-mesh",
        source_tool_call_ids=("call-solid-mesh",),
        context=authoring_context_from_snapshot(before),
        draft_revision=5,
        part_id="P1",
        mesh_intent=MeshIntent("hexahedron", 1, global_size=0.3),
    )
    bridge.register_proposal(proposal)
    receipt = bridge.accept_from_gui_control(proposal.proposal_id)

    assert receipt.state is ProposalState.FAILED
    assert "mesh.hex.unsupported-shape" in receipt.message
    assert started == []
    assert session.snapshot() == before
