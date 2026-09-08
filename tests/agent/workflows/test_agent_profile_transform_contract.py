from __future__ import annotations

import pytest

from fem.application import ModelSession, UnitContext
from fem.geometry import describe_recipe_topology
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.geometry_authoring import geometry_contract_proof
from fem_agent.providers.fake import FakeProvider
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)

from tests.helpers.agent_provider_fixtures import text_response
from tests.helpers.fixtures.profile_transform_baseline import concentric_ring_fixture


pytestmark = pytest.mark.local_session


def _ring_controller() -> tuple[ModelSession, object]:
    fixture = concentric_ring_fixture()
    session = ModelSession()
    session.create_native_project_with_first_part(
        "model",
        UnitContext("mm", "N", "MPa"),
        fixture.sketch,
        part_name="part",
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
    return session, controller


def test_concentric_ring_catalog_preserves_profile_and_hole_lineage() -> None:
    fixture = concentric_ring_fixture()
    catalog = fixture.feature_catalog

    assert catalog["dimension"] == 2
    assert catalog["exact"] is True
    entities = catalog["entities"]
    assert isinstance(entities, list)
    profiles = [
        item for item in entities
        if item["semantic_role"] == "sketch.profile"
    ]
    assert [item["logical_id"] for item in profiles] == [
        fixture.source_face_id,
    ]
    hole_loops = [
        item for item in entities
        if item["logical_id"] == "edge:hole-loop"
    ]
    assert hole_loops == [
        {
            "kind": "edge",
            "logical_id": "edge:hole-loop",
            "semantic_role": "boundary.hole-loop",
            "selectable": True,
            "topology_links": ["edge:C2"],
        }
    ]
    assert catalog["features"] == [
        {
            "feature_id": "Sketch-1",
            "kind": "sketch",
            "summary": "草图  点=2，曲线=2，Profile=1，孔=1",
        }
    ]


def test_ring_extrusion_proves_one_body_two_caps_and_hole_side() -> None:
    fixture = concentric_ring_fixture()
    topology = describe_recipe_topology(fixture.extrusion)
    proof = geometry_contract_proof(fixture.extrusion)

    assert topology.exact
    assert proof.exact
    assert proof.expected_body_count == 1
    assert topology.signature.logical_ids == (
        "edge:bottom/C1",
        "edge:bottom/C2",
        "edge:top/C1",
        "edge:top/C2",
        "face:bottom",
        "face:top",
        "face:side/C1",
        "face:side/C2",
        "body:domain",
    )
    assert topology.entity("face:bottom").semantic_role == "copy.bottom.sketch.profile"
    assert topology.entity("face:top").semantic_role == "copy.top.sketch.profile"
    assert topology.entity("face:side/C1").semantic_role == "sweep.boundary.outer"
    assert topology.entity("face:side/C2").semantic_role == "sweep.boundary.hole"
    assert topology.entity("body:domain").semantic_role == "sweep.domain"


def test_mesh_ready_publishes_transform_seam() -> None:
    _session, controller = _ring_controller()

    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    definitions = controller.definitions
    names = {item.name for item in definitions}
    assert {
        "read_profile_transform_context",
        "prepare_profile_extrusion",
        "prepare_profile_revolution",
        "prepare_profile_path_sweep",
    } <= names


def test_repeated_refusal_gets_one_correction_and_local_recovery(tmp_path) -> None:
    _session, controller = _ring_controller()
    refusal = "拉伸不受支持；必须先生成网格。"
    provider = FakeProvider(
        [
            text_response(refusal)
            for _ in range(2)
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "agent-private",
        provider,
        dynamic_tools=controller,
    )

    events = engine.send_message("拉伸成3d")
    assert not any(item.event is EngineEventType.TOOL_STARTED for item in events)
    assert not any(
        item.event is EngineEventType.MESSAGE_DELTA
        and item.data.get("text") == refusal
        for item in events
    )
