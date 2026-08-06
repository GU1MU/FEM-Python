from __future__ import annotations

import json

import pytest

from fem.application import ModelSession, UnitContext
from fem.application.preprocessing import generate_fem_model
from fem.geometry import describe_recipe_topology
from fem.mesh.settings import MeshSettings
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.geometry_authoring import geometry_contract_proof
from fem_agent.providers.base import AssistantMessage, ProviderResponse, ToolDefinition
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from tests.fixtures.profile_transform_baseline import (
    concentric_ring_fixture,
)
from tests.helpers.profile_transform_capture import (
    RequestCaptureProvider,
    tool_schema_hash,
)


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


def test_phase0_concentric_ring_catalog_freezes_profile_and_hole_lineage() -> None:
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


def test_phase0_ring_extrusion_proves_one_body_two_caps_and_hole_side() -> None:
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


@pytest.mark.gmsh
def test_phase0_ring_extrusion_generates_88_nodes_and_192_tet4(
    real_gmsh,
) -> None:
    fixture = concentric_ring_fixture()
    generated = generate_fem_model(
        fixture.extrusion,
        MeshSettings(20.0, cell_shape="tetrahedron"),
    )

    assert generated.mesh.num_nodes == 88
    assert generated.mesh.num_elements == 192
    assert {element.type for element in generated.mesh.elements} == {"Tet4"}


def test_phase0_mesh_ready_publishes_transform_seam() -> None:
    _session, controller = _ring_controller()

    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    definitions = controller.definitions
    names = tuple(item.name for item in definitions)
    assert names == (
        "read_authoring_context",
        "read_geometry_feature_catalog",
        "read_profile_transform_context",
        "prepare_profile_extrusion",
        "prepare_profile_revolution",
        "prepare_profile_path_sweep",
        "set_authoring_requirements",
        "read_mesh_refinement_context",
        "read_geometry_edit_context",
        "prepare_geometry_edit",
        "request_project_save",
        "read_deletable_objects",
        "prepare_delete_proposal",
    )
    hashes = {item.name: tool_schema_hash(item) for item in definitions}
    assert all(len(value) == 64 for value in hashes.values())

    geometry_edit = next(
        item for item in definitions if item.name == "prepare_geometry_edit"
    )
    variants = geometry_edit.parameters["properties"]["edit"]["oneOf"]
    transform_operations = {
        item["properties"]["operation"]["const"]
        for item in variants
        if item["properties"]["operation"].get("const")
        in {"extrude_profiles", "revolve_profile", "path_sweep_profile"}
    }
    assert transform_operations == set()
    assert {
        item.name
        for item in definitions
        if item.name.startswith("prepare_profile_")
    } == {
        "prepare_profile_extrusion",
        "prepare_profile_revolution",
        "prepare_profile_path_sweep",
    }


def test_phase0_request_capture_keeps_only_redacted_context_and_schema_hashes() -> None:
    tool = ToolDefinition(
        "prepare_geometry_edit",
        "Read-only baseline seam.",
        {"type": "object", "additionalProperties": False},
    )
    messages = (
        AssistantMessage(
            "system",
            "Current local state (structured metadata only): "
            + json.dumps(
                {
                    "session_id": "session-sensitive",
                    "phase": "empty",
                    "revision": 0,
                    "active_run_id": "run-sensitive",
                }
            ),
        ),
        AssistantMessage("system", "secret=fixture-only C:\\fixture\\model.femproj"),
        AssistantMessage("user", "opaque-user-payload"),
    )

    provider = RequestCaptureProvider()
    provider.complete(messages, (tool,))
    request = provider.requests[0]

    assert request.tool_names == ("prepare_geometry_edit",)
    assert request.schema_hashes == {"prepare_geometry_edit": tool_schema_hash(tool)}
    encoded = json.dumps(request.to_dict(), ensure_ascii=False)
    assert "session-sensitive" not in encoded
    assert "run-sensitive" not in encoded
    assert "fixture-only" not in encoded
    assert "opaque-user-payload" not in encoded
    assert "<session-redacted>" in encoded
    assert "<path-redacted>" in encoded
    assert len(request.schema_hashes["prepare_geometry_edit"]) == 64


def test_phase0_capture_reproduces_minimal_no_tool_call_failure(tmp_path) -> None:
    _session, controller = _ring_controller()
    provider = RequestCaptureProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    content="拉伸不受支持；必须先生成网格。",
                ),
                finish_reason="stop",
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "agent-private",
        provider,
        dynamic_tools=controller,
    )

    events = engine.send_message("拉伸成3d")
    request = provider.requests[0]
    system_context = "\n".join(request.system_context)
    failure_evidence = {
        "user_request": "拉伸成3d",
        "authoring_capability": {
            "workflow_stage": "mesh_ready",
            "published_transform_tool": "prepare_profile_extrusion",
        },
        "tool_calls": False,
        "final_capability_statement": "拉伸不受支持；必须先生成网格。",
    }

    assert failure_evidence["tool_calls"] is False
    assert "read_profile_transform_context" in request.tool_names
    assert "prepare_profile_extrusion" in request.tool_names
    assert "read_profile_transform_context" in request.schema_hashes
    assert "prepare_profile_extrusion" in request.schema_hashes
    assert "active_part_id" not in system_context
    assert "recipe_kind" not in system_context
    assert not any(item.event is EngineEventType.TOOL_STARTED for item in events)
    # Phase 0 keeps the no-tool-call evidence, while Phase 2 prevents the
    # unsupported/mesh-first text from reaching the user and performs one
    # bounded correction retry in the same turn.
    assert len(provider.requests) == 2
    assert not any(
        item.event is EngineEventType.MESSAGE_DELTA
        and item.data.get("text") == failure_evidence["final_capability_statement"]
        for item in events
    )
    assert "route_hint" in system_context
    assert any(
        "Local geometry route correction" in item
        for item in provider.requests[1].system_context
    )
    assert any(
        item.event is EngineEventType.MESSAGE_DELTA
        and item.data.get("text") == "当前几何能力检查未完成，请重试。"
        for item in events
    )
    assert "<session-redacted>" in system_context
