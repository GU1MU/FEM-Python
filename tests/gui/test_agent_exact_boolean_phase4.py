from __future__ import annotations

import json

import pytest

from fem.application import ModelSession, UnitContext
from fem.application.preprocessing import generate_fem_model
from fem.application.recipe_compiler import compile_recipe
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    PathSweptGeometry,
    RectangleGeometry,
    RevolvedGeometry,
    SolidBody,
    WireGeometry,
    WireMember,
    WirePoint,
    model,
)
from fem.io.project import decode_project, encode_project
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import ProposalState
from fem_agent.geometry_authoring import (
    geometry_recipe_from_payload,
    geometry_recipe_to_payload,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)


def _controller(session: ModelSession):
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    return bridge, controller


def _part_session(target, tool) -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent Boolean",
        UnitContext("mm", "N", "MPa"),
        target,
        part_name="Target Part",
    )
    session.add_native_part(tool, name="Tool Part")
    return session


def _part_call(operation: str, *, result_name: str = "Boolean Result"):
    return {
        "part_id": "P1",
        "edit": {
            "operation": "part_boolean",
            "boolean_operation": operation,
            "tool_part_id": "P2",
            "result_name": result_name,
            "tool_handling": "consume_tool_part",
        },
    }


def _body_call(operation: str, *, result_name: str = "Body Result"):
    return {
        "part_id": "P1",
        "edit": {
            "operation": "body_boolean",
            "boolean_operation": operation,
            "target_body_id": "B1",
            "tool_body_id": "B2",
            "result_name": result_name,
            "tool_handling": "consume_tool_body",
        },
    }


def _path_solid() -> PathSweptGeometry:
    return PathSweptGeometry(
        RectangleGeometry("Path Profile", 2.0, 1.0),
        WireGeometry(
            "Ordered Path",
            (
                WirePoint("A", 0.0, 0.0, 0.0),
                WirePoint("B", 0.0, 0.0, 2.0),
            ),
            (WireMember("AB", "A", "B"),),
        ),
        ("face:domain",),
        "transport",
    )


def _multi_body_session() -> ModelSession:
    geometry = MultiBodyGeometry(
        "Canonical same-Part Bodies",
        (
            SolidBody("B1", "Target", BoxGeometry("Target", 2.0, 1.0, 1.0)),
            SolidBody(
                "B2",
                "Tool",
                MovedGeometry(BoxGeometry("Tool", 1.0, 1.0, 1.0), 1.5, 0.0, 0.0),
            ),
            SolidBody(
                "B3",
                "Unaffected",
                MovedGeometry(
                    BoxGeometry("Unaffected", 1.0, 1.0, 1.0),
                    5.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent Body Boolean",
        UnitContext("mm", "N", "MPa"),
        geometry,
        part_name="MultiBody Part",
    )
    return session


def _json_depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


@pytest.mark.parametrize("root_kind", ("boolean", "multi_body"))
def test_phase4_boolean_payload_rejects_more_than_512_mapping_nodes(
    root_kind: str,
) -> None:
    payload = {
        "schema_version": 1,
        "kind": root_kind,
        "padding": [{"node": index} for index in range(512)],
    }
    assert 1 + len(payload["padding"]) > 512
    assert len(json.dumps(payload, separators=(",", ":")).encode("utf-8")) < 65536
    assert _json_depth(payload) <= 16

    with pytest.raises(ValueError, match="node budget"):
        geometry_recipe_from_payload(payload)


@pytest.mark.gmsh
def test_phase4_schema_closes_intersect_fragment_and_context_diagnoses() -> None:
    session = _part_session(
        BoxGeometry("Target", 2.0, 1.0, 1.0),
        MovedGeometry(BoxGeometry("Tool", 1.0, 1.0, 1.0), 1.5, 0.0, 0.0),
    )
    _bridge, controller = _controller(session)
    definition = next(
        item for item in controller.definitions if item.name == "prepare_geometry_edit"
    )
    variants = definition.parameters["properties"]["edit"]["oneOf"]
    by_operation = {
        item["properties"]["operation"]["const"]: item for item in variants
    }

    assert by_operation["part_boolean"]["properties"]["boolean_operation"]["enum"] == [
        "fuse",
        "cut",
    ]
    body_operation_schema = by_operation["body_boolean"]["properties"][
        "boolean_operation"
    ]
    assert body_operation_schema["enum"] == [
        "fuse",
        "cut",
    ]
    assert by_operation["part_boolean"]["additionalProperties"] is False
    assert by_operation["body_boolean"]["additionalProperties"] is False

    context = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": "P1"},
        ToolExecutionContext("phase4", 0, "read-boolean"),
    )
    disabled = context.data["exact_boolean"]["disabled_operations"]
    assert [item["operation"] for item in disabled] == ["intersect", "fragment"]
    assert {item["code"] for item in disabled} == {
        "boolean.agent.operation-disabled"
    }


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("target", "tool", "operation"),
    (
        (
            ExtrudedGeometry(RectangleGeometry("Extruded", 2.0, 1.0), 1.0),
            MovedGeometry(BoxGeometry("Basic", 1.0, 1.0, 1.0), 1.5, 0.0, 0.0),
            "fuse",
        ),
        (
            ExtrudedGeometry(RectangleGeometry("Extruded A", 2.0, 1.0), 1.0),
            MovedGeometry(
                ExtrudedGeometry(RectangleGeometry("Extruded B", 1.0, 1.0), 1.0),
                1.5,
                0.0,
                0.0,
            ),
            "cut",
        ),
        (
            _path_solid(),
            MovedGeometry(
                ExtrudedGeometry(RectangleGeometry("Path Tool", 1.0, 1.0), 2.0),
                1.5,
                0.0,
                0.0,
            ),
            "fuse",
        ),
        (
            RevolvedGeometry(
                RectangleGeometry("Revolve Profile", 2.0, 1.0),
                "x",
                180.0,
                ("face:domain",),
            ),
            MovedGeometry(BoxGeometry("Revolve Tool", 1.0, 1.0, 1.0), 0.5, -0.5, 0.0),
            "cut",
        ),
    ),
)
def test_phase4_real_agent_part_boolean_matrix(target, tool, operation: str) -> None:
    session = _part_session(target, tool)
    bridge, controller = _controller(session)
    before = session.snapshot()

    prepared = controller.dispatch(
        "prepare_geometry_edit",
        _part_call(operation),
        ToolExecutionContext("phase4", 0, f"part-{operation}"),
    )

    assert prepared.ok, prepared.summary
    assert session.snapshot() == before
    proposal = bridge._records[prepared.data["proposal_id"]].proposal
    assert proposal.base_session_revision == before.session_revision
    assert proposal.display_summary["operation"] == operation
    assert proposal.display_summary["tool_handling"] == "consume_tool_part"
    assert proposal.display_summary["lineage_entity_count"] > 0
    recipe_bytes = proposal.operations[0].parameters["recipe_json"].encode("utf-8")
    assert len(recipe_bytes) < 32768
    recipe_payload = json.loads(proposal.operations[0].parameters["recipe_json"])
    round_tripped = geometry_recipe_from_payload(recipe_payload)
    assert geometry_recipe_to_payload(round_tripped) == recipe_payload

    receipt = bridge.accept_from_gui_control(proposal.proposal_id)
    assert receipt.state is ProposalState.SUCCEEDED
    committed = session.snapshot()
    assert committed.part("P1").suppressed
    assert committed.part("P2").suppressed
    result = committed.part("P3")
    assert not result.suppressed
    assert isinstance(result.geometry_recipe, BooleanGeometry)
    assert result.geometry_recipe.operation == operation
    assert any(record.kind in {"fuse", "cut"} for record in result.feature_history)


@pytest.mark.gmsh
def test_phase4_cut_target_tool_order_is_persisted_and_not_exchangeable() -> None:
    target = BoxGeometry("Large Target", 2.0, 1.0, 1.0)
    tool = MovedGeometry(BoxGeometry("Small Tool", 1.0, 1.0, 1.0), 1.5, 0.0, 0.0)
    volumes = []
    for reverse in (False, True):
        session = _part_session(*(tool, target) if reverse else (target, tool))
        bridge, controller = _controller(session)
        prepared = controller.dispatch(
            "prepare_geometry_edit",
            _part_call("cut", result_name=f"Cut {reverse}"),
            ToolExecutionContext(
                "phase4",
                int(reverse),
                f"cut-order-{str(reverse).lower()}",
            ),
        )
        assert prepared.ok, prepared.summary
        proposal = bridge._records[prepared.data["proposal_id"]].proposal
        assert proposal.display_summary["target_part_id"] == "P1"
        assert proposal.display_summary["tool_part_id"] == "P2"
        assert bridge.accept_from_gui_control(proposal.proposal_id).state is ProposalState.SUCCEEDED
        recipe = session.snapshot().part("P3").geometry_recipe
        with model(f"phase4-cut-order-{reverse}", dimension=3) as cad:
            compiled = compile_recipe(cad, recipe)
            volumes.append(cad.volume(compiled.domain[0]))
        assert recipe.part_context.target_part_id == "P1"
        assert recipe.part_context.tool_part_id == "P2"
    assert volumes[0] != pytest.approx(volumes[1])


@pytest.mark.gmsh
def test_phase4_body_boolean_preserves_same_part_target_and_unaffected_body() -> None:
    session = _multi_body_session()
    bridge, controller = _controller(session)
    before = session.snapshot()

    prepared = controller.dispatch(
        "prepare_geometry_edit",
        _body_call("fuse"),
        ToolExecutionContext("phase4", 0, "body-fuse"),
    )

    assert prepared.ok, prepared.summary
    assert session.snapshot() == before
    proposal = bridge._records[prepared.data["proposal_id"]].proposal
    assert proposal.expected_changes["preserved_target_body_id"] == "B1"
    assert proposal.expected_changes["consumed_tool_body_id"] == "B2"
    assert bridge.accept_from_gui_control(proposal.proposal_id).state is ProposalState.SUCCEEDED

    committed = session.snapshot()
    assert tuple(part.id for part in committed.parts) == ("P1",)
    geometry = committed.part("P1").geometry_recipe
    assert type(geometry) is MultiBodyGeometry
    assert tuple(body.id for body in geometry.bodies) == ("B1", "B3")
    assert geometry.body("B3") == before.part("P1").geometry_recipe.body("B3")
    assert geometry.body("B1").recipe.name == "Body Result"
    assert "B2" in geometry.retired_body_ids


@pytest.mark.gmsh
def test_phase4_body_boolean_save_reopen_undo_replay_and_remesh() -> None:
    session = _multi_body_session()
    bridge, controller = _controller(session)
    prepared = controller.dispatch(
        "prepare_geometry_edit",
        _body_call("cut", result_name="Persisted Cut"),
        ToolExecutionContext("phase4", 0, "body-cut-persist"),
    )
    assert prepared.ok, prepared.summary
    assert bridge.accept_from_gui_control(prepared.data["proposal_id"]).state is ProposalState.SUCCEEDED
    committed_geometry = session.snapshot().part("P1").geometry_recipe

    encoded = encode_project(session.prepare_project_save())
    reopened = ModelSession()
    assert reopened.replace_from_snapshot(decode_project(encoded).snapshot).accepted
    reopened_geometry = reopened.snapshot().part("P1").geometry_recipe
    assert reopened_geometry == committed_geometry
    with model("phase4-body-replay", dimension=3) as cad:
        assert len(compile_recipe(cad, reopened_geometry).domain) == 2

    coarse = generate_fem_model(
        reopened_geometry,
        MeshSettings(0.7, cell_shape="tetrahedron"),
    )
    refined = generate_fem_model(
        reopened_geometry,
        MeshSettings(0.45, cell_shape="tetrahedron"),
    )
    assert {element.type for element in coarse.mesh.elements} == {"Tet4"}
    assert len(refined.mesh.elements) > len(coarse.mesh.elements)

    reopened.undo_body_boolean("P1", "B1")
    restored = reopened.snapshot().part("P1").geometry_recipe
    assert tuple(body.id for body in restored.bodies) == ("B1", "B2", "B3")
    assert restored.body("B1").recipe == _multi_body_session().snapshot().part("P1").geometry_recipe.body("B1").recipe
    assert restored.body("B2").recipe == _multi_body_session().snapshot().part("P1").geometry_recipe.body("B2").recipe


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("tool", "operation", "diagnostic"),
    (
        (
            MovedGeometry(BoxGeometry("Disjoint", 1.0, 1.0, 1.0), 4.0, 0.0, 0.0),
            "fuse",
            "volume-count",
        ),
        (
            MovedGeometry(BoxGeometry("Touching", 1.0, 1.0, 1.0), 2.0, 0.0, 0.0),
            "fuse",
            "non-positive-overlap",
        ),
        (
            MovedGeometry(BoxGeometry("Splitter", 1.0, 1.0, 1.0), 0.5, 0.0, 0.0),
            "cut",
            "volume-count",
        ),
    ),
)
def test_phase4_rejected_boolean_preflight_is_atomic(tool, operation: str, diagnostic: str) -> None:
    session = _part_session(BoxGeometry("Target", 2.0, 1.0, 1.0), tool)
    _bridge, controller = _controller(session)
    before = session.snapshot()

    outcome = controller.dispatch(
        "prepare_geometry_edit",
        _part_call(operation),
        ToolExecutionContext("phase4", 0, f"reject-{diagnostic}"),
    )

    assert not outcome.ok
    assert diagnostic in outcome.summary
    assert session.snapshot() == before


@pytest.mark.gmsh
def test_phase4_reject_and_stale_commit_never_mutate_session() -> None:
    session = _part_session(
        BoxGeometry("Target", 2.0, 1.0, 1.0),
        MovedGeometry(BoxGeometry("Tool", 1.0, 1.0, 1.0), 1.5, 0.0, 0.0),
    )
    bridge, controller = _controller(session)
    before = session.snapshot()
    rejected = controller.dispatch(
        "prepare_geometry_edit",
        _part_call("fuse", result_name="Rejected Result"),
        ToolExecutionContext("phase4", 0, "reject"),
    )
    assert bridge.reject_from_gui_control(rejected.data["proposal_id"]).state is ProposalState.REJECTED
    assert session.snapshot() == before

    bridge, controller = _controller(session)
    stale = controller.dispatch(
        "prepare_geometry_edit",
        _part_call("cut", result_name="Stale Result"),
        ToolExecutionContext("phase4", 0, "stale"),
    )
    assert stale.ok, stale
    session.rename_native_part("P1", "Changed Target")
    changed = session.snapshot()
    assert bridge.accept_from_gui_control(stale.data["proposal_id"]).state is ProposalState.FAILED
    assert session.snapshot() == changed
