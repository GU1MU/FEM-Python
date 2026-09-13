from __future__ import annotations

from tests.helpers.agent_boolean_fixtures import (
    make_boolean_part_session,
    build_part_boolean_call,
    build_body_boolean_call,
    make_multi_body_session,
)

import json

import pytest

from fem.application import ModelSession
from fem.application.recipe_compiler import compile_recipe
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    RectangleGeometry,
    model,
)
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


def _json_depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _boolean_tree_payload(leaf_count: int) -> dict[str, object]:
    if leaf_count == 1:
        return {"kind": "box", "name": "B", "width": 1, "depth": 1, "height": 1}
    left_count = leaf_count // 2
    return {
        "kind": "boolean",
        "name": "B",
        "operation": "fuse",
        "object": _boolean_tree_payload(left_count),
        "tool": _boolean_tree_payload(leaf_count - left_count),
        "body_context": None,
        "planar_context": None,
        "part_context": None,
    }


def _feature_payload(root_kind: str, *, oversized: bool) -> dict[str, object]:
    if root_kind == "boolean":
        payload = _boolean_tree_payload(2049 if oversized else 2)
    else:
        payload = {
            "kind": "multi_body",
            "name": "B",
            "bodies": [
                {
                    "id": f"B{index + 1}",
                    "name": f"B{index + 1}",
                    "recipe": _boolean_tree_payload(16 if oversized else 4),
                }
                for index in range(128 if oversized else 2)
            ],
            "retired_body_ids": [],
            "retired_boolean_feature_ids": [],
        }
    return {"schema_version": 1, **payload}


def _mapping_count(value: object) -> int:
    if isinstance(value, dict):
        return 1 + sum(_mapping_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(_mapping_count(child) for child in value)
    return 0


@pytest.mark.parametrize("root_kind", ("boolean", "multi_body"))
def test_boolean_payload_rejects_excessive_recipe_nodes(root_kind: str) -> None:
    representative = _feature_payload(root_kind, oversized=False)
    recipe = geometry_recipe_from_payload(representative)
    expected_type = BooleanGeometry if root_kind == "boolean" else MultiBodyGeometry
    assert isinstance(recipe, expected_type)

    payload = _feature_payload(root_kind, oversized=True)
    # Cross the supported 4096-node budget using schema-valid recipe fields;
    # keep the other public payload bounds out of the rejection path.
    if root_kind == "multi_body":
        assert len(payload["bodies"]) <= 128
    assert _mapping_count(payload) == 4097
    assert len(json.dumps(payload, separators=(",", ":")).encode("utf-8")) < 524_288
    assert _json_depth(payload) <= 16
    with pytest.raises(ValueError, match="node budget"):
        geometry_recipe_from_payload(payload)


@pytest.mark.parametrize("root_kind", ("boolean", "multi_body"))
def test_boolean_payload_rejects_unknown_recipe_fields(root_kind: str) -> None:
    payload = _feature_payload(root_kind, oversized=False)
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match"):
        geometry_recipe_from_payload(payload)


@pytest.mark.usefixtures("real_gmsh")
def test_schema_closes_intersect_fragment_and_context_diagnoses() -> None:
    session = make_boolean_part_session(
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
        ToolExecutionContext("exact-boolean", 0, "read-boolean"),
    )
    disabled = context.data["exact_boolean"]["disabled_operations"]
    assert [item["operation"] for item in disabled] == ["intersect", "fragment"]
    assert {item["code"] for item in disabled} == {
        "boolean.agent.operation-disabled"
    }


@pytest.mark.usefixtures("real_gmsh")
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
    ),
)
def test_real_agent_part_boolean_matrix(target, tool, operation: str) -> None:
    session = make_boolean_part_session(target, tool)
    bridge, controller = _controller(session)
    before = session.snapshot()

    prepared = controller.dispatch(
        "prepare_geometry_edit",
        build_part_boolean_call(operation),
        ToolExecutionContext("exact-boolean", 0, f"part-{operation}"),
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


@pytest.mark.usefixtures("real_gmsh")
def test_cut_target_tool_order_is_persisted_and_not_exchangeable() -> None:
    target = BoxGeometry("Large Target", 2.0, 1.0, 1.0)
    tool = MovedGeometry(BoxGeometry("Small Tool", 1.0, 1.0, 1.0), 1.5, 0.0, 0.0)
    volumes = []
    for reverse in (False, True):
        session = make_boolean_part_session(*(tool, target) if reverse else (target, tool))
        bridge, controller = _controller(session)
        prepared = controller.dispatch(
            "prepare_geometry_edit",
            build_part_boolean_call("cut", result_name=f"Cut {reverse}"),
            ToolExecutionContext(
                "exact-boolean",
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
        with model(f"exact-boolean-cut-order-{reverse}", dimension=3) as cad:
            compiled = compile_recipe(cad, recipe)
            volumes.append(cad.volume(compiled.domain[0]))
        assert recipe.part_context.target_part_id == "P1"
        assert recipe.part_context.tool_part_id == "P2"
    assert volumes[0] != pytest.approx(volumes[1])


@pytest.mark.usefixtures("real_gmsh")
def test_body_boolean_preserves_same_part_target_and_unaffected_body() -> None:
    session = make_multi_body_session()
    bridge, controller = _controller(session)
    before = session.snapshot()

    prepared = controller.dispatch(
        "prepare_geometry_edit",
        build_body_boolean_call("fuse"),
        ToolExecutionContext("exact-boolean", 0, "body-fuse"),
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


@pytest.mark.usefixtures("real_gmsh")
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
def test_rejected_boolean_preflight_is_atomic(tool, operation: str, diagnostic: str) -> None:
    session = make_boolean_part_session(BoxGeometry("Target", 2.0, 1.0, 1.0), tool)
    _bridge, controller = _controller(session)
    before = session.snapshot()

    outcome = controller.dispatch(
        "prepare_geometry_edit",
        build_part_boolean_call(operation),
        ToolExecutionContext("exact-boolean", 0, f"reject-{diagnostic}"),
    )

    assert not outcome.ok
    assert diagnostic in outcome.summary
    assert session.snapshot() == before


@pytest.mark.usefixtures("real_gmsh")
def test_reject_and_stale_commit_never_mutate_session() -> None:
    session = make_boolean_part_session(
        BoxGeometry("Target", 2.0, 1.0, 1.0),
        MovedGeometry(BoxGeometry("Tool", 1.0, 1.0, 1.0), 1.5, 0.0, 0.0),
    )
    bridge, controller = _controller(session)
    before = session.snapshot()
    rejected = controller.dispatch(
        "prepare_geometry_edit",
        build_part_boolean_call("fuse", result_name="Rejected Result"),
        ToolExecutionContext("exact-boolean", 0, "reject"),
    )
    assert bridge.reject_from_gui_control(rejected.data["proposal_id"]).state is ProposalState.REJECTED
    assert session.snapshot() == before

    bridge, controller = _controller(session)
    stale = controller.dispatch(
        "prepare_geometry_edit",
        build_part_boolean_call("cut", result_name="Stale Result"),
        ToolExecutionContext("exact-boolean", 0, "stale"),
    )
    assert stale.ok, stale
    session.rename_native_part("P1", "Changed Target")
    changed = session.snapshot()
    assert bridge.accept_from_gui_control(stale.data["proposal_id"]).state is ProposalState.FAILED
    assert session.snapshot() == changed
