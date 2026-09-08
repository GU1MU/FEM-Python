from __future__ import annotations

from tests.helpers.agent_planar_construction import (
    tool_response,
    text_response,
    ControllerDynamicTools,
    make_planar_authoring_controller,
    build_planar_arguments,
    build_rectangle_arguments,
    dispatch_planar_construction,
)

import json

import pytest

from fem.application import ModelSession
from fem_agent.authoring import ProposalState
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import ToolExecutionContext


def test_publishes_strict_schema_and_bounded_context() -> None:
    _bridge, controller = make_planar_authoring_controller(ModelSession())
    definition = next(
        item
        for item in controller.definitions
        if item.name == "prepare_planar_construction_proposal"
    )
    outputs = definition.parameters["properties"]["output"]["oneOf"]
    assert outputs[0] == {"const": "planar"}
    assert {item["properties"]["kind"]["const"] for item in outputs[1:]} == {
        "planar",
        "extrusion",
        "revolution",
        "path_sweep",
    }
    branches = {item["properties"]["kind"]["const"]: item for item in outputs[1:]}
    assert branches["extrusion"]["required"] == [
        "kind",
        "profile_selection",
        "height",
    ]
    assert branches["revolution"]["required"] == [
        "kind",
        "profile_selection",
        "axis",
        "angle_degrees",
    ]
    assert branches["path_sweep"]["required"] == [
        "kind",
        "profile_selection",
        "path",
        "frame_strategy",
    ]
    assert all(branch["additionalProperties"] is False for branch in branches.values())

    planar_object = next(
        item
        for item in outputs[1:]
        if item["properties"]["kind"]["const"] == "planar"
    )
    assert planar_object["required"] == ["kind"]
    assert planar_object["additionalProperties"] is False
    construction = definition.parameters["properties"]["construction"]
    variants = construction["properties"]["nodes"]["items"]["oneOf"]
    assert {item["properties"]["kind"]["const"] for item in variants} == {
        "rectangle",
        "circle",
        "polygon",
        "path_stroke",
        "union",
        "difference",
        "intersection",
        "translate",
        "rotate",
        "mirror",
        "linear_pattern",
        "rectangular_pattern",
        "circular_pattern",
    }
    assert all(item["additionalProperties"] is False for item in variants)
    rectangle = next(
        item for item in variants if item["properties"]["kind"]["const"] == "rectangle"
    )
    circle = next(
        item for item in variants if item["properties"]["kind"]["const"] == "circle"
    )
    polygon = next(
        item for item in variants if item["properties"]["kind"]["const"] == "polygon"
    )
    path_stroke = next(
        item
        for item in variants
        if item["properties"]["kind"]["const"] == "path_stroke"
    )
    assert "lower-left" in rectangle["properties"]["x"]["description"]
    assert "not the center" in rectangle["properties"]["y"]["description"]
    assert "never diameter" in circle["properties"]["radius"]["description"]
    assert "Default closed-boundary representation" in (
        polygon["properties"]["vertices"]["description"]
    )
    assert "decomposing one shaped slot into rectangles" in (
        polygon["properties"]["vertices"]["description"]
    )
    assert "non-branching centerline" in (
        path_stroke["properties"]["points"]["description"]
    )
    path_points_description = path_stroke["properties"]["points"]["description"]
    assert "multiple bends" in path_points_description
    assert "straight [[10,10],[40,10]]" in path_points_description
    assert "[[10,10],[10,30],[35,30],[35,50]]" in path_points_description
    assert "first and last points must differ" in path_points_description
    assert "never append the first point" in path_points_description
    nodes_description = construction["properties"]["nodes"]["description"]
    assert "two independent slots" in nodes_description
    assert nodes_description.count('"kind":"path_stroke"') == 2
    assert nodes_description.count('"kind":"difference"') == 2
    assert '"base":"cut_a"' in nodes_description
    assert "never concatenate disconnected centerlines" in nodes_description
    assert "Use polygon as the default closed-boundary" in definition.description
    assert "path_stroke as the preferred compact form" in definition.description
    provider_surface = json.dumps(
        {
            "description": definition.description,
            "parameters": definition.parameters,
        },
        ensure_ascii=False,
    )
    assert "S-shaped" not in provider_surface
    assert "U-shaped" not in provider_surface
    assert "H-shaped" not in provider_surface

    context = controller.dispatch(
        "read_authoring_context",
        {},
        ToolExecutionContext("phase3-planar", 0, "context"),
    )
    capability = context.data["context"]["planar_construction_ir"]
    assert capability["output_kinds"] == ["planar", "extrusion", "revolution", "path_sweep"]
    assert capability["schema_version"] == 1
    assert capability["plane"] == "XY"
    assert capability["budgets"]["max_node_count"] == 64
    assert capability["budgets"]["max_pattern_instances"] == 256
    assert capability["coordinate_conventions"] == {
        "rectangle_anchor": "lower_left",
        "rectangle_extent": "x..x+width, y..y+height",
        "circle_position": "center_x, center_y",
        "circle_size": "radius; use diameter/2 when the request gives a diameter",
        "pattern_seed": "included_as_instance_zero",
    }
    assert capability["slot_representation_policy"] == {
        "closed_boundary_default": "polygon",
        "centerline_compact_form": "path_stroke",
        "path_stroke_condition": (
            "one_open_non_branching_polyline_and_one_constant_width"
        ),
        "multiple_bends_supported": True,
        "primitive_union_role": "fallback_for_genuinely_composite_geometry",
        "rectangle_decomposition_for_one_connected_slot": "avoid",
    }
    serialized = json.dumps(capability).casefold()
    assert "gmsh" not in serialized
    assert "occ" not in serialized


def test_rejects_misanchored_cutters_before_presenting_a_card() -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    before = session.snapshot()
    construction = {
        "schema_version": 1,
        "name": "misanchored_plate",
        "plane": "XY",
        "nodes": [
            {
                "id": "plate",
                "kind": "rectangle",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 300,
            },
            {
                "id": "centered_as_if_x_y_were_centers",
                "kind": "rectangle",
                "x": -20,
                "y": -40,
                "width": 40,
                "height": 80,
            },
            {
                "id": "outside_hole",
                "kind": "circle",
                "center_x": -35,
                "center_y": -135,
                "radius": 3,
            },
            {
                "id": "result",
                "kind": "difference",
                "base": "plate",
                "subtract": ["centered_as_if_x_y_were_centers", "outside_hole"],
            },
        ],
        "result_node_id": "result",
    }

    result = controller.dispatch(
        "prepare_planar_construction_proposal",
        {
            "part_function": "错误锚点回归样例",
            "construction": construction,
            "output": "planar",
        },
        ToolExecutionContext("phase3-planar", 0, "misanchored"),
    )

    assert result.ok is False
    assert result.data["diagnostic"]["code"] == "planar-ir.subtract-no-effect"
    assert result.data["diagnostic"]["node_id"] in {
        "centered_as_if_x_y_were_centers",
        "outside_hole",
    }
    assert result.data["diagnostic"]["model_unchanged"] is True
    assert bridge._records == {}
    assert session.snapshot() == before


@pytest.mark.parametrize(
    "edit",
    [
        {
            "operation": "add_path_slot",
            "points": [
                {"x": 30.0, "y": 190.0},
                {"x": 30.0, "y": 220.0},
                {"x": 60.0, "y": 220.0},
                {"x": 30.0, "y": 190.0},
            ],
            "width": 6.0,
            "cap": "butt",
            "join": "miter",
        },
        {
            "operation": "planar_boolean",
            "boolean_operation": "cut",
            "tool": {
                "kind": "path_stroke",
                "points": [
                    {"x": 30.0, "y": 190.0},
                    {"x": 30.0, "y": 220.0},
                    {"x": 60.0, "y": 220.0},
                    {"x": 30.0, "y": 190.0},
                ],
                "width": 6.0,
                "cap": "butt",
                "join": "miter",
            },
        },
    ],
)
def test_closed_path_slot_edit_returns_same_representation_repair_guidance(
    edit: dict[str, object],
) -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    initial = dispatch_planar_construction(
        controller, key="closed-slot-initial", arguments=build_rectangle_arguments()
    )
    receipt = bridge.accept_from_gui_control(str(initial.data["proposal_id"]))
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state, receipt.message)
    context = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": "P1"},
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "closed-slot-context",
        ),
    )
    assert context.ok, context.summary
    policy = context.data["freeform_profile_policy"]
    assert policy["constant_width_slot_operation_priority"] == [
        "add_path_slot",
        "planar_boolean(tool.kind=path_stroke)",
    ]
    assert policy["planar_boolean_path_stroke_role"] == "lower_level_equivalent"

    result = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": edit,
        },
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "closed-slot-edit",
        ),
    )

    assert not result.ok
    assert "error" in result.data, result.data
    error = result.data["error"]
    assert error["code"] == "planar-ir.invalid-path-stroke"
    assert error["failed_operation"] == edit["operation"]
    assert error["representation"] == "path_stroke"
    assert error["evidence"]["points_role"] == "centerline"
    assert error["evidence"]["required_topology"] == "open"
    assert error["evidence"]["first_equals_last"] is True
    remediation = error["remediation"]
    assert remediation["action"] == "retry_same_path_slot_with_revised_centerline"
    assert remediation["preserve_representation"] is True
    assert remediation["preferred_operation"] == "add_path_slot"
    assert remediation["lower_level_equivalent"] == (
        "planar_boolean(tool.kind=path_stroke)"
    )
    assert remediation["polygon_fallback_only_when"] == [
        "centerline_has_junction",
        "width_varies",
    ]


def test_fake_provider_uses_one_card_and_continues_from_new_snapshot(
    tmp_path,
) -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    dynamic = ControllerDynamicTools(controller)
    provider = FakeProvider(
        [
            tool_response(
                "prepare-ir",
                "prepare_planar_construction_proposal",
                build_rectangle_arguments(),
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "phase3-fake-provider",
        provider,
        dynamic_tools=dynamic,
    )
    before = session.snapshot()

    events = engine.send_message("创建二维矩形板")

    assert session.snapshot() == before
    assert len(provider.requests) == 1
    assert [
        event.data["tool"]
        for event in events
        if event.event is EngineEventType.TOOL_STARTED
    ] == ["prepare_planar_construction_proposal"]
    assert len(bridge._records) == 1
    audit = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    assert audit["entries"][-1]["tool_call_flags"]["called_tool_names"] == [
        "prepare_planar_construction_proposal"
    ]
    proposal_id, record = next(iter(bridge._records.items()))
    assert record.state is ProposalState.PENDING_CONFIRMATION
    checkpoint = next(
        event.data["result"]["data"]["continuation_checkpoint"]
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
    )

    receipt = bridge.accept_from_gui_control(proposal_id)
    controller.record_proposal_state("geometry", receipt.state, receipt.message)
    dynamic.refresh_turn_snapshot(tuple(item.name for item in controller.definitions))
    provider.queue(
        tool_response("read-new", "read_authoring_context", {}),
        text_response("二维部件已进入后续建模阶段。"),
    )
    continuation = engine.continue_after_proposal(
        proposal_id,
        checkpoint["proposal_hash"],
        checkpoint["source_turn_id"],
        checkpoint["model_revision"],
        receipt.state.value,
        receipt.message,
    )

    assert session.snapshot().parts[0].dimension == 2
    assert dynamic.provider_snapshot.active_part_dimension == 2
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "read_authoring_context"
        for event in continuation
    )
    assert len([item for item in engine._history if item.role == "user"]) == 1
    assert not any(
        "confirm" in str(event.data.get("text", "")).casefold()
        or "确认" in str(event.data.get("text", ""))
        for event in continuation
        if event.event is EngineEventType.MESSAGE_DELTA
    )


def test_invalid_ir_fails_without_a_card_or_model_change() -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    before = session.snapshot()
    arguments = build_planar_arguments()
    arguments["construction"]["nodes"][-1]["subtract"] = ["missing"]

    result = controller.dispatch(
        "prepare_planar_construction_proposal",
        arguments,
        ToolExecutionContext("phase3-planar", 0, "invalid"),
    )

    assert not result.ok
    assert result.data["diagnostic"]["code"] == "planar-ir.reference-missing"
    assert result.data["diagnostic"]["model_unchanged"] is True
    assert "proposal_id" not in result.data
    assert not bridge._records
    assert session.snapshot() == before


def test_legacy_planar_profiles_remains_callable_and_auditable() -> None:
    session = ModelSession()
    _bridge, controller = make_planar_authoring_controller(session)
    assert {item.name for item in controller.definitions} >= {
        "prepare_geometry_proposal",
        "prepare_planar_construction_proposal",
    }

    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "兼容矩形板",
            "geometry": {
                "kind": "planar_profiles",
                "profiles": [
                    {
                        "kind": "rectangle",
                        "x": 0.0,
                        "y": 0.0,
                        "width": 10.0,
                        "height": 5.0,
                    }
                ],
            },
        },
        ToolExecutionContext("phase3-legacy", 0, "legacy"),
    )

    assert result.ok
    assert result.data["authoring_path"] == "legacy_planar_profiles"
