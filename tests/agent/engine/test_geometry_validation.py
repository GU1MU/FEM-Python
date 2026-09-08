import pytest

from fem_agent.engine import (
    AgentSessionEngine,
    EngineEventType,
    _missing_requested_geometry_features,
)
from fem_agent.providers.base import ToolCall
from fem_agent.providers.fake import FakeProvider

from tests.helpers.agent_engine_providers import (
    _tool_response,
)
from tests.helpers.agent_engine_registry_fixtures import (
    _AdditionalModelToolRegistry,
    _GeometryEditToolRegistry,
)


pytestmark = pytest.mark.integration


def test_explicit_2d_request_cannot_fall_back_to_a_derived_3d_output(tmp_path):
    construction = {
        "schema_version": 1,
        "name": "2D平板",
        "plane": "XY",
        "nodes": [
            {
                "id": "plate",
                "kind": "rectangle",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 300,
            }
        ],
        "result_node_id": "plate",
    }
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "wrong-3d",
                    "prepare_planar_construction_proposal",
                    {
                        "part_function": "2D平板",
                        "construction": construction,
                        "output": {
                            "kind": "extrusion",
                            "profile_selection": "unique_material_profile",
                            "height": 10,
                        },
                    },
                )
            ),
            _tool_response(
                ToolCall(
                    "correct-2d",
                    "prepare_planar_construction_proposal",
                    {
                        "part_function": "2D平板",
                        "construction": construction,
                        "output": {"kind": "planar"},
                    },
                )
            ),
        ]
    )
    tools = _AdditionalModelToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_planar_dimension_guard",
        dynamic_tools=tools,
    )

    events = engine.send_message("创建一个二维平板")

    assert len(provider.requests) == 2
    assert tools.calls == [
        (
            "prepare_planar_construction_proposal",
            {
                "part_function": "2D平板",
                "construction": construction,
                "output": {"kind": "planar"},
            },
        )
    ]
    first_result = next(
        event.data["result"]
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["call_id"] == "wrong-3d"
    )
    assert first_result["data"]["required_output"] == "planar"


def test_branching_slot_rejects_single_path_before_dispatch(tmp_path):
    invalid = {
        "part_function": "二维平板，中央分叉槽",
        "construction": {
            "schema_version": 1,
            "name": "invalid_branching_slot",
            "plane": "XY",
            "nodes": [
                {
                    "id": "plate",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 300,
                    "height": 100,
                },
                {
                    "id": "slot",
                    "kind": "path_stroke",
                    "points": [
                        {"x": 140, "y": 35},
                        {"x": 160, "y": 35},
                        {"x": 160, "y": 65},
                        {"x": 140, "y": 65},
                    ],
                    "width": 10,
                    "cap": "butt",
                    "join": "miter",
                },
                {
                    "id": "result",
                    "kind": "difference",
                    "base": "plate",
                    "subtract": ["slot"],
                },
            ],
            "result_node_id": "result",
        },
        "output": {"kind": "planar"},
    }
    corrected = {
        "part_function": "二维平板，中央分叉槽",
        "construction": {
            "schema_version": 1,
            "name": "connected_branching_slot",
            "plane": "XY",
            "nodes": [
                {
                    "id": "plate",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 300,
                    "height": 100,
                },
                {
                    "id": "left_stem",
                    "kind": "rectangle",
                    "x": 130,
                    "y": 30,
                    "width": 10,
                    "height": 40,
                },
                {
                    "id": "cross_stem",
                    "kind": "rectangle",
                    "x": 130,
                    "y": 45,
                    "width": 40,
                    "height": 10,
                },
                {
                    "id": "right_stem",
                    "kind": "rectangle",
                    "x": 160,
                    "y": 30,
                    "width": 10,
                    "height": 40,
                },
                {
                    "id": "slot",
                    "kind": "union",
                    "operands": ["left_stem", "cross_stem", "right_stem"],
                },
                {
                    "id": "result",
                    "kind": "difference",
                    "base": "plate",
                    "subtract": ["slot"],
                },
            ],
            "result_node_id": "result",
        },
        "output": {"kind": "planar"},
    }
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "invalid-slot",
                    "prepare_planar_construction_proposal",
                    invalid,
                )
            ),
            _tool_response(
                ToolCall(
                    "corrected-slot",
                    "prepare_planar_construction_proposal",
                    corrected,
                )
            ),
        ]
    )
    tools = _AdditionalModelToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_branching_slot_guard",
        dynamic_tools=tools,
    )

    events = engine.send_message(
        "创建一个2D平板，在中央做一条中心线包含分叉节点的槽"
    )

    assert tools.calls == [
        ("prepare_planar_construction_proposal", corrected)
    ]
    assert len(provider.requests) == 2
    assert "single non-branching open centerline" in (
        provider.requests[1].messages[-1].content or ""
    )
    completed = [
        event
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_planar_construction_proposal"
    ]
    assert completed[0].data["result"]["ok"] is False
    assert completed[-1].data["result"]["data"]["state"] == "pending_confirmation"


def test_nonbranching_path_slot_rejects_disconnected_rectangle_fallback(tmp_path):
    correct_edit = {
        "part_id": "P1",
        "edit": {
            "operation": "add_path_slot",
            "points": [
                {"x": 75, "y": 80},
                {"x": 40, "y": 80},
                {"x": 40, "y": 50},
                {"x": 75, "y": 50},
                {"x": 75, "y": 20},
                {"x": 40, "y": 20},
            ],
            "width": 6,
            "cap": "square",
            "join": "miter",
        },
    }
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "read-edit",
                    "read_geometry_edit_context",
                    {"part_id": "P1"},
                )
            ),
            _tool_response(
                ToolCall(
                    "bad-edit",
                    "prepare_geometry_edit",
                    {
                        "part_id": "P1",
                        "edit": {
                            "operation": "batch",
                            "edits": [
                                {
                                    "operation": "add_rectangle",
                                    "x": 40,
                                    "y": 70,
                                    "width": 35,
                                    "height": 6,
                                },
                                {
                                    "operation": "add_rectangle",
                                    "x": 40,
                                    "y": 47,
                                    "width": 35,
                                    "height": 6,
                                },
                                {
                                    "operation": "add_rectangle",
                                    "x": 40,
                                    "y": 24,
                                    "width": 35,
                                    "height": 6,
                                },
                            ],
                        },
                    },
                )
            ),
            _tool_response(
                ToolCall("correct-edit", "prepare_geometry_edit", correct_edit)
            ),
        ]
    )
    tools = _GeometryEditToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_nonbranching_path_slot_guard",
        dynamic_tools=tools,
    )

    events = engine.send_message(
        "请加入一条由单条开放无分叉中心线定义的连续定宽槽"
    )

    assert [name for name, _arguments in tools.calls] == [
        "read_geometry_edit_context",
        "prepare_geometry_edit",
    ]
    assert tools.calls[-1][1] == correct_edit
    bad_result = next(
        event.data["result"]
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["call_id"] == "bad-edit"
    )
    assert not bad_result["ok"]
    assert bad_result["data"]["required_operation"] == "add_path_slot"
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["call_id"] == "correct-edit"
        and event.data["result"]["data"]["state"] == "pending_confirmation"
        for event in events
    )
    preview = next(
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
        and "方案预览" in event.data["text"]
    )
    assert "切除一条连续定宽槽" in preview
    assert "(75, 80) → (40, 80)" in preview
    assert "add_path_slot" not in preview
    assert "P1" not in preview
    assert "{" not in preview
    assert any(
        event.event is EngineEventType.MESSAGE_DELTA
        and "上一版方案未通过校验" in event.data["text"]
        for event in events
    )


def test_generic_geometry_feature_guard_requires_requested_slot_and_holes():
    partial = {
        "part_function": "宽平板",
        "geometry": {
            "kind": "planar_profiles",
            "profiles": [
                {
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 500,
                    "height": 300,
                }
            ],
        },
    }
    complete = {
        "part_function": "宽平板，中央开槽，四周开孔",
        "geometry": {
            "kind": "planar_profiles",
            "profiles": [
                {
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 500,
                    "height": 300,
                    "role": "material",
                },
                {
                    "kind": "rectangle",
                    "x": 225,
                    "y": 110,
                    "width": 50,
                    "height": 80,
                    "role": "hole",
                },
                {
                    "kind": "circle",
                    "center_x": 40,
                    "center_y": 40,
                    "radius": 10,
                    "role": "hole",
                },
            ],
        },
    }
    request = "做一个宽平板，中央开槽，四周开孔"
    assert _missing_requested_geometry_features(request, partial) == (
        "slot_or_cutout",
        "holes",
    )
    assert _missing_requested_geometry_features(request, complete) == ()
