from __future__ import annotations

import pytest

from fem_agent.authoring_runtime import AuthoringTurnSnapshot
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import ProviderResponse, ToolCall, ToolDefinition
from fem_agent.providers.fake import FakeProvider
from fem_agent.routing import geometry_route_hint
from fem_agent.schemas import ToolResult

from tests.helpers.agent_provider_fixtures import text_response, tool_response


_TOOLS = (
    ToolDefinition(
        "read_profile_transform_context",
        "Read bounded native geometry transform context.",
        {
            "type": "object",
            "properties": {"part_id": {"type": "string"}},
            "required": ["part_id"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "prepare_profile_extrusion",
        "Prepare a native geometry edit proposal.",
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    ),
)


class _DynamicRegistry:
    def __init__(self, *, dimension: int = 2, tools=_TOOLS):
        self.definitions = tuple(tools)
        self.calls = []
        self.provider_snapshot = AuthoringTurnSnapshot(
            available=True,
            source_kind="native",
            workflow_stage="mesh_ready",
            document_id="document:profile-transform",
            session_id="native-profile-transform",
            session_revision=1,
            active_part_id="P1",
            active_part_dimension=dimension,
            active_part_recipe_kind="planar_sketch",
            active_part_suppressed=False,
            mesh_present=True,
            mesh_current=True,
            enabled_capabilities=("edit_native_geometry",),
            published_tool_names=tuple(item.name for item in self.definitions),
            snapshot_generation=1,
        )

    def refresh_turn_snapshot(self, published_tool_names=()):
        return self.provider_snapshot

    def dispatch(self, name, arguments, context):
        self.calls.append((name, dict(arguments)))
        return ToolResult(
            ok=True,
            session_id=context.session_id,
            input_revision=context.expected_revision,
            idempotency_key=context.idempotency_key,
            summary=f"{name} read",
        )


def _refusal() -> ProviderResponse:
    return text_response("拉伸不受支持；必须先生成网格。")


@pytest.mark.parametrize("text, operation, missing", [
    ("把这个截面加厚到 20 mm", "extrude_profiles", ()),
    ("extrude this profile by 10 mm", "extrude_profiles", ()),
    ("revolve around the x axis by 90 degrees", "revolve_profile", ()),
    ("沿路径扫掠", "path_sweep_profile", ("path",)),
    ("sweep this profile along A-B-C", "path_sweep_profile", ()),
    ("做扫掠六面体网格", "swept_mesh", ()),
    ("做扫掠", "sweep", ("sweep_type",)),
])
def test_geometry_intent_routes_by_operation_and_required_fields(text, operation, missing):
    hint = geometry_route_hint(text)
    assert hint is not None
    assert hint.requested_operation == operation
    assert hint.missing_fields == missing


@pytest.mark.parametrize("text", ["extrude this profile; any size", "拉伸成3d，尺寸任意"])
def test_explicit_arbitrary_size_needs_no_clarification(text):
    hint = geometry_route_hint(text)
    assert hint.allow_arbitrary_size
    assert not hint.missing_fields


def test_guard_allows_a_first_round_missing_field_question(tmp_path) -> None:
    provider = FakeProvider(
        [
            text_response("请提供拉伸高度。")
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "first-missing-field",
        provider,
        dynamic_tools=_DynamicRegistry(),
    )

    events = engine.send_message("拉伸成3d")

    assert len(provider.requests) == 1
    assert any(
        item.data.get("text") == "请提供拉伸高度。"
        for item in events
        if item.event is EngineEventType.MESSAGE_DELTA
    )


def test_guard_retry_continues_after_the_required_probe(tmp_path) -> None:
    provider = FakeProvider(
        [
            _refusal(),
            tool_response(
                ToolCall(
                    "corrected-probe",
                    "read_profile_transform_context",
                    {"part_id": "P1"},
                ),
            ),
            text_response("请提供拉伸高度。"),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "corrected-probe",
        provider,
        dynamic_tools=_DynamicRegistry(),
    )

    events = engine.send_message("拉伸成3d")

    assert len(provider.requests) == 3
    assert any(
        item.event is EngineEventType.TOOL_STARTED
        and item.data.get("tool") == "read_profile_transform_context"
        for item in events
    )
    visible = [
        item.data.get("text")
        for item in events
        if item.event is EngineEventType.MESSAGE_DELTA
    ]
    assert visible[-1] == "请提供拉伸高度。"


def test_guard_second_refusal_returns_local_recovery(tmp_path) -> None:
    provider = FakeProvider([_refusal(), _refusal()])
    engine = AgentSessionEngine(
        tmp_path / "agent-private",
        provider,
        dynamic_tools=_DynamicRegistry(),
    )

    events = engine.send_message("拉伸成3d")

    assert len(provider.requests) == 2
    visible = [
        item.data.get("text")
        for item in events
        if item.event is EngineEventType.MESSAGE_DELTA
    ]
    assert visible == ["当前几何能力检查未完成，请重试。"]

    english_provider = FakeProvider([_refusal(), _refusal()])
    english_engine = AgentSessionEngine(
        tmp_path / "english-recovery",
        english_provider,
        dynamic_tools=_DynamicRegistry(),
    )
    english_events = english_engine.send_message("extrude this profile")
    english_visible = [
        item.data.get("text")
        for item in english_events
        if item.event is EngineEventType.MESSAGE_DELTA
    ]
    assert english_visible == [
        "The current geometry capability check was not completed; please retry."
    ]


def test_guard_allows_typed_unsupported_and_mesh_intent(tmp_path) -> None:
    unsupported_provider = FakeProvider([_refusal()])
    unsupported_engine = AgentSessionEngine(
        tmp_path / "unsupported",
        unsupported_provider,
        dynamic_tools=_DynamicRegistry(dimension=3),
    )
    unsupported_events = unsupported_engine.send_message("拉伸成3d")
    assert len(unsupported_provider.requests) == 1
    assert any(
        item.data.get("text") == "拉伸不受支持；必须先生成网格。"
        for item in unsupported_events
        if item.event is EngineEventType.MESSAGE_DELTA
    )

    mesh_provider = FakeProvider([_refusal()])
    mesh_engine = AgentSessionEngine(
        tmp_path / "mesh",
        mesh_provider,
        dynamic_tools=_DynamicRegistry(),
    )
    mesh_engine.send_message("做扫掠六面体网格")
    assert len(mesh_provider.requests) == 1


def test_guard_does_not_intercept_missing_fields_diagnostics_or_cancel(
    tmp_path,
) -> None:
    cases = (
        ("拉伸成3d", "请提供拉伸高度。"),
        ("沿路径扫掠", "请提供路径。"),
        ("拉伸成3d", "profile-transform.source-not-planar: typed diagnostic"),
        ("拉伸成3d", "操作已取消。"),
    )
    for index, (request, response_text) in enumerate(cases):
        provider = FakeProvider(
            [
                text_response(response_text)
            ]
        )
        engine = AgentSessionEngine(
            tmp_path / f"no-guard-{index}",
            provider,
            dynamic_tools=_DynamicRegistry(),
        )
        events = engine.send_message(request)
        assert len(provider.requests) == 1
        assert any(
            item.data.get("text") == response_text
            for item in events
            if item.event is EngineEventType.MESSAGE_DELTA
        )


def test_guard_requires_published_transform_tools(tmp_path) -> None:
    provider = FakeProvider([_refusal()])
    engine = AgentSessionEngine(
        tmp_path / "tools-unpublished",
        provider,
        dynamic_tools=_DynamicRegistry(tools=()),
    )

    events = engine.send_message("拉伸成3d")

    assert len(provider.requests) == 1
    assert any(
        item.data.get("text") == "拉伸不受支持；必须先生成网格。"
        for item in events
        if item.event is EngineEventType.MESSAGE_DELTA
    )
