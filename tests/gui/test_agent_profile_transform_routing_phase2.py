from __future__ import annotations

import json

from fem_agent.authoring_runtime import AuthoringTurnSnapshot
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)
from fem_agent.providers.fake import FakeProvider
from fem_agent.routing import geometry_route_hint
from fem_agent.schemas import ToolResult


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
            document_id="document:phase2",
            session_id="native-phase2",
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
    return ProviderResponse(
        AssistantMessage("assistant", "拉伸不受支持；必须先生成网格。"),
        finish_reason="stop",
    )


def test_phase2_route_hint_distinguishes_transform_and_mesh_intents() -> None:
    extrusion = geometry_route_hint("把这个截面加厚到 20 mm")
    assert extrusion is not None
    assert extrusion.requested_operation == "extrude_profiles"
    assert extrusion.missing_fields == ()
    assert extrusion.mesh_prerequisite is False
    assert extrusion.required_probe_tool == "read_profile_transform_context"
    assert extrusion.required_prepare_tool == "prepare_profile_extrusion"

    missing_path = geometry_route_hint("沿路径扫掠")
    assert missing_path is not None
    assert missing_path.requested_operation == "path_sweep_profile"
    assert missing_path.missing_fields == ("path",)
    assert missing_path.required_prepare_tool == "prepare_profile_path_sweep"

    mesh = geometry_route_hint("做扫掠六面体网格")
    assert mesh is not None
    assert mesh.intent_kind == "meshing"
    assert mesh.requested_operation == "swept_mesh"
    assert mesh.required_probe_tool is None

    ambiguous = geometry_route_hint("做扫掠")
    assert ambiguous is not None
    assert ambiguous.intent_kind == "ambiguous"
    assert ambiguous.missing_fields == ("sweep_type",)


def test_phase2_route_hint_covers_bilingual_transform_fields_and_arbitrary_size() -> None:
    english_extrude = geometry_route_hint("extrude this profile by 10 mm")
    assert english_extrude is not None
    assert english_extrude.requested_operation == "extrude_profiles"
    assert english_extrude.missing_fields == ()

    english_revolve = geometry_route_hint(
        "revolve around the x axis by 90 degrees"
    )
    assert english_revolve is not None
    assert english_revolve.requested_operation == "revolve_profile"
    assert english_revolve.missing_fields == ()
    assert english_revolve.required_prepare_tool == "prepare_profile_revolution"

    english_path = geometry_route_hint("path sweep A-B-C")
    assert english_path is not None
    assert english_path.requested_operation == "path_sweep_profile"
    assert english_path.missing_fields == ()

    natural_english_path = geometry_route_hint(
        "sweep this profile along A-B-C"
    )
    assert natural_english_path is not None
    assert natural_english_path.requested_operation == "path_sweep_profile"
    assert natural_english_path.missing_fields == ()

    coordinate_path = geometry_route_hint(
        "path sweep through (0, 0, 0), (10, 0, 0), (10, 5, 2)"
    )
    assert coordinate_path is not None
    assert coordinate_path.requested_operation == "path_sweep_profile"
    assert coordinate_path.missing_fields == ()

    arbitrary = geometry_route_hint("extrude this profile; any size")
    assert arbitrary is not None
    assert arbitrary.allow_arbitrary_size is True
    assert arbitrary.missing_fields == ()

    arbitrary_zh = geometry_route_hint("拉伸成3d，尺寸任意")
    assert arbitrary_zh is not None
    assert arbitrary_zh.allow_arbitrary_size is True
    assert arbitrary_zh.missing_fields == ()

    fixed_frame = geometry_route_hint("path sweep with fixed-frame")
    assert fixed_frame is not None
    assert fixed_frame.missing_fields == ("path",)

    assert geometry_route_hint("普通聊天") is None


def test_phase2_guard_retry_allows_a_clarification_before_the_probe(tmp_path) -> None:
    provider = FakeProvider(
        [
            _refusal(),
            ProviderResponse(
                AssistantMessage("assistant", "请提供拉伸高度。"),
                finish_reason="stop",
            ),
        ]
    )
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
    assert visible == ["请提供拉伸高度。"]
    correction = provider.requests[1].messages[-1]
    assert correction.role == "system"
    assert "required_probe_tool" in (correction.content or "")
    assert not any(
        item.role == "system"
        and "Local geometry route correction" in (item.content or "")
        for item in engine._history
    )
    context = next(
        item.content
        for item in provider.requests[0].messages
        if item.role == "system" and "Current local state" in (item.content or "")
    )
    assert "route_hint" in context


def test_phase2_guard_allows_a_first_round_missing_field_question(tmp_path) -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", "请提供拉伸高度。"),
                finish_reason="stop",
            )
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


def test_phase2_guard_retry_continues_after_the_required_probe(tmp_path) -> None:
    provider = FakeProvider(
        [
            _refusal(),
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "corrected-probe",
                            "read_profile_transform_context",
                            {"part_id": "P1"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                AssistantMessage("assistant", "请提供拉伸高度。"),
                finish_reason="stop",
            ),
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
    assert "当前几何能力检查未完成，请重试。" not in visible


def test_phase2_refusal_correction_allows_read_only_discovery(tmp_path) -> None:
    feature_catalog = ToolDefinition(
        "read_geometry_feature_catalog",
        "Read bounded native geometry features.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    provider = FakeProvider(
        [
            _refusal(),
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "read-features",
                            "read_geometry_feature_catalog",
                            {},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "read-transform",
                            "read_profile_transform_context",
                            {"part_id": "P1"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                AssistantMessage("assistant", "请提供拉伸高度。"),
                finish_reason="stop",
            ),
        ]
    )
    registry = _DynamicRegistry(
        tools=(_TOOLS[0], feature_catalog, _TOOLS[1])
    )
    engine = AgentSessionEngine(
        tmp_path / "read-only-after-refusal",
        provider,
        dynamic_tools=registry,
    )

    events = engine.send_message("拉伸成3d")

    assert [name for name, _arguments in registry.calls] == [
        "read_geometry_feature_catalog",
        "read_profile_transform_context",
    ]
    visible = [
        item.data.get("text")
        for item in events
        if item.event is EngineEventType.MESSAGE_DELTA
    ]
    assert visible[-1] == "请提供拉伸高度。"
    assert "当前几何能力检查未完成，请重试。" not in visible


def test_phase2_guard_second_refusal_returns_local_recovery(tmp_path) -> None:
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


def test_phase2_guard_allows_typed_unsupported_and_mesh_intent(tmp_path) -> None:
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


def test_phase2_guard_does_not_intercept_missing_fields_diagnostics_or_cancel(
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
                ProviderResponse(
                    AssistantMessage("assistant", response_text),
                    finish_reason="stop",
                )
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


def test_phase2_guard_does_not_repeat_after_probe_call(tmp_path) -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "probe-1",
                            "read_profile_transform_context",
                            {"part_id": "P1"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            _refusal(),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "probe-called",
        provider,
        dynamic_tools=_DynamicRegistry(),
    )

    events = engine.send_message("拉伸成3d")

    assert len(provider.requests) == 2
    assert any(
        item.data.get("text") == "拉伸不受支持；必须先生成网格。"
        for item in events
        if item.event is EngineEventType.MESSAGE_DELTA
    )


def test_phase2_guard_requires_published_transform_tools(tmp_path) -> None:
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


def test_phase2_route_audit_is_bounded_json(tmp_path) -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", "请提供高度。"),
                finish_reason="stop",
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "audit",
        provider,
        dynamic_tools=_DynamicRegistry(),
    )
    engine.send_message("拉伸成3d")
    payload = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    hint = payload["entries"][0]["route_hint"]
    assert hint["requested_operation"] == "extrude_profiles"
    assert hint["mesh_prerequisite"] is False
