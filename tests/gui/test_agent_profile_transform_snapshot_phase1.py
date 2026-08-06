from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import fem_agent.engine as engine_module
from fem_agent.authoring import (
    AuthoringContext,
    CapabilitySummary,
    LocalModelBinding,
    MeshSummary,
    PartSummary,
)
from fem_agent.authoring_runtime import (
    AUTHORING_TURN_SNAPSHOT_MAX_BYTES,
    AuthoringTurnSnapshot,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
)
from fem_agent.engine import AgentSessionEngine, EngineConfig
from fem_agent.providers.base import AssistantMessage, ProviderResponse, ToolCall
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import tool_schema_hash
from fem_gui.agent_runtime import QtAgentRuntime


def _context(revision: int = 4) -> AuthoringContext:
    return AuthoringContext(
        binding=LocalModelBinding(
            "document:phase1",
            "native-phase1",
            revision,
            "native",
            True,
        ),
        model_name="phase1-model",
        active_part_id="part-profile",
        parts=(
            PartSummary(
                "part-profile",
                "profile body",
                "planar_sketch",
                2,
                False,
            ),
        ),
        mesh=MeshSummary(True, True),
        capabilities=(
            CapabilitySummary("read_authoring_context", True),
            CapabilitySummary("prepare_geometry_edit", True),
            CapabilitySummary("hidden_operation", False),
        ),
    )


def test_phase1_snapshot_binds_revision_and_preserves_owner_cache() -> None:
    context = _context()
    controller = AuthoringWorkflowController(lambda: context, {})

    controller.observe_binding(context)
    names = ("read_authoring_context", "prepare_geometry_edit")
    snapshot = controller.set_published_tool_names(names)

    assert snapshot.available
    assert snapshot.source_kind == "native"
    assert snapshot.workflow_stage == controller.stage.value
    assert snapshot.document_id == "document:phase1"
    assert snapshot.session_id == "native-phase1"
    assert snapshot.session_revision == 4
    assert snapshot.active_part_id == "part-profile"
    assert snapshot.active_part_dimension == 2
    assert snapshot.active_part_recipe_kind == "planar_sketch"
    assert snapshot.active_part_suppressed is False
    assert snapshot.mesh_present and snapshot.mesh_current
    assert snapshot.enabled_capabilities == names
    assert snapshot.published_tool_names == names
    assert controller.provider_snapshot == snapshot

    with pytest.raises(AttributeError):
        snapshot.workflow_stage = "stale"  # type: ignore[misc]

    controller.invalidate_binding("document switched")
    stale = controller.turn_snapshot
    assert stale.snapshot_generation > snapshot.snapshot_generation
    assert not stale.available
    assert stale.document_id is None
    assert stale.session_id is None
    assert stale.session_revision is None
    assert not controller.provider_snapshot.available

    fresh_context = _context(revision=5)
    controller.observe_binding(fresh_context)
    fresh = controller.set_published_tool_names(names)
    assert fresh.snapshot_generation > stale.snapshot_generation
    assert fresh.available
    assert fresh.session_revision == 5
    assert fresh.workflow_stage == AuthoringWorkflowStage.STALE.value


def test_phase1_snapshot_is_bounded_and_deterministically_clipped() -> None:
    names = (
        "read_authoring_context",
        "prepare_geometry_edit",
        *(f"operation_{index:03d}_" + "x" * 96 for index in range(96)),
    )
    first = AuthoringTurnSnapshot(
        available=True,
        source_kind="native",
        workflow_stage="geometry_ready",
        document_id="document:phase1",
        session_id="native-phase1",
        session_revision=4,
        active_part_id="part-profile",
        active_part_dimension=2,
        active_part_recipe_kind="planar_sketch",
        active_part_suppressed=False,
        mesh_present=True,
        mesh_current=True,
        enabled_capabilities=names,
        published_tool_names=names,
        snapshot_generation=8,
    )
    second = AuthoringTurnSnapshot(
        available=True,
        source_kind="native",
        workflow_stage="geometry_ready",
        document_id="document:phase1",
        session_id="native-phase1",
        session_revision=4,
        active_part_id="part-profile",
        active_part_dimension=2,
        active_part_recipe_kind="planar_sketch",
        active_part_suppressed=False,
        mesh_present=True,
        mesh_current=True,
        enabled_capabilities=names,
        published_tool_names=names,
        snapshot_generation=8,
    )

    first_payload = first.to_provider_dict()
    encoded_compact = json.dumps(
        first_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded_default = json.dumps(first_payload, ensure_ascii=False).encode("utf-8")
    assert len(encoded_compact) <= AUTHORING_TURN_SNAPSHOT_MAX_BYTES
    assert len(encoded_default) <= AUTHORING_TURN_SNAPSHOT_MAX_BYTES
    assert first.truncated
    assert first == second
    assert first.enabled_capabilities[:2] == names[:2]
    assert first.published_tool_names[:2] == names[:2]


def test_phase1_unavailable_refresh_does_not_reuse_previous_document() -> None:
    current: object = _context()
    controller = AuthoringWorkflowController(lambda: current, {})
    controller.refresh_turn_snapshot(("read_authoring_context",))
    assert controller.turn_snapshot.available

    current = {"binding": {"document_id": "document:other"}}
    unavailable = controller.refresh_turn_snapshot(("read_authoring_context",))

    assert not unavailable.available
    assert unavailable.document_id is None
    assert unavailable.session_id is None
    assert controller.provider_snapshot.available is False


def test_phase1_stale_review_projects_the_new_typed_binding() -> None:
    old_context = _context()
    new_context = _context(revision=5)
    controller = AuthoringWorkflowController(lambda: old_context, {})
    controller.observe_binding(old_context)
    controller.set_published_tool_names(("read_authoring_context",))
    controller._stage = AuthoringWorkflowStage.REVIEW_PENDING
    controller._review_binding = (
        old_context.binding.document_id,
        old_context.binding.session_id,
        old_context.binding.session_revision,
    )

    assert controller.stale_review_for_binding(new_context)
    snapshot = controller.turn_snapshot
    assert snapshot.available
    assert snapshot.document_id == new_context.binding.document_id
    assert snapshot.session_id == new_context.binding.session_id
    assert snapshot.session_revision == new_context.binding.session_revision


def test_phase1_runtime_publish_failures_atomically_drop_provider_cache(
    tmp_path,
    monkeypatch,
) -> None:
    context = _context()
    controller = AuthoringWorkflowController(lambda: context, {})
    controller.observe_binding(context)
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=FakeProvider,
        authoring_controller=controller,
    )
    try:
        available = runtime.refresh_authoring_turn_snapshot_from_gui()
        assert available.available
        previous_generation = available.snapshot_generation

        def fail_publish(_names):
            raise RuntimeError("publish failed")

        monkeypatch.setattr(controller, "set_published_tool_names", fail_publish)
        runtime._try_publish_authoring_tool_cache_owner_thread()
        failed = runtime.authoring_turn_snapshot
        assert not failed.available
        assert failed.snapshot_generation > previous_generation
        assert runtime._authoring_tool_definitions() == ()

        monkeypatch.undo()
        runtime.refresh_authoring_turn_snapshot_from_gui()
        assert runtime.authoring_turn_snapshot.available

        def fail_refresh(_names=()):
            raise RuntimeError("refresh failed")

        monkeypatch.setattr(controller, "refresh_turn_snapshot", fail_refresh)
        controller.invalidate_turn_snapshot()
        runtime._try_refresh_authoring_turn_snapshot_from_gui()
        failed_refresh = runtime.authoring_turn_snapshot
        assert not failed_refresh.available
        assert runtime._authoring_tool_definitions() == ()
    finally:
        runtime.shutdown()


def test_phase1_runtime_binding_invalidation_hides_old_tools_until_rebind(
    tmp_path,
) -> None:
    current = _context()
    controller = AuthoringWorkflowController(lambda: current, {})
    controller.observe_binding(current)
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", content="invalidated"),
                finish_reason="stop",
            ),
            ProviderResponse(
                AssistantMessage("assistant", content="rebound"),
                finish_reason="stop",
            ),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / "agent-private-binding-invalidation",
        provider_factory=lambda: provider,
        authoring_controller=controller,
    )
    try:
        published = runtime.refresh_authoring_turn_snapshot_from_gui()
        old_names = {
            item.name for item in runtime._authoring_tool_definitions()
        }
        assert published.available
        assert old_names

        previous_generation = published.snapshot_generation
        runtime.invalidate_authoring_binding_from_gui("document switched")
        invalidated = runtime.authoring_turn_snapshot
        assert not invalidated.available
        assert invalidated.snapshot_generation > previous_generation
        assert runtime._authoring_tool_definitions() == ()

        engine = runtime._ensure_engine()
        engine.send_message("after document switch")
        first_request_names = {item.name for item in provider.requests[0].tools}
        assert not old_names.intersection(first_request_names)

        controller.reset_for_binding()
        controller.observe_binding(current)
        rebound = runtime.refresh_authoring_turn_snapshot_from_gui()
        assert rebound.available
        assert runtime._authoring_tool_definitions()

        engine.send_message("after rebind")
        second_request_names = {item.name for item in provider.requests[1].tools}
        assert old_names.intersection(second_request_names)
    finally:
        runtime.shutdown()


def test_phase1_engine_context_and_audit_are_round_scoped_and_safe(tmp_path) -> None:
    context = _context()
    controller = AuthoringWorkflowController(lambda: context, {})
    controller.observe_binding(context)
    dynamic_names = tuple(item.name for item in controller.definitions)
    controller.set_published_tool_names(dynamic_names)

    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", content="已读取当前建模状态。"),
                finish_reason="stop",
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "agent-private",
        provider,
        dynamic_tools=controller,
    )

    events = engine.send_message("读取当前建模状态")
    assert events
    request = provider.requests[0]
    state_message = next(
        item
        for item in request.messages
        if item.role == "system"
        and item.content.startswith("Current local state")
    )
    state = json.loads(state_message.content.split(": ", 1)[1])
    snapshot = state["authoring_turn_snapshot"]
    assert snapshot["available"] is True
    assert snapshot["active_part_id"] == "part-profile"
    assert snapshot["active_part_dimension"] == 2
    assert dynamic_names
    assert dynamic_names[0] in snapshot["published_tool_names"]

    audit = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    assert audit["schema_version"] == 2
    assert len(audit["entries"]) == 1
    entry = audit["entries"][0]
    assert set(entry) == {
        "session_id",
        "workflow_stage",
        "revision",
        "published_tool_names",
        "schema_hashes",
        "route_hint",
        "tool_call_flags",
    }
    assert entry["tool_call_flags"] == {
        "provider_called": False,
        "called_tool_names": [],
        "read_tool_called": False,
        "prepare_tool_called": False,
    }
    assert entry["schema_hashes"][dynamic_names[0]] == tool_schema_hash(
        next(item for item in request.tools if item.name == dynamic_names[0])
    )
    encoded = json.dumps(audit, ensure_ascii=False)
    assert "phase1-model" not in encoded
    assert "document:phase1" not in encoded
    assert "Current local state" not in encoded


def _show_capabilities_response(call_id: str) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage(
            "assistant",
            tool_calls=(ToolCall(call_id, "show_capabilities", {}),),
        ),
        finish_reason="tool_calls",
    )


def test_phase1_audit_batches_rounds_into_one_atomic_write(tmp_path, monkeypatch) -> None:
    provider = FakeProvider(
        [
            _show_capabilities_response("round-1"),
            _show_capabilities_response("round-2"),
            ProviderResponse(
                AssistantMessage("assistant", content="完成。"),
                finish_reason="stop",
            ),
        ]
    )
    engine = AgentSessionEngine(tmp_path / "agent-private", provider)
    writes: list[Path] = []
    original_write = engine_module.atomic_write_json

    def capture_write(path, payload, *, overwrite=False):
        if Path(path).name == "tool-audit.json":
            writes.append(Path(path))
        return original_write(path, payload, overwrite=overwrite)

    monkeypatch.setattr(engine_module, "atomic_write_json", capture_write)
    assert not engine._audit_path().exists()

    engine.send_message("检查能力")

    assert writes == [engine._audit_path()]
    audit = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    assert len(audit["entries"]) == 3
    assert [
        item["tool_call_flags"]["called_tool_names"]
        for item in audit["entries"]
    ] == [["show_capabilities"], ["show_capabilities"], []]


def test_phase1_deferred_audit_flush_is_explicit_and_close_safe(
    tmp_path,
    monkeypatch,
) -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", content="done"),
                finish_reason="stop",
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "agent-private-deferred-audit",
        provider,
        defer_audit_persistence=True,
    )
    writes: list[Path] = []
    original_write = engine_module.atomic_write_json

    def capture_write(path, payload, *, overwrite=False):
        if Path(path).name == "tool-audit.json":
            writes.append(Path(path))
        return original_write(path, payload, overwrite=overwrite)

    monkeypatch.setattr(engine_module, "atomic_write_json", capture_write)
    engine.send_message("defer audit")
    assert writes == []
    assert not engine._audit_path().exists()

    engine.flush_round_audit()
    assert writes == [engine._audit_path()]
    audit = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    assert len(audit["entries"]) == 1

    engine.close_session()
    assert writes == [engine._audit_path()]


@pytest.mark.parametrize("terminal", ["provider_error", "tool_limit"])
def test_phase1_audit_batch_flushes_on_terminal_provider_paths(
    tmp_path,
    monkeypatch,
    terminal,
) -> None:
    if terminal == "provider_error":
        provider = FakeProvider(
            [_show_capabilities_response("before-error"), RuntimeError("boom")]
        )
        config = None
    else:
        provider = FakeProvider(
            [
                _show_capabilities_response("before-limit"),
                _show_capabilities_response("over-limit"),
            ]
        )
        config = EngineConfig(max_tool_calls=1)
    engine = AgentSessionEngine(
        tmp_path / f"agent-private-{terminal}",
        provider,
        config=config,
    )
    writes: list[Path] = []
    original_write = engine_module.atomic_write_json

    def capture_write(path, payload, *, overwrite=False):
        if Path(path).name == "tool-audit.json":
            writes.append(Path(path))
        return original_write(path, payload, overwrite=overwrite)

    monkeypatch.setattr(engine_module, "atomic_write_json", capture_write)
    events = engine.send_message("触发终止路径")

    assert events
    assert writes == [engine._audit_path()]
    audit = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    assert len(audit["entries"]) == 1
    assert audit["entries"][0]["tool_call_flags"]["provider_called"] is True


class _BlockingRoundProvider:
    provider_name = "fake"
    model_name = "blocking-test"

    def __init__(self) -> None:
        self.calls = 0
        self.second_call_started = threading.Event()
        self.release_second_call = threading.Event()

    def complete(self, messages, tools):
        del messages, tools
        self.calls += 1
        if self.calls == 1:
            return _show_capabilities_response("before-cancel")
        self.second_call_started.set()
        self.release_second_call.wait(timeout=5)
        return ProviderResponse(
            AssistantMessage("assistant", content="late"),
            finish_reason="stop",
        )


def test_phase1_audit_batch_flushes_when_cancelled_and_closed(tmp_path, monkeypatch) -> None:
    provider = _BlockingRoundProvider()
    engine = AgentSessionEngine(tmp_path / "agent-private", provider)
    writes: list[Path] = []
    original_write = engine_module.atomic_write_json

    def capture_write(path, payload, *, overwrite=False):
        if Path(path).name == "tool-audit.json":
            writes.append(Path(path))
        return original_write(path, payload, overwrite=overwrite)

    monkeypatch.setattr(engine_module, "atomic_write_json", capture_write)
    result: list[tuple[object, ...]] = []
    worker = threading.Thread(
        target=lambda: result.append(engine.send_message("取消并关闭")),
        daemon=True,
    )
    worker.start()
    assert provider.second_call_started.wait(timeout=5)
    closer = threading.Thread(target=engine.close_session, daemon=True)
    closer.start()
    provider.release_second_call.set()
    worker.join(timeout=5)
    closer.join(timeout=5)

    assert not worker.is_alive()
    assert not closer.is_alive()
    assert result
    assert writes == [engine._audit_path()]
    audit = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    assert len(audit["entries"]) == 1
    assert audit["entries"][0]["tool_call_flags"]["called_tool_names"] == [
        "show_capabilities"
    ]
