"""Deterministic owner-dispatch timeout and whitelist-consistency tests.

Phase 3 (方案 D) of the planar feature-chain plan: an owner-thread dispatch
deadline must return a structured ``ToolResult`` instead of leaking an
uncaught ``TimeoutError``, and the high-timeout whitelist must stay in sync
with the registered dynamic authoring tools.
"""

from __future__ import annotations

import threading

from PySide6.QtWidgets import QApplication

from fem.application import ModelSession
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.schemas import ToolResult
from fem_agent.tools.registry import ToolExecutionContext
import fem_gui.agent_runtime as agent_runtime
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from fem_gui.agent_runtime import QtAgentRuntime


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _controller(session: ModelSession):
    holder: dict[str, object] = {}

    def refresh() -> None:
        bridge.bind_snapshot(session.snapshot())
        controller = holder.get("controller")
        if controller is not None:
            controller.observe_binding(bridge.context)  # type: ignore[arg-type]

    bridge = AgentAuthoringBridge(SessionGeometryAuthoringPort(session, refresh))
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    holder["controller"] = controller
    return bridge, controller


def _dispatch_from_other_thread(runtime, name: str, key: str) -> ToolResult:
    """Call the cross-thread dispatch path from a non-owner thread.

    The owner thread never processes the queued ``authoringToolRequested``
    signal (no event loop spins here), so the owner side never completes and
    the dispatch deadline is the only outcome.
    """

    outcome: dict[str, object] = {}

    def run() -> None:
        outcome["result"] = runtime._dispatch_authoring_tool(
            name,
            {},
            ToolExecutionContext("timeout-session", 0, key),
        )

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    # The owner-dispatch deadline is 0.15s; the join bound only guards it
    # and must respect the GUI test real-wait policy.
    worker.join(timeout=2.0)
    assert not worker.is_alive(), "dispatch did not honor its deadline"
    result = outcome["result"]
    assert type(result) is ToolResult
    return result


def test_dispatch_timeout_returns_structured_result_and_session_continues(
    monkeypatch,
    tmp_path,
) -> None:
    _application()
    monkeypatch.setattr(
        agent_runtime, "_AUTHORING_TOOL_OWNER_TIMEOUT_SECONDS", 0.15
    )
    _bridge, controller = _controller(ModelSession())
    runtime = QtAgentRuntime(
        tmp_path / "agent-timeout",
        authoring_controller=controller,
    )

    result = _dispatch_from_other_thread(
        runtime, "read_authoring_context", "timeout-default"
    )

    assert result.ok is False
    assert result.session_id == "timeout-session"
    assert result.idempotency_key == "timeout-default"
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "WORKER_TIMEOUT"
    assert diagnostic.remediation
    assert result.data is not None
    error = result.data["error"]
    assert error["code"] == "authoring.owner-dispatch-timeout"
    assert error["tool"] == "read_authoring_context"
    assert error["timeout_seconds"] == 0.15
    assert error["elapsed_seconds"] >= 0.15
    assert error["advice"]

    # The session continues: the very next owner-thread dispatch succeeds.
    follow_up = runtime._dispatch_authoring_tool(
        "read_authoring_context",
        {},
        ToolExecutionContext("timeout-session", 0, "timeout-follow-up"),
    )
    assert follow_up.ok is True


def test_whitelisted_tools_get_the_long_budget_tier(monkeypatch, tmp_path) -> None:
    _application()
    monkeypatch.setattr(
        agent_runtime, "_AUTHORING_TOOL_OWNER_TIMEOUT_SECONDS", 0.15
    )
    monkeypatch.setattr(
        agent_runtime, "_AUTHORING_TOOL_LONG_OWNER_TIMEOUT_SECONDS", 0.45
    )
    _bridge, controller = _controller(ModelSession())
    runtime = QtAgentRuntime(
        tmp_path / "agent-timeout-tier",
        authoring_controller=controller,
    )

    whitelisted = "prepare_planar_construction_proposal"
    assert whitelisted in agent_runtime._AUTHORING_TOOL_LONG_TIMEOUT_NAMES
    result = _dispatch_from_other_thread(runtime, whitelisted, "timeout-long")

    assert result.ok is False
    assert result.data is not None
    error = result.data["error"]
    assert error["code"] == "authoring.owner-dispatch-timeout"
    assert error["timeout_seconds"] == 0.45
    assert error["elapsed_seconds"] >= 0.45


def test_long_timeout_whitelist_matches_registered_dynamic_tools() -> None:
    names = agent_runtime._AUTHORING_TOOL_LONG_TIMEOUT_NAMES
    assert names, "the long-timeout whitelist must be explicit and non-empty"
    assert names == {
        "prepare_planar_construction_proposal",
        "run_native_preflight",
    }
    _bridge, controller = _controller(ModelSession())
    # ``definitions`` is stage-gated (an empty session never publishes
    # ``run_native_preflight``), so the consistency check must cover the
    # full handler registry the controller was wired with.
    registered = set(controller._handlers) | {
        item.name for item in controller.definitions
    }
    assert names <= registered, (
        "every long-timeout tool must stay a registered dynamic tool"
    )
