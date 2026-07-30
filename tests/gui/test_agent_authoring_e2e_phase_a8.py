from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton

from fem.application import ModelSession, UnitContext
from fem.geometry import (
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    legacy_sketch_to_strict,
)
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import (
    FakeAuthoringPort,
    ProposalState,
)
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
)
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ToolCall,
)
from fem_agent.providers.fake import FakeProvider
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from fem_gui.agent_events import EventType
from fem_gui.agent_runtime import QtAgentRuntime
from fem_gui.main_window import FEMMainWindow
from fem_gui.widgets.agent_chat import AgentChatDrawer
from tests.test_agent_authoring_phase_a4 import _plate_model
from tests.test_agent_authoring_phase_a8 import (
    _context,
    _controller,
    _requirements_for,
)


def _tool_response(call: ToolCall) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage("assistant", tool_calls=(call,)),
        finish_reason="tool_calls",
    )


def _text_response(text: str) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage("assistant", content=text),
        finish_reason="stop",
    )


def _generic_a4_session() -> ModelSession:
    session = ModelSession()
    recipe = legacy_sketch_to_strict(
        SketchGeometry(
            "草图-通用孔板",
            (
                SketchRectangle("material", 0.0, 0.0, 10.0, 6.0),
                SketchCircle("cut", 6.5, 2.0, 1.0),
            ),
        )
    )
    session.create_native_project_with_first_part(
        "模型-通用孔板",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="部件-通用孔板",
    )
    task = session.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0),
        "a" * 64,
        expected_session_revision=session.session_revision,
    )
    assert session.accept_agent_generated_model(
        task.token,
        _plate_model(),
    ).accepted
    return session


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, *, timeout_ms: int = 15_000) -> None:
    deadline = time.monotonic() + timeout_ms / 1000
    application = _application()
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    application.processEvents()
    assert predicate()


def test_a8_qt_runtime_dispatches_dynamic_tool_on_owner_thread(
    tmp_path,
) -> None:
    _application()
    owner_thread = threading.get_ident()
    reader_threads: list[int] = []

    def read_context():
        reader_threads.append(threading.get_ident())
        return {
            "schema_version": "1.0",
            "binding": {
                "document_id": "document:a8",
                "session_id": "native-a8",
                "session_revision": 0,
                "source_kind": "blank",
                "supported": True,
            },
            "model_name": None,
            "parts": [],
            "mesh": {"generated": False},
            "definitions": {"analysis_step_count": 0},
        }

    controller = AuthoringWorkflowController(read_context, {})
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall("context-a8", "read_authoring_context", {})
            ),
            _text_response("已读取有界建模上下文。"),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: provider,
        authoring_controller=controller,
    )
    events = []
    runtime.agentEventReady.connect(events.append)

    assert runtime.send_message("读取当前建模上下文")
    _wait_until(lambda: not runtime.busy)

    assert reader_threads == [owner_thread]
    assert provider.requests
    available_names = {
        item.name for item in provider.requests[0].tools
    }
    assert "read_authoring_context" in available_names
    assert "set_authoring_requirements" in available_names
    assert "set_unit_context" not in available_names
    assert any(
        event.event_type.value == "tool_requested"
        and event.payload["tool_name"] == "read_authoring_context"
        for event in events
    )
    assert any(event.event_type.value == "tool_result" for event in events)
    runtime.shutdown()


def test_a8_qt_runtime_gui_terminal_notifications_require_owner_thread(
    tmp_path,
) -> None:
    _application()
    controller = AuthoringWorkflowController(lambda: {}, {})
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=FakeProvider,
        authoring_controller=controller,
    )
    controller._stage = AuthoringWorkflowStage.MESH_PENDING
    controller._pending_operation = "mesh"

    runtime.record_authoring_proposal_state_from_gui(
        "mesh",
        ProposalState.FAILED,
        "deterministic mesh failure",
    )
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    assert controller.terminal_records[-1].state == "failed"

    failures: list[BaseException] = []

    def cross_thread_call() -> None:
        try:
            runtime.invalidate_authoring_binding_from_gui(
                "cross-thread document switch"
            )
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=cross_thread_call)
    thread.start()
    thread.join(timeout=5.0)

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    runtime.shutdown()


@pytest.mark.parametrize(
    ("abort_kind", "error_type"),
    [
        ("timeout", TimeoutError),
        ("shutdown", RuntimeError),
    ],
)
def test_a8_delayed_authoring_invocation_cannot_run_after_abort(
    tmp_path,
    monkeypatch,
    abort_kind,
    error_type,
) -> None:
    _application()
    calls = []

    def handler(_arguments, _controller):
        calls.append("mutated")
        return AuthoringToolOutcome("unexpected", {"state": "succeeded"})

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"prepare_geometry_proposal": handler},
    )
    controller._stage = AuthoringWorkflowStage.GEOMETRY_READY
    runtime = QtAgentRuntime(
        tmp_path / f"agent-private-{abort_kind}",
        provider_factory=FakeProvider,
        authoring_controller=controller,
    )
    errors = []

    if abort_kind == "timeout":
        monkeypatch.setattr(
            "fem_gui.agent_runtime._AUTHORING_TOOL_OWNER_TIMEOUT_SECONDS",
            0.05,
        )

    def dispatch_from_worker() -> None:
        try:
            runtime._dispatch_authoring_tool(
                "prepare_geometry_proposal",
                {},
                ToolExecutionContext("session-a8", 0, f"{abort_kind}-a8"),
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=dispatch_from_worker)
    worker.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with runtime._lock:
            if runtime._authoring_invocations:
                break
        time.sleep(0.005)
    else:
        raise AssertionError("authoring invocation was not queued")

    if abort_kind == "shutdown":
        runtime.shutdown()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], error_type)
    assert calls == []

    _application().processEvents()
    assert calls == []
    if abort_kind == "timeout":
        runtime.shutdown()


def test_a8_geometry_operation_emits_the_only_local_confirmation_card(
    tmp_path,
) -> None:
    _application()
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    controller, _calls, _queries = _controller(bridge)
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "requirements-a8",
                            "set_authoring_requirements",
                            {
                                "turn_id": "turn-a8",
                                "requirements": _requirements_for("geometry"),
                            },
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
                            "geometry-a8",
                            "prepare_geometry_proposal",
                            {},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            _text_response("几何创建等待确认。"),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / "agent-private-operation-card",
        provider_factory=lambda: provider,
        authoring_controller=controller,
    )
    events = []
    runtime.agentEventReady.connect(events.append)

    assert runtime.send_message("创建这个带孔平板")
    _wait_until(lambda: not runtime.busy)

    proposals = [
        event
        for event in events
        if event.event_type is EventType.PROPOSAL_REQUESTED
    ]
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.payload["proposal_kind"] == "geometry"
    assert proposal.payload["confirm_label"] == "加入模型"
    assert controller.pending_review is None
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    runtime.shutdown()


@pytest.mark.parametrize(
    ("button_name", "expected_stage"),
    [
        (
            "agentChatProposalAcceptButton",
            AuthoringWorkflowStage.GEOMETRY_PENDING,
        ),
        (
            "agentChatProposalRejectButton",
            AuthoringWorkflowStage.GEOMETRY_READY,
        ),
    ],
)
def test_a8_geometry_card_buttons_reach_gui_boundary(
    tmp_path,
    button_name,
    expected_stage,
) -> None:
    application = _application()
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    controller, _calls, _queries = _controller(bridge)
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "requirements-a8",
                            "set_authoring_requirements",
                            {
                                "turn_id": "turn-a8",
                                "requirements": _requirements_for("geometry"),
                            },
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
                            "geometry-a8",
                            "prepare_geometry_proposal",
                            {},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            _text_response("几何创建等待确认。"),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / f"agent-private-{button_name}",
        provider_factory=lambda: provider,
        authoring_controller=controller,
    )
    drawer = AgentChatDrawer(
        agent_runtime=runtime,
        authoring_bridge=bridge,
    )
    drawer.resize(720, 800)
    drawer.show()
    assert runtime.send_message("创建这个带孔平板")
    _wait_until(lambda: not runtime.busy)

    button = drawer.findChild(QToolButton, button_name)
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    assert button is not None and button.isEnabled()

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    application.processEvents()

    assert controller.stage is expected_stage, drawer.composer_hint.text()
    drawer.close()
    runtime.shutdown()


def test_a8_new_agent_session_discards_pending_geometry_proposal(
    tmp_path,
) -> None:
    _application()
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    controller, _calls, _queries = _controller(bridge)
    controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-a8",
            "requirements": _requirements_for("geometry"),
        },
        ToolExecutionContext("session-a8", 0, "requirements-before-reset"),
    )
    prepared = controller.dispatch(
        "prepare_geometry_proposal",
        {},
        ToolExecutionContext("session-a8", 0, "geometry-before-reset"),
    )
    proposal_id = prepared.data["proposal_id"]
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    assert bridge.state(proposal_id) is ProposalState.PENDING_CONFIRMATION

    runtime = QtAgentRuntime(
        tmp_path / "agent-private-session-reset",
        provider_factory=FakeProvider,
        authoring_controller=controller,
    )
    drawer = AgentChatDrawer(
        agent_runtime=runtime,
        authoring_bridge=bridge,
    )

    assert runtime.new_session()
    _wait_until(lambda: not runtime.busy)

    assert controller.stage is AuthoringWorkflowStage.REQUIREMENTS
    assert controller.ledger.entries == ()
    assert bridge.state(proposal_id) is ProposalState.STALE
    drawer.close()
    runtime.shutdown()


def test_a8_production_main_window_injects_the_real_controller() -> None:
    _application()
    window = FEMMainWindow()
    runtime = window.viewport_panel.agent_chat_drawer.agent_runtime

    assert runtime.authoring_controller is window.agent_authoring_controller
    assert runtime.authoring_controller is not None
    assert "read_authoring_context" in {
        item.name for item in runtime.authoring_controller.definitions
    }

    runtime.shutdown()
    window.close()


def test_a8_production_geometry_waits_for_one_gui_acceptance() -> None:
    _application()
    window = FEMMainWindow()
    controller = window.agent_authoring_controller
    bridge = window.agent_authoring_bridge
    before = window.session.snapshot()
    assert before.source_kind is None

    recorded = controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-a8",
            "requirements": _requirements_for("geometry"),
        },
        ToolExecutionContext("session-a8", 0, "requirements-a8"),
    )
    prepared = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "偏心孔板",
            "geometry": {
                "kind": "planar_profiles",
                "profiles": [
                    {
                        "kind": "rectangle",
                        "x": 0.0,
                        "y": 0.0,
                        "width": 120.0,
                        "height": 60.0,
                    },
                    {
                        "kind": "circle",
                        "center_x": 68.0,
                        "center_y": 26.0,
                        "radius": 8.0,
                    },
                ],
            },
        },
        ToolExecutionContext("session-a8", 0, "geometry-a8"),
    )
    proposal_id = prepared.data["proposal_id"]

    draft_state = window.session.snapshot()
    assert recorded.ok and prepared.ok
    assert controller.pending_review is None
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    assert draft_state.session_revision == before.session_revision
    assert draft_state.parts == ()

    accepted = bridge.accept_from_gui_control(proposal_id)
    controller.record_proposal_state("geometry", accepted.state)
    after = window.session.snapshot()

    assert accepted.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    assert after.source_kind == "native"
    assert after.session_revision == before.session_revision + 1
    assert [part.name for part in after.parts] == ["部件-偏心孔板"]

    window.viewport_panel.agent_chat_drawer.agent_runtime.shutdown()
    window.close()


def test_a8_production_mesh_uses_catalogued_generic_local_refinement() -> None:
    session = ModelSession()
    recipe = legacy_sketch_to_strict(
        SketchGeometry(
            "草图-通用孔板",
            (
                SketchRectangle("material", 0.0, 0.0, 10.0, 6.0),
                SketchCircle("cut", 6.5, 2.0, 1.0),
            ),
        )
    )
    session.create_native_project_with_first_part(
        "模型-通用孔板",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="部件-通用孔板",
    )
    port = SessionGeometryAuthoringPort(session, lambda: None)
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    controller._stage = AuthoringWorkflowStage.MESH_READY
    assert controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-mesh-refinement",
            "requirements": _requirements_for("mesh"),
        },
        ToolExecutionContext("session-a8", 0, "mesh-requirements"),
    ).ok

    catalog = controller.dispatch(
        "read_mesh_refinement_context",
        {},
        ToolExecutionContext("session-a8", 0, "mesh-refinement-context"),
    )
    assert catalog.ok, catalog.diagnostics[0].message
    available_ids = {
        item["logical_id"] for item in catalog.data["entities"]
    }
    assert "edge:C5" in available_ids
    target_radius_entity = next(
        item
        for item in catalog.data["entities"]
        if item["logical_id"] == "edge:C5"
    )
    line_entity = next(
        item
        for item in catalog.data["entities"]
        if item["logical_id"] == "edge:L1"
    )
    assert "target_radius" in target_radius_entity[
        "allowed_falloff_references"
    ]
    assert line_entity["allowed_falloff_references"] == ["global_size"]

    rejected = controller.dispatch(
        "prepare_mesh_proposal",
        {
            "local_refinements": [
                {
                    "target": "edge:not-current",
                    "size": 0.2,
                    "falloff": {
                        "reference": "global_size",
                        "start_factor": 0.0,
                        "end_factor": 1.5,
                    },
                }
            ]
        },
        ToolExecutionContext("session-a8", 0, "mesh-refinement-stale-target"),
    )
    assert not rejected.ok
    assert controller.stage is AuthoringWorkflowStage.MESH_READY

    unsupported_radius = controller.dispatch(
        "prepare_mesh_proposal",
        {
            "local_refinements": [
                {
                    "target": "edge:C1",
                    "size": 0.2,
                    "falloff": {
                        "reference": "target_radius",
                        "start_factor": 0.25,
                        "end_factor": 2.0,
                    },
                }
            ]
        },
        ToolExecutionContext("session-a8", 0, "mesh-radius-unsupported"),
    )
    assert not unsupported_radius.ok
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    assert bridge._records == {}

    prepared = controller.dispatch(
        "prepare_mesh_proposal",
        {
            "local_refinements": [
                {
                    "target": "edge:C5",
                    "size": 0.2,
                    "falloff": {
                        "reference": "target_radius",
                        "start_factor": 0.25,
                        "end_factor": 2.0,
                    },
                }
            ]
        },
        ToolExecutionContext("session-a8", 0, "mesh-refinement-proposal"),
    )

    assert catalog.ok and prepared.ok
    assert prepared.data["local_refinements"] == [
        {
            "target": "edge:C5",
            "size": 0.2,
            "falloff": {
                "reference": "target_radius",
                "start_factor": 0.25,
                "end_factor": 2.0,
            },
        }
    ]
    proposal = bridge._records[prepared.data["proposal_id"]].proposal
    mesh_intent = proposal.operations[0].parameters["mesh_intent"]
    assert mesh_intent["local_controls"][0]["target"] == "edge:C5"
    assert controller.stage is AuthoringWorkflowStage.MESH_PENDING


def test_a8_existing_mesh_can_be_remeshed_without_preemptive_deletion() -> None:
    session = _generic_a4_session()
    before = session.snapshot()
    started = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        start_mesh_task=lambda request: started.append(request) or True,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(before)
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )

    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    initial_tools = {tool.name for tool in controller.definitions}
    assert {
        "set_authoring_requirements",
        "read_mesh_refinement_context",
    } <= initial_tools
    assert "prepare_mesh_proposal" not in initial_tools
    recorded = controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-remesh",
            "requirements": {
                "mesh_cell_shape": "triangle",
                "mesh_order": 1,
                "mesh_global_size": 0.8,
            },
        },
        ToolExecutionContext(
            before.session_id,
            before.session_revision,
            "remesh-requirements",
        ),
    )
    assert recorded.ok

    first = controller.dispatch(
        "prepare_mesh_proposal",
        {},
        ToolExecutionContext(
            before.session_id,
            before.session_revision,
            "remesh-reject",
        ),
    )
    assert first.ok
    unchanged = session.snapshot()
    assert unchanged.session_revision == before.session_revision
    assert unchanged.mesh_current
    assert unchanged.mesh_settings == before.mesh_settings

    rejected = bridge.reject_from_gui_control(first.data["proposal_id"])
    controller.record_proposal_state("mesh", rejected.state)
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    still_unchanged = session.snapshot()
    assert still_unchanged.session_revision == before.session_revision
    assert still_unchanged.mesh_current
    assert still_unchanged.mesh_settings == before.mesh_settings

    second = controller.dispatch(
        "prepare_mesh_proposal",
        {},
        ToolExecutionContext(
            before.session_id,
            before.session_revision,
            "remesh-accept",
        ),
    )
    running = bridge.accept_from_gui_control(second.data["proposal_id"])
    assert running.state is ProposalState.RUNNING
    assert session.snapshot().mesh_current
    assert session.snapshot().mesh_settings == before.mesh_settings

    accepted = port.accept_mesh_result(
        second.data["proposal_id"],
        _plate_model(),
    )
    assert accepted.accepted
    terminal = bridge._records[second.data["proposal_id"]]
    controller.record_proposal_state("mesh", terminal.state, terminal.message)
    after = session.snapshot()

    assert terminal.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    assert after.session_revision == before.session_revision + 1
    assert after.mesh_current
    assert after.mesh_settings is not None
    assert after.mesh_settings.size == 0.8
    assert len(started) == 1


def test_a8_production_geometry_edit_adds_second_hole_in_place() -> None:
    session = _generic_a4_session()
    before = session.snapshot()
    part_id = str(before.parts[0].id)
    refreshes: list[int] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: refreshes.append(session.session_revision),
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(before)
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    controller._stage = AuthoringWorkflowStage.DEFINITIONS_READY

    tool_names = {tool.name for tool in controller.definitions}
    assert {
        "read_geometry_edit_context",
        "prepare_geometry_edit",
    } <= tool_names
    context_result = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": part_id},
        ToolExecutionContext("session-a8", 0, "geometry-edit-context"),
    )
    prepared = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": part_id,
            "edit": {
                "operation": "add_circle",
                "center_x": 6.5,
                "center_y": 4.0,
                "radius": 0.5,
            },
        },
        ToolExecutionContext("session-a8", 0, "geometry-edit-proposal"),
    )

    assert context_result.ok and prepared.ok
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    proposal_id = prepared.data["proposal_id"]
    accepted = bridge.accept_from_gui_control(proposal_id)
    controller.record_proposal_state("geometry", accepted.state)
    after = session.snapshot()

    assert accepted.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    assert len(after.parts) == 1
    assert str(after.parts[0].id) == part_id
    assert type(after.parts[0].geometry_recipe) is SketchGeometry
    assert len(
        [
            curve
            for curve in after.parts[0].geometry_recipe.curves
            if isinstance(curve, SketchCircle)
        ]
    ) == 2
    assert not after.model_current
    assert not after.mesh_current
    assert refreshes == [after.session_revision]


def test_a8_direct_definition_actions_apply_one_by_one_and_refresh_gui() -> None:
    session = _generic_a4_session()
    projections = []
    controller_holder = {}

    def project_definition_delta(delta) -> None:
        projections.append(delta)
        stale_ids = bridge.bind_snapshot(session.snapshot())
        controller_holder["controller"].observe_binding(
            bridge.context,
            proposal_staled=bool(stale_ids),
        )

    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        apply_definition_delta=project_definition_delta,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    controller_holder["controller"] = controller
    controller._stage = AuthoringWorkflowStage.DEFINITIONS_READY
    before_revision = session.session_revision

    topology = controller.dispatch(
        "read_model_topology_context",
        {},
        ToolExecutionContext(
            "session-a8",
            session.session_revision,
            "definition-topology",
        ),
    )
    assert topology.ok, topology.to_json()

    def topology_entry(semantic_role: str, mesh_kind: str):
        return next(
            item
            for item in topology.data["entries"]
            if item["semantic_role"] == semantic_role
            and item["mesh_kind"] == mesh_kind
        )

    def scope_parameters(
        name: str,
        semantic_role: str,
        mesh_kind: str,
    ) -> dict[str, object]:
        entry = topology_entry(semantic_role, mesh_kind)
        return {
            "name": name,
            "part_id": entry["part_id"],
            "logical_ids": [entry["logical_id"]],
            "mesh_kind": entry["mesh_kind"],
            "expected_count": entry["matched_count"],
        }

    actions = [
        (
            "create_named_region",
            scope_parameters(
                "边-固定端",
                "boundary.left",
                "edge",
            ),
        ),
        (
            "create_named_region",
            scope_parameters(
                "边-加载端",
                "boundary.right",
                "edge",
            ),
        ),
        (
            "create_named_region",
            scope_parameters(
                "边-孔边",
                "boundary.hole-loop",
                "edge",
            ),
        ),
        (
            "create_named_region",
            scope_parameters("域-板体", "domain", "element"),
        ),
        (
            "create_material",
            {
                "name": "材料-铝合金",
                "properties": {"E": 70000.0, "nu": 0.33},
            },
        ),
        (
            "create_section",
            {
                "name": "截面-平面应力",
                "material": "材料-铝合金",
                "plane_type": "stress",
                "properties": {},
                "thickness": 1.0,
            },
        ),
        (
            "assign_section",
            {
                "section_name": "截面-平面应力",
                "region_name": "域-板体",
            },
        ),
        ("create_static_step", {"name": "分析步-静力"}),
        (
            "create_boundary_condition",
            {
                "name": "位移-固定端",
                "step_name": "分析步-静力",
                "target_scope": "边-固定端",
                "target_kind": "edge",
                "first_component": 1,
                "last_component": 2,
                "value": 0.0,
                "unit": "mm",
                "distribution": "uniform",
                "confirmed": True,
            },
        ),
        (
            "create_load",
            {
                "name": "载荷-拉伸",
                "step_name": "分析步-静力",
                "target_scope": "边-加载端",
                "entity_type": "edge",
                "load_type": "edge_traction",
                "component": None,
                "vector": [10.0, 0.0],
                "magnitude": None,
                "direction": "global_xy",
                "unit": "N/mm",
                "distribution": "uniform",
                "confirmed": True,
            },
        ),
        (
            "create_result_request",
            {
                "name": "结果请求-位移反力",
                "step_name": "分析步-静力",
                "target": "node",
                "variables": ["U", "RF"],
                "units": ["mm", "N"],
                "confirmed": True,
            },
        ),
    ]

    results = []
    for index, (action, parameters) in enumerate(actions):
        result = controller.dispatch(
            "apply_model_definition",
            {"action": action, "parameters": parameters},
            ToolExecutionContext(
                "session-a8",
                session.session_revision,
                f"direct-{index}",
            ),
        )
        assert result.ok, result.to_json()
        assert result.data["gui_synchronized"] is True
        assert "proposal_id" not in result.data
        results.append(result)

    after = session.snapshot()
    assert after.session_revision == before_revision + len(actions)
    assert len(projections) == len(actions)
    assert set(after.named_regions) == {
        "边-固定端",
        "边-加载端",
        "边-孔边",
        "域-板体",
    }
    assert [item.name for item in after.materials] == ["材料-铝合金"]
    assert [item.name for item in after.sections] == ["截面-平面应力"]
    assert [item.name for item in after.steps] == ["分析步-静力"]
    assert len(after.steps[0].boundaries) == 1
    assert len(after.steps[0].edge_loads) == 1
    assert len(after.steps[0].outputs) == 1
    assert controller.stage is AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY

    before_unsafe = session.snapshot()
    projected_count = len(projections)
    unsafe = controller.dispatch(
        "apply_model_definition",
        {
            "action": "create_material",
            "parameters": {
                "name": "材料-D:\\private\\steel",
                "properties": {"E": 210000.0, "nu": 0.3},
            },
        },
        ToolExecutionContext(
            "session-a8",
            session.session_revision,
            "direct-provider-unsafe",
        ),
    )
    assert not unsafe.ok
    assert session.snapshot() == before_unsafe
    assert len(projections) == projected_count
