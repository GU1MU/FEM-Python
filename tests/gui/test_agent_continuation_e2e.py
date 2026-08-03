from __future__ import annotations

import json
import os
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton

from fem_agent.authoring import (
    AgentProposal,
    FakeAuthoringPort,
    ModelOperation,
    OperationKind,
    ProposalKind,
    ProposalPortRecord,
    ProposalState,
)
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
)
from fem_agent.providers.base import AssistantMessage, ProviderResponse, ToolCall
from fem_agent.providers.fake import FakeProvider
from fem_gui.agent_authoring import AgentAuthoringBridge
from fem_gui.agent_events import EventType
from fem_gui.agent_runtime import QtAgentRuntime
from fem_gui.widgets.agent_chat import AgentChatDrawer
from tests.test_agent_authoring_phase_a8 import _context, _requirements_for


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, *, timeout_ms: int = 20_000) -> None:
    deadline = time.monotonic() + timeout_ms / 1_000
    application = _application()
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    application.processEvents()
    assert predicate()


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage(
            "assistant",
            tool_calls=(ToolCall(call_id, name, arguments),),
        ),
        finish_reason="tool_calls",
    )


def _text(value: str) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage("assistant", content=value),
        finish_reason="stop",
    )


class _ImmediateSuccessPort(FakeAuthoringPort):
    """Small local-job seam: GUI authorization completes deterministically."""

    def accept(self, proposal_id: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        succeeded = replace(
            record,
            state=ProposalState.SUCCEEDED,
            message=(
                f"{record.proposal.proposal_kind.value} completed locally at "
                "D:\\private\\terminal-sensitive.log"
            ),
        )
        self._records[proposal_id] = succeeded
        self.calls.append(("accept", proposal_id))
        return succeeded


def _proposal_handler(
    bridge: AgentAuthoringBridge,
    kind: ProposalKind,
    operation: ModelOperation,
):
    def handle(_arguments, controller: AuthoringWorkflowController):
        metadata = controller.invocation_metadata(kind.value)
        context = bridge.context
        assert context is not None
        suffix = str(metadata["identity_suffix"])
        proposal = AgentProposal.create(
            proposal_id=f"proposal-{suffix}",
            proposal_kind=kind,
            agent_session_id=str(metadata["agent_session_id"]),
            turn_id=str(metadata["turn_id"]),
            source_tool_call_ids=tuple(metadata["source_tool_call_ids"]),
            target_document_id=context.binding.document_id,
            target_session_id=context.binding.session_id,
            base_session_revision=context.binding.session_revision,
            draft_revision=int(metadata["draft_revision"]),
            operations=(operation,),
            preconditions={"local_gui_confirmation": True},
            expected_changes={"stage": kind.value},
            invalidation_impact={"accepted_results": kind is not ProposalKind.SOLVE},
            display_summary={
                "title": f"{kind.value} proposal",
                "summary": f"Run deterministic {kind.value} stage",
                "confirm_label": "确认",
            },
        )
        bridge.register_proposal(proposal)
        data = {
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.proposal_hash,
            "state": ProposalState.PENDING_CONFIRMATION.value,
            "proposal_view": {
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.proposal_hash,
                "proposal_kind": kind.value,
                "title": f"{kind.value} proposal",
                "summary": f"Run deterministic {kind.value} stage",
                "impact": "Bounded local model operation",
                "confirm_label": "确认",
                "target_document_id": context.binding.document_id,
                "target_session_id": context.binding.session_id,
                "base_session_revision": context.binding.session_revision,
            },
            "continuation_checkpoint": {
                "session_id": proposal.agent_session_id,
                "source_turn_id": proposal.turn_id,
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.proposal_hash,
                "model_revision": proposal.base_session_revision,
            },
        }
        return AuthoringToolOutcome(f"{kind.value} is waiting for GUI confirmation", data)

    return handle


def _controller(bridge: AgentAuthoringBridge, stages: list[str]) -> AuthoringWorkflowController:
    def definition(arguments, _controller):
        action = str(arguments["action"])
        stages.append(action)
        return AuthoringToolOutcome(
            f"{action} applied locally",
            {
                "state": "succeeded",
                "definition_object_type": (
                    "analysis_step" if action == "create_static_step" else "named_region"
                ),
            },
        )

    def preflight(_arguments, _controller):
        stages.append("preflight")
        return AuthoringToolOutcome(
            "preflight passed locally",
            {"passed": True, "blocking_diagnostic_count": 0},
        )

    def result_catalog(_arguments, _controller):
        stages.append("result")
        return AuthoringToolOutcome(
            "accepted result catalog read locally",
            {
                "state": "succeeded",
                "result_id": "result-e2e",
                "field_count": 2,
                "fields": ["U", "S"],
            },
        )

    geometry = ModelOperation(
        OperationKind.ADD_NATIVE_PART,
        {"part_name": "部件-E2E", "recipe": {"kind": "planar_sketch"}},
    )
    mesh = ModelOperation(
        OperationKind.REQUEST_MESH,
        {"part_id": "part-e2e", "mesh_intent_hash": "a" * 64},
    )
    solve = ModelOperation(
        OperationKind.REQUEST_SOLVE,
        {"step_name": "分析步-静力", "validation_stamp": "b" * 64},
    )
    return AuthoringWorkflowController(
        lambda: _context(),
        {
            "prepare_geometry_proposal": _proposal_handler(
                bridge, ProposalKind.GEOMETRY, geometry
            ),
            "prepare_mesh_proposal": _proposal_handler(
                bridge, ProposalKind.MESH, mesh
            ),
            "apply_model_definition": definition,
            "run_native_preflight": preflight,
            "prepare_solve_proposal": _proposal_handler(
                bridge, ProposalKind.SOLVE, solve
            ),
            "read_accepted_result_catalog": result_catalog,
        },
    )


def _click_current_accept(drawer: AgentChatDrawer) -> None:
    buttons = [
        button
        for button in drawer.findChildren(
            QToolButton,
            "agentChatProposalAcceptButton",
        )
        if button.isEnabled()
    ]
    assert len(buttons) == 1
    QTest.mouseClick(buttons[0], Qt.MouseButton.LeftButton)


def test_fake_provider_real_runtime_bridge_continues_full_authoring_workflow(
    tmp_path,
) -> None:
    _application()
    port = _ImmediateSuccessPort()
    bridge = AgentAuthoringBridge(port)
    bridge.bind_context(_context())
    stages: list[str] = []
    controller = _controller(bridge, stages)
    provider = FakeProvider(
        [
            _tool(
                "requirements-geometry",
                "set_authoring_requirements",
                {"turn_id": "requirements-geometry", "requirements": _requirements_for("geometry")},
            ),
            _tool(
                "proposal-geometry",
                "prepare_geometry_proposal",
                {
                    "part_function": "平面拉伸试样",
                    "geometry": {
                        "kind": "planar_profiles",
                        "profiles": [
                            {
                                "kind": "rectangle",
                                "x": 0.0,
                                "y": 0.0,
                                "width": 20.0,
                                "height": 10.0,
                            }
                        ],
                    },
                },
            ),
            _text("geometry awaiting local GUI"),
            _tool(
                "requirements-mesh",
                "set_authoring_requirements",
                {"turn_id": "requirements-mesh", "requirements": _requirements_for("mesh")},
            ),
            _tool("proposal-mesh", "prepare_mesh_proposal", {}),
            _text("mesh awaiting local GUI"),
            _tool(
                "requirements-definitions",
                "set_authoring_requirements",
                {
                    "turn_id": "requirements-definitions",
                    "requirements": _requirements_for("definitions"),
                },
            ),
            _tool(
                "definition-material",
                "apply_model_definition",
                {
                    "action": "create_material",
                    "parameters": {
                        "name": "材料-结构钢",
                        "properties": {"E": 210000.0, "nu": 0.3},
                    },
                },
            ),
            _tool(
                "definition-step",
                "apply_model_definition",
                {"action": "create_static_step", "parameters": {"name": "分析步-静力"}},
            ),
            _tool("definition-preflight", "run_native_preflight", {}),
            _tool("proposal-solve", "prepare_solve_proposal", {}),
            _text("solve awaiting local GUI"),
            _tool("result-catalog", "read_accepted_result_catalog", {}),
            _text("需求、几何、网格、定义、求解与结果均已完成。"),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: provider,
        authoring_controller=controller,
    )
    drawer = AgentChatDrawer(agent_runtime=runtime, authoring_bridge=bridge)
    events = []
    runtime.agentEventReady.connect(events.append)
    drawer.show()

    assert runtime.send_message("完成 D:\\private\\sensitive-model.inp 的全流程")
    _wait_until(lambda: not runtime.busy)
    _wait_until(lambda: len(drawer.event_presentation.turns[0].proposals) == 1)

    heartbeat: list[bool] = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    _click_current_accept(drawer)
    _wait_until(lambda: len(drawer.event_presentation.turns) >= 2 and not runtime.busy)
    _wait_until(
        lambda: sum(len(turn.proposals) for turn in drawer.event_presentation.turns) == 2
    )
    assert heartbeat == [True]
    assert controller.stage is AuthoringWorkflowStage.MESH_PENDING

    _click_current_accept(drawer)
    _wait_until(
        lambda: sum(len(turn.proposals) for turn in drawer.event_presentation.turns) == 3
        and not runtime.busy
    )
    assert controller.stage is AuthoringWorkflowStage.SOLVE_PENDING

    _click_current_accept(drawer)
    _wait_until(lambda: controller.stage is AuthoringWorkflowStage.RESULTS_READY)
    _wait_until(lambda: not runtime.busy)

    assert stages == ["create_material", "create_static_step", "preflight", "result"]
    assert [call for call in port.calls if call[0] == "accept"] == [
        ("accept", record.proposal.proposal_id) for record in port.records
    ]
    assert sum(event.event_type is EventType.CONTINUATION_STARTED for event in events) == 3
    assert provider.requests[3].messages[-1].role == "system"
    assert "proposal_terminal" in provider.requests[3].messages[-1].content
    assert "set_authoring_requirements" in {tool.name for tool in provider.requests[3].tools}
    assert "set_authoring_requirements" in {tool.name for tool in provider.requests[6].tools}
    assert "read_accepted_result_catalog" in {tool.name for tool in provider.requests[12].tools}

    encoded = json.dumps(
        [
            {
                "messages": [message.content for message in request.messages],
                "tools": [tool.parameters for tool in request.tools],
            }
            for request in provider.requests
        ],
        ensure_ascii=False,
    )
    assert "sensitive-model.inp" not in encoded
    assert "terminal-sensitive.log" not in encoded
    assert "PySide6" not in encoded and "vtk" not in encoded.casefold()
    assert '"nodes":' not in encoded and '"elements":' not in encoded
    assert '"result_arrays":' not in encoded

    drawer.close()
    runtime.shutdown()
