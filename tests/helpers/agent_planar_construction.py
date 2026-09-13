from __future__ import annotations

from copy import deepcopy

from fem.application import ModelSession
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)

from tests.helpers.fixtures.planar_construction_baseline import EXPECTED_H_CONSTRUCTION


def make_planar_authoring_controller(session: ModelSession):
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


def make_h_plate_construction() -> dict[str, object]:
    return deepcopy(EXPECTED_H_CONSTRUCTION)


def build_planar_arguments() -> dict[str, object]:
    return {
        "part_function": "带组合槽和四角孔的二维板",
        "construction": make_h_plate_construction(),
        "output": "planar",
    }


def build_rectangle_arguments() -> dict[str, object]:
    """Small real geometry for proposal lifecycle tests, without repeated H-slot CAD."""
    return {
        "part_function": "二维矩形板",
        "construction": {
            "schema_version": 1,
            "name": "lifecycle rectangle",
            "plane": "XY",
            "nodes": [
                {"id": "plate", "kind": "rectangle", "x": 0, "y": 0,
                 "width": 100, "height": 300},
            ],
            "result_node_id": "plate",
        },
        "output": "planar",
    }


def dispatch_planar_construction(controller, *, key: str = "prepare-planar-construction", arguments=None):
    return controller.dispatch(
        "prepare_planar_construction_proposal",
        build_planar_arguments() if arguments is None else arguments,
        ToolExecutionContext("planar-construction", 0, key),
    )
