from __future__ import annotations

import math

import pytest

from fem.application import ModelSession, UnitContext
from fem.geometry import RectangleGeometry, WireGeometry, WireMember, WirePoint
from fem.io import load_project, save_project
from fem_agent.authoring import AuthoringAuthorizationError, ProposalState
from fem_agent.authoring_runtime import AuthoringWorkflowStage, _PREPARE_GEOMETRY
from fem_agent.geometry_authoring import (
    geometry_recipe_from_payload,
    wire_geometry,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)


def _frame_arguments() -> dict[str, object]:
    return {
        "part_function": "空间框架",
        "geometry": {
            "kind": "wire",
            "points": [
                {"name": "A", "x": 0.0, "y": 0.0, "z": 0.0},
                {"name": "B", "x": 2.0, "y": 0.0, "z": 0.0},
                {"name": "C", "x": 2.0, "y": 1.0, "z": 1.0},
                {"name": "D", "x": 0.0, "y": 1.0, "z": 1.0},
            ],
            "members": [
                {"name": "AB", "start": "A", "end": "B"},
                {"name": "BC", "start": "B", "end": "C"},
                {"name": "CD", "start": "C", "end": "D"},
                {"name": "DA", "start": "D", "end": "A"},
            ],
        },
    }


def test_phase1_wire_draft_payload_preview_and_round_trip_preserve_identity() -> None:
    recipe = WireGeometry(
        "线框-折线",
        (
            WirePoint("A", 0.0, 0.0, 0.0),
            WirePoint("B", 1.0, 0.0, 1.0),
            WirePoint("C", 1.0, 2.0, 1.0),
        ),
        (
            WireMember("AB", "A", "B"),
            WireMember("BC", "B", "C"),
        ),
    )

    draft = wire_geometry(
        recipe.name,
        points=recipe.points,
        members=recipe.members,
    )
    restored = geometry_recipe_from_payload(draft.recipe_payload)
    preview = draft.preview.to_dict()

    assert restored == recipe
    assert draft.recipe_payload["kind"] == "wire"
    assert preview["dimension"] == 1
    assert preview["point_names"] == ["A", "B", "C"]
    assert preview["member_names"] == ["AB", "BC"]
    assert preview["lines"] == [[0, 1], [1, 2]]


def test_phase1_wire_keeps_coincident_names_and_unshared_crossings_independent() -> None:
    coincident = WireGeometry(
        "线框-同坐标",
        (
            WirePoint("A1", 0.0, 0.0, 0.0),
            WirePoint("A2", 0.0, 0.0, 0.0),
            WirePoint("B1", 1.0, 0.0, 0.0),
            WirePoint("B2", 0.0, 1.0, 0.0),
        ),
        (
            WireMember("M1", "A1", "B1"),
            WireMember("M2", "A2", "B2"),
        ),
    )
    crossing = WireGeometry(
        "线框-交叉",
        (
            WirePoint("L", -1.0, 0.0, 0.0),
            WirePoint("R", 1.0, 0.0, 0.0),
            WirePoint("B", 0.0, -1.0, 0.0),
            WirePoint("T", 0.0, 1.0, 0.0),
        ),
        (
            WireMember("H", "L", "R"),
            WireMember("V", "B", "T"),
        ),
    )

    assert wire_geometry(
        coincident.name,
        points=coincident.points,
        members=coincident.members,
    ).preview.points[0] == wire_geometry(
        coincident.name,
        points=coincident.points,
        members=coincident.members,
    ).preview.points[1]
    assert wire_geometry(
        crossing.name,
        points=crossing.points,
        members=crossing.members,
    ).preview.lines == ((0, 1), (2, 3))


@pytest.mark.parametrize(
    "points,members,error",
    [
        (
            (WirePoint("A", 0, 0, 0), WirePoint("a", 1, 0, 0)),
            (WireMember("M", "A", "a"),),
            "duplicate wire point name",
        ),
        (
            (WirePoint("A", 0, 0, 0), WirePoint("B", 1, 0, 0)),
            (WireMember("M", "A", "missing"),),
            "unknown point",
        ),
        (
            (WirePoint("A", 0, 0, 0), WirePoint("B", 1, 0, 0)),
            (WireMember("M", "A", "A"),),
            "same point",
        ),
        (
            (WirePoint("A", 0, 0, 0), WirePoint("B", 1, 0, 0)),
            (
                WireMember("M1", "A", "B"),
                WireMember("M2", "B", "A"),
            ),
            "duplicate endpoint pair",
        ),
    ],
)
def test_phase1_wire_rejects_invalid_topology_deterministically(
    points,
    members,
    error,
) -> None:
    with pytest.raises(ValueError, match=error):
        wire_geometry("线框-无效", points=points, members=members)


def test_phase1_wire_rejects_non_finite_coordinates_and_payload_extensions() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        WirePoint("A", math.nan, 0.0, 0.0)

    with pytest.raises(ValueError, match="wire point fields do not match"):
        geometry_recipe_from_payload(
            {
                "kind": "wire",
                "name": "线框-无效",
                "points": [
                    {"name": "A", "x": 0, "y": 0, "z": 0, "tag": 1},
                    {"name": "B", "x": 1, "y": 0, "z": 0},
                ],
                "members": [{"name": "M", "start": "A", "end": "B"}],
            }
        )


def test_phase1_prepare_geometry_schema_exposes_only_bounded_named_wire() -> None:
    variants = _PREPARE_GEOMETRY.parameters["properties"]["geometry"]["oneOf"]
    wire = next(
        item
        for item in variants
        if item["properties"]["kind"] == {"const": "wire"}
    )

    assert wire["additionalProperties"] is False
    assert wire["properties"]["points"]["maxItems"] == 128
    assert wire["properties"]["members"]["maxItems"] == 128
    assert wire["properties"]["points"]["items"]["additionalProperties"] is False
    assert wire["properties"]["members"]["items"]["additionalProperties"] is False


def test_phase1_gui_bridge_commits_wire_once_then_project_round_trips(tmp_path) -> None:
    session = ModelSession()
    refreshes: list[int] = []
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(
            session,
            lambda: refreshes.append(session.session_revision),
        )
    )
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    before = session.snapshot()

    prepared = controller.dispatch(
        "prepare_geometry_proposal",
        _frame_arguments(),
        ToolExecutionContext("wire-session", 0, "wire-proposal"),
    )
    pending = session.snapshot()

    assert prepared.ok
    assert pending.session_revision == before.session_revision
    assert pending.parts == ()
    assert refreshes == []

    receipt = bridge.accept_from_gui_control(prepared.data["proposal_id"])
    accepted = session.snapshot()

    assert receipt.state is ProposalState.SUCCEEDED
    assert accepted.session_revision == before.session_revision + 1
    assert len(accepted.parts) == 1
    assert isinstance(accepted.parts[0].geometry_recipe, WireGeometry)
    assert refreshes == [accepted.session_revision]

    path = save_project(tmp_path / "agent-wire.femproj", session.prepare_project_save().snapshot)
    reopened = load_project(path).snapshot
    assert reopened.parts[0].geometry_recipe == accepted.parts[0].geometry_recipe


def test_phase1_rejected_wire_proposal_leaves_blank_session_unchanged() -> None:
    session = ModelSession()
    refreshes: list[int] = []
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(
            session,
            lambda: refreshes.append(session.session_revision),
        )
    )
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    before = session.snapshot()
    prepared = controller.dispatch(
        "prepare_geometry_proposal",
        _frame_arguments(),
        ToolExecutionContext("wire-session-reject", 0, "wire-reject"),
    )

    receipt = bridge.reject_from_gui_control(prepared.data["proposal_id"])
    after = session.snapshot()

    assert receipt.state is ProposalState.REJECTED
    assert after.session_revision == before.session_revision
    assert after.parts == ()
    assert refreshes == []


def test_phase1_stale_wire_proposal_cannot_add_a_part() -> None:
    session = ModelSession()
    units = UnitContext("mm", "N", "MPa", convention="N-mm-MPa")
    session.create_native_project_with_first_part(
        "模型-既有",
        units,
        RectangleGeometry("实体-既有", 2.0, 1.0),
        part_name="部件-既有",
    )
    refreshes: list[int] = []
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(
            session,
            lambda: refreshes.append(session.session_revision),
        )
    )
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    controller._stage = AuthoringWorkflowStage.GEOMETRY_READY
    requirements = controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "wire-stale-requirements",
            "requirements": {
                "length_unit": "mm",
                "force_unit": "N",
                "stress_unit": "MPa",
            },
        },
        ToolExecutionContext(
            "wire-session-stale",
            0,
            "wire-stale-requirements",
        ),
    )
    assert requirements.ok, requirements.summary
    prepared = controller.dispatch(
        "prepare_geometry_proposal",
        _frame_arguments(),
        ToolExecutionContext("wire-session-stale", 0, "wire-stale"),
    )
    assert prepared.ok, prepared.summary
    proposal_id = prepared.data["proposal_id"]

    session.add_native_part(
        RectangleGeometry("实体-外部", 1.0, 1.0),
        name="部件-外部",
        mesh_settings=None,
        unit_context=units,
    )
    bridge.bind_snapshot(session.snapshot())
    count_after_external_edit = len(session.snapshot().parts)

    assert bridge.state(proposal_id) is ProposalState.STALE
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_from_gui_control(proposal_id)
    assert len(session.snapshot().parts) == count_after_external_edit
    assert refreshes == []
