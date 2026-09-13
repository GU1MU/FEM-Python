from __future__ import annotations

import pytest

from fem.application import ModelSession, UnitContext
from fem.geometry import (
    PathSweptGeometry,
    RectangleGeometry,
    SketchRectangle,
    WireGeometry,
    WireMember,
    WirePoint,
    describe_recipe_topology,
)
from fem_agent.authoring import ProposalState
from fem_agent.geometry_authoring import planar_sketch_geometry
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui import agent_authoring
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)


def _controller(session: ModelSession):
    bridge = AgentAuthoringBridge(SessionGeometryAuthoringPort(session, lambda: None))
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    return bridge, controller


def _strict_session() -> tuple[ModelSession, str]:
    sketch = planar_sketch_geometry(
        "Agent sweep sketch",
        contours=(SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),),
    ).recipe
    source = next(
        entity.logical_id
        for entity in describe_recipe_topology(sketch).entities
        if entity.kind == "face" and entity.semantic_role == "sketch.profile"
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent sweep",
        UnitContext("mm", "N", "MPa"),
        sketch,
        part_name="Sketch",
    )
    return session, source


def test_path_rejects_disconnected_branch_self_intersection_and_zero_segment() -> None:
    profile = RectangleGeometry("Profile", 1.0, 1.0)
    points = (
        WirePoint("A", 0.0, 0.0, 0.0),
        WirePoint("B", 0.0, 0.0, 1.0),
        WirePoint("C", 1.0, 0.0, 1.0),
        WirePoint("D", 1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="two endpoints"):
        PathSweptGeometry(
            profile,
            WireGeometry(
                "Disconnected",
                points,
                (
                    WireMember("AB", "A", "B"),
                    WireMember("CD", "C", "D"),
                ),
            ),
            ("face:domain",),
        )
    with pytest.raises(ValueError, match="branch"):
        PathSweptGeometry(
            profile,
            WireGeometry(
                "Branched",
                points,
                (
                    WireMember("AB", "A", "B"),
                    WireMember("BC", "B", "C"),
                    WireMember("BD", "B", "D"),
                ),
            ),
            ("face:domain",),
        )
    crossing_points = (
        WirePoint("A", -1.0, 0.0, 0.0),
        WirePoint("B", 1.0, 1.0, 1.0),
        WirePoint("C", -1.0, 1.0, 1.0),
        WirePoint("D", 1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="self-intersect"):
        PathSweptGeometry(
            profile,
            WireGeometry(
                "Self crossing",
                crossing_points,
                (
                    WireMember("AB", "A", "B"),
                    WireMember("BC", "B", "C"),
                    WireMember("CD", "C", "D"),
                ),
            ),
            ("face:domain",),
        )
    with pytest.raises(ValueError, match="zero length"):
        WireGeometry(
            "Zero",
            (WirePoint("A", 0.0, 0.0, 0.0), WirePoint("B", 0.0, 0.0, 0.0)),
            (WireMember("AB", "A", "B"),),
        )


@pytest.mark.usefixtures("real_gmsh")
def test_dedicated_path_prepare_preserves_atomic_proposal(monkeypatch) -> None:
    session, source = _strict_session()
    bridge, controller = _controller(session)
    monkeypatch.setattr(agent_authoring, "_preflight_derived_geometry", lambda _recipe: None)
    before = session.snapshot()
    prepared = controller.dispatch(
        "prepare_profile_path_sweep",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": before.session_revision,
            "path": {
                "points": [
                    {"name": "A", "x": 0.0, "y": 0.0, "z": 0.0},
                    {"name": "B", "x": 0.0, "y": 0.0, "z": 2.0},
                ],
                "members": [
                    {"name": "AB", "start": "A", "end": "B"},
                ],
            },
            "frame_strategy": "fixed",
        },
        ToolExecutionContext("profile-sweep-dedicated", before.session_revision, "path"),
    )
    assert prepared.ok, prepared.summary
    assert session.snapshot() == before
    proposal_id = prepared.data["proposal_id"]
    assert bridge.accept_from_gui_control(proposal_id).state is ProposalState.SUCCEEDED
    assert isinstance(session.snapshot().parts[0].geometry_recipe, PathSweptGeometry)


@pytest.mark.usefixtures("real_gmsh")
def test_agent_path_proposal_is_atomic_revision_bound() -> None:
    session, source = _strict_session()
    bridge, controller = _controller(session)
    before = session.snapshot()
    prepared = controller.dispatch(
        "prepare_profile_path_sweep",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": session.session_revision,
            "path": {
                "points": [
                    {"name": "A", "x": 0.0, "y": 0.0, "z": 0.0},
                    {"name": "B", "x": 0.0, "y": 0.0, "z": 2.0},
                    {"name": "C", "x": 1.0, "y": 0.0, "z": 3.0},
                ],
                "members": [
                    {"name": "AB", "start": "A", "end": "B"},
                    {"name": "BC", "start": "B", "end": "C"},
                ],
            },
            "frame_strategy": "transport",
        },
        ToolExecutionContext("profile-sweep", 0, "path-sweep"),
    )

    assert prepared.ok, prepared.summary
    assert session.snapshot() == before
    proposal_id = prepared.data["proposal_id"]
    proposal = bridge._records[proposal_id].proposal
    assert proposal.display_summary["frame_strategy"] == "transport"
    assert proposal.base_session_revision == before.session_revision

    receipt = bridge.accept_from_gui_control(proposal_id)
    assert receipt.state is ProposalState.SUCCEEDED
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is PathSweptGeometry


@pytest.mark.usefixtures("real_gmsh")
def test_stale_path_proposal_does_not_mutate() -> None:
    session, source = _strict_session()
    bridge, controller = _controller(session)
    prepared = controller.dispatch(
        "prepare_profile_revolution",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": session.session_revision,
            "axis": "x",
            "angle_degrees": 180.0,
        },
        ToolExecutionContext("profile-sweep", 0, "revolve"),
    )
    session.rename_native_part("P1", "Changed")
    stale_state = session.snapshot()

    receipt = bridge.accept_from_gui_control(prepared.data["proposal_id"])

    assert receipt.state is ProposalState.FAILED
    assert session.snapshot() == stale_state


@pytest.mark.usefixtures("real_gmsh")
def test_preflight_failure_and_gui_reject_are_atomic() -> None:
    session, source = _strict_session()
    bridge, controller = _controller(session)
    before = session.snapshot()
    failed = controller.dispatch(
        "prepare_profile_path_sweep",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": session.session_revision,
            "path": {
                "points": [
                    {"name": "A", "x": 1.0, "y": 0.0, "z": 0.0},
                    {"name": "B", "x": 1.0, "y": 0.0, "z": 2.0},
                ],
                "members": [
                    {"name": "AB", "start": "A", "end": "B"},
                ],
            },
            "frame_strategy": "fixed",
        },
        ToolExecutionContext("profile-sweep", 0, "invalid-start"),
    )

    assert not failed.ok
    assert session.snapshot() == before

    prepared = controller.dispatch(
        "prepare_profile_revolution",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": session.session_revision,
            "axis": "x",
            "angle_degrees": 180.0,
        },
        ToolExecutionContext("profile-sweep", 0, "reject-revolve"),
    )
    receipt = bridge.reject_from_gui_control(prepared.data["proposal_id"])

    assert receipt.state is ProposalState.REJECTED
    assert session.snapshot() == before
