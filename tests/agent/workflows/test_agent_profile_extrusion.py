from __future__ import annotations

from copy import deepcopy

import pytest

from fem.application import ModelSession, UnitContext
from fem.geometry import ExtrudedGeometry, RectangleGeometry, SketchCircle, SketchRectangle, describe_recipe_topology
from fem.io.project import decode_project, encode_project
from fem_agent.authoring import ProposalState
from fem_agent.geometry_authoring import (
    geometry_recipe_from_payload,
    planar_sketch_geometry,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui import agent_authoring
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from tests.helpers.profile_sketches import (
    hole_profile_sketch,
    profile_face_id,
    two_profile_sketch,
)


pytestmark = pytest.mark.local_session


def _controller(session: ModelSession):
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    return bridge, controller


def _native_session(sketch) -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent extrusion",
        UnitContext("mm", "N", "MPa"),
        sketch,
        part_name="Sketch",
    )
    return session


def _dispatch(controller, *, source_face_ids, height=2.5):
    revision = controller.turn_snapshot.session_revision
    return controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": list(source_face_ids),
            "context_revision": revision,
            "height": height,
        },
        ToolExecutionContext("profile-extrusion", revision, "prepare"),
    )


def test_dedicated_extrusion_reads_unique_profile_and_accepts_atomically(
    monkeypatch,
) -> None:
    sketch = hole_profile_sketch()
    session = _native_session(sketch)
    bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    before = session.snapshot()
    context = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        ToolExecutionContext("agent-phase3", before.session_revision, "read"),
    )
    assert context.ok
    assert context.data["material_profile_count"] == 1
    assert context.data["hole_count"] == 1
    source = context.data["profiles"][0]["face_id"]

    prepared = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.5,
        },
        ToolExecutionContext("agent-phase3", before.session_revision, "prepare"),
    )
    assert prepared.ok, prepared.summary
    assert session.snapshot() == before
    assert (
        bridge.accept_from_gui_control(prepared.data["proposal_id"]).state
        is ProposalState.SUCCEEDED
    )
    recipe = session.snapshot().parts[0].geometry_recipe
    assert isinstance(recipe, ExtrudedGeometry)
    assert recipe.source_face_ids == (source,)
    topology = describe_recipe_topology(recipe)
    assert topology.entity("face:side/L5").semantic_role == "sweep.boundary.hole"


def test_explicit_profile_ids_require_same_revision(monkeypatch) -> None:
    session = _native_session(RectangleGeometry("Revision rectangle", 4.0, 3.0))
    bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    before = session.snapshot()
    context = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        ToolExecutionContext("agent-phase3", before.session_revision, "read"),
    )
    source = context.data["profiles"][0]["face_id"]

    prepared = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": before.session_revision,
            "height": 2.5,
        },
        ToolExecutionContext("agent-phase3", before.session_revision, "prepare"),
    )
    assert prepared.ok, prepared.summary
    assert session.snapshot() == before
    assert (
        bridge.accept_from_gui_control(prepared.data["proposal_id"]).state
        is ProposalState.SUCCEEDED
    )


def test_explicit_profile_ids_without_revision_are_rejected(monkeypatch) -> None:
    session = _native_session(RectangleGeometry("Missing revision", 4.0, 3.0))
    _bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    before = session.snapshot()
    context = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        ToolExecutionContext("agent-phase3", before.session_revision, "read"),
    )
    source = context.data["profiles"][0]["face_id"]
    rejected = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "height": 2.5,
        },
        ToolExecutionContext("agent-phase3", before.session_revision, "prepare"),
    )
    assert not rejected.ok
    assert rejected.data["diagnostic"]["code"] == "profile-transform.stale-context"
    assert session.snapshot() == before


def test_explicit_profile_ids_with_stale_revision_are_rejected(monkeypatch) -> None:
    session = _native_session(RectangleGeometry("Stale revision", 4.0, 3.0))
    _bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    before = session.snapshot()
    context = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        ToolExecutionContext("agent-phase3", before.session_revision, "read"),
    )
    source = context.data["profiles"][0]["face_id"]
    session.add_native_part(RectangleGeometry("Revision bump", 1.0, 1.0))
    changed = session.snapshot()
    rejected = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": before.session_revision,
            "height": 2.5,
        },
        ToolExecutionContext("agent-phase3", changed.session_revision, "prepare"),
    )
    assert not rejected.ok
    assert rejected.data["diagnostic"]["code"] == "profile-transform.stale-context"
    assert session.snapshot() == changed


def test_dedicated_extrusion_requires_explicit_selection_for_multiple_profiles(
    monkeypatch,
) -> None:
    session = _native_session(two_profile_sketch())
    _bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    before = session.snapshot()
    ambiguous = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.5,
        },
        ToolExecutionContext("agent-phase3", before.session_revision, "ambiguous"),
    )
    assert not ambiguous.ok
    assert ambiguous.data["diagnostic"]["code"] == (
        "profile-transform.ambiguous-material-profiles"
    )
    assert session.snapshot() == before


@pytest.mark.parametrize(
    "contour",
    (
        SketchRectangle("material", 0.0, 0.0, 3.0, 2.0),
        SketchCircle("material", 0.0, 0.0, 1.5),
    ),
)
def test_agent_extrudes_rectangle_and_circle_profiles(
    monkeypatch,
    contour,
) -> None:
    sketch = planar_sketch_geometry("Agent sketch", contours=(contour,)).recipe
    source = next(
        entity.logical_id
        for entity in describe_recipe_topology(sketch).entities
        if entity.kind == "face" and entity.semantic_role == "sketch.profile"
    )
    session = _native_session(sketch)
    bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )

    prepared = _dispatch(controller, source_face_ids=(source,))
    accepted = bridge.accept_from_gui_control(prepared.data["proposal_id"])

    assert accepted.state is ProposalState.SUCCEEDED
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is ExtrudedGeometry
    assert recipe.source_face_ids == (source,)


def test_multi_profile_proposal_is_atomic_and_commits_independent_parts(
    monkeypatch,
) -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    second = profile_face_id(sketch, "L5")
    session = _native_session(sketch)
    bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    before = session.snapshot()

    prepared = _dispatch(controller, source_face_ids=(second, first))

    assert prepared.ok, prepared.summary
    assert session.snapshot() == before
    assert "生成 2 个独立 Part" in prepared.data["proposal_view"]["summary"]
    proposal_id = prepared.data["proposal_id"]
    proposal = bridge._records[proposal_id].proposal
    assert proposal.display_summary["source"] == [first, second]
    assert proposal.display_summary["expected_part_count"] == 2
    assert proposal.display_summary["direction"] == "positive_sketch_normal"
    assert proposal.invalidation_impact == {
        "mesh": True, "definitions": True, "results": True
    }

    receipt = bridge.accept_from_gui_control(proposal_id)
    accepted = session.snapshot()

    assert receipt.state is ProposalState.SUCCEEDED
    assert accepted.session_revision == before.session_revision + 1
    assert [str(part.id) for part in accepted.parts] == ["P1", "P2"]
    assert all(type(part.geometry_recipe) is ExtrudedGeometry for part in accepted.parts)
    assert [part.geometry_recipe.source_face_ids for part in accepted.parts] == [
        (first,), (second,)
    ]


def test_hole_profile_retains_cap_outer_hole_and_body_lineage(
    monkeypatch,
) -> None:
    sketch = hole_profile_sketch()
    source = profile_face_id(sketch, "L1")
    session = _native_session(sketch)
    bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )

    prepared = _dispatch(controller, source_face_ids=(source,))
    bridge.accept_from_gui_control(prepared.data["proposal_id"])
    recipe = session.snapshot().parts[0].geometry_recipe
    topology = describe_recipe_topology(recipe)

    assert topology.entity("face:bottom").selectable
    assert topology.entity("face:top").selectable
    assert topology.entity("body:domain").selectable
    assert all(topology.entity(f"face:side/L{index}").selectable for index in range(1, 9))
    assert all(
        topology.entity(f"face:side/L{index}").semantic_role
        == "sweep.boundary.outer"
        for index in range(1, 5)
    )
    assert all(
        topology.entity(f"face:side/L{index}").semantic_role
        == "sweep.boundary.hole"
        for index in range(5, 9)
    )


@pytest.mark.parametrize(
    ("source_face_ids", "height"),
    [
        ((), 1.0),
        (("face:does-not-exist",), 1.0),
        (("face:hole",), 1.0),
        (("edge:L1",), 1.0),
        (("face:domain",), 0.0),
    ],
)
def test_rejects_ambiguous_nonmaterial_and_zero_height_without_mutation(
    monkeypatch,
    source_face_ids,
    height,
) -> None:
    session = _native_session(two_profile_sketch())
    _bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    before = session.snapshot()

    result = _dispatch(
        controller,
        source_face_ids=source_face_ids,
        height=height,
    )

    assert not result.ok
    assert session.snapshot() == before


def test_preflight_failure_and_reject_keep_session_unchanged(monkeypatch) -> None:
    sketch = two_profile_sketch()
    source = profile_face_id(sketch, "L1")
    session = _native_session(sketch)
    bridge, controller = _controller(session)
    before = session.snapshot()

    def fail(_recipes):
        raise RuntimeError("unexpected multiple volumes")

    monkeypatch.setattr(agent_authoring, "_preflight_profile_extrusions", fail)
    failed = _dispatch(controller, source_face_ids=(source,))
    assert not failed.ok
    assert session.snapshot() == before

    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    prepared = _dispatch(controller, source_face_ids=(source,))
    rejected = bridge.reject_from_gui_control(prepared.data["proposal_id"])
    assert rejected.state is ProposalState.REJECTED
    assert session.snapshot() == before


def test_stale_accept_and_operation_tampering_are_atomic(monkeypatch) -> None:
    sketch = two_profile_sketch()
    source = profile_face_id(sketch, "L1")
    session = _native_session(sketch)
    bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    prepared = _dispatch(controller, source_face_ids=(source,))
    proposal_id = prepared.data["proposal_id"]
    session.rename_native_part("P1", "Changed")
    changed = session.snapshot()

    receipt = bridge.accept_from_gui_control(proposal_id)
    assert receipt.state is ProposalState.FAILED
    assert session.snapshot() == changed

    operation = bridge._records[proposal_id].proposal.operations[0]
    payload = deepcopy(operation.parameters["base_recipe"])
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        geometry_recipe_from_payload(payload)


def test_project_round_trip_preserves_selected_source_and_lineage(
    monkeypatch,
) -> None:
    sketch = hole_profile_sketch()
    source = profile_face_id(sketch, "L1")
    session = _native_session(sketch)
    bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: None,
    )
    prepared = _dispatch(controller, source_face_ids=(source,), height=4.0)
    bridge.accept_from_gui_control(prepared.data["proposal_id"])
    accepted = session.snapshot()

    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    recipe = reopened.parts[0].geometry_recipe

    assert type(recipe) is ExtrudedGeometry
    assert recipe.height == 4.0
    assert recipe.source_face_ids == (source,)
    assert describe_recipe_topology(recipe) == describe_recipe_topology(
        accepted.parts[0].geometry_recipe
    )

    before_ids = describe_recipe_topology(recipe).signature.logical_ids
    edited = ExtrudedGeometry(recipe.base, 6.0, recipe.source_face_ids)
    session.replace_part_geometry("P1", edited)
    after = session.snapshot().parts[0].geometry_recipe

    assert after.height == 6.0
    assert describe_recipe_topology(after).signature.logical_ids == before_ids
