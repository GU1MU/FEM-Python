from __future__ import annotations

import pytest

from fem.application import ModelSession, UnitContext
from fem.application.recipe_compiler import TopologyResolutionError, compile_recipe
from fem.application.preprocessing import generate_fem_model
from fem.geometry import (
    MovedGeometry,
    PathSweptGeometry,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SketchRectangle,
    WireGeometry,
    WireMember,
    WirePoint,
    describe_recipe_topology,
    is_single_solid_recipe,
    model,
)
from fem.io.project import decode_project, encode_project
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import ProposalState
from fem_agent.geometry_authoring import planar_sketch_geometry
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)


def _path(frame: str = "transport") -> PathSweptGeometry:
    profile = RectangleGeometry("Profile", 2.0, 1.0)
    path = WireGeometry(
        "Ordered path",
        (
            WirePoint("A", 0.0, 0.0, 0.0),
            WirePoint("B", 0.0, 0.0, 2.0),
            WirePoint("C", 1.0, 0.0, 3.0),
        ),
        (WireMember("AB", "A", "B"), WireMember("BC", "B", "C")),
    )
    return PathSweptGeometry(profile, path, ("face:domain",), frame)


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


@pytest.mark.gmsh
@pytest.mark.parametrize("frame", ("fixed", "transport"))
def test_phase3_real_path_sweep_proves_volume_caps_sides_and_frame(frame: str) -> None:
    recipe = _path(frame)

    with model(f"phase3-path-{frame}", dimension=3) as cad:
        compiled = compile_recipe(cad, recipe)

        assert len(compiled.domain) == 1
        assert cad.volume(compiled.domain[0]) > 0.0
        assert len(compiled.logical_entities["face:start"]) == 1
        assert len(compiled.logical_entities["face:end"]) == 1
        assert all(
            len(compiled.logical_entities[f"face:side/{name}"]) == 2
            for name in ("bottom", "right", "top", "left")
        )
    assert describe_recipe_topology(recipe).exact
    assert is_single_solid_recipe(recipe)


@pytest.mark.gmsh
@pytest.mark.parametrize(
    "recipe",
    (
        MovedGeometry(_path(), 2.0, 3.0, 4.0),
        RotatedGeometry(_path("fixed"), "y", 37.0),
    ),
)
def test_phase3_path_sweep_rigid_transform_rebinds_all_lineage(recipe) -> None:
    with model(f"phase3-path-transform-{type(recipe).__name__}", dimension=3) as cad:
        compiled = compile_recipe(cad, recipe)

        assert len(compiled.domain) == 1
        assert cad.volume(compiled.domain[0]) > 0.0
        assert set(compiled.logical_entities) == set(
            describe_recipe_topology(recipe).signature.logical_ids
        )


@pytest.mark.gmsh
def test_phase3_path_sweep_remeshes_as_one_tet_solid() -> None:
    recipe = _path()

    coarse = generate_fem_model(
        recipe,
        MeshSettings(0.8, cell_shape="tetrahedron"),
    )
    refined = generate_fem_model(
        recipe,
        MeshSettings(0.5, cell_shape="tetrahedron"),
    )

    assert {element.type for element in coarse.mesh.elements} == {"Tet4"}
    assert {element.type for element in refined.mesh.elements} == {"Tet4"}
    assert len(refined.mesh.elements) > len(coarse.mesh.elements)


@pytest.mark.gmsh
def test_phase3_half_full_revolve_and_cross_axis_rejection() -> None:
    profile = RectangleGeometry("Profile", 2.0, 1.0)
    for angle in (180.0, 360.0):
        with model(f"phase3-revolve-{angle:g}", dimension=3) as cad:
            compiled = compile_recipe(
                cad,
                RevolvedGeometry(profile, "x", angle, ("face:domain",)),
            )
            assert len(compiled.domain) == 1
            assert cad.volume(compiled.domain[0]) > 0.0

    crossing = MovedGeometry(RectangleGeometry("Crossing", 2.0, 2.0), 0.0, -1.0, 0.0)
    with model("phase3-revolve-crossing", dimension=3) as cad:
        with pytest.raises(TopologyResolutionError, match="crosses-axis"):
            compile_recipe(
                cad,
                RevolvedGeometry(crossing, "x", 180.0, ("face:domain",)),
            )


def test_phase3_path_rejects_disconnected_branch_self_intersection_and_zero_segment() -> None:
    profile = RectangleGeometry("Profile", 1.0, 1.0)
    points = (
        WirePoint("A", 0.0, 0.0, 0.0),
        WirePoint("B", 0.0, 0.0, 1.0),
        WirePoint("C", 1.0, 0.0, 1.0),
        WirePoint("D", 1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="两个端点"):
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
    with pytest.raises(ValueError, match="分支"):
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
    with pytest.raises(ValueError, match="自相交"):
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


def test_phase3_runtime_schema_is_strict_for_revolve_and_ordered_path() -> None:
    session, _source = _strict_session()
    _bridge, controller = _controller(session)
    definition = next(
        item for item in controller.definitions if item.name == "prepare_geometry_edit"
    )
    variants = definition.parameters["properties"]["edit"]["oneOf"]
    by_operation = {
        item["properties"]["operation"]["const"]: item for item in variants
    }

    assert by_operation["revolve_profile"]["required"] == [
        "operation", "source_face_id", "axis", "angle_degrees"
    ]
    path = by_operation["path_sweep_profile"]
    assert path["required"] == [
        "operation", "source_face_id", "path", "frame_strategy"
    ]
    assert path["properties"]["frame_strategy"]["enum"] == ["fixed", "transport"]
    assert path["additionalProperties"] is False


@pytest.mark.gmsh
def test_phase3_agent_path_proposal_is_atomic_revision_bound_and_persistent() -> None:
    session, source = _strict_session()
    bridge, controller = _controller(session)
    before = session.snapshot()
    prepared = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "path_sweep_profile",
                "source_face_id": source,
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
        },
        ToolExecutionContext("phase3", 0, "path-sweep"),
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
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe


def test_phase3_stale_path_proposal_does_not_mutate() -> None:
    session, source = _strict_session()
    bridge, controller = _controller(session)
    prepared = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "revolve_profile",
                "source_face_id": source,
                "axis": "x",
                "angle_degrees": 180.0,
            },
        },
        ToolExecutionContext("phase3", 0, "revolve"),
    )
    session.rename_native_part("P1", "Changed")
    stale_state = session.snapshot()

    receipt = bridge.accept_from_gui_control(prepared.data["proposal_id"])

    assert receipt.state is ProposalState.FAILED
    assert session.snapshot() == stale_state


def test_phase3_preflight_failure_and_gui_reject_are_atomic() -> None:
    session, source = _strict_session()
    bridge, controller = _controller(session)
    before = session.snapshot()
    failed = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "path_sweep_profile",
                "source_face_id": source,
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
        },
        ToolExecutionContext("phase3", 0, "invalid-start"),
    )

    assert not failed.ok
    assert session.snapshot() == before

    prepared = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "revolve_profile",
                "source_face_id": source,
                "axis": "x",
                "angle_degrees": 180.0,
            },
        },
        ToolExecutionContext("phase3", 0, "reject-revolve"),
    )
    receipt = bridge.reject_from_gui_control(prepared.data["proposal_id"])

    assert receipt.state is ProposalState.REJECTED
    assert session.snapshot() == before
