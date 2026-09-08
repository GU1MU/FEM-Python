from __future__ import annotations


from fem.application import ModelSession, compile_planar_construction
from fem.geometry import (
    ExtrudedGeometry,
    PlanarConstructionIR,
    RevolvedGeometry,
    describe_recipe_topology,
)
from fem_agent.geometry_authoring import profile_transform_context
from fem_agent.tools.registry import ToolExecutionContext
from tests.helpers.agent_planar_construction import make_planar_authoring_controller

import pytest


def _construction(*nodes: dict[str, object], result: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "Phase 4 construction",
        "plane": "XY",
        "nodes": list(nodes),
        "result_node_id": result,
    }


def _ring_construction() -> dict[str, object]:
    return _construction(
        {
            "id": "outer",
            "kind": "circle",
            "center_x": 0.0,
            "center_y": 3.0,
            "radius": 1.0,
        },
        {
            "id": "inner",
            "kind": "circle",
            "center_x": 0.0,
            "center_y": 3.0,
            "radius": 0.4,
        },
        {
            "id": "ring",
            "kind": "difference",
            "base": "outer",
            "subtract": ["inner"],
        },
        result="ring",
    )


def _dispatch(controller, arguments: dict[str, object], key: str):
    return controller.dispatch(
        "prepare_planar_construction_proposal",
        arguments,
        ToolExecutionContext("phase4-ir-transform", 0, key),
    )


@pytest.mark.usefixtures("real_gmsh")
def test_direct_ring_revolution_proves_one_body() -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    result = _dispatch(
        controller,
        {
            "part_function": "旋转圆环",
            "construction": _ring_construction(),
            "output": {
                "kind": "revolution",
                "profile_selection": "unique_material_profile",
                "axis": "x",
                "angle_degrees": 360.0,
            },
        },
        "ring-revolution",
    )

    assert result.ok, result.summary
    bridge.accept_from_gui_control(result.data["proposal_id"])
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is RevolvedGeometry
    topology = describe_recipe_topology(recipe)
    assert topology.exact
    assert len(topology.entities_of("body", selectable_only=True)) == 1


@pytest.mark.usefixtures("real_gmsh")
def test_ir_planar_part_uses_existing_profile_transform_tools() -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    planar = _dispatch(
        controller,
        {
            "part_function": "后续拉伸圆环",
            "construction": _ring_construction(),
            "output": "planar",
        },
        "planar-ring",
    )
    assert planar.ok, planar.summary
    bridge.accept_from_gui_control(planar.data["proposal_id"])
    snapshot = session.snapshot()
    part = snapshot.parts[0]
    bridge, controller = make_planar_authoring_controller(session)

    read = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": part.id},
        ToolExecutionContext("phase4-ir-transform", snapshot.session_revision, "read"),
    )
    direct_context = profile_transform_context(
        part.geometry_recipe,
        part_id=part.id,
        session_revision=snapshot.session_revision,
    )
    assert read.ok
    assert read.data["profiles"] == direct_context["profiles"]
    assert read.data["extrusion"] == direct_context["extrusion"]

    transformed = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": part.id,
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
        ToolExecutionContext(
            "phase4-ir-transform",
            snapshot.session_revision,
            "dedicated-extrusion",
        ),
    )
    assert transformed.ok, transformed.summary
    proposal = bridge._records[transformed.data["proposal_id"]].proposal
    assert proposal.invalidation_impact == {
        "mesh": True,
        "definitions": True,
        "results": True,
    }
    bridge.accept_from_gui_control(transformed.data["proposal_id"])
    transformed_snapshot = session.snapshot()
    assert transformed_snapshot.session_revision == snapshot.session_revision + 1
    assert type(transformed_snapshot.parts[0].geometry_recipe) is ExtrudedGeometry


@pytest.mark.usefixtures("real_gmsh")
def test_multiple_materials_require_explicit_selection() -> None:
    construction = _construction(
        {
            "id": "left",
            "kind": "rectangle",
            "x": 0.0,
            "y": 0.0,
            "width": 1.0,
            "height": 1.0,
        },
        {
            "id": "right",
            "kind": "rectangle",
            "x": 3.0,
            "y": 0.0,
            "width": 1.0,
            "height": 1.0,
        },
        {"id": "result", "kind": "union", "operands": ["left", "right"]},
        result="result",
    )
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    before = session.snapshot()
    ambiguous = _dispatch(
        controller,
        {
            "part_function": "双材料区",
            "construction": construction,
            "output": {
                "kind": "extrusion",
                "profile_selection": "unique_material_profile",
                "height": 1.0,
            },
        },
        "ambiguous",
    )
    assert not ambiguous.ok
    assert ambiguous.data["diagnostic"]["code"] == "planar-ir.transform-invalid"
    assert session.snapshot() == before
    assert not bridge._records

    compiled = compile_planar_construction(PlanarConstructionIR.from_dict(construction))
    context = profile_transform_context(compiled.recipe)
    face_ids = [item["face_id"] for item in context["profiles"]]
    explicit = _dispatch(
        controller,
        {
            "part_function": "双材料区",
            "construction": construction,
            "output": {
                "kind": "extrusion",
                "profile_selection": face_ids,
                "height": 1.0,
            },
        },
        "explicit",
    )
    assert explicit.ok, explicit.summary
    bridge.accept_from_gui_control(explicit.data["proposal_id"])
    topology = describe_recipe_topology(session.snapshot().parts[0].geometry_recipe)
    assert len(topology.entities_of("body", selectable_only=True)) == 2
