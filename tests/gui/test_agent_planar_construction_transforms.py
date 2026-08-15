from __future__ import annotations

from copy import deepcopy

from fem import geometry as geometry_runtime
from fem.application import ModelSession, compile_planar_construction
from fem.application.preprocessing import generate_fem_model
from fem.application.recipe_compiler import compile_recipe
from fem.geometry import (
    ExtrudedGeometry,
    LogicalEntityRef,
    PathSweptGeometry,
    PlanarConstructionIR,
    RevolvedGeometry,
    describe_recipe_topology,
)
from fem.io.project import decode_project, encode_project
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import ProposalState
from fem_agent.geometry_authoring import profile_transform_context
from fem_agent.tools.registry import ToolExecutionContext
from tests.gui.test_agent_planar_construction_ir import (
    _arguments as planar_arguments,
    _controller,
)


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


def _hole_profile_construction() -> dict[str, object]:
    return _construction(
        {
            "id": "plate",
            "kind": "rectangle",
            "x": -0.5,
            "y": -0.5,
            "width": 1.0,
            "height": 1.0,
        },
        {
            "id": "hole",
            "kind": "circle",
            "center_x": 0.0,
            "center_y": 0.0,
            "radius": 0.15,
        },
        {
            "id": "profile",
            "kind": "difference",
            "base": "plate",
            "subtract": ["hole"],
        },
        result="profile",
    )


def _path_output() -> dict[str, object]:
    return {
        "kind": "path_sweep",
        "profile_selection": "unique_material_profile",
        "path": {
            "points": [
                {"name": "A", "x": 0.0, "y": 0.0, "z": 0.0},
                {"name": "B", "x": 0.0, "y": 0.0, "z": 3.0},
                {"name": "C", "x": 2.0, "y": 0.0, "z": 4.0},
            ],
            "members": [
                {"name": "AB", "start": "A", "end": "B"},
                {"name": "BC", "start": "B", "end": "C"},
            ],
        },
        "frame_strategy": "transport",
    }


def _dispatch(controller, arguments: dict[str, object], key: str):
    return controller.dispatch(
        "prepare_planar_construction_proposal",
        arguments,
        ToolExecutionContext("phase4-ir-transform", 0, key),
    )


def test_phase4_output_schema_and_context_publish_four_strict_kinds() -> None:
    _bridge, controller = _controller(ModelSession())
    definition = next(
        item
        for item in controller.definitions
        if item.name == "prepare_planar_construction_proposal"
    )
    outputs = definition.parameters["properties"]["output"]["oneOf"]
    assert outputs[0] == {"const": "planar"}
    branches = {item["properties"]["kind"]["const"]: item for item in outputs[1:]}
    assert branches["extrusion"]["required"] == [
        "kind",
        "profile_selection",
        "height",
    ]
    assert branches["revolution"]["required"] == [
        "kind",
        "profile_selection",
        "axis",
        "angle_degrees",
    ]
    assert branches["path_sweep"]["required"] == [
        "kind",
        "profile_selection",
        "path",
        "frame_strategy",
    ]
    assert all(branch["additionalProperties"] is False for branch in branches.values())

    context = controller.dispatch(
        "read_authoring_context",
        {},
        ToolExecutionContext("phase4-ir-transform", 0, "context"),
    )
    assert context.data["context"]["planar_construction_ir"]["output_kinds"] == [
        "planar",
        "extrusion",
        "revolution",
        "path_sweep",
    ]


def test_phase4_direct_h_plate_extrusion_is_one_atomic_3d_proposal() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    arguments = deepcopy(planar_arguments())
    arguments["output"] = {
        "kind": "extrusion",
        "profile_selection": "unique_material_profile",
        "height": 10.0,
    }
    before = session.snapshot()

    result = _dispatch(controller, arguments, "h-extrusion")

    assert result.ok, result.summary
    assert session.snapshot() == before
    assert len(bridge._records) == 1
    proposal = bridge._records[result.data["proposal_id"]].proposal
    assert proposal.preconditions["local_evidence"]["output_kind"] == "extrusion"
    assert proposal.display_summary["recipe_type"] == "ExtrudedGeometry"
    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    snapshot = session.snapshot()
    assert snapshot.session_revision == 1
    assert len(snapshot.parts) == 1
    recipe = snapshot.parts[0].geometry_recipe
    assert type(recipe) is ExtrudedGeometry
    assert recipe.base.is_strict
    topology = describe_recipe_topology(recipe)
    assert topology.exact
    assert topology.entities_of("body", selectable_only=True)
    roles = {entity.semantic_role for entity in topology.entities_of("face")}
    assert "copy.bottom.sketch.profile" in roles
    assert "copy.top.sketch.profile" in roles
    assert "sweep.boundary.outer" in roles
    assert "sweep.boundary.hole" in roles
    with geometry_runtime.model("phase4-ir-h-lineage", dimension=3) as cad:
        compiled = compile_recipe(cad, recipe)
        assert len(compiled.domain) == 1
        assert cad.volume(compiled.domain[0]) > 0.0
        hole_sides = tuple(
            entity.logical_id
            for entity in compiled.catalog.entities_of("face")
            if entity.semantic_role == "sweep.boundary.hole"
        )
        assert hole_sides
        assert all(
            compiled.resolve(LogicalEntityRef(logical_id)) for logical_id in hole_sides
        )
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe


def test_phase4_direct_ring_revolution_proves_one_body() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
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


def test_phase4_direct_hole_path_sweep_preserves_channel_and_tet_chain() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = _dispatch(
        controller,
        {
            "part_function": "带孔路径扫掠",
            "construction": _hole_profile_construction(),
            "output": _path_output(),
        },
        "hole-path",
    )

    assert result.ok, result.summary
    bridge.accept_from_gui_control(result.data["proposal_id"])
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is PathSweptGeometry
    topology = describe_recipe_topology(recipe)
    assert topology.exact
    assert any(
        entity.semantic_role == "sweep.boundary.hole"
        for entity in topology.entities_of("face")
    )
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe
    mesh = generate_fem_model(recipe, MeshSettings(0.8, cell_shape="tetrahedron"))
    assert mesh.mesh.elements
    assert {element.type for element in mesh.mesh.elements} == {"Tet4"}


def test_phase4_ir_planar_part_uses_existing_profile_transform_tools() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
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
    bridge, controller = _controller(session)

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


def test_phase4_multiple_materials_require_explicit_selection() -> None:
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
    bridge, controller = _controller(session)
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
