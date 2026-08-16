from __future__ import annotations

import pytest

from fem import geometry as geometry_runtime
from fem.application import (
    ModelSession,
    compile_planar_construction,
    compile_planar_feature_recipe,
    derive_feature_history,
)
from fem.application.preprocessing import generate_fem_model
from fem.application.recipe_compiler import compile_recipe
from fem.geometry import (
    BooleanGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    PlanarConstructionIR,
)
from fem.io.project import load_project, save_project
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import ProposalState
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import ToolExecutionContext
from tests.gui.test_agent_planar_construction_ir import (
    _ControllerDynamicTools,
    _controller,
    _tool,
)


def _h_plate() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "H slot plate",
        "plane": "XY",
        "nodes": [
            {
                "id": "plate",
                "kind": "rectangle",
                "x": 0.0,
                "y": 0.0,
                "width": 30.0,
                "height": 10.0,
            },
            {
                "id": "left_bar",
                "kind": "rectangle",
                "x": 11.0,
                "y": 2.0,
                "width": 1.0,
                "height": 6.0,
            },
            {
                "id": "cross_bar",
                "kind": "rectangle",
                "x": 11.0,
                "y": 4.5,
                "width": 8.0,
                "height": 1.0,
            },
            {
                "id": "right_bar",
                "kind": "rectangle",
                "x": 18.0,
                "y": 2.0,
                "width": 1.0,
                "height": 6.0,
            },
            {
                "id": "slot",
                "kind": "union",
                "operands": ["left_bar", "cross_bar", "right_bar"],
            },
            {
                "id": "hole_seed",
                "kind": "circle",
                "center_x": 1.0,
                "center_y": 1.0,
                "radius": 0.35,
            },
            {
                "id": "corner_holes",
                "kind": "rectangular_pattern",
                "seed": "hole_seed",
                "count_x": 2,
                "count_y": 2,
                "spacing_x": 28.0,
                "spacing_y": 8.0,
            },
            {
                "id": "cuts",
                "kind": "union",
                "operands": ["slot", "corner_holes"],
            },
            {
                "id": "result",
                "kind": "difference",
                "base": "plate",
                "subtract": ["cuts"],
            },
        ],
        "result_node_id": "result",
    }


def _planar_arguments(
    construction: dict[str, object],
    *,
    part_function: str,
) -> dict[str, object]:
    return {
        "part_function": part_function,
        "construction": construction,
        "output": "planar",
    }


def _dispatch(controller, arguments: dict[str, object], key: str):
    return controller.dispatch(
        "prepare_planar_construction_proposal",
        arguments,
        ToolExecutionContext("phase7-planar", 0, key),
    )


def _save_and_reopen(session: ModelSession, path):
    target = save_project(path, session.prepare_project_save())
    assert target.suffix == ".fempy"
    return load_project(target).snapshot


@pytest.mark.gmsh
def test_phase7_h_plate_preview_mesh_disk_roundtrip_and_edit(
    real_gmsh,
    tmp_path,
) -> None:
    del real_gmsh
    session = ModelSession()
    bridge, controller = _controller(session)
    before = session.snapshot()

    result = _dispatch(
        controller,
        _planar_arguments(_h_plate(), part_function="带 H 槽和四角孔的二维板"),
        "h-planar",
    )

    assert result.ok, result.summary
    proposal_id = result.data["proposal_id"]
    preview = bridge._proposal_previews[proposal_id]
    assert preview.dimension == 2
    assert preview.faces and preview.edges
    assert session.snapshot() == before
    receipt = bridge.accept_from_gui_control(proposal_id)
    assert receipt.state is ProposalState.SUCCEEDED
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is BooleanGeometry
    assert [feature.kind for feature in derive_feature_history(recipe)] == [
        "sketch",
        "cut",
        "cut",
    ]

    mesh = generate_fem_model(
        recipe,
        MeshSettings(1.0, cell_shape="triangle"),
    )
    assert mesh.mesh.elements
    assert {element.type for element in mesh.mesh.elements} == {"Tri3"}

    reopened = _save_and_reopen(session, tmp_path / "h-plate.fempy")
    assert reopened.parts[0].geometry_recipe == recipe
    reopened_session = ModelSession()
    assert reopened_session.replace_from_snapshot(reopened).accepted
    edited_construction = _h_plate()
    edited_construction["nodes"][0]["width"] = 32.0
    edited_construction["nodes"][6]["spacing_x"] = 30.0
    edited_ir = PlanarConstructionIR.from_dict(edited_construction)
    edited_flattened = compile_planar_construction(edited_ir)
    edited = compile_planar_feature_recipe(
        edited_ir,
        compiled=edited_flattened,
    )
    current = reopened_session.snapshot()
    reopened_session.replace_part_geometry(
        current.parts[0].id,
        edited,
        expected_session_revision=current.session_revision,
    )
    assert reopened_session.snapshot().parts[0].geometry_recipe == edited


def _composite_slot(bars: list[tuple[float, float, float, float]]) -> dict[str, object]:
    nodes: list[dict[str, object]] = [
        {
            "id": "plate",
            "kind": "rectangle",
            "x": -2.0,
            "y": -2.0,
            "width": 14.0,
            "height": 14.0,
        }
    ]
    for index, (x, y, width, height) in enumerate(bars):
        nodes.append(
            {
                "id": f"bar_{index}",
                "kind": "rectangle",
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )
    nodes.extend(
        (
            {
                "id": "slot",
                "kind": "union",
                "operands": [f"bar_{index}" for index in range(len(bars))],
            },
            {
                "id": "result",
                "kind": "difference",
                "base": "plate",
                "subtract": ["slot"],
            },
        )
    )
    return {
        "schema_version": 1,
        "name": "generic composite slot",
        "plane": "XY",
        "nodes": nodes,
        "result_node_id": "result",
    }


@pytest.mark.gmsh
@pytest.mark.parametrize(
    "bars",
    (
        [(1.0, 8.0, 8.0, 1.0), (4.5, 1.0, 1.0, 8.0)],
        [
            (1.0, 1.0, 1.0, 8.0),
            (1.0, 8.0, 7.0, 1.0),
            (1.0, 4.5, 6.0, 1.0),
            (1.0, 1.0, 7.0, 1.0),
        ],
        [(4.5, 1.0, 1.0, 8.0), (1.0, 4.5, 8.0, 1.0)],
    ),
    ids=("tee", "e", "cross"),
)
def test_phase7_three_named_shapes_use_only_generic_nodes(real_gmsh, bars) -> None:
    del real_gmsh
    construction = _composite_slot(bars)
    assert {node["kind"] for node in construction["nodes"]} == {
        "rectangle",
        "union",
        "difference",
    }
    session = ModelSession()
    bridge, controller = _controller(session)
    result = _dispatch(
        controller,
        _planar_arguments(construction, part_function="通用组合槽板"),
        f"generic-{len(bars)}",
    )
    assert result.ok, result.summary
    bridge.accept_from_gui_control(result.data["proposal_id"])
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is BooleanGeometry
    assert [feature.kind for feature in derive_feature_history(recipe)] == [
        "sketch",
        "cut",
    ]
    proof = compile_planar_construction(
        PlanarConstructionIR.from_dict(construction)
    ).proof
    assert proof.material_profile_count == 1
    assert proof.hole_count == 1


@pytest.mark.gmsh
def test_phase7_u_path_stroke_preview_mesh_and_disk_roundtrip(
    real_gmsh,
    tmp_path,
) -> None:
    del real_gmsh
    construction = {
        "schema_version": 1,
        "name": "U path slot plate",
        "plane": "XY",
        "nodes": [
            {
                "id": "plate",
                "kind": "rectangle",
                "x": 0.0,
                "y": 0.0,
                "width": 20.0,
                "height": 20.0,
            },
            {
                "id": "slot",
                "kind": "path_stroke",
                "points": [[5.0, 15.0], [5.0, 5.0], [15.0, 5.0], [15.0, 15.0]],
                "width": 2.0,
                "cap": "round",
                "join": "round",
            },
            {
                "id": "result",
                "kind": "difference",
                "base": "plate",
                "subtract": ["slot"],
            },
        ],
        "result_node_id": "result",
    }
    session = ModelSession()
    bridge, controller = _controller(session)
    result = _dispatch(
        controller,
        _planar_arguments(construction, part_function="U 形定宽路径槽板"),
        "u-path",
    )
    assert result.ok, result.summary
    assert bridge._proposal_previews[result.data["proposal_id"]].faces
    bridge.accept_from_gui_control(result.data["proposal_id"])
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is BooleanGeometry
    proof = compile_planar_construction(
        PlanarConstructionIR.from_dict(construction)
    ).proof
    assert proof.hole_count == 1
    mesh = generate_fem_model(recipe, MeshSettings(1.0, cell_shape="triangle"))
    assert {element.type for element in mesh.mesh.elements} == {"Tri3"}
    reopened = _save_and_reopen(session, tmp_path / "u-path-slot.fempy")
    assert reopened.parts[0].geometry_recipe == recipe


@pytest.mark.gmsh
def test_phase7_h_sketch_dedicated_extrusion_lineage_tet_and_roundtrip(
    real_gmsh,
    tmp_path,
) -> None:
    del real_gmsh
    session = ModelSession()
    bridge, controller = _controller(session)
    planar = _dispatch(
        controller,
        _planar_arguments(_h_plate(), part_function="后续拉伸的 H 槽板"),
        "h-before-extrusion",
    )
    bridge.accept_from_gui_control(planar.data["proposal_id"])
    current = session.snapshot()
    part = current.parts[0]
    bridge, controller = _controller(session)
    read = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": part.id},
        ToolExecutionContext("phase7-transform", current.session_revision, "read"),
    )
    assert read.ok
    transformed = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": part.id,
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
        ToolExecutionContext(
            "phase7-transform",
            current.session_revision,
            "extrude",
        ),
    )
    assert transformed.ok, transformed.summary
    bridge.accept_from_gui_control(transformed.data["proposal_id"])
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is ExtrudedGeometry
    with geometry_runtime.model("phase7-h-extrusion", dimension=3) as cad:
        compiled = compile_recipe(cad, recipe)
        hole_sides = tuple(
            entity.logical_id
            for entity in compiled.catalog.entities_of("face")
            if entity.semantic_role == "sweep.boundary.hole"
        )
        # Feature-chain persistence keeps boolean-split boundaries: the H
        # slot contributes twelve selectable sides and each of the four
        # corner holes is cut into four arc sides (4 x 4), giving 28.
        assert len(hole_sides) == 28
        assert all(
            compiled.resolve(LogicalEntityRef(logical_id)) for logical_id in hole_sides
        )
    mesh = generate_fem_model(
        recipe,
        MeshSettings(1.2, cell_shape="tetrahedron"),
    )
    assert {element.type for element in mesh.mesh.elements} == {"Tet4"}
    reopened = _save_and_reopen(session, tmp_path / "h-extrusion.fempy")
    assert reopened.parts[0].geometry_recipe == recipe


@pytest.mark.gmsh
def test_phase7_blank_direct_extrusion_is_one_provider_round_and_one_final_card(
    real_gmsh,
    tmp_path,
) -> None:
    del real_gmsh
    session = ModelSession()
    bridge, controller = _controller(session)
    dynamic = _ControllerDynamicTools(controller)
    arguments = {
        "part_function": "直接生成带组合槽的厚板",
        "construction": _h_plate(),
        "output": {
            "kind": "extrusion",
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
    }
    provider = FakeProvider(
        [_tool("direct-extrusion", "prepare_planar_construction_proposal", arguments)]
    )
    engine = AgentSessionEngine(
        tmp_path / "phase7-direct-extrusion",
        provider,
        dynamic_tools=dynamic,
    )
    before = session.snapshot()

    events = engine.send_message("创建带组合槽的厚板，厚度 2")

    assert len(provider.requests) == 1
    assert session.snapshot() == before
    assert [
        event.data["tool"]
        for event in events
        if event.event is EngineEventType.TOOL_STARTED
    ] == ["prepare_planar_construction_proposal"]
    assert len(bridge._records) == 1
    proposal_id, record = next(iter(bridge._records.items()))
    assert record.proposal.display_summary["recipe_type"] == "ExtrudedGeometry"
    receipt = bridge.accept_from_gui_control(proposal_id)
    assert receipt.state is ProposalState.SUCCEEDED
    accepted = session.snapshot()
    assert len(accepted.parts) == 1
    assert accepted.parts[0].dimension == 3
    assert type(accepted.parts[0].geometry_recipe) is ExtrudedGeometry
