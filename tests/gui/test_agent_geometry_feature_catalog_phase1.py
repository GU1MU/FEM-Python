from __future__ import annotations

from fem import geometry as geometry_runtime
from fem.application import prepare_planar_boolean
from fem.application import ModelSession, UnitContext
from fem.geometry import (
    BooleanGeometry,
    ExtrudedGeometry,
    PlanarBooleanContext,
    RectangleGeometry,
    describe_recipe_topology,
)
from fem_agent.geometry_authoring import (
    GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
    geometry_feature_catalog_tool_schema,
    planar_polygon_geometry,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import AgentToolRegistry, ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)


def _extrusion(name: str) -> ExtrudedGeometry:
    sketch = RectangleGeometry(name, 2.0, 1.0)
    face_id = next(
        item.logical_id
        for item in describe_recipe_topology(sketch).entities
        if item.kind == "face"
    )
    return ExtrudedGeometry(sketch, 3.0, (face_id,))


def _controller(session: ModelSession):
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot())
    return create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )


def test_phase1_geometry_catalog_tool_schema_accepts_optional_part_filter() -> None:
    schema = geometry_feature_catalog_tool_schema()

    assert schema["name"] == GEOMETRY_FEATURE_CATALOG_TOOL_NAME
    assert schema["input_schema"] == {
        "type": "object",
        "properties": {
            "part_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 96,
            }
        },
        "required": [],
        "additionalProperties": False,
    }


def test_phase1_unmeshed_native_catalog_is_visible_bounded_and_read_only() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Model",
        UnitContext("mm", "N", "MPa"),
        _extrusion("Profile"),
        part_name="Active",
    )
    session.add_native_part(RectangleGeometry("Hidden", 1.0, 1.0), name="Hidden")
    session.suppress_native_part("P2")
    before = session.snapshot()
    controller = _controller(session)

    definitions = {item.name: item for item in controller.definitions}
    assert GEOMETRY_FEATURE_CATALOG_TOOL_NAME in definitions
    result = controller.dispatch(
        GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
        {},
        ToolExecutionContext("catalog-session", 0, "catalog-read"),
    )
    after = session.snapshot()

    assert result.ok, result.summary
    assert result.data["kind"] == "native_geometry_feature_catalog"
    assert result.data["session_revision"] == before.session_revision
    assert result.data["truncated"] is False
    assert [item["part_id"] for item in result.data["parts"]] == ["P1"]
    entities = result.data["parts"][0]["entities"]
    assert 0 < len(entities) <= 128
    assert any(item["logical_id"] == "body:domain" for item in entities)
    flattened = repr(result.data).casefold()
    assert "gmsh" not in flattened
    assert "occ" not in flattened
    assert before == after
    assert not before.mesh_current


def test_phase1_geometry_catalog_filters_part_and_rejects_unknown_part() -> None:
    blank = ModelSession()
    blank_controller = _controller(blank)
    assert GEOMETRY_FEATURE_CATALOG_TOOL_NAME not in {
        item.name for item in blank_controller.definitions
    }

    session = ModelSession()
    session.create_native_project_with_first_part(
        "Model",
        UnitContext("mm", "N", "MPa"),
        _extrusion("Profile"),
    )
    controller = _controller(session)
    result = controller.dispatch(
        GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
        {"part_id": "P1"},
        ToolExecutionContext("catalog-session", 0, "catalog-filtered"),
    )
    assert result.ok
    assert [item["part_id"] for item in result.data["parts"]] == ["P1"]

    missing = controller.dispatch(
        GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
        {"part_id": "P9"},
        ToolExecutionContext("catalog-session", 0, "catalog-missing"),
    )
    assert not missing.ok


def test_phase1_planar_boolean_catalog_exposes_exact_tool_recipe_and_bounds() -> None:
    plate = RectangleGeometry("Plate", 100.0, 300.0)
    h_tool = planar_polygon_geometry(
        "H slot",
        vertices=(
            (37.0, 120.0),
            (45.0, 120.0),
            (45.0, 146.0),
            (55.0, 146.0),
            (55.0, 120.0),
            (63.0, 120.0),
            (63.0, 180.0),
            (55.0, 180.0),
            (55.0, 154.0),
            (45.0, 154.0),
            (45.0, 180.0),
            (37.0, 180.0),
        ),
    ).recipe
    target_face_id = next(
        item.logical_id
        for item in describe_recipe_topology(plate).entities
        if item.kind == "face"
    )
    tool_face_id = next(
        item.logical_id
        for item in describe_recipe_topology(h_tool).entities
        if item.kind == "face"
    )
    recipe = BooleanGeometry(
        "Plate with H cut",
        "cut",
        plate,
        h_tool,
        planar_context=PlanarBooleanContext(
            "PB1",
            target_face_id,
            (tool_face_id,),
        ),
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Planar features",
        UnitContext("mm", "N", "MPa"),
        recipe,
    )
    controller = _controller(session)

    result = controller.dispatch(
        GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
        {"part_id": "P1"},
        ToolExecutionContext("catalog-planar", 0, "catalog-planar-read"),
    )

    assert result.ok, result.summary
    geometry = result.data["parts"][0]["planar_feature_geometry"]
    feature = geometry["features"][0]
    assert feature["feature_id"] == "PB1"
    assert feature["operation"] == "cut"
    assert feature["bounding_box"] == [37.0, 120.0, 63.0, 180.0]
    assert feature["tool_geometry_recipe"]["kind"] == "planar_sketch"
    assert len(feature["tool_geometry_recipe"]["points"]) == 12

    edit_context = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": "P1"},
        ToolExecutionContext("catalog-planar", 0, "catalog-planar-edit-read"),
    )
    assert edit_context.ok, edit_context.diagnostics[0].message
    edit_feature = edit_context.data["planar_feature_geometry"]["features"][0]
    assert edit_feature["feature_id"] == "PB1"
    assert edit_feature["bounding_box"] == [37.0, 120.0, 63.0, 180.0]


def test_phase1_stale_geometry_read_resynchronizes_without_unknown_tool(
    tmp_path,
) -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Model",
        UnitContext("mm", "N", "MPa"),
        _extrusion("Profile"),
    )
    controller = _controller(session)
    controller.cancel_turn("test cancellation")
    controller.set_published_tool_names(
        tuple(item.name for item in controller.definitions)
    )
    registry = AgentToolRegistry(tmp_path / "agent", dynamic_tools=controller)

    result = registry.dispatch(
        GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
        {"part_id": "P1"},
        ToolExecutionContext("catalog-stale", 0, "catalog-stale-read"),
    )

    assert result.ok, result.summary
    assert result.data["parts"][0]["part_id"] == "P1"
    assert controller.stage.value != "cancelled"


def test_phase1_planar_edit_verifies_general_feature_clearance(real_gmsh) -> None:
    del real_gmsh
    plate = RectangleGeometry("Plate", 100.0, 300.0)
    h_tool = planar_polygon_geometry(
        "Reference cut",
        vertices=(
            (37.0, 120.0),
            (45.0, 120.0),
            (45.0, 146.0),
            (55.0, 146.0),
            (55.0, 120.0),
            (63.0, 120.0),
            (63.0, 180.0),
            (55.0, 180.0),
            (55.0, 154.0),
            (45.0, 154.0),
            (45.0, 180.0),
            (37.0, 180.0),
        ),
    ).recipe
    target_face_id = next(
        item.logical_id
        for item in describe_recipe_topology(plate).entities
        if item.kind == "face"
    )
    tool_face_id = next(
        item.logical_id
        for item in describe_recipe_topology(h_tool).entities
        if item.kind == "face"
    )
    with geometry_runtime.model("reference-cut", dimension=2) as cad:
        base = prepare_planar_boolean(
            cad,
            plate,
            target_face_id,
            h_tool,
            (tool_face_id,),
            "cut",
        ).geometry
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Spatial relation",
        UnitContext("mm", "N", "MPa"),
        base,
    )
    controller = _controller(session)
    edit = {
        "operation": "planar_boolean",
        "boolean_operation": "cut",
        "tool": {
            "kind": "polygon",
            "vertices": [
                {"x": 28.0, "y": 190.0},
                {"x": 28.0, "y": 230.0},
                {"x": 36.0, "y": 230.0},
                {"x": 36.0, "y": 198.0},
                {"x": 64.0, "y": 198.0},
                {"x": 64.0, "y": 230.0},
                {"x": 72.0, "y": 230.0},
                {"x": 72.0, "y": 190.0},
            ],
        },
    }
    mismatch = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": edit,
            "spatial_relation": {
                "reference_feature_id": "PB1",
                "relation": "above",
                "clearance": 5.0,
            },
        },
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "spatial-clearance-mismatch",
        ),
    )
    assert not mismatch.ok

    context = ToolExecutionContext(
        session.session_id,
        session.session_revision,
        "spatial-clearance-valid",
    )

    result = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": edit,
            "spatial_relation": {
                "reference_feature_id": "PB1",
                "relation": "above",
                "clearance": 10.0,
            },
        },
        context,
    )

    assert result.ok, result.summary
    proof = result.data["spatial_relation_proof"]
    assert proof["reference_feature_id"] == "PB1"
    assert proof["target_feature_id"] == "PB2"
    assert proof["relation"] == "above"
    assert proof["requested_clearance"] == 10.0
    assert proof["measured_clearance"] == 10.0
    assert proof["verified"] is True


def test_phase1_geometry_catalog_omitted_count_uses_all_active_parts() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Many Parts",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("R1", 1.0, 1.0),
        part_name="Part 1",
    )
    for index in range(2, 130):
        session.add_native_part(
            RectangleGeometry(f"R{index}", 1.0, 1.0),
            name=f"Part {index}",
        )
    controller = _controller(session)

    result = controller.dispatch(
        GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
        {},
        ToolExecutionContext("catalog-many", 0, "catalog-many-read"),
    )

    assert result.ok, result.summary
    assert result.data["truncated"] is True
    assert result.data["omitted_part_count"] == 129 - len(result.data["parts"])
    assert result.data["omitted_part_count"] >= 1
