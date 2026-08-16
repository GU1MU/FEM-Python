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
    resolve_extrusion_source_faces,
)
from fem_agent.geometry_authoring import (
    GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
    geometry_feature_catalog_tool_schema,
    planar_feature_geometry_catalog,
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
    controller, _bridge = _controller_and_bridge(session)
    return controller


def _controller_and_bridge(session: ModelSession):
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot())
    return (
        create_session_authoring_workflow_controller(
            session,
            bridge,
            AgentResultQueryBridge(SessionResultQueryPort(session)),
        ),
        bridge,
    )


def _proven_cut(base, tool, model_name: str):
    target_face_id = resolve_extrusion_source_faces(base).face_ids[0]
    tool_face_ids = resolve_extrusion_source_faces(tool).face_ids
    with geometry_runtime.model(model_name, dimension=2) as cad:
        return prepare_planar_boolean(
            cad,
            base,
            target_face_id,
            tool,
            tool_face_ids,
            "cut",
        ).geometry


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
    part_context = result.data["parts"][0]
    geometry = part_context["planar_feature_geometry"]
    feature = geometry["features"][0]
    assert feature["feature_id"] == "PB1"
    assert feature["operation"] == "cut"
    assert feature["bounding_box"] == [37.0, 120.0, 63.0, 180.0]
    assert feature["tool_geometry_recipe"]["kind"] == "planar_sketch"
    assert len(feature["tool_geometry_recipe"]["points"]) == 12
    assert feature["tool_geometry_recipe_scope"] == {
        "kind": "planar_boolean_feature_tool_snapshot",
        "feature_id": "PB1",
        "editable": False,
        "logical_ids": "feature_local_read_only",
        "replacement_operation": "replace_planar_boolean_feature",
    }

    edit_context = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": "P1"},
        ToolExecutionContext("catalog-planar", 0, "catalog-planar-edit-read"),
    )
    assert edit_context.ok, edit_context.diagnostics[0].message
    policy = edit_context.data["freeform_profile_policy"]
    assert policy["primary_operation_for_closed_boundary_slot"] == (
        "planar_boolean(tool.kind=polygon)"
    )
    assert policy["rectangle_decomposition_for_one_connected_slot"] == "avoid"
    edit_feature = edit_context.data["planar_feature_geometry"]["features"][0]
    assert edit_feature["feature_id"] == "PB1"
    assert edit_feature["bounding_box"] == [37.0, 120.0, 63.0, 180.0]

    definition = next(
        item for item in controller.definitions if item.name == "prepare_geometry_edit"
    )
    operations = {
        branch["properties"]["operation"]["const"]
        for branch in definition.parameters["properties"]["edit"]["oneOf"]
    }
    assert operations == {
        "planar_boolean",
        "replace_planar_boolean_feature",
        "add_path_slot",
        "translate",
        "rotate",
    }
    planar_boolean = next(
        branch
        for branch in definition.parameters["properties"]["edit"]["oneOf"]
        if branch["properties"]["operation"]["const"] == "planar_boolean"
    )
    tool_schema = planar_boolean["properties"]["tool"]
    assert "Prefer polygon" in tool_schema["description"]
    polygon_tool = next(
        branch
        for branch in tool_schema["oneOf"]
        if branch["properties"]["kind"]["const"] == "polygon"
    )
    path_tool = next(
        branch
        for branch in tool_schema["oneOf"]
        if branch["properties"]["kind"]["const"] == "path_stroke"
    )
    assert "Primary boundary" in polygon_tool["properties"]["vertices"][
        "description"
    ]
    assert "Multiple bends" in path_tool["properties"]["points"]["description"]
    assert definition.parameters["properties"]["part_id"] == {"const": "P1"}


def test_phase1_scoped_geometry_edit_schema_rejects_sketch_ids_before_handler() -> None:
    plate = RectangleGeometry("Plate", 100.0, 300.0)
    tool = planar_polygon_geometry(
        "Wrong cut",
        vertices=((40.0, 135.0), (60.0, 135.0), (60.0, 165.0), (40.0, 165.0)),
    ).recipe
    target_face_id = resolve_extrusion_source_faces(plate).face_ids[0]
    tool_face_id = resolve_extrusion_source_faces(tool).face_ids[0]
    recipe = BooleanGeometry(
        "Plate with wrong cut",
        "cut",
        plate,
        tool,
        planar_context=PlanarBooleanContext("PB1", target_face_id, (tool_face_id,)),
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Scoped schema",
        UnitContext("mm", "N", "MPa"),
        recipe,
    )
    controller = _controller(session)
    read = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": "P1"},
        ToolExecutionContext(session.session_id, 1, "read-scoped-schema"),
    )
    assert read.ok

    invalid = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "batch",
                "edits": [
                    {
                        "operation": "delete_curves",
                        "curve_ids": ["pc-c0001"],
                    }
                ],
            },
        },
        ToolExecutionContext(session.session_id, 1, "invalid-scoped-operation"),
    )

    assert not invalid.ok
    assert invalid.data["error"]["code"] == "geometry-edit.schema-invalid"
    assert invalid.data["error"]["path"] == "arguments.edit.operation"
    assert "replace_planar_boolean_feature" in invalid.data["error"][
        "allowed_values"
    ]

    missing_operation = controller.dispatch(
        "prepare_geometry_edit",
        {"part_id": "P1", "edit": {"edits": []}},
        ToolExecutionContext(session.session_id, 1, "missing-operation"),
    )
    assert not missing_operation.ok
    assert missing_operation.data["error"]["path"] == "arguments.edit.operation"

    nested_part_id = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "replace_planar_boolean_feature",
                "feature_id": "PB1",
                "part_id": "P1",
                "tool": {
                    "kind": "rectangle",
                    "x": 40.0,
                    "y": 135.0,
                    "width": 20.0,
                    "height": 30.0,
                },
            },
        },
        ToolExecutionContext(session.session_id, 1, "nested-part-id"),
    )
    assert not nested_part_id.ok
    assert nested_part_id.data["error"]["path"] == "arguments.edit.part_id"

    incomplete_tool = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "replace_planar_boolean_feature",
                "feature_id": "PB1",
                "tool": {
                    "kind": "rectangle",
                    "x": 40.0,
                    "y": 135.0,
                    "width": 20.0,
                },
            },
        },
        ToolExecutionContext(session.session_id, 1, "incomplete-tool"),
    )
    assert not incomplete_tool.ok
    assert incomplete_tool.data["error"]["code"] == "geometry-edit.schema-invalid"
    assert incomplete_tool.data["error"]["path"] == "arguments.edit.tool.height"


def test_phase1_replaces_planar_boolean_feature_and_replays_later_history(
    real_gmsh,
) -> None:
    del real_gmsh
    plate = RectangleGeometry("Plate", 100.0, 300.0)
    wrong_tool = planar_polygon_geometry(
        "Wrong rectangular cut",
        vertices=((40.0, 135.0), (60.0, 135.0), (60.0, 165.0), (40.0, 165.0)),
    ).recipe
    wrong_cut = _proven_cut(plate, wrong_tool, "wrong-cut")
    corner_hole = planar_polygon_geometry(
        "Following cut",
        vertices=((3.5, 3.5), (6.5, 3.5), (6.5, 6.5), (3.5, 6.5)),
    ).recipe
    accepted = _proven_cut(
        wrong_cut,
        corner_hole,
        "following-hole",
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Feature replacement",
        UnitContext("mm", "N", "MPa"),
        accepted,
    )
    controller, bridge = _controller_and_bridge(session)
    read = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": "P1"},
        ToolExecutionContext(session.session_id, 1, "read-before-replacement"),
    )
    assert read.ok

    no_op_append = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "planar_boolean",
                "boolean_operation": "cut",
                "tool": {
                    "kind": "polygon",
                    "vertices": [
                        {"x": 40.0, "y": 135.0},
                        {"x": 50.0, "y": 135.0},
                        {"x": 50.0, "y": 145.0},
                        {"x": 60.0, "y": 145.0},
                        {"x": 60.0, "y": 155.0},
                        {"x": 50.0, "y": 155.0},
                        {"x": 50.0, "y": 165.0},
                        {"x": 40.0, "y": 165.0},
                    ],
                },
            },
        },
        ToolExecutionContext(session.session_id, 1, "no-op-append"),
    )
    assert not no_op_append.ok
    assert no_op_append.data["error"]["code"] == (
        "geometry-edit.planar-boolean-lineage"
    )
    assert no_op_append.data["error"]["reason"].startswith("planar-boolean.")
    assert no_op_append.data["error"]["remediation"]["replacement_operation"] == (
        "replace_planar_boolean_feature"
    )

    replacement = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "replace_planar_boolean_feature",
                "feature_id": "PB1",
                "tool": {
                    "kind": "polygon",
                    "vertices": [
                        {"x": 30.0, "y": 135.0},
                        {"x": 40.0, "y": 135.0},
                        {"x": 40.0, "y": 145.0},
                        {"x": 60.0, "y": 145.0},
                        {"x": 60.0, "y": 135.0},
                        {"x": 70.0, "y": 135.0},
                        {"x": 70.0, "y": 165.0},
                        {"x": 60.0, "y": 165.0},
                        {"x": 60.0, "y": 155.0},
                        {"x": 40.0, "y": 155.0},
                        {"x": 40.0, "y": 165.0},
                        {"x": 30.0, "y": 165.0},
                    ],
                },
            },
        },
        ToolExecutionContext(session.session_id, 1, "replace-pb1"),
    )

    assert replacement.ok, replacement.data
    assert replacement.data["replaced_feature_id"] == "PB1"
    assert replacement.data["replayed_feature_count"] == 2
    receipt = bridge.accept_from_gui_control(replacement.data["proposal_id"])
    assert receipt.state.value == "succeeded"
    features = planar_feature_geometry_catalog(
        session.snapshot().parts[0].geometry_recipe
    )["features"]
    assert [feature["feature_id"] for feature in features] == ["PB1", "PB2"]
    assert features[0]["bounding_box"] == [30.0, 135.0, 70.0, 165.0]
    assert features[1]["bounding_box"] == [3.5, 3.5, 6.5, 6.5]


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
