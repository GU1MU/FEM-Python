from __future__ import annotations

from fem.application import ModelSession, UnitContext
from fem.geometry import ExtrudedGeometry, RectangleGeometry, describe_recipe_topology
from fem_agent.geometry_authoring import (
    GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
    geometry_feature_catalog_tool_schema,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
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


def test_phase1_geometry_catalog_tool_schema_is_strictly_no_argument() -> None:
    schema = geometry_feature_catalog_tool_schema()

    assert schema["name"] == GEOMETRY_FEATURE_CATALOG_TOOL_NAME
    assert schema["input_schema"] == {
        "type": "object",
        "properties": {},
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


def test_phase1_geometry_catalog_rejects_arguments_and_blank_visibility() -> None:
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
        ToolExecutionContext("catalog-session", 0, "catalog-invalid"),
    )
    assert not result.ok


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
