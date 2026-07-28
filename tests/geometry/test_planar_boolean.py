from __future__ import annotations

import pytest

from fem.geometry import (
    BooleanGeometry,
    LogicalEntityRef,
    PlanarBooleanContext,
    PlanarBooleanSelectionError,
    RectangleGeometry,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    analyze_sketch_profiles,
    resolve_planar_boolean_faces,
)


def _two_profile_tool() -> tuple[SketchGeometry, tuple[str, ...]]:
    sketch = SketchGeometry(
        "Tool",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.5, 0.5),
            SketchPoint("P2", 1.0, 0.5),
            SketchPoint("P3", 1.0, 1.0),
            SketchPoint("P4", 0.5, 1.0),
            SketchPoint("P5", 2.0, 0.5),
            SketchPoint("P6", 2.5, 0.5),
            SketchPoint("P7", 2.5, 1.0),
            SketchPoint("P8", 2.0, 1.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
            SketchLine("L5", "P5", "P6"),
            SketchLine("L6", "P6", "P7"),
            SketchLine("L7", "P7", "P8"),
            SketchLine("L8", "P8", "P5"),
        ),
    )
    face_ids = tuple(
        f"face:{profile.id}" for profile in analyze_sketch_profiles(sketch).profiles
    )
    return sketch, face_ids


def test_planar_selection_canonicalizes_one_target_and_multiple_profiles() -> None:
    tool, tool_face_ids = _two_profile_tool()

    selection = resolve_planar_boolean_faces(
        RectangleGeometry("Target", 3.0, 2.0),
        LogicalEntityRef("face:domain"),
        tool,
        reversed(tool_face_ids),
    )

    assert selection.target_face_id == "face:domain"
    assert set(selection.tool_face_ids) == set(tool_face_ids)
    assert selection.tool.boundary_edge_ids == tuple(
        f"edge:L{index}" for index in range(1, 9)
    )


def test_planar_selection_rejects_wrong_target_and_non_strict_tool() -> None:
    tool, tool_face_ids = _two_profile_tool()

    with pytest.raises(PlanarBooleanSelectionError) as wrong_target:
        resolve_planar_boolean_faces(
            RectangleGeometry("Target", 3.0, 2.0),
            "edge:bottom",
            tool,
            tool_face_ids,
        )
    with pytest.raises(PlanarBooleanSelectionError) as primitive_tool:
        resolve_planar_boolean_faces(
            RectangleGeometry("Target", 3.0, 2.0),
            "face:domain",
            RectangleGeometry("Tool", 1.0, 1.0),
            ("face:domain",),
        )

    assert wrong_target.value.code == "planar-boolean.target.wrong-kind"
    assert primitive_tool.value.code == "planar-boolean.tool.strict-sketch-required"


def test_planar_selection_rejects_open_or_missing_profiles() -> None:
    open_tool = SketchGeometry(
        "Open",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 1.0, 0.0),
        ),
        (SketchLine("L1", "P1", "P2"),),
    )

    with pytest.raises(PlanarBooleanSelectionError) as open_profile:
        resolve_planar_boolean_faces(
            RectangleGeometry("Target", 3.0, 2.0),
            "face:domain",
            open_tool,
            ("face:domain",),
        )
    with pytest.raises(PlanarBooleanSelectionError) as missing:
        resolve_planar_boolean_faces(
            RectangleGeometry("Target", 3.0, 2.0),
            "face:domain",
            open_tool,
            (),
        )

    assert open_profile.value.code == "planar-boolean.tool.topology-unproven"
    assert missing.value.code == "planar-boolean.tool.required"


def test_planar_selection_rejects_non_xy_target_or_tool_planes() -> None:
    xy_tool, tool_face_ids = _two_profile_tool()
    vertical_plane = SketchPlane(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    vertical = SketchGeometry(
        "Vertical",
        vertical_plane,
        xy_tool.points,
        xy_tool.curves,
    )

    with pytest.raises(PlanarBooleanSelectionError) as target_plane:
        resolve_planar_boolean_faces(
            vertical,
            tool_face_ids[0],
            xy_tool,
            tool_face_ids,
        )
    with pytest.raises(PlanarBooleanSelectionError) as tool_plane:
        resolve_planar_boolean_faces(
            RectangleGeometry("Target", 3.0, 2.0),
            "face:domain",
            vertical,
            tool_face_ids,
        )

    assert target_plane.value.code == "planar-boolean.target.plane-unsupported"
    assert tool_plane.value.code == "planar-boolean.tool.plane-unsupported"


def test_planar_context_and_boolean_validate_feature_contract() -> None:
    tool, tool_face_ids = _two_profile_tool()
    context = PlanarBooleanContext(
        "PB1",
        "face:domain",
        tool_face_ids,
    )
    recipe = BooleanGeometry(
        "Cut",
        "cut",
        RectangleGeometry("Target", 3.0, 2.0),
        tool,
        planar_context=context,
    )

    assert recipe.planar_context == context
    assert not context.proven

    with pytest.raises(ValueError, match="PB1"):
        PlanarBooleanContext("PB0", "face:domain", tool_face_ids)
    with pytest.raises(ValueError, match="fuse or cut"):
        BooleanGeometry(
            "Fragment",
            "fragment",
            RectangleGeometry("Target", 3.0, 2.0),
            tool,
            planar_context=context,
        )
