from __future__ import annotations

import pytest

from fem.geometry import (
    ExtrudedGeometry,
    ExtrusionSourceResolutionError,
    LogicalEntityRef,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    analyze_sketch_profiles,
    resolve_extrusion_source_faces,
)
from fem.geometry.recipe_topology import describe_recipe_topology
from fem_gui.geometry_preview import build_geometry_preview


def two_profile_sketch() -> SketchGeometry:
    points = (
        SketchPoint("P1", 0.0, 0.0),
        SketchPoint("P2", 2.0, 0.0),
        SketchPoint("P3", 2.0, 1.0),
        SketchPoint("P4", 0.0, 1.0),
        SketchPoint("P5", 5.0, 0.0),
        SketchPoint("P6", 7.0, 0.0),
        SketchPoint("P7", 7.0, 1.0),
        SketchPoint("P8", 5.0, 1.0),
    )
    curves = (
        SketchLine("L1", "P1", "P2"),
        SketchLine("L2", "P2", "P3"),
        SketchLine("L3", "P3", "P4"),
        SketchLine("L4", "P4", "P1"),
        SketchLine("L5", "P5", "P6"),
        SketchLine("L6", "P6", "P7"),
        SketchLine("L7", "P7", "P8"),
        SketchLine("L8", "P8", "P5"),
    )
    return SketchGeometry("Profiles", SketchPlane.xy(), points, curves)


def profile_face_id(sketch: SketchGeometry, edge_id: str) -> str:
    profile = next(
        profile
        for profile in analyze_sketch_profiles(sketch).profiles
        if edge_id in {item.lstrip("-") for item in profile.curve_ids}
    )
    return f"face:{profile.id}"


def hole_profile_sketch() -> SketchGeometry:
    points = (
        SketchPoint("P1", 0.0, 0.0),
        SketchPoint("P2", 4.0, 0.0),
        SketchPoint("P3", 4.0, 3.0),
        SketchPoint("P4", 0.0, 3.0),
        SketchPoint("P5", 1.0, 1.0),
        SketchPoint("P6", 2.0, 1.0),
        SketchPoint("P7", 2.0, 2.0),
        SketchPoint("P8", 1.0, 2.0),
    )
    curves = (
        SketchLine("L1", "P1", "P2"),
        SketchLine("L2", "P2", "P3"),
        SketchLine("L3", "P3", "P4"),
        SketchLine("L4", "P4", "P1"),
        SketchLine("L5", "P5", "P6"),
        SketchLine("L6", "P6", "P7"),
        SketchLine("L7", "P7", "P8"),
        SketchLine("L8", "P8", "P5"),
    )
    return SketchGeometry("Perforated", SketchPlane.xy(), points, curves)


def test_resolver_canonicalizes_aliases_and_filters_boundary_closure() -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    second = profile_face_id(sketch, "L5")

    all_sources = resolve_extrusion_source_faces(sketch)
    selected = resolve_extrusion_source_faces(sketch, (first,))

    assert set(all_sources.face_ids) == {first, second}
    assert selected.face_ids == (first,)
    assert selected.boundary_edge_ids == (
        "edge:L1",
        "edge:L2",
        "edge:L3",
        "edge:L4",
    )
    assert selected.boundary_point_ids == (
        "point:P1",
        "point:P2",
        "point:P3",
        "point:P4",
    )


def test_single_profile_alias_is_saved_as_primary_face() -> None:
    sketch = SketchGeometry(
        "Single",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 1.0, 0.0),
            SketchPoint("P3", 1.0, 1.0),
            SketchPoint("P4", 0.0, 1.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
    )
    primary = profile_face_id(sketch, "L1")

    recipe = ExtrudedGeometry(
        sketch,
        2.0,
        ("face:domain", primary),
    )

    assert recipe.source_face_ids == (primary,)


def test_invalid_extrusion_sources_have_stable_codes() -> None:
    sketch = two_profile_sketch()

    with pytest.raises(ExtrusionSourceResolutionError) as wrong_kind:
        resolve_extrusion_source_faces(
            sketch,
            (LogicalEntityRef("edge:L1"),),
        )
    with pytest.raises(ExtrusionSourceResolutionError) as unknown:
        resolve_extrusion_source_faces(sketch, ("face:missing",))

    assert wrong_kind.value.code == "extrude.source-face.wrong-kind"
    assert unknown.value.code == "extrude.source-face.unknown"


def test_selected_one_and_selected_two_have_stable_topology_namespaces() -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    second = profile_face_id(sketch, "L5")

    single = describe_recipe_topology(
        ExtrudedGeometry(sketch, 2.0, (first,))
    )
    multiple = describe_recipe_topology(
        ExtrudedGeometry(sketch, 2.0, (second, first))
    )

    assert single.exact
    assert single.entity("face:bottom").selectable
    assert single.entity("face:side/L1").selectable
    assert "face:side/L5" not in single.signature.logical_ids
    first_name = first.split(":", 1)[1]
    second_name = second.split(":", 1)[1]
    assert multiple.exact
    assert multiple.entity(f"face:bottom/{first_name}").selectable
    assert multiple.entity(f"face:top/{second_name}").selectable
    assert multiple.entity(f"face:side/{first_name}/L1").selectable
    assert multiple.entity(f"face:side/{second_name}/L5").selectable
    assert multiple.entity("body:domain").selectable


def test_height_only_edit_preserves_profile_extrusion_signature() -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")

    before = describe_recipe_topology(
        ExtrudedGeometry(sketch, 1.0, (first,))
    )
    after = describe_recipe_topology(
        ExtrudedGeometry(sketch, 5.0, (first,))
    )

    assert before.signature == after.signature


def test_hole_boundary_is_part_of_selected_material_profile_closure() -> None:
    sketch = hole_profile_sketch()
    source = profile_face_id(sketch, "L1")

    selection = resolve_extrusion_source_faces(sketch, (source,))
    topology = describe_recipe_topology(
        ExtrudedGeometry(sketch, 2.0, (source,))
    )

    assert selection.boundary_edge_ids == tuple(
        f"edge:L{index}" for index in range(1, 9)
    )
    assert topology.entity("face:side/L5").selectable
    assert topology.entity("face:side/L8").selectable


def test_selected_only_preview_omits_unselected_profile() -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")

    preview = build_geometry_preview(
        ExtrudedGeometry(sketch, 2.0, (first,))
    )

    assert max(point[0] for point in preview.points) == pytest.approx(2.0)
    assert min(point[2] for point in preview.points) == pytest.approx(0.0)
    assert max(point[2] for point in preview.points) == pytest.approx(2.0)
    assert "face:side/L1" in preview.face_logical_ids
    assert "face:side/L5" not in preview.face_logical_ids
    assert preview.body_logical_id == "body:domain"
