from __future__ import annotations

import pytest

from fem_agent.geometry_authoring import (
    add_planar_circle,
    add_planar_polygon,
    box_geometry,
    cylinder_geometry,
    disk_geometry,
    planar_polygon_geometry,
    plate_with_hole_geometry,
    rectangle_geometry,
    rotate_geometry,
    translate_geometry,
    update_planar_point,
)


def test_recipe_tools_cover_required_primitives_transforms_and_bounded_preview() -> (
    None
):
    rectangle = rectangle_geometry("实体-矩形", width=10.0, height=4.0)
    disk = disk_geometry("实体-圆盘", radius=2.0)
    box = box_geometry("实体-长方体", width=3.0, depth=2.0, height=1.0)
    cylinder = cylinder_geometry("实体-圆柱", radius=1.0, height=5.0)
    moved = translate_geometry(rectangle, dx=2.0, dy=-1.0)
    rotated = rotate_geometry(box, axis="z", angle_degrees=30.0)

    assert [draft.preview.dimension for draft in (rectangle, disk)] == [2, 2]
    assert [draft.preview.dimension for draft in (box, cylinder)] == [3, 3]
    assert moved.recipe_payload["kind"] == "translated"
    assert rotated.recipe_payload["kind"] == "rotated"
    for draft in (rectangle, disk, box, cylinder, moved, rotated):
        assert 0 < len(draft.preview.points) <= 64
        assert draft.preview.to_dict()["kind"] == "bounded_wireframe"


def test_plate_hole_accepts_coordinates_or_offset_and_rejects_incomplete_hole() -> (
    None
):
    by_coordinate = plate_with_hole_geometry(
        "实体-偏心孔板",
        width=10.0,
        height=6.0,
        hole_radius=1.0,
        hole_center=(6.5, 2.0),
    )
    by_offset = plate_with_hole_geometry(
        "实体-偏心孔板",
        width=10.0,
        height=6.0,
        hole_radius=1.0,
        center_offset=(1.5, -1.0),
    )

    assert by_coordinate.recipe == by_offset.recipe
    assert by_coordinate.transforms[0]["kind"] == "hole_center"
    assert by_offset.transforms[0]["kind"] == "center_offset"

    with pytest.raises(ValueError, match="exactly one"):
        plate_with_hole_geometry(
            "实体-孔板",
            width=10.0,
            height=6.0,
            hole_radius=1.0,
        )
    with pytest.raises(ValueError, match="entirely inside"):
        plate_with_hole_geometry(
            "实体-孔板",
            width=10.0,
            height=6.0,
            hole_radius=1.0,
            hole_center=(0.5, 3.0),
        )


def test_general_polygon_profile_can_be_extended_and_reshaped() -> None:
    polygon = planar_polygon_geometry(
        "草图-三角板",
        vertices=((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)),
    )
    with_cutout = add_planar_circle(
        polygon.recipe,
        center_x=2.0,
        center_y=2.0,
        radius=0.5,
    )
    reshaped = update_planar_point(
        with_cutout.recipe,
        point_id="P2",
        x=12.0,
    )
    with_second_profile = add_planar_polygon(
        reshaped.recipe,
        vertices=((6.0, 1.0), (7.0, 1.0), (6.5, 2.0)),
    )

    assert with_second_profile.recipe_payload["kind"] == "planar_sketch"
    assert with_second_profile.recipe.point("P2").u == 12.0
    assert len(with_second_profile.recipe.curves) == 7
