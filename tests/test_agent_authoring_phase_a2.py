from __future__ import annotations

import pytest

from fem_agent.authoring import UnitContextSummary
from fem_agent.geometry_authoring import (
    box_geometry,
    cylinder_geometry,
    disk_geometry,
    plate_with_hole_geometry,
    rectangle_geometry,
    rotate_geometry,
    translate_geometry,
)
from fem_agent.naming import NameAllocator, NamePolicy, NamePolicyError


def test_a2_name_policy_allocates_normalized_unique_stable_names() -> None:
    policy = NamePolicy()
    allocator = NameAllocator(
        {
            "parts": (
                "部件-偏心孔板",
                "部件-偏心孔板-2",
                "部件-Ａ板",
            ),
            "models": ("模型-偏心孔板",),
        },
        policy=policy,
    )

    assert allocator.allocate("parts", "部件", "偏心孔板") == "部件-偏心孔板-3"
    assert allocator.allocate("models", "部件", "偏心孔板") == "部件-偏心孔板"
    assert allocator.allocate("parts", "部件", "A板") == "部件-A板-2"
    assert policy.compose("边", "固定端") == "边-固定端"

    with pytest.raises(NamePolicyError):
        policy.compose("部件", " 偏心孔板")
    with pytest.raises(NamePolicyError):
        policy.compose("未知", "偏心孔板")
    with pytest.raises(NamePolicyError):
        policy.compose("部件", "Part-1")
    with pytest.raises(NamePolicyError):
        policy.validate("部件-Ａ板")


def test_a2_recipe_tools_cover_required_primitives_transforms_and_bounded_preview() -> (
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


def test_a2_plate_hole_accepts_coordinates_or_offset_and_rejects_incomplete_hole() -> (
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
    with pytest.raises(ValueError, match="完整位于"):
        plate_with_hole_geometry(
            "实体-孔板",
            width=10.0,
            height=6.0,
            hole_radius=1.0,
            hole_center=(0.5, 3.0),
        )


def test_a2_unit_summary_keeps_explicit_not_applicable_fields() -> None:
    units = UnitContextSummary(
        length="mm",
        force="N",
        stress="MPa",
        density=None,
        acceleration=None,
        convention="N-mm-MPa",
    )

    assert units.to_dict() == {
        "length": "mm",
        "force": "N",
        "stress": "MPa",
        "density": None,
        "acceleration": None,
        "convention": "N-mm-MPa",
    }
