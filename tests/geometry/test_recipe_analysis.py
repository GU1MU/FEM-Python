from __future__ import annotations

import pytest

from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CircleFrame,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    RectangleFrame,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    axis_aligned_rectangle,
    expand_sketch_recipe,
    recipe_characteristic_size,
    supports_structured_hexahedron,
    transformed_circle,
)
from fem.geometry.recipe_topology import topology_fingerprint_for_recipe


def test_sketch_expansion_is_deterministic_and_detached() -> None:
    recipe = SketchGeometry(
        "Composite",
        (
            SketchRectangle("material", 2.0, 3.0, 4.0, 5.0),
            SketchCircle("material", -1.0, 2.0, 0.5),
            SketchCircle("cut", 3.0, 4.0, 0.25),
        ),
    )
    fingerprint = topology_fingerprint_for_recipe(recipe)

    first = expand_sketch_recipe(recipe)
    second = expand_sketch_recipe(recipe)

    assert first == second
    assert first is not second
    assert isinstance(first, BooleanGeometry)
    assert first.operation == "cut"
    assert first.name == recipe.name
    assert isinstance(first.object_geometry, BooleanGeometry)
    assert first.object_geometry.operation == "fuse"
    assert isinstance(first.tool_geometry, MovedGeometry)
    assert first.tool_geometry.base.name == "Composite-Contour-3"
    assert (first.tool_geometry.dx, first.tool_geometry.dy) == (3.0, 4.0)
    assert recipe.contours[0].operation == "material"
    assert topology_fingerprint_for_recipe(recipe) == fingerprint


def test_axis_aligned_rectangle_uses_one_frozen_tolerance() -> None:
    rectangle = MovedGeometry(
        RectangleGeometry("Plate", 4.0, 2.0),
        3.0,
        -1.0,
    )

    assert axis_aligned_rectangle(rectangle) == RectangleFrame(
        3.0,
        -1.0,
        4.0,
        2.0,
    )
    assert axis_aligned_rectangle(
        RotatedGeometry(rectangle, "z", 360.0 + 5.0e-13)
    ) == RectangleFrame(3.0, -1.0, 4.0, 2.0)
    assert axis_aligned_rectangle(
        RotatedGeometry(rectangle, "z", 90.0)
    ) is None
    assert axis_aligned_rectangle(
        SketchGeometry(
            "Sketch",
            (SketchRectangle("material", 2.0, 1.0, 3.0, 4.0),),
        )
    ) == RectangleFrame(2.0, 1.0, 3.0, 4.0)


def test_transformed_circle_tracks_move_rotation_and_sketch_expansion() -> None:
    transformed = RotatedGeometry(
        MovedGeometry(DiskGeometry("Disk", 2.0), 1.0, 2.0),
        "z",
        90.0,
    )

    frame = transformed_circle(transformed)

    assert frame is not None
    assert frame.center_x == pytest.approx(-2.0)
    assert frame.center_y == pytest.approx(1.0)
    assert frame.radius == pytest.approx(2.0)
    assert transformed_circle(
        SketchGeometry(
            "Circle",
            (SketchCircle("material", 4.0, -3.0, 0.75),),
        )
    ) == CircleFrame(4.0, -3.0, 0.75)


@pytest.mark.parametrize(
    ("recipe", "expected"),
    (
        (RectangleGeometry("Rectangle", 4.0, 2.0), 2.0),
        (DiskGeometry("Disk", 1.5), 3.0),
        (BoxGeometry("Box", 4.0, 3.0, 2.0), 2.0),
        (CylinderGeometry("Cylinder", 2.0, 3.0), 3.0),
        (
            SketchGeometry(
                "Sketch",
                (
                    SketchRectangle("material", 0.0, 0.0, 6.0, 4.0),
                    SketchCircle("cut", 3.0, 2.0, 0.5),
                ),
            ),
            1.0,
        ),
        (
            ExtrudedGeometry(
                SketchGeometry(
                    "Extrusion",
                    (SketchRectangle("material", 0.0, 0.0, 6.0, 4.0),),
                ),
                1.5,
            ),
            1.5,
        ),
    ),
)
def test_recipe_characteristic_size_is_shared_for_every_recipe_family(
    recipe,
    expected,
) -> None:
    assert recipe_characteristic_size(recipe) == pytest.approx(expected)
    assert recipe_characteristic_size(
        RotatedGeometry(MovedGeometry(recipe, 1.0, -2.0), "z", 37.0)
    ) == pytest.approx(expected)


def test_structured_hexahedron_eligibility_matches_native_mesher() -> None:
    box = BoxGeometry("Box", 2.0, 3.0, 4.0)
    rectangular_extrusion = ExtrudedGeometry(
        SketchGeometry(
            "Sketch",
            (SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),),
        ),
        3.0,
    )
    cut_extrusion = ExtrudedGeometry(
        SketchGeometry(
            "Cut",
            (
                SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),
                SketchCircle("cut", 1.0, 0.5, 0.2),
            ),
        ),
        3.0,
    )

    assert supports_structured_hexahedron(box)
    assert supports_structured_hexahedron(
        RotatedGeometry(MovedGeometry(box, 1.0, 2.0, 3.0), "z", 20.0)
    )
    assert supports_structured_hexahedron(rectangular_extrusion)
    assert not supports_structured_hexahedron(cut_extrusion)
    assert not supports_structured_hexahedron(
        ExtrudedGeometry(RectangleGeometry("Rectangle", 2.0, 1.0), 3.0)
    )


@pytest.mark.parametrize(
    "helper",
    (
        axis_aligned_rectangle,
        transformed_circle,
        recipe_characteristic_size,
        supports_structured_hexahedron,
    ),
)
def test_recipe_analysis_rejects_unsupported_values(helper) -> None:
    with pytest.raises(TypeError, match="Unsupported native geometry recipe"):
        helper(object())


def test_sketch_expansion_rejects_non_sketch_values() -> None:
    with pytest.raises(TypeError, match="SketchGeometry"):
        expand_sketch_recipe(RectangleGeometry("Rectangle", 2.0, 1.0))
