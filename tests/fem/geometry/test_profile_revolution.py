from __future__ import annotations

from math import pi

import pytest

from fem.application.recipe_compiler import TopologyResolutionError, compile_recipe
from fem.geometry import (
    LogicalEntityRef,
    MovedGeometry,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    geometry_dimension,
    model,
)
from fem.geometry.recipe_topology import describe_recipe_topology


def _rectangle() -> RectangleGeometry:
    return RectangleGeometry("扫掠草图", 2.0, 1.0)


def test_revolved_recipe_validates_axis_angle_and_source_profile() -> None:
    recipe = RevolvedGeometry(
        _rectangle(),
        "Y",
        180.0,
        ("face:domain",),
    )

    assert recipe.axis == "y"
    assert recipe.angle_degrees == 180.0
    assert recipe.source_face_ids == ("face:domain",)
    assert geometry_dimension(recipe) == 3
    assert describe_recipe_topology(recipe).signature.logical_ids == (
        "face:start",
        "face:end",
        "face:sides",
        "body:domain",
    )
    with pytest.raises(ValueError, match="sweep axis"):
        RevolvedGeometry(_rectangle(), "a", 90.0)
    with pytest.raises(ValueError, match="sweep angle"):
        RevolvedGeometry(_rectangle(), "x", 0.0)
    with pytest.raises(ValueError, match="sweep angle"):
        RevolvedGeometry(_rectangle(), "x", 361.0)


def test_revolved_recipe_compiles_positive_volume_and_rejects_degenerate_axis(
    real_gmsh,
) -> None:
    recipe = RevolvedGeometry(
        _rectangle(),
        "x",
        180.0,
        ("face:domain",),
    )
    with model("profile-sweep-test", dimension=3) as cad:
        compiled = compile_recipe(cad, recipe)
        assert len(compiled.domain) == 1
        assert cad.volume(compiled.domain[0]) == pytest.approx(pi)

    degenerate = RevolvedGeometry(
        _rectangle(),
        "z",
        90.0,
        ("face:domain",),
    )
    with model("profile-sweep-degenerate-test", dimension=3) as cad:
        with pytest.raises(TopologyResolutionError, match="zero-volume"):
            compile_recipe(cad, degenerate)


@pytest.mark.parametrize(
    "recipe",
    (
        MovedGeometry(
            RevolvedGeometry(
                _rectangle(),
                "x",
                180.0,
                ("face:domain",),
            ),
            1.0,
            2.0,
            3.0,
        ),
        RotatedGeometry(
            RevolvedGeometry(
                _rectangle(),
                "x",
                180.0,
                ("face:domain",),
            ),
            "z",
            37.0,
        ),
    ),
)
def test_rigid_transforms_preserve_revolved_volume_and_body_identity(
    real_gmsh,
    recipe,
) -> None:
    with model(
        f"profile-sweep-{type(recipe).__name__}-test",
        dimension=3,
    ) as cad:
        compiled = compile_recipe(cad, recipe)

        assert len(compiled.domain) == 1
        assert cad.volume(compiled.domain[0]) == pytest.approx(pi)
        assert compiled.resolve(LogicalEntityRef("body:domain")) == (
            compiled.domain[0],
        )
