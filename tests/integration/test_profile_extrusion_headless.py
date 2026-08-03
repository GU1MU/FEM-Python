from __future__ import annotations

from dataclasses import replace

import pytest

from fem import geometry
from fem.application.definitions import NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.application.preprocessing import generate_fem_model
from fem.application.recipe_compiler import compile_recipe
from fem.geometry import (
    ExtrudedGeometry,
    LogicalEntityRef,
    SketchPlane,
)
from fem.application.native_regions import RecipeRegionSelector
from fem.io.project import load_project, save_project
from fem.mesh.settings import MeshSettings
from tests.geometry.test_profile_extrusion import (
    hole_profile_sketch,
    profile_face_id,
    two_profile_sketch,
)


def test_selected_one_and_two_profiles_compile_to_expected_volume_count(
    real_gmsh,
) -> None:
    del real_gmsh
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    second = profile_face_id(sketch, "L5")

    with geometry.model("selected-one-profile", dimension=3) as cad:
        compiled = compile_recipe(
            cad,
            ExtrudedGeometry(sketch, 1.0, (first,)),
        )
        assert len(compiled.domain) == 1
        assert compiled.resolve(LogicalEntityRef("face:bottom"))
        assert compiled.resolve(LogicalEntityRef("face:side/L1"))

    first_name = first.split(":", 1)[1]
    second_name = second.split(":", 1)[1]
    with geometry.model("selected-two-profiles", dimension=3) as cad:
        compiled = compile_recipe(
            cad,
            ExtrudedGeometry(sketch, 1.0, (first, second)),
        )
        assert len(compiled.domain) == 2
        assert len(compiled.resolve(LogicalEntityRef("body:domain"))) == 2
        assert compiled.resolve(
            LogicalEntityRef(f"face:bottom/{first_name}")
        )
        assert compiled.resolve(
            LogicalEntityRef(f"face:top/{second_name}")
        )

    with geometry.model("legacy-all-profiles", dimension=3) as cad:
        compiled = compile_recipe(
            cad,
            ExtrudedGeometry(sketch, 1.0),
        )
        assert len(compiled.domain) == 2


def test_selected_only_profile_mesh_omits_other_planar_domain(real_gmsh) -> None:
    del real_gmsh
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")

    model = generate_fem_model(
        ExtrudedGeometry(sketch, 1.0, (first,)),
        MeshSettings(0.5),
    )

    assert model.mesh.nodes
    assert max(node.x for node in model.mesh.nodes) == pytest.approx(
        2.0,
        abs=1.0e-7,
    )


def test_selected_profile_save_reopen_and_remesh(
    real_gmsh,
    tmp_path,
) -> None:
    del real_gmsh
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    recipe = ExtrudedGeometry(sketch, 1.0, (first,))
    settings = MeshSettings(0.5)
    target = save_project(
        tmp_path / "selected-profile.femproj",
        ProjectSnapshot(
            source_kind="native",
            parts=(NativePart(),),
            geometry_recipe=recipe,
            mesh_settings=settings,
            feature_history=derive_feature_history(recipe),
        ),
    )

    reopened = load_project(target).snapshot
    model = generate_fem_model(
        reopened.geometry_recipe,
        reopened.mesh_settings,
    )

    assert isinstance(reopened.geometry_recipe, ExtrudedGeometry)
    assert reopened.geometry_recipe.source_face_ids == (first,)
    assert model.mesh.nodes
    assert max(node.x for node in model.mesh.nodes) == pytest.approx(
        2.0,
        abs=1.0e-7,
    )


def test_strict_hole_profile_keeps_inner_side_lineage(real_gmsh) -> None:
    del real_gmsh
    sketch = hole_profile_sketch()
    source = profile_face_id(sketch, "L1")

    with geometry.model("strict-hole-profile", dimension=3) as cad:
        compiled = compile_recipe(
            cad,
            ExtrudedGeometry(sketch, 1.0, (source,)),
        )

        inner_sides = {
            entity
            for edge_id in ("L5", "L6", "L7", "L8")
            for entity in compiled.resolve(
                LogicalEntityRef(f"face:side/{edge_id}")
            )
        }
        outer_sides = {
            entity
            for edge_id in ("L1", "L2", "L3", "L4")
            for entity in compiled.resolve(
                LogicalEntityRef(f"face:side/{edge_id}")
            )
        }
        assert inner_sides
        assert inner_sides.isdisjoint(outer_sides)
        assert set(
            compiled.region_bindings[RecipeRegionSelector.HOLE]
        ) == inner_sides


def test_strict_profile_extrudes_along_positive_sketch_normal(real_gmsh) -> None:
    del real_gmsh
    sketch = two_profile_sketch()
    sketch = replace(
        sketch,
        plane=SketchPlane(
            origin=(1.0, 2.0, 3.0),
            x_direction=(1.0, 0.0, 0.0),
            y_direction=(0.0, 0.0, 1.0),
        ),
    )
    source = profile_face_id(sketch, "L1")

    with geometry.model("positive-sketch-normal", dimension=3) as cad:
        compiled = compile_recipe(
            cad,
            ExtrudedGeometry(sketch, 2.0, (source,)),
        )
        bottom = compiled.resolve(LogicalEntityRef("face:bottom"))[0]
        top = compiled.resolve(LogicalEntityRef("face:top"))[0]
        bottom_center = cad.center_of_mass(bottom)
        top_center = cad.center_of_mass(top)

        assert tuple(
            top_value - bottom_value
            for bottom_value, top_value in zip(
                bottom_center, top_center, strict=True
            )
        ) == pytest.approx((0.0, -2.0, 0.0), abs=1.0e-8)
