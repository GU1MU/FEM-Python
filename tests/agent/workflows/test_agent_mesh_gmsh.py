from __future__ import annotations


from fem.application.preprocessing import generate_fem_model
from fem.geometry import DiskGeometry, LogicalEntityRef, RectangleGeometry, SketchCircle, SketchRectangle
from fem.geometry.errors import GeometryError
from fem.mesh.settings import LocalMeshControl, MeshSizeFalloff
from fem_agent.mesh_authoring import MeshIntent
from fem_agent.geometry_authoring import planar_sketch_geometry


def _types(model) -> set[str]:
    return {str(element.type) for element in model.mesh.elements}


def test_strict_quad_refinement_increases_mesh_density(real_gmsh) -> None:
    del real_gmsh
    recipe = RectangleGeometry("Rectangle", 8.0, 4.0)
    models = [
        generate_fem_model(
            recipe,
            MeshIntent("quadrilateral", 1, global_size=size).to_mesh_settings(recipe),
        )
        for size in (1.0, 0.4)
    ]
    assert all(_types(model) == {"Quad4"} for model in models)
    assert len(models[1].mesh.nodes) > len(models[0].mesh.nodes)


def test_real_auto_triangle_preserves_local_refinement(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = planar_sketch_geometry(
        "Refined plate",
        contours=(
            SketchRectangle("material", 0.0, 0.0, 10.0, 6.0),
            SketchCircle("cut", 6.5, 2.0, 1.0),
        ),
    ).recipe
    intent = MeshIntent(
        "triangle",
        1,
        auto_level=4,
        local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("edge:C5"),
                    0.1,
                    MeshSizeFalloff("target_radius", 0.25, 2.0),
                ),
        ),
    )
    settings = intent.to_mesh_settings(recipe)

    model = generate_fem_model(recipe, settings)

    assert settings.auto_level == 4
    assert settings.size > settings.local_controls[0].size
    assert _types(model) == {"Tri3"}
    assert len(model.mesh.nodes) > 0


def test_real_strict_quad_failure_does_not_downgrade_to_triangles(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = DiskGeometry("实体-圆盘", 2.0)
    settings = MeshIntent(
        "quadrilateral",
        1,
        global_size=0.5,
    ).to_mesh_settings(recipe)

    try:
        model = generate_fem_model(recipe, settings)
    except GeometryError as error:
        assert "strict" in str(error).casefold() or "quad" in str(error).casefold()
    else:
        assert _types(model) == {"Quad4"}
        assert "Tri3" not in _types(model)
