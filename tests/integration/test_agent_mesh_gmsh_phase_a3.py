from __future__ import annotations

from fem.application.preprocessing import generate_fem_model
from fem.geometry import (
    DiskGeometry,
    LogicalEntityRef,
    RectangleGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    legacy_sketch_to_strict,
)
from fem.geometry.errors import GeometryError
from fem.mesh import gmsh as gmsh_meshing
from fem.mesh.settings import LocalMeshControl, MeshSizeFalloff
from fem_agent.mesh_authoring import MeshIntent


def _types(model) -> set[str]:
    return {str(element.type) for element in model.mesh.elements}


def test_real_a3_explicit_strict_quad_uses_requested_size_without_fallback(
    real_gmsh,
    monkeypatch,
) -> None:
    del real_gmsh
    specs = []
    original_generate = gmsh_meshing.Mesher.generate

    def capture_generate(self, spec):
        specs.append(spec)
        return original_generate(self, spec)

    monkeypatch.setattr(gmsh_meshing.Mesher, "generate", capture_generate)
    recipe = RectangleGeometry("实体-矩形", 8.0, 4.0)
    coarse = MeshIntent(
        "quadrilateral",
        1,
        global_size=1.0,
    ).to_mesh_settings(recipe)
    fine = MeshIntent(
        "quadrilateral",
        1,
        global_size=0.4,
    ).to_mesh_settings(recipe)

    coarse_model = generate_fem_model(recipe, coarse)
    fine_model = generate_fem_model(recipe, fine)

    assert _types(coarse_model) == {"Quad4"}
    assert _types(fine_model) == {"Quad4"}
    assert len(fine_model.mesh.nodes) > len(coarse_model.mesh.nodes)
    assert all(
        isinstance(spec, gmsh_meshing.AutoMeshSpec)
        and spec.level == 3
        and spec.cell_shape == "quad"
        for spec in specs
    )


def test_real_a3_auto_triangle_with_generic_local_refinement_keeps_absolute_sizes(
    real_gmsh,
    monkeypatch,
) -> None:
    del real_gmsh
    specs = []
    original_generate = gmsh_meshing.Mesher.generate

    def capture_generate(self, spec):
        specs.append(spec)
        return original_generate(self, spec)

    monkeypatch.setattr(gmsh_meshing.Mesher, "generate", capture_generate)
    recipe = legacy_sketch_to_strict(
        SketchGeometry(
            "草图-通用孔板",
            (
                SketchRectangle("material", 0.0, 0.0, 10.0, 6.0),
                SketchCircle("cut", 6.5, 2.0, 1.0),
            ),
        )
    )
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
    assert len(specs) == 1
    assert isinstance(specs[0], gmsh_meshing.AutoMeshSpec)
    assert specs[0].level == 3
    assert specs[0].cell_shape == "tri"


def test_real_a3_strict_quad_failure_does_not_downgrade_to_triangles(
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
