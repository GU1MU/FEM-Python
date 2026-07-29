from __future__ import annotations

from fem.application.preprocessing import generate_fem_model
from fem.geometry import RectangleGeometry, RevolvedGeometry
from fem.mesh.settings import MeshSettings


def test_axis_angle_profile_sweep_generates_tetrahedral_model(real_gmsh) -> None:
    del real_gmsh
    recipe = RevolvedGeometry(
        RectangleGeometry("扫掠体", 2.0, 1.0),
        "x",
        180.0,
        ("face:domain",),
    )

    model = generate_fem_model(
        recipe,
        MeshSettings(0.5, cell_shape="tetrahedron"),
    )

    assert model.mesh.nodes
    assert model.mesh.elements
    assert {element.type for element in model.mesh.elements} == {"Tet4"}
