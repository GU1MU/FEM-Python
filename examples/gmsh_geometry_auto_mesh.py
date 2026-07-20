"""Generate and inspect a typed automatic full-quad mesh."""

from fem import geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing


MODEL_NAME = "gmsh_geometry_auto_mesh"


def main() -> None:
    """Run the headless automatic-meshing workflow."""
    with geometry.model(MODEL_NAME, dimension=2) as cad:
        surface = cad.disk(0.0, 0.0, 1.0)
        boundary = cad.boundary([surface])
        if not boundary:
            raise RuntimeError("expected a disk boundary")

        mesher = gmsh_meshing.Mesher(cad)
        native_mesh = mesher.generate(
            gmsh_meshing.AutoMeshSpec(
                level=3,
                cell_shape="quad",
                order=2,
            )
        )
        mesh = gmsh_io.read(native_mesh)

    element_types = {element.type for element in mesh.elements}
    if element_types != {"Quad8"}:
        raise RuntimeError(
            f"expected only Quad8 elements, imported {sorted(element_types)!r}"
        )
    print(
        "Generated level-3 automatic Quad8 mesh: "
        f"{mesh.num_nodes} nodes, "
        f"{mesh.num_elements} elements"
    )


if __name__ == "__main__":
    main()
