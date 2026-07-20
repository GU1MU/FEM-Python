"""Mesh and import a filleted box with typed Gmsh geometry."""

from fem.core import Mesh3D
from fem import geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing


MODEL_NAME = "gmsh_geometry_filleted_box"


def main() -> None:
    """Run the headless fillet-to-Tet4 import workflow."""
    with geometry.model(MODEL_NAME, dimension=3) as cad:
        box = cad.box(0.0, 0.0, 0.0, 1.0, 0.7, 0.5)
        surface = cad.boundary((box,), combined=False)[0]
        curve = cad.boundary((surface,), combined=False)[0]
        filleted = cad.fillet((box,), (curve,), (0.1,))
        if not filleted.primary or any(
            entity.dimension != 3 for entity in filleted.primary
        ):
            raise RuntimeError("fillet did not create a replacement solid")

        native_mesh = gmsh_meshing.Mesher(cad).generate(
            gmsh_meshing.MeshSpec(size=0.2)
        )
        mesh = gmsh_io.read(native_mesh)

    if not isinstance(mesh, Mesh3D):
        raise RuntimeError("expected a three-dimensional imported mesh")
    element_types = sorted({element.type for element in mesh.elements})
    if element_types != ["Tet4"]:
        raise RuntimeError(f"expected Tet4 elements, got {element_types}")
    print(
        f"Imported filleted box: {mesh.num_nodes} nodes, "
        f"{mesh.num_elements} Tet4 elements"
    )


if __name__ == "__main__":
    main()
