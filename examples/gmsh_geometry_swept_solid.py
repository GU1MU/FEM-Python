"""Mesh and import a swept solid with typed Gmsh geometry."""

from fem.core import Mesh3D
from fem import geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing


MODEL_NAME = "gmsh_geometry_swept_solid"


def main() -> None:
    """Run the headless sweep-to-Tet4 import workflow."""
    with geometry.model(MODEL_NAME, dimension=3) as cad:
        profile = cad.disk(0.0, 0.0, 0.25)
        path_start = cad.point(0.0, 0.0, 0.0)
        path_end = cad.point(0.0, 0.0, 1.25)
        path_curve = cad.line(path_start, path_end)
        path = cad.wire((cad.orient(path_curve),), closed=False)
        swept = cad.sweep((profile,), path, frame="corrected_frenet")
        if not swept.primary or any(
            entity.dimension != 3 for entity in swept.primary
        ):
            raise RuntimeError("sweep did not create a solid")

        native_mesh = gmsh_meshing.Mesher(cad).generate(
            gmsh_meshing.MeshSpec(size=0.25)
        )
        mesh = gmsh_io.read(native_mesh)

    if not isinstance(mesh, Mesh3D):
        raise RuntimeError("expected a three-dimensional imported mesh")
    element_types = sorted({element.type for element in mesh.elements})
    if element_types != ["Tet4"]:
        raise RuntimeError(f"expected Tet4 elements, got {element_types}")
    print(
        f"Imported swept solid: {mesh.num_nodes} nodes, "
        f"{mesh.num_elements} Tet4 elements"
    )


if __name__ == "__main__":
    main()
