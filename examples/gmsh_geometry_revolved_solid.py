"""Build, solve, and export a revolved solid with typed Gmsh geometry."""

import math
from pathlib import Path

import numpy as np

from fem import materials, post, steps
from fem.core import FEMModel, Mesh3D, validate_model
from fem.geometry import gmsh as geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import elements, nodes
from fem.solvers import static_linear


MODEL_NAME = "gmsh_geometry_revolved_solid"
HEIGHT = 1.0
AXIAL_FORCE = 1000.0


def main() -> None:
    """Run the headless revolve-to-Tet4-to-VTK workflow."""
    with geometry.model(MODEL_NAME, dimension=3) as cad:
        profile = cad.rectangle(0.35, 0.0, 0.35, HEIGHT)
        revolved = cad.revolve(
            (profile,),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            2.0 * math.pi,
        )
        if not revolved.primary or any(
            entity.dimension != 3 for entity in revolved.primary
        ):
            raise RuntimeError("revolution did not create a solid")

        native_mesh = gmsh_meshing.Mesher(cad).generate(
            gmsh_meshing.MeshSpec(size=0.3)
        )
        mesh = gmsh_io.read(native_mesh)

    if not isinstance(mesh, Mesh3D):
        raise RuntimeError("expected a three-dimensional imported mesh")
    element_types = sorted({element.type for element in mesh.elements})
    if element_types != ["Tet4"]:
        raise RuntimeError(f"expected Tet4 elements, got {element_types}")

    model = FEMModel(mesh=mesh, name=MODEL_NAME)
    solid = elements.set_all(mesh, "SOLID")
    fixed = nodes.set_by_y(mesh, "FIXED", 0.0)
    loaded = nodes.set_by_y(mesh, "LOADED", HEIGHT)
    if not fixed.node_ids or not loaded.node_ids:
        raise RuntimeError("expected nodes on both ends of the revolved solid")
    model.element_sets[solid.name] = solid
    model.node_sets[fixed.name] = fixed
    model.node_sets[loaded.name] = loaded

    elastic = materials.linear_elastic.material(
        "elastic",
        E=70.0e9,
        nu=0.3,
    )
    materials.add(model, elastic)
    materials.assign(model, elastic, "SOLID")

    load_step = steps.static("axial_pull")
    steps.displacement(load_step, "FIXED", components=(1, 2, 3))
    steps.nodal_load(
        load_step,
        "LOADED",
        component=2,
        value=AXIAL_FORCE / len(loaded.node_ids),
    )
    steps.add(model, load_step)
    validate_model(model)

    result = static_linear.solve(model, load_step, name=MODEL_NAME)
    if not np.all(np.isfinite(result.U)):
        raise RuntimeError("the revolved-solid displacement solution is not finite")
    if not np.all(np.isfinite(result.reactions)):
        raise RuntimeError("the revolved-solid reaction solution is not finite")

    output_dir = Path("results") / MODEL_NAME
    vtk_path = output_dir / f"{MODEL_NAME}.vtk"
    post.vtk.export.from_result(
        result,
        output_dir=output_dir,
        name=MODEL_NAME,
    )
    print(
        f"Solved {mesh.num_nodes} nodes and {mesh.num_elements} Tet4 elements; "
        f"wrote {vtk_path.resolve()}"
    )


if __name__ == "__main__":
    main()
