"""Build, solve, and export a scripted plate with a circular hole."""

from pathlib import Path

import numpy as np

from fem import materials, post, steps
from fem.core import FEMModel, validate_model
from fem.geometry import gmsh as geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import edges, elements, nodes
from fem.solvers import static_linear


def main() -> None:
    """Run the headless geometry-to-VTK workflow."""
    with geometry.model("gmsh_geometry_plate", dimension=2) as cad:
        plate = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        hole = cad.disk(1.0, 0.5, 0.2)
        domain = cad.cut([plate], [hole]).of_dimension(2)

        boundary = cad.boundary(domain)
        fixed = cad.select(boundary, x=0.0)
        traction = cad.select(boundary, x=2.0)
        if not fixed or not traction:
            raise RuntimeError("expected left and right plate boundaries")

        mesher = gmsh_meshing.Mesher(cad)
        native_mesh = mesher.generate(
            gmsh_meshing.MeshSpec(size=0.2, order=2)
        )
        mesh = gmsh_io.read(native_mesh)

    model = FEMModel(mesh=mesh, name="gmsh_geometry_plate")
    domain_elements = elements.set_all(mesh, "DOMAIN")
    fixed_nodes = nodes.set_by_x(mesh, "FIXED", 0.0)
    traction_nodes = nodes.set_by_x(mesh, "TRACTION", 2.0)
    traction_edges = edges.edge_by_x(mesh, "TRACTION", 2.0)
    model.element_sets[domain_elements.name] = domain_elements
    model.node_sets[fixed_nodes.name] = fixed_nodes
    model.node_sets[traction_nodes.name] = traction_nodes
    model.edges[traction_edges.name] = traction_edges

    elastic = materials.linear_elastic.material(
        "elastic",
        E=1000.0,
        nu=0.3,
    )
    materials.add(model, elastic)
    materials.assign(model, "elastic", "DOMAIN")

    load_step = steps.static("pull")
    steps.displacement(load_step, "FIXED", components=(1, 2))
    steps.edge_traction(load_step, "TRACTION", vector=(1.0, 0.0))
    steps.add(model, load_step)
    validate_model(model)

    result = static_linear.solve(model, "pull")
    if not np.all(np.isfinite(result.U)):
        raise RuntimeError("the displacement solution is not finite")
    if not np.all(np.isfinite(result.reactions)):
        raise RuntimeError("the reaction solution is not finite")

    output_name = model.name or "gmsh_geometry_plate"
    output_dir = Path("results") / output_name
    post.vtk.export.from_result(
        result,
        output_dir=output_dir,
        name=output_name,
    )
    print(
        f"Solved {model.mesh.num_elements} elements and wrote "
        f"{output_dir / f'{output_name}.vtk'}"
    )


if __name__ == "__main__":
    main()
