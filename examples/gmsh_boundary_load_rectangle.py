"""Solve a caller-owned Gmsh rectangle with FEM-side boundary selections."""

from pathlib import Path

import gmsh as gmsh_api
import numpy as np

from fem import materials, post, steps
from fem.core import FEMModel, validate_model
from fem.io import gmsh as gmsh_io
from fem.selection import edges, elements, nodes
from fem.solvers import static_linear


def main() -> None:
    owns_session = not gmsh_api.isInitialized()
    if owns_session:
        gmsh_api.initialize()

    try:
        gmsh_api.option.setNumber("General.Terminal", 0)
        gmsh_api.model.add("gmsh_boundary_load_rectangle")
        gmsh_api.model.occ.addRectangle(0.0, 0.0, 0.0, 2.0, 1.0)
        gmsh_api.model.occ.synchronize()

        gmsh_api.model.mesh.setSize(gmsh_api.model.getEntities(0), 0.35)
        gmsh_api.option.setNumber("Mesh.ElementOrder", 1)
        gmsh_api.option.setNumber("Mesh.RecombineAll", 0)
        gmsh_api.model.mesh.generate(2)

        mesh = gmsh_io.from_model(dimension=2)
        model = FEMModel(mesh=mesh, name="gmsh_boundary_load_rectangle")
        domain_elements = elements.set_all(mesh, "DOMAIN")
        fixed_nodes = nodes.set_by_x(mesh, "FIXED", 0.0)
        traction_nodes = nodes.set_by_x(mesh, "TRACTION", 2.0)
        traction_edges = edges.edge_by_x(mesh, "TRACTION", 2.0)
        model.element_sets[domain_elements.name] = domain_elements
        model.node_sets[fixed_nodes.name] = fixed_nodes
        model.node_sets[traction_nodes.name] = traction_nodes
        model.edges[traction_edges.name] = traction_edges

        print(
            f"TRACTION contains {len(traction_edges.edges)} selected FEM edge(s): "
            f"{traction_edges.edges[:3]}"
        )

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
            raise RuntimeError("the imported distributed-load solution is not finite")

        output_dir = Path("results") / model.name
        post.vtk.export.from_result(
            result,
            output_dir=output_dir,
            name=model.name,
        )
        print(f"Wrote {output_dir / f'{model.name}.vtk'}")
    finally:
        if owns_session:
            gmsh_api.finalize()


if __name__ == "__main__":
    main()
