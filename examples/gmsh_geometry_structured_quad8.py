"""Build, solve, and export a typed structured Quad8 rectangle."""

from pathlib import Path

import numpy as np

from fem import materials, post, steps
from fem.core import FEMModel, validate_model
from fem import geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import edges, elements, nodes
from fem.solvers import static_linear


def main() -> None:
    """Run the headless typed-facade structured-mesh workflow."""
    with geometry.model("gmsh_geometry_structured_quad8", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        curves = cad.boundary([surface])
        horizontal = cad.select(curves, y=0.0) + cad.select(curves, y=1.0)
        vertical = cad.select(curves, x=0.0) + cad.select(curves, x=2.0)
        if len(horizontal) != 2 or len(vertical) != 2:
            raise RuntimeError("expected two horizontal and two vertical boundaries")

        mesher = gmsh_meshing.Mesher(cad)
        for curve in horizontal:
            mesher.transfinite_curve(curve, num_nodes=5)
        for curve in vertical:
            mesher.transfinite_curve(curve, num_nodes=3)
        mesher.transfinite_surface(surface)
        mesher.recombine(surface)

        native_mesh = mesher.generate(
            gmsh_meshing.MeshSpec(order=2, recombine=False)
        )
        mesh = gmsh_io.read(native_mesh)

    if mesh.num_elements != 8:
        raise RuntimeError(
            f"expected 8 structured cells, imported {mesh.num_elements}"
        )
    element_types = {element.type for element in mesh.elements}
    if element_types != {"Quad8"}:
        raise RuntimeError(f"expected only Quad8 elements, imported {element_types!r}")

    model = FEMModel(mesh=mesh, name="gmsh_geometry_structured_quad8")
    domain_elements = elements.set_all(mesh, "DOMAIN")
    fixed_nodes = nodes.set_by_x(mesh, "FIXED", 0.0)
    traction_nodes = nodes.set_by_x(mesh, "TRACTION", 2.0)
    traction_edges = edges.edge_by_x(mesh, "TRACTION", 2.0)
    model.element_sets[domain_elements.name] = domain_elements
    model.node_sets[fixed_nodes.name] = fixed_nodes
    model.node_sets[traction_nodes.name] = traction_nodes
    model.edges[traction_edges.name] = traction_edges

    elastic = materials.linear_elastic.material("elastic", E=1000.0, nu=0.3)
    materials.add(model, elastic)
    materials.assign(model, "elastic", "DOMAIN")

    load_step = steps.static("pull")
    steps.displacement(load_step, "FIXED", components=(1, 2))
    steps.edge_traction(load_step, "TRACTION", vector=(1.0, 0.0))
    steps.add(model, load_step)
    validate_model(model)

    result = static_linear.solve(model, load_step)
    if not np.all(np.isfinite(result.U)):
        raise RuntimeError("the structured Quad8 displacement solution is not finite")

    output_name = model.name or "gmsh_geometry_structured_quad8"
    output_dir = Path("results") / output_name
    post.vtk.export.from_result(
        result,
        output_dir=output_dir,
        name=output_name,
    )
    print(
        f"Solved {model.mesh.num_elements} structured Quad8 elements and wrote "
        f"{output_dir / f'{output_name}.vtk'}"
    )


if __name__ == "__main__":
    main()
