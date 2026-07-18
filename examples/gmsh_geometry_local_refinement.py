"""Locally refine, solve, and export a typed plate-with-hole model."""

from pathlib import Path

import numpy as np

from fem import materials, post, steps
from fem.core import FEMModel, validate_model
from fem.geometry import gmsh as geometry
from fem.io import gmsh as gmsh_io
from fem.selection import edges, elements, nodes
from fem.solvers import static_linear


MODEL_NAME = "gmsh_geometry_local_refinement"


def main() -> None:
    """Run the headless local-refinement-to-VTK workflow."""
    with geometry.model(MODEL_NAME, dimension=2) as cad:
        plate = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        hole = cad.disk(1.0, 0.5, 0.2)
        domain = cad.cut([plate], [hole]).of_dimension(2)
        if len(domain) != 1:
            raise RuntimeError("expected one plate surface after cutting the hole")

        boundary = cad.boundary(domain)
        fixed = cad.select(boundary, x=0.0)
        traction = cad.select(boundary, x=2.0)
        bottom = cad.select(boundary, y=0.0)
        top = cad.select(boundary, y=1.0)
        outer_curve_groups = (fixed, traction, bottom, top)
        if any(len(group) != 1 for group in outer_curve_groups):
            raise RuntimeError("expected four axis-aligned outer plate boundaries")

        outer_curves = frozenset(
            curve for group in outer_curve_groups for curve in group
        )
        if len(outer_curves) != 4:
            raise RuntimeError("expected four distinct outer plate boundaries")
        hole_curves = tuple(curve for curve in boundary if curve not in outer_curves)
        if not hole_curves:
            raise RuntimeError("expected at least one circular hole boundary")

        hole_distance = cad.distance_field(curves=hole_curves, sampling=100)
        hole_refinement = cad.threshold_field(
            hole_distance,
            size_min=0.025,
            size_max=0.20,
            dist_min=0.05,
            dist_max=0.35,
        )
        cad.background_field(hole_refinement)
        native_mesh = cad.generate_mesh(order=1)
        mesh = gmsh_io.read(
            native_mesh,
            plane_type="stress",
            thickness=1.0,
        )

    element_types = {element.type for element in mesh.elements}
    if element_types != {"Tri3"}:
        raise RuntimeError(
            f"expected only first-order triangles, imported {element_types!r}"
        )
    model = FEMModel(mesh=mesh, name=MODEL_NAME)
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

    output_dir = Path("results") / MODEL_NAME
    post.vtk.export.from_result(
        result,
        output_dir=output_dir,
        name=MODEL_NAME,
    )
    print(
        f"Solved {model.mesh.num_elements} locally refined Tri3 elements and wrote "
        f"{output_dir / f'{MODEL_NAME}.vtk'}"
    )


if __name__ == "__main__":
    main()
