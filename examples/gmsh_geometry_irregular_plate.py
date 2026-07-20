"""Build, solve, and export a typed irregular plate with a circular-arc hole."""

from pathlib import Path

import numpy as np

from fem import materials, post, steps
from fem.core import FEMModel, validate_model
from fem import geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import edges, elements, nodes
from fem.solvers import static_linear


MODEL_NAME = "gmsh_geometry_irregular_plate"


def main() -> None:
    """Run the typed irregular-profile-to-Tri3-to-VTK workflow."""
    with geometry.model(MODEL_NAME, dimension=2) as cad:
        lower_left = cad.point(0.0, 0.0)
        lower_right = cad.point(2.0, 0.0)
        upper_right = cad.point(2.0, 1.0)
        arc_center = cad.point(1.5, 1.0)
        arc_end = cad.point(1.5, 1.5)
        upper_left = cad.point(0.0, 1.5)

        bottom = cad.line(lower_left, lower_right)
        right = cad.line(lower_right, upper_right)
        rounded_corner = cad.circular_arc(upper_right, arc_center, arc_end)
        top = cad.line(arc_end, upper_left)
        left = cad.line(upper_left, lower_left)
        outer_loop = cad.curve_loop(
            [
                cad.orient(bottom),
                cad.orient(right),
                cad.orient(rounded_corner),
                cad.orient(top),
                cad.orient(left),
            ]
        )

        hole_center = cad.point(0.9, 0.65)
        hole_east = cad.point(1.08, 0.65)
        hole_north = cad.point(0.9, 0.83)
        hole_west = cad.point(0.72, 0.65)
        hole_south = cad.point(0.9, 0.47)
        hole_arcs = (
            cad.circular_arc(hole_east, hole_center, hole_north),
            cad.circular_arc(hole_north, hole_center, hole_west),
            cad.circular_arc(hole_west, hole_center, hole_south),
            cad.circular_arc(hole_south, hole_center, hole_east),
        )
        hole_loop = cad.curve_loop(
            [
                cad.orient(hole_arcs[3], reversed=True),
                cad.orient(hole_arcs[2], reversed=True),
                cad.orient(hole_arcs[1], reversed=True),
                cad.orient(hole_arcs[0], reversed=True),
            ]
        )
        cad.plane_surface(outer_loop, holes=[hole_loop])

        mesher = gmsh_meshing.Mesher(cad)
        native_mesh = mesher.generate(
            gmsh_meshing.MeshSpec(size=0.12, order=1, recombine=False)
        )
        mesh = gmsh_io.read(
            native_mesh,
            plane_type="stress",
            thickness=1.0,
        )

    element_types = {element.type for element in mesh.elements}
    if element_types != {"Tri3"}:
        raise RuntimeError(
            f"expected only Tri3 elements, imported {sorted(element_types)!r}"
        )

    model = FEMModel(mesh=mesh, name=MODEL_NAME)
    domain = elements.set_all(mesh, "DOMAIN")
    fixed = nodes.set_by_x(mesh, "FIXED", 0.0)
    traction_nodes = nodes.set_by_x(mesh, "TRACTION", 2.0)
    traction_edge = edges.edge_by_x(mesh, "TRACTION", 2.0)
    model.element_sets[domain.name] = domain
    model.node_sets[fixed.name] = fixed
    model.node_sets[traction_nodes.name] = traction_nodes
    model.edges[traction_edge.name] = traction_edge

    elastic = materials.linear_elastic.material(
        "elastic",
        E=1000.0,
        nu=0.3,
    )
    materials.add(model, elastic)
    materials.assign(model, elastic, "DOMAIN")

    load_step = steps.static("pull")
    steps.displacement(load_step, "FIXED", components=(1, 2))
    steps.edge_traction(load_step, "TRACTION", vector=(1.0, 0.0))
    steps.add(model, load_step)
    validate_model(model)

    result = static_linear.solve(model, load_step)
    if not np.all(np.isfinite(result.U)):
        raise RuntimeError("the irregular-plate displacement solution is not finite")
    if not np.all(np.isfinite(result.reactions)):
        raise RuntimeError("the irregular-plate reaction solution is not finite")

    output_dir = Path("results") / MODEL_NAME
    post.vtk.export.from_result(
        result,
        output_dir=output_dir,
        name=MODEL_NAME,
    )
    print(
        f"Solved {model.mesh.num_elements} irregular-profile Tri3 elements and wrote "
        f"{output_dir / f'{MODEL_NAME}.vtk'}"
    )


if __name__ == "__main__":
    main()
