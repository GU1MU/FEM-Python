"""Build, solve, and export a typed structured Quad8 rectangle."""

from pathlib import Path

import numpy as np

from fem import materials, post, steps
from fem.core import validate_model
from fem.geometry import gmsh as geometry
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

        for curve in horizontal:
            cad.transfinite_curve(curve, num_nodes=5)
        for curve in vertical:
            cad.transfinite_curve(curve, num_nodes=3)
        cad.transfinite_surface(surface)
        cad.recombine(surface)

        fixed = cad.select(curves, x=0.0)
        traction = cad.select(curves, x=2.0)
        cad.physical("DOMAIN", [surface])
        cad.physical("FIXED", fixed)
        cad.physical("TRACTION", traction)
        imported = cad.generate_mesh(order=2, recombine=False)

    if imported.mesh.num_elements != 8:
        raise RuntimeError(
            f"expected 8 structured cells, imported {imported.mesh.num_elements}"
        )
    element_types = {element.type for element in imported.mesh.elements}
    if element_types != {"Quad8"}:
        raise RuntimeError(f"expected only Quad8 elements, imported {element_types!r}")

    model = imported.to_fem_model("gmsh_geometry_structured_quad8")
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
