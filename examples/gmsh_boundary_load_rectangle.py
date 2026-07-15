"""Solve a Gmsh rectangle using physical names for its distributed load."""

from pathlib import Path

import gmsh as gmsh_api
import numpy as np

from fem import materials, post, steps
from fem.core import validate_model
from fem.io import gmsh as gmsh_io
from fem.solvers import static_linear


def _vertical_boundaries(surface_tag: int, x: float) -> list[int]:
    boundary_tags: list[int] = []
    for dimension, tag in gmsh_api.model.getBoundary(
        [(2, surface_tag)],
        oriented=False,
        recursive=False,
    ):
        if dimension != 1:
            continue
        x_min, _, _, x_max, _, _ = gmsh_api.model.getBoundingBox(dimension, tag)
        if abs(x_min - x) <= 1e-6 and abs(x_max - x) <= 1e-6:
            boundary_tags.append(tag)
    return boundary_tags


def main() -> None:
    owns_session = not gmsh_api.isInitialized()
    if owns_session:
        gmsh_api.initialize()

    try:
        gmsh_api.option.setNumber("General.Terminal", 0)
        gmsh_api.model.add("gmsh_boundary_load_rectangle")
        surface = gmsh_api.model.occ.addRectangle(0.0, 0.0, 0.0, 2.0, 1.0)
        gmsh_api.model.occ.synchronize()

        domain_group = gmsh_api.model.addPhysicalGroup(2, [surface])
        gmsh_api.model.setPhysicalName(2, domain_group, "DOMAIN")
        fixed_group = gmsh_api.model.addPhysicalGroup(
            1,
            _vertical_boundaries(surface, 0.0),
        )
        gmsh_api.model.setPhysicalName(1, fixed_group, "FIXED")
        traction_group = gmsh_api.model.addPhysicalGroup(
            1,
            _vertical_boundaries(surface, 2.0),
        )
        gmsh_api.model.setPhysicalName(1, traction_group, "TRACTION")

        gmsh_api.model.mesh.setSize(gmsh_api.model.getEntities(0), 0.35)
        gmsh_api.option.setNumber("Mesh.ElementOrder", 1)
        gmsh_api.option.setNumber("Mesh.RecombineAll", 0)
        gmsh_api.model.mesh.generate(2)

        imported = gmsh_io.from_model(dimension=2)
        imported_edge = imported.edges["TRACTION"]
        print(
            f"TRACTION contains {len(imported_edge.edges)} imported FEM edge(s): "
            f"{imported_edge.edges[:3]}"
        )

        model = imported.to_fem_model("gmsh_boundary_load_rectangle")
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
