"""Generate and import a caller-owned quadratic Gmsh rectangle."""

import gmsh as gmsh_api

from fem.io import gmsh as gmsh_io
from fem.selection import elements, nodes


def main() -> None:
    owns_session = not gmsh_api.isInitialized()
    if owns_session:
        gmsh_api.initialize()

    try:
        gmsh_api.option.setNumber("General.Terminal", 0)
        gmsh_api.model.add("quad8_rectangle")
        surface = gmsh_api.model.occ.addRectangle(0.0, 0.0, 0.0, 2.0, 1.0)
        gmsh_api.model.occ.synchronize()

        boundaries = [
            tag
            for dimension, tag in gmsh_api.model.getBoundary(
                [(2, surface)],
                oriented=False,
                recursive=False,
            )
            if dimension == 1
        ]
        for boundary in boundaries:
            gmsh_api.model.mesh.setTransfiniteCurve(boundary, 5)
        gmsh_api.model.mesh.setTransfiniteSurface(surface)
        gmsh_api.model.mesh.setRecombine(2, surface)

        gmsh_api.option.setNumber("Mesh.RecombineAll", 1)
        gmsh_api.option.setNumber("Mesh.ElementOrder", 2)
        gmsh_api.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        gmsh_api.model.mesh.generate(2)

        mesh = gmsh_io.from_model(dimension=2)
        element_types = {element.type for element in mesh.elements}
        if element_types != {"Quad8"}:
            raise RuntimeError(
                f"expected only Quad8 elements, imported {sorted(element_types)!r}"
            )

        domain = elements.set_all(mesh, "DOMAIN")
        left = nodes.set_by_x(mesh, "LEFT", 0.0)
        right = nodes.set_by_x(mesh, "RIGHT", 2.0)
        print(
            "Imported Gmsh Quad8 rectangle: "
            f"{mesh.num_nodes} nodes, {mesh.num_elements} elements, "
            f"node sets={sorted((left.name, right.name))}, "
            f"element sets={[domain.name]}"
        )
    finally:
        if owns_session:
            gmsh_api.finalize()


if __name__ == "__main__":
    main()
