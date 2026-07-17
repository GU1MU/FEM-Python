"""Generate and import a Gmsh rectangle without transferring session ownership."""

import gmsh as gmsh_api

from fem.io import gmsh as gmsh_io


def _vertical_boundaries(surface_tag: int, x: float) -> list[int]:
    boundary_tags: list[int] = []
    for dimension, tag in gmsh_api.model.getBoundary(
        [(2, surface_tag)],
        oriented=False,
        recursive=False,
    ):
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
        gmsh_api.model.add("rectangle")
        surface = gmsh_api.model.occ.addRectangle(0.0, 0.0, 0.0, 2.0, 1.0)
        gmsh_api.model.occ.synchronize()

        domain_group = gmsh_api.model.addPhysicalGroup(2, [surface])
        gmsh_api.model.setPhysicalName(2, domain_group, "DOMAIN")
        left_group = gmsh_api.model.addPhysicalGroup(
            1,
            _vertical_boundaries(surface, 0.0),
        )
        gmsh_api.model.setPhysicalName(1, left_group, "LEFT")
        right_group = gmsh_api.model.addPhysicalGroup(
            1,
            _vertical_boundaries(surface, 2.0),
        )
        gmsh_api.model.setPhysicalName(1, right_group, "RIGHT")

        gmsh_api.model.mesh.setSize(gmsh_api.model.getEntities(0), 0.35)
        gmsh_api.option.setNumber("Mesh.ElementOrder", 1)
        gmsh_api.option.setNumber("Mesh.RecombineAll", 0)
        gmsh_api.model.mesh.generate(2)

        imported = gmsh_io.from_model(dimension=2)
        model = imported.to_fem_model("gmsh_rectangle")
        print(
            "Imported Gmsh rectangle: "
            f"{model.mesh.num_nodes} nodes, {model.mesh.num_elements} elements, "
            f"node sets={sorted(model.node_sets)}, "
            f"element sets={sorted(model.element_sets)}"
        )
    finally:
        if owns_session:
            gmsh_api.finalize()


if __name__ == "__main__":
    main()
