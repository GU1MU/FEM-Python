"""Generate and inspect a typed automatic full-quad mesh."""

from fem.geometry import gmsh as geometry


MODEL_NAME = "gmsh_geometry_auto_mesh"


def main() -> None:
    """Run the headless automatic-meshing workflow."""
    with geometry.model(MODEL_NAME, dimension=2) as cad:
        surface = cad.disk(0.0, 0.0, 1.0)
        boundary = cad.boundary([surface])
        if not boundary:
            raise RuntimeError("expected a disk boundary")

        cad.physical("DOMAIN", [surface])
        cad.physical("BOUNDARY", boundary)
        imported = cad.generate_auto_mesh(
            level=3,
            cell_shape="quad",
            order=2,
        )

    element_types = {element.type for element in imported.mesh.elements}
    if element_types != {"Quad8"}:
        raise RuntimeError(
            f"expected only Quad8 elements, imported {sorted(element_types)!r}"
        )
    if "DOMAIN" not in imported.element_sets:
        raise RuntimeError("expected the DOMAIN element set to be imported")
    if "BOUNDARY" not in imported.node_sets:
        raise RuntimeError("expected the BOUNDARY node set to be imported")
    if "BOUNDARY" not in imported.edges:
        raise RuntimeError("expected the BOUNDARY edge collection to be imported")

    print(
        "Generated level-3 automatic Quad8 mesh: "
        f"{imported.mesh.num_nodes} nodes, "
        f"{imported.mesh.num_elements} elements, "
        f"node sets={sorted(imported.node_sets)}, "
        f"element sets={sorted(imported.element_sets)}, "
        f"edges={sorted(imported.edges)}"
    )


if __name__ == "__main__":
    main()
