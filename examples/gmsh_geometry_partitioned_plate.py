"""Build, partition, solve, and export a typed conformal plate model."""

from pathlib import Path

import numpy as np

from fem import materials, post, steps
from fem.core import FEMModel, validate_model
from fem import geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import edges, elements, nodes
from fem.solvers import static_linear


MODEL_NAME = "gmsh_geometry_partitioned_plate"


def main() -> None:
    """Run a copied-partition geometry through the explicit FEM workflow."""
    with geometry.model(MODEL_NAME, dimension=2) as cad:
        plate = cad.rectangle(0.0, 0.0, 2.0, 1.0)

        lower_partition_point = cad.point(0.75, 0.0)
        upper_partition_point = cad.point(0.75, 1.0)
        left_partition = cad.line(lower_partition_point, upper_partition_point)
        right_partition = cad.copy([left_partition])[0]
        cad.translate([right_partition], 0.5, 0.0, 0.0)

        partitioned = cad.fragment(
            [plate],
            [left_partition, right_partition],
        )
        domains = partitioned.of_dimension(2)
        imprints = partitioned.of_dimension(1)
        if len(domains) != 3:
            raise RuntimeError(
                f"expected three partitioned plate surfaces, got {len(domains)}"
            )
        if len(imprints) < 2:
            raise RuntimeError("expected both copied partition curves in the result")

        exterior = cad.boundary(domains)
        if len(cad.select(exterior, x=0.0)) != 1:
            raise RuntimeError("expected one fixed exterior boundary")
        if len(cad.select(exterior, x=2.0)) != 1:
            raise RuntimeError("expected one traction exterior boundary")

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
    domain_elements = elements.set_all(mesh, "DOMAIN")
    fixed_nodes = nodes.set_by_x(mesh, "FIXED", 0.0)
    traction_nodes = nodes.set_by_x(mesh, "TRACTION", 2.0)
    left_partition_nodes = nodes.set_by_x(mesh, "PARTITION_LEFT", 0.75)
    right_partition_nodes = nodes.set_by_x(mesh, "PARTITION_RIGHT", 1.25)
    traction_edges = edges.edge_by_x(mesh, "TRACTION", 2.0)
    coordinate_tolerance = 1.0e-8
    region_node_ids = (
        nodes.in_box(mesh, xmax=0.75 + coordinate_tolerance),
        nodes.in_box(
            mesh,
            xmin=0.75 - coordinate_tolerance,
            xmax=1.25 + coordinate_tolerance,
        ),
        nodes.in_box(mesh, xmin=1.25 - coordinate_tolerance),
    )
    region_elements = tuple(
        elements.set_by_nodes(mesh, name, node_ids, mode="all")
        for name, node_ids in zip(
            ("REGION_LEFT", "REGION_CENTER", "REGION_RIGHT"),
            region_node_ids,
            strict=True,
        )
    )
    region_id_sets = tuple(
        set(element_set.element_ids) for element_set in region_elements
    )
    if any(not element_ids for element_ids in region_id_sets):
        raise RuntimeError("expected every imprinted plate region to contain elements")
    if (
        set().union(*region_id_sets) != set(domain_elements.element_ids)
        or sum(len(element_ids) for element_ids in region_id_sets)
        != len(domain_elements.element_ids)
    ):
        raise RuntimeError("imprinted plate regions must partition the FEM domain")

    model.element_sets[domain_elements.name] = domain_elements
    for element_set in region_elements:
        model.element_sets[element_set.name] = element_set
    for node_set in (
        fixed_nodes,
        traction_nodes,
        left_partition_nodes,
        right_partition_nodes,
    ):
        model.node_sets[node_set.name] = node_set
    model.edges[traction_edges.name] = traction_edges

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
        raise RuntimeError("the partitioned-plate displacement solution is not finite")
    if not np.all(np.isfinite(result.reactions)):
        raise RuntimeError("the partitioned-plate reaction solution is not finite")

    output_dir = Path("results") / MODEL_NAME
    post.vtk.export.from_result(
        result,
        output_dir=output_dir,
        name=MODEL_NAME,
    )
    print(
        f"Solved {model.mesh.num_elements} conformal partitioned Tri3 elements "
        f"and wrote {output_dir / f'{MODEL_NAME}.vtk'}"
    )


if __name__ == "__main__":
    main()
