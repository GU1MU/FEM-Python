"""Build, solve, verify, and export a spatial Truss2 bar with typed Gmsh geometry."""

from pathlib import Path

import numpy as np

from fem import materials, post, steps
from fem.core import FEMModel, validate_model
from fem.elements import get_element_kernel
from fem.geometry import gmsh as geometry
from fem.io import gmsh as gmsh_io
from fem.selection import elements, nodes
from fem.solvers import static_linear


MODEL_NAME = "gmsh_geometry_truss2"
LENGTH = 2.0
ELASTIC_MODULUS = 210.0e9
AREA = 1.0e-4
AXIAL_FORCE = 1.0e4


def main() -> None:
    """Run the headless geometry-to-Truss2-to-VTK workflow."""
    with geometry.model(MODEL_NAME, dimension=1) as cad:
        start = cad.point(0.0, 0.5, -0.25)
        end = cad.point(LENGTH, 0.5, -0.25)
        cad.line(start, end)
        native_mesh = cad.generate_mesh(size=0.5)
        mesh = gmsh_io.read(
            native_mesh,
            line_element_type="Truss2",
        )

    model = FEMModel(mesh=mesh, name=MODEL_NAME)
    members = elements.set_all(mesh, "MEMBERS")
    fixed = nodes.set_by_x(mesh, "FIXED", 0.0)
    tip_nodes = nodes.set_by_x(mesh, "TIP", LENGTH)
    model.element_sets[members.name] = members
    model.node_sets[fixed.name] = fixed
    model.node_sets[tip_nodes.name] = tip_nodes

    steel = materials.linear_elastic.material(
        "steel",
        E=ELASTIC_MODULUS,
        nu=0.3,
    )
    materials.add(model, steel)
    materials.assign(model, steel, "MEMBERS", area=AREA)

    load_step = steps.static("axial_pull")
    steps.displacement(load_step, "FIXED", components=(1, 2, 3))
    fixed_node_id = model.node_sets["FIXED"].node_ids[0]
    for node_id in model.mesh.node_ids:
        if node_id != fixed_node_id:
            steps.displacement(load_step, node_id, components=(2, 3))
    steps.nodal_load(load_step, "TIP", component=1, value=AXIAL_FORCE)
    steps.add(model, load_step)
    validate_model(model)

    result = static_linear.solve(model, load_step, name=MODEL_NAME)
    tip_node_id = model.node_sets["TIP"].node_ids[0]
    tip_displacement = result.U[model.mesh.global_dof(tip_node_id, 0)]
    expected_displacement = AXIAL_FORCE * LENGTH / (ELASTIC_MODULUS * AREA)
    if not np.isclose(tip_displacement, expected_displacement, rtol=1.0e-9):
        raise RuntimeError(
            "Truss2 tip displacement does not match F*L/(E*A): "
            f"got {tip_displacement}, expected {expected_displacement}"
        )

    expected_stress = AXIAL_FORCE / AREA
    stresses = [
        get_element_kernel(element.type).element_stress(
            model.mesh,
            element,
            result.U,
        )[1]
        for element in model.mesh.elements
    ]
    if not np.allclose(stresses, expected_stress, rtol=1.0e-9):
        raise RuntimeError(
            "Truss2 axial stress does not match F/A: "
            f"got {stresses}, expected {expected_stress}"
        )

    output_dir = Path("results") / MODEL_NAME
    post.vtk.export.from_result(
        result,
        output_dir=output_dir,
        name=MODEL_NAME,
    )
    print(
        f"Verified u={tip_displacement:.6g}, sigma={expected_stress:.6g}; "
        f"wrote {output_dir / f'{MODEL_NAME}.vtk'}"
    )


if __name__ == "__main__":
    main()
