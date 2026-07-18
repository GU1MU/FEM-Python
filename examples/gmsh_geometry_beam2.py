"""Build, solve, verify, and export a Beam2 cantilever with typed Gmsh geometry."""

from pathlib import Path

import numpy as np

from fem import materials, post, steps
from fem.core import FEMModel, validate_model
from fem.elements.beam_section import parse_beam2_section
from fem.geometry import gmsh as geometry
from fem.io import gmsh as gmsh_io
from fem.selection import elements, nodes
from fem.solvers import static_linear


MODEL_NAME = "gmsh_geometry_beam2"
LENGTH = 2.0
ELASTIC_MODULUS = 210.0e9
TIP_FORCE = 1.0e3
LINE_LOAD = 5.0e2
HEIGHT = 0.2
WIDTH = 0.1


def _fixed_step(name: str):
    step = steps.static(name)
    steps.displacement(step, "FIXED", components=(1, 2, 3, 4, 5, 6))
    return step


def _require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1.0e-8, atol=1.0e-12):
        raise RuntimeError(f"{label}: got {actual}, expected {expected}")


def main() -> None:
    """Run the headless geometry-to-Beam2-to-VTK workflow."""
    with geometry.model(MODEL_NAME, dimension=1) as cad:
        root = cad.point(0.0, 0.0, 0.0)
        tip = cad.point(LENGTH, 0.0, 0.0)
        cad.line(root, tip)
        native_mesh = cad.generate_mesh(size=0.5)
        mesh = gmsh_io.read(
            native_mesh,
            line_element_type="Beam2",
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
    materials.assign(
        model,
        steel,
        "MEMBERS",
        section_type="rectangle",
        height=HEIGHT,
        width=WIDTH,
    )

    tip_y_step = _fixed_step("tip_y")
    steps.nodal_load(tip_y_step, "TIP", component=2, value=TIP_FORCE)
    steps.add(model, tip_y_step)

    tip_z_step = _fixed_step("tip_z")
    steps.nodal_load(tip_z_step, "TIP", component=3, value=TIP_FORCE)
    steps.add(model, tip_z_step)

    distributed_step = _fixed_step("distributed_y")
    steps.line_load(
        distributed_step,
        "MEMBERS",
        vector=(0.0, LINE_LOAD, 0.0),
        coordinate_system="global",
    )
    steps.add(model, distributed_step)
    validate_model(model)

    tip_y_result = static_linear.solve(model, tip_y_step)
    section = parse_beam2_section(model.mesh.elements[0].props)
    tip_z_result = static_linear.solve(model, tip_z_step)
    distributed_result = static_linear.solve(
        model,
        distributed_step,
        name=MODEL_NAME,
    )

    tip_node_id = model.node_sets["TIP"].node_ids[0]
    tip_y = tip_y_result.U[model.mesh.global_dof(tip_node_id, 1)]
    tip_z = tip_z_result.U[model.mesh.global_dof(tip_node_id, 2)]
    distributed_y = distributed_result.U[model.mesh.global_dof(tip_node_id, 1)]
    expected_tip_y = TIP_FORCE * LENGTH**3 / (
        3.0 * ELASTIC_MODULUS * section.Izz
    )
    expected_tip_z = TIP_FORCE * LENGTH**3 / (
        3.0 * ELASTIC_MODULUS * section.Iyy
    )
    expected_distributed_y = LINE_LOAD * LENGTH**4 / (
        8.0 * ELASTIC_MODULUS * section.Izz
    )
    _require_close(tip_y, expected_tip_y, "Beam2 global-Y tip response using Izz")
    _require_close(tip_z, expected_tip_z, "Beam2 global-Z tip response using Iyy")
    _require_close(
        distributed_y,
        expected_distributed_y,
        "Beam2 distributed global-Y response using Izz",
    )

    envelope = post.stress.beam.nodal_envelope(distributed_result)
    if not envelope or max(row.absolute_maximum for row in envelope) <= 0.0:
        raise RuntimeError("Beam2 distributed load produced no axial-stress envelope")

    output_dir = Path("results") / MODEL_NAME
    post.vtk.export.from_result(
        distributed_result,
        output_dir=output_dir,
        name=MODEL_NAME,
    )
    print(
        f"Verified uy={tip_y:.6g}, uz={tip_z:.6g}, q-tip={distributed_y:.6g}; "
        f"wrote {output_dir / f'{MODEL_NAME}.vtk'}"
    )


if __name__ == "__main__":
    main()
