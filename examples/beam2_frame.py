# Example: mixed-section spatial Beam2 frame under a distributed line load.

from pathlib import Path

from fem import materials, post, selection, solvers, steps
from fem.core import ElementSet, FEMModel, NodeSet
from fem.io import csv as mesh_csv


DATA_DIR = Path(__file__).resolve().parent / "examples_data"


mesh = mesh_csv.read_beam2(DATA_DIR / "beam2.csv")
model = FEMModel(mesh=mesh, name="beam2_frame")

model.node_sets["fixed"] = NodeSet("fixed", (1,))
model.node_sets["tips"] = NodeSet("tips", (8, 13))

mast_nodes = selection.nodes.set_by_x(mesh, "mast_nodes", 0.0)
right_arm_nodes = selection.nodes.set_in_box(
    mesh, "right_arm_nodes", xmin=0.0, ymin=5.0, ymax=5.0
)
left_arm_nodes = selection.nodes.set_in_box(
    mesh, "left_arm_nodes", xmax=0.0, ymin=5.0, ymax=5.0
)
for node_set in (mast_nodes, right_arm_nodes, left_arm_nodes):
    model.node_sets[node_set.name] = node_set

mast = selection.elements.set_by_nodes(
    mesh, "mast", mast_nodes.node_ids, mode="all"
)
right_arm = selection.elements.set_by_nodes(
    mesh, "right_arm", right_arm_nodes.node_ids, mode="all"
)
left_arm = selection.elements.set_by_nodes(
    mesh, "left_arm", left_arm_nodes.node_ids, mode="all"
)
for element_set in (mast, right_arm, left_arm):
    model.element_sets[element_set.name] = element_set
model.element_sets["arms"] = ElementSet(
    "arms", right_arm.element_ids + left_arm.element_ids
)

aluminum = materials.linear_elastic.material(
    "aluminum", E=70.0e3, nu=0.33, rho=2700.0
)
materials.add(model, aluminum)
materials.assign(
    model,
    aluminum,
    mast,
    section_type="hollow_circle",
    outer_radius=0.025,
    inner_radius=0.015,
)
materials.assign(
    model,
    aluminum,
    right_arm,
    section_type="rectangle",
    height=0.04,
    width=0.03,
)
materials.assign(
    model,
    aluminum,
    left_arm,
    section_type="solid_circle",
    radius=0.02,
)

# Fix all six Beam2 degrees of freedom at the mast base, then apply a
# constant line load to both arms in the negative global-y direction.
load_step = steps.static("distributed_load")
steps.displacement(load_step, "fixed", components=(1, 2, 3, 4, 5, 6))
steps.line_load(
    load_step,
    "arms",
    vector=(0.0, -1.0e-5, 0.0),
    coordinate_system="global",
)
steps.add(model, load_step)

result = solvers.static_linear.solve(model, step="distributed_load")

for node_id in model.node_sets["tips"].node_ids:
    values = tuple(float(result.U[dof]) for dof in mesh.node_dofs(node_id))
    print(f"Node {node_id} (ux, uy, uz, rx, ry, rz):", values)

stress_rows = post.stress.beam.nodal_envelope(result)
critical = max(stress_rows, key=lambda row: row.absolute_maximum)
print(
    "Critical axial stress envelope: "
    f"node={critical.node_id}, "
    f"max={critical.maximum:.6g}, "
    f"min={critical.minimum:.6g}, "
    f"abs_max={critical.absolute_maximum:.6g}"
)

output_dir = Path("results") / model.name
post.vtk.export.from_result(result, output_dir=output_dir)
print("Results:", output_dir.resolve())
