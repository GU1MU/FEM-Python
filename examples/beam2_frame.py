# Example: mixed-section spatial Beam2 frame under a distributed line load.

from pathlib import Path

from fem import post, solvers, steps
from fem.core import ElementSet, FEMModel, NodeSet
from fem.io import csv as mesh_csv


DATA_DIR = Path(__file__).resolve().parent / "examples_data"
MATERIALS_PATH = DATA_DIR / "examples_materials.csv"


# The CSV assigns a hollow circular section to the mast, a rectangular
# section to the right arm, and a solid circular section to the left arm.
mesh = mesh_csv.read_beam2(DATA_DIR / "beam2.csv", MATERIALS_PATH)
model = FEMModel(mesh=mesh, name="beam2_frame")

model.node_sets["fixed"] = NodeSet("fixed", (1,))
model.node_sets["tips"] = NodeSet("tips", (8, 13))
model.element_sets["arms"] = ElementSet("arms", tuple(range(3, 13)))

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
