# Example: mixed Hex8 and Tet4 linear static model.

import os
from pathlib import Path

from fem import materials, post, solvers, steps
from fem.core import ElementSet, FEMModel, NodeSet, model_element_info
from fem.io import csv as mesh_csv


DATA_DIR = Path(__file__).resolve().parent / "examples_data"


mesh = mesh_csv.read_mixed3d(DATA_DIR / "mixed_hex8_tet4.csv")
model = FEMModel(mesh=mesh, name="mixed_hex8_tet4")

model.element_sets["hexes"] = ElementSet("hexes", (1,))
model.element_sets["tets"] = ElementSet("tets", (2,))
model.node_sets["fixed"] = NodeSet("fixed", (1, 4, 5, 8))
model.node_sets["tip"] = NodeSet("tip", (9,))

steel = materials.linear_elastic.material("steel", E=210000.0, nu=0.3)
aluminum = materials.linear_elastic.material("aluminum", E=70000.0, nu=0.33)
materials.add(model, steel)
materials.add(model, aluminum)
materials.assign(model, "steel", "hexes")
materials.assign(model, "aluminum", "tets")

load_step = steps.static("pull")
steps.displacement(load_step, "fixed", components=(1, 2, 3))
steps.nodal_load(load_step, "tip", component=1, value=100.0)
steps.add(model, load_step)

result = solvers.static_linear.solve(model, "pull")
element_infos = [model_element_info(model, elem.id) for elem in mesh.elements]

print("Element info:")
for info in element_infos:
    print(
        f"  elem {info.elem_id}: "
        f"type={info.element_type}, "
        f"material={info.material}, "
        f"E={info.properties.get('E')}, "
        f"nu={info.properties.get('nu')}"
    )
print("Tip ux:", float(result.U[mesh.global_dof(9, 0)]))

output_dir = os.environ.get(
    "FEM_MIXED_EXAMPLE_OUTPUT_DIR",
    str(Path("results") / "mixed_hex8_tet4"),
)
post.vtk.export.from_result(result, output_dir=output_dir, name="mixed_hex8_tet4")
