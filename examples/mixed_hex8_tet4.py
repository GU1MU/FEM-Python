# Example: mixed Hex8 and Tet4 linear static model.

import csv
import os
from pathlib import Path

from fem import materials, post, solvers, steps
from fem.core import Element3D, ElementSet, FEMModel, Node3D, NodeSet, model_element_info
from fem.core.mesh import HexMesh3D


DATA_DIR = Path(__file__).resolve().parent / "geometry_data"


def read_mixed_mesh(nodes_path, elements_path):
    nodes = []
    with Path(nodes_path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            nodes.append(
                Node3D(
                    int(row["node_id"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                )
            )

    elements = []
    with Path(elements_path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            node_ids = [
                int(row[f"node{i}"])
                for i in range(1, 9)
                if row.get(f"node{i}", "").strip()
            ]
            elements.append(Element3D(int(row["elem_id"]), node_ids, row["type"]))

    return HexMesh3D(nodes=nodes, elements=elements)


mesh = read_mixed_mesh(
    DATA_DIR / "mixed_hex8_tet4_nodes.csv",
    DATA_DIR / "mixed_hex8_tet4_elements.csv",
)
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
