# Example: mixed Hex8 and Tet4 linear static model.

import os

from fem import materials, post, solvers, steps
from fem.core import Element3D, ElementSet, FEMModel, Node3D, NodeSet, model_element_info
from fem.core.mesh import HexMesh3D


nodes = [
    Node3D(1, 0.0, 0.0, 0.0),
    Node3D(2, 1.0, 0.0, 0.0),
    Node3D(3, 1.0, 1.0, 0.0),
    Node3D(4, 0.0, 1.0, 0.0),
    Node3D(5, 0.0, 0.0, 1.0),
    Node3D(6, 1.0, 0.0, 1.0),
    Node3D(7, 1.0, 1.0, 1.0),
    Node3D(8, 0.0, 1.0, 1.0),
    Node3D(9, 2.0, 0.0, 0.0),
]
elements = [
    Element3D(1, [1, 2, 3, 4, 5, 6, 7, 8], "Hex8"),
    Element3D(2, [2, 9, 3, 6], "Tet4"),
]
mesh = HexMesh3D(nodes=nodes, elements=elements)
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

post.vtk.export.from_result(result, output_dir=r'results', name="mixed_hex8_tet4")
