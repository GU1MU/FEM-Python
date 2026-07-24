"""Gmsh Beam2 portal frame with service-load and support-settlement cases."""

from pathlib import Path

from fem import geometry, materials, post, steps
from fem.core import FEMModel
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import elements, nodes
from fem.solvers import static_linear


MODEL_NAME = "beam2_frame"
# SI units: m, N, Pa
SPAN = 4.0
EAVE_HEIGHT = 2.0
RIDGE_HEIGHT = 3.0
MESH_LEVEL = 3
ROOF_LOAD = 12.0e3
HORIZONTAL_LOAD = 20.0e3
SETTLEMENT = -0.01
TOL = 1.0e-8


# Geometry and Beam2 mesh
with geometry.model(MODEL_NAME, dimension=1) as cad:
    left_base = cad.point(0.0, 0.0, 0.0)
    left_eave = cad.point(0.0, EAVE_HEIGHT, 0.0)
    ridge = cad.point(SPAN / 2.0, RIDGE_HEIGHT, 0.0)
    right_eave = cad.point(SPAN, EAVE_HEIGHT, 0.0)
    right_base = cad.point(SPAN, 0.0, 0.0)

    cad.line(left_base, left_eave)
    cad.line(left_eave, ridge)
    cad.line(ridge, right_eave)
    cad.line(right_eave, right_base)

    native_mesh = gmsh_meshing.Mesher(cad).generate(
        gmsh_meshing.AutoMeshSpec(level=MESH_LEVEL)
    )
    mesh = gmsh_io.read(native_mesh, line_element_type="Beam2")


# Model sets, material, and sections
model = FEMModel(mesh=mesh, name=MODEL_NAME)
left_column_node_ids = nodes.by_x(mesh, 0.0)
right_column_node_ids = nodes.by_x(mesh, SPAN)
column_node_ids = left_column_node_ids + right_column_node_ids
roof_node_ids = nodes.in_box(mesh, ymin=EAVE_HEIGHT - TOL)
model.element_sets["COLUMNS"] = elements.set_by_nodes(
    mesh, "COLUMNS", column_node_ids
)
model.element_sets["ROOF"] = elements.set_by_nodes(
    mesh, "ROOF", roof_node_ids
)
model.node_sets["SUPPORTS"] = nodes.set_by_y(mesh, "SUPPORTS", 0.0)
model.node_sets["RIGHT_BASE"] = nodes.set_by_coord(
    mesh, "RIGHT_BASE", x=SPAN, y=0.0, z=0.0
)
model.node_sets["RIGHT_EAVE"] = nodes.set_by_coord(
    mesh, "RIGHT_EAVE", x=SPAN, y=EAVE_HEIGHT, z=0.0
)
model.node_sets["RIDGE"] = nodes.set_by_coord(
    mesh, "RIDGE", x=SPAN / 2.0, y=RIDGE_HEIGHT, z=0.0
)

steel = materials.linear_elastic.material("steel", E=210.0e9, nu=0.3)
materials.add(model, steel)
materials.assign(
    model, steel, "COLUMNS", section_type="rectangle", height=0.24, width=0.16
)
materials.assign(
    model, steel, "ROOF", section_type="rectangle", height=0.20, width=0.12
)


initial = steps.static("Initial")
steps.displacement(initial, "SUPPORTS", components=(1, 2, 3, 4, 5, 6))
steps.add(model, initial)

service_load = steps.static("service_load")
steps.line_load(service_load, "ROOF", (0.0, -ROOF_LOAD, 0.0))
steps.nodal_load(service_load, "RIGHT_EAVE", component=1, value=HORIZONTAL_LOAD)
steps.add(model, service_load)

support_settlement = steps.static("support_settlement")
steps.line_load(support_settlement, "ROOF", (0.0, -ROOF_LOAD, 0.0))
steps.nodal_load(
    support_settlement, "RIGHT_EAVE", component=1, value=HORIZONTAL_LOAD
)
steps.displacement(
    support_settlement, "RIGHT_BASE", components=2, value=SETTLEMENT
)
steps.add(model, support_settlement)


# Solve, inspect, and export both load cases
results = static_linear.solve(
    model,
    steps=("service_load", "support_settlement"),
)
ridge_id = model.node_sets["RIDGE"].node_ids[0]
right_eave_id = model.node_sets["RIGHT_EAVE"].node_ids[0]
support_ids = model.node_sets["SUPPORTS"].node_ids
output_dir = Path("results") / MODEL_NAME

for result in results:
    eave_ux = result.nodal_displacement(right_eave_id, component=1)
    ridge_uy = result.nodal_displacement(ridge_id, component=2)
    reaction_x = sum(
        result.nodal_reaction(node_id, component=1) for node_id in support_ids
    )
    reaction_y = sum(
        result.nodal_reaction(node_id, component=2) for node_id in support_ids
    )
    max_stress = post.stress.beam.absolute_maximum(result)
    post.vtk.export.from_result(result, output_dir=output_dir)
    print(
        f"{result.step.name}: eave ux={eave_ux * 1e3:.3f} mm, "
        f"ridge uy={ridge_uy * 1e3:.3f} mm, "
        f"support (Rx, Ry)=({reaction_x / 1e3:.3f}, "
        f"{reaction_y / 1e3:.3f}) kN, "
        f"max |stress|={max_stress / 1e6:.3f} MPa, "
        f"VTK={output_dir / f'{result.name}.vtk'}"
    )
