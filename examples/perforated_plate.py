"""Locally refined first-order plate with a circular hole."""

from pathlib import Path

from fem import geometry, materials, post, steps
from fem.core import FEMModel
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import edges, elements, nodes
from fem.solvers import static_linear


MODEL_NAME = "perforated_plate"
# SI units: m, Pa
WIDTH = 4.0
HEIGHT = 2.0
HOLE_X = 1.2
HOLE_Y = 1.0
HOLE_RADIUS = 0.3
THICKNESS = 0.01
EDGE_TRACTION = 1.0e6


# Geometry and locally refined mesh
with geometry.model(MODEL_NAME, dimension=2) as cad:
    plate = cad.rectangle(0.0, 0.0, WIDTH, HEIGHT)
    hole = cad.disk(HOLE_X, HOLE_Y, HOLE_RADIUS)
    domain = cad.cut([plate], [hole]).of_dimension(2)

    boundary = cad.boundary(domain)
    outer_curves = (
        cad.select(boundary, x=0.0)
        + cad.select(boundary, x=WIDTH)
        + cad.select(boundary, y=0.0)
        + cad.select(boundary, y=HEIGHT)
    )
    hole_curves = tuple(
        curve for curve in boundary if curve not in outer_curves
    )

    mesher = gmsh_meshing.Mesher(cad)
    hole_distance = mesher.distance_field(curves=hole_curves, sampling=100)
    hole_refinement = mesher.threshold_field(
        hole_distance,
        size_min=0.05,
        size_max=0.40,
        dist_min=0.10,
        dist_max=0.80,
    )
    mesher.background_field(hole_refinement)

    native_mesh = mesher.generate(gmsh_meshing.MeshSpec(order=1))
    mesh = gmsh_io.read(
        native_mesh,
        plane_type="stress",
        thickness=THICKNESS,
    )


# Model, material, and boundary load
model = FEMModel(mesh=mesh, name=MODEL_NAME)
model.element_sets["DOMAIN"] = elements.set_all(mesh, "DOMAIN")
model.node_sets["FIXED"] = nodes.set_by_x(mesh, "FIXED", 0.0)
model.node_sets["LOADED"] = nodes.set_by_x(mesh, "LOADED", WIDTH)
model.edges["TRACTION"] = edges.edge_by_x(mesh, "TRACTION", WIDTH)

steel = materials.linear_elastic.material("steel", E=210.0e9, nu=0.3)
materials.add(model, steel)
materials.assign(model, steel, "DOMAIN")

pull = steps.static("pull")
steps.displacement(pull, "FIXED", components=(1, 2))
steps.edge_traction(pull, "TRACTION", vector=(EDGE_TRACTION, 0.0))
steps.add(model, pull)


# Solve, inspect, and export
result = static_linear.solve(model)
loaded_ux = max(
    result.nodal_displacement(node_id, component=1)
    for node_id in model.node_sets["LOADED"].node_ids
)
reaction_x = sum(
    result.nodal_reaction(node_id, component=1)
    for node_id in model.node_sets["FIXED"].node_ids
)

output_dir = Path("results") / MODEL_NAME
post.vtk.export.from_result(result, output_dir=output_dir)
print(
    f"elements={mesh.num_elements}, "
    f"max loaded ux={loaded_ux * 1e3:.4f} mm, "
    f"support Rx={reaction_x / 1e3:.3f} kN, "
    f"VTK={output_dir / f'{MODEL_NAME}.vtk'}"
)
