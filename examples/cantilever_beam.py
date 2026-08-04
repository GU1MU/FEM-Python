"""Abaqus Hex20 cantilever under surface traction and gravity."""

from pathlib import Path

from fem import post
from fem.io import inp
from fem.selection import nodes
from fem.solvers import static_linear


MODEL_NAME = "cantilever_beam_3d"
DATA_DIR = Path(__file__).resolve().parent / "examples_data"


# Import mesh, material, constraints, surface traction, and gravity
model = inp.read(DATA_DIR / f"{MODEL_NAME}.inp")
result = static_linear.solve(model)


# Inspect and export the imported analysis
tip_node_id = nodes.nearest(model.mesh, 5.0, 5.0, 0.0)
fixed_node_ids = model.node_sets["Set-Fixed"].node_ids
tip_uy = result.nodal_displacement(tip_node_id, component=2)
reaction_y = sum(
    result.nodal_reaction(node_id, component=2)
    for node_id in fixed_node_ids
)

output_dir = Path("results") / MODEL_NAME
post.vtk.export.from_result(result, output_dir=output_dir)
print(
    f"mesh={model.mesh.num_elements} Hex20, "
    f"tip uy={tip_uy:.6f} mm, "
    f"support Ry={reaction_y:.6f} N, "
    f"VTK={output_dir / f'{MODEL_NAME}.vtk'}"
)
