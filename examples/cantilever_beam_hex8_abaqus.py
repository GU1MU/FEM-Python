# Example: cantilever beam solved with Hex8 solid elements.

from pathlib import Path

from fem import abaqus, post, solvers

# Read Abaqus model data and solve its first analysis step.
model = abaqus.read(r"examples\cantilever_beam_hex8.inp")
model.name = "cantilever_beam_hex8_abaqus"
step = solvers.static_linear.get_step(model)

print("Model:", model.name)
print("Step:", step.name if step is not None else "None")
print("Node sets:", sorted(model.node_sets))
print("Element sets:", sorted(model.element_sets))

result = solvers.static_linear.solve(model, step)

# Export nodal displacements, stresses, and VTK for visualization.
post.vtk.export.from_result(result, output_dir=Path("results") / model.name)
