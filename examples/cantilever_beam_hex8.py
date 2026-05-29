# Example: hand-written model workflow built from an Abaqus Hex8 mesh.

from pathlib import Path

from fem import materials, post, selection, solvers, steps
from fem.core import ElementSet, FEMModel, NodeSet
from fem.io import inp


DATA_DIR = Path(__file__).resolve().parent / "examples_data"

# Read only mesh topology and coordinates from Abaqus input.
mesh = inp.read_hex8(DATA_DIR / "cantilever_beam_hex8.inp")

model = FEMModel(mesh=mesh, name="cantilever_beam_hex8")

# Define element sets, material, and section data by hand.
model.element_sets["solid"] = ElementSet("solid", selection.elements.all(mesh))
steel = materials.linear_elastic.material("steel", E=220000.0, nu=0.3, rho=7800.0)
materials.add(model, steel)
materials.assign(model, material="steel", element_set="solid")

# Define node sets from geometric selections.
xs = [node.x for node in mesh.nodes]
zs = [node.z for node in mesh.nodes]
x_min, x_max = min(xs), max(xs)
z_max = max(zs)

fixed = NodeSet("fixed", selection.nodes.by_x(mesh, x_min))
loaded = NodeSet("loaded", selection.nodes.by_coord(mesh, x=x_max, z=z_max))
model.node_sets[fixed.name] = fixed
model.node_sets[loaded.name] = loaded

print("Fixed nodes:", fixed.node_ids)
print("Loaded nodes:", loaded.node_ids)

# Define a static loading step by hand.
load_step = steps.static("load")
steps.displacement(load_step, target="fixed", components=(1, 2, 3))
steps.nodal_load(load_step, target="loaded", component=3, value=-50.0)
steps.add(model, load_step)

result = solvers.static_linear.solve(model, step="load")

print("Step:", result.step.name)
first_loaded_displacement = tuple(
    float(result.U[dof]) for dof in mesh.node_dofs(loaded.node_ids[0])
)
print("First loaded node displacement:", first_loaded_displacement)

post.vtk.export.from_result(result, output_dir=Path("results") / model.name)
