import os
import runpy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fem import materials, steps
from fem.core.model import ElementSet, FEMModel, MaterialDefinition, NodeSet
from fem.solvers import static_linear
from tests.helpers.mesh_builders import make_mixed_hex8_tet4_mesh
from tests.helpers.model_builders import make_truss_workflow_model


class ModelWorkflowIntegrationTests(unittest.TestCase):
    def test_core_model_supports_hand_written_mesh_model_solve_result_flow(self):
        model = make_truss_workflow_model(name="manual_bar", loaded_set_name="tip")
        mesh = model.mesh

        material = materials.linear_elastic.material("steel", E=100.0, nu=0.3)
        materials.add(model, material)
        section = materials.assign(model, "steel", "bar", area=2.0)
        step = steps.static("pull")
        steps.displacement(step, "fixed", components=(1, 2))
        steps.displacement(step, 2, components=2)
        steps.nodal_load(step, "tip", component=1, value=100.0)
        steps.add(model, step)

        result = static_linear.solve(model, "pull")

        self.assertEqual(material, MaterialDefinition("steel", {"E": 100.0, "nu": 0.3}))
        self.assertEqual(model.element_sets["bar"], ElementSet("bar", (1,)))
        self.assertEqual(section.element_set, "bar")
        self.assertEqual(section.properties["area"], 2.0)
        self.assertEqual(model.node_sets["fixed"], NodeSet("fixed", (1,)))
        self.assertEqual(model.node_sets["tip"], NodeSet("tip", (2,)))
        self.assertEqual(step.name, "pull")
        self.assertEqual(len(step.boundaries), 2)
        self.assertEqual(len(step.cloads), 1)
        self.assertEqual(mesh.elements[0].props["E"], 100.0)
        self.assertEqual(mesh.elements[0].props["area"], 2.0)
        self.assertAlmostEqual(result.U[mesh.global_dof(2, 0)], 0.5)
        self.assertAlmostEqual(result.reactions[mesh.global_dof(1, 0)], -100.0)

    def test_manual_workflow_uses_materials_steps_and_static_solver(self):
        model = make_truss_workflow_model(
            name="manual_workflow",
            loaded_set_name="loaded",
            element_props={"area": 2.0},
        )
        mesh = model.mesh

        steel = materials.linear_elastic.material("steel", E=100.0, nu=0.3)
        materials.add(model, steel)
        materials.assign(model, material="steel", element_set="bar")

        step = steps.static("pull")
        steps.displacement(step, target="fixed", components=(1, 2))
        steps.displacement(step, target="loaded", components=2)
        steps.nodal_load(step, target="loaded", component=1, value=10.0)
        steps.add(model, step)

        result = static_linear.solve(model, step="pull")

        self.assertAlmostEqual(result.U[mesh.global_dof(2, 0)], 0.05)
        self.assertAlmostEqual(result.U[mesh.global_dof(2, 1)], 0.0)


class MixedElementWorkflowIntegrationTests(unittest.TestCase):
    def test_mixed_solid_model_assigns_materials_by_element_set_and_solves(self):
        mesh = make_mixed_hex8_tet4_mesh()
        model = FEMModel(mesh=mesh, name="mixed_hex8_tet4")
        model.element_sets["hexes"] = ElementSet("hexes", (1,))
        model.element_sets["tets"] = ElementSet("tets", (2,))
        model.node_sets["fixed"] = NodeSet("fixed", (1, 4, 5, 8))
        model.node_sets["tip"] = NodeSet("tip", (9,))

        steel = materials.linear_elastic.material("steel", E=210.0, nu=0.3)
        aluminum = materials.linear_elastic.material("aluminum", E=120.0, nu=0.25)
        materials.add(model, steel)
        materials.add(model, aluminum)
        materials.assign(model, "steel", "hexes")
        materials.assign(model, "aluminum", "tets")

        step = steps.static("pull")
        steps.displacement(step, "fixed", components=(1, 2, 3))
        steps.nodal_load(step, "tip", component=1, value=1.0)
        steps.add(model, step)

        result = static_linear.solve(model, "pull")

        self.assertEqual(mesh.elements[0].type, "Hex8")
        self.assertEqual(mesh.elements[1].type, "Tet4")
        self.assertEqual(mesh.elements[0].props["material"], "steel")
        self.assertEqual(mesh.elements[1].props["material"], "aluminum")
        self.assertTrue(np.all(np.isfinite(result.U)))
        self.assertGreater(abs(float(result.U[mesh.global_dof(9, 0)])), 0.0)


class MixedElementExampleTests(unittest.TestCase):
    def test_mixed_hex8_tet4_example_import_runs(self):
        old_output_dir = os.environ.get("FEM_MIXED_EXAMPLE_OUTPUT_DIR")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FEM_MIXED_EXAMPLE_OUTPUT_DIR"] = tmp
            try:
                namespace = runpy.run_path("examples/mixed_hex8_tet4.py")
                self.assertTrue((Path(tmp) / "mixed_hex8_tet4.vtk").exists())
            finally:
                if old_output_dir is None:
                    os.environ.pop("FEM_MIXED_EXAMPLE_OUTPUT_DIR", None)
                else:
                    os.environ["FEM_MIXED_EXAMPLE_OUTPUT_DIR"] = old_output_dir

        self.assertIn("result", namespace)
        result = namespace["result"]
        self.assertEqual([elem.type for elem in result.model.mesh.elements], ["Hex8", "Tet4"])


if __name__ == "__main__":
    unittest.main()
