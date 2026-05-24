import unittest

from fem import materials, steps
from fem.core.model import ElementSet, MaterialDefinition, NodeSet
from fem.solvers import static_linear
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


if __name__ == "__main__":
    unittest.main()
