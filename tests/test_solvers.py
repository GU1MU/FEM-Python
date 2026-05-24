import unittest
import importlib.util

import numpy as np
from scipy.sparse import csr_matrix

from fem import solvers
from fem.core.result import ModelResult, ModelResults
from fem.solvers import static_linear
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
)


class StaticLinearSolverTests(unittest.TestCase):
    def test_static_linear_solver_builds_step_boundary_and_solves_case(self):
        model = make_static_pull_truss_model()
        mesh = model.mesh

        bc = static_linear.boundary_for_step(model, "pull")
        self.assertEqual(len(bc.prescribed_displacements), 3)
        self.assertAlmostEqual(sum(bc.nodal_forces.values()), 100.0)

        U = static_linear.solve(model, "pull").U

        self.assertAlmostEqual(U[mesh.global_dof(2, 0)], 0.5)
        self.assertAlmostEqual(U[mesh.global_dof(2, 1)], 0.0)

    def test_static_linear_solver_returns_result_with_displacements_and_reactions(self):
        model = make_static_pull_truss_model()
        mesh = model.mesh

        result = static_linear.solve(
            model,
            "pull",
            name="pull_case",
        )

        self.assertIsInstance(result, ModelResult)
        self.assertEqual(result.step.name, "pull")
        self.assertFalse(hasattr(result, "output_dir"))
        self.assertFalse(hasattr(result, "boundary"))
        self.assertEqual(result.name, "pull_case")
        self.assertAlmostEqual(result.U[mesh.global_dof(2, 0)], 0.5)
        self.assertAlmostEqual(result.U[mesh.global_dof(2, 1)], 0.0)
        self.assertAlmostEqual(result.reactions[mesh.global_dof(1, 0)], -100.0)
        self.assertAlmostEqual(result.reactions[mesh.global_dof(2, 0)], 0.0)

    def test_static_linear_solver_solve_all_returns_step_result_collection(self):
        model = make_two_step_static_pull_truss_model()
        mesh = model.mesh

        results = static_linear.solve_all(model)

        self.assertIsInstance(results, ModelResults)
        self.assertEqual(len(results.results), 2)
        self.assertEqual(tuple(result.step.name for result in results.results), ("pull1", "pull2"))
        pull1, pull2 = results.results
        self.assertAlmostEqual(pull1.U[mesh.global_dof(2, 0)], 0.5)
        self.assertAlmostEqual(pull2.U[mesh.global_dof(2, 0)], 1.0)
        self.assertEqual(pull1.name, "bar_pull1")
        self.assertEqual(pull2.name, "bar_pull2")


class LinearSolverTests(unittest.TestCase):
    def test_solver_package_exposes_linear_solver_only(self):
        self.assertTrue(hasattr(solvers, "linear"))
        self.assertTrue(hasattr(solvers.linear, "solve"))
        self.assertIsNone(importlib.util.find_spec("fem.solve"))

    def test_linear_solver_solves_sparse_system_and_rejects_dense_matrix(self):
        K = csr_matrix([[2.0, 0.0], [0.0, 4.0]])
        F = np.array([6.0, 8.0])

        U = solvers.linear.solve(K, F)

        self.assertTrue(np.allclose(U, [3.0, 2.0]))
        with self.assertRaises(TypeError):
            solvers.linear.solve(np.eye(2), F)

    def test_linear_solver_rejects_singular_sparse_matrix(self):
        K = csr_matrix([[1.0, 0.0], [0.0, 0.0]])
        F = np.array([1.0, 1.0])

        with self.assertRaises(RuntimeError):
            solvers.linear.solve(K, F)


if __name__ == "__main__":
    unittest.main()
