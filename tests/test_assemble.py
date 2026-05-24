import importlib
import sys
import unittest

import numpy as np

from fem.assemble import assemble_global_stiffness, assemble_global_stiffness_sparse
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
    make_hex8_stiffness_mesh,
    make_quad4_stiffness_mesh,
    make_quad8_stiffness_mesh,
    make_tet4_stiffness_mesh,
    make_tet10_stiffness_mesh,
    make_tri3_stiffness_mesh,
    make_truss_stiffness_mesh,
)


class LineAssemblyTests(unittest.TestCase):
    def test_sparse_assembly_accepts_mesh_for_truss_and_beam(self):
        for mesh in (make_truss_stiffness_mesh(), make_beam_stiffness_mesh()):
            with self.subTest(mesh_type=type(mesh).__name__):
                K = assemble_global_stiffness_sparse(mesh)

                self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
                self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_and_sparse_assembly_accept_mesh(self):
        for mesh in (make_truss_stiffness_mesh(), make_beam_stiffness_mesh()):
            with self.subTest(mesh_type=type(mesh).__name__):
                K_dense = assemble_global_stiffness(mesh)
                K_sparse = assemble_global_stiffness_sparse(mesh)

                self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))

    def test_assembly_requires_mesh_only(self):
        mesh = make_truss_stiffness_mesh()

        with self.assertRaises(TypeError):
            assemble_global_stiffness()
        with self.assertRaises(TypeError):
            assemble_global_stiffness_sparse()
        with self.assertRaises(TypeError):
            assemble_global_stiffness(mesh, elements=mesh.elements)
        with self.assertRaises(TypeError):
            assemble_global_stiffness_sparse(mesh.num_dofs, num_elements=1)
        with self.assertRaises(TypeError):
            assemble_global_stiffness(num_dofs=mesh.num_dofs)
        with self.assertRaises(TypeError):
            assemble_global_stiffness_sparse(
                mesh,
                get_element_dofs=lambda eid: mesh.element_dofs(mesh.elements[eid]),
            )

    def test_assemble_package_exposes_stiffness_module(self):
        import fem.assemble as assemble
        from fem.assemble import stiffness

        self.assertTrue(hasattr(assemble, "__path__"))
        self.assertIs(
            assemble.assemble_global_stiffness_sparse,
            stiffness.assemble_global_stiffness_sparse,
        )
        self.assertIs(assemble.assemble_global_stiffness, stiffness.assemble_global_stiffness)


class PlaneAssemblyTests(unittest.TestCase):
    def test_sparse_assembly_accepts_mesh_for_quad4(self):
        mesh = make_quad4_stiffness_mesh()

        K = assemble_global_stiffness_sparse(mesh)

        self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
        self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_assembly_matches_sparse_for_quad4(self):
        mesh = make_quad4_stiffness_mesh()

        K_dense = assemble_global_stiffness(mesh)
        K_sparse = assemble_global_stiffness_sparse(mesh)

        self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))

    def test_sparse_assembly_accepts_mesh_for_tri3_and_quad8(self):
        for mesh in (make_tri3_stiffness_mesh(), make_quad8_stiffness_mesh()):
            with self.subTest(mesh_type=type(mesh).__name__):
                K = assemble_global_stiffness_sparse(mesh)

                self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
                self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_assembly_matches_sparse_for_tri3_and_quad8(self):
        for mesh in (make_tri3_stiffness_mesh(), make_quad8_stiffness_mesh()):
            with self.subTest(mesh_type=type(mesh).__name__):
                K_dense = assemble_global_stiffness(mesh)
                K_sparse = assemble_global_stiffness_sparse(mesh)

                self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))


class SolidAssemblyTests(unittest.TestCase):
    def test_sparse_assembly_accepts_mesh_for_hex8(self):
        mesh = make_hex8_stiffness_mesh()

        K = assemble_global_stiffness_sparse(mesh)

        self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
        self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_assembly_matches_sparse_for_hex8(self):
        mesh = make_hex8_stiffness_mesh()

        K_dense = assemble_global_stiffness(mesh)
        K_sparse = assemble_global_stiffness_sparse(mesh)

        self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))

    def test_sparse_assembly_accepts_mesh_for_tet4_and_tet10(self):
        for mesh in (make_tet4_stiffness_mesh(), make_tet10_stiffness_mesh()):
            with self.subTest(mesh_type=type(mesh).__name__):
                K = assemble_global_stiffness_sparse(mesh)

                self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
                self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_assembly_matches_sparse_for_tet4_and_tet10(self):
        for mesh in (make_tet4_stiffness_mesh(), make_tet10_stiffness_mesh()):
            with self.subTest(mesh_type=type(mesh).__name__):
                K_dense = assemble_global_stiffness(mesh)
                K_sparse = assemble_global_stiffness_sparse(mesh)

                self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))


class StiffnessModuleRemovalTests(unittest.TestCase):
    def test_stiffness_module_is_removed_in_favor_of_element_kernels(self):
        sys.modules.pop("fem.stiffness", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.stiffness")


if __name__ == "__main__":
    unittest.main()
