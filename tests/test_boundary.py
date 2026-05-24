import unittest

import numpy as np

from fem import boundary
from fem.elements import get_element_kernel
from tests.helpers.mesh_builders import make_quad4_boundary_mesh, make_tet4_stiffness_mesh


class BoundaryKernelTests(unittest.TestCase):
    def test_boundary_load_vector_matches_kernel_dispatch(self):
        mesh = make_quad4_boundary_mesh()
        elem = mesh.elements[0]
        bc = boundary.condition.BoundaryCondition()
        bc.add_body_force_element(elem.id, 4.0, -5.0)
        bc.add_surface_traction(elem.id, 0, 7.0, -11.0)

        F = boundary.loads.build_load_vector(mesh, bc)
        kernel = get_element_kernel(elem.type)
        expected = kernel.body_force(mesh, elem, (4.0, -5.0))
        expected += kernel.edge_traction(mesh, elem, 0, (7.0, -11.0))

        self.assertTrue(np.allclose(F, expected))

        mesh3d = make_tet4_stiffness_mesh()
        elem3d = mesh3d.elements[0]
        bc3d = boundary.condition.BoundaryCondition()
        bc3d.add_body_force_element(elem3d.id, 0.0, 0.0, -6.0)
        bc3d.add_surface_traction(elem3d.id, 3, 0.0, 0.0, -2.0)

        F3d = boundary.loads.build_load_vector(mesh3d, bc3d)
        kernel3d = get_element_kernel(elem3d.type)
        expected3d = kernel3d.body_force(mesh3d, elem3d, (0.0, 0.0, -6.0))
        expected3d += kernel3d.face_traction(mesh3d, elem3d, 3, (0.0, 0.0, -2.0))

        self.assertTrue(np.allclose(F3d, expected3d))

    def test_boundary_package_exposes_explicit_modules_only(self):
        self.assertTrue(hasattr(boundary, "body"))
        self.assertTrue(hasattr(boundary, "condition"))
        self.assertTrue(hasattr(boundary, "loads"))
        self.assertTrue(hasattr(boundary, "nodal"))
        self.assertTrue(hasattr(boundary, "constraints"))
        self.assertTrue(hasattr(boundary, "traction"))
        self.assertTrue(callable(boundary.body.add_forces))
        self.assertTrue(callable(boundary.nodal.add_forces))
        self.assertTrue(callable(boundary.traction.add_forces))
        self.assertFalse(hasattr(boundary, "BoundaryCondition2D"))
        self.assertFalse(hasattr(boundary, "BoundaryCondition3D"))
        self.assertFalse(hasattr(boundary, "build_load_vector_3d"))

    def test_3d_nodal_forces_accumulate_like_2d(self):
        mesh = make_tet4_stiffness_mesh()
        bc = boundary.condition.BoundaryCondition()
        bc.add_nodal_force(node_id=1, component=2, value=-2.0, mesh=mesh)
        bc.add_nodal_force(node_id=1, component=2, value=-3.0, mesh=mesh)

        F = boundary.loads.build_load_vector(mesh, bc)

        self.assertAlmostEqual(F[mesh.global_dof(1, 2)], -5.0)


if __name__ == "__main__":
    unittest.main()
