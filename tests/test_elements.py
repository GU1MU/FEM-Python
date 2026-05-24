import importlib
import sys
import unittest

import numpy as np

from fem import boundary
from fem.elements import get_element_kernel
from fem.elements.hexahedron import Hex8Kernel, hex8_shape_funcs_grads
from fem.elements.quadrilateral import (
    Quad4PlaneKernel,
    Quad8PlaneKernel,
    quad4_shape_grad_xi_eta,
    quad8_shape_funcs_grads,
)
from fem.elements.registry import register_element_kernel
from fem.elements.tetrahedron import Tet4Kernel, Tet10Kernel, tet10_shape_funcs_grads
from fem.elements.triangle import Tri3PlaneKernel
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
    make_hex8_solid_stress_mesh,
    make_hex8_stiffness_mesh,
    make_quad4_boundary_mesh,
    make_quad4_stiffness_mesh,
    make_quad8_load_mesh,
    make_quad8_stiffness_mesh,
    make_tet4_stiffness_mesh,
    make_tet10_stiffness_mesh,
    make_tri3_load_mesh,
    make_tri3_stiffness_mesh,
    make_truss_stiffness_mesh,
)


def _node_lookup(mesh):
    return {node.id: node for node in mesh.nodes}


def _assert_kernel_matches_explicit_node_lookup(test_case, mesh):
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)

    ke = kernel.stiffness(mesh, elem)
    expected = kernel.stiffness(mesh, elem, _node_lookup(mesh))

    test_case.assertTrue(np.allclose(ke, expected))


class LineElementKernelTests(unittest.TestCase):
    def test_truss_kernel_builds_node_lookup_when_omitted(self):
        _assert_kernel_matches_explicit_node_lookup(self, make_truss_stiffness_mesh())

    def test_beam_kernel_builds_node_lookup_when_omitted(self):
        _assert_kernel_matches_explicit_node_lookup(self, make_beam_stiffness_mesh())

    def test_truss_kernel_provides_element_stress(self):
        mesh = make_truss_stiffness_mesh()
        elem = mesh.elements[0]
        U = np.array([0.0, 0.0, 0.02, 0.0], dtype=float)

        axial_strain, axial_stress, mises = get_element_kernel("Truss2D").element_stress(
            mesh, elem, U
        )

        self.assertAlmostEqual(axial_strain, 0.01)
        self.assertAlmostEqual(axial_stress, 2.1)
        self.assertAlmostEqual(mises, 2.1)


class Quad4ElementKernelTests(unittest.TestCase):
    def test_quad4_stiffness_builds_node_lookup_from_mesh_when_omitted(self):
        mesh = make_quad4_stiffness_mesh()

        ke = get_element_kernel("Quad4Plane").stiffness(mesh, mesh.elements[0])

        self.assertEqual(ke.shape, (8, 8))
        self.assertTrue(np.allclose(ke, ke.T))

    def test_quad4_kernel_matches_explicit_node_lookup(self):
        _assert_kernel_matches_explicit_node_lookup(self, make_quad4_stiffness_mesh())


class PlaneElementKernelTests(unittest.TestCase):
    def test_tri3_kernel_matches_explicit_node_lookup(self):
        _assert_kernel_matches_explicit_node_lookup(self, make_tri3_stiffness_mesh())

    def test_quad8_kernel_matches_explicit_node_lookup(self):
        _assert_kernel_matches_explicit_node_lookup(self, make_quad8_stiffness_mesh())


class PlaneLoadKernelTests(unittest.TestCase):
    def test_plane_kernels_provide_body_force_and_edge_traction(self):
        cases = [
            (make_tri3_load_mesh(), 0, 2.0, 4.0),
            (make_quad4_boundary_mesh(), 0, 4.0, 4.0),
            (make_quad8_load_mesh(), 0, 6.0, 3.0),
        ]

        for mesh, edge, body_measure, edge_measure in cases:
            with self.subTest(element_type=mesh.elements[0].type):
                elem = mesh.elements[0]
                kernel = get_element_kernel(elem.type)

                body = kernel.body_force(mesh, elem, (4.0, -5.0))
                edge_load = kernel.edge_traction(mesh, elem, edge, (7.0, -11.0))

                self.assertEqual(body.shape, (len(elem.node_ids) * 2,))
                self.assertEqual(edge_load.shape, (len(elem.node_ids) * 2,))
                self.assertAlmostEqual(float(body[0::2].sum()), 4.0 * body_measure)
                self.assertAlmostEqual(float(body[1::2].sum()), -5.0 * body_measure)
                self.assertAlmostEqual(float(edge_load[0::2].sum()), 7.0 * edge_measure)
                self.assertAlmostEqual(float(edge_load[1::2].sum()), -11.0 * edge_measure)

    def test_tet_kernels_provide_body_force_and_face_traction(self):
        for mesh in (make_tet4_stiffness_mesh(), make_tet10_stiffness_mesh()):
            with self.subTest(element_type=mesh.elements[0].type):
                elem = mesh.elements[0]
                kernel = get_element_kernel(elem.type)

                body = kernel.body_force(mesh, elem, (0.0, 0.0, -6.0))
                face = kernel.face_traction(mesh, elem, 3, (0.0, 0.0, -2.0))

                self.assertEqual(body.shape, (len(elem.node_ids) * 3,))
                self.assertEqual(face.shape, (len(elem.node_ids) * 3,))
                self.assertAlmostEqual(float(body[2::3].sum()), -1.0)
                self.assertAlmostEqual(float(face[2::3].sum()), -1.0)

    def test_plane_load_kernels_do_not_require_elastic_props(self):
        for mesh in (make_quad4_boundary_mesh(), make_quad8_load_mesh()):
            with self.subTest(element_type=mesh.elements[0].type):
                elem = mesh.elements[0]
                elem.props = {"thickness": elem.props["thickness"]}
                kernel = get_element_kernel(elem.type)

                body = kernel.body_force(mesh, elem, (4.0, -5.0))
                edge_load = kernel.edge_traction(mesh, elem, 0, (7.0, -11.0))

                self.assertEqual(body.shape, (len(elem.node_ids) * 2,))
                self.assertEqual(edge_load.shape, (len(elem.node_ids) * 2,))


class PlaneStressKernelTests(unittest.TestCase):
    def test_plane_kernels_provide_stress_interfaces_without_post_helpers(self):
        cases = [
            (make_tri3_stiffness_mesh(), None),
            (make_quad4_stiffness_mesh(), 2),
            (make_quad8_stiffness_mesh(), 3),
        ]

        for mesh, gauss_order in cases:
            with self.subTest(element_type=mesh.elements[0].type):
                elem = mesh.elements[0]
                U = np.linspace(0.01, 0.01 * mesh.num_dofs, mesh.num_dofs)
                kernel = get_element_kernel(elem.type)

                if "tri3" in elem.type.lower():
                    node_vals, plane_type, nu = kernel.nodal_stress(
                        mesh, elem, U, _node_lookup(mesh)
                    )
                else:
                    node_vals, plane_type, nu = kernel.nodal_stress(
                        mesh, elem, U, _node_lookup(mesh), gauss_order
                    )

                self.assertEqual(node_vals.shape, (len(elem.node_ids), 3))
                self.assertEqual(plane_type, "stress")
                self.assertEqual(nu, elem.props["nu"])


class Hex8KernelTests(unittest.TestCase):
    def test_hex8_body_force_uses_actual_element_volume(self):
        mesh = make_hex8_stiffness_mesh()
        bc = boundary.condition.BoundaryCondition()
        bc.add_body_force_element(1, 0.0, 0.0, -2.0)

        loads = boundary.loads.build_load_vector(mesh, bc)

        self.assertAlmostEqual(float(loads[2::3].sum()), -48.0)
        self.assertTrue(np.allclose(loads[2::3], np.full(8, -6.0)))

    def test_hex8_face_traction_uses_actual_face_area(self):
        mesh = make_hex8_stiffness_mesh()
        bc = boundary.condition.BoundaryCondition()
        bc.add_surface_traction(1, 1, 0.0, 0.0, -5.0)

        loads = boundary.loads.build_load_vector(mesh, bc)

        self.assertAlmostEqual(float(loads[2::3].sum()), -30.0)
        self.assertTrue(np.allclose(loads[2::3][:4], np.zeros(4)))
        self.assertTrue(np.allclose(loads[2::3][4:], np.full(4, -7.5)))

    def test_hex8_kernel_matches_explicit_node_lookup(self):
        _assert_kernel_matches_explicit_node_lookup(self, make_hex8_stiffness_mesh())


class TetKernelTests(unittest.TestCase):
    def test_tet4_kernel_matches_explicit_node_lookup(self):
        _assert_kernel_matches_explicit_node_lookup(self, make_tet4_stiffness_mesh())

    def test_tet10_kernel_matches_explicit_node_lookup(self):
        _assert_kernel_matches_explicit_node_lookup(self, make_tet10_stiffness_mesh())


class SolidStressKernelTests(unittest.TestCase):
    def test_solid_kernels_provide_nodal_stress_matching_post_helpers(self):
        tet10_coords = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ]

        for mesh in (
            make_hex8_solid_stress_mesh(),
            make_tet4_stiffness_mesh(),
            make_tet10_stiffness_mesh(),
        ):
            with self.subTest(element_type=mesh.elements[0].type):
                elem = mesh.elements[0]
                U = np.linspace(0.01, 0.01 * mesh.num_dofs, mesh.num_dofs)
                node_lookup = _node_lookup(mesh)
                kernel = get_element_kernel(elem.type)

                if elem.type.lower() == "hex8":
                    a = 1.0 / np.sqrt(3.0)
                    gps = [
                        (-a, -a, -a),
                        (a, -a, -a),
                        (a, a, -a),
                        (-a, a, -a),
                        (-a, -a, a),
                        (a, -a, a),
                        (a, a, a),
                        (-a, a, a),
                    ]
                    gp_stresses = [
                        kernel.stress_at(mesh, elem, U, *gp, node_lookup)
                        for gp in gps
                    ]
                    expected = np.tile(np.mean(gp_stresses, axis=0), (8, 1))
                    node_vals = kernel.nodal_stress(mesh, elem, U, node_lookup)
                elif elem.type.lower() == "tet4":
                    stress = kernel.stress_at(mesh, elem, U, 0.25, 0.25, 0.25, node_lookup)
                    expected = np.tile(stress, (4, 1))
                    node_vals = kernel.nodal_stress(mesh, elem, U, node_lookup)
                else:
                    expected = np.array(
                        [
                            kernel.stress_at(mesh, elem, U, *coords, node_lookup)
                            for coords in tet10_coords
                        ],
                        dtype=float,
                    )
                    node_vals = kernel.nodal_stress(mesh, elem, U, node_lookup)

                self.assertTrue(np.allclose(node_vals, expected))


class ElementsPackageTests(unittest.TestCase):
    def test_elements_use_family_modules_and_registry_module(self):
        self.assertIs(type(get_element_kernel("Quad4Plane")), Quad4PlaneKernel)
        self.assertIs(type(get_element_kernel("Quad8Plane")), Quad8PlaneKernel)
        self.assertIs(type(get_element_kernel("Tri3Plane")), Tri3PlaneKernel)
        self.assertIs(type(get_element_kernel("Hex8")), Hex8Kernel)
        self.assertIs(type(get_element_kernel("Tet4")), Tet4Kernel)
        self.assertIs(type(get_element_kernel("Tet10")), Tet10Kernel)
        self.assertTrue(callable(quad4_shape_grad_xi_eta))
        self.assertTrue(callable(quad8_shape_funcs_grads))
        self.assertTrue(callable(hex8_shape_funcs_grads))
        self.assertTrue(callable(tet10_shape_funcs_grads))
        self.assertTrue(callable(register_element_kernel))

        for old_module in (
            "fem.elements.quad4",
            "fem.elements.quad8",
            "fem.elements.tri3",
            "fem.elements.tet",
            "fem.elements.hex8",
        ):
            sys.modules.pop(old_module, None)
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(old_module)


if __name__ == "__main__":
    unittest.main()
