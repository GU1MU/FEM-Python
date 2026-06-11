import unittest

import numpy as np
from scipy.sparse import csr_matrix

from fem import boundary, materials, post, selection, solvers
from fem.assemble import assemble_global_stiffness, assemble_global_stiffness_sparse
from fem.elements import get_element_kernel
from fem.elements.hexahedron import hex8_shape_funcs_grads
from fem.elements.tetrahedron import TET10_NATURAL_NODE_COORDS, tet10_gauss_points
from fem.core.dof import DofMap
from fem.core.mesh import BeamMesh2D, Element2D, Element3D, HexMesh3D, Node2D, Node3D, PlaneMesh2D, TetMesh3D, TrussMesh2D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementFace,
    ElementSet,
    FEMModel,
    MaterialDefinition,
    NodalLoad,
    NodeSet,
    OutputRequest,
    SectionAssignment,
    Surface,
)
from fem.core.result import ModelResult, ModelResults
from tests.helpers.file_builders import write_inp
from tests.helpers.paths import (
    fresh_test_output_dir,
    temporary_directory,
    test_output_path as output_path_for_test,
)


class DofMapRegressionTests(unittest.TestCase):
    def test_dof_map_handles_mesh_nodes_independent_of_dimension(self):
        nodes = [Node2D(20, 0.0, 0.0), Node2D(10, 1.0, 0.0)]

        dof_map = DofMap.from_nodes(nodes, dofs_per_node=2)

        self.assertEqual(dof_map.node_ids, [10, 20])
        self.assertEqual(dof_map.node_dofs(10), [0, 1])
        self.assertEqual(dof_map.node_dofs(20), [2, 3])
        self.assertEqual(dof_map.element_dofs([20, 10]), [2, 3, 0, 1])

    def test_dof_map_rejects_duplicate_nodes_and_invalid_components(self):
        nodes = [Node2D(1, 0.0, 0.0), Node2D(1, 1.0, 0.0)]

        with self.assertRaises(ValueError):
            DofMap.from_nodes(nodes, dofs_per_node=2)

        dof_map = DofMap.from_nodes([Node2D(1, 0.0, 0.0)], dofs_per_node=2)
        with self.assertRaises(IndexError):
            dof_map.global_dof(1, -1)
        with self.assertRaises(IndexError):
            dof_map.global_dof(1, 2)

    def test_core_dof_exposes_dof_map_without_legacy_dof_manager(self):
        import importlib
        import sys

        from fem.core import dof

        self.assertIs(dof.DofMap, DofMap)
        self.assertFalse(hasattr(dof, "DofManager2D"))
        self.assertFalse(hasattr(dof, "DofManager3D"))

        sys.modules.pop("fem.dof_manager", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.dof_manager")

    def test_meshes_expose_dof_interface_without_dof_manager_access(self):
        import importlib
        import sys

        from fem.core import mesh as core_mesh

        self.assertIs(core_mesh.HexMesh3D, HexMesh3D)
        sys.modules.pop("fem.mesh", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.mesh")

        meshes = [
            TrussMesh2D(
                nodes=[Node2D(20, 0.0, 0.0), Node2D(10, 1.0, 0.0)],
                elements=[Element2D(1, [20, 10], type="Truss2D")],
            ),
            BeamMesh2D(
                nodes=[Node2D(20, 0.0, 0.0), Node2D(10, 1.0, 0.0)],
                elements=[Element2D(1, [20, 10], type="Beam2D")],
            ),
            PlaneMesh2D(
                nodes=[Node2D(20, 0.0, 0.0), Node2D(10, 1.0, 0.0)],
                elements=[Element2D(1, [20, 10], type="Tri3Plane")],
            ),
            HexMesh3D(
                nodes=[Node3D(20, 0.0, 0.0, 0.0), Node3D(10, 1.0, 0.0, 0.0)],
                elements=[Element3D(1, [20, 10], type="Hex8")],
            ),
            TetMesh3D(
                nodes=[Node3D(20, 0.0, 0.0, 0.0), Node3D(10, 1.0, 0.0, 0.0)],
                elements=[Element3D(1, [20, 10], type="Tet4")],
            ),
        ]

        for mesh in meshes:
            self.assertTrue(hasattr(mesh, "dof_map"))
            self.assertFalse(hasattr(mesh, "dof_manager"))
            self.assertEqual(mesh.node_ids, [10, 20])
            self.assertEqual(mesh.node_dofs(10), list(range(mesh.dofs_per_node)))
            self.assertEqual(
                mesh.element_dofs(mesh.elements[0])[:mesh.dofs_per_node],
                mesh.node_dofs(20),
            )

    def test_core_model_stores_sets_surfaces_materials_and_sections(self):
        mesh = HexMesh3D(
            nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
            elements=[Element3D(1, [1, 2], type="Hex8")],
        )
        node_set = NodeSet("FIXED", [1, 2])
        element_set = ElementSet("SOLID", [1])
        surface = Surface("LOAD", [ElementFace(1, 0, [1, 2])])
        material = MaterialDefinition("STEEL", {"E": 210.0, "nu": 0.3})
        section = SectionAssignment("SOLID", "STEEL")
        model = FEMModel(
            mesh=mesh,
            node_sets={node_set.name: node_set},
            element_sets={element_set.name: element_set},
            surfaces={surface.name: surface},
            materials={material.name: material},
            sections=[section],
            name="job",
        )

        self.assertEqual(model.name, "job")
        self.assertEqual(model.node_sets["FIXED"].node_ids, (1, 2))
        self.assertEqual(model.element_sets["SOLID"].element_ids, (1,))
        self.assertEqual(model.surfaces["LOAD"].faces[0].local_index, 0)
        self.assertEqual(model.materials["STEEL"].properties["E"], 210.0)
        self.assertEqual(model.sections[0].element_set, "SOLID")

    def test_core_model_stores_analysis_steps(self):
        mesh = HexMesh3D(
            nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
            elements=[Element3D(1, [1, 2], type="Hex8")],
        )
        step = AnalysisStep(
            "load",
            procedure="static",
            boundaries=[DisplacementConstraint("FIXED", 1, 3, 0.0)],
            cloads=[NodalLoad("TIP", 3, -100.0)],
            metadata={"nlgeom": "NO"},
        )
        model = FEMModel(mesh=mesh, steps=[step])

        self.assertEqual(model.steps[0].name, "load")
        self.assertEqual(model.steps[0].boundaries[0].target, "FIXED")
        self.assertEqual(model.steps[0].cloads[0].component, 3)
        self.assertEqual(model.steps[0].metadata["nlgeom"], "NO")

    def test_core_model_has_no_solver_or_boundary_pipeline_methods(self):
        forbidden = (
            "boundary",
            "from_mesh",
            "add_node_set",
            "add_element_set",
            "add_surface",
            "add_material",
            "assign_section",
            "add_step",
            "add_displacement",
            "add_nodal_load",
            "add_surface_traction",
            "add_surface_pressure",
            "add_output_request",
            "get_step",
            "boundary_for_step",
            "assemble_stiffness",
            "load_vector",
            "solve",
            "run",
            "run_all",
        )
        for name in forbidden:
            self.assertFalse(hasattr(FEMModel, name))

    def test_static_linear_solver_builds_step_boundary_and_solves_case(self):
        from fem.solvers import static_linear

        mesh = TrussMesh2D(
            nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
            elements=[Element2D(1, [1, 2], type="Truss2D", props={"E": 100.0, "area": 2.0})],
        )
        step = AnalysisStep(
            "pull",
            boundaries=[
                DisplacementConstraint("FIXED", 1, 2, 0.0),
                DisplacementConstraint(2, 2, 2, 0.0),
            ],
            cloads=[NodalLoad("TIP", 1, 100.0)],
        )
        model = FEMModel(
            mesh=mesh,
            node_sets={
                "FIXED": NodeSet("FIXED", [1]),
                "TIP": NodeSet("TIP", [2]),
            },
            steps=[step],
        )

        bc = static_linear.boundary_for_step(model, "pull")
        self.assertEqual(len(bc.prescribed_displacements), 3)
        self.assertAlmostEqual(sum(bc.nodal_forces.values()), 100.0)

        U = static_linear.solve(model, "pull").U

        self.assertAlmostEqual(U[mesh.global_dof(2, 0)], 0.5)
        self.assertAlmostEqual(U[mesh.global_dof(2, 1)], 0.0)

    def test_static_linear_solver_returns_result_with_displacements_and_reactions(self):
        from fem.solvers import static_linear

        mesh = TrussMesh2D(
            nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
            elements=[Element2D(1, [1, 2], type="Truss2D", props={"E": 100.0, "area": 2.0})],
        )
        step = AnalysisStep(
            "pull",
            boundaries=[
                DisplacementConstraint("FIXED", 1, 2, 0.0),
                DisplacementConstraint(2, 2, 2, 0.0),
            ],
            cloads=[NodalLoad("TIP", 1, 100.0)],
        )
        model = FEMModel(
            mesh=mesh,
            node_sets={
                "FIXED": NodeSet("FIXED", [1]),
                "TIP": NodeSet("TIP", [2]),
            },
            steps=[step],
        )

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
        from fem.solvers import static_linear

        mesh = TrussMesh2D(
            nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
            elements=[Element2D(1, [1, 2], type="Truss2D", props={"E": 100.0, "area": 2.0})],
        )
        model = FEMModel(
            mesh=mesh,
            node_sets={
                "FIXED": NodeSet("FIXED", [1]),
                "TIP": NodeSet("TIP", [2]),
            },
            steps=[
                AnalysisStep("Initial", boundaries=[
                    DisplacementConstraint("FIXED", 1, 2, 0.0),
                    DisplacementConstraint(2, 2, 2, 0.0),
                ]),
                AnalysisStep("pull1", cloads=[NodalLoad("TIP", 1, 100.0)]),
                AnalysisStep("pull2", cloads=[NodalLoad("TIP", 1, 200.0)]),
            ],
            name="bar",
        )

        results = static_linear.solve_all(model)

        self.assertIsInstance(results, ModelResults)
        self.assertEqual(len(results.results), 2)
        self.assertEqual(tuple(result.step.name for result in results.results), ("pull1", "pull2"))
        pull1, pull2 = results.results
        self.assertAlmostEqual(pull1.U[mesh.global_dof(2, 0)], 0.5)
        self.assertAlmostEqual(pull2.U[mesh.global_dof(2, 0)], 1.0)
        self.assertEqual(pull1.name, "bar_pull1")
        self.assertEqual(pull2.name, "bar_pull2")

    def test_core_model_supports_hand_written_mesh_model_solve_result_flow(self):
        from fem.solvers import static_linear

        mesh = TrussMesh2D(
            nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
            elements=[Element2D(1, [1, 2], type="Truss2D")],
        )

        from fem import steps

        model = FEMModel(mesh=mesh, name="manual_bar")
        material = materials.linear_elastic.material("steel", E=100.0, nu=0.3)
        materials.add(model, material)
        element_set = ElementSet("bar", [1])
        model.element_sets[element_set.name] = element_set
        section = materials.assign(model, "steel", "bar", area=2.0)
        fixed = NodeSet("fixed", [1])
        tip = NodeSet("tip", [2])
        model.node_sets[fixed.name] = fixed
        model.node_sets[tip.name] = tip
        step = steps.static("pull")
        steps.displacement(step, "fixed", components=(1, 2))
        steps.displacement(step, 2, components=2)
        steps.nodal_load(step, "tip", component=1, value=100.0)
        steps.add(model, step)

        result = static_linear.solve(model, "pull")

        self.assertEqual(material, MaterialDefinition("steel", {"E": 100.0, "nu": 0.3}))
        self.assertEqual(element_set, ElementSet("bar", (1,)))
        self.assertEqual(section.element_set, "bar")
        self.assertEqual(section.properties["area"], 2.0)
        self.assertEqual(fixed, NodeSet("fixed", (1,)))
        self.assertEqual(tip, NodeSet("tip", (2,)))
        self.assertEqual(step.name, "pull")
        self.assertEqual(len(step.boundaries), 2)
        self.assertEqual(len(step.cloads), 1)
        self.assertEqual(mesh.elements[0].props["E"], 100.0)
        self.assertEqual(mesh.elements[0].props["area"], 2.0)
        self.assertAlmostEqual(result.U[mesh.global_dof(2, 0)], 0.5)
        self.assertAlmostEqual(result.reactions[mesh.global_dof(1, 0)], -100.0)


class SelectionRegressionTests(unittest.TestCase):
    def test_selection_package_exposes_nodes_edges_and_faces_only(self):
        import importlib.util

        self.assertTrue(hasattr(selection, "nodes"))
        self.assertTrue(hasattr(selection, "edges"))
        self.assertTrue(hasattr(selection, "elements"))
        self.assertTrue(hasattr(selection, "faces"))
        self.assertIsNone(importlib.util.find_spec("fem.helper"))

    def test_nodes_select_2d_and_3d_coordinates(self):
        mesh2d = PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 1.0, 0.0),
                Node2D(3, 1.0, 2.0),
            ],
            elements=[],
        )
        mesh3d = HexMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 1.0, 2.0, 3.0),
            ],
            elements=[],
        )

        self.assertEqual(selection.nodes.by_x(mesh2d, 1.0), [2, 3])
        self.assertEqual(selection.nodes.by_coord(mesh2d, x=1.0, y=2.0), [3])
        self.assertEqual(selection.nodes.by_z(mesh3d, 3.0), [3])
        self.assertEqual(selection.nodes.by_coord(mesh3d, x=1.0, z=0.0), [2])

    def test_edges_select_boundary_edges_by_coordinate(self):
        mesh = PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 1.0, 0.0),
                Node2D(3, 1.0, 1.0),
                Node2D(4, 0.0, 1.0),
            ],
            elements=[Element2D(1, [1, 2, 3, 4], type="Quad4")],
        )

        self.assertEqual(len(selection.edges.boundary(mesh)), 4)
        self.assertEqual(selection.edges.by_x(mesh, 0.0), [(1, 3, [4, 1])])
        self.assertEqual(selection.edges.by_y(mesh, 1.0), [(1, 2, [3, 4])])

    def test_elements_select_by_id_and_type_and_build_sets(self):
        mesh = PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 1.0, 0.0),
                Node2D(3, 1.0, 1.0),
                Node2D(4, 0.0, 1.0),
            ],
            elements=[
                Element2D(1, [1, 2, 3], type="Tri3"),
                Element2D(2, [1, 3, 4], type="Quad4"),
            ],
        )

        self.assertEqual(selection.elements.all(mesh), [1, 2])
        self.assertEqual(selection.elements.by_type(mesh, "quad4"), [2])
        self.assertEqual(selection.elements.by_ids(mesh, [2, 3]), [2])
        self.assertEqual(
            selection.elements.set_by_type(mesh, "QUADS", "quad4"),
            ElementSet("QUADS", (2,)),
        )

    def test_faces_select_boundary_faces_by_coordinate(self):
        mesh = HexMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 2.0, 0.0, 0.0),
                Node3D(3, 2.0, 3.0, 0.0),
                Node3D(4, 0.0, 3.0, 0.0),
                Node3D(5, 0.0, 0.0, 4.0),
                Node3D(6, 2.0, 0.0, 4.0),
                Node3D(7, 2.0, 3.0, 4.0),
                Node3D(8, 0.0, 3.0, 4.0),
            ],
            elements=[Element3D(1, [1, 2, 3, 4, 5, 6, 7, 8], type="Hex8")],
        )

        self.assertEqual(len(selection.faces.boundary(mesh)), 6)
        self.assertEqual(selection.faces.by_z(mesh, 4.0), [(1, 1, [5, 6, 7, 8])])
        self.assertEqual(selection.faces.by_x(mesh, 2.0), [(1, 5, [2, 3, 7, 6])])

    def test_selection_can_build_model_sets_and_surfaces(self):
        mesh = HexMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 2.0, 0.0, 0.0),
                Node3D(3, 2.0, 3.0, 0.0),
                Node3D(4, 0.0, 3.0, 0.0),
                Node3D(5, 0.0, 0.0, 4.0),
                Node3D(6, 2.0, 0.0, 4.0),
                Node3D(7, 2.0, 3.0, 4.0),
                Node3D(8, 0.0, 3.0, 4.0),
            ],
            elements=[Element3D(1, [1, 2, 3, 4, 5, 6, 7, 8], type="Hex8")],
        )

        fixed = selection.nodes.set_by_x(mesh, "FIXED", 0.0)
        load_surface = selection.faces.surface_by_x(mesh, "LOAD", 2.0)

        self.assertIsInstance(fixed, NodeSet)
        self.assertEqual(fixed.node_ids, (1, 4, 5, 8))
        self.assertIsInstance(load_surface, Surface)
        self.assertEqual(load_surface.faces, (ElementFace(1, 5, (2, 3, 7, 6)),))


class MaterialsRegressionTests(unittest.TestCase):
    def test_materials_package_exposes_linear_elastic_module_only(self):
        self.assertTrue(hasattr(materials, "linear_elastic"))
        self.assertFalse(hasattr(materials, "compute_plane_stress_matrix"))
        self.assertFalse(hasattr(materials, "compute_plane_strain_matrix"))
        self.assertFalse(hasattr(materials, "compute_plane_elastic_matrix"))
        self.assertFalse(hasattr(materials, "compute_3d_elastic_matrix"))

    def test_linear_elastic_constitutive_matrices(self):
        E = 210.0
        nu = 0.3

        plane_stress = materials.linear_elastic.plane_stress_matrix(E, nu)
        plane_matrix = materials.linear_elastic.plane_matrix(E, nu, "stress")
        solid = materials.linear_elastic.solid_3d_matrix(E, nu)

        self.assertEqual(plane_stress.shape, (3, 3))
        self.assertTrue(np.allclose(plane_matrix, plane_stress))
        self.assertEqual(solid.shape, (6, 6))
        self.assertAlmostEqual(plane_stress[0, 1], E * nu / (1.0 - nu ** 2))


class SolverRegressionTests(unittest.TestCase):
    def test_solver_package_exposes_linear_solver_only(self):
        import importlib.util

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


class LineElementKernelRegressionTests(unittest.TestCase):
    def make_truss_mesh(self):
        return TrussMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2],
                    type="Truss2D",
                    props={"E": 210.0, "area": 0.5},
                )
            ],
        )

    def make_beam_mesh(self):
        return BeamMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2],
                    type="Beam2D",
                    props={"E": 210.0, "area": 0.5, "Izz": 0.25},
                )
            ],
        )

    def test_truss_kernel_builds_node_lookup_when_omitted(self):
        mesh = self.make_truss_mesh()

        kernel = get_element_kernel("Truss2D")
        ke = kernel.stiffness(mesh, mesh.elements[0])

        node_lookup = {node.id: node for node in mesh.nodes}
        expected = kernel.stiffness(mesh, mesh.elements[0], node_lookup)
        self.assertTrue(np.allclose(ke, expected))

    def test_beam_kernel_builds_node_lookup_when_omitted(self):
        mesh = self.make_beam_mesh()

        kernel = get_element_kernel("Beam2D")
        ke = kernel.stiffness(mesh, mesh.elements[0])

        node_lookup = {node.id: node for node in mesh.nodes}
        expected = kernel.stiffness(mesh, mesh.elements[0], node_lookup)
        self.assertTrue(np.allclose(ke, expected))

    def test_truss_kernel_provides_element_stress(self):
        mesh = self.make_truss_mesh()
        elem = mesh.elements[0]
        U = np.array([0.0, 0.0, 0.02, 0.0], dtype=float)

        axial_strain, axial_stress, mises = get_element_kernel("Truss2D").element_stress(
            mesh, elem, U
        )

        self.assertAlmostEqual(axial_strain, 0.01)
        self.assertAlmostEqual(axial_stress, 2.1)
        self.assertAlmostEqual(mises, 2.1)

    def test_sparse_assembly_accepts_mesh_for_truss_and_beam(self):
        for mesh in (self.make_truss_mesh(), self.make_beam_mesh()):
            K = assemble_global_stiffness_sparse(mesh)

            self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
            self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_and_sparse_assembly_accept_mesh(self):
        for mesh in (self.make_truss_mesh(), self.make_beam_mesh()):
            K_dense = assemble_global_stiffness(mesh)
            K_sparse = assemble_global_stiffness_sparse(mesh)

            self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))

    def test_assembly_requires_mesh_only(self):
        mesh = self.make_truss_mesh()

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


class Quad4StiffnessRegressionTests(unittest.TestCase):
    def make_quad4_mesh(self):
        return PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
                Node2D(3, 2.0, 1.0),
                Node2D(4, 0.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2, 3, 4],
                    type="Quad4Plane",
                    props={
                        "E": 210.0,
                        "nu": 0.3,
                        "thickness": 1.0,
                        "plane_type": "stress",
                    },
                )
            ],
        )

    def test_quad4_stiffness_builds_node_lookup_from_mesh_when_omitted(self):
        mesh = self.make_quad4_mesh()

        ke = get_element_kernel("Quad4Plane").stiffness(mesh, mesh.elements[0])

        self.assertEqual(ke.shape, (8, 8))
        self.assertTrue(np.allclose(ke, ke.T))

    def test_quad4_kernel_matches_explicit_node_lookup(self):
        mesh = self.make_quad4_mesh()

        kernel = get_element_kernel("Quad4Plane")
        ke = kernel.stiffness(mesh, mesh.elements[0])

        node_lookup = {node.id: node for node in mesh.nodes}
        expected = kernel.stiffness(mesh, mesh.elements[0], node_lookup)
        self.assertTrue(np.allclose(ke, expected))

    def test_sparse_assembly_accepts_mesh_for_quad4(self):
        mesh = self.make_quad4_mesh()

        K = assemble_global_stiffness_sparse(mesh)

        self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
        self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_assembly_matches_sparse_for_quad4(self):
        mesh = self.make_quad4_mesh()

        K_dense = assemble_global_stiffness(mesh)
        K_sparse = assemble_global_stiffness_sparse(mesh)

        self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))


class PlaneElementKernelRegressionTests(unittest.TestCase):
    def make_tri3_mesh(self):
        return PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
                Node2D(3, 0.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2, 3],
                    type="Tri3Plane",
                    props={
                        "E": 210.0,
                        "nu": 0.3,
                        "thickness": 1.0,
                        "plane_type": "stress",
                    },
                )
            ],
        )

    def make_quad8_mesh(self):
        return PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
                Node2D(3, 2.0, 2.0),
                Node2D(4, 0.0, 2.0),
                Node2D(5, 1.0, 0.0),
                Node2D(6, 2.0, 1.0),
                Node2D(7, 1.0, 2.0),
                Node2D(8, 0.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                    type="Quad8Plane",
                    props={
                        "E": 210.0,
                        "nu": 0.3,
                        "thickness": 1.0,
                        "plane_type": "stress",
                    },
                )
            ],
        )

    def test_tri3_kernel_matches_explicit_node_lookup(self):
        mesh = self.make_tri3_mesh()

        kernel = get_element_kernel("Tri3Plane")
        ke = kernel.stiffness(mesh, mesh.elements[0])

        node_lookup = {node.id: node for node in mesh.nodes}
        expected = kernel.stiffness(mesh, mesh.elements[0], node_lookup)
        self.assertTrue(np.allclose(ke, expected))

    def test_quad8_kernel_matches_explicit_node_lookup(self):
        mesh = self.make_quad8_mesh()

        kernel = get_element_kernel("Quad8Plane")
        ke = kernel.stiffness(mesh, mesh.elements[0])

        node_lookup = {node.id: node for node in mesh.nodes}
        expected = kernel.stiffness(mesh, mesh.elements[0], node_lookup)
        self.assertTrue(np.allclose(ke, expected))

    def test_sparse_assembly_accepts_mesh_for_tri3_and_quad8(self):
        for mesh in (self.make_tri3_mesh(), self.make_quad8_mesh()):
            K = assemble_global_stiffness_sparse(mesh)

            self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
            self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_assembly_matches_sparse_for_tri3_and_quad8(self):
        for mesh in (self.make_tri3_mesh(), self.make_quad8_mesh()):
            K_dense = assemble_global_stiffness(mesh)
            K_sparse = assemble_global_stiffness_sparse(mesh)

            self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))


class BoundaryKernelRegressionTests(unittest.TestCase):
    def make_tri3_mesh(self):
        return PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
                Node2D(3, 0.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2, 3],
                    type="Tri3Plane",
                    props={"E": 210.0, "nu": 0.3, "thickness": 2.0},
                )
            ],
        )

    def make_quad4_mesh(self):
        return PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
                Node2D(3, 2.0, 1.0),
                Node2D(4, 0.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2, 3, 4],
                    type="Quad4Plane",
                    props={"E": 210.0, "nu": 0.3, "thickness": 2.0},
                )
            ],
        )

    def make_quad8_mesh(self):
        return PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
                Node2D(3, 2.0, 2.0),
                Node2D(4, 0.0, 2.0),
                Node2D(5, 1.0, 0.0),
                Node2D(6, 2.0, 1.0),
                Node2D(7, 1.0, 2.0),
                Node2D(8, 0.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                    type="Quad8Plane",
                    props={"E": 210.0, "nu": 0.3, "thickness": 1.5},
                )
            ],
        )

    def make_tet4_mesh(self):
        return TetMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 0.0, 1.0, 0.0),
                Node3D(4, 0.0, 0.0, 1.0),
            ],
            elements=[
                Element3D(
                    id=1,
                    node_ids=[1, 2, 3, 4],
                    type="Tet4",
                    props={"E": 210.0, "nu": 0.3},
                )
            ],
        )

    def make_tet10_mesh(self):
        return TetMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 0.0, 1.0, 0.0),
                Node3D(4, 0.0, 0.0, 1.0),
                Node3D(5, 0.5, 0.0, 0.0),
                Node3D(6, 0.5, 0.5, 0.0),
                Node3D(7, 0.0, 0.5, 0.0),
                Node3D(8, 0.0, 0.0, 0.5),
                Node3D(9, 0.5, 0.0, 0.5),
                Node3D(10, 0.0, 0.5, 0.5),
            ],
            elements=[
                Element3D(
                    id=1,
                    node_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    type="Tet10",
                    props={"E": 210.0, "nu": 0.3},
                )
            ],
        )

    def test_plane_kernels_provide_body_force_and_edge_traction(self):
        cases = [
            (self.make_tri3_mesh(), 0, 2.0, 4.0),
            (self.make_quad4_mesh(), 0, 4.0, 4.0),
            (self.make_quad8_mesh(), 0, 6.0, 3.0),
        ]

        for mesh, edge, body_measure, edge_measure in cases:
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
        for mesh in (self.make_tet4_mesh(), self.make_tet10_mesh()):
            elem = mesh.elements[0]
            kernel = get_element_kernel(elem.type)

            body = kernel.body_force(mesh, elem, (0.0, 0.0, -6.0))
            face = kernel.face_traction(mesh, elem, 3, (0.0, 0.0, -2.0))

            self.assertEqual(body.shape, (len(elem.node_ids) * 3,))
            self.assertEqual(face.shape, (len(elem.node_ids) * 3,))
            self.assertAlmostEqual(float(body[2::3].sum()), -1.0)
            self.assertAlmostEqual(float(face[2::3].sum()), -1.0)

    def test_boundary_load_vector_matches_kernel_dispatch(self):
        mesh = self.make_quad4_mesh()
        elem = mesh.elements[0]
        bc = boundary.condition.BoundaryCondition()
        bc.add_body_force_element(elem.id, 4.0, -5.0)
        bc.add_surface_traction(elem.id, 0, 7.0, -11.0)

        F = boundary.loads.build_load_vector(mesh, bc)
        kernel = get_element_kernel(elem.type)
        expected = kernel.body_force(mesh, elem, (4.0, -5.0))
        expected += kernel.edge_traction(mesh, elem, 0, (7.0, -11.0))

        self.assertTrue(np.allclose(F, expected))

        mesh3d = self.make_tet4_mesh()
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
        mesh = self.make_tet4_mesh()
        bc = boundary.condition.BoundaryCondition()
        bc.add_nodal_force(node_id=1, component=2, value=-2.0, mesh=mesh)
        bc.add_nodal_force(node_id=1, component=2, value=-3.0, mesh=mesh)

        F = boundary.loads.build_load_vector(mesh, bc)

        self.assertAlmostEqual(F[mesh.global_dof(1, 2)], -5.0)

    def test_plane_load_kernels_do_not_require_elastic_props(self):
        for mesh in (self.make_quad4_mesh(), self.make_quad8_mesh()):
            elem = mesh.elements[0]
            elem.props = {"thickness": elem.props["thickness"]}
            kernel = get_element_kernel(elem.type)

            body = kernel.body_force(mesh, elem, (4.0, -5.0))
            edge_load = kernel.edge_traction(mesh, elem, 0, (7.0, -11.0))

            self.assertEqual(body.shape, (len(elem.node_ids) * 2,))
            self.assertEqual(edge_load.shape, (len(elem.node_ids) * 2,))


class PlaneStressKernelRegressionTests(unittest.TestCase):
    def make_tri3_mesh(self):
        return PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
                Node2D(3, 0.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2, 3],
                    type="Tri3Plane",
                    props={"E": 210.0, "nu": 0.3, "thickness": 1.0, "plane_type": "stress"},
                )
            ],
        )

    def make_quad4_mesh(self):
        return PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
                Node2D(3, 2.0, 1.0),
                Node2D(4, 0.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2, 3, 4],
                    type="Quad4Plane",
                    props={"E": 210.0, "nu": 0.3, "thickness": 1.0, "plane_type": "stress"},
                )
            ],
        )

    def make_quad8_mesh(self):
        return PlaneMesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 2.0, 0.0),
                Node2D(3, 2.0, 2.0),
                Node2D(4, 0.0, 2.0),
                Node2D(5, 1.0, 0.0),
                Node2D(6, 2.0, 1.0),
                Node2D(7, 1.0, 2.0),
                Node2D(8, 0.0, 1.0),
            ],
            elements=[
                Element2D(
                    id=1,
                    node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                    type="Quad8Plane",
                    props={"E": 210.0, "nu": 0.3, "thickness": 1.0, "plane_type": "stress"},
                )
            ],
        )

    def test_plane_kernels_provide_stress_interfaces_without_post_helpers(self):
        cases = [
            (self.make_tri3_mesh(), None),
            (self.make_quad4_mesh(), 2),
            (self.make_quad8_mesh(), 3),
        ]

        for mesh, gauss_order in cases:
            elem = mesh.elements[0]
            U = np.linspace(0.01, 0.01 * mesh.num_dofs, mesh.num_dofs)
            node_lookup = {node.id: node for node in mesh.nodes}
            kernel = get_element_kernel(elem.type)

            if "tri3" in elem.type.lower():
                node_vals, plane_type, nu = kernel.nodal_stress(mesh, elem, U, node_lookup)
            elif "quad4" in elem.type.lower():
                node_vals, plane_type, nu = kernel.nodal_stress(mesh, elem, U, node_lookup, gauss_order)
            else:
                node_vals, plane_type, nu = kernel.nodal_stress(mesh, elem, U, node_lookup, gauss_order)

            self.assertEqual(node_vals.shape, (len(elem.node_ids), 3))
            self.assertEqual(plane_type, "stress")
            self.assertEqual(nu, elem.props["nu"])


class Hex8LoadRegressionTests(unittest.TestCase):
    def make_box_mesh(self):
        nodes = [
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 2.0, 0.0, 0.0),
            Node3D(3, 2.0, 3.0, 0.0),
            Node3D(4, 0.0, 3.0, 0.0),
            Node3D(5, 0.0, 0.0, 4.0),
            Node3D(6, 2.0, 0.0, 4.0),
            Node3D(7, 2.0, 3.0, 4.0),
            Node3D(8, 0.0, 3.0, 4.0),
        ]
        elem = Element3D(
            id=1,
            node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            type="Hex8",
            props={"E": 1.0, "nu": 0.25},
        )
        return HexMesh3D(nodes=nodes, elements=[elem])

    def test_hex8_body_force_uses_actual_element_volume(self):
        mesh = self.make_box_mesh()
        bc = boundary.condition.BoundaryCondition()
        bc.add_body_force_element(1, 0.0, 0.0, -2.0)

        loads = boundary.loads.build_load_vector(mesh, bc)

        self.assertAlmostEqual(float(loads[2::3].sum()), -48.0)
        self.assertTrue(np.allclose(loads[2::3], np.full(8, -6.0)))

    def test_hex8_face_traction_uses_actual_face_area(self):
        mesh = self.make_box_mesh()
        bc = boundary.condition.BoundaryCondition()
        bc.add_surface_traction(1, 1, 0.0, 0.0, -5.0)

        loads = boundary.loads.build_load_vector(mesh, bc)

        self.assertAlmostEqual(float(loads[2::3].sum()), -30.0)
        self.assertTrue(np.allclose(loads[2::3][:4], np.zeros(4)))
        self.assertTrue(np.allclose(loads[2::3][4:], np.full(4, -7.5)))

    def test_hex8_kernel_matches_explicit_node_lookup(self):
        mesh = self.make_box_mesh()

        kernel = get_element_kernel("Hex8")
        ke = kernel.stiffness(mesh, mesh.elements[0])

        node_lookup = {node.id: node for node in mesh.nodes}
        expected = kernel.stiffness(mesh, mesh.elements[0], node_lookup)
        self.assertTrue(np.allclose(ke, expected))

    def test_sparse_assembly_accepts_mesh_for_hex8(self):
        mesh = self.make_box_mesh()

        K = assemble_global_stiffness_sparse(mesh)

        self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
        self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_assembly_matches_sparse_for_hex8(self):
        mesh = self.make_box_mesh()

        K_dense = assemble_global_stiffness(mesh)
        K_sparse = assemble_global_stiffness_sparse(mesh)

        self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))


class TetElementKernelRegressionTests(unittest.TestCase):
    def make_tet4_mesh(self):
        return TetMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 0.0, 1.0, 0.0),
                Node3D(4, 0.0, 0.0, 1.0),
            ],
            elements=[
                Element3D(
                    id=1,
                    node_ids=[1, 2, 3, 4],
                    type="Tet4",
                    props={"E": 210.0, "nu": 0.3},
                )
            ],
        )

    def make_tet10_mesh(self):
        return TetMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 0.0, 1.0, 0.0),
                Node3D(4, 0.0, 0.0, 1.0),
                Node3D(5, 0.5, 0.0, 0.0),
                Node3D(6, 0.5, 0.5, 0.0),
                Node3D(7, 0.0, 0.5, 0.0),
                Node3D(8, 0.0, 0.0, 0.5),
                Node3D(9, 0.5, 0.0, 0.5),
                Node3D(10, 0.0, 0.5, 0.5),
            ],
            elements=[
                Element3D(
                    id=1,
                    node_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    type="Tet10",
                    props={"E": 210.0, "nu": 0.3},
                )
            ],
        )

    def test_tet4_kernel_matches_explicit_node_lookup(self):
        mesh = self.make_tet4_mesh()

        kernel = get_element_kernel("Tet4")
        ke = kernel.stiffness(mesh, mesh.elements[0])

        node_lookup = {node.id: node for node in mesh.nodes}
        expected = kernel.stiffness(mesh, mesh.elements[0], node_lookup)
        self.assertTrue(np.allclose(ke, expected))

    def test_tet10_kernel_matches_explicit_node_lookup(self):
        mesh = self.make_tet10_mesh()

        kernel = get_element_kernel("Tet10")
        ke = kernel.stiffness(mesh, mesh.elements[0])

        node_lookup = {node.id: node for node in mesh.nodes}
        expected = kernel.stiffness(mesh, mesh.elements[0], node_lookup)
        self.assertTrue(np.allclose(ke, expected))

    def test_sparse_assembly_accepts_mesh_for_tet4_and_tet10(self):
        for mesh in (self.make_tet4_mesh(), self.make_tet10_mesh()):
            K = assemble_global_stiffness_sparse(mesh)

            self.assertEqual(K.shape, (mesh.num_dofs, mesh.num_dofs))
            self.assertTrue(np.allclose(K.toarray(), K.toarray().T))

    def test_dense_assembly_matches_sparse_for_tet4_and_tet10(self):
        for mesh in (self.make_tet4_mesh(), self.make_tet10_mesh()):
            K_dense = assemble_global_stiffness(mesh)
            K_sparse = assemble_global_stiffness_sparse(mesh)

            self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))


class SolidStressKernelRegressionTests(unittest.TestCase):
    def make_hex8_mesh(self):
        nodes = [
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 2.0, 0.0, 0.0),
            Node3D(3, 2.0, 3.0, 0.0),
            Node3D(4, 0.0, 3.0, 0.0),
            Node3D(5, 0.0, 0.0, 4.0),
            Node3D(6, 2.0, 0.0, 4.0),
            Node3D(7, 2.0, 3.0, 4.0),
            Node3D(8, 0.0, 3.0, 4.0),
        ]
        elem = Element3D(
            id=1,
            node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            type="Hex8",
            props={"E": 210.0, "nu": 0.3},
        )
        return HexMesh3D(nodes=nodes, elements=[elem])

    def make_tet4_mesh(self):
        return TetMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 0.0, 1.0, 0.0),
                Node3D(4, 0.0, 0.0, 1.0),
            ],
            elements=[
                Element3D(
                    id=1,
                    node_ids=[1, 2, 3, 4],
                    type="Tet4",
                    props={"E": 210.0, "nu": 0.3},
                )
            ],
        )

    def make_tet10_mesh(self):
        return TetMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 0.0, 1.0, 0.0),
                Node3D(4, 0.0, 0.0, 1.0),
                Node3D(5, 0.5, 0.0, 0.0),
                Node3D(6, 0.5, 0.5, 0.0),
                Node3D(7, 0.0, 0.5, 0.0),
                Node3D(8, 0.0, 0.0, 0.5),
                Node3D(9, 0.5, 0.0, 0.5),
                Node3D(10, 0.0, 0.5, 0.5),
            ],
            elements=[
                Element3D(
                    id=1,
                    node_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    type="Tet10",
                    props={"E": 210.0, "nu": 0.3},
                )
            ],
        )

    def test_solid_kernels_provide_nodal_stress_matching_post_helpers(self):
        hex8_mesh = self.make_hex8_mesh()
        tet4_mesh = self.make_tet4_mesh()
        tet10_mesh = self.make_tet10_mesh()

        for mesh in (hex8_mesh, tet4_mesh, tet10_mesh):
            elem = mesh.elements[0]
            U = np.linspace(0.01, 0.01 * mesh.num_dofs, mesh.num_dofs)
            node_lookup = {node.id: node for node in mesh.nodes}
            kernel = get_element_kernel(elem.type)

            if elem.type.lower() == "hex8":
                a = 1.0 / np.sqrt(3.0)
                gauss_points = [
                    (-a, -a, -a),
                    (a, -a, -a),
                    (a, a, -a),
                    (-a, a, -a),
                    (-a, -a, a),
                    (a, -a, a),
                    (a, a, a),
                    (-a, a, a),
                ]
                gp_stresses = np.array([
                    kernel.stress_at(mesh, elem, U, *gp, node_lookup)
                    for gp in gauss_points
                ], dtype=float)
                n_gp = np.array([
                    hex8_shape_funcs_grads(*gp)[0]
                    for gp in gauss_points
                ], dtype=float)
                expected = np.linalg.solve(n_gp, gp_stresses)
                node_vals = kernel.nodal_stress(mesh, elem, U, node_lookup)
            elif elem.type.lower() == "tet4":
                stress = kernel.stress_at(mesh, elem, U, 0.25, 0.25, 0.25, node_lookup)
                expected = np.tile(stress, (4, 1))
                node_vals = kernel.nodal_stress(mesh, elem, U, node_lookup)
            else:
                gauss_coords = [(xi, eta, zeta) for xi, eta, zeta, _ in tet10_gauss_points()]
                gp_stresses = np.array([
                    kernel.stress_at(mesh, elem, U, *coords, node_lookup)
                    for coords in gauss_coords
                ], dtype=float)
                a_gp = np.array([[1.0, *coords] for coords in gauss_coords], dtype=float)
                coeffs = np.linalg.solve(a_gp, gp_stresses)
                a_nodes = np.array([
                    [1.0, *coords]
                    for coords in TET10_NATURAL_NODE_COORDS
                ], dtype=float)
                expected = a_nodes @ coeffs
                node_vals = kernel.nodal_stress(mesh, elem, U, node_lookup)

            self.assertTrue(np.allclose(node_vals, expected))


class VtkExportRefactorTests(unittest.TestCase):
    def test_vtk_export_lives_inside_post_package(self):
        import importlib
        import sys

        from fem.post.vtk.polar import convert_nodal_displacement

        mesh = PlaneMesh2D(
            nodes=[Node2D(1, 1.0, 0.0), Node2D(2, 0.0, 1.0)],
            elements=[],
        )

        polar = convert_nodal_displacement(
            mesh,
            {
                1: {"ux": 2.0, "uy": 0.0, "rz": 0.5},
                2: {"ux": 0.0, "uy": 3.0, "rz": 0.0},
            },
            [0.0, 0.0],
        )

        self.assertAlmostEqual(polar[1]["ux"], 2.0)
        self.assertAlmostEqual(polar[1]["uy"], 0.0)
        self.assertAlmostEqual(polar[1]["rz"], 0.5)
        self.assertAlmostEqual(polar[2]["ux"], 3.0)
        self.assertAlmostEqual(polar[2]["uy"], 0.0)

        sys.modules.pop("fem.vtk_export", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.vtk_export")
        sys.modules.pop("fem.post.vtk_export", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.post.vtk_export")


class IoPackageRefactorTests(unittest.TestCase):
    def test_io_package_exposes_split_readers_without_legacy_facade(self):
        import importlib
        import sys

        from fem.io import csv as csv_io
        from fem.io import inp, materials as materials_io

        self.assertTrue(callable(materials_io.read))
        self.assertTrue(callable(materials_io.linear_elastic))
        self.assertTrue(callable(csv_io.read_truss2d))
        self.assertTrue(callable(csv_io.read_hex8))
        self.assertTrue(callable(csv_io.read_mixed3d))
        self.assertTrue(callable(inp.read_hex8))
        self.assertTrue(callable(inp.read_tet4))
        self.assertFalse(hasattr(inp, "read_hex8_3d_abaqus"))
        self.assertFalse(hasattr(csv_io, "read_hex8_csv"))

        sys.modules.pop("fem.mesh_io", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.mesh_io")
        for old_module in (
            "fem.io.materials_io",
            "fem.io.mesh_io_csv",
            "fem.io.mesh_io_inp",
        ):
            sys.modules.pop(old_module, None)
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(old_module)

    def test_inp_readers_only_read_mesh_without_material_coupling(self):
        import inspect

        from fem.io import inp

        for reader_name in (
            "read_tri3",
            "read_quad4",
            "read_quad8",
            "read_tet4",
            "read_tet10",
            "read_hex8",
        ):
            signature = inspect.signature(getattr(inp, reader_name))
            self.assertNotIn("material_id", signature.parameters)
            self.assertNotIn("material_path", signature.parameters)

        with temporary_directory() as tmp:
            mesh_path = write_inp(
                tmp,
                "hex8_mesh_only.inp",
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8",
                    "1, 1,2,3,4,5,6,7,8",
                ],
            )
            mesh = inp.read_hex8(mesh_path)
        self.assertEqual(mesh.elements[0].props, {})


class AbaqusModelReaderTests(unittest.TestCase):
    def test_abaqus_read_builds_model_with_sets_surfaces_materials_and_steps(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_model.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Heading",
                    "** minimal model reader fixture",
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Nset, nset=FIXED",
                    "1,4,5,8",
                    "*Nset, nset=TIP",
                    "2,3,6,7",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=TIP_FACE",
                    "SOLID, S4",
                    "*Material, name=STEEL",
                    "*Density",
                    "7.85",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=LOAD, nlgeom=NO",
                    "*Static",
                    "1., 1., 1e-05, 1.",
                    "*Boundary",
                    "FIXED, 1, 3, 0.",
                    "*Cload",
                    "TIP, 3, -50.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)

        self.assertEqual(model.name, "test_abaqus_model")
        self.assertEqual(model.mesh.num_nodes, 8)
        self.assertEqual(model.mesh.elements[0].type, "Hex8")
        self.assertNotIn("E", model.mesh.elements[0].props)
        self.assertEqual(model.materials["STEEL"].properties["E"], 210.0)
        self.assertEqual(model.materials["STEEL"].properties["nu"], 0.3)
        self.assertEqual(model.materials["STEEL"].properties["rho"], 7.85)
        self.assertEqual(model.node_sets["FIXED"].node_ids, (1, 4, 5, 8))
        self.assertEqual(model.element_sets["SOLID"].element_ids, (1,))
        self.assertEqual(model.surfaces["TIP_FACE"].faces[0], ElementFace(1, 5, (2, 3, 7, 6)))
        self.assertEqual(model.sections[0], SectionAssignment("SOLID", "STEEL"))
        self.assertEqual(model.steps[0].name, "LOAD")
        self.assertEqual(model.steps[0].boundaries[0], DisplacementConstraint("FIXED", 1, 3, 0.0))
        self.assertEqual(model.steps[0].cloads[0], NodalLoad("TIP", 3, -50.0))
        self.assertFalse(hasattr(model, "boundary"))
        from fem.solvers import static_linear

        bc = static_linear.boundary_for_step(model, "LOAD")
        self.assertEqual(len(bc.prescribed_displacements), 12)
        self.assertAlmostEqual(sum(bc.nodal_forces.values()), -200.0)

    def test_abaqus_read_inherits_initial_boundaries_across_steps(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_initial_steps.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0.",
                    "2, 1., 0.",
                    "*Element, type=CPS3, elset=SOLID",
                    "1, 1,2,1",
                    "*Nset, nset=FIXED",
                    "1",
                    "*Nset, nset=TIP",
                    "2",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Boundary",
                    "FIXED, 1, 2, 0.",
                    "*Step, name=STEP-1",
                    "*Static",
                    "*Cload",
                    "TIP, 1, 10.",
                    "*End Step",
                    "*Step, name=STEP-2",
                    "*Static",
                    "*Cload",
                    "TIP, 2, 20.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)

        self.assertEqual([step.name for step in model.steps], ["Initial", "STEP-1", "STEP-2"])
        from fem.solvers import static_linear

        self.assertEqual(static_linear.get_step(model).name, "STEP-1")
        step2_bc = static_linear.boundary_for_step(model, "STEP-2")
        self.assertEqual(len(step2_bc.prescribed_displacements), 2)
        self.assertAlmostEqual(sum(step2_bc.nodal_forces.values()), 20.0)

    def test_abaqus_read_prefers_assembly_node_set_over_part_set_for_load_targets(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_scoped_node_sets.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Part, name=BLOCK",
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "*Element, type=C3D4, elset=SOLID",
                    "1, 1,2,3,4",
                    "*Nset, nset=LOADSET, generate",
                    "1, 4, 1",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*End Part",
                    "*Assembly, name=Assembly",
                    "*Instance, name=BLOCK-1, part=BLOCK",
                    "*End Instance",
                    "*Nset, nset=LOADSET, instance=BLOCK-1",
                    "2",
                    "*Nset, nset=FIXED, instance=BLOCK-1",
                    "1",
                    "*End Assembly",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Boundary",
                    "FIXED, 1, 3, 0.",
                    "*Cload",
                    "LOADSET, 2, -1000.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)
        from fem.solvers import static_linear

        bc = static_linear.boundary_for_step(model, "LOAD")

        self.assertEqual(model.node_sets["LOADSET"].node_ids, (2,))
        self.assertEqual(len(bc.nodal_forces), 1)
        self.assertAlmostEqual(sum(bc.nodal_forces.values()), -1000.0)

    def test_abaqus_read_converts_dsload_and_dload_pressure_to_surface_tractions(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_surface_loads.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=TIP_FACE",
                    "SOLID, S4",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "TIP_FACE, P, 2.",
                    "*Dload",
                    "SOLID, P4, 3.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)
        from fem.solvers import static_linear

        bc = static_linear.boundary_for_step(model, "LOAD")

        self.assertEqual(len(model.steps[0].surface_loads), 2)
        self.assertIn("TIP_FACE", model.surfaces)
        self.assertEqual(len(bc.surface_tractions), 2)
        self.assertTrue(np.allclose(bc.surface_tractions[0].vector, (-2.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(bc.surface_tractions[1].vector, (-3.0, 0.0, 0.0)))

    def test_abaqus_read_projects_trshr_direction_to_surface_tangent(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_trshr_surface_load.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=TOP",
                    "SOLID, S2",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "TOP, TRSHR, 10., 2., 0., 2.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)
        from fem.solvers import static_linear

        bc = static_linear.boundary_for_step(model, "LOAD")

        self.assertEqual(len(model.steps[0].surface_loads), 1)
        self.assertEqual(model.steps[0].surface_loads[0].load_type, "shear_traction")
        self.assertEqual(len(bc.surface_tractions), 1)
        self.assertTrue(np.allclose(bc.surface_tractions[0].vector, (10.0, 0.0, 0.0)))

    def test_abaqus_trshr_rejects_nonplanar_faces(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_trshr_nonplanar_face.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.2",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=WARPED_TOP",
                    "SOLID, S2",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "WARPED_TOP, TRSHR, 10., 1., 0., 0.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)
        from fem.solvers import static_linear

        with self.assertRaisesRegex(ValueError, "non-planar"):
            static_linear.boundary_for_step(model, "LOAD")

    def test_abaqus_read_maps_tetra_face_labels_to_local_faces(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_tet_face_labels.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "*Element, type=C3D4, elset=SOLID",
                    "1, 1,2,3,4",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=FACE_1",
                    "SOLID, S1",
                    "*Surface, type=ELEMENT, name=FACE_2",
                    "SOLID, S2",
                    "*Surface, type=ELEMENT, name=FACE_3",
                    "SOLID, S3",
                    "*Surface, type=ELEMENT, name=FACE_4",
                    "SOLID, S4",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)

        self.assertEqual(model.surfaces["FACE_1"].faces[0], ElementFace(1, 3, (1, 2, 3)))
        self.assertEqual(model.surfaces["FACE_2"].faces[0], ElementFace(1, 2, (1, 2, 4)))
        self.assertEqual(model.surfaces["FACE_3"].faces[0], ElementFace(1, 0, (2, 3, 4)))
        self.assertEqual(model.surfaces["FACE_4"].faces[0], ElementFace(1, 1, (1, 3, 4)))

    def test_abaqus_read_maps_hex_face_labels_to_local_faces(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_hex_face_labels.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=FACE_1",
                    "SOLID, S1",
                    "*Surface, type=ELEMENT, name=FACE_2",
                    "SOLID, S2",
                    "*Surface, type=ELEMENT, name=FACE_3",
                    "SOLID, S3",
                    "*Surface, type=ELEMENT, name=FACE_4",
                    "SOLID, S4",
                    "*Surface, type=ELEMENT, name=FACE_5",
                    "SOLID, S5",
                    "*Surface, type=ELEMENT, name=FACE_6",
                    "SOLID, S6",
                    "*Surface, type=ELEMENT, name=LOADED",
                    "SOLID, S1",
                    "SOLID, S2",
                    "SOLID, S3",
                    "SOLID, S4",
                    "SOLID, S5",
                    "SOLID, S6",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "LOADED, P, 2.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)

        self.assertEqual(model.surfaces["FACE_1"].faces[0], ElementFace(1, 0, (1, 4, 3, 2)))
        self.assertEqual(model.surfaces["FACE_2"].faces[0], ElementFace(1, 1, (5, 6, 7, 8)))
        self.assertEqual(model.surfaces["FACE_3"].faces[0], ElementFace(1, 2, (1, 2, 6, 5)))
        self.assertEqual(model.surfaces["FACE_4"].faces[0], ElementFace(1, 5, (2, 3, 7, 6)))
        self.assertEqual(model.surfaces["FACE_5"].faces[0], ElementFace(1, 3, (3, 4, 8, 7)))
        self.assertEqual(model.surfaces["FACE_6"].faces[0], ElementFace(1, 4, (1, 5, 8, 4)))

        from fem.solvers import static_linear

        bc = static_linear.boundary_for_step(model, "LOAD")
        node_lookup = {node.id: node for node in model.mesh.nodes}
        elem = model.mesh.elements[0]
        elem_xyz = np.array(
            [[node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
             for node_id in elem.node_ids],
            dtype=float,
        )
        elem_center = elem_xyz.mean(axis=0)

        for face, traction in zip(model.surfaces["LOADED"].faces, bc.surface_tractions):
            face_xyz = np.array(
                [[node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
                 for node_id in face.node_ids],
                dtype=float,
            )
            inward = elem_center - face_xyz.mean(axis=0)

            self.assertGreater(np.dot(np.array(traction.vector), inward), 0.0)

    def test_abaqus_read_maps_tet10_face_labels_and_pressure_direction(self):
        from fem import abaqus
        from fem.solvers import static_linear

        path = output_path_for_test("test_abaqus_tet10_face_labels.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "5, 0.5, 0., 0.",
                    "6, 0.5, 0.5, 0.",
                    "7, 0., 0.5, 0.",
                    "8, 0., 0., 0.5",
                    "9, 0.5, 0., 0.5",
                    "10, 0., 0.5, 0.5",
                    "*Element, type=C3D10, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8,9,10",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=FACE_1",
                    "SOLID, S1",
                    "*Surface, type=ELEMENT, name=FACE_2",
                    "SOLID, S2",
                    "*Surface, type=ELEMENT, name=FACE_3",
                    "SOLID, S3",
                    "*Surface, type=ELEMENT, name=FACE_4",
                    "SOLID, S4",
                    "*Surface, type=ELEMENT, name=LOADED",
                    "SOLID, S1",
                    "SOLID, S2",
                    "SOLID, S3",
                    "SOLID, S4",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "LOADED, P, 2.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)

        self.assertEqual(model.surfaces["FACE_1"].faces[0], ElementFace(1, 3, (1, 2, 3, 5, 6, 7)))
        self.assertEqual(model.surfaces["FACE_2"].faces[0], ElementFace(1, 2, (1, 2, 4, 5, 9, 8)))
        self.assertEqual(model.surfaces["FACE_3"].faces[0], ElementFace(1, 0, (2, 3, 4, 6, 10, 9)))
        self.assertEqual(model.surfaces["FACE_4"].faces[0], ElementFace(1, 1, (1, 3, 4, 7, 10, 8)))

        bc = static_linear.boundary_for_step(model, "LOAD")
        node_lookup = {node.id: node for node in model.mesh.nodes}
        elem = model.mesh.elements[0]
        elem_xyz = np.array(
            [[node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
             for node_id in elem.node_ids],
            dtype=float,
        )
        elem_center = elem_xyz.mean(axis=0)

        for face, traction in zip(model.surfaces["LOADED"].faces, bc.surface_tractions):
            face_xyz = np.array(
                [[node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
                 for node_id in face.node_ids],
                dtype=float,
            )
            inward = elem_center - face_xyz.mean(axis=0)

            self.assertGreater(np.dot(np.array(traction.vector), inward), 0.0)

    def test_abaqus_pressure_points_into_tetra_element_for_all_faces(self):
        from fem import abaqus
        from fem.solvers import static_linear

        path = output_path_for_test("test_abaqus_tet_pressure_direction.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "*Element, type=C3D4, elset=SOLID",
                    "1, 1,2,3,4",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=LOADED",
                    "SOLID, S1",
                    "SOLID, S2",
                    "SOLID, S3",
                    "SOLID, S4",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "LOADED, P, 2.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)
        bc = static_linear.boundary_for_step(model, "LOAD")
        node_lookup = {node.id: node for node in model.mesh.nodes}
        elem = model.mesh.elements[0]
        elem_xyz = np.array(
            [[node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
             for node_id in elem.node_ids],
            dtype=float,
        )
        elem_center = elem_xyz.mean(axis=0)

        for face, traction in zip(model.surfaces["LOADED"].faces, bc.surface_tractions):
            face_xyz = np.array(
                [[node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
                 for node_id in face.node_ids],
                dtype=float,
            )
            inward = elem_center - face_xyz.mean(axis=0)

            self.assertGreater(np.dot(np.array(traction.vector), inward), 0.0)

    def test_abaqus_read_accumulates_repeated_sets_and_scales_trvec_loads(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_sets_trvec.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Nset, nset=FIXED",
                    "1,4",
                    "*Nset, nset=FIXED",
                    "5,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=TIP_FACE",
                    "SOLID, S6",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "TIP_FACE, TRVEC, 10., 0., 0., -2.",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)
        from fem.solvers import static_linear

        bc = static_linear.boundary_for_step(model, "LOAD")

        self.assertEqual(model.node_sets["FIXED"].node_ids, (1, 4, 5, 8))
        self.assertEqual(bc.surface_tractions[0].vector, (0.0, 0.0, -10.0))

    def test_abaqus_read_stores_output_requests_on_steps(self):
        from fem import abaqus

        path = output_path_for_test("test_abaqus_output_requests.inp")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0., 0.",
                    "2, 1., 0.",
                    "*Element, type=CPS3, elset=SOLID",
                    "1, 1,2,1",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=OUTPUT",
                    "*Static",
                    "*Output, field, variable=PRESELECT",
                    "*Node Output",
                    "U, RF",
                    "*Element Output, directions=YES",
                    "S, E",
                    "*Output, history, variable=PRESELECT",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )

        model = abaqus.read(path)
        outputs = model.steps[0].outputs

        self.assertEqual(outputs[0], OutputRequest("field", "preselect", ("PRESELECT",), {"variable": "PRESELECT"}))
        self.assertEqual(outputs[1], OutputRequest("field", "node", ("U", "RF"), {}))
        self.assertEqual(outputs[2], OutputRequest("field", "element", ("S", "E"), {"directions": "YES"}))
        self.assertEqual(outputs[3], OutputRequest("history", "preselect", ("PRESELECT",), {"variable": "PRESELECT"}))


class PostPackageRefactorTests(unittest.TestCase):
    def test_post_package_exposes_submodules_without_function_facade(self):
        import importlib
        import sys

        import fem.post as post
        from fem.post import displacement, path, polar, stress, vtk
        from fem.post.polar import convert_nodal_solution_into_polar_coord

        self.assertTrue(hasattr(post, "__path__"))
        self.assertIs(post.displacement, displacement)
        self.assertIs(post.path, path)
        self.assertIs(post.polar, polar)
        self.assertIs(post.stress, stress)
        self.assertIs(post.vtk, vtk)
        self.assertFalse(hasattr(post, "export_nodal_displacements_csv"))
        self.assertFalse(hasattr(post, "export_hex8_element_stress_csv"))
        self.assertTrue(hasattr(displacement, "__path__"))
        self.assertTrue(callable(displacement.export.nodal))
        self.assertFalse(hasattr(displacement, "export_nodal_displacement"))
        self.assertTrue(callable(path.extract_path_data))
        self.assertTrue(callable(convert_nodal_solution_into_polar_coord))
        self.assertTrue(hasattr(stress, "__path__"))
        self.assertTrue(callable(stress.dispatch.resolve_type_key))
        self.assertTrue(callable(stress.element.by_type))
        self.assertTrue(callable(stress.export.element))
        self.assertTrue(callable(stress.export.nodal))
        self.assertTrue(callable(stress.invariants.von_mises_3d))
        self.assertTrue(callable(stress.nodal.by_type))
        self.assertFalse(hasattr(stress, "export_hex8_element_stress"))
        self.assertFalse(hasattr(stress, "_compute_hex8_element_stress_at_point"))
        self.assertTrue(hasattr(vtk, "__path__"))
        self.assertTrue(hasattr(vtk, "cells"))
        self.assertTrue(callable(vtk.export.from_csv))
        self.assertTrue(hasattr(vtk, "fields"))
        self.assertTrue(hasattr(vtk, "polar"))
        self.assertTrue(hasattr(vtk, "writer"))
        self.assertFalse(hasattr(vtk, "export_from_csv_3d"))

        for old_module in (
            "fem.post.displacement_export",
            "fem.post.path_export",
            "fem.post.stress_export",
            "fem.post.vtk_export",
        ):
            sys.modules.pop(old_module, None)
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(old_module)

    def test_stress_export_infers_single_element_type_from_mesh(self):
        import csv
        from fem.post import stress

        mesh = HexMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 1.0, 1.0, 0.0),
                Node3D(4, 0.0, 1.0, 0.0),
                Node3D(5, 0.0, 0.0, 1.0),
                Node3D(6, 1.0, 0.0, 1.0),
                Node3D(7, 1.0, 1.0, 1.0),
                Node3D(8, 0.0, 1.0, 1.0),
            ],
            elements=[
                Element3D(
                    id=1,
                    node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                    type="Hex8",
                    props={"E": 210.0, "nu": 0.3},
                )
            ],
        )

        elem_path = output_path_for_test("test_post_stress_element.csv")
        nodal_path = output_path_for_test("test_post_stress_nodal.csv")

        stress.export.element(mesh, np.zeros(mesh.num_dofs), elem_path)
        stress.export.nodal(mesh, np.zeros(mesh.num_dofs), nodal_path)

        with open(elem_path, "r", encoding="utf-8") as f:
            elem_rows = list(csv.reader(f))
        with open(nodal_path, "r", encoding="utf-8") as f:
            nodal_rows = list(csv.reader(f))

        self.assertEqual(elem_rows[0][0], "elem_id")
        self.assertEqual(len(elem_rows), 2)
        self.assertEqual(nodal_rows[0][0], "source_elem_id")
        self.assertEqual(nodal_rows[0][2], "original_node_id")
        self.assertEqual(len(nodal_rows), 9)

    def test_vtk_export_from_result_materializes_missing_csvs(self):
        from fem.post import vtk

        output_dir = fresh_test_output_dir("test_vtk_from_result")
        mesh = HexMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 1.0, 1.0, 0.0),
                Node3D(4, 0.0, 1.0, 0.0),
                Node3D(5, 0.0, 0.0, 1.0),
                Node3D(6, 1.0, 0.0, 1.0),
                Node3D(7, 1.0, 1.0, 1.0),
                Node3D(8, 0.0, 1.0, 1.0),
            ],
            elements=[
                Element3D(
                    1,
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    "Hex8",
                    {"E": 210.0, "nu": 0.3},
                )
            ],
        )
        model = FEMModel(mesh=mesh, name="vtk_auto")
        result = ModelResult(
            model,
            AnalysisStep("load"),
            np.zeros(mesh.num_dofs),
            np.zeros(mesh.num_dofs),
        )

        vtk.export.from_result(result, output_dir=output_dir)

        self.assertTrue((output_dir / "vtk_auto_nodal_displacement.csv").exists())
        self.assertTrue((output_dir / "vtk_auto_element_stress.csv").exists())
        self.assertTrue((output_dir / "vtk_auto_nodal_stress.csv").exists())
        self.assertTrue((output_dir / "vtk_auto.vtk").exists())

    def test_vtk_export_from_result_overwrites_derived_csvs(self):
        from fem.post import vtk

        output_dir = fresh_test_output_dir("test_vtk_from_result_overwrite")
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh = HexMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 1.0, 1.0, 0.0),
                Node3D(4, 0.0, 1.0, 0.0),
                Node3D(5, 0.0, 0.0, 1.0),
                Node3D(6, 1.0, 0.0, 1.0),
                Node3D(7, 1.0, 1.0, 1.0),
                Node3D(8, 0.0, 1.0, 1.0),
            ],
            elements=[
                Element3D(
                    1,
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    "Hex8",
                    {"E": 210.0, "nu": 0.3},
                )
            ],
        )
        stale_disp = output_dir / "vtk_overwrite_nodal_displacement.csv"
        stale_disp.write_text("node_id,x,y,z,ux,uy,uz\n1,0,0,0,999,999,999\n", encoding="utf-8")
        result = ModelResult(
            FEMModel(mesh=mesh, name="vtk_overwrite"),
            AnalysisStep("load"),
            np.zeros(mesh.num_dofs),
            np.zeros(mesh.num_dofs),
        )

        vtk.export.from_result(result, output_dir=output_dir)

        self.assertNotIn("999", stale_disp.read_text(encoding="utf-8"))

    def test_vtk_export_from_result_skips_unsupported_nodal_stress(self):
        from fem.post import vtk

        output_dir = fresh_test_output_dir("test_vtk_from_result_truss")
        mesh = TrussMesh2D(
            nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
            elements=[Element2D(1, [1, 2], "Truss2D", {"E": 100.0, "area": 1.0})],
        )
        result = ModelResult(
            FEMModel(mesh=mesh, name="vtk_truss"),
            AnalysisStep("load"),
            np.zeros(mesh.num_dofs),
            np.zeros(mesh.num_dofs),
        )

        vtk.export.from_result(result, output_dir=output_dir)

        self.assertTrue((output_dir / "vtk_truss_nodal_displacement.csv").exists())
        self.assertTrue((output_dir / "vtk_truss_element_stress.csv").exists())
        self.assertFalse((output_dir / "vtk_truss_nodal_stress.csv").exists())
        self.assertTrue((output_dir / "vtk_truss.vtk").exists())

    def test_vtk_element_stress_reader_averages_repeated_element_rows(self):
        from fem.post.vtk import fields

        path = output_path_for_test("test_vtk_element_stress_average.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "elem_id,node_id,local_node,sig_x,sig_y,tau_xy,mises_stress\n"
            "1,1,1,1,2,3,4\n"
            "1,2,2,3,4,5,6\n",
            encoding="utf-8",
        )

        fields_by_name = fields.read_element_stress(path)

        self.assertAlmostEqual(fields_by_name["sig_x"][1], 2.0)
        self.assertAlmostEqual(fields_by_name["mises_stress"][1], 5.0)

    def test_direct_post_exports_create_parent_dirs_and_beam_uses_rz(self):
        from fem.post import displacement

        mesh = BeamMesh2D(
            nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
            elements=[Element2D(1, [1, 2], "Beam2D")],
        )
        path = output_path_for_test("nested", "beam_displacement.csv")

        displacement.export.nodal(mesh, np.zeros(mesh.num_dofs), path)

        header = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("rz", header)
        self.assertNotIn("uz", header)


class ElementsPackageRefactorTests(unittest.TestCase):
    def test_elements_use_family_modules_and_registry_module(self):
        import importlib
        import sys

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


class StiffnessModuleRemovalTests(unittest.TestCase):
    def test_stiffness_module_is_removed_in_favor_of_element_kernels(self):
        import importlib
        import sys

        sys.modules.pop("fem.stiffness", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.stiffness")


class ManualWorkflowRefactorTests(unittest.TestCase):
    def test_manual_workflow_uses_materials_steps_and_static_solver(self):
        from fem import steps
        from fem.solvers import static_linear

        mesh = TrussMesh2D(
            nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
            elements=[
                Element2D(
                    1,
                    [1, 2],
                    "Truss2D",
                    {"area": 2.0},
                )
            ],
        )
        model = FEMModel(
            mesh=mesh,
            name="manual_workflow",
            node_sets={
                "fixed": NodeSet("fixed", [1]),
                "loaded": NodeSet("loaded", [2]),
            },
            element_sets={"bar": ElementSet("bar", [1])},
        )

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
