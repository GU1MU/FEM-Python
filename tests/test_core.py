import importlib
import importlib.util
import sys
import unittest

import numpy as np

from fem import materials, selection
from fem.core import mesh as core_mesh
from fem.core import dof
from fem.core.dof import DofMap
from fem.core.mesh import HexMesh3D, Node2D, Node3D, PlaneMesh2D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementFace,
    ElementSet,
    FEMModel,
    MaterialDefinition,
    NodalLoad,
    NodeSet,
    SectionAssignment,
    Surface,
)
from tests.helpers.mesh_builders import (
    make_dof_order_meshes,
    make_minimal_hex_mesh,
    make_selection_hex_mesh,
    make_selection_mixed_plane_mesh,
    make_selection_quad_mesh,
)


class DofMapCoreTests(unittest.TestCase):
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
        self.assertIs(dof.DofMap, DofMap)
        self.assertFalse(hasattr(dof, "DofManager2D"))
        self.assertFalse(hasattr(dof, "DofManager3D"))

        sys.modules.pop("fem.dof_manager", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.dof_manager")

    def test_meshes_expose_dof_interface_without_dof_manager_access(self):
        self.assertIs(core_mesh.HexMesh3D, HexMesh3D)
        sys.modules.pop("fem.mesh", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.mesh")

        for mesh in make_dof_order_meshes():
            with self.subTest(mesh_type=type(mesh).__name__):
                self.assertTrue(hasattr(mesh, "dof_map"))
                self.assertFalse(hasattr(mesh, "dof_manager"))
                self.assertEqual(mesh.node_ids, [10, 20])
                self.assertEqual(mesh.node_dofs(10), list(range(mesh.dofs_per_node)))
                self.assertEqual(
                    mesh.element_dofs(mesh.elements[0])[:mesh.dofs_per_node],
                    mesh.node_dofs(20),
                )

    def test_core_model_stores_sets_surfaces_materials_and_sections(self):
        mesh = make_minimal_hex_mesh()
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
        step = AnalysisStep(
            "load",
            procedure="static",
            boundaries=[DisplacementConstraint("FIXED", 1, 3, 0.0)],
            cloads=[NodalLoad("TIP", 3, -100.0)],
            metadata={"nlgeom": "NO"},
        )
        model = FEMModel(mesh=make_minimal_hex_mesh(), steps=[step])

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
            with self.subTest(name=name):
                self.assertFalse(hasattr(FEMModel, name))


class SelectionTests(unittest.TestCase):
    def test_selection_package_exposes_nodes_edges_and_faces_only(self):
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
        mesh = make_selection_quad_mesh()

        self.assertEqual(len(selection.edges.boundary(mesh)), 4)
        self.assertEqual(selection.edges.by_x(mesh, 0.0), [(1, 3, [4, 1])])
        self.assertEqual(selection.edges.by_y(mesh, 1.0), [(1, 2, [3, 4])])

    def test_elements_select_by_id_and_type_and_build_sets(self):
        mesh = make_selection_mixed_plane_mesh()

        self.assertEqual(selection.elements.all(mesh), [1, 2])
        self.assertEqual(selection.elements.by_type(mesh, "quad4"), [2])
        self.assertEqual(selection.elements.by_ids(mesh, [2, 3]), [2])
        self.assertEqual(
            selection.elements.set_by_type(mesh, "QUADS", "quad4"),
            ElementSet("QUADS", (2,)),
        )

    def test_faces_select_boundary_faces_by_coordinate(self):
        mesh = make_selection_hex_mesh()

        self.assertEqual(len(selection.faces.boundary(mesh)), 6)
        self.assertEqual(selection.faces.by_z(mesh, 4.0), [(1, 1, [5, 6, 7, 8])])
        self.assertEqual(selection.faces.by_x(mesh, 2.0), [(1, 5, [2, 3, 7, 6])])

    def test_selection_can_build_model_sets_and_surfaces(self):
        mesh = make_selection_hex_mesh()

        fixed = selection.nodes.set_by_x(mesh, "FIXED", 0.0)
        load_surface = selection.faces.surface_by_x(mesh, "LOAD", 2.0)

        self.assertIsInstance(fixed, NodeSet)
        self.assertEqual(fixed.node_ids, (1, 4, 5, 8))
        self.assertIsInstance(load_surface, Surface)
        self.assertEqual(load_surface.faces, (ElementFace(1, 5, (2, 3, 7, 6)),))


class MaterialsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
