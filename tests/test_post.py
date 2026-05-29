import csv
import importlib
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

import fem.post as post
from fem.core.mesh import BeamMesh2D, Element2D, Node2D, PlaneMesh2D
from fem.post import displacement, path, polar, stress, vtk
from fem.post.stress import dispatch
from fem.post.polar import convert_nodal_solution_into_polar_coord
from fem.post.vtk.polar import convert_nodal_displacement
from tests.helpers.mesh_builders import (
    make_mixed_hex8_tet4_mesh,
    make_mixed_quad4_quad8_mesh,
    make_mixed_tet4_tet10_mesh,
    make_mixed_tri3_quad4_mesh,
    make_unit_hex8_mesh,
)
from tests.helpers.model_builders import make_simple_truss_mesh
from tests.helpers.result_builders import make_zero_result


class VtkExportTests(unittest.TestCase):
    def test_vtk_export_lives_inside_post_package(self):
        mesh = PlaneMesh2D(
            nodes=[Node2D(1, 1.0, 0.0), Node2D(2, 0.0, 1.0)],
            elements=[],
        )

        polar_values = convert_nodal_displacement(
            mesh,
            {
                1: {"ux": 2.0, "uy": 0.0, "rz": 0.5},
                2: {"ux": 0.0, "uy": 3.0, "rz": 0.0},
            },
            [0.0, 0.0],
        )

        self.assertAlmostEqual(polar_values[1]["ux"], 2.0)
        self.assertAlmostEqual(polar_values[1]["uy"], 0.0)
        self.assertAlmostEqual(polar_values[1]["rz"], 0.5)
        self.assertAlmostEqual(polar_values[2]["ux"], 3.0)
        self.assertAlmostEqual(polar_values[2]["uy"], 0.0)

        sys.modules.pop("fem.vtk_export", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.vtk_export")
        sys.modules.pop("fem.post.vtk_export", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fem.post.vtk_export")


class PostPackageTests(unittest.TestCase):
    def test_post_package_exposes_submodules_without_function_facade(self):
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
        mesh = make_unit_hex8_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            elem_path = output_dir / "test_post_stress_element.csv"
            nodal_path = output_dir / "test_post_stress_nodal.csv"

            stress.export.element(mesh, np.zeros(mesh.num_dofs), elem_path)
            stress.export.nodal(mesh, np.zeros(mesh.num_dofs), nodal_path)

            with elem_path.open("r", encoding="utf-8") as f:
                elem_rows = list(csv.reader(f))
            with nodal_path.open("r", encoding="utf-8") as f:
                nodal_rows = list(csv.reader(f))

        self.assertEqual(elem_rows[0][0], "elem_id")
        self.assertEqual(len(elem_rows), 2)
        self.assertEqual(nodal_rows[0][0], "node_id")
        self.assertEqual(len(nodal_rows), 9)

    def test_vtk_export_from_result_materializes_missing_csvs(self):
        result = make_zero_result(make_unit_hex8_mesh(), "vtk_auto")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            vtk.export.from_result(result, output_dir=output_dir)

            self.assertTrue((output_dir / "vtk_auto_nodal_displacement.csv").exists())
            self.assertTrue((output_dir / "vtk_auto_element_stress.csv").exists())
            self.assertTrue((output_dir / "vtk_auto_nodal_stress.csv").exists())
            self.assertTrue((output_dir / "vtk_auto.vtk").exists())

    def test_vtk_export_from_result_overwrites_derived_csvs(self):
        result = make_zero_result(make_unit_hex8_mesh(), "vtk_overwrite")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stale_disp = output_dir / "vtk_overwrite_nodal_displacement.csv"
            stale_disp.write_text(
                "node_id,x,y,z,ux,uy,uz\n1,0,0,0,999,999,999\n",
                encoding="utf-8",
            )

            vtk.export.from_result(result, output_dir=output_dir)

            self.assertNotIn("999", stale_disp.read_text(encoding="utf-8"))

    def test_vtk_export_from_result_skips_unsupported_nodal_stress(self):
        result = make_zero_result(make_simple_truss_mesh(E=100.0, area=1.0), "vtk_truss")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            vtk.export.from_result(result, output_dir=output_dir)

            self.assertTrue((output_dir / "vtk_truss_nodal_displacement.csv").exists())
            self.assertTrue((output_dir / "vtk_truss_element_stress.csv").exists())
            self.assertFalse((output_dir / "vtk_truss_nodal_stress.csv").exists())
            self.assertTrue((output_dir / "vtk_truss.vtk").exists())

    def test_vtk_element_stress_reader_averages_repeated_element_rows(self):
        from fem.post.vtk import fields

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "test_vtk_element_stress_average.csv"
            csv_path.write_text(
                "elem_id,node_id,local_node,sig_x,sig_y,tau_xy,mises_stress\n"
                "1,1,1,1,2,3,4\n"
                "1,2,2,3,4,5,6\n",
                encoding="utf-8",
            )

            fields_by_name = fields.read_element_stress(csv_path)

        self.assertAlmostEqual(fields_by_name["sig_x"][1], 2.0)
        self.assertAlmostEqual(fields_by_name["mises_stress"][1], 5.0)

    def test_direct_post_exports_create_parent_dirs_and_beam_uses_rz(self):
        mesh = BeamMesh2D(
            nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
            elements=[Element2D(1, [1, 2], "Beam2D")],
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "nested" / "beam_displacement.csv"

            displacement.export.nodal(mesh, np.zeros(mesh.num_dofs), output_path)

            header = output_path.read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("rz", header)
        self.assertNotIn("uz", header)


class MixedStressDispatchTests(unittest.TestCase):
    def test_dispatch_resolves_compatible_mixed_solid_type_keys(self):
        mesh = make_mixed_hex8_tet4_mesh()

        self.assertEqual(dispatch.resolve_type_keys(mesh, None), ("hex8", "tet4"))
        self.assertEqual(dispatch.stress_group_for_keys(("hex8", "tet4")), "solid")
        self.assertTrue(dispatch.element_stress_supported(("hex8", "tet4")))
        self.assertTrue(dispatch.nodal_stress_supported(("hex8", "tet4")))

    def test_dispatch_resolves_compatible_mixed_plane_type_keys(self):
        mesh = make_mixed_tri3_quad4_mesh()

        self.assertEqual(dispatch.resolve_type_keys(mesh, None), ("tri3", "quad4"))
        self.assertEqual(dispatch.stress_group_for_keys(("tri3", "quad4")), "plane")


class MixedStressExportTests(unittest.TestCase):
    def test_element_stress_export_writes_mixed_solid_rows(self):
        mesh = make_mixed_hex8_tet4_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mixed_element_stress.csv"
            stress.export.element(mesh, np.zeros(mesh.num_dofs), csv_path)
            with csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0][0], "elem_id")
        self.assertEqual(len(rows), len(mesh.elements) + 1)
        self.assertEqual([row[0] for row in rows[1:]], ["1", "2"])

    def test_nodal_stress_export_writes_mixed_solid_nodes(self):
        mesh = make_mixed_hex8_tet4_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mixed_nodal_stress.csv"
            stress.export.nodal(mesh, np.zeros(mesh.num_dofs), csv_path)
            with csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0][0], "node_id")
        self.assertEqual(len(rows), len(mesh.nodes) + 1)
        self.assertEqual({row[0] for row in rows[1:]}, {str(node.id) for node in mesh.nodes})

    def test_stress_exports_write_mixed_plane_rows_and_nodes(self):
        mesh = make_mixed_tri3_quad4_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            elem_path = output_dir / "mixed_plane_element_stress.csv"
            nodal_path = output_dir / "mixed_plane_nodal_stress.csv"
            stress.export.element(mesh, np.zeros(mesh.num_dofs), elem_path)
            stress.export.nodal(mesh, np.zeros(mesh.num_dofs), nodal_path)
            with elem_path.open("r", encoding="utf-8") as f:
                elem_rows = list(csv.reader(f))
            with nodal_path.open("r", encoding="utf-8") as f:
                nodal_rows = list(csv.reader(f))

        self.assertEqual(elem_rows[0][0], "elem_id")
        self.assertEqual(len(elem_rows), 8)
        self.assertEqual(nodal_rows[0][0], "node_id")
        self.assertEqual(len(nodal_rows), len(mesh.nodes) + 1)

    def test_stress_exports_cover_higher_order_mixed_types(self):
        solid_mesh = make_mixed_tet4_tet10_mesh()
        plane_mesh = make_mixed_quad4_quad8_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            solid_elem = output_dir / "mixed_tet_element_stress.csv"
            solid_nodal = output_dir / "mixed_tet_nodal_stress.csv"
            solid_vtk = make_zero_result(solid_mesh, "mixed_tet_vtk")
            stress.export.element(solid_mesh, np.zeros(solid_mesh.num_dofs), solid_elem)
            stress.export.nodal(solid_mesh, np.zeros(solid_mesh.num_dofs), solid_nodal)
            vtk.export.from_result(solid_vtk, output_dir=output_dir)

            plane_elem = output_dir / "mixed_quad_element_stress.csv"
            plane_nodal = output_dir / "mixed_quad_nodal_stress.csv"
            stress.export.element(plane_mesh, np.zeros(plane_mesh.num_dofs), plane_elem)
            stress.export.nodal(plane_mesh, np.zeros(plane_mesh.num_dofs), plane_nodal)

            with solid_elem.open("r", encoding="utf-8") as f:
                solid_elem_rows = list(csv.reader(f))
            with solid_nodal.open("r", encoding="utf-8") as f:
                solid_nodal_rows = list(csv.reader(f))
            with plane_elem.open("r", encoding="utf-8") as f:
                plane_elem_rows = list(csv.reader(f))
            with plane_nodal.open("r", encoding="utf-8") as f:
                plane_nodal_rows = list(csv.reader(f))
            vtk_text = (output_dir / "mixed_tet_vtk.vtk").read_text(encoding="utf-8")

        self.assertEqual([row[0] for row in solid_elem_rows[1:]], ["1", "2"])
        self.assertEqual(len(solid_nodal_rows), len(solid_mesh.nodes) + 1)
        self.assertEqual(len(plane_elem_rows), 13)
        self.assertEqual(len(plane_nodal_rows), len(plane_mesh.nodes) + 1)
        self.assertIn("\n10\n", vtk_text)
        self.assertIn("\n24\n", vtk_text)

    def test_vtk_export_from_result_materializes_mixed_stress_csvs(self):
        result = make_zero_result(make_mixed_hex8_tet4_mesh(), "mixed_vtk")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            vtk.export.from_result(result, output_dir=output_dir)

            self.assertTrue((output_dir / "mixed_vtk_nodal_displacement.csv").exists())
            self.assertTrue((output_dir / "mixed_vtk_element_stress.csv").exists())
            self.assertTrue((output_dir / "mixed_vtk_nodal_stress.csv").exists())
            vtk_text = (output_dir / "mixed_vtk.vtk").read_text(encoding="utf-8")

        self.assertIn("CELL_TYPES 2", vtk_text)
        self.assertIn("\n12\n", vtk_text)
        self.assertIn("\n10\n", vtk_text)

    def test_vtk_cells_report_unsupported_element_type(self):
        mesh = make_mixed_hex8_tet4_mesh()
        mesh.elements[1].type = "UnsupportedSolid"
        result = make_zero_result(mesh, "unsupported_vtk")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsupported element type for VTK export: UnsupportedSolid"):
                vtk.export.from_result(result, output_dir=Path(tmp))


if __name__ == "__main__":
    unittest.main()
