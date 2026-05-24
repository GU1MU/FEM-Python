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
from fem.post.polar import convert_nodal_solution_into_polar_coord
from fem.post.vtk.polar import convert_nodal_displacement
from tests.helpers.mesh_builders import make_unit_hex8_mesh
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


if __name__ == "__main__":
    unittest.main()
