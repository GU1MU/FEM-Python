import csv
import importlib
import sys
from pathlib import Path
import unittest

import numpy as np

import fem.post as post
from fem.core.mesh import BeamMesh2D, Element2D, Element3D, Node2D, Node3D, PlaneMesh2D, TetMesh3D
from fem.core.model import ElementSet, FEMModel, MaterialDefinition, SectionAssignment
from fem.post import displacement, path, polar, stress, vtk
from fem.post.stress import dispatch
from fem.post.stress.averaging import (
    ElementNodalContribution,
    RegionKey,
    StressAveragingPolicy,
    average_solid_nodal_contributions,
)
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
from tests.helpers.paths import temporary_directory
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

        with temporary_directory() as tmp:
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
        self.assertEqual(nodal_rows[0][0], "source_elem_id")
        self.assertEqual(nodal_rows[0][2], "original_node_id")
        self.assertEqual(len(nodal_rows), 9)

    def test_vtk_export_from_result_materializes_missing_csvs(self):
        result = make_zero_result(make_unit_hex8_mesh(), "vtk_auto")

        with temporary_directory() as tmp:
            output_dir = Path(tmp)

            vtk.export.from_result(result, output_dir=output_dir)

            self.assertTrue((output_dir / "vtk_auto_nodal_displacement.csv").exists())
            self.assertTrue((output_dir / "vtk_auto_element_stress.csv").exists())
            self.assertTrue((output_dir / "vtk_auto_nodal_stress.csv").exists())
            self.assertTrue((output_dir / "vtk_auto.vtk").exists())

    def test_vtk_export_from_result_overwrites_derived_csvs(self):
        result = make_zero_result(make_unit_hex8_mesh(), "vtk_overwrite")

        with temporary_directory() as tmp:
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

        with temporary_directory() as tmp:
            output_dir = Path(tmp)

            vtk.export.from_result(result, output_dir=output_dir)

            self.assertTrue((output_dir / "vtk_truss_nodal_displacement.csv").exists())
            self.assertTrue((output_dir / "vtk_truss_element_stress.csv").exists())
            self.assertFalse((output_dir / "vtk_truss_nodal_stress.csv").exists())
            self.assertTrue((output_dir / "vtk_truss.vtk").exists())

    def test_vtk_element_stress_reader_averages_repeated_element_rows(self):
        from fem.post.vtk import fields

        with temporary_directory() as tmp:
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

        with temporary_directory() as tmp:
            output_path = Path(tmp) / "nested" / "beam_displacement.csv"

            displacement.export.nodal(mesh, np.zeros(mesh.num_dofs), output_path)

            header = output_path.read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("rz", header)
        self.assertNotIn("uz", header)


class StressAveragingTests(unittest.TestCase):
    def _contribution(
        self,
        elem_id,
        stress,
        *,
        node_id=10,
        local_node=1,
        material="steel",
        section="solid",
        element_type="hex8",
    ):
        return ElementNodalContribution(
            source_elem_id=elem_id,
            source_local_node=local_node,
            original_node_id=node_id,
            region_key=RegionKey(material, section, element_type),
            x=1.0,
            y=2.0,
            z=3.0,
            stress=stress,
        )

    def test_averaging_keeps_same_region_below_mises_threshold_in_one_cluster(self):
        rows = average_solid_nodal_contributions(
            [
                self._contribution(1, (100.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                self._contribution(2, (110.0, 10.0, 0.0, 0.0, 0.0, 0.0), local_node=2),
                self._contribution(
                    3,
                    (1000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    node_id=99,
                ),
            ],
            StressAveragingPolicy(threshold=75.0),
        )

        node_rows = [row for row in rows if row.original_node_id == 10]
        self.assertEqual({row.cluster_id for row in node_rows}, {0})
        self.assertEqual(len(node_rows), 2)
        for row in node_rows:
            self.assertEqual(row.original_node_id, 10)
            self.assertEqual(row.region_id, 1)
            self.assertEqual(row.stress, (105.0, 5.0, 0.0, 0.0, 0.0, 0.0))

    def test_averaging_splits_same_region_when_mises_threshold_is_exceeded(self):
        rows = average_solid_nodal_contributions(
            [
                self._contribution(1, (100.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                self._contribution(2, (105.0, 0.0, 0.0, 0.0, 0.0, 0.0), local_node=2),
                self._contribution(3, (500.0, 0.0, 0.0, 0.0, 0.0, 0.0), local_node=3),
            ],
            StressAveragingPolicy(threshold=75.0),
        )

        self.assertEqual([row.cluster_id for row in rows], [0, 1, 2])
        self.assertEqual(rows[0].stress[0], 100.0)
        self.assertEqual(rows[1].stress[0], 105.0)
        self.assertEqual(rows[2].stress[0], 500.0)

    def test_averaging_uses_region_mises_range_for_threshold(self):
        rows = average_solid_nodal_contributions(
            [
                self._contribution(1, (452.3077, 0.0, 0.0, 0.0, 0.0, 0.0)),
                self._contribution(
                    2,
                    (735.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    local_node=2,
                ),
                self._contribution(
                    3,
                    (662.3077, 0.0, 0.0, 0.0, 0.0, 0.0),
                    local_node=3,
                ),
                self._contribution(
                    4,
                    (759.2308, 0.0, 0.0, 0.0, 0.0, 0.0),
                    local_node=4,
                ),
                self._contribution(
                    5,
                    (476.5385, 0.0, 0.0, 0.0, 0.0, 0.0),
                    local_node=5,
                ),
                self._contribution(
                    6,
                    (549.2308, 0.0, 0.0, 0.0, 0.0, 0.0),
                    local_node=6,
                ),
            ],
            StressAveragingPolicy(threshold=75.0),
        )

        self.assertEqual([row.cluster_id for row in rows], [0, 1, 2, 3, 4, 5])
        self.assertEqual([round(row.stress[0], 4) for row in rows], [
            452.3077,
            735.0,
            662.3077,
            759.2308,
            476.5385,
            549.2308,
        ])

    def test_averaging_does_not_cross_material_regions(self):
        rows = average_solid_nodal_contributions(
            [
                self._contribution(1, (100.0, 0.0, 0.0, 0.0, 0.0, 0.0), material="steel"),
                self._contribution(
                    2,
                    (110.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    local_node=2,
                    material="aluminum",
                ),
            ],
        )

        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0].region_id, rows[1].region_id)
        self.assertEqual([row.stress[0] for row in rows], [110.0, 100.0])

    def test_averaging_does_not_cross_element_type_regions(self):
        rows = average_solid_nodal_contributions(
            [
                self._contribution(1, (100.0, 0.0, 0.0, 0.0, 0.0, 0.0), element_type="hex8"),
                self._contribution(
                    2,
                    (110.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    local_node=2,
                    element_type="tet4",
                ),
            ],
        )

        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0].region_id, rows[1].region_id)
        self.assertEqual(sorted(row.element_type_id for row in rows), [1, 2])


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

        with temporary_directory() as tmp:
            csv_path = Path(tmp) / "mixed_element_stress.csv"
            stress.export.element(mesh, np.zeros(mesh.num_dofs), csv_path)
            with csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0][0], "elem_id")
        self.assertEqual(len(rows), len(mesh.elements) + 1)
        self.assertEqual([row[0] for row in rows[1:]], ["1", "2"])

    def test_nodal_stress_export_writes_mixed_solid_nodes(self):
        mesh = make_mixed_hex8_tet4_mesh()

        with temporary_directory() as tmp:
            csv_path = Path(tmp) / "mixed_nodal_stress.csv"
            stress.export.nodal(mesh, np.zeros(mesh.num_dofs), csv_path)
            with csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0][0], "source_elem_id")
        self.assertEqual(rows[0][1], "source_local_node")
        self.assertEqual(rows[0][2], "original_node_id")
        self.assertEqual(len(rows), 13)
        self.assertEqual({row[2] for row in rows[1:]}, {str(node.id) for node in mesh.nodes})
        self.assertEqual([row[0] for row in rows[1:]].count("1"), 8)
        self.assertEqual([row[0] for row in rows[1:]].count("2"), 4)

    def test_nodal_stress_export_does_not_average_across_model_sections(self):
        mesh = TetMesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 0.0, 1.0, 0.0),
                Node3D(4, 0.0, 0.0, 1.0),
                Node3D(5, 2.0, 0.0, 0.0),
                Node3D(6, 0.0, 2.0, 0.0),
                Node3D(7, 0.0, 0.0, 2.0),
            ],
            elements=[
                Element3D(1, [1, 2, 3, 4], "Tet4", {"E": 210.0, "nu": 0.3}),
                Element3D(2, [1, 5, 6, 7], "Tet4", {"E": 210.0, "nu": 0.3}),
            ],
        )
        model = FEMModel(
            mesh=mesh,
            element_sets={
                "section_a": ElementSet("section_a", [1]),
                "section_b": ElementSet("section_b", [2]),
            },
            materials={"steel": MaterialDefinition("steel", {"E": 210.0, "nu": 0.3})},
            sections=[
                SectionAssignment("section_a", "steel", "solid", {"name": "A"}),
                SectionAssignment("section_b", "steel", "solid", {"name": "B"}),
            ],
        )

        with temporary_directory() as tmp:
            csv_path = Path(tmp) / "section_nodal_stress.csv"
            stress.export.nodal(mesh, np.zeros(mesh.num_dofs), csv_path, model=model)
            with csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        shared_rows = [row for row in rows if row["original_node_id"] == "1"]
        self.assertEqual(len(shared_rows), 2)
        self.assertEqual({row["material_id"] for row in shared_rows}, {"1"})
        self.assertEqual(len({row["section_id"] for row in shared_rows}), 2)
        self.assertEqual(len({row["region_id"] for row in shared_rows}), 2)

    def test_stress_exports_write_mixed_plane_rows_and_nodes(self):
        mesh = make_mixed_tri3_quad4_mesh()

        with temporary_directory() as tmp:
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

        with temporary_directory() as tmp:
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
        mesh = make_mixed_hex8_tet4_mesh()
        result = make_zero_result(mesh, "mixed_vtk")

        with temporary_directory() as tmp:
            output_dir = Path(tmp)
            vtk.export.from_result(result, output_dir=output_dir)

            self.assertTrue((output_dir / "mixed_vtk_nodal_displacement.csv").exists())
            self.assertTrue((output_dir / "mixed_vtk_element_stress.csv").exists())
            self.assertTrue((output_dir / "mixed_vtk_nodal_stress.csv").exists())
            vtk_text = (output_dir / "mixed_vtk.vtk").read_text(encoding="utf-8")
            region_rows = vtk.fields.read_region_nodal_stress(
                output_dir / "mixed_vtk_nodal_stress.csv"
            )
            _, vtk_cells, _, _ = vtk.cells.build_region_aware(mesh, region_rows)

        self.assertIn("CELL_TYPES 2", vtk_text)
        self.assertIn("POINTS 12 float", vtk_text)
        self.assertIn("VECTORS displacement float", vtk_text)
        self.assertIn("SCALARS sig_x float 1", vtk_text)
        self.assertIn("SCALARS mises float 1", vtk_text)
        self.assertNotIn("SCALARS original_node_id", vtk_text)
        self.assertNotIn("SCALARS region_id", vtk_text)
        self.assertNotIn("SCALARS cluster_id", vtk_text)
        self.assertNotIn("SCALARS original_element_id", vtk_text)
        self.assertIn("\n12\n", vtk_text)
        self.assertIn("\n10\n", vtk_text)
        self.assertNotEqual(vtk_cells[0][2], vtk_cells[1][1])

    def test_vtk_cells_report_unsupported_element_type(self):
        mesh = make_mixed_hex8_tet4_mesh()
        mesh.elements[1].type = "UnsupportedSolid"
        result = make_zero_result(mesh, "unsupported_vtk")

        with temporary_directory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsupported element type for VTK export: UnsupportedSolid"):
                vtk.export.from_result(result, output_dir=Path(tmp))


if __name__ == "__main__":
    unittest.main()
