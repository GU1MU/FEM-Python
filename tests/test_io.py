import importlib
import inspect
import sys
import unittest

from tests.helpers.file_builders import write_inp
from tests.helpers.paths import temporary_directory


class IoPackageTests(unittest.TestCase):
    def test_io_package_exposes_split_readers_without_legacy_facade(self):
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

    def test_csv_read_mixed3d_reads_hex8_and_tet4_from_sectioned_csv(self):
        from fem.io import csv as csv_io

        with temporary_directory() as tmp:
            mesh_path = write_inp(
                tmp,
                "mixed_hex8_tet4.csv",
                [
                    "# NODES",
                    "node_id,x,y,z",
                    "1,0.0,0.0,0.0",
                    "2,1.0,0.0,0.0",
                    "3,1.0,1.0,0.0",
                    "4,0.0,1.0,0.0",
                    "5,0.0,0.0,1.0",
                    "6,1.0,0.0,1.0",
                    "7,1.0,1.0,1.0",
                    "8,0.0,1.0,1.0",
                    "9,2.0,0.0,0.0",
                    "",
                    "# ELEMENTS",
                    "elem_id,type,node1,node2,node3,node4,node5,node6,node7,node8",
                    "1,Hex8,1,2,3,4,5,6,7,8",
                    "2,Tet4,2,9,3,6,,,,",
                ],
            )

            mesh = csv_io.read_mixed3d(mesh_path)

        self.assertEqual(mesh.num_nodes, 9)
        self.assertEqual(mesh.num_elements, 2)
        self.assertEqual([elem.type for elem in mesh.elements], ["Hex8", "Tet4"])
        self.assertEqual(mesh.elements[0].node_ids, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(mesh.elements[1].node_ids, [2, 9, 3, 6])
        self.assertEqual(mesh.dofs_per_node, 3)

    def test_material_csv_builds_named_linear_elastic_material(self):
        from fem.io import materials as materials_io

        with temporary_directory() as tmp:
            material_path = write_inp(
                tmp,
                "materials.csv",
                [
                    "material_id,name,E,rho,nu",
                    "1,steel,220e3,7800,0.3",
                    "2,aluminum,70e3,2700,0.33",
                ],
            )

            steel = materials_io.linear_elastic(material_path, "steel")
            aluminum = materials_io.linear_elastic(material_path, "aluminum")

        self.assertEqual(steel.name, "steel")
        self.assertEqual(steel.properties["E"], 220000.0)
        self.assertEqual(steel.properties["rho"], 7800.0)
        self.assertEqual(steel.properties["nu"], 0.3)
        self.assertEqual(aluminum.name, "aluminum")
        self.assertEqual(aluminum.properties["E"], 70000.0)
        self.assertEqual(aluminum.properties["rho"], 2700.0)
        self.assertEqual(aluminum.properties["nu"], 0.33)


if __name__ == "__main__":
    unittest.main()
