import importlib
import inspect
import sys
import unittest



class IoPackageTests(unittest.TestCase):
    def test_io_package_exposes_split_readers_without_legacy_facade(self):
        from fem.io import csv as csv_io
        from fem.io import inp, materials as materials_io

        self.assertTrue(callable(materials_io.read))
        self.assertTrue(callable(csv_io.read_truss2d))
        self.assertTrue(callable(csv_io.read_hex8))
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

        mesh = inp.read_hex8(r"examples\cantilever_beam_hex8.inp")
        self.assertEqual(mesh.elements[0].props, {})


if __name__ == "__main__":
    unittest.main()
