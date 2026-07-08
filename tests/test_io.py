import importlib
import inspect
import sys

import pytest

from tests.helpers.file_builders import write_inp


def test_io_package_exposes_split_readers_without_legacy_facade():
    from fem.io import csv as csv_io
    from fem.io import inp, materials as materials_io

    assert callable(materials_io.read)
    assert callable(materials_io.linear_elastic)
    assert callable(csv_io.read_truss2d)
    assert callable(csv_io.read_hex8)
    assert callable(csv_io.read_mixed3d)
    assert callable(inp.read_tri6)
    assert callable(inp.read_mixed2d)
    assert callable(inp.read_hex8)
    assert callable(inp.read_tet4)
    assert not hasattr(inp, "read_hex8_3d_abaqus")
    assert not hasattr(csv_io, "read_hex8_csv")

    sys.modules.pop("fem.mesh_io", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fem.mesh_io")
    for old_module in (
        "fem.io.materials_io",
        "fem.io.mesh_io_csv",
        "fem.io.mesh_io_inp",
    ):
        sys.modules.pop(old_module, None)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old_module)


def test_inp_readers_only_read_mesh_without_material_coupling(tmp_path):
    from fem.io import inp

    for reader_name in (
        "read_tri3",
        "read_tri6",
        "read_quad4",
        "read_quad8",
        "read_mixed2d",
        "read_tet4",
        "read_tet10",
        "read_hex8",
    ):
        signature = inspect.signature(getattr(inp, reader_name))
        assert "material_id" not in signature.parameters
        assert "material_path" not in signature.parameters

    mesh_path = write_inp(
        tmp_path,
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

    assert mesh.elements[0].props == {}


def test_inp_read_tri6_reads_cps6_mesh(tmp_path):
    from fem.io import inp

    mesh_path = write_inp(
        tmp_path,
        "tri6_mesh.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 2., 0.",
            "3, 0., 1.",
            "4, 1., 0.",
            "5, 1., 0.5",
            "6, 0., 0.5",
            "*Element, type=CPS6",
            "1, 1,2,3,4,5,6",
        ],
    )

    mesh = inp.read_tri6(mesh_path)

    assert mesh.dofs_per_node == 2
    assert mesh.elements[0].type == "Tri6Plane"
    assert mesh.elements[0].node_ids == [1, 2, 3, 4, 5, 6]
    assert mesh.elements[0].props["plane_type"] == "stress"
    assert mesh.elements[0].props["thickness"] == 1.0


def test_inp_read_mixed2d_reads_linear_tri3_and_quad4(tmp_path):
    from fem.io import inp

    mesh_path = write_inp(
        tmp_path,
        "mixed_linear_2d.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 1., 1.",
            "4, 0., 1.",
            "5, 2., 0.",
            "*Element, type=CPS3",
            "1, 1,2,4",
            "*Element, type=CPS4",
            "2, 2,5,3,4",
        ],
    )

    mesh = inp.read_mixed2d(mesh_path)

    assert [elem.type for elem in mesh.elements] == ["Tri3Plane", "Quad4Plane"]
    assert [elem.node_ids for elem in mesh.elements] == [[1, 2, 4], [2, 5, 3, 4]]
    assert all(elem.props["plane_type"] == "stress" for elem in mesh.elements)


def test_inp_read_mixed2d_reads_quadratic_tri6_and_quad8(tmp_path):
    from fem.io import inp

    mesh_path = write_inp(
        tmp_path,
        "mixed_quadratic_2d.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 2., 0.",
            "3, 0., 1.",
            "4, 1., 0.",
            "5, 1., 0.5",
            "6, 0., 0.5",
            "7, 3., 0.",
            "8, 5., 0.",
            "9, 5., 2.",
            "10, 3., 2.",
            "11, 4., 0.",
            "12, 5., 1.",
            "13, 4., 2.",
            "14, 3., 1.",
            "*Element, type=CPS6",
            "1, 1,2,3,4,5,6",
            "*Element, type=CPS8",
            "2, 7,8,9,10,11,12,13,14",
        ],
    )

    mesh = inp.read_mixed2d(mesh_path)

    assert [elem.type for elem in mesh.elements] == ["Tri6Plane", "Quad8Plane"]
    assert mesh.elements[0].node_ids == [1, 2, 3, 4, 5, 6]
    assert mesh.elements[1].node_ids == [7, 8, 9, 10, 11, 12, 13, 14]


def test_inp_read_mixed2d_rejects_linear_and_quadratic_mix(tmp_path):
    from fem.io import inp

    mesh_path = write_inp(
        tmp_path,
        "mixed_order_2d.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 0., 1.",
            "4, 0.5, 0.",
            "5, 0.5, 0.5",
            "6, 0., 0.5",
            "*Element, type=CPS3",
            "1, 1,2,3",
            "*Element, type=CPS6",
            "2, 1,2,3,4,5,6",
        ],
    )

    with pytest.raises(ValueError, match="same polynomial order"):
        inp.read_mixed2d(mesh_path)


def test_csv_read_mixed3d_reads_hex8_and_tet4_from_sectioned_csv(tmp_path):
    from fem.io import csv as csv_io

    mesh_path = write_inp(
        tmp_path,
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

    assert mesh.num_nodes == 9
    assert mesh.num_elements == 2
    assert [elem.type for elem in mesh.elements] == ["Hex8", "Tet4"]
    assert mesh.elements[0].node_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert mesh.elements[1].node_ids == [2, 9, 3, 6]
    assert mesh.dofs_per_node == 3


def test_material_csv_builds_named_linear_elastic_material(tmp_path):
    from fem.io import materials as materials_io

    material_path = write_inp(
        tmp_path,
        "materials.csv",
        [
            "material_id,name,E,rho,nu",
            "1,steel,220e3,7800,0.3",
            "2,aluminum,70e3,2700,0.33",
        ],
    )

    steel = materials_io.linear_elastic(material_path, "steel")
    aluminum = materials_io.linear_elastic(material_path, "aluminum")

    assert steel.name == "steel"
    assert steel.properties["E"] == 220000.0
    assert steel.properties["rho"] == 7800.0
    assert steel.properties["nu"] == 0.3
    assert aluminum.name == "aluminum"
    assert aluminum.properties["E"] == 70000.0
    assert aluminum.properties["rho"] == 2700.0
    assert aluminum.properties["nu"] == 0.33
