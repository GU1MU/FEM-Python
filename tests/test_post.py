import csv
import importlib
import sys

import numpy as np
import pytest

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
    make_mixed_tri6_quad8_mesh,
    make_tri6_stiffness_mesh,
    make_unit_hex8_mesh,
)
from tests.helpers.model_builders import make_simple_truss_mesh
from tests.helpers.result_builders import make_zero_result


def test_vtk_export_lives_inside_post_package():
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

    assert polar_values[1]["ux"] == pytest.approx(2.0)
    assert polar_values[1]["uy"] == pytest.approx(0.0)
    assert polar_values[1]["rz"] == pytest.approx(0.5)
    assert polar_values[2]["ux"] == pytest.approx(3.0)
    assert polar_values[2]["uy"] == pytest.approx(0.0)

    sys.modules.pop("fem.vtk_export", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fem.vtk_export")
    sys.modules.pop("fem.post.vtk_export", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fem.post.vtk_export")


def test_post_package_exposes_submodules_without_function_facade():
    assert hasattr(post, "__path__")
    assert post.displacement is displacement
    assert post.path is path
    assert post.polar is polar
    assert post.stress is stress
    assert post.vtk is vtk
    assert not hasattr(post, "export_nodal_displacements_csv")
    assert not hasattr(post, "export_hex8_element_stress_csv")
    assert hasattr(displacement, "__path__")
    assert callable(displacement.export.nodal)
    assert not hasattr(displacement, "export_nodal_displacement")
    assert callable(path.extract_path_data)
    assert callable(convert_nodal_solution_into_polar_coord)
    assert hasattr(stress, "__path__")
    assert callable(stress.dispatch.resolve_type_key)
    assert callable(stress.element.by_type)
    assert callable(stress.export.element)
    assert callable(stress.export.nodal)
    assert callable(stress.invariants.von_mises_3d)
    assert callable(stress.nodal.by_type)
    assert not hasattr(stress, "export_hex8_element_stress")
    assert not hasattr(stress, "_compute_hex8_element_stress_at_point")
    assert hasattr(vtk, "__path__")
    assert hasattr(vtk, "cells")
    assert callable(vtk.export.from_csv)
    assert hasattr(vtk, "fields")
    assert hasattr(vtk, "polar")
    assert hasattr(vtk, "writer")
    assert not hasattr(vtk, "export_from_csv_3d")

    for old_module in (
        "fem.post.displacement_export",
        "fem.post.path_export",
        "fem.post.stress_export",
        "fem.post.vtk_export",
    ):
        sys.modules.pop(old_module, None)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old_module)


def test_stress_export_infers_single_element_type_from_mesh(tmp_path):
    mesh = make_unit_hex8_mesh()
    elem_path = tmp_path / "test_post_stress_element.csv"
    nodal_path = tmp_path / "test_post_stress_nodal.csv"

    stress.export.element(mesh, np.zeros(mesh.num_dofs), elem_path)
    stress.export.nodal(mesh, np.zeros(mesh.num_dofs), nodal_path)

    with elem_path.open("r", encoding="utf-8") as f:
        elem_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_rows = list(csv.reader(f))

    assert elem_rows[0][0] == "elem_id"
    assert len(elem_rows) == 2
    assert nodal_rows[0][0] == "node_id"
    assert len(nodal_rows) == 9


def test_vtk_export_from_result_materializes_missing_csvs(tmp_path):
    result = make_zero_result(make_unit_hex8_mesh(), "vtk_auto")

    vtk.export.from_result(result, output_dir=tmp_path)

    assert (tmp_path / "vtk_auto_nodal_displacement.csv").exists()
    assert (tmp_path / "vtk_auto_element_stress.csv").exists()
    assert (tmp_path / "vtk_auto_nodal_stress.csv").exists()
    assert (tmp_path / "vtk_auto.vtk").exists()


def test_vtk_export_from_result_overwrites_derived_csvs(tmp_path):
    result = make_zero_result(make_unit_hex8_mesh(), "vtk_overwrite")
    stale_disp = tmp_path / "vtk_overwrite_nodal_displacement.csv"
    stale_disp.write_text(
        "node_id,x,y,z,ux,uy,uz\n1,0,0,0,999,999,999\n",
        encoding="utf-8",
    )

    vtk.export.from_result(result, output_dir=tmp_path)

    assert "999" not in stale_disp.read_text(encoding="utf-8")


def test_vtk_export_from_result_skips_unsupported_nodal_stress(tmp_path):
    result = make_zero_result(make_simple_truss_mesh(E=100.0, area=1.0), "vtk_truss")

    vtk.export.from_result(result, output_dir=tmp_path)

    assert (tmp_path / "vtk_truss_nodal_displacement.csv").exists()
    assert (tmp_path / "vtk_truss_element_stress.csv").exists()
    assert not (tmp_path / "vtk_truss_nodal_stress.csv").exists()
    assert (tmp_path / "vtk_truss.vtk").exists()


def test_vtk_element_stress_reader_averages_repeated_element_rows(tmp_path):
    from fem.post.vtk import fields

    csv_path = tmp_path / "test_vtk_element_stress_average.csv"
    csv_path.write_text(
        "elem_id,node_id,local_node,sig_x,sig_y,tau_xy,mises_stress\n"
        "1,1,1,1,2,3,4\n"
        "1,2,2,3,4,5,6\n",
        encoding="utf-8",
    )

    fields_by_name = fields.read_element_stress(csv_path)

    assert fields_by_name["sig_x"][1] == pytest.approx(2.0)
    assert fields_by_name["mises_stress"][1] == pytest.approx(5.0)


def test_direct_post_exports_create_parent_dirs_and_beam_uses_rz(tmp_path):
    mesh = BeamMesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
        elements=[Element2D(1, [1, 2], "Beam2D")],
    )
    output_path = tmp_path / "nested" / "beam_displacement.csv"

    displacement.export.nodal(mesh, np.zeros(mesh.num_dofs), output_path)

    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert "rz" in header
    assert "uz" not in header


def test_dispatch_resolves_compatible_mixed_solid_type_keys():
    mesh = make_mixed_hex8_tet4_mesh()

    assert dispatch.resolve_type_keys(mesh, None) == ("hex8", "tet4")
    assert dispatch.stress_group_for_keys(("hex8", "tet4")) == "solid"
    assert dispatch.element_stress_supported(("hex8", "tet4"))
    assert dispatch.nodal_stress_supported(("hex8", "tet4"))


def test_dispatch_resolves_compatible_mixed_plane_type_keys():
    mesh = make_mixed_tri3_quad4_mesh()

    assert dispatch.resolve_type_keys(mesh, None) == ("tri3", "quad4")
    assert dispatch.stress_group_for_keys(("tri3", "quad4")) == "plane"


def test_dispatch_resolves_compatible_mixed_quadratic_plane_type_keys():
    mesh = make_mixed_tri6_quad8_mesh()

    assert dispatch.resolve_type_keys(mesh, None) == ("tri6", "quad8")
    assert dispatch.stress_group_for_keys(("tri6", "quad8")) == "plane"
    assert dispatch.element_stress_supported(("tri6", "quad8"))
    assert dispatch.nodal_stress_supported(("tri6", "quad8"))


def test_element_stress_export_writes_mixed_solid_rows(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    csv_path = tmp_path / "mixed_element_stress.csv"

    stress.export.element(mesh, np.zeros(mesh.num_dofs), csv_path)
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0][0] == "elem_id"
    assert len(rows) == len(mesh.elements) + 1
    assert [row[0] for row in rows[1:]] == ["1", "2"]


def test_nodal_stress_export_writes_mixed_solid_nodes(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    csv_path = tmp_path / "mixed_nodal_stress.csv"

    stress.export.nodal(mesh, np.zeros(mesh.num_dofs), csv_path)
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0][0] == "node_id"
    assert len(rows) == len(mesh.nodes) + 1
    assert {row[0] for row in rows[1:]} == {str(node.id) for node in mesh.nodes}


def test_stress_exports_write_mixed_plane_rows_and_nodes(tmp_path):
    mesh = make_mixed_tri3_quad4_mesh()
    elem_path = tmp_path / "mixed_plane_element_stress.csv"
    nodal_path = tmp_path / "mixed_plane_nodal_stress.csv"

    stress.export.element(mesh, np.zeros(mesh.num_dofs), elem_path)
    stress.export.nodal(mesh, np.zeros(mesh.num_dofs), nodal_path)
    with elem_path.open("r", encoding="utf-8") as f:
        elem_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_rows = list(csv.reader(f))

    assert elem_rows[0][0] == "elem_id"
    assert len(elem_rows) == 8
    assert nodal_rows[0][0] == "node_id"
    assert len(nodal_rows) == len(mesh.nodes) + 1


def test_stress_exports_cover_higher_order_mixed_types(tmp_path):
    solid_mesh = make_mixed_tet4_tet10_mesh()
    plane_mesh = make_mixed_tri6_quad8_mesh()
    solid_elem = tmp_path / "mixed_tet_element_stress.csv"
    solid_nodal = tmp_path / "mixed_tet_nodal_stress.csv"
    solid_vtk = make_zero_result(solid_mesh, "mixed_tet_vtk")
    plane_elem = tmp_path / "mixed_quad_element_stress.csv"
    plane_nodal = tmp_path / "mixed_quad_nodal_stress.csv"

    stress.export.element(solid_mesh, np.zeros(solid_mesh.num_dofs), solid_elem)
    stress.export.nodal(solid_mesh, np.zeros(solid_mesh.num_dofs), solid_nodal)
    vtk.export.from_result(solid_vtk, output_dir=tmp_path)
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
    vtk_text = (tmp_path / "mixed_tet_vtk.vtk").read_text(encoding="utf-8")

    assert [row[0] for row in solid_elem_rows[1:]] == ["1", "2"]
    assert len(solid_nodal_rows) == len(solid_mesh.nodes) + 1
    assert len(plane_elem_rows) == 15
    assert len(plane_nodal_rows) == len(plane_mesh.nodes) + 1
    assert "\n10\n" in vtk_text
    assert "\n24\n" in vtk_text


def test_vtk_cells_support_tri6_quadratic_triangle(tmp_path):
    result = make_zero_result(make_tri6_stiffness_mesh(), "tri6_vtk")

    vtk.export.from_result(result, output_dir=tmp_path)
    vtk_text = (tmp_path / "tri6_vtk.vtk").read_text(encoding="utf-8")

    assert "CELL_TYPES 1" in vtk_text
    assert "\n22\n" in vtk_text


def test_vtk_export_from_result_materializes_mixed_stress_csvs(tmp_path):
    result = make_zero_result(make_mixed_hex8_tet4_mesh(), "mixed_vtk")

    vtk.export.from_result(result, output_dir=tmp_path)

    assert (tmp_path / "mixed_vtk_nodal_displacement.csv").exists()
    assert (tmp_path / "mixed_vtk_element_stress.csv").exists()
    assert (tmp_path / "mixed_vtk_nodal_stress.csv").exists()
    vtk_text = (tmp_path / "mixed_vtk.vtk").read_text(encoding="utf-8")

    assert "CELL_TYPES 2" in vtk_text
    assert "\n12\n" in vtk_text
    assert "\n10\n" in vtk_text


def test_vtk_cells_report_unsupported_element_type(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[1].type = "UnsupportedSolid"
    result = make_zero_result(mesh, "unsupported_vtk")

    with pytest.raises(ValueError, match="Unsupported element type for VTK export: UnsupportedSolid"):
        vtk.export.from_result(result, output_dir=tmp_path)
