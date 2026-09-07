import csv

import numpy as np
import pytest

from fem.core.mesh import (
    Element2D,
    Mesh2D,
    Node2D,
)
from fem.elements import get_element_kernel
from fem.post import (
    displacement,
    vtk,
)
from fem.post.stress import dispatch
from tests.helpers.mesh_builders import (
    make_hex20_stiffness_mesh,
    make_mixed_hex20_tet10_mesh,
    make_mixed_hex8_hex20_mesh,
    make_mixed_hex8_tet4_mesh,
    make_mixed_tet4_tet10_mesh,
    make_mixed_tri3_quad4_mesh,
    make_mixed_tri6_quad8_mesh,
    make_tri6_stiffness_mesh,
    make_unit_hex8_mesh,
)
from tests.helpers.model_builders import make_simple_truss_mesh
from tests.helpers.result_builders import make_zero_result


from tests.helpers.post_builders import (
    _affine_solid_displacement, _write_current_element_stress, _write_current_nodal_stress,
)


def test_vtk_duplicates_points_for_unaveraged_nodal_stress_rows(tmp_path):
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3"),
            Element2D(2, [3, 4, 5], "Tri3"),
        ],
    )
    displacement_path = tmp_path / "displacement.csv"
    nodal_stress_path = tmp_path / "nodal_stress.csv"
    vtk_path = tmp_path / "split.vtk"
    displacement_path.write_text(
        "node_id,x,y,ux,uy\n"
        "1,0,0,0,0\n2,1,0,0,0\n3,0,1,3,4\n4,-1,0,0,0\n5,0,-1,0,0\n",
        encoding="utf-8",
    )
    nodal_stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy,mises\n"
        "1,0,0,1,1,false,0,0,0,0\n"
        "2,1,0,1,2,false,0,0,0,0\n"
        "3,0,1,1,3,false,10,0,0,10\n"
        "3,0,1,2,1,false,20,0,0,20\n"
        "4,-1,0,2,2,false,0,0,0,0\n"
        "5,0,-1,2,3,false,0,0,0,0\n",
        encoding="utf-8",
    )

    vtk.export.from_csv(
        mesh,
        displacement_path,
        None,
        vtk_path,
        nodal_stress_path,
    )

    lines = vtk_path.read_text(encoding="utf-8").splitlines()
    cells_index = lines.index("CELLS 2 8")
    displacement_index = lines.index("VECTORS displacement float")
    sig_x_index = lines.index("SCALARS sig_x float 1")
    assert "POINTS 6 float" in lines
    assert "POINT_DATA 6" in lines
    assert lines[cells_index + 1 : cells_index + 3] == ["3 0 1 2", "3 3 4 5"]
    np.testing.assert_allclose(
        [[float(value) for value in line.split()] for line in lines[displacement_index + 1 : displacement_index + 7]],
        [[0, 0, 0], [0, 0, 0], [3, 4, 0], [3, 4, 0], [0, 0, 0], [0, 0, 0]],
    )
    np.testing.assert_allclose(
        [float(value) for value in lines[sig_x_index + 2 : sig_x_index + 8]],
        [0, 0, 10, 20, 0, 0],
    )


def test_vtk_from_result_uses_threshold_for_csv_and_topology(tmp_path):
    props = {"E": 100.0, "nu": 0.25, "plane_type": "stress", "thickness": 1.0}
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3", dict(props)),
            Element2D(2, [3, 4, 5], "Tri3", dict(props)),
        ],
    )

    vtk.export.from_result(
        make_zero_result(mesh, "threshold_zero"),
        output_dir=tmp_path,
        threshold=0.0,
    )

    stress_path = tmp_path / "threshold_zero_nodal_stress.csv"
    with stress_path.open("r", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    shared_rows = [row for row in csv_rows if row["node_id"] == "3"]
    vtk_lines = (tmp_path / "threshold_zero.vtk").read_text(encoding="utf-8").splitlines()
    assert [(row["elem_id"], row["averaged"]) for row in shared_rows] == [
        ("1", "false"),
        ("2", "false"),
    ]
    assert "POINTS 6 float" in vtk_lines


@pytest.mark.parametrize(
    ("row", "message"),
    (
        (
            vtk.fields.NodalStressCsvRow(1, 1, None, False, {"sig_x": 10.0}),
            r"node 1.*requires.*provenance.*missing local_node",
        ),
        (
            vtk.fields.NodalStressCsvRow(1, 1, 2, False, {"sig_x": 10.0}),
            r"node 1.*element 1.*local node 2.*connectivity",
        ),
    ),
)
def test_vtk_validates_single_nonaveraged_row_provenance(row, message):
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 0.0, 1.0)],
        elements=[Element2D(1, [1, 2, 3], "Tri3")],
    )

    with pytest.raises(ValueError, match=message):
        vtk.cells.build_result(mesh, (row,))


def test_vtk_uses_shared_zero_fallback_for_incident_element_without_raw_row():
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3"),
            Element2D(2, [3, 4, 5], "Tri3"),
        ],
    )
    row = vtk.fields.NodalStressCsvRow(3, 1, 3, False, {"sig_x": 10.0})

    topology = vtk.cells.build_result(mesh, (row,))
    point_fields = vtk.fields.point_fields(
        vtk.fields.NodalStressCsv(("sig_x",), (row,)),
        topology.point_rows,
    )

    selected_point = topology.cells[0][3]
    fallback_point = topology.cells[1][1]
    assert len(topology.cells) == 2
    assert selected_point != fallback_point
    assert topology.point_node_ids[selected_point] == 3
    assert topology.point_node_ids[fallback_point] == 3
    assert topology.point_rows[selected_point] == row
    assert topology.point_rows[fallback_point] is None
    assert point_fields["sig_x"][selected_point] == pytest.approx(10.0)
    assert point_fields["sig_x"][fallback_point] == pytest.approx(0.0)


def test_vtk_uses_single_boundary_raw_row_for_its_element_local_point():
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 0.0, 1.0)],
        elements=[Element2D(1, [1, 2, 3], "Tri3")],
    )
    row = vtk.fields.NodalStressCsvRow(2, 1, 2, False, {"sig_x": 10.0})

    topology = vtk.cells.build_result(mesh, (row,))

    point_index = topology.cells[0][2]
    assert topology.point_node_ids[point_index] == 2
    assert topology.point_rows[point_index] == row


def test_vtk_rejects_repeated_nodal_stress_rows_without_provenance():
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 0.0, 1.0)],
        elements=[Element2D(1, [1, 2, 3], "Tri3")],
    )
    rows = (
        vtk.fields.NodalStressCsvRow(1, 1, 1, False, {"sig_x": 10.0}),
        vtk.fields.NodalStressCsvRow(1, None, None, False, {"sig_x": 20.0}),
    )
    with pytest.raises(ValueError, match="requires.*provenance"):
        vtk.cells.build_result(mesh, rows)


def test_isolated_node_export_uses_averaged_zero_row_and_reaches_vtk(tmp_path):
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, 2.0, 2.0),
        ],
        elements=[
            Element2D(
                1,
                [1, 2, 3],
                "Tri3",
                {"E": 100.0, "nu": 0.25, "thickness": 1.0, "plane_type": "stress"},
            )
        ],
    )

    vtk.export.from_result(
        make_zero_result(mesh, "isolated_node"),
        output_dir=tmp_path,
    )

    data = vtk.fields.read_nodal_stress_rows(
        tmp_path / "isolated_node_nodal_stress.csv"
    )
    isolated_row = next(row for row in data.rows if row.node_id == 4)
    vtk_text = (tmp_path / "isolated_node.vtk").read_text(encoding="utf-8")
    assert isolated_row.elem_id is None
    assert isolated_row.local_node is None
    assert isolated_row.averaged is True
    assert set(isolated_row.values.values()) == {0.0}
    assert "POINTS 4 float" in vtk_text


def test_explicit_hex8_nodal_stress_subset_exports_mixed_mesh_to_vtk(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    U = _affine_solid_displacement(mesh)
    disp_path = tmp_path / "mixed_subset_nodal_displacement.csv"
    stress_path = tmp_path / "mixed_subset_nodal_stress.csv"
    vtk_path = tmp_path / "mixed_subset.vtk"

    displacement.export.nodal(mesh, U, disp_path)
    _write_current_nodal_stress(
        mesh,
        U,
        stress_path,
        element_type="hex8",
    )

    nodal_data = vtk.fields.read_nodal_stress_rows(stress_path)
    topology = vtk.cells.build_result(mesh, nodal_data.rows)
    point_fields = vtk.fields.point_fields(nodal_data, topology.point_rows)
    hex8_point = topology.cells[0][2]
    tet4_fallback_point = topology.cells[1][1]

    assert len(topology.cells) == 2
    assert topology.point_node_ids[hex8_point] == 2
    assert topology.point_node_ids[tet4_fallback_point] == 2
    assert topology.point_rows[hex8_point].elem_id == 1
    assert topology.point_rows[hex8_point].local_node == 2
    assert topology.point_rows[tet4_fallback_point] is None
    assert abs(point_fields["sig_x"][hex8_point]) > 0.0
    assert point_fields["sig_x"][tet4_fallback_point] == pytest.approx(0.0)

    vtk.export.from_csv(
        mesh,
        disp_path,
        None,
        vtk_path,
        stress_path,
    )

    vtk_text = vtk_path.read_text(encoding="utf-8")
    assert "\nCELLS 2 " in vtk_text
    assert "\nCELL_TYPES 2\n" in vtk_text


def test_vtk_cells_support_hex20_in_abaqus_node_order(tmp_path):
    mesh = make_hex20_stiffness_mesh(curved=True)

    vtk_cells, cell_types, elems_for_cell = vtk.cells.build(mesh)

    assert vtk_cells == [[20, *range(20)]]
    assert cell_types == [25]
    assert elems_for_cell == mesh.elements

    result = make_zero_result(mesh, "hex20_vtk")
    vtk.export.from_result(result, output_dir=tmp_path)
    vtk_text = (tmp_path / "hex20_vtk.vtk").read_text(encoding="utf-8")

    assert "20 " + " ".join(str(index) for index in range(20)) in vtk_text
    assert "CELL_TYPES 1" in vtk_text
    assert "\n25\n" in vtk_text


def test_vtk_cells_reject_reduced_integration_hex20_without_type_25():
    element_type = "C3D20R"
    mesh = make_hex20_stiffness_mesh()
    mesh.elements[0].type = element_type

    with pytest.raises(
        ValueError,
        match=rf"Unsupported element type for VTK export: {element_type}",
    ):
        vtk.cells.build(mesh)


@pytest.mark.parametrize(
    ("builder", "expected_types"),
    [(make_unit_hex8_mesh, [12]), (make_mixed_hex8_tet4_mesh, [12, 10])],
    ids=["single-solid", "mixed-solids"],
)
def test_vtk_export_from_result_materializes_missing_csvs(tmp_path, builder, expected_types):
    result = make_zero_result(builder(), "vtk_auto")

    vtk.export.from_result(result, output_dir=tmp_path)

    assert (tmp_path / "vtk_auto_nodal_displacement.csv").exists()
    assert (tmp_path / "vtk_auto_element_stress.csv").exists()
    assert (tmp_path / "vtk_auto_nodal_stress.csv").exists()
    assert (tmp_path / "vtk_auto.vtk").exists()
    lines = (tmp_path / "vtk_auto.vtk").read_text(encoding="utf-8").splitlines()
    type_index = lines.index(f"CELL_TYPES {len(expected_types)}")
    assert [int(value) for value in lines[type_index + 1:type_index + 1 + len(expected_types)]] == expected_types
    with (tmp_path / "vtk_auto_element_stress.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["elem_id"]) for row in rows] == ([1] if len(expected_types) == 1 else [1, 2])


def test_vtk_export_from_result_overwrites_derived_csvs(tmp_path):
    result = make_zero_result(make_unit_hex8_mesh(), "vtk_overwrite")
    stale_disp = tmp_path / "vtk_overwrite_nodal_displacement.csv"
    stale_disp.write_text(
        "node_id,x,y,z,ux,uy,uz\n1,0,0,0,999,999,999\n",
        encoding="utf-8",
    )

    vtk.export.from_result(result, output_dir=tmp_path)

    with stale_disp.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["node_id"]) for row in rows] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert all(float(row[key]) == 0.0 for row in rows for key in ("ux", "uy", "uz"))


def test_vtk_export_from_result_skips_unsupported_nodal_stress(tmp_path):
    result = make_zero_result(make_simple_truss_mesh(E=100.0, area=1.0), "vtk_truss")

    vtk.export.from_result(result, output_dir=tmp_path)

    assert (tmp_path / "vtk_truss_nodal_displacement.csv").exists()
    assert (tmp_path / "vtk_truss_element_stress.csv").exists()
    assert not (tmp_path / "vtk_truss_nodal_stress.csv").exists()
    assert (tmp_path / "vtk_truss.vtk").exists()


def test_stress_exports_cover_higher_order_mixed_types(tmp_path):
    solid_mesh = make_mixed_tet4_tet10_mesh()
    plane_mesh = make_mixed_tri6_quad8_mesh()
    solid_elem = tmp_path / "mixed_tet_element_stress.csv"
    solid_nodal = tmp_path / "mixed_tet_nodal_stress.csv"
    solid_vtk = make_zero_result(solid_mesh, "mixed_tet_vtk")
    plane_elem = tmp_path / "mixed_quad_element_stress.csv"
    plane_nodal = tmp_path / "mixed_quad_nodal_stress.csv"

    _write_current_element_stress(
        solid_mesh,
        np.zeros(solid_mesh.num_dofs),
        solid_elem,
    )
    _write_current_nodal_stress(
        solid_mesh,
        np.zeros(solid_mesh.num_dofs),
        solid_nodal,
    )
    vtk.export.from_result(solid_vtk, output_dir=tmp_path)
    _write_current_element_stress(
        plane_mesh,
        np.zeros(plane_mesh.num_dofs),
        plane_elem,
    )
    _write_current_nodal_stress(
        plane_mesh,
        np.zeros(plane_mesh.num_dofs),
        plane_nodal,
    )

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


@pytest.mark.parametrize(
    ("mesh_builder", "name", "element_rows", "nodal_rows", "cell_types"),
    (
        (make_mixed_hex8_hex20_mesh, "mixed_hex8_hex20", 3, 29, [12, 25]),
        (make_mixed_hex20_tet10_mesh, "mixed_hex20_tet10", 3, 31, [25, 24]),
    ),
)
def test_mixed_hex20_stress_and_vtk_exports_have_exact_rows_and_cell_types(
    tmp_path,
    mesh_builder,
    name,
    element_rows,
    nodal_rows,
    cell_types,
):
    mesh = mesh_builder()
    U = _affine_solid_displacement(mesh)
    element_path = tmp_path / f"{name}_direct_element_stress.csv"
    nodal_path = tmp_path / f"{name}_direct_nodal_stress.csv"

    _write_current_element_stress(mesh, U, element_path)
    _write_current_nodal_stress(mesh, U, nodal_path)
    vtk.export.from_result(make_zero_result(mesh, name), output_dir=tmp_path)

    with element_path.open("r", encoding="utf-8") as f:
        element_stress_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_stress_rows = list(csv.reader(f))
    vtk_lines = (tmp_path / f"{name}.vtk").read_text(encoding="utf-8").splitlines()
    cell_types_index = vtk_lines.index("CELL_TYPES 2")

    assert len(element_stress_rows) == element_rows
    assert len(nodal_stress_rows) == nodal_rows
    assert [int(value) for value in vtk_lines[cell_types_index + 1 : cell_types_index + 3]] == cell_types
    rows_by_node = {int(row[0]): row for row in nodal_stress_rows[1:]}
    hex20_elem = next(
        elem
        for elem in mesh.elements
        if dispatch.type_key_from_name(elem.type) == "hex20"
    )
    expected = get_element_kernel(hex20_elem.type).nodal_stress(mesh, hex20_elem, U)
    for local_index in (0, 8):
        node_id = hex20_elem.node_ids[local_index]
        exported = np.array([float(value) for value in rows_by_node[node_id][7:13]])
        assert not np.allclose(expected[local_index], 0.0)
        assert np.allclose(exported, expected[local_index])


def test_vtk_cells_support_tri6_quadratic_triangle(tmp_path):
    result = make_zero_result(make_tri6_stiffness_mesh(), "tri6_vtk")

    vtk.export.from_result(result, output_dir=tmp_path)
    vtk_text = (tmp_path / "tri6_vtk.vtk").read_text(encoding="utf-8")

    assert "CELL_TYPES 1" in vtk_text
    assert "\n22\n" in vtk_text


def test_vtk_cells_support_mixed_tri3_quad4_topology(tmp_path):
    result = make_zero_result(make_mixed_tri3_quad4_mesh(), "mixed_plane_vtk")

    vtk.export.from_result(result, output_dir=tmp_path)

    vtk_lines = (tmp_path / "mixed_plane_vtk.vtk").read_text(
        encoding="utf-8"
    ).splitlines()
    cell_types_index = vtk_lines.index("CELL_TYPES 2")

    assert [int(value) for value in vtk_lines[cell_types_index + 1 : cell_types_index + 3]] == [
        5,
        9,
    ]


def test_vtk_cells_report_unsupported_element_type(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[1].type = "UnsupportedSolid"
    result = make_zero_result(mesh, "unsupported_vtk")

    with pytest.raises(ValueError, match="Unsupported element type for VTK export: UnsupportedSolid"):
        vtk.export.from_result(result, output_dir=tmp_path)
