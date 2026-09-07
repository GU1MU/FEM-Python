"""Legacy CSV compatibility: permissive scalar fallback and strict provenance boundaries."""

import csv

import pytest

from fem.core.mesh import (
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
)
from fem.post import (
    path,
    vtk,
)
from fem.post.polar import convert_nodal_solution_into_polar_coord


@pytest.mark.parametrize(
    ("header", "row", "message"),
    (
        ("x,y,elem_id,local_node,averaged,sig_x", "0,0,,,true,10", "node_id"),
        ("node_id,x,y,local_node,averaged,sig_x", "1,0,0,,true,10", "elem_id"),
        ("node_id,x,y,elem_id,averaged,sig_x", "1,0,0,,true,10", "local_node"),
        ("node_id,x,y,elem_id,local_node,sig_x", "1,0,0,,,10", "averaged"),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "bad,0,0,,,true,10",
            "expected an integer",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,bad,1,false,10",
            "expected an integer",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,1,bad,false,10",
            "expected an integer",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,1,1,maybe,10",
            "expected true or false",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,,1,false,10",
            "missing elem_id",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,1,,false,10",
            "missing local_node",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,1,0,false,10",
            "one-based",
        ),
    ),
)
def test_nodal_stress_reader_enforces_current_metadata_contract(
    tmp_path,
    header,
    row,
    message,
):
    stress_path = tmp_path / "nodal_stress.csv"
    stress_path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        vtk.fields.read_nodal_stress_rows(stress_path)


def test_nodal_stress_reader_accepts_averaged_rows_without_provenance(tmp_path):
    stress_path = tmp_path / "averaged_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x\n"
        "1,0,0,,,true,10\n"
        "2,1,0,,,true,20\n",
        encoding="utf-8",
    )

    data = vtk.fields.read_nodal_stress_rows(stress_path)

    assert [(row.node_id, row.elem_id, row.local_node, row.averaged) for row in data.rows] == [
        (1, None, None, True),
        (2, None, None, True),
    ]
    assert [row.values["sig_x"] for row in data.rows] == [10.0, 20.0]


@pytest.mark.parametrize(
    "consumer",
    ("vtk", "path", "polar"),
)
def test_nodal_stress_consumers_reject_malformed_current_rows(tmp_path, consumer):
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
        elements=[],
    )
    stress_path = tmp_path / "invalid_current_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy\n"
        "1,0,0,1,1,not-bool,10,2,3\n",
        encoding="utf-8",
    )

    def call():
        if consumer == "vtk":
            return vtk.fields.read_nodal_stress_rows(stress_path)
        if consumer == "path":
            return path.extract_path_data(
                mesh,
                1,
                2,
                2,
                "sig_x",
                path=tmp_path / "path.csv",
                stress_csv_path=stress_path,
            )
        return convert_nodal_solution_into_polar_coord(
            stress_path,
            (0.0, 0.0),
            tmp_path / "polar.csv",
        )

    with pytest.raises(ValueError, match="averaged"):
        call()


def test_path_general_reader_keeps_last_duplicate_row_and_invalid_scalar_zero(tmp_path):
    mesh = Mesh2D(nodes=[Node2D(1, 0.0, 0.0)], elements=[])
    disp_path = tmp_path / "duplicate_nodal_displacement.csv"
    out_path = tmp_path / "nodes.csv"
    disp_path.write_text(
        "node_id,x,y,ux,uy\n"
        "1,0,0,10,0\n"
        "1,0,0,invalid,0\n",
        encoding="utf-8",
    )

    path.extract_nodes_data(
        mesh,
        [1],
        ["ux"],
        path=out_path,
        disp_csv_path=disp_path,
    )

    with out_path.open("r", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["ux"]) == 0.0


def test_averaged_whitespace_provenance_is_accepted_across_stress_entries(tmp_path):
    mesh = Mesh2D(nodes=[Node2D(1, 1.0, 0.0)], elements=[])
    stress_path = tmp_path / "averaged_whitespace_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy\n"
        "1,1,0,   ,   , TRUE ,10,2,3\n"
        "1,1,0,   ,   , TRUE ,20,4,6\n"
        "1,1,0, 1 , 2 ,false,30,6,9\n",
        encoding="utf-8",
    )

    vtk_data = vtk.fields.read_nodal_stress_rows(stress_path)
    assert [(row.elem_id, row.local_node, row.averaged) for row in vtk_data.rows] == [
        (None, None, True),
        (None, None, True),
        (1, 2, False),
    ]

    path_stress_path = tmp_path / "path_averaged_whitespace_nodal_stress.csv"
    path_stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy\n"
        "1,1,0,   ,   , TRUE ,10,2,3\n",
        encoding="utf-8",
    )
    path_out = tmp_path / "nodes.csv"
    path.extract_nodes_data(
        mesh,
        [1],
        ["sig_x"],
        path=path_out,
        stress_csv_path=path_stress_path,
    )
    with path_out.open("r", encoding="utf-8") as stream:
        path_row = next(csv.DictReader(stream))
    assert float(path_row["sig_x"]) == pytest.approx(10.0)

    polar_out = tmp_path / "polar.csv"
    convert_nodal_solution_into_polar_coord(
        stress_path,
        (0.0, 0.0),
        polar_out,
    )
    with polar_out.open("r", encoding="utf-8") as stream:
        polar_rows = list(csv.DictReader(stream))
    assert [float(row["sig_r"]) for row in polar_rows] == pytest.approx(
        [10.0, 20.0, 30.0]
    )
    assert [float(row["sig_t"]) for row in polar_rows] == pytest.approx([2.0, 4.0, 6.0])
    assert [float(row["tau_rt"]) for row in polar_rows] == pytest.approx([3.0, 6.0, 9.0])


def test_vtk_element_stress_reader_averages_repeated_element_rows(tmp_path):
    from fem.post.vtk import fields

    csv_path = tmp_path / "test_vtk_element_stress_average.csv"
    csv_path.write_text(
        "elem_id,node_id,local_node,sig_x,sig_y,tau_xy,mises\n"
        "1,1,1,1,2,3,4\n"
        "1,2,2,3,4,5,6\n",
        encoding="utf-8",
    )

    fields_by_name = fields.read_element_stress(csv_path)

    assert fields_by_name["sig_x"][1] == pytest.approx(2.0)
    assert fields_by_name["mises"][1] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("field", "row", "expected"),
    (
        ("node_id", "bad,1,2,3,4", "expected an integer"),
        ("ux", "1,bad,2,3,4", "expected a numeric value"),
        ("uy", "1,1,bad,3,4", "expected a numeric value"),
        ("uz", "1,1,2,bad,4", "expected a numeric value"),
        ("rz", "1,1,2,3,bad", "expected a numeric value"),
    ),
)
def test_vtk_displacement_reader_reports_invalid_leaf_with_context(
    tmp_path,
    field,
    row,
    expected,
):
    mesh = Mesh3D(nodes=[Node3D(1, 0.0, 0.0, 0.0)], elements=[])
    csv_path = tmp_path / f"invalid_displacement_{field}.csv"
    csv_path.write_text(
        "node_id,ux,uy,uz,rz\n" + row + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        vtk.fields.read_displacement(mesh, csv_path)

    message = str(exc_info.value)
    assert str(csv_path) in message
    assert "line 2" in message
    assert f"field {field}" in message
    assert "raw value 'bad'" in message
    assert expected in message


def test_vtk_element_stress_reader_reports_invalid_elem_id_with_context(tmp_path):
    csv_path = tmp_path / "invalid_element_stress_elem_id.csv"
    csv_path.write_text(
        "elem_id,sig_x\nbad,10\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        vtk.fields.read_element_stress(csv_path)

    message = str(exc_info.value)
    assert str(csv_path) in message
    assert "line 2" in message
    assert "field elem_id" in message
    assert "raw value 'bad'" in message
    assert "expected an integer" in message


def test_vtk_element_stress_reader_keeps_invalid_scalar_zero_in_average(tmp_path):
    csv_path = tmp_path / "invalid_element_stress_scalar.csv"
    csv_path.write_text(
        "elem_id,sig_x\n1,4\n1,invalid\n",
        encoding="utf-8",
    )

    fields_by_name = vtk.fields.read_element_stress(csv_path)

    assert fields_by_name["sig_x"][1] == pytest.approx(2.0)


def test_vtk_reader_parses_beam2_nodal_stress_csv_as_three_scalars(tmp_path):
    path = tmp_path / "beam_nodal_stress.csv"
    path.write_text(
        "node_id,x,y,z,axial_stress_max,axial_stress_min,axial_stress_abs_max\n"
        "1,0,0,0,12,-4,12\n"
        "2,1,0,0,8,-6,8\n",
        encoding="utf-8",
    )

    data = vtk.fields.read_nodal_stress_rows(path)

    assert data.field_names == (
        "axial_stress_max",
        "axial_stress_min",
        "axial_stress_abs_max",
    )
    assert [row.node_id for row in data.rows] == [1, 2]
    assert all(row.averaged for row in data.rows)
    assert data.rows[0].values == {
        "axial_stress_max": 12.0,
        "axial_stress_min": -4.0,
        "axial_stress_abs_max": 12.0,
    }
