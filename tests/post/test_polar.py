import csv

import pytest

from fem.core.mesh import (
    Mesh2D,
    Node2D,
)
from fem.post import (
    vtk,
)
from fem.post.polar import convert_nodal_solution_into_polar_coord
from fem.post.vtk.polar import convert_nodal_displacement


def test_polar_conversion_preserves_distinct_repeated_nodal_rows():
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 1.0)],
        elements=[],
    )
    data = vtk.fields.NodalStressCsv(
        ("sig_x", "sig_y", "tau_xy", "mises"),
        (
            vtk.fields.NodalStressCsvRow(
                1, 1, 1, False, {"sig_x": 10.0, "sig_y": 0.0, "tau_xy": 0.0, "mises": 10.0}
            ),
            vtk.fields.NodalStressCsvRow(
                1, 2, 1, False, {"sig_x": 20.0, "sig_y": 0.0, "tau_xy": 0.0, "mises": 20.0}
            ),
        ),
    )

    converted = vtk.polar.convert_nodal_stress_rows(mesh, data, (0.0, 0.0))

    assert [row.values["sig_t"] for row in converted.rows] == pytest.approx([10.0, 20.0])
    assert [(row.elem_id, row.local_node) for row in converted.rows] == [(1, 1), (2, 1)]


def test_polar_csv_conversion_reports_non_numeric_center_with_context(tmp_path):
    center = ("bad-radius", 0.0)

    with pytest.raises(ValueError) as exc_info:
        convert_nodal_solution_into_polar_coord(
            tmp_path / "unused.csv",
            center,
            tmp_path / "unused_polar.csv",
        )

    message = str(exc_info.value)
    assert repr(center) in message
    assert "2 numeric values" in message
    assert "expected numeric x/y" in message


def test_polar_csv_conversion_reports_missing_numeric_value_with_context(tmp_path):
    csv_path = tmp_path / "incomplete_displacement.csv"
    csv_path.write_text(
        "node_id,x,y,ux,uy\n"
        "1,0,0,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        convert_nodal_solution_into_polar_coord(
            csv_path,
            (0.0, 0.0),
            tmp_path / "incomplete_polar.csv",
        )

    message = str(exc_info.value)
    assert str(csv_path) in message
    assert "line 2" in message
    assert "field uy" in message
    assert "raw value None" in message
    assert "expected a numeric value" in message


def test_polar_csv_conversion_accepts_current_stress_metadata(tmp_path):
    csv_path = tmp_path / "current_nodal_stress.csv"
    out_path = tmp_path / "current_polar.csv"
    csv_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy\n"
        "1,1,0,1,1,false,10,2,3\n",
        encoding="utf-8",
    )

    convert_nodal_solution_into_polar_coord(csv_path, (0.0, 0.0), out_path)

    with out_path.open("r", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["sig_r"]) == pytest.approx(10.0)
    assert float(row["sig_t"]) == pytest.approx(2.0)
    assert float(row["tau_rt"]) == pytest.approx(3.0)


def test_polar_csv_conversion_keeps_displacement_schema_unchanged(tmp_path):
    csv_path = tmp_path / "displacement.csv"
    out_path = tmp_path / "polar_displacement.csv"
    csv_path.write_text(
        "node_id,x,y,ux,uy\n1,0,1,2,0\n",
        encoding="utf-8",
    )

    convert_nodal_solution_into_polar_coord(csv_path, (0.0, 0.0), out_path)

    with out_path.open("r", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["ur"]) == pytest.approx(0.0)
    assert float(row["ut"]) == pytest.approx(-2.0)


def test_polar_displacement_fills_mesh_nodes_and_ignores_unknown_ids():
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 1.0, 0.0),
            Node2D(2, 0.0, 1.0),
            Node2D(3, -1.0, 0.0),
            Node2D(4, 0.0, 0.0),
        ],
        elements=[],
    )

    polar_values = convert_nodal_displacement(
        mesh,
        {
            1: {"ux": 2.0, "uy": 0.0, "rz": 0.5},
            2: {"ux": 0.0, "uy": 3.0, "rz": 0.0},
            4: {"ux": 4.0},
            999: {"ux": 9.0, "uy": 9.0, "rz": 9.0},
        },
        [0.0, 0.0],
    )

    assert set(polar_values) == {1, 2, 3, 4}
    assert polar_values[1] == {"ux": 2.0, "uy": 0.0, "rz": 0.5}
    assert polar_values[2] == {"ux": 3.0, "uy": 0.0, "rz": 0.0}
    assert polar_values[3] == {"ux": 0.0, "uy": 0.0, "rz": 0.0}
    assert polar_values[4] == {"ux": 4.0, "uy": 0.0, "rz": 0.0}
