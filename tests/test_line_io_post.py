import csv

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import AnalysisStep, FEMModel
from fem.core.result import ModelResult
from fem.io import csv as csv_io
from fem.post import stress, vtk


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _materials(tmp_path):
    return _write(
        tmp_path / "materials.csv",
        ["material_id,name,E,rho,nu", "1,steel,210,7.8,0.25"],
    )


def test_read_truss2_requires_and_preserves_three_dimensional_contract(tmp_path):
    mesh_path = _write(
        tmp_path / "truss.csv",
        [
            "node_id,x,y,z",
            "10,0,0,0",
            "20,2,3,6",
            "elem_id,node_i,node_j,area,material_id",
            "1,10,20,2.5,1",
        ],
    )

    mesh = csv_io.read_truss2(mesh_path, _materials(tmp_path))

    assert isinstance(mesh, Mesh3D)
    assert mesh.dofs_per_node == 3
    assert mesh.nodes == [Node3D(10, 0.0, 0.0, 0.0), Node3D(20, 2.0, 3.0, 6.0)]
    assert mesh.elements[0].type == "Truss2"
    assert mesh.elements[0].props == {
        "area": 2.5,
        "material_id": 1,
        "E": 210.0,
        "rho": 7.8,
    }


def test_read_beam2_reads_topology_only(tmp_path):
    mesh_path = _write(
        tmp_path / "beam.csv",
        [
            "node_id,x,y,z",
            "1,0,0,0",
            "2,4,0,0",
            "elem_id,node_i,node_j",
            "10,1,2",
        ],
    )

    mesh = csv_io.read_beam2(mesh_path)

    assert isinstance(mesh, Mesh3D)
    assert mesh.dofs_per_node == 6
    assert mesh.elements[0].type == "Beam2"
    assert mesh.elements[0].node_ids == [1, 2]
    assert mesh.elements[0].props == {}


@pytest.mark.parametrize(
    ("reader_name", "node_header", "node_rows", "element_header", "element_row", "message"),
    [
        (
            "read_truss2",
            "node_id,x,y,z",
            ["1,0,0,0", "2,1,0,0"],
            "elem_id,node_i,node_j,material_id",
            "1,1,2,1",
            "element header",
        ),
        (
            "read_truss2",
            "node_id,x,y,z",
            ["1,0,0,0", "2,1,0,0"],
            "elem_id,node_j,node_i,area,material_id",
            "1,1,2,3,1",
            "element header",
        ),
        (
            "read_truss2",
            "node_id,x,y,z",
            ["1,0,0,0", "2,1,0,0"],
            "elem_id,node_i,node_j,area,material_id,note",
            "1,1,2,3,1,unexpected",
            "element header",
        ),
        (
            "read_beam2",
            "node_id,x,y,z",
            ["1,0,0,0", "2,1,0,0"],
            "elem_id,node_i",
            "1,1",
            "element header",
        ),
        (
            "read_beam2",
            "node_id,x,y,z",
            ["1,0,0,0", "2,1,0,0"],
            "elem_id,node_j,node_i",
            "1,2,1",
            "element header",
        ),
        (
            "read_beam2",
            "node_id,x,y,z",
            ["1,0,0,0", "2,1,0,0"],
            "elem_id,node_i,node_j,note",
            "1,1,2,unexpected",
            "element header",
        ),
        (
            "read_truss2",
            "node_id,x,y",
            ["1,0,0", "2,1,0"],
            "elem_id,node_i,node_j,area,material_id",
            "1,1,2,1,1",
            "node_id,x,y,z",
        ),
        (
            "read_beam2",
            "node_id,x,y",
            ["1,0,0", "2,1,0"],
            "elem_id,node_i,node_j",
            "1,1,2",
            "node_id,x,y,z",
        ),
    ],
)
def test_line_csv_readers_require_current_node_and_element_headers(
    tmp_path,
    reader_name,
    node_header,
    node_rows,
    element_header,
    element_row,
    message,
):
    mesh_path = _write(
        tmp_path / "bad_header.csv",
        [node_header, *node_rows, element_header, element_row],
    )

    with pytest.raises(ValueError, match=message):
        getattr(csv_io, reader_name)(mesh_path)


def _result(mesh, displacement_values):
    model = FEMModel(mesh=mesh, name="line_result")
    return ModelResult(
        model,
        AnalysisStep("result"),
        np.asarray(displacement_values, dtype=float),
        np.zeros(mesh.num_dofs),
    )


def test_truss2_result_export_writes_3d_displacement_element_stress_and_vtk(tmp_path):
    mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], "Truss2", {"E": 100.0, "area": 2.0})],
    )
    vtk.export.from_result(_result(mesh, [0, 0, 0, 0.2, 0, 0]), tmp_path, overwrite=True)

    displacement_path = tmp_path / "line_result_nodal_displacement.csv"
    stress_path = tmp_path / "line_result_element_stress.csv"
    vtk_path = tmp_path / "line_result.vtk"
    assert next(csv.reader(displacement_path.open(encoding="utf-8"))) == [
        "node_id", "x", "y", "z", "ux", "uy", "uz"
    ]
    assert next(csv.reader(stress_path.open(encoding="utf-8"))) == [
        "elem_id", "node_i", "node_j", "axial_strain", "axial_stress", "mises"
    ]
    assert "VECTORS displacement float" in vtk_path.read_text(encoding="utf-8")
    assert not (tmp_path / "line_result_nodal_stress.csv").exists()


def test_beam2_result_export_writes_six_components_rotation_vector_and_no_element_stress(tmp_path):
    mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        elements=[
            Element3D(
                1,
                [1, 2],
                "Beam2",
                {
                    "E": 100.0,
                    "nu": 0.25,
                    "section_type": "rectangle",
                    "height": 2.0,
                    "width": 1.0,
                },
            )
        ],
        dofs_per_node=6,
    )
    values = np.array([0, 0, 0, 0.1, 0.2, 0.3, 1, 2, 3, 0.4, 0.5, 0.6])
    result = _result(mesh, values)
    vtk.export.from_result(result, tmp_path, overwrite=True)

    displacement_path = tmp_path / "line_result_nodal_displacement.csv"
    rows = list(csv.reader(displacement_path.open(encoding="utf-8")))
    assert rows[0] == ["node_id", "x", "y", "z", "ux", "uy", "uz", "rx", "ry", "rz"]
    vtk_text = (tmp_path / "line_result.vtk").read_text(encoding="utf-8")
    assert "VECTORS displacement float" in vtk_text
    assert "VECTORS rotation float" in vtk_text
    assert "0.1 0.2 0.3" in vtk_text
    assert not (tmp_path / "line_result_element_stress.csv").exists()
    stress_path = tmp_path / "line_result_nodal_stress.csv"
    stress_rows = list(csv.reader(stress_path.open(encoding="utf-8")))
    assert stress_rows[0] == [
        "node_id",
        "x",
        "y",
        "z",
        "axial_stress_max",
        "axial_stress_min",
        "axial_stress_abs_max",
    ]
    assert [int(row[0]) for row in stress_rows[1:]] == [1, 2]
    for scalar in (
        "axial_stress_max",
        "axial_stress_min",
        "axial_stress_abs_max",
    ):
        assert f"SCALARS {scalar} float 1" in vtk_text
    assert "VECTORS axial_stress" not in vtk_text


def test_beam2_direct_nodal_export_requires_result_load_context(tmp_path):
    mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        elements=[
            Element3D(
                1,
                [1, 2],
                "Beam2",
                {
                    "E": 100.0,
                    "nu": 0.25,
                    "section_type": "solid_circle",
                    "radius": 1.0,
                },
            )
        ],
        dofs_per_node=6,
    )
    result = _result(mesh, np.zeros(mesh.num_dofs))
    path = tmp_path / "beam_nodal_stress.csv"

    with pytest.raises(ValueError, match="ModelResult.*load context"):
        stress.export.nodal(mesh, result.U, path)

    stress.export.nodal_from_result(result, path)

    assert path.exists()


def test_beam2_nodal_stress_csv_contains_every_mesh_node_once(tmp_path):
    mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 2.0, 0.0, 0.0),
        ],
        dofs_per_node=6,
        elements=[
            Element3D(
                1,
                [1, 2],
                "Beam2",
                {
                    "E": 100.0,
                    "nu": 0.25,
                    "section_type": "solid_circle",
                    "radius": 1.0,
                },
            )
        ],
    )

    vtk.export.from_result(_result(mesh, np.zeros(mesh.num_dofs)), tmp_path)

    rows = list(
        csv.DictReader(
            (tmp_path / "line_result_nodal_stress.csv").open(encoding="utf-8")
        )
    )
    assert [int(row["node_id"]) for row in rows] == [1, 2, 3]
    assert float(rows[2]["axial_stress_abs_max"]) == pytest.approx(0.0)
