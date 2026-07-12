import csv
import importlib
from pathlib import Path

import numpy as np
import pytest

from fem.core.mesh import BeamMesh3D, Element2D, Element3D, Node2D, Node3D, TrussMesh3D
from fem.core.model import AnalysisStep, FEMModel
from fem.core.result import ModelResult
from fem.elements import get_element_kernel
from fem.io import csv as csv_io
from fem.post import displacement, stress, vtk


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

    assert isinstance(mesh, TrussMesh3D)
    assert mesh.nodes == [Node3D(10, 0.0, 0.0, 0.0), Node3D(20, 2.0, 3.0, 6.0)]
    assert mesh.elements[0].type == "Truss2"
    assert mesh.elements[0].props == {
        "area": 2.5,
        "material_id": 1,
        "E": 210.0,
        "rho": 7.8,
    }


def test_read_beam2_reads_section_orientation_and_material_properties(tmp_path):
    mesh_path = _write(
        tmp_path / "beam.csv",
        [
            "node_id,x,y,z",
            "1,0,0,0",
            "2,4,0,0",
            "elem_id,node_i,node_j,section_type,radius,outer_radius,inner_radius,size_y,size_z,local_y_x,local_y_y,local_y_z,material_id",
            "10,1,2,rectangle,,,,3,2,0,1,0,1",
        ],
    )

    mesh = csv_io.read_beam2(mesh_path, _materials(tmp_path))

    assert isinstance(mesh, BeamMesh3D)
    assert mesh.elements[0].type == "Beam2"
    assert mesh.elements[0].props == {
        "section_type": "rectangle",
        "size_y": 3.0,
        "size_z": 2.0,
        "local_y": (0.0, 1.0, 0.0),
        "material_id": 1,
        "E": 210.0,
        "nu": 0.25,
        "rho": 7.8,
    }


def test_read_beam2_preserves_shape_specific_circle_dimensions(tmp_path):
    mesh_path = _write(
        tmp_path / "circles.csv",
        [
            "node_id,x,y,z",
            "1,0,0,0",
            "2,1,0,0",
            "3,2,0,0",
            "elem_id,node_i,node_j,section_type,radius,outer_radius,inner_radius,size_y,size_z,local_y_x,local_y_y,local_y_z,material_id",
            "1,1,2,solid_circle,0.5,,,,,0,1,0,1",
            "2,2,3,hollow_circle,,0.6,0.4,,,0,1,0,1",
        ],
    )

    mesh = csv_io.read_beam2(mesh_path)

    assert mesh.elements[0].props == {
        "section_type": "solid_circle",
        "radius": 0.5,
        "local_y": (0.0, 1.0, 0.0),
        "material_id": 1,
    }
    assert mesh.elements[1].props == {
        "section_type": "hollow_circle",
        "outer_radius": 0.6,
        "inner_radius": 0.4,
        "local_y": (0.0, 1.0, 0.0),
        "material_id": 1,
    }


def test_read_beam2_rejects_nonempty_irrelevant_section_dimension(tmp_path):
    mesh_path = _write(
        tmp_path / "irrelevant_dimension.csv",
        [
            "node_id,x,y,z",
            "1,0,0,0",
            "2,1,0,0",
            "elem_id,node_i,node_j,section_type,radius,outer_radius,inner_radius,size_y,size_z,local_y_x,local_y_y,local_y_z,material_id",
            "1,1,2,solid_circle,0.5,0.6,,,,0,1,0,1",
        ],
    )

    with pytest.raises(ValueError, match="solid_circle.*outer_radius"):
        csv_io.read_beam2(mesh_path)


@pytest.mark.parametrize(
    ("reader_name", "header", "row"),
    [
        ("read_truss2", "elem_id,node_i,node_j,section_area,material_id", "1,1,2,3,1"),
        ("read_truss2", "elem_id,node_j,node_i,area,material_id", "1,1,2,3,1"),
        ("read_truss2", "elem_id,node_i,node_j,area,material_id,extra", "1,1,2,3,1,9"),
        (
            "read_beam2",
            "elem_id,node_i,node_j,area,Iyy,Izz,J,local_y_x,local_y_y,local_y_z,material_id",
            "1,1,2,3,5,7,2,0,1,0,1",
        ),
        (
            "read_beam2",
            "elem_id,node_i,node_j,section_type,radius,outer_radius,inner_radius,size_y,size_z,local_y_x,local_y_y,local_y_z,material,extra",
            "1,1,2,solid_circle,1,,,,,0,1,0,1,9",
        ),
    ],
)
def test_line_csv_readers_reject_noncanonical_element_headers(
    tmp_path, reader_name, header, row
):
    mesh_path = _write(
        tmp_path / "bad_header.csv",
        ["node_id,x,y,z", "1,0,0,0", "2,1,0,0", header, row],
    )

    with pytest.raises(ValueError, match="element header"):
        getattr(csv_io, reader_name)(mesh_path)


@pytest.mark.parametrize("reader_name", ["read_truss2", "read_beam2"])
def test_line_csv_readers_reject_legacy_two_dimensional_nodes(tmp_path, reader_name):
    mesh_path = _write(
        tmp_path / "legacy.csv",
        ["node_id,x,y", "1,0,0", "2,1,0", "elem_id,node_i,node_j,area,material_id", "1,1,2,1,1"],
    )

    with pytest.raises(ValueError, match="node_id,x,y,z"):
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
    mesh = TrussMesh3D(
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
    mesh = BeamMesh3D(
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
                    "size_y": 2.0,
                    "size_z": 1.0,
                    "local_y": (0, 1, 0),
                },
            )
        ],
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
    mesh = BeamMesh3D(
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
                    "local_y": (0.0, 1.0, 0.0),
                },
            )
        ],
    )
    result = _result(mesh, np.zeros(mesh.num_dofs))
    path = tmp_path / "beam_nodal_stress.csv"

    with pytest.raises(ValueError, match="ModelResult.*load context"):
        stress.export.nodal(mesh, result.U, path)

    stress.export.nodal_from_result(result, path)

    assert path.exists()


def test_beam2_nodal_stress_csv_contains_every_mesh_node_once(tmp_path):
    mesh = BeamMesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 2.0, 0.0, 0.0),
        ],
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
                    "local_y": (0.0, 1.0, 0.0),
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


def test_old_planar_line_apis_are_removed_and_plane_elements_remain_available():
    mesh_module = importlib.import_module("fem.core.mesh")
    line_module = importlib.import_module("fem.elements.line")

    for name in ("TrussMesh2D", "BeamMesh2D"):
        assert not hasattr(mesh_module, name)
    for name in ("Truss2DKernel", "Beam2DKernel"):
        assert not hasattr(line_module, name)
    for name in ("read_truss2d", "read_beam2d"):
        assert not hasattr(csv_io, name)
    for element_type in ("Truss2D", "Beam2D", "Truss2DLegacy"):
        with pytest.raises(NotImplementedError, match="Unsupported element type"):
            get_element_kernel(element_type)

    with pytest.raises(TypeError):
        Element2D(1, [1, 2])
    plane = Element2D(1, [1, 2, 3], "Tri3Plane")
    assert plane.type == "Tri3Plane"


def test_active_source_and_examples_contain_no_legacy_line_api_names():
    project_root = Path(__file__).resolve().parents[1]
    legacy_names = (
        "Truss2D", "Beam2D", "Truss2DKernel", "Beam2DKernel",
        "TrussMesh2D", "BeamMesh2D", "read_truss2d", "read_beam2d",
    )
    offenders = []
    for root_name in ("src", "exam" + "ples"):
        for path in (project_root / root_name).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".csv"}:
                text = path.read_text(encoding="utf-8")
                if any(name in text for name in legacy_names):
                    offenders.append(path.relative_to(project_root).as_posix())
    assert offenders == []


def test_active_source_and_examples_contain_no_old_beam2_input_contract():
    project_root = Path(__file__).resolve().parents[1]
    old_header = "elem_id,node_i,node_j,area,Iyy,Izz,J"
    old_property_reads = (
        'elem.props["Iyy"]',
        'elem.props["Izz"]',
        'elem.props["J"]',
    )
    offenders = []
    for root_name in ("src", "exam" + "ples"):
        for path in (project_root / root_name).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".csv"}:
                text = path.read_text(encoding="utf-8")
                if old_header in text or any(value in text for value in old_property_reads):
                    offenders.append(path.relative_to(project_root).as_posix())
    assert offenders == []
