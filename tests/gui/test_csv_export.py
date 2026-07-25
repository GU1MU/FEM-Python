from __future__ import annotations

import csv

import numpy as np

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import FEMModel
from fem.core.result import ModelResult
from fem_gui.visualization.csv_export import export_field_csv
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data


def test_beam_envelope_uses_the_common_current_field_csv(tmp_path):
    mesh = Mesh3D(
        [Node3D(10, 0.0, 0.0, 0.0), Node3D(20, 1.0, 0.0, 0.0)],
        [Element3D(30, [10, 20], "Beam2", {
            "E": 200.0,
            "nu": 0.3,
            "section_type": "rectangle",
            "height": 0.1,
            "width": 0.2,
        })],
        dofs_per_node=6,
    )
    model = FEMModel(mesh)
    displacement = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.01, 0.02, 0.0, 0.0, 0.0, 0.01,
    ])
    result = ModelResult(model, None, displacement, np.zeros(12))
    data = build_result_data(result, build_model_geometry(model))
    target = export_field_csv(
        data, "NODAL:S11AbsMax", tmp_path / "beam_envelope.csv"
    )

    with target.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["node_id"] for row in rows] == ["10", "20"]
    assert {row["field"] for row in rows} == {"S11AbsMax"}
    assert {row["position"] for row in rows} == {"nodal"}
    assert {row["association"] for row in rows} == {"point"}


def test_truss_centroid_uses_cell_ids_in_the_same_csv_schema(tmp_path):
    mesh = Mesh3D(
        [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        [Element3D(9, [1, 2], "Truss2", {"E": 100.0, "area": 2.0})],
        dofs_per_node=3,
    )
    model = FEMModel(mesh)
    result = ModelResult(
        model,
        None,
        np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0]),
        np.zeros(6),
    )
    data = build_result_data(result, build_model_geometry(model))
    target = export_field_csv(data, "CENTROID:S11", tmp_path / "truss.csv")

    with target.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["elem_id"] == "9"
    assert row["node_id"] == ""
    assert row["position"] == "centroid"
    assert row["association"] == "cell"
