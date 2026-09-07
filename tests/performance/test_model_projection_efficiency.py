from __future__ import annotations

import numpy as np

from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import AnalysisStep, FEMModel
from fem_gui.inspection_service import InspectionService
from fem_gui.visualization.model_adapter import build_model_geometry, pyvista_cell_array


def test_large_import_projection_stays_flat_and_inspection_stays_lazy():
    model = _plate_model(5_000)

    geometry = build_model_geometry(model)
    service = InspectionService(model)

    assert geometry.points.shape == (len(model.mesh.nodes), 3)
    assert geometry.cell_array.shape == (5 * len(model.mesh.elements),)
    assert geometry.cell_array.dtype == np.int64
    assert geometry.cell_array.flags.c_contiguous
    assert pyvista_cell_array(geometry) is geometry.cell_array
    assert len(geometry.cells) == len(model.mesh.elements)
    assert service._element_record_cached.cache_info().currsize == 0

    service.element_record(model.mesh.elements[-1].id)
    assert service._element_record_cached.cache_info().currsize == 1


def _plate_model(element_count: int) -> FEMModel:
    columns = 100
    rows = (element_count + columns - 1) // columns
    nodes = [
        Node2D(
            row * (columns + 1) + column + 1,
            float(column),
            float(row),
        )
        for row in range(rows + 1)
        for column in range(columns + 1)
    ]
    elements = []
    for index in range(element_count):
        row, column = divmod(index, columns)
        lower_left = row * (columns + 1) + column + 1
        elements.append(
            Element2D(
                index + 1,
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + columns + 2,
                    lower_left + columns + 1,
                ],
                "Quad4",
            )
        )
    return FEMModel(
        Mesh2D(nodes, elements),
        steps=[AnalysisStep("load")],
    )
