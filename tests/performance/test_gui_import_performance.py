from __future__ import annotations

import os
from time import perf_counter

import pytest

from fem.boundary.step import boundary_for_step
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import AnalysisStep, Edge, EdgeLoad, ElementEdge, FEMModel
from fem_gui.inspection_service import InspectionService
from fem_gui.visualization.model_adapter import build_model_geometry


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("FEM_RUN_SLOW_PERF") != "1",
        reason=(
            "[slow-opt-in] set FEM_RUN_SLOW_PERF=1 to run "
            "scalability benchmarks"
        ),
    ),
]


def _plate_model(element_count: int, pressure: bool) -> FEMModel:
    nx = max(int(element_count ** 0.5), 1)
    ny = max((element_count + nx - 1) // nx, 1)
    nodes = [
        Node2D(row * (nx + 1) + column + 1, float(column), float(row))
        for row in range(ny + 1)
        for column in range(nx + 1)
    ]
    elements = []
    loaded_edges = []
    for index in range(element_count):
        row, column = divmod(index, nx)
        lower_left = row * (nx + 1) + column + 1
        node_ids = (
            lower_left, lower_left + 1,
            lower_left + nx + 2, lower_left + nx + 1,
        )
        element_id = index + 1
        elements.append(Element2D(element_id, node_ids, "Quad4Plane"))
        if pressure and index >= element_count - min(nx, element_count):
            loaded_edges.append(ElementEdge(
                element_id, 2, (lower_left + nx + 2, lower_left + nx + 1)
            ))
    edges = {"TOP": Edge("TOP", loaded_edges)} if pressure else {}
    steps = [AnalysisStep(
        "load",
        edge_loads=[EdgeLoad("TOP", magnitude=2.0, load_type="pressure")]
        if pressure else (),
    )]
    return FEMModel(Mesh2D(nodes, elements), edges=edges, steps=steps)


@pytest.mark.parametrize(
    ("element_count", "pressure"),
    [(10, False), (1_000, False), (50_000, False), (50_000, True)],
)
def test_import_pipeline_scalability(element_count, pressure, record_property):
    started = perf_counter()
    model = _plate_model(element_count, pressure)
    model_time = perf_counter() - started
    started = perf_counter()
    geometry = build_model_geometry(model)
    geometry_time = perf_counter() - started
    started = perf_counter()
    service = InspectionService(model)
    inspection_time = perf_counter() - started
    started = perf_counter()
    boundary = boundary_for_step(model, "load")
    boundary_time = perf_counter() - started

    record_property("FEMModel", model_time)
    record_property("geometry", geometry_time)
    record_property("inspection", inspection_time)
    record_property("boundary", boundary_time)
    assert len(geometry.cells) == element_count
    assert service._element_record_cached.cache_info().currsize == 0
    assert len(boundary.edge_tractions) == (len(model.edges.get("TOP", ()).edges) if pressure else 0)
