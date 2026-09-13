"""Model preparation measurements and lazy inspection contracts."""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Callable
from time import perf_counter
from typing import Any

from fem.application.preflight import run_static_preflight
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import AnalysisStep, FEMModel
from fem_gui.inspection_service import InspectionService
from fem_gui.visualization.model_adapter import build_model_geometry


def _measure(function: Callable[[], Any]) -> tuple[Any, float, int]:
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    value = function()
    elapsed = perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak_bytes


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


def _measure_model_preparation(element_count: int) -> dict[str, Any]:
    model, model_seconds, model_peak = _measure(
        lambda: _plate_model(element_count)
    )
    report, preflight_seconds, preflight_peak = _measure(
        lambda: run_static_preflight(
            model,
            "load",
            check_numerical_stability=False,
            copy_model=False,
            quick_check=True,
        )
    )
    geometry, geometry_seconds, geometry_peak = _measure(
        lambda: build_model_geometry(model)
    )
    inspection, inspection_seconds, inspection_peak = _measure(
        lambda: InspectionService(model)
    )
    return {
        "elements": element_count,
        "nodes": len(model.mesh.nodes),
        "dofs": model.mesh.num_dofs,
        "model_seconds": model_seconds,
        "model_peak_bytes": model_peak,
        "quick_preflight_seconds": preflight_seconds,
        "quick_preflight_peak_bytes": preflight_peak,
        "geometry_seconds": geometry_seconds,
        "geometry_peak_bytes": geometry_peak,
        "geometry_cell_array_bytes": geometry.cell_array.nbytes,
        "inspection_seconds": inspection_seconds,
        "inspection_peak_bytes": inspection_peak,
        "inspection_cached_records": (
            inspection._element_record_cached.cache_info().currsize
        ),
        "preflight_diagnostic_codes": tuple(
            diagnostic.code for diagnostic in report.diagnostics
        ),
        "numerical_stability_checked": report.numerical_stability_checked,
    }


def test_model_preparation_preserves_lazy_inspection(record_property) -> None:
    metrics = _measure_model_preparation(5_000)
    for name, value in metrics.items():
        record_property(name, value)
    assert metrics["nodes"] == 5_151
    assert metrics["dofs"] == 10_302
    assert metrics["geometry_cell_array_bytes"] == 5_000 * 5 * 8
    assert metrics["inspection_cached_records"] == 0
    assert metrics["numerical_stability_checked"] is False
    assert "static.boundary.missing_displacement" in metrics["preflight_diagnostic_codes"]
