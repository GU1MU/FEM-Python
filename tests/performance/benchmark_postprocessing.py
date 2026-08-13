from __future__ import annotations

import argparse
import gc
import json
import tracemalloc
from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np

from fem.application.results import (
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultCellKind,
    ResultFieldId,
    ResultFieldTopology,
    ResultSourceKey,
    ResultValueLayout,
    ResultVariable,
    ScalarFieldSelection,
    build_result_provider,
)
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import FEMModel
from fem.core.result import ModelResult
from fem_gui.visualization.result_renderer import (
    build_result_render_payload,
    validate_result_render_payload,
)


def _measure(function: Callable[[], Any]) -> tuple[Any, float, int]:
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    value = function()
    elapsed = perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak_bytes


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="benchmark-result",
        session_id="benchmark-session",
        artifact_id="benchmark-artifact",
        model_revision=1,
        step_name="Step-1",
        run_id="benchmark-run",
    )


def _selection(component: str) -> ScalarFieldSelection:
    key = FieldMaterializationKey(
        FieldRequest(
            ResultFieldId(ResultVariable.U, FieldPosition.NODE),
        ),
        recovery_contract=1,
    )
    return ScalarFieldSelection(key, component)


def _quad_topology(
    element_count: int,
    *,
    component: str,
) -> ResultFieldTopology:
    columns = 100
    rows = (element_count + columns - 1) // columns
    points = np.asarray(
        [
            (float(column), float(row), 0.0)
            for row in range(rows + 1)
            for column in range(columns + 1)
        ],
        dtype=float,
    )
    cells = []
    for index in range(element_count):
        row, column = divmod(index, columns)
        lower_left = row * (columns + 1) + column
        cells.append(
            (
                lower_left,
                lower_left + 1,
                lower_left + columns + 2,
                lower_left + columns + 1,
            )
        )
    cell_tuple = tuple(cells)
    return ResultFieldTopology(
        source=_source(),
        materialization_generation=1,
        selection=_selection(component),
        deformation_scale=1.0,
        points=points,
        cells=cell_tuple,
        cell_kinds=(ResultCellKind.FEM_ELEMENT,) * element_count,
        canonical_element_types=("Quad4",) * element_count,
        values=np.linspace(0.0, 1.0, len(points)),
        value_layout=ResultValueLayout.POINT,
        point_locations=(None,) * len(points),
        cell_locations=(None,) * element_count,
    )


def _quad_result(element_count: int) -> ModelResult:
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
    properties = {
        "E": 210_000.0,
        "nu": 0.3,
        "plane_type": "stress",
        "thickness": 1.0,
    }
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
                properties,
            )
        )
    mesh = Mesh2D(nodes, elements)
    displacement = np.zeros(mesh.num_dofs, dtype=float)
    displacement[0::2] = np.linspace(0.0, 0.01, len(nodes))
    return ModelResult(
        FEMModel(mesh, name="postprocessing-benchmark"),
        None,
        displacement,
        np.zeros(mesh.num_dofs, dtype=float),
    )


def run(
    element_count: int,
    *,
    include_recovery: bool = False,
) -> dict[str, float | int]:
    first = _quad_topology(element_count, component="U1")
    second = _quad_topology(element_count, component="U2")
    payload, build_seconds, build_peak = _measure(
        lambda: build_result_render_payload(first),
    )
    _, validate_seconds, validate_peak = _measure(
        lambda: validate_result_render_payload(payload),
    )
    rebound, reuse_seconds, reuse_peak = _measure(
        lambda: build_result_render_payload(second, reusable=payload),
    )
    result: dict[str, float | int] = {
        "elements": element_count,
        "points": len(first.point_locations),
        "build_seconds": build_seconds,
        "build_peak_bytes": build_peak,
        "validate_seconds": validate_seconds,
        "validate_peak_bytes": validate_peak,
        "reuse_seconds": reuse_seconds,
        "reuse_peak_bytes": reuse_peak,
        "dataset_reused": int(rebound.dataset is payload.dataset),
    }
    if include_recovery:
        provider = build_result_provider(_source(), _quad_result(element_count))
        key = provider.resolve_request(
            FieldRequest(
                ResultFieldId(ResultVariable.S, FieldPosition.CENTROID),
            )
        )
        patch, recovery_seconds, recovery_peak = _measure(
            lambda: provider.materialize((key,)),
        )
        result.update(
            recovery_rows=len(patch.fields[0].locations),
            recovery_seconds=recovery_seconds,
            recovery_peak_bytes=recovery_peak,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=20_000)
    parser.add_argument("--include-recovery", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.elements,
                include_recovery=arguments.include_recovery,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
