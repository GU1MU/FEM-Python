from __future__ import annotations

import math
from time import perf_counter


import numpy as np

from fem.application.results import (
    FieldPosition,
    ResultSourceKey,
    ScalarFieldSelection,
    build_result_provider,
    execute_output_requests,
    prepare_result_export_snapshot,
)
from fem.core.model import (
    OutputRequest,
)
from fem.elements import (
    get_element_kernel,
)
from fem.post.stress.beam import recover_integration_point_stress
from fem.solvers import static_linear


from tests.helpers.beam_section_builders import (
    _inline_cantilever,
    _stress_fields,
)


def _best_batch_time(action, *, repeats: int) -> float:
    best = math.inf
    for _sample in range(3):
        checksum = 0.0
        started = perf_counter()
        for _repeat in range(repeats):
            checksum += float(action())
        best = min(best, perf_counter() - started)
        assert math.isfinite(checksum)
    return best


def _section_resultant_workload(result) -> float:
    mesh = result.model.mesh
    element = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    forces = kernel.local_integration_point_forces(mesh, element, result.U)
    return sum(
        abs(value)
        for value in (
            forces.N,
            forces.Vy,
            forces.Vz,
            forces.T,
            forces.My,
            forces.Mz,
        )
    )


def _four_point_workload(result) -> float:
    recovered = recover_integration_point_stress(result)
    point_values = sum(
        abs(component)
        for field in recovered.section_points
        for row in field.rows
        for component in row.values().values()
    )
    return point_values


def _export_switch_checksum(export) -> float:
    return float(
        export.materialization_generation
        + len(export.field.locations)
        + len(export.field.descriptor.columns)
    )


def benchmark(repeats: int, switches: int) -> dict:
    result = static_linear.solve(
        _inline_cantilever("rectangle", {"height": 0.4, "width": 0.2}),
        "Load",
    )
    source = ResultSourceKey(
        "performance-result",
        "performance-session",
        "performance-artifact",
        1,
        "Load",
        "performance-run",
    )
    base_provider = build_result_provider(source, result)
    output_requests = (OutputRequest("field", "element", ("S",)),)
    outcome = execute_output_requests(base_provider, output_requests)
    provider = outcome.provider_draft
    stress_fields = _stress_fields(provider)

    value_bytes = sum(field.values.nbytes for field in stress_fields)
    section_resultant_bytes = 6 * np.dtype(float).itemsize
    byte_ratio = value_bytes / section_resultant_bytes

    section_resultant_seconds = _best_batch_time(
        lambda: _section_resultant_workload(result),
        repeats=repeats,
    )
    four_point_materialization_seconds = _best_batch_time(
        lambda: _four_point_workload(result),
        repeats=repeats,
    )

    point_selections = tuple(
        ScalarFieldSelection(field.key, "S11")
        for field in stress_fields
        if field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
    )
    base_switch_seconds = _best_batch_time(
        lambda: _export_switch_checksum(
            prepare_result_export_snapshot(
                provider.snapshot,
                point_selections[0],
            )
        ),
        repeats=switches,
    )
    point_index = 0

    def switch_point() -> float:
        nonlocal point_index
        export = prepare_result_export_snapshot(
            provider.snapshot,
            point_selections[point_index % len(point_selections)],
        )
        point_index += 1
        return _export_switch_checksum(export)

    four_point_switch_seconds = _best_batch_time(
        switch_point,
        repeats=switches,
    )

    metrics = {
        "materialization_repeats": repeats,
        "export_switches": switches,
        "section_resultant_value_bytes": section_resultant_bytes,
        "four_point_section_value_bytes": value_bytes,
        "four_point_value_byte_ratio": byte_ratio,
        "section_resultant_materialization_seconds": (
            section_resultant_seconds
        ),
        "four_point_materialization_seconds": (four_point_materialization_seconds),
        "four_point_materialization_time_ratio": (
            four_point_materialization_seconds / section_resultant_seconds
        ),
        "base_switch_seconds": base_switch_seconds,
        "four_point_switch_seconds": four_point_switch_seconds,
    }
    return metrics


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Measure beam section-point materialization and export switching.")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--switches", type=int, default=1200)
    args = parser.parse_args()
    if args.repeats < 1 or args.switches < 1:
        parser.error("workload counts must be positive")
    print(json.dumps(benchmark(args.repeats, args.switches), indent=2))
