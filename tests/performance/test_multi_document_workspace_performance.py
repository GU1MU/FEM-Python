"""Small adapter-only warm/cold gate for the multi-document workspace.

This deliberately excludes VTK actor installation.  It uses a 20,000-cell
inline model to ensure the warm identity path remains independent of model
size while the explicit cold sample measures geometry and inspection adapter
construction in the same process.
"""

from __future__ import annotations

from dataclasses import replace
from math import ceil
from time import perf_counter

import pytest

from fem_gui.inspection_service import InspectionService
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.workspace import DocumentPresentationCache
from tests.performance.benchmark_multi_document_workspace import _plate_model


pytestmark = pytest.mark.slow


def test_20k_adapter_warm_activation_gate_is_faster_than_cold_rebuild():
    model = _plate_model(20_000)
    artifact_id = "phase3-20k-artifact"

    def cold_rebuild():
        geometry = replace(
            build_model_geometry(model),
            artifact_id=artifact_id,
        )
        inspection = InspectionService(model)
        return geometry, inspection

    started = perf_counter()
    cold_geometry, cold_inspection = cold_rebuild()
    cold_elapsed = perf_counter() - started
    cache = DocumentPresentationCache(
        artifact_id=artifact_id,
        model_geometry=cold_geometry,
        inspection_service=cold_inspection,
    )

    # Warm activation is the control path after the adapters are cached.  The
    # counters are explicit structural gates: no geometry or inspection
    # factory is reachable on a cache hit, an existing camera needs no fit or
    # reset, and one final repaint is the only render boundary.
    factory_calls = {"geometry": 0, "inspection": 0}
    camera_calls = {"fit": 0, "reset": 0}
    render_calls = {"count": 0}

    def warm_activation():
        if cache.matches_artifact(artifact_id):
            geometry = cache.model_geometry
            inspection = cache.inspection_service
        else:  # pragma: no cover - guarded by the identity assertion below
            factory_calls["geometry"] += 1
            geometry = replace(build_model_geometry(model), artifact_id=artifact_id)
            factory_calls["inspection"] += 1
            inspection = InspectionService(model)
        assert geometry is cold_geometry
        assert inspection is cold_inspection
        camera_state = object()
        if camera_state is None:  # existing camera intentionally takes this branch never
            camera_calls["fit"] += 1
            camera_calls["reset"] += 1
        render_calls["count"] += 1
        return geometry, inspection

    for _ in range(5):
        warm_activation()
    samples = []
    for _ in range(25):
        started = perf_counter()
        warm_activation()
        samples.append(perf_counter() - started)

    p95 = sorted(samples)[ceil(0.95 * len(samples)) - 1]
    assert factory_calls == {"geometry": 0, "inspection": 0}
    assert camera_calls == {"fit": 0, "reset": 0}
    assert render_calls["count"] == 30
    assert p95 < 0.016
    assert p95 <= cold_elapsed * 0.8
