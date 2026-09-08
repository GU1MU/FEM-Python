"""Worker-path equivalence tests for the Phase-3 planar compile worker.

Phase 3 (方案 D) of the planar feature-chain plan: the planar construction
compile runs on a dedicated worker thread (gmsh affinity) and must return a
field-for-field identical result to the Phase-2 main-thread compile.  These
tests drive the real ``prepare_planar_construction_proposal`` dispatch, spy
on the worker boundary, and compare against a direct same-thread compile.
"""

from __future__ import annotations

from copy import deepcopy
import threading

from fem.application import (
    ModelSession,
    compile_planar_construction,
    compile_planar_feature_recipe,
)
from fem_agent.tools.registry import ToolExecutionContext
from fem.geometry.construction_ir import PlanarConstructionIR
import fem_gui.agent_authoring as agent_authoring
from tests.helpers.agent_planar_construction import (
    make_planar_authoring_controller as _controller,
    build_rectangle_arguments,
)
from tests.helpers.fixtures.planar_construction_phase0 import EXPECTED_H_CONSTRUCTION
from tests.helpers.fixtures.planar_feature_chain_baseline import (
    feature_recipe_fingerprint,
)

import pytest


pytestmark = pytest.mark.local_session


def _spy_worker(monkeypatch) -> dict[str, object]:
    """Capture the worker payload and the thread that produced it."""

    captured: dict[str, object] = {}
    original = agent_authoring.run_owned_worker

    def spy(workload, **kwargs):
        def wrapped():
            captured["thread"] = threading.get_ident()
            result = workload()
            captured["payload"] = result
            return result

        return original(wrapped, **kwargs)

    monkeypatch.setattr(agent_authoring, "run_owned_worker", spy)
    return captured


def _dispatch(controller, construction: dict[str, object], *, key: str):
    return controller.dispatch(
        "prepare_planar_construction_proposal",
        {
            "part_function": "二维板",
            "construction": construction,
            "output": "planar",
        },
        ToolExecutionContext("phase3-worker", 0, key),
    )


def test_worker_path_matches_direct_compile_field_for_field(
    real_gmsh, monkeypatch,
) -> None:
    del real_gmsh
    captured = _spy_worker(monkeypatch)
    _bridge, controller = _controller(ModelSession())

    raw = build_rectangle_arguments()["construction"]
    raw["nodes"].extend([
        {"id": "hole", "kind": "circle", "center_x": 20, "center_y": 20, "radius": 2},
        {"id": "result", "kind": "difference", "base": "plate", "subtract": ["hole"]},
    ])
    raw["result_node_id"] = "result"
    result = _dispatch(controller, raw, key="equivalence")

    assert result.ok is True
    assert captured["thread"] != threading.get_ident()
    worker_compiled, worker_feature, kind, _recipe, _mesh = captured["payload"]
    assert kind == "planar"

    construction = PlanarConstructionIR.from_dict(raw)
    direct_compiled = compile_planar_construction(construction)
    direct_feature = compile_planar_feature_recipe(
        construction, compiled=direct_compiled
    )

    assert worker_compiled.proof == direct_compiled.proof
    assert feature_recipe_fingerprint(
        worker_feature
    ) == feature_recipe_fingerprint(direct_feature)


def test_cancelled_compile_stops_at_checkpoint_and_keeps_model(
    real_gmsh, monkeypatch,
) -> None:
    del real_gmsh
    _bridge, controller = _controller(ModelSession())

    # Gate the first CAD model open until the cancel event fires, so the
    # worker deterministically stops at a cancellation checkpoint.
    def gated_factory(cancel_event):
        def factory(*args, **kwargs):
            # Cancellation fires within ~0.2s; the bound only guards the gate
            # itself and must respect the GUI test real-wait policy.
            if cancel_event.wait(timeout=2.0):
                raise agent_authoring.PlanarCompileCancelled(
                    "test gate observed cancellation"
                )
            raise AssertionError("gate opened before cancellation")

        return factory

    monkeypatch.setattr(
        agent_authoring, "_cancellable_planar_model_factory", gated_factory
    )

    def cancel_soon() -> None:
        threading.Event().wait(0.2)
        controller.cancel_turn("test cancellation")

    canceller = threading.Thread(target=cancel_soon, daemon=True)
    canceller.start()
    result = _dispatch(
        controller, deepcopy(EXPECTED_H_CONSTRUCTION), key="cancelled"
    )
    canceller.join(timeout=2.0)

    assert result.ok is False
    assert result.data["diagnostic"]["code"] == "planar-ir.cancelled"
    assert result.data["diagnostic"]["model_unchanged"] is True
