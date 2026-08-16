"""Worker-path equivalence tests for the Phase-3 planar compile worker.

Phase 3 (方案 D) of the planar feature-chain plan: the planar construction
compile runs on a dedicated worker thread (gmsh affinity) and must return a
field-for-field identical result to the Phase-2 main-thread compile.  These
tests drive the real ``prepare_planar_construction_proposal`` dispatch, spy
on the worker boundary, and compare against a direct same-thread compile.
"""

from __future__ import annotations

from copy import deepcopy
import math
import threading

import pytest

from fem.application import (
    ModelSession,
    compile_planar_construction,
    compile_planar_feature_recipe,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem.geometry.construction_ir import PlanarConstructionIR
import fem_gui.agent_authoring as agent_authoring
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from tests.helpers.fixtures.planar_construction_phase0 import EXPECTED_H_CONSTRUCTION
from tests.helpers.fixtures.planar_feature_chain_baseline import (
    BASELINE_AREA,
    BASELINE_BOUNDING_BOX,
    BASELINE_COMPONENT_COUNT,
    BASELINE_FEATURE_RECIPE_SHA256,
    BASELINE_HOLE_COUNT,
    PLATE_SLOT_SHU_IR_DICT,
    feature_recipe_fingerprint,
)


def _controller(session: ModelSession):
    holder: dict[str, object] = {}

    def refresh() -> None:
        bridge.bind_snapshot(session.snapshot())
        controller = holder.get("controller")
        if controller is not None:
            controller.observe_binding(bridge.context)  # type: ignore[arg-type]

    bridge = AgentAuthoringBridge(SessionGeometryAuthoringPort(session, refresh))
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    holder["controller"] = controller
    return bridge, controller


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

    result = _dispatch(
        controller, deepcopy(EXPECTED_H_CONSTRUCTION), key="equivalence"
    )

    assert result.ok is True
    assert captured["thread"] != threading.get_ident()
    worker_compiled, worker_feature, kind, _recipe, _mesh = captured["payload"]
    assert kind == "planar"

    construction = PlanarConstructionIR.from_dict(EXPECTED_H_CONSTRUCTION)
    direct_compiled = compile_planar_construction(construction)
    direct_feature = compile_planar_feature_recipe(
        construction, compiled=direct_compiled
    )

    assert worker_compiled.proof == direct_compiled.proof
    assert feature_recipe_fingerprint(
        worker_feature
    ) == feature_recipe_fingerprint(direct_feature)


@pytest.mark.slow
def test_worker_path_reproduces_baseline_oracle(real_gmsh, monkeypatch) -> None:
    del real_gmsh
    captured = _spy_worker(monkeypatch)
    _bridge, controller = _controller(ModelSession())

    result = _dispatch(
        controller, deepcopy(PLATE_SLOT_SHU_IR_DICT), key="baseline-oracle"
    )

    assert result.ok is True
    worker_compiled, worker_feature, *_ = captured["payload"]
    proof = worker_compiled.proof
    assert math.isclose(proof.area, BASELINE_AREA, rel_tol=1.0e-9, abs_tol=1.0e-6)
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-7)
        for left, right in zip(proof.bounding_box, BASELINE_BOUNDING_BOX)
    )
    assert proof.component_count == BASELINE_COMPONENT_COUNT
    assert proof.hole_count == BASELINE_HOLE_COUNT
    assert feature_recipe_fingerprint(worker_feature) == BASELINE_FEATURE_RECIPE_SHA256


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
