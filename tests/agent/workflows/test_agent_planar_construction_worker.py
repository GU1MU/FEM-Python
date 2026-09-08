from __future__ import annotations

import threading

from fem.application import ModelSession
from fem_agent.tools.registry import ToolExecutionContext
import fem_gui.agent_authoring as agent_authoring
from tests.helpers.agent_planar_construction import (
    make_planar_authoring_controller as _controller,
    build_rectangle_arguments,
)


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


def test_planar_worker_returns_hole_proof_and_preview(
    real_gmsh, monkeypatch,
) -> None:
    del real_gmsh
    captured = _spy_worker(monkeypatch)
    session = ModelSession()
    _bridge, controller = _controller(session)

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

    assert worker_compiled.proof.equivalent
    assert worker_compiled.proof.material_profile_count == 1
    assert worker_compiled.proof.hole_count == 1
    assert worker_compiled.preview.faces
    assert 0 < len(worker_compiled.preview.points) <= 4096


def test_cancelled_compile_stops_at_checkpoint_and_keeps_model(
    real_gmsh, monkeypatch,
) -> None:
    del real_gmsh
    session = ModelSession()
    _bridge, controller = _controller(session)

    started = threading.Event()
    before = session.snapshot()
    def gated_factory(cancel_event):
        def factory(*args, **kwargs):
            started.set()
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
        assert started.wait(timeout=2.0)
        controller.cancel_turn("test cancellation")

    canceller = threading.Thread(target=cancel_soon, daemon=True)
    canceller.start()
    result = _dispatch(
        controller, build_rectangle_arguments()["construction"], key="cancelled"
    )
    canceller.join(timeout=2.0)

    assert result.ok is False
    assert result.data["diagnostic"]["code"] == "planar-ir.cancelled"
    assert result.data["diagnostic"]["model_unchanged"] is True
    assert not canceller.is_alive()
    assert session.snapshot() == before
