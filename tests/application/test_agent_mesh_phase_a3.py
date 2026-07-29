from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import time

import pytest

from fem.application import ModelSession, UnitContext
from fem.core.model import FEMModel
from fem.geometry import PlateWithHoleGeometry
from fem.geometry.gmsh_coordinator import (
    GmshExecutionCancelled,
    GmshExecutionCoordinator,
)
from fem.mesh.settings import MeshSettings
from tests.helpers.mesh_builders import (
    make_selection_mixed_plane_mesh,
    make_selection_quad_mesh,
)


def _recipe() -> PlateWithHoleGeometry:
    return PlateWithHoleGeometry(
        "实体-偏心孔板",
        10.0,
        6.0,
        6.5,
        2.0,
        1.0,
    )


def _model(name: str, *, mixed: bool = False) -> FEMModel:
    return FEMModel(
        mesh=(
            make_selection_mixed_plane_mesh()
            if mixed
            else make_selection_quad_mesh()
        ),
        name=name,
    )


def _meshed_session() -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        _recipe(),
        part_name="部件-偏心孔板",
    )
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(1.0, cell_shape="triangle"),
    )
    task = session.prepare_mesh_generation()
    accepted = session.accept_generated_model(task.token, _model("旧网格"))
    assert accepted.accepted
    return session


def _agent_settings() -> MeshSettings:
    return MeshSettings(
        0.5,
        cell_shape="quadrilateral",
        auto_level=4,
        strict_cell_shape=True,
    )


def test_a3_prepare_keeps_current_intent_and_mesh_then_accepts_both_atomically() -> (
    None
):
    session = _meshed_session()
    before = session.snapshot()

    task = session.prepare_agent_mesh_generation(
        "P1",
        _agent_settings(),
        "a" * 64,
        expected_session_revision=before.session_revision,
    )
    prepared = session.snapshot()

    assert prepared.session_revision == before.session_revision
    assert prepared.parts[0].mesh_settings == before.parts[0].mesh_settings
    assert prepared.artifact == before.artifact
    assert task.parts[0].mesh_settings == _agent_settings()
    delta = session.accept_agent_generated_model(
        task.token,
        _model("新网格"),
    )
    after = session.snapshot()
    assert delta.accepted
    assert after.session_revision == before.session_revision + 1
    assert after.parts[0].mesh_settings == _agent_settings()
    assert after.artifact is not None
    assert after.artifact.model.name == "新网格"
    assert session.validate_task_token(task.token).value == "already_completed"


def test_a3_failed_accept_and_explicit_termination_keep_old_mesh() -> None:
    session = _meshed_session()
    before = session.snapshot()
    task = session.prepare_agent_mesh_generation(
        "P1",
        _agent_settings(),
        "b" * 64,
        expected_session_revision=before.session_revision,
    )

    class BrokenCandidate:
        def __deepcopy__(self, _memo):
            raise RuntimeError("candidate copy failed")

    with pytest.raises(RuntimeError, match="candidate copy failed"):
        session.accept_agent_generated_model(task.token, BrokenCandidate())

    assert session.snapshot() == before
    terminated = session.terminate_agent_mesh_task(
        task.token,
        "strict quadrilateral generation failed",
    )
    assert terminated.accepted
    assert session.snapshot() == before
    assert session.validate_task_token(task.token).value == "already_completed"


def test_a3_cancelled_task_consumes_token_and_keeps_old_mesh() -> None:
    session = _meshed_session()
    before = session.snapshot()
    task = session.prepare_agent_mesh_generation(
        "P1",
        _agent_settings(),
        "c" * 64,
        expected_session_revision=before.session_revision,
    )

    receipt = session.terminate_agent_mesh_task(task.token, "cancelled")

    assert receipt.accepted
    assert session.snapshot() == before
    assert session.validate_task_token(task.token).value == "already_completed"


def test_a3_stale_result_is_discarded_then_token_is_consumed() -> None:
    session = _meshed_session()
    task = session.prepare_agent_mesh_generation(
        "P1",
        _agent_settings(),
        "d" * 64,
        expected_session_revision=session.session_revision,
    )
    session.rename_native_model("模型-用户修改")
    current = session.snapshot()

    stale = session.accept_agent_generated_model(task.token, _model("晚到网格"))

    assert stale.accepted is False
    assert stale.token_status is not None
    assert session.snapshot() == current
    session.terminate_agent_mesh_task(task.token, "stale")
    assert session.snapshot() == current
    assert session.validate_task_token(task.token).value == "already_completed"


def test_a3_process_gmsh_coordinator_serializes_threads_and_recovers() -> None:
    coordinator = GmshExecutionCoordinator()
    state_lock = Lock()
    active = 0
    maximum = 0

    def own(index: int) -> int:
        nonlocal active, maximum
        with coordinator.acquire(f"task-{index}"):
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
        return index

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert sorted(pool.map(own, range(6))) == list(range(6))

    assert maximum == 1
    with pytest.raises(RuntimeError, match="boom"):
        with coordinator.acquire("failing owner"):
            raise RuntimeError("boom")
    with coordinator.acquire("owner after failure"):
        assert coordinator.snapshot().depth == 1
    assert coordinator.snapshot().owner_thread_id is None


def test_a3_waiting_gmsh_owner_is_cancellable() -> None:
    coordinator = GmshExecutionCoordinator()
    cancel = Event()
    entered = Event()
    release = Event()

    def hold() -> None:
        with coordinator.acquire("holder"):
            entered.set()
            release.wait(timeout=2.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        holder = pool.submit(hold)
        assert entered.wait(timeout=1.0)
        waiting = pool.submit(
            coordinator.acquire,
            "cancelled waiter",
            cancelled=cancel,
            poll_interval=0.01,
        )
        cancel.set()
        with pytest.raises(GmshExecutionCancelled):
            waiting.result(timeout=1.0)
        release.set()
        holder.result(timeout=1.0)
