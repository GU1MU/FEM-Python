from __future__ import annotations

import os
from threading import Event
from time import monotonic, perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem.application import MeshEntityRef, NamedRegion, NamedRegionEditBatch
from fem.core.mesh import Mesh3D, Node3D
from fem.core.model import FEMModel
from fem_gui.commands import GuiCommandStatus
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.task_controller import BackgroundTaskState
from fem_gui.visualization.model_adapter import build_model_geometry


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _process_until(predicate, timeout: float = 5.0) -> None:
    application = _application()
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline:
        application.processEvents()
        Event().wait(0.001)
    application.processEvents()
    assert predicate()


def _loaded_window(gui_inp_path) -> FEMMainWindow:
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    window.scope_background_reference_threshold = 0
    return window


def _single_node_batch(window: FEMMainWindow, name: str) -> NamedRegionEditBatch:
    node_id = int(window.document.model.mesh.nodes[0].id)
    return NamedRegionEditBatch(
        window.document.session_revision,
        (NamedRegion(name, (MeshEntityRef.node(node_id),)),),
    )


def test_large_scope_task_launch_stays_below_50_ms(monkeypatch, tmp_path) -> None:
    _application()
    window = FEMMainWindow()
    model = FEMModel(
        Mesh3D(
            [
                Node3D(index, float(index), 0.0, 0.0)
                for index in range(1, 100_001)
            ],
            [],
        )
    )
    imported = window.session.prepare_import(tmp_path / "large.inp")
    window.session.accept_imported_model(imported.token, model)
    window.document = window.session.projection_snapshot()
    window.scope_background_reference_threshold = 1
    references = tuple(
        MeshEntityRef.node(index) for index in range(1, 100_001)
    )
    batch = NamedRegionEditBatch(
        window.document.session_revision,
        (NamedRegion("Large", references),),
    )
    captured = []

    def capture_start(workload, **options):
        captured.append((workload, options))
        return 1

    monkeypatch.setattr(window.task_controller, "start", capture_start)
    started = perf_counter()
    receipt = window.apply_named_region_edit(batch)
    elapsed = perf_counter() - started

    assert receipt.status is GuiCommandStatus.PENDING
    assert elapsed < 0.05
    assert len(captured) == 1
    window.close()


def test_background_scope_success_uses_incremental_projection(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    window = _loaded_window(gui_inp_path)
    topology = window._scope_selection_topology()
    mesh = window.document.model.mesh
    install_calls = []
    monkeypatch.setattr(
        window,
        "_install_model",
        lambda *args, **kwargs: install_calls.append((args, kwargs)),
    )

    receipt = window.apply_named_region_edit(
        _single_node_batch(window, "AsyncNode")
    )
    terminal = receipt.completion.result(5.0)

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert "AsyncNode" in window.document.named_regions
    assert window.document.model.mesh is mesh
    assert window._scope_selection_topology_cache is topology
    assert install_calls == []
    assert not window.busy
    window.close()


def test_dependent_scope_workflow_resumes_only_after_background_commit(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    window = _loaded_window(gui_inp_path)
    entered = Event()
    release = Event()
    original = main_window_module.compile_named_region_edit

    def blocked(task):
        entered.set()
        release.wait(5.0)
        return original(task)

    monkeypatch.setattr(main_window_module, "compile_named_region_edit", blocked)
    reference = MeshEntityRef.node(
        int(window.document.model.mesh.nodes[0].id)
    )
    resumed = []

    name = window._create_region_from_current_mesh_selection(
        requested_name="Deferred",
        references=(reference,),
        on_committed=resumed.append,
    )
    _process_until(entered.is_set)

    assert name is None
    assert resumed == []
    assert "Deferred" not in window.document.named_regions

    release.set()
    _process_until(lambda: not window.busy and resumed == ["Deferred"])

    assert "Deferred" in window.document.named_regions
    window.close()


def test_background_scope_cancel_is_busy_and_atomic(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    window = _loaded_window(gui_inp_path)
    entered = Event()
    release = Event()
    original = main_window_module.compile_named_region_edit

    def blocked(task):
        entered.set()
        release.wait(5.0)
        return original(task)

    monkeypatch.setattr(main_window_module, "compile_named_region_edit", blocked)
    before = window.session.snapshot()
    receipt = window.apply_named_region_edit(
        _single_node_batch(window, "Cancelled")
    )
    _process_until(entered.is_set)

    assert receipt.status is GuiCommandStatus.PENDING
    assert window.busy
    assert window.cancel_current_task()
    release.set()
    terminal = receipt.completion.result(5.0)

    assert terminal.state is BackgroundTaskState.CANCELLED
    assert not window.busy
    after = window.session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.named_regions == before.named_regions
    assert after.model.node_sets == before.model.node_sets
    window.close()


def test_later_scope_edit_supersedes_running_request(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    window = _loaded_window(gui_inp_path)
    entered = Event()
    release = Event()
    original = main_window_module.compile_named_region_edit
    call_count = 0

    def block_first(task):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            entered.set()
            release.wait(5.0)
        return original(task)

    monkeypatch.setattr(
        main_window_module,
        "compile_named_region_edit",
        block_first,
    )
    first = window.apply_named_region_edit(
        _single_node_batch(window, "First")
    )
    _process_until(entered.is_set)
    second = window.apply_named_region_edit(
        _single_node_batch(window, "Second")
    )
    release.set()
    first_terminal = first.completion.result(5.0)
    second_terminal = second.completion.result(5.0)

    assert first_terminal.state is BackgroundTaskState.CANCELLED
    assert second_terminal.state is BackgroundTaskState.SUCCEEDED
    assert tuple(window.document.named_regions) == ("Second",)
    assert call_count == 2
    _process_until(lambda: not window.busy)
    assert not window.busy
    window.close()


def test_background_scope_exception_shows_one_error_and_keeps_ui_atomic(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    window = _loaded_window(gui_inp_path)
    shown = []
    monkeypatch.setattr(
        main_window_module,
        "compile_named_region_edit",
        lambda _task: (_ for _ in ()).throw(RuntimeError("scope exploded")),
    )
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: shown.append((title, message)),
    )
    before = window.session.snapshot()
    artifact_id = window.viewport.artifact_id
    tree_root = window.model_tree.topLevelItem(0).text(0)

    receipt = window.apply_named_region_edit(
        _single_node_batch(window, "Failure")
    )
    terminal = receipt.completion.result(5.0)
    _process_until(lambda: not window.busy)

    assert terminal.state is BackgroundTaskState.FAILED
    assert shown == [("提交作用域失败", "scope exploded")]
    after = window.session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.named_regions == before.named_regions
    assert window.viewport.artifact_id == artifact_id
    assert window.model_tree.topLevelItem(0).text(0) == tree_root
    window.close()
