from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import fem_gui.main_window as main_window_module
from fem.io import save_result_archive
from fem_gui.main_window import FEMMainWindow
from fem_gui.main_window import DisplayState
from fem_gui.workspace import canonical_path
from fem_gui.task_controller import BackgroundTaskState
from fem_gui.widgets.result_tree import (
    ROLE_DOCUMENT_ID,
    ROLE_RESULT_KIND,
    ROLE_RESULT_SOURCE,
    ROLE_RUN_ID,
    ROLE_SELECTION,
    ResultTree,
)
from tests.gui.test_result_document_workflow_phase5 import _snapshot
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)
from tests.gui.test_result_tree import _catalog


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_open(window: FEMMainWindow, receipt) -> None:
    assert receipt.completion is not None
    terminal = receipt.completion.result(5.0)
    app = _application()
    controller = window.workspace.open_controller
    deadline = time.monotonic() + 5.0
    while (
        controller is not None
        and controller.busy
        and time.monotonic() < deadline
    ):
        app.processEvents()
        time.sleep(0.001)
    app.processEvents()
    assert not (controller is not None and controller.busy)
    assert terminal.state is BackgroundTaskState.SUCCEEDED


def _archive_path(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.femres"
    save_result_archive(
        path,
        _snapshot(make_continuum_nodal_semantics_result, name),
    )
    return path


def _open_archive(window: FEMMainWindow, path: Path) -> None:
    _wait_open(window, window.open_result_path(path))


def _projection(catalog, name: str = "model") -> SimpleNamespace:
    run = SimpleNamespace(
        run_id=catalog.source.run_id,
        name="Job-1",
        step_name=catalog.source.step_name,
        has_result=True,
    )
    return SimpleNamespace(
        model_name=name,
        source_path=Path(f"{name}.femres"),
        runs=(run,),
        displayed_result_run_id=run.run_id,
    )


def _first_leaf(item, selection=None):
    if item.data(0, ROLE_SELECTION) is not None and item.childCount() == 0:
        if selection is None or item.data(0, ROLE_SELECTION) == selection:
            return item
    for index in range(item.childCount()):
        leaf = _first_leaf(item.child(index), selection)
        if leaf is not None:
            return leaf
    return None


def test_result_tree_appends_model_and_archive_roots_incrementally(monkeypatch) -> None:
    _application()
    tree = ResultTree()
    catalog = _catalog()
    projection = _projection(catalog, "A")
    tree.upsert_model_runs(11, projection, display_name="A", catalog=catalog)
    tree.upsert_archive(22, projection, display_name="external.femres", catalog=catalog)

    assert set(tree.roots) == {11, 22}
    assert tree.topLevelItemCount() == 2
    assert tree.roots[11].text(0) == "A"
    assert tree.roots[11].child(0).text(0) == "Job-1"
    assert tree.roots[22].text(0) == "external"
    assert tree.roots[22].toolTip(0).endswith("A.femres")

    clear_calls: list[bool] = []
    monkeypatch.setattr(tree, "clear", lambda: clear_calls.append(True))
    previous = tree.roots[11]
    tree.upsert_model_runs(11, _projection(catalog, "A-updated"), catalog=catalog)
    assert clear_calls == []
    assert tree.roots[22].text(0) == "external"
    assert tree.roots[11] is not previous


def test_result_field_items_carry_document_run_source_and_typed_selection() -> None:
    _application()
    tree = ResultTree()
    catalog = _catalog()
    tree.upsert_model_runs(7, _projection(catalog), catalog=catalog)
    leaf = _first_leaf(tree.roots[7], catalog.default_selection)
    assert leaf is not None
    assert leaf.data(0, ROLE_DOCUMENT_ID) == 7
    assert leaf.data(0, ROLE_RUN_ID) == catalog.source.run_id
    assert leaf.data(0, ROLE_RESULT_SOURCE) == catalog.source
    assert leaf.data(0, ROLE_SELECTION) == catalog.default_selection

    routed = []
    tree.fieldSelectionRouted.connect(
        lambda document_id, run_id, source, selection: routed.append(
            (document_id, run_id, source, selection)
        )
    )
    tree._activate_item(leaf)
    assert routed == [
        (7, catalog.source.run_id, catalog.source, catalog.default_selection)
    ]


def test_result_root_removal_is_indexed_and_preserves_other_documents() -> None:
    _application()
    tree = ResultTree()
    catalog = _catalog()
    projection = _projection(catalog)
    tree.upsert_model_runs(1, projection, catalog=catalog)
    tree.upsert_archive(2, projection, catalog=catalog)
    assert tree.remove_archive(2)
    assert set(tree.roots) == {1}
    assert tree.topLevelItemCount() == 1
    assert not tree.remove_archive(2)


def test_result_run_item_emits_routed_activation() -> None:
    _application()
    tree = ResultTree()
    catalog = _catalog()
    tree.upsert_model_runs(19, _projection(catalog), catalog=catalog)
    run_item = tree.roots[19].child(0)
    assert run_item.data(0, ROLE_RESULT_KIND) == "run"
    activated = []
    tree.runActivated.connect(
        lambda document_id, run_id: activated.append((document_id, run_id))
    )
    tree._activate_item(run_item)
    assert activated == [(19, catalog.source.run_id)]


def test_same_selection_is_scoped_to_document_and_source() -> None:
    _application()
    tree = ResultTree()
    first = _catalog()
    second = replace(
        first,
        source=replace(
            first.source,
            result_id="result-2",
            session_id="session-2",
            artifact_id="artifact-2",
            run_id="run-2",
        ),
    )
    tree.upsert_model_runs(31, _projection(first, "A"), catalog=first)
    tree.upsert_model_runs(32, _projection(second, "B"), catalog=second)
    selection = first.default_selection
    assert tree.select_selection(
        selection,
        document_id=32,
        source=second.source,
    )
    assert tree.currentItem().data(0, ROLE_DOCUMENT_ID) == 32
    assert tree.currentItem().data(0, ROLE_RESULT_SOURCE) == second.source
    assert not tree.select_selection(
        selection,
        document_id=32,
        source=first.source,
    )


def test_main_window_appends_three_results_and_duplicate_is_index_hit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    paths = tuple(_archive_path(tmp_path, f"append-{index}") for index in range(3))
    load_calls: list[Path] = []
    geometry_calls: list[object] = []
    view_calls: list[object] = []
    inspection_calls: list[object] = []
    original_load = main_window_module.load_result_archive
    original_geometry = main_window_module.build_result_archive_geometry
    original_view = main_window_module.build_result_archive_model_view
    original_inspection = main_window_module.InspectionService

    def counted_load(path):
        load_calls.append(Path(path))
        return original_load(path)

    monkeypatch.setattr(main_window_module, "load_result_archive", counted_load)
    monkeypatch.setattr(
        main_window_module,
        "build_result_archive_geometry",
        lambda projection: geometry_calls.append(projection)
        or original_geometry(projection),
    )
    monkeypatch.setattr(
        main_window_module,
        "build_result_archive_model_view",
        lambda *args, **kwargs: view_calls.append(args)
        or original_view(*args, **kwargs),
    )

    def counted_inspection(*args, **kwargs):
        inspection_calls.append(args)
        return original_inspection(*args, **kwargs)

    monkeypatch.setattr(main_window_module, "InspectionService", counted_inspection)
    window = FEMMainWindow()
    try:
        for path in paths:
            _open_archive(window, path)
        assert len(window.workspace.results) == 3
        assert len(load_calls) == 3
        assert len(geometry_calls) == 3
        assert len(view_calls) == 3
        assert len(inspection_calls) == 3
        result_ids = set(window.workspace.results)
        assert result_ids <= set(window.result_tree.roots)
        active_before = window.workspace.active_document_id
        duplicate = window.open_result_path(paths[1])
        assert duplicate.status.value == "accepted"
        assert len(load_calls) == 3
        assert len(geometry_calls) == 3
        assert len(view_calls) == 3
        assert len(inspection_calls) == 3
        assert window.workspace.active_document_id == window.workspace.result_paths[
            canonical_path(paths[1])
        ]
        assert active_before != window.workspace.active_document_id
    finally:
        window.close()


def test_main_window_failed_archive_open_is_atomic(tmp_path: Path, monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    try:
        before_context = window.workspace.active_document_id
        before_roots = set(window.result_tree.roots)

        def fail_load(_path):
            raise ValueError("invalid archive")

        monkeypatch.setattr(main_window_module, "load_result_archive", fail_load)
        monkeypatch.setattr(window, "_show_command_rejection", lambda *_args: None)
        monkeypatch.setattr(window, "_show_error", lambda *_args: None)
        monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
        receipt = window.open_result_path(tmp_path / "broken.femres")
        assert receipt.completion is not None
        terminal = receipt.completion.result(5.0)
        assert terminal.state is BackgroundTaskState.FAILED
        _application().processEvents()
        assert window.workspace.active_document_id == before_context
        assert set(window.result_tree.roots) == before_roots
        assert not window.workspace.results
        assert not window.workspace.result_paths
    finally:
        window.close()


def test_result_presentation_state_isolated_across_warm_a_b_a_switch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    try:
        path_a = _archive_path(tmp_path, "warm-a")
        path_b = _archive_path(tmp_path, "warm-b")
        _open_archive(window, path_a)
        context_a = window.workspace.active_document()
        assert context_a is not None
        window._scale_mode = "custom"
        window._scale_value = 3.5
        window._contour_options["minimum"] = 42.0
        window._overlay_undeformed = True
        window._display = DisplayState("deformed", True)
        selection_a = window.result_selection
        source_a = context_a.session.current_result_identity()[0]
        _open_archive(window, path_b)
        context_b = window.workspace.active_document()
        assert context_b is not None and context_b is not context_a
        assert window._contour_options["minimum"] != 42.0
        assert "averaging_threshold" in window._contour_options
        assert window._scale_mode == "auto"
        assert not window._overlay_undeformed
        assert context_a.presentation_state.result_scale_mode == "custom"
        assert context_a.presentation_state.result_scale_value == 3.5
        assert context_a.presentation_state.contour_options["minimum"] == 42.0
        assert context_a.presentation_state.overlay_undeformed
        assert context_a.presentation_state.display_state == DisplayState(
            "deformed", True
        )
        assert context_a.presentation_state.result_selection == selection_a
        context_a.presentation_state.camera_state = object()
        restores: list[object] = []
        monkeypatch.setattr(
            main_window_module,
            "_restore_camera_state",
            lambda plotter, state: restores.append((plotter, state)),
        )
        fit_calls: list[bool] = []
        render_calls: list[bool] = []
        monkeypatch.setattr(
            window.viewport,
            "fit",
            lambda *args, **kwargs: fit_calls.append(True),
        )
        monkeypatch.setattr(
            window.viewport,
            "render",
            lambda *args, **kwargs: render_calls.append(True),
        )
        window.viewport._plotter = object()
        monkeypatch.setattr(window, "_apply_session_delta", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(window, "_project_viewport_for_module", lambda *_args, **_kwargs: None)
        assert window._activate_workspace_context(context_a)
        assert (
            window.result_tree.roots[context_a.document_id].data(
                0,
                ROLE_RESULT_SOURCE,
            )
            == source_a
        )
        assert restores
        assert fit_calls == []
        assert len(render_calls) == 1
        assert window.workspace.active_document_id == context_a.document_id
    finally:
        window.close()


def test_closing_external_result_releases_cache_and_preserves_other_root(
    tmp_path: Path,
) -> None:
    _application()
    window = FEMMainWindow()
    try:
        _open_archive(window, _archive_path(tmp_path, "close-a"))
        closed = window.workspace.active_document()
        assert closed is not None
        _open_archive(window, _archive_path(tmp_path, "close-b"))
        survivor = window.workspace.active_document()
        assert survivor is not None and survivor is not closed
        survivor_identity = survivor.presentation_cache.result_source
        closed_id = closed.document_id
        assert closed.presentation_cache.result_model_view is not None
        assert window.close_model(confirm=False, document_id=closed_id)
        assert closed.presentation_cache.result_model_view is None
        assert closed.presentation_cache.model_geometry is None
        assert closed.presentation_cache.result_source is None
        assert closed.presentation_cache.inspection_service is None
        assert closed_id not in window.workspace.results
        assert closed_id not in window.result_tree.roots
        assert survivor.document_id in window.workspace.results
        assert survivor.presentation_cache.result_source == survivor_identity
    finally:
        window.close()


def test_main_window_run_route_is_owned_by_document(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    try:
        context = window.workspace.add_model(display_name="Model-B")
        window.workspace.activate(context)
        window._active_context = context
        monkeypatch.setattr(window, "_activate_workspace_context", lambda _ctx: True)
        monkeypatch.setattr(
            context.session,
            "current_result_identity",
            lambda: None,
        )
        calls: list[str] = []
        monkeypatch.setattr(
            window,
            "select_run_result",
            lambda run_id: calls.append(run_id)
            or SimpleNamespace(diagnostic=None),
        )
        monkeypatch.setattr(window, "_refresh_result_tree_for_context", lambda *_args, **_kwargs: None)
        ribbon_calls: list[str] = []
        monkeypatch.setattr(
            window.ribbon,
            "set_current",
            lambda module_name: ribbon_calls.append(module_name),
        )
        window._activate_routed_result_run(context.document_id, "run-B")
        assert calls == ["run-B"]
        assert ribbon_calls == ["结果"]
    finally:
        window.close()
