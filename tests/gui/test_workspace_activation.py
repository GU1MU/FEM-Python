from __future__ import annotations

import os
from pathlib import Path
from math import ceil
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from fem.application import ModelSession
from fem.application.definitions import MeshEntityRef
from fem.geometry import LogicalEntityRef
from fem_gui.main_window import FEMMainWindow
from fem_gui.workspace import DocumentPresentationCache
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.scene import DisplayState
from fem_gui.widgets.viewport import FEMViewport
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
)

import fem_gui.main_window as main_window_module


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _context_with_model(
    window: FEMMainWindow,
    name: str,
    *,
    model=None,
):
    model = make_static_pull_truss_model() if model is None else model
    session = ModelSession()
    initial_projection = session.projection_snapshot()
    task = session.prepare_import(Path(f"{name}.inp"))
    delta = session.accept_imported_model(task.token, model)
    context = window.workspace.add_model(
        session=session,
        projection=initial_projection,
        display_name=name,
        source_path=Path(f"{name}.inp"),
    )
    assert window._apply_session_delta(
        delta,
        context=context,
        model_geometry=build_model_geometry(model),
    )
    window._refresh_model_tree_for_context(context)
    return context


def test_activation_cache_reuses_geometry_and_inspection_identity(dispose_gui_widget):
    _application()
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        model = make_static_pull_truss_model()
        task = first.session.prepare_import(Path("A.inp"))
        delta = first.session.accept_imported_model(task.token, model)
        geometry = build_model_geometry(model)
        assert window._apply_session_delta(delta, model_geometry=geometry)
        first_geometry = window.geometry
        first_inspection = window.inspection_service
        assert first.presentation_cache.model_geometry is first_geometry

        second = _context_with_model(window, "B")
        assert window._activate_workspace_context(second)
        assert window._activate_workspace_context(first)
        assert window.geometry is first_geometry
        assert window.inspection_service is first_inspection
    finally:
        dispose_gui_widget(window)


def test_result_cache_identity_requires_a_cached_model_view():
    cache = DocumentPresentationCache(
        result_source="run",
        result_generation=3,
    )
    assert not cache.matches_result("run", 3)
    cache.result_model_view = object()
    assert cache.matches_result("run", 3)


def test_activation_native_preview_projection_passes_render_and_reset_false(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        preview = object()
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            window,
            "_current_native_geometry_preview",
            lambda: preview,
        )
        monkeypatch.setattr(
            window.viewport,
            "show_geometry_preview",
            lambda _preview, **kwargs: calls.append(kwargs),
        )
        window._workspace_activation = True
        window._project_viewport_for_module(
            "几何",
            render=False,
            reset_camera=False,
        )
        assert calls == [{"render": False, "reset_camera": False}]
    finally:
        window._workspace_activation = False
        dispose_gui_widget(window)


def test_nonactive_delta_updates_context_without_viewport_calls(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        second = _context_with_model(window, "B")
        assert window._activate_workspace_context(second)
        assert window._activate_workspace_context(first)
        calls: list[str] = []
        for name in (
            "set_model",
            "clear_model",
            "show_geometry_preview",
            "clear_selection",
            "render",
            "fit",
        ):
            monkeypatch.setattr(
                window.viewport,
                name,
                lambda *args, _name=name, **kwargs: calls.append(_name),
            )
        before = second.projection
        model = make_static_pull_truss_model()
        task = second.session.prepare_import(Path("B2.inp"))
        delta = second.session.accept_imported_model(task.token, model)
        assert window._apply_session_delta(delta, context=second)
        assert second.projection is not before
        assert calls == []
    finally:
        dispose_gui_widget(window)


def test_nonactive_result_identity_change_invalidates_result_cache(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        target = _context_with_model(window, "result-cache")
        old_source = object()
        new_source = object()
        cache = target.presentation_cache
        cache.result_source = old_source
        cache.result_generation = 1
        cache.result_model_view = object()
        monkeypatch.setattr(
            target.session,
            "current_result_identity",
            lambda: (new_source, 2),
        )
        model = make_static_pull_truss_model()
        task = target.session.prepare_import(Path("result-cache-2.inp"))
        delta = target.session.accept_imported_model(task.token, model)

        assert window._apply_session_delta(delta, context=target)
        assert not cache.matches_result(new_source, 2)
        assert cache.result_model_view is None
    finally:
        dispose_gui_widget(window)


def test_warm_cache_control_path_p95_under_16ms_for_20k_entries():
    """The non-VTK warm lookup remains O(1) with a 20k-entry workspace."""

    caches = {
        index: DocumentPresentationCache(
            artifact_id=f"artifact-{index}",
            model_geometry=object(),
        )
        for index in range(20_000)
    }
    target = caches[19_999]
    for _ in range(5):
        assert caches[19_999] is target
        assert target.matches_artifact("artifact-19999")

    samples: list[float] = []
    for _ in range(25):
        started = perf_counter()
        assert caches[19_999].matches_artifact("artifact-19999")
        samples.append(perf_counter() - started)
    p95 = sorted(samples)[ceil(0.95 * len(samples)) - 1]
    assert p95 < 0.016


def test_artifact_change_invalidates_only_target_cache(
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        model = make_static_pull_truss_model()
        task = first.session.prepare_import(Path("A.inp"))
        assert window._apply_session_delta(
            first.session.accept_imported_model(task.token, model),
            model_geometry=build_model_geometry(model),
        )
        second = _context_with_model(window, "B")
        assert window._activate_workspace_context(second)
        assert window._activate_workspace_context(first)
        first_cache = first.presentation_cache
        second_cache = second.presentation_cache
        assert first_cache.model_geometry is not None
        assert second_cache.model_geometry is not None
        task = first.session.prepare_import(Path("A2.inp"))
        delta = first.session.accept_imported_model(task.token, model)
        assert window._apply_session_delta(
            delta,
            context=first,
            model_geometry=build_model_geometry(model),
        )
        assert first_cache.model_geometry is not None
        assert first_cache.artifact_id == first.projection.artifact.artifact_id
        assert second_cache.model_geometry is not None
    finally:
        dispose_gui_widget(window)


def test_activation_restores_camera_step_display_and_batches_to_one_render(
    monkeypatch,
    dispose_gui_widget,
):
    """A-B-A restores the source presentation after one batched repaint."""

    _application()
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        model = make_two_step_static_pull_truss_model()
        task = first.session.prepare_import(Path("A.inp"))
        delta = first.session.accept_imported_model(task.token, model)
        assert window._apply_session_delta(
            delta,
            model_geometry=build_model_geometry(model),
        )
        second = _context_with_model(window, "B")

        window._set_current_step("pull2", refresh_viewport=False)
        window._display = DisplayState("deformed", True)

        # Keep camera capture/restore deterministic without creating a native
        # VTK backend in the offscreen test process.
        window.viewport._plotter = object()
        captured: list[str] = []
        restored: list[str] = []
        fit_calls: list[dict[str, object]] = []
        render_calls: list[object] = []
        symbol_refresh_calls: list[dict[str, object]] = []
        child_calls: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(
            main_window_module,
            "_capture_camera_state",
            lambda _plotter: captured.append(f"camera-{len(captured)}")
            or captured[-1],
        )
        monkeypatch.setattr(
            main_window_module,
            "_restore_camera_state",
            lambda _plotter, state: restored.append(state),
        )
        monkeypatch.setattr(
            window,
            "_project_viewport_for_module",
            lambda *_args, **_kwargs: None,
        )
        for name in (
            "clear_selection",
            "set_model",
            "set_edges_visible",
            "set_nodes_visible",
            "set_symbol_settings",
            "set_symbols_visible",
            "show_boundary_and_loads",
            "set_selection_mode",
        ):
            monkeypatch.setattr(
                window.viewport,
                name,
                lambda *args, _name=name, **kwargs: child_calls.append(
                    (_name, kwargs)
                ),
            )
        monkeypatch.setattr(
            window.viewport,
            "fit",
            lambda *args, **kwargs: fit_calls.append(kwargs),
        )
        monkeypatch.setattr(
            window.viewport,
            "render",
            lambda *args, **kwargs: render_calls.append(kwargs),
        )
        monkeypatch.setattr(
            window.viewport,
            "_refresh_symbols_for_camera",
            lambda **kwargs: symbol_refresh_calls.append(kwargs),
        )

        assert window._activate_workspace_context(second)
        assert window._activate_workspace_context(first)

        assert first.presentation_state.step_name == "pull2"
        assert first.presentation_state.display_state == DisplayState(
            "deformed",
            True,
        )
        assert restored == ["camera-0"]
        assert fit_calls == [{"render": False}]
        assert symbol_refresh_calls == [{"render": False}]
        assert not any(
            name == "show_boundary_and_loads"
            for name, _kwargs in child_calls
        )
        assert len(render_calls) == 2
        assert all(
            kwargs.get("render") is False
            for name, kwargs in child_calls
            if name in {
                "clear_selection",
                "set_model",
                "set_edges_visible",
                "set_nodes_visible",
                "set_symbol_settings",
                "set_symbols_visible",
                "show_boundary_and_loads",
                "set_selection_mode",
            }
        )
        assert all(
            kwargs.get("reset_camera") is False
            for name, kwargs in child_calls
            if name == "set_model"
        )
    finally:
        window.viewport._plotter = None
        dispose_gui_widget(window)


def test_activation_clears_selection_highlight_and_inspection_transients(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        second = _context_with_model(window, "B")
        window.selection.select_node(1)
        window._selected_geometry_refs.add(LogicalEntityRef("body:test"))
        window._selected_mesh_scope_refs.add(MeshEntityRef("node", node_id=1))
        window.viewport._selected_kind = "node"
        window.viewport._selected_id = 1
        popup = QWidget(window)
        window._inspection_windows.append(popup)
        window._mesh_browser = popup
        clear_calls: list[dict[str, object]] = []
        original_clear = window.viewport.clear_selection
        monkeypatch.setattr(
            window.viewport,
            "clear_selection",
            lambda *args, **kwargs: (
                clear_calls.append(kwargs),
                original_clear(*args, **kwargs),
            )[1],
        )

        assert window._activate_workspace_context(second)
        assert window.selection.node_id is None
        assert window.selection.element_id is None
        assert window._selected_geometry_refs == set()
        assert window._selected_mesh_scope_refs == set()
        assert window.viewport._selected_kind is None
        assert window.viewport._selected_id is None
        assert window._inspection_windows == []
        assert window._mesh_browser is None
        assert clear_calls
        assert all(call.get("render") is False for call in clear_calls)
    finally:
        dispose_gui_widget(window)


def test_warm_activation_uses_cached_adapters_without_fit_or_construction(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        model = make_static_pull_truss_model()
        task = first.session.prepare_import(Path("A.inp"))
        assert window._apply_session_delta(
            first.session.accept_imported_model(task.token, model),
            model_geometry=build_model_geometry(model),
        )
        second = _context_with_model(window, "B")
        assert window._activate_workspace_context(second)
        cached_geometry = first.presentation_cache.model_geometry
        cached_inspection = first.presentation_cache.inspection_service
        assert cached_geometry is not None
        assert cached_inspection is not None

        # The source camera was already established before leaving A.  The
        # following A activation must restore it rather than fit/reset it.
        first.presentation_state.camera_state = object()
        window.viewport._plotter = object()
        monkeypatch.setattr(main_window_module, "_capture_camera_state", lambda _: None)
        monkeypatch.setattr(main_window_module, "_restore_camera_state", lambda *_: None)
        monkeypatch.setattr(
            window,
            "_project_viewport_for_module",
            lambda *_args, **_kwargs: None,
        )
        calls: list[tuple[str, dict[str, object]]] = []
        for name in (
            "clear_selection",
            "set_model",
            "set_edges_visible",
            "set_nodes_visible",
            "set_symbol_settings",
            "set_symbols_visible",
            "show_boundary_and_loads",
            "set_selection_mode",
        ):
            monkeypatch.setattr(
                window.viewport,
                name,
                lambda *args, _name=name, **kwargs: calls.append(
                    (_name, kwargs)
                ),
            )
        fit_calls: list[dict[str, object]] = []
        render_calls: list[object] = []
        monkeypatch.setattr(
            window.viewport,
            "fit",
            lambda *args, **kwargs: fit_calls.append(kwargs),
        )
        monkeypatch.setattr(
            window.viewport,
            "render",
            lambda *args, **kwargs: render_calls.append(kwargs),
        )

        def fail_build(_model):
            raise AssertionError("warm activation rebuilt model geometry")

        class FailInspection:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("warm activation rebuilt inspection")

        monkeypatch.setattr(main_window_module, "build_model_geometry", fail_build)
        monkeypatch.setattr(main_window_module, "InspectionService", FailInspection)
        assert window._activate_workspace_context(first)

        assert window.geometry is cached_geometry
        assert window.inspection_service is cached_inspection
        assert fit_calls == []
        assert len(render_calls) == 1
        assert all(kwargs.get("render") is False for _, kwargs in calls)
        assert all(
            kwargs.get("reset_camera") is False
            for name, kwargs in calls
            if name == "set_model"
        )
    finally:
        window.viewport._plotter = None
        dispose_gui_widget(window)


def test_activation_is_blocked_while_editor_is_active_without_mutating_state(
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        before = window.workspace.active_document()
        assert before is not None
        target = _context_with_model(window, "B")
        before_projection = before.projection
        before_step = window._current_step_name
        before_display = window._display
        window._wire_editor_controller = object()

        assert not window._activate_workspace_context(target)
        assert window.workspace.active_document() is before
        assert window._active_context is before
        assert window.document is before_projection
        assert window._current_step_name == before_step
        assert window._display is before_display
        assert target.presentation_cache.model_geometry is None
    finally:
        window._wire_editor_controller = None
        dispose_gui_widget(window)


def test_activation_failure_restores_previous_aliases_scene_and_camera(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        model = make_static_pull_truss_model()
        task = first.session.prepare_import(Path("failure-A.inp"))
        assert window._apply_session_delta(
            first.session.accept_imported_model(task.token, model),
            model_geometry=build_model_geometry(model),
        )
        second = _context_with_model(window, "failure-B")
        previous_document = window.document
        previous_geometry = window.geometry
        previous_inspection = window.inspection_service
        previous_step = window._current_step_name
        previous_display = window._display
        window.viewport._plotter = object()
        restored_cameras: list[object] = []
        renders: list[object] = []
        symbol_refresh_calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            main_window_module,
            "_capture_camera_state",
            lambda _plotter: "previous-camera",
        )
        monkeypatch.setattr(
            main_window_module,
            "_restore_camera_state",
            lambda _plotter, state: restored_cameras.append(state),
        )
        monkeypatch.setattr(
            window,
            "_project_viewport_for_module",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected activation failure")
            ),
        )
        monkeypatch.setattr(
            window.viewport,
            "render",
            lambda *args, **kwargs: renders.append(kwargs),
        )
        monkeypatch.setattr(
            window.viewport,
            "_refresh_symbols_for_camera",
            lambda **kwargs: symbol_refresh_calls.append(kwargs),
        )

        assert not window._activate_workspace_context(second)
        assert window.workspace.active_document() is first
        assert window._active_context is first
        assert window.document is previous_document
        assert window.geometry is previous_geometry
        assert window.inspection_service is previous_inspection
        assert window._current_step_name == previous_step
        assert window._display is previous_display
        assert restored_cameras == ["previous-camera"]
        assert symbol_refresh_calls == [{"render": False}]
        assert len(renders) <= 1
    finally:
        window.viewport._plotter = None
        dispose_gui_widget(window)


def test_cold_activation_without_camera_fits_once_then_renders_once(
    monkeypatch,
    dispose_gui_widget,
):
    """A first display with no saved camera performs one fit and one repaint."""

    _application()
    window = FEMMainWindow()
    try:
        target = _context_with_model(window, "cold")
        fit_calls: list[dict[str, object]] = []
        render_calls: list[object] = []
        monkeypatch.setattr(
            window,
            "_project_viewport_for_module",
            lambda *_args, **_kwargs: None,
        )
        for name in (
            "clear_selection",
            "set_model",
            "set_edges_visible",
            "set_nodes_visible",
            "set_symbol_settings",
            "set_symbols_visible",
            "show_boundary_and_loads",
            "set_selection_mode",
        ):
            monkeypatch.setattr(
                window.viewport,
                name,
                lambda *args, _name=name, **kwargs: None,
            )
        monkeypatch.setattr(
            window.viewport,
            "fit",
            lambda *args, **kwargs: fit_calls.append(kwargs),
        )
        monkeypatch.setattr(
            window.viewport,
            "render",
            lambda *args, **kwargs: render_calls.append(kwargs),
        )

        assert target.presentation_state.camera_state is None
        assert window._activate_workspace_context(target)
        assert fit_calls == [{"render": False}]
        assert len(render_calls) <= 1
    finally:
        dispose_gui_widget(window)


def test_context_activation_keeps_one_viewport_backend_and_render_window(
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        viewport = window.viewport
        plotter = getattr(viewport, "_plotter", None)
        render_window = (
            getattr(plotter, "ren_win", None)
            if plotter is not None
            else None
        )
        contexts = [
            _context_with_model(window, f"M{index}")
            for index in range(3)
        ]
        for context in contexts:
            assert window._activate_workspace_context(context)
            assert window.viewport is viewport
            assert window.findChildren(FEMViewport) == [viewport]
            if plotter is not None:
                assert window.viewport._plotter is plotter
            if render_window is not None:
                assert getattr(window.viewport._plotter, "ren_win", None) is render_window
    finally:
        dispose_gui_widget(window)
