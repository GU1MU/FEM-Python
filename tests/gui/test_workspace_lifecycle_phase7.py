"""Focused Phase 7 workspace lifecycle and bounded stability checks."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from fem.application import ModelSession
from fem_gui.main_window import FEMMainWindow


class _DeferredController(QObject):
    """Small cooperative controller double for document-close tests."""

    busy_changed = Signal(bool)
    cancelling_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._busy = True
        self._cancel_requested = False
        self._after_cleanup = None

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def current_task_name(self) -> str:
        return "phase7-test" if self._busy else ""

    def request_cancel(self, *, after_cleanup=None) -> bool:
        if after_cleanup is not None:
            self._after_cleanup = after_cleanup
        if not self._busy:
            return False
        if self._cancel_requested:
            return False
        self._cancel_requested = True
        self.cancelling_changed.emit(True)
        return True

    def finish(self) -> None:
        self._busy = False
        self.cancelling_changed.emit(False)
        self.busy_changed.emit(False)
        callback = self._after_cleanup
        self._after_cleanup = None
        if callback is not None:
            callback()


def _app() -> QApplication:
    application = QApplication.instance()
    assert application is not None
    return application


def _add_model(window: FEMMainWindow, name: str):
    context = window.workspace.add_model(
        session=ModelSession(),
        display_name=name,
    )
    window.model_tree.insert_document(
        context.document_id,
        context.projection,
        model_name=name,
    )
    return context


def _add_result(window: FEMMainWindow, name: str):
    context = window.workspace.add_result(
        session=ModelSession(),
        display_name=name,
    )
    window.result_tree.upsert_archive(
        context.document_id,
        context.projection,
        display_name=name,
        source_path=Path(f"{name}.femres"),
    )
    return context


def test_busy_document_close_is_deferred_until_cooperative_cleanup(
    dispose_gui_widget,
) -> None:
    window = FEMMainWindow()
    try:
        controller = _DeferredController()
        context = window.workspace.add_model(
            session=ModelSession(),
            display_name="busy-model",
            task_controller=controller,
        )
        window.model_tree.insert_document(context.document_id, context.projection)

        assert window.close_model(confirm=False, document_id=context.document_id)
        assert context.document_id in window.workspace.models
        assert controller.cancel_requested

        controller.finish()
        _app().processEvents()
        assert context.document_id not in window.workspace.models
        assert context.document_id not in window.model_tree.roots
    finally:
        dispose_gui_widget(window)


def test_exit_confirmation_is_read_only_and_cancel_preserves_all_contexts(
    monkeypatch,
    dispose_gui_widget,
) -> None:
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        second = _add_model(window, "second")
        external = _add_result(window, "external")
        before = tuple(
            (
                context.document_id,
                id(context.session),
                context.revision,
                context.source_path,
            )
            for context in window.workspace.documents()
        )
        calls: list[int] = []
        monkeypatch.setattr(
            window,
            "_confirm_workspace_context_close",
            lambda context, _confirm: calls.append(context.document_id) or False,
        )
        window.show()
        event = QCloseEvent()
        window.closeEvent(event)

        assert not event.isAccepted()
        assert tuple(
            (
                context.document_id,
                id(context.session),
                context.revision,
                context.source_path,
            )
            for context in window.workspace.documents()
        ) == before
        assert set(calls) == {first.document_id}
        assert second.document_id in window.workspace.models
        assert external.document_id in window.workspace.results
    finally:
        dispose_gui_widget(window)


def test_workspace_switch_loops_keep_single_viewport_and_stable_roots(
    dispose_gui_widget,
) -> None:
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        models = [first, _add_model(window, "A"), _add_model(window, "B")]
        results = [_add_result(window, "R1"), _add_result(window, "R2"), _add_result(window, "R3")]
        for context in models:
            assert window._activate_workspace_context(context)
        for context in results:
            assert window._activate_workspace_context(context)
        initial_model_roots = set(window.model_tree.roots)
        initial_result_roots = set(window.result_tree.roots)
        for _ in range(99):
            for context in models:
                assert window._activate_workspace_context(context)
            for context in results:
                assert window._activate_workspace_context(context)

        assert set(window.model_tree.roots) == initial_model_roots
        assert set(window.result_tree.roots) == initial_result_roots
        assert len(window.findChildren(type(window.viewport))) == 1
        assert window.workspace.document_count == 6
    finally:
        dispose_gui_widget(window)


def test_repeated_small_document_open_close_leaves_no_workspace_references(
    dispose_gui_widget,
) -> None:
    window = FEMMainWindow()
    try:
        initial = window.workspace.active_document()
        assert initial is not None
        opened = []
        for index in range(20):
            opened.append(_add_model(window, f"model-{index}"))
            opened.append(_add_result(window, f"result-{index}"))
        for context in tuple(opened):
            assert window.close_model(confirm=False, document_id=context.document_id)
        assert all(
            context.document_id not in window.workspace.models
            and context.document_id not in window.workspace.results
            for context in opened
        )
        assert window.workspace.document_count == 1
        assert window.workspace.active_document() is initial
    finally:
        dispose_gui_widget(window)


def test_root_lifecycle_routes_and_shared_shutdown_are_singleton(
    monkeypatch,
    dispose_gui_widget,
) -> None:
    window = FEMMainWindow()
    try:
        model = window.workspace.active_document()
        assert model is not None
        result = _add_result(window, "external")
        runtime = window.viewport_panel.agent_chat_drawer.agent_runtime
        assert runtime is window.viewport_panel.agent_chat_drawer.agent_runtime
        assert window.model_tree.rootActionRequested is not None
        assert window.result_tree.rootActionRequested is not None

        activated: list[int] = []
        monkeypatch.setattr(
            window,
            "_activate_workspace_context",
            lambda context: activated.append(context.document_id) or True,
        )
        window._model_root_action_requested(model.document_id, "activate")
        window._result_root_action_requested(result.document_id, "activate")
        assert activated == [model.document_id, result.document_id]

        overlay = window.viewport_panel.overlay_host
        backend = window.viewport
        overlay_calls: list[bool] = []
        backend_calls: list[bool] = []
        monkeypatch.setattr(overlay, "shutdown", lambda **_kwargs: overlay_calls.append(True))
        monkeypatch.setattr(backend, "shutdown_backend", lambda: backend_calls.append(True))
        window._shutdown_shared_resources()
        window._shutdown_shared_resources()
        assert overlay_calls == [True]
        assert backend_calls == [True]
    finally:
        dispose_gui_widget(window)
