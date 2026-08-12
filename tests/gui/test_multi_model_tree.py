from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.application import ModelSession, ProjectSnapshot
from fem.io.project import LoadedProject
import fem_gui.main_window as main_window_module
from fem_gui.commands import (
    GuiCommandDiagnostic,
    GuiCommandReceipt,
    GuiCommandStatus,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.task_controller import BackgroundTaskState
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.widgets.viewport import FEMViewport
from fem_gui.widgets.model_tree import (
    ModelTree,
    ROLE_DOCUMENT_ID,
    ROLE_KIND,
)
from tests.helpers.gui_command_receipts import await_succeeded
from tests.helpers.model_builders import make_static_pull_truss_model


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _model(name: str = "shared") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        mesh=SimpleNamespace(nodes=[None], elements=[None]),
        node_sets={},
        element_sets={},
        surfaces={},
        edges={},
        materials={"P1": object()},
        sections=[],
        steps=[],
    )


def _items(root):
    stack = [root]
    while stack:
        item = stack.pop()
        yield item
        stack.extend(item.child(index) for index in range(item.childCount()))


def test_model_tree_appends_roots_and_routes_same_named_children():
    _application()
    tree = ModelTree()
    first = tree.insert_document(11, _model(), source_path="A.fempy")
    second = tree.insert_document(22, _model(), source_path="B.fempy")

    assert tree.topLevelItemCount() == 2
    assert tree.roots[11] is first
    assert tree.roots[22] is second
    assert first.text(0) == second.text(0) == "shared"
    assert first.toolTip(0).endswith("A.fempy")
    assert all(
        item.data(0, ROLE_DOCUMENT_ID) == 11
        for item in _items(first)
    )
    assert all(
        item.data(0, ROLE_DOCUMENT_ID) == 22
        for item in _items(second)
    )

    routed = []
    tree.highlightRequested.connect(
        lambda document_id, kind, key: routed.append(
            (document_id, kind, key)
        )
    )
    material = next(
        item for item in _items(second)
        if item.data(0, ROLE_KIND) == "material"
    )
    tree._on_clicked(material)
    assert routed == [(22, "material", "P1")]


def test_model_tree_incremental_update_and_remove_never_clear_other_roots(
    monkeypatch,
):
    _application()
    tree = ModelTree()
    tree.insert_document(1, _model("A"))
    tree.insert_document(2, _model("B"))
    clear_calls = []
    original_clear = tree.clear
    monkeypatch.setattr(tree, "clear", lambda: clear_calls.append(True))

    tree.update_document(2, _model("B-updated"))

    assert clear_calls == []
    assert tree.roots[1].text(0) == "A"
    assert tree.roots[2].text(0) == "B-updated"
    assert tree.remove_document(2)
    assert 1 in tree.roots
    assert tree.topLevelItemCount() == 1


def test_model_tree_fifty_roots_use_one_indexed_tree():
    _application()
    tree = ModelTree()
    for document_id in range(1, 51):
        tree.insert_document(document_id, _model(f"Model-{document_id}"))

    assert len(tree.roots) == 50
    tree.set_active_document(37)
    assert tree.currentItem() is tree.roots[37]
    assert tree.topLevelItemCount() == 50


def test_fifty_workspace_roots_keep_one_shared_viewport(
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        viewport = window.viewport
        for document_id in range(1, 51):
            window.model_tree.insert_document(
                document_id,
                _model(f"Model-{document_id}"),
            )
        assert window.viewport is viewport
        assert len(window.findChildren(FEMViewport)) == 1
        assert len(window.model_tree.roots) == 50
    finally:
        dispose_gui_widget(window)


def _loaded_project(path: Path, name: str) -> LoadedProject:
    return LoadedProject(
        ProjectSnapshot(
            source_kind="native",
            source_path=path,
            model_name=name,
        ),
        path,
        13,
        (),
    )


def _await_terminal(receipt):
    assert receipt.status is GuiCommandStatus.PENDING
    completion = receipt.completion
    assert completion is not None
    application = QApplication.instance() or QApplication([])
    deadline = monotonic() + 5.0
    while not completion.done and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    return completion.result(0.0)


def test_main_window_open_appends_two_models_and_duplicate_path_activates(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    paths = (Path("C:/phase2-a.fempy"), Path("C:/phase2-b.fempy"))
    loaded = {
        paths[0]: _loaded_project(paths[0], "Model-A"),
        paths[1]: _loaded_project(paths[1], "Model-B"),
    }
    calls = []

    def decode(path):
        calls.append(path)
        return loaded[path]

    monkeypatch.setattr(main_window_module, "load_project", decode)
    window = FEMMainWindow()
    monkeypatch.setattr(window, "_show_error", lambda *_args, **_kwargs: None)
    try:
        first = window.open_project_path(paths[0])
        await_succeeded(first, timeout=5.0)
        first_id = window.workspace.active_document_id
        second = window.open_project_path(paths[1])
        await_succeeded(second, timeout=5.0)
        second_id = window.workspace.active_document_id

        assert first_id != second_id
        assert len(window.workspace.models) == 3
        assert set(calls) == set(paths)
        assert window.model_tree.topLevelItemCount() == 3
        assert window.model_tree.roots[first_id].text(0) == "Model-A"
        assert window.model_tree.roots[second_id].text(0) == "Model-B"

        duplicate = window.open_project_path(paths[0])
        assert duplicate.status is GuiCommandStatus.ACCEPTED
        assert window.workspace.active_document_id == first_id
        assert calls.count(paths[0]) == 1
    finally:
        dispose_gui_widget(window)


def test_main_window_failed_open_does_not_add_root_or_change_active(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    path = Path("C:/phase2-broken.fempy")

    def decode(_path):
        raise ValueError("broken project")

    monkeypatch.setattr(main_window_module, "load_project", decode)
    window = FEMMainWindow()
    monkeypatch.setattr(window, "_show_error", lambda *_args, **_kwargs: None)
    try:
        before = window.workspace.active_document_id
        receipt = window.open_project_path(path)
        terminal = _await_terminal(receipt)
        assert terminal.state is BackgroundTaskState.FAILED
        assert window.workspace.active_document_id == before
        assert len(window.workspace.models) == 1
        assert path not in tuple(
            context.source_path
            for context in window.workspace.models.values()
        )
        assert window.model_tree.topLevelItemCount() == 1
    finally:
        dispose_gui_widget(window)


def _native_context(window: FEMMainWindow, name: str, path: Path):
    session = ModelSession()
    session.new_native_project(name)
    context = window.workspace.add_model(
        session=session,
        projection=session.projection_snapshot(),
        display_name=name,
        source_path=path,
    )
    window._refresh_model_tree_for_context(context)
    return context


def test_save_routes_only_target_context(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    saved = []

    def fake_save(path, snapshot, **_kwargs):
        saved.append((Path(path), snapshot))
        return Path(path)

    monkeypatch.setattr(main_window_module, "save_project", fake_save)
    try:
        first = _native_context(window, "A", Path("C:/A.fempy"))
        second = _native_context(window, "B", Path("C:/B.fempy"))
        window.workspace.activate(second)
        window._active_context = second
        receipt = window.save_project_path(
            Path("C:/saved-A.fempy"),
            document_id=first.document_id,
        )
        await_succeeded(receipt, timeout=5.0)

        assert len(saved) == 1
        assert saved[0][0].name == "saved-A.fempy"
        assert window.workspace.active_document_id == second.document_id
        assert first.source_path == Path("C:/saved-A.fempy")
        assert second.source_path == Path("C:/B.fempy")
    finally:
        dispose_gui_widget(window)


def test_nonactive_model_edit_does_not_change_other_session_projection(
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        first = _native_context(window, "A", Path("C:/A.fempy"))
        second = _native_context(window, "B", Path("C:/B.fempy"))
        assert window._activate_workspace_context(second)
        before = second.projection

        delta = first.session.new_native_project("A-edited")
        assert window._apply_session_delta(delta, context=first)

        assert second.projection is before
        assert second.session.session_revision == before.session_revision
        assert second.projection.parts == before.parts
        assert second.projection.sections == before.sections
        assert second.projection.runs == before.runs
        assert window.workspace.active_document() is second
    finally:
        dispose_gui_widget(window)


def test_close_routes_only_clean_target_context(
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        first = _native_context(window, "A", Path("C:/A.fempy"))
        second = _native_context(window, "B", Path("C:/B.fempy"))
        window.workspace.activate(second)
        window._active_context = second
        assert window.close_model(
            confirm=False,
            document_id=first.document_id,
        )
        assert first.document_id not in window.workspace.models
        assert second.document_id in window.workspace.models
        assert window.workspace.active_document_id == second.document_id
        assert second.document_id in window.model_tree.roots
    finally:
        dispose_gui_widget(window)


def test_close_active_context_projects_replacement_before_returning(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        initial = window.workspace.active_document()
        assert initial is not None
        window.workspace.remove(initial)
        window._active_context = None
        window.model_tree.remove_document(initial.document_id)
        first = _native_context(window, "A", Path("C:/A.fempy"))
        second = _native_context(window, "B", Path("C:/B.fempy"))
        window.workspace.activate(first)
        window._active_context = first
        window._refresh_model_tree_for_context(first)
        projected = []
        monkeypatch.setattr(
            window,
            "_refresh_model_tree_for_context",
            lambda context: projected.append(context),
        )

        assert window.close_model(
            confirm=False,
            document_id=first.document_id,
        )

        assert window._active_context is second
        assert window.workspace.active_document_id == second.document_id
        assert window.document.model_name == "B"
        assert projected and projected[-1] is second
        assert second.document_id in window.model_tree.roots
    finally:
        dispose_gui_widget(window)


def test_close_inactive_context_preserves_active_display_state(
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    try:
        first = _native_context(window, "A", Path("C:/A.fempy"))
        second = _native_context(window, "B", Path("C:/B.fempy"))
        window.workspace.activate(second)
        window._active_context = second
        window._refresh_model_tree_for_context(second)
        display = window._display
        document = window.document
        selection_mode = window.selection.mode

        assert window.close_model(
            confirm=False,
            document_id=first.document_id,
        )

        assert window._active_context is second
        assert window.document is document
        assert window._display is display
        assert window.selection.mode == selection_mode
    finally:
        dispose_gui_widget(window)


def test_set_active_document_only_touches_old_and_new_roots(monkeypatch):
    _application()
    tree = ModelTree()

    class SpyRoot:
        def __init__(self):
            self.selected = []

        def setSelected(self, value):
            self.selected.append(bool(value))

    roots = {index: SpyRoot() for index in range(1, 51)}
    tree._roots = roots
    tree._active_document_id = 1
    current = []
    monkeypatch.setattr(tree, "setCurrentItem", current.append)

    tree.set_active_document(50)

    assert roots[1].selected == [False]
    assert roots[50].selected == [True]
    assert sum(len(root.selected) for root in roots.values()) == 2
    assert current == [roots[50]]


def test_set_active_document_preserves_root_order():
    _application()
    tree = ModelTree()
    for document_id in range(1, 4):
        tree.insert_document(document_id, _model(f"Model-{document_id}"))
    original = [tree.topLevelItem(index) for index in range(3)]

    tree.set_active_document(3)
    tree.set_active_document(1)

    assert [tree.topLevelItem(index) for index in range(3)] == original


def test_remove_last_root_adds_placeholder_without_clear(monkeypatch):
    _application()
    tree = ModelTree()
    tree.insert_document(1, _model("A"))
    clear_calls = []
    monkeypatch.setattr(tree, "clear", lambda: clear_calls.append(True))

    assert tree.remove_document(1)
    assert clear_calls == []
    assert tree.roots == {}
    assert tree.topLevelItemCount() == 1
    assert tree.topLevelItem(0).data(0, ROLE_KIND) == "empty"


def test_empty_projection_update_keeps_document_root():
    _application()
    tree = ModelTree()
    session = ModelSession()
    projection = session.projection_snapshot()
    root = tree.insert_document(7, projection)

    assert tree.roots[7] is root
    tree.update_document(7, projection)
    assert tree.roots[7] is not None
    assert tree.topLevelItemCount() == 1


def test_active_artifact_delta_installs_tree_once_without_extra_refresh(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    geometry = build_model_geometry(model)
    try:
        task = window.session.prepare_import(Path("phase2-first.inp"))
        first_delta = window.session.accept_imported_model(
            task.token,
            model,
        )
        assert window._apply_session_delta(
            first_delta,
            model_geometry=geometry,
        )

        replacement_task = window.session.prepare_import(
            Path("phase2-second.inp")
        )
        replacement_delta = window.session.accept_imported_model(
            replacement_task.token,
            model,
        )
        installs = []
        refreshes = []
        monkeypatch.setattr(
            window,
            "_install_model",
            lambda *args, **kwargs: installs.append((args, kwargs)),
        )
        monkeypatch.setattr(
            window,
            "_refresh_model_tree_for_context",
            lambda context: refreshes.append(context),
        )

        assert window._apply_session_delta(
            replacement_delta,
            model_geometry=geometry,
        )
        assert len(installs) == 1
        assert refreshes == []
    finally:
        dispose_gui_widget(window)


def test_create_native_models_append_roots_without_global_tree_clear(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    clear_calls = []
    monkeypatch.setattr(window.model_tree, "clear", lambda: clear_calls.append(True))
    try:
        window._create_native_model("Model-2")
        window._create_native_model("Model-3")

        assert len(window.workspace.models) == 3
        assert len(window.model_tree.roots) == 3
        assert clear_calls == []
        assert {
            context.display_name
            for context in window.workspace.models.values()
        } == {"Model-1", "Model-2", "Model-3"}
    finally:
        dispose_gui_widget(window)


def test_new_native_model_does_not_confirm_dirty_active_document(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    responses = iter((("Model-2", True),))
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: next(responses),
    )
    monkeypatch.setattr(
        window,
        "_confirm_discard_changes",
        lambda: (_ for _ in ()).throw(AssertionError("must not confirm")),
    )
    try:
        window.new_native_model()
        assert len(window.workspace.models) == 2
        assert window.workspace.active_document().display_name == "Model-2"
    finally:
        dispose_gui_widget(window)


def test_rejected_native_model_creation_removes_context_and_restores_active(
    monkeypatch,
    dispose_gui_widget,
):
    _application()
    window = FEMMainWindow()
    before = window._active_context
    assert before is not None
    monkeypatch.setattr(
        window,
        "new_native_project",
        lambda _command: GuiCommandReceipt.rejected(
            9001,
            GuiCommandDiagnostic("test.rejected", "rejected"),
        ),
    )
    monkeypatch.setattr(window, "_show_command_rejection", lambda *_args: None)
    try:
        window._create_native_model("Rejected")
        assert window._active_context is before
        assert window.workspace.active_document_id == before.document_id
        assert len(window.workspace.models) == 1
        assert set(window.model_tree.roots) == {before.document_id}
    finally:
        dispose_gui_widget(window)


def test_no_artifact_delta_refreshes_target_tree_once(monkeypatch, dispose_gui_widget):
    _application()
    window = FEMMainWindow()
    calls = []
    monkeypatch.setattr(
        window,
        "_refresh_model_tree_for_context",
        lambda context: calls.append(context),
    )
    try:
        assert window._apply_session_delta(window.session.new_native_project())
        assert calls == [window._active_context]
    finally:
        dispose_gui_widget(window)
