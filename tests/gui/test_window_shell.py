from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem_gui.main_window import FEMMainWindow


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _visible_text(window: FEMMainWindow) -> str:
    menu_text = [action.text() for action in window.menuBar().actions()]
    action_text = [action.text() for action in window.actions.values()]
    tree_text: list[str] = []
    root = window.model_tree.invisibleRootItem()
    stack = [root.child(index) for index in range(root.childCount())]
    while stack:
        item = stack.pop()
        tree_text.append(item.text(0))
        stack.extend(item.child(index) for index in range(item.childCount()))
    return "\n".join((*menu_text, *action_text, *tree_text))


def test_chinese_shell_has_no_unimplemented_modules_or_placeholder_terms():
    _application()
    window = FEMMainWindow()
    text = _visible_text(window)

    assert [action.text() for action in window.menuBar().actions()] == ["文件", "编辑", "视图", "分析", "结果", "帮助"]
    assert all(action.text() for action in window.actions.values())
    for forbidden in ("Planned", "Coming Soon", "Placeholder", "暂未实现", "后续支持", "Part", "Sketch", "Interaction", "Optimization", "AI"):
        assert forbidden not in text
    assert window.statusBar().height() == 22
    window.close()


def test_action_states_without_model_or_result():
    _application()
    window = FEMMainWindow()

    assert not window.actions["submit_job"].isEnabled()
    assert not window.actions["select_point"].isEnabled()
    assert not window.actions["deformed"].isEnabled()
    assert not window.actions["query"].isEnabled()

    window.close()


def test_startup_model_tree_has_no_automatic_part_placeholder():
    _application()
    window = FEMMainWindow()
    document_id = window.workspace.active_document_id

    assert document_id is not None
    root = window.model_tree.roots[document_id]
    assert root.text(0) == "模型-1"
    assert root.childCount() == 0
    assert window.document.parts == ()

    window.close()


def test_main_window_close_explicitly_releases_viewport_backend(monkeypatch):
    _application()
    window = FEMMainWindow()
    runtime = window.viewport_panel.agent_chat_drawer.agent_runtime
    calls: list[bool] = []
    monkeypatch.setattr(
        window.viewport,
        "shutdown_backend",
        lambda: calls.append(True),
    )

    window.close()

    assert calls == [True]
    assert runtime.is_shutdown
