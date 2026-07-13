from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread
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


def test_action_states_without_model_result_and_while_busy():
    _application()
    window = FEMMainWindow()

    assert not window.actions["submit_job"].isEnabled()
    assert not window.actions["select_node"].isEnabled()
    assert not window.actions["deformed"].isEnabled()
    assert not window.actions["query"].isEnabled()

    window._thread = QThread(window)
    window._update_action_states()
    assert not window.actions["open"].isEnabled()
    assert not window.actions["submit_job"].isEnabled()
    window._thread = None
    window.close()
