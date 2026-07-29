from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt
from PySide6.QtGui import QAction, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from fem_gui.widgets.agent_chat import (
    ModelViewportOverlayHost,
    ToolActivityPreview,
)
from fem_gui.widgets.viewport import FEMViewport
from fem_gui.widgets.viewport_toolbar import ViewportPanel


_VIEWPORT_ACTIONS = (
    "fit",
    "iso",
    "top",
    "front",
    "bottom",
    "back",
    "left",
    "right",
    "select_node",
    "select_element",
    "select_edge",
    "selected_info",
    "geometry_select_point",
    "geometry_select_edge",
    "geometry_select_face",
    "geometry_select_body",
    "nodes",
    "edges",
    "node_labels",
    "element_labels",
    "symbols",
    "undeformed",
    "deformed",
    "contour",
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _actions(parent: QWidget) -> dict[str, QAction]:
    return {
        name: QAction(name, parent)
        for name in _VIEWPORT_ACTIONS
    }


def _global_rect(widget: QWidget) -> QRect:
    return QRect(widget.mapToGlobal(QPoint(0, 0)), widget.size())


def _belongs_to(widget: QWidget, ancestor: QWidget) -> bool:
    current: QWidget | None = widget
    while current is not None:
        if current is ancestor:
            return True
        current = current.parentWidget()
    return False


class _ViewportProbe(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.mouse_presses = 0
        self.wheel_events = 0
        self._picker_event_targets = {self}

    def mousePressEvent(self, event) -> None:
        self.mouse_presses += 1
        event.accept()

    def wheelEvent(self, event) -> None:
        self.wheel_events += 1
        event.accept()


def test_drawer_open_close_and_width_only_change_overlay_geometry():
    application = _application()
    viewport = FEMViewport()
    host = ModelViewportOverlayHost(viewport)
    host.resize(720, 460)
    host.show()
    application.processEvents()

    baseline_geometry = QRect(viewport.geometry())
    baseline_size = QSize(viewport.size())
    assert host.layout().count() == 1
    assert host.layout().itemAt(0).widget() is viewport
    assert host.layout().indexOf(host.agent_chat_drawer) == -1
    assert host.agent_chat_drawer.parentWidget() is host
    assert viewport.parentWidget() is host

    host.agent_chat_drawer.close_button.click()
    QTest.qWait(host.ANIMATION_DURATION_MS + 30)
    assert host.agent_chat_drawer.isHidden()
    assert host.chat_launcher.isVisible()
    launcher_mask = host.chat_launcher.mask()
    assert launcher_mask.contains(host.chat_launcher.rect().center())
    assert not launcher_mask.contains(host.chat_launcher.rect().topLeft())
    assert not launcher_mask.contains(host.chat_launcher.rect().topRight())
    assert viewport.geometry() == baseline_geometry
    assert viewport.size() == baseline_size

    host.chat_launcher.click()
    QTest.qWait(host.ANIMATION_DURATION_MS + 30)
    assert host.agent_chat_drawer.isVisible()
    assert host.chat_launcher.isHidden()
    assert viewport.geometry() == baseline_geometry
    assert viewport.size() == baseline_size

    host.set_drawer_width(448)
    application.processEvents()
    assert host.agent_chat_drawer.width() == 448
    assert host.agent_chat_drawer.geometry().right() == host.rect().right()
    assert viewport.geometry() == baseline_geometry
    assert viewport.size() == baseline_size

    host.agent_chat_drawer.resize_handle.dragDelta.emit(24)
    application.processEvents()
    assert host.agent_chat_drawer.width() == 472
    assert viewport.geometry() == baseline_geometry
    assert viewport.size() == baseline_size

    host.set_drawer_open(False, animated=False)
    assert viewport.geometry() == baseline_geometry
    host.close()


def test_drawer_stays_between_toolbar_and_scope_tray_at_800_by_600():
    application = _application()
    viewport = _ViewportProbe()
    panel = ViewportPanel(viewport, _actions(viewport))
    panel.resize(800, 600)
    panel.show()
    panel.scope_creation_bar.begin("Set", "NodeSet-1")
    application.processEvents()

    assert panel.size() == QSize(800, 600)
    baseline_geometry = QRect(viewport.geometry())
    baseline_size = QSize(viewport.size())
    host_rect = _global_rect(panel.overlay_host)
    drawer_rect = _global_rect(panel.agent_chat_drawer)
    toolbar_rect = _global_rect(panel.toolbar)
    tray_rect = _global_rect(panel.scope_creation_tray)

    assert drawer_rect.top() == host_rect.top()
    assert drawer_rect.bottom() == host_rect.bottom()
    assert not drawer_rect.intersects(toolbar_rect)
    assert not drawer_rect.intersects(tray_rect)
    assert viewport.geometry() == panel.overlay_host.rect()

    panel.overlay_host.set_drawer_open(False, animated=False)
    panel.overlay_host.set_drawer_width(500)
    panel.overlay_host.set_drawer_open(True, animated=False)
    application.processEvents()

    assert panel.size() == QSize(800, 600)
    assert viewport.geometry() == baseline_geometry
    assert viewport.size() == baseline_size
    assert panel.agent_chat_drawer.width() == 500
    assert not _global_rect(panel.agent_chat_drawer).intersects(
        _global_rect(panel.toolbar)
    )
    assert not _global_rect(panel.agent_chat_drawer).intersects(
        _global_rect(panel.scope_creation_tray)
    )
    panel.close()


def test_drawer_hit_area_consumes_input_and_outside_remains_viewport():
    application = _application()
    viewport = _ViewportProbe()
    host = ModelViewportOverlayHost(viewport)
    host.resize(680, 420)
    host.show()
    application.processEvents()

    inside = QPoint(host.width() - 20, 90)
    inside_target = host.childAt(inside)
    assert inside_target is not None
    assert _belongs_to(inside_target, host.agent_chat_drawer)
    assert host.agent_chat_drawer not in viewport._picker_event_targets
    assert all(
        target not in viewport._picker_event_targets
        for target in host.agent_chat_drawer.findChildren(QWidget)
    )

    QTest.mouseClick(inside_target, Qt.MouseButton.LeftButton)
    wheel = QWheelEvent(
        QPointF(2.0, 2.0),
        QPointF(inside_target.mapToGlobal(QPoint(2, 2))),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(inside_target, wheel)
    application.processEvents()
    assert viewport.mouse_presses == 0
    assert viewport.wheel_events == 0

    outside = QPoint(30, 90)
    outside_target = host.childAt(outside)
    assert outside_target is viewport
    QTest.mouseClick(outside_target, Qt.MouseButton.LeftButton)
    QApplication.sendEvent(
        outside_target,
        QWheelEvent(
            QPointF(2.0, 2.0),
            QPointF(outside_target.mapToGlobal(QPoint(2, 2))),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        ),
    )
    assert viewport.mouse_presses == 1
    assert viewport.wheel_events == 1
    host.close()


def test_static_preview_controls_do_not_create_files_or_agent_dependencies(
    tmp_path,
    monkeypatch,
):
    application = _application()
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fem_gui"
        / "widgets"
        / "agent_chat.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "fem_agent" not in imported_roots
    assert imported_roots <= {
        "__future__",
        "PySide6",
        "agent_events",
        "agent_workspace",
        "collections",
        "html",
        "re",
    }

    monkeypatch.chdir(tmp_path)
    viewport = _ViewportProbe()
    host = ModelViewportOverlayHost(viewport)
    host.resize(600, 400)
    host.show()
    application.processEvents()

    drawer = host.agent_chat_drawer
    tools = drawer.findChild(ToolActivityPreview)
    assert tools is not None
    assert tools.details.isHidden()
    tools.summary_button.click()
    assert tools.details.isVisible()
    assert [action.text() for action in drawer.add_menu.actions()] == [
        "选择工作区…",
    ]
    assert drawer.add_button.menu() is None
    assert drawer.add_button.text() == "＋"
    assert drawer.send_button.text() == ""
    assert not drawer.send_button.icon().isNull()
    assert drawer.send_button.iconSize() == QSize(16, 16)

    drawer.input.setPlainText("@")
    application.processEvents()
    assert drawer.suggestion.isVisible()
    assert "各种类型" in drawer.suggestion_item.text()
    drawer.send_button.click()
    assert drawer.send_state.currentWidget() is drawer.stop_button
    assert not drawer.input.isEnabled()
    drawer.stop_button.click()
    assert drawer.send_state.currentWidget() is drawer.send_button
    assert drawer.input.isEnabled()
    assert list(tmp_path.iterdir()) == []
    host.close()
