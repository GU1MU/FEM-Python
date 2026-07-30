from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt, Signal
from PySide6.QtGui import QAction, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from fem_gui.agent_events import FakeAgentEventStream
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


def _widget_at_host_point(
    host: QWidget,
    point: QPoint,
) -> QWidget | None:
    return QApplication.widgetAt(host.mapToGlobal(point))


def _belongs_to(widget: QWidget, ancestor: QWidget) -> bool:
    current: QWidget | None = widget
    while current is not None:
        if current is ancestor:
            return True
        current = current.parentWidget()
    return False


class _ViewportProbe(QWidget):
    nativeSurfaceUpdated = Signal()

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


def test_drawer_removes_phase_copy_and_uses_compact_composer_controls():
    application = _application()
    viewport = _ViewportProbe()
    host = ModelViewportOverlayHost(viewport)
    host.resize(720, 520)
    host.show()
    application.processEvents()

    drawer = host.agent_chat_drawer
    labels = drawer.findChildren(QLabel)
    title = drawer.findChild(QLabel, "agentChatTitle")
    header = drawer.findChild(QWidget, "agentChatHeader")

    assert title is not None
    assert header is not None
    assert title.text() == "FEM Agent"
    assert title.font().pointSizeF() >= 12
    assert header.height() <= 40
    assert (
        drawer.palette().color(drawer.backgroundRole()).name()
        == "#ffffff"
    )
    assert (
        drawer.conversation_widget.palette()
        .color(drawer.conversation_widget.backgroundRole())
        .name()
        == "#ffffff"
    )
    assert all("Phase 5" not in label.text() for label in labels)
    assert drawer.findChild(QWidget, "agentChatPreviewBadge") is None
    assert drawer.findChild(QWidget, "agentChatSubtitle") is None
    assert drawer.findChild(QWidget, "agentChatWelcome") is None
    assert drawer.findChild(QWidget, "agentChatAuthoringBinding") is None
    assert not hasattr(drawer, "new_session_button")
    assert drawer.input.font().pointSizeF() >= 10
    assert drawer.input.height() == 44
    assert drawer.close_button.width() <= 30
    assert drawer.close_button.height() <= 30
    assert drawer.send_state.size() == QSize(30, 30)
    host.close()


def test_composer_input_expands_for_multiple_lines_and_collapses_when_cleared():
    application = _application()
    viewport = _ViewportProbe()
    host = ModelViewportOverlayHost(viewport)
    host.resize(720, 520)
    host.show()
    application.processEvents()

    editor = host.agent_chat_drawer.input
    collapsed_height = editor.height()
    line_height = editor.fontMetrics().lineSpacing()
    for line_count in range(1, 6):
        editor.setPlainText(
            "\n".join(f"第 {index} 行" for index in range(line_count))
        )
        application.processEvents()
        assert editor.height() == (
            collapsed_height + (line_count - 1) * line_height
        )
        assert editor.verticalScrollBarPolicy() == (
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    maximum_height = collapsed_height + 4 * line_height
    editor.setPlainText("\n".join(f"第 {index} 行" for index in range(6)))
    application.processEvents()
    assert editor.height() == maximum_height
    assert editor.verticalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )

    editor.setPlainText("自动折行内容" * 100)
    application.processEvents()
    assert editor.height() == maximum_height
    assert editor.verticalScrollBar().maximum() > 0

    editor.clear()
    application.processEvents()
    assert editor.height() == collapsed_height
    host.close()


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
    assert (
        _global_rect(host.agent_chat_drawer).right()
        == _global_rect(host).right()
    )
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


def test_native_viewport_cannot_cover_independent_tool_overlays():
    application = _application()
    viewport = _ViewportProbe()
    viewport.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
    host = ModelViewportOverlayHost(viewport)
    host.resize(720, 460)
    host.show()
    application.processEvents()

    baseline_geometry = QRect(viewport.geometry())
    baseline_size = QSize(viewport.size())
    assert host.agent_chat_drawer.isWindow()
    assert host.chat_launcher.isWindow()
    assert (
        host.agent_chat_drawer.windowType()
        == Qt.WindowType.Tool
    )
    assert host.chat_launcher.windowType() == Qt.WindowType.Tool
    assert host.agent_chat_drawer.testAttribute(
        Qt.WidgetAttribute.WA_ShowWithoutActivating
    )
    assert host.chat_launcher.testAttribute(
        Qt.WidgetAttribute.WA_ShowWithoutActivating
    )
    assert (
        host.chat_launcher.windowFlags()
        & Qt.WindowType.WindowDoesNotAcceptFocus
    )

    host.set_drawer_open(False, animated=False)
    viewport.raise_()
    viewport.nativeSurfaceUpdated.emit()
    application.processEvents()
    assert host.chat_launcher.isVisible()
    QTest.mouseClick(
        host.chat_launcher,
        Qt.MouseButton.LeftButton,
    )
    application.processEvents()
    assert host.drawer_is_open
    assert viewport.geometry() == baseline_geometry
    assert viewport.size() == baseline_size

    host.set_drawer_open(True, animated=False)
    viewport.raise_()
    viewport.nativeSurfaceUpdated.emit()
    application.processEvents()
    assert host.agent_chat_drawer.isVisible()
    QTest.mouseClick(
        host.agent_chat_drawer.close_button,
        Qt.MouseButton.LeftButton,
    )
    application.processEvents()
    assert not host.drawer_is_open
    assert viewport.geometry() == baseline_geometry
    assert viewport.size() == baseline_size
    host.close()


def test_tool_overlays_follow_host_window_lifecycle():
    application = _application()
    viewport = _ViewportProbe()
    host = ModelViewportOverlayHost(viewport)
    host.resize(720, 460)
    host.move(60, 70)
    host.show()
    host.activateWindow()
    application.processEvents()
    host.set_drawer_open(False, animated=False)
    application.processEvents()

    baseline_viewport_geometry = QRect(viewport.geometry())
    baseline_viewport_size = QSize(viewport.size())
    first_host_rect = _global_rect(host)
    first_launcher_rect = _global_rect(host.chat_launcher)
    assert first_host_rect.contains(first_launcher_rect)

    host.move(100, 120)
    application.processEvents()
    moved_launcher_rect = _global_rect(host.chat_launcher)
    assert moved_launcher_rect.topLeft() - first_launcher_rect.topLeft() == (
        _global_rect(host).topLeft() - first_host_rect.topLeft()
    )
    assert viewport.geometry() == baseline_viewport_geometry
    assert viewport.size() == baseline_viewport_size

    host.resize(640, 400)
    application.processEvents()
    resized_host_rect = _global_rect(host)
    resized_launcher_rect = _global_rect(host.chat_launcher)
    assert resized_host_rect.contains(resized_launcher_rect)
    assert (
        resized_host_rect.right() - resized_launcher_rect.right()
        == host.LAUNCHER_MARGIN
    )
    assert viewport.geometry() == host.rect()

    host.hide()
    application.processEvents()
    assert host.agent_chat_drawer.isHidden()
    assert host.chat_launcher.isHidden()

    host.show()
    host.activateWindow()
    application.processEvents()
    assert host.chat_launcher.isVisible()

    host.showMinimized()
    application.processEvents()
    assert host.agent_chat_drawer.isHidden()
    assert host.chat_launcher.isHidden()

    host.showNormal()
    host.activateWindow()
    application.processEvents()
    assert host.chat_launcher.isVisible()
    host.close()


def test_drawer_and_scope_bar_overlay_without_resizing_viewport():
    application = _application()
    viewport = _ViewportProbe()
    panel = ViewportPanel(viewport, _actions(viewport))
    panel.resize(800, 600)
    panel.show()
    application.processEvents()
    assert panel.layout().count() == 2
    assert panel.overlay_host.geometry().bottom() == panel.rect().bottom()
    assert panel.scope_creation_bar.isHidden()
    baseline_geometry = QRect(viewport.geometry())
    baseline_size = QSize(viewport.size())
    panel.scope_creation_bar.begin("Set", "NodeSet-1")
    application.processEvents()

    assert panel.size() == QSize(800, 600)
    assert viewport.geometry() == baseline_geometry
    assert viewport.size() == baseline_size
    host_rect = _global_rect(panel.overlay_host)
    drawer_rect = _global_rect(panel.agent_chat_drawer)
    toolbar_rect = _global_rect(panel.toolbar)
    scope_bar_rect = _global_rect(panel.scope_creation_bar)

    assert drawer_rect.top() == host_rect.top()
    assert drawer_rect.bottom() == host_rect.bottom()
    assert not drawer_rect.intersects(toolbar_rect)
    assert scope_bar_rect.bottom() == host_rect.bottom()
    assert not drawer_rect.intersects(scope_bar_rect)
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
        _global_rect(panel.scope_creation_bar)
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
    inside_target = _widget_at_host_point(host, inside)
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
    outside_target = _widget_at_host_point(host, outside)
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


def test_static_preview_controls_do_not_create_files_or_unbounded_dependencies(
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
    assert imported_roots <= {
        "__future__",
        "PySide6",
        "agent_events",
        "agent_runtime",
        "agent_workspace",
        "collections",
        "fem_agent",
        "html",
        "pathlib",
        "re",
    }

    monkeypatch.chdir(tmp_path)
    viewport = _ViewportProbe()
    host = ModelViewportOverlayHost(viewport)
    host.resize(600, 400)
    host.show()
    application.processEvents()

    drawer = host.agent_chat_drawer
    drawer.replay_agent_events(
        FakeAgentEventStream().review_preview()
    )
    application.processEvents()
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
    assert not drawer.agent_runtime.busy
    assert list(tmp_path.iterdir()) == []
    host.close()
