"""三维视口顶部常驻的高频操作工具栏。"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..agent_authoring import (
    AgentAuthoringBridge,
    AuthoringWorkflowController,
)
from ..icons import icon
from .agent_chat import ModelViewportOverlayHost


class ViewportToolBar(QToolBar):
    """仅复用主窗口 QAction 的紧凑视口工具栏。"""

    def __init__(self, actions: Mapping[str, QAction], parent=None) -> None:
        super().__init__("视口工具", parent)
        self._action_widgets: dict[str, QToolButton] = {}
        self.setObjectName("viewportToolbar")
        self.setMovable(False)
        self.setFloatable(False)
        # Coordinate-view pictograms already contain the complete axis legend. Keep
        # this toolbar icon-only so the QAction text cannot overlap the X/Y/Z
        # labels drawn inside the image.
        # Keep the viewport toolbar compact, but let the artwork use most of
        # each button.  The PNG assets are trimmed to their alpha bounds, so
        # a 32 px base size is now visually balanced instead of appearing
        # undersized inside the 44 px toolbar.
        self.setIconSize(QSize(32, 32))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setFixedHeight(44)

        self._add_group(actions, ("fit", "iso", "top", "front"))
        views = QMenu("其他标准视角", self)
        for key in ("bottom", "back", "left", "right"):
            views.addAction(actions[key])
        view_button = QToolButton(self)
        view_button.setObjectName("viewportMoreViews")
        view_button.setIcon(icon("view_more"))
        view_button.setIconSize(QSize(34, 34))
        view_button.setToolTip("其他标准视角")
        view_button.setMenu(views)
        view_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.addWidget(view_button)
        self.addSeparator()

        self._add_group(
            actions,
            ("select_point", "select_element", "select_edge", "select_face", "select_body"),
        )
        self.addSeparator()
        self._add_group(actions, ("nodes", "edges", "node_labels", "element_labels", "symbols"))

    def _add_group(self, actions: Mapping[str, QAction], names: tuple[str, ...]) -> None:
        for name in names:
            action = actions[name]
            self.addAction(action)
            button = self.widgetForAction(action)
            if isinstance(button, QToolButton):
                button.setObjectName(f"viewportAction_{name}")
                self._action_widgets[name] = button
            if isinstance(button, QToolButton) and name in {
                "symbols", "select_element",
            }:
                # These pictograms contain thin structural outlines.  Give
                # them the full button height so the beam/support details do
                # not disappear at the default toolbar size.
                button.setIconSize(QSize(40, 40))
            if name in {"top", "bottom", "front", "back", "left", "right", "iso"}:
                if isinstance(button, QToolButton):
                    button.setIconSize(QSize(36, 36))
                    button.setMinimumWidth(40)

    def set_geometry_context(self, enabled: bool) -> None:
        """Compatibility hook; semantic selection buttons are always shared."""


class ScopeCreationBar(QWidget):
    """Persistent completion controls for one guided scope selection."""

    createRequested = Signal()
    cancelRequested = Signal()
    activeChanged = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("scopeCreationBar")
        self.setAutoFillBackground(True)
        self.type_value = QLabel("—", self)
        self.type_value.setObjectName("scopeCreationType")
        self.name_edit = QLineEdit(self)
        self.name_edit.setObjectName("scopeCreationName")
        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.setObjectName("scopeCreationCancel")
        self.cancel_button.clicked.connect(self.cancelRequested)
        self.create_button = QPushButton("创建", self)
        self.create_button.setObjectName("scopeCreationSubmit")
        self.create_button.setEnabled(False)
        self.create_button.clicked.connect(self.createRequested)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        layout.addWidget(QLabel("类型", self))
        layout.addWidget(self.type_value)
        layout.addSpacing(12)
        layout.addWidget(QLabel("作用域名称", self))
        layout.addWidget(self.name_edit, 1)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.create_button)
        self.hide()

    def begin(self, scope_type: str, suggested_name: str) -> None:
        self.type_value.setText(str(scope_type))
        self.name_edit.setText(str(suggested_name))
        self.name_edit.selectAll()
        self.create_button.setEnabled(False)
        self.activeChanged.emit(True)

    def set_selection_ready(self, ready: bool) -> None:
        self.create_button.setEnabled(bool(ready))

    def scope_name(self) -> str:
        return self.name_edit.text().strip()

    def finish(self) -> None:
        self.create_button.setEnabled(False)
        self.hide()
        self.activeChanged.emit(False)


class PlanarBooleanFaceBar(QWidget):
    """One-step target-Face prompt shared by 2D and 3D Boolean workflows."""

    confirmRequested = Signal()
    cancelRequested = Signal()
    activeChanged = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("planarBooleanFaceBar")
        self.setAutoFillBackground(True)
        self.prompt_label = QLabel(self)
        self.prompt_label.setObjectName("planarBooleanFacePrompt")
        self.confirm_button = QPushButton("确定", self)
        self.confirm_button.setObjectName("planarBooleanFaceConfirm")
        self.confirm_button.setEnabled(False)
        self.confirm_button.clicked.connect(self.confirmRequested)
        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.setObjectName("planarBooleanFaceCancel")
        self.cancel_button.clicked.connect(self.cancelRequested)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        layout.addWidget(self.prompt_label, 1)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.confirm_button)
        self.hide()

    def begin(self, operation: str) -> None:
        if str(operation) not in {"fuse", "cut"}:
            raise ValueError("布尔操作必须是合并或切除")
        self.prompt_label.setText("请选择目标面")
        self.confirm_button.setEnabled(False)
        self.activeChanged.emit(True)

    def set_selection_ready(self, ready: bool) -> None:
        self.confirm_button.setEnabled(bool(ready))

    def finish(self) -> None:
        self.confirm_button.setEnabled(False)
        self.hide()
        self.activeChanged.emit(False)


class ViewportPanel(QWidget):
    """组合常驻工具栏与有限元三维视口。"""

    def __init__(
        self,
        viewport: QWidget,
        actions: Mapping[str, QAction],
        parent=None,
        *,
        authoring_bridge: AgentAuthoringBridge | None = None,
        authoring_controller: AuthoringWorkflowController | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("viewportPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toolbar = ViewportToolBar(actions, self)
        self.viewport = viewport
        self.overlay_host = ModelViewportOverlayHost(
            viewport,
            self,
            authoring_bridge=authoring_bridge,
            authoring_controller=authoring_controller,
        )
        self.agent_chat_drawer = self.overlay_host.agent_chat_drawer
        self._active_bottom_overlay: QWidget | None = None
        self.scope_creation_bar = ScopeCreationBar(self.overlay_host)
        self.planar_boolean_face_bar = PlanarBooleanFaceBar(self.overlay_host)
        self.scope_creation_bar.activeChanged.connect(
            lambda active: self._set_bottom_overlay_active(
                self.scope_creation_bar,
                active,
            )
        )
        self.planar_boolean_face_bar.activeChanged.connect(
            lambda active: self._set_bottom_overlay_active(
                self.planar_boolean_face_bar,
                active,
            )
        )
        self.overlay_host.set_bottom_overlay(self.scope_creation_bar)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.overlay_host, 1)

    def _set_bottom_overlay_active(
        self,
        overlay: QWidget,
        active: bool,
    ) -> None:
        if active:
            self._active_bottom_overlay = overlay
            self.overlay_host.set_bottom_overlay(overlay)
            self.overlay_host.set_bottom_overlay_visible(True)
        elif self._active_bottom_overlay is overlay:
            self._active_bottom_overlay = None
            self.overlay_host.set_bottom_overlay_visible(False)

    def set_geometry_context(self, enabled: bool) -> None:
        self.toolbar.set_geometry_context(enabled)
