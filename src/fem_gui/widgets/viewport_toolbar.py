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

from ..icons import icon


class ViewportToolBar(QToolBar):
    """仅复用主窗口 QAction 的紧凑视口工具栏。"""

    def __init__(self, actions: Mapping[str, QAction], parent=None) -> None:
        super().__init__("视口工具", parent)
        self._action_widgets: dict[str, QToolButton] = {}
        self.setObjectName("viewportToolbar")
        self.setMovable(False)
        self.setFloatable(False)
        # Coordinate-plane PNGs already contain the complete axis legend.  Keep
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

        self._model_selection_actions = (
            "select_node", "select_element", "select_edge", "selected_info",
        )
        self._geometry_selection_actions = (
            "geometry_select_point", "geometry_select_edge",
            "geometry_select_face", "geometry_select_body",
        )
        self._add_group(actions, self._model_selection_actions)
        self._add_group(actions, self._geometry_selection_actions)
        self.set_geometry_context(False)
        self.addSeparator()
        self._add_group(actions, ("nodes", "edges", "node_labels", "element_labels", "symbols"))
        self.addSeparator()
        self._add_group(actions, ("undeformed", "deformed", "contour"))

    def _add_group(self, actions: Mapping[str, QAction], names: tuple[str, ...]) -> None:
        for name in names:
            action = actions[name]
            self.addAction(action)
            button = self.widgetForAction(action)
            if isinstance(button, QToolButton):
                button.setObjectName(f"viewportAction_{name}")
                self._action_widgets[name] = button
            if isinstance(button, QToolButton) and name in {
                "symbols", "undeformed", "deformed", "overlay", "contour", "select_element",
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
        """Swap FEM and CAD selection buttons without changing shared actions."""
        for name in self._model_selection_actions:
            widget = self._action_widgets.get(name)
            if widget is not None:
                widget.setVisible(not enabled)
        for name in self._geometry_selection_actions:
            widget = self._action_widgets.get(name)
            if widget is not None:
                widget.setVisible(enabled)


class ScopeCreationBar(QWidget):
    """Persistent completion controls for one guided scope selection."""

    createRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("scopeCreationBar")
        self.type_value = QLabel("—", self)
        self.type_value.setObjectName("scopeCreationType")
        self.name_edit = QLineEdit(self)
        self.name_edit.setObjectName("scopeCreationName")
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
        layout.addWidget(self.create_button)
        self.hide()

    def begin(self, scope_type: str, suggested_name: str) -> None:
        self.type_value.setText(str(scope_type))
        self.name_edit.setText(str(suggested_name))
        self.name_edit.selectAll()
        self.create_button.setEnabled(False)
        self.show()

    def set_selection_ready(self, ready: bool) -> None:
        self.create_button.setEnabled(bool(ready))

    def scope_name(self) -> str:
        return self.name_edit.text().strip()

    def finish(self) -> None:
        self.create_button.setEnabled(False)
        self.hide()


class ViewportPanel(QWidget):
    """组合常驻工具栏与有限元三维视口。"""

    def __init__(self, viewport: QWidget, actions: Mapping[str, QAction], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("viewportPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toolbar = ViewportToolBar(actions, self)
        self.scope_creation_bar = ScopeCreationBar(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(viewport, 1)
        layout.addWidget(self.scope_creation_bar)

    def set_geometry_context(self, enabled: bool) -> None:
        self.toolbar.set_geometry_context(enabled)
