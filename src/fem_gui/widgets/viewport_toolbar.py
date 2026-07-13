"""三维视口顶部常驻的高频操作工具栏。"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QToolBar, QToolButton, QVBoxLayout, QWidget

from ..icons import icon


class ViewportToolBar(QToolBar):
    """仅复用主窗口 QAction 的紧凑视口工具栏。"""

    def __init__(self, actions: Mapping[str, QAction], parent=None) -> None:
        super().__init__("视口工具", parent)
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

        self._add_group(actions, ("select_node", "select_element", "clear_selection", "selected_info"))
        self.addSeparator()
        self._add_group(actions, ("nodes", "edges", "node_labels", "element_labels", "symbols"))
        self.addSeparator()
        self._add_group(actions, ("undeformed", "deformed", "contour"))

    def _add_group(self, actions: Mapping[str, QAction], names: tuple[str, ...]) -> None:
        for name in names:
            action = actions[name]
            self.addAction(action)
            button = self.widgetForAction(action)
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


class ViewportPanel(QWidget):
    """组合常驻工具栏与有限元三维视口。"""

    def __init__(self, viewport: QWidget, actions: Mapping[str, QAction], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("viewportPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toolbar = ViewportToolBar(actions, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(viewport, 1)
