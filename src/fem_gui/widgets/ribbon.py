"""使用原生 PySide6 控件构成的紧凑 Ribbon。"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
)


class RibbonGroup(QFrame):
    """带底部名称的 Ribbon 命令组。"""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ribbonGroup")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(5, 3, 6, 1)
        outer.setSpacing(0)
        self._content = QHBoxLayout()
        self._content.setContentsMargins(0, 0, 0, 0)
        self._content.setSpacing(3)
        outer.addLayout(self._content, 1)
        label = QLabel(title, self)
        label.setObjectName("ribbonGroupTitle")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(label)
        self._small_grid: QGridLayout | None = None
        self._small_count = 0

    def add_action(
        self,
        action: QAction,
        *,
        large: bool = False,
        compact: bool = False,
        tile: bool = False,
    ) -> QToolButton:
        """添加绑定现有 QAction 的大按钮、方形按钮或两行小按钮。"""
        button = QToolButton(self)
        button.setDefaultAction(action)
        text_width = button.fontMetrics().horizontalAdvance(action.text())
        if tile:
            button.setObjectName("ribbonTileButton")
            button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextUnderIcon
            )
            button.setIconSize(QSize(30, 30))
            button.setFixedSize(92, 68)
            button.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            self._content.addWidget(button)
            self._small_grid = None
            self._small_count = 0
        elif large:
            button.setObjectName("ribbonLargeButton")
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(QSize(34, 34))
            button.setFixedHeight(60)
            # Keep the complete command label visible.  A fixed 54 px
            # minimum works for pictograms, but elides longer labels such as
            # Keep semantic command labels visible when the page is laid out tightly.
            button.setMinimumWidth(max(54, text_width + 16))
            self._content.addWidget(button)
            # Keep later compact commands in a new run so a large command
            # inserted between two compact runs remains in its declared
            # position instead of being appended after the existing grid.
            self._small_grid = None
            self._small_count = 0
        else:
            button.setObjectName("ribbonSmallButton")
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setIconSize(QSize(24, 24))
            button.setFixedHeight(30)
            # Reserve room for the icon, the inter-item gap and the text.
            # The previous fixed 72 px minimum allowed labels such as
            # “导出 CSV” to be elided inside a two-column group.
            button_width = max(72, text_width + 38)
            button.setMinimumWidth(button_width)
            button.setMaximumWidth(max(146, button_width))
            if compact:
                button.setObjectName("ribbonCompactButton")
                self._content.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
                return button
            if self._small_grid is None:
                host = QWidget(self)
                self._small_grid = QGridLayout(host)
                self._small_grid.setContentsMargins(0, 0, 0, 0)
                self._small_grid.setHorizontalSpacing(2)
                self._small_grid.setVerticalSpacing(1)
                self._content.addWidget(host)
            row = self._small_count % 2
            column = self._small_count // 2
            self._small_grid.addWidget(button, row, column)
            self._small_count += 1
        return button

    def add_widget(self, widget: QWidget) -> None:
        """添加分析步等真实上下文控件。"""
        # A real widget starts a new horizontal segment.  Without resetting
        # the small-button grid, a later small action would be appended to a
        # grid that is already positioned before this widget, making the
        # visual order differ from the declared order.
        self._small_grid = None
        self._small_count = 0
        self._content.addWidget(widget)


class RibbonPage(QWidget):
    """一个工作流模块对应的 Ribbon 页面。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ribbonPage")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(2, 0, 2, 0)
        self._layout.setSpacing(0)
        self._layout.addStretch(1)

    def add_group(self, title: str) -> RibbonGroup:
        group = RibbonGroup(title, self)
        self._layout.insertWidget(self._layout.count() - 1, group)
        return group


class RibbonWidget(QWidget):
    """模块页签与命令页组成的无第三方依赖 Ribbon。"""

    moduleChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ribbonWidget")
        self.setFixedHeight(110)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.tab_bar = QTabBar(self)
        self.tab_bar.setObjectName("ribbonTabs")
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setExpanding(False)
        self.tab_bar.setFixedHeight(29)
        self.stack = QStackedWidget(self)
        self.stack.setObjectName("ribbonStack")
        layout.addWidget(self.tab_bar)
        layout.addWidget(self.stack, 1)
        self.tab_bar.currentChanged.connect(self._change_page)

    def add_page(self, name: str) -> RibbonPage:
        page = RibbonPage(self)
        page.setObjectName(f"ribbonPage_{name}")
        self.tab_bar.addTab(name)
        self.stack.addWidget(page)
        return page

    def set_current(self, name: str) -> None:
        for index in range(self.tab_bar.count()):
            if self.tab_bar.tabText(index) == name:
                self.tab_bar.setCurrentIndex(index)
                return

    def _change_page(self, index: int) -> None:
        if index < 0:
            return
        self.stack.setCurrentIndex(index)
        self.moduleChanged.emit(self.tab_bar.tabText(index))
