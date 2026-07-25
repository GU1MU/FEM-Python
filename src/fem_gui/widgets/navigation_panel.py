"""左侧模型与结果双页导航区。"""

from __future__ import annotations

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .model_tree import ModelTree
from .result_tree import ResultTree


class NavigationPanel(QWidget):
    """组合精简模型树和当前结果树。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("navigationPanel")
        self.setMinimumWidth(210)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("navigationTabs")
        self.model_tree = ModelTree(self.tabs)
        self.result_tree = ResultTree(self.tabs)
        self.tabs.addTab(self.model_tree, "模型")
        self.tabs.addTab(self.result_tree, "结果")
        layout.addWidget(self.tabs)

    def show_model(self) -> None:
        self.tabs.setCurrentWidget(self.model_tree)

    def show_result(self) -> None:
        self.tabs.setCurrentWidget(self.result_tree)
