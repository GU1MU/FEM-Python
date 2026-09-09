"""Left navigation panel with model and result tabs."""

from __future__ import annotations

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .model_tree import ModelTree
from .result_tree import ResultTree


class NavigationPanel(QWidget):
    """Combine the compact model tree and current result tree."""

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
        self.tabs.addTab(self.model_tree, "Model")
        self.tabs.addTab(self.result_tree, "Results")
        layout.addWidget(self.tabs)

    def show_model(self) -> None:
        self.tabs.setCurrentWidget(self.model_tree)

    def show_result(self) -> None:
        self.tabs.setCurrentWidget(self.result_tree)
