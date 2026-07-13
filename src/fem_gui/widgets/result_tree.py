"""只展示真实可用结果族的精简结果树。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

from ..visualization.result_adapter import ResultData


ROLE_FIELD = int(Qt.ItemDataRole.UserRole)


class ResultTree(QTreeWidget):
    """按位移、反力和应力组织当前单步结果。"""

    fieldActivated = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("resultTree")
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.itemDoubleClicked.connect(self._activate_item)
        self.clear_result()

    def clear_result(self) -> None:
        self.clear()
        item = QTreeWidgetItem(["尚无分析结果"])
        item.setData(0, ROLE_FIELD, None)
        self.addTopLevelItem(item)

    def set_result(self, step_name: str, data: ResultData) -> None:
        self.clear()
        root = QTreeWidgetItem(["分析结果"])
        step = QTreeWidgetItem([step_name or "当前分析步"])
        root.addChild(step)
        for label, key in self._families(data):
            item = QTreeWidgetItem([label])
            item.setData(0, ROLE_FIELD, key)
            step.addChild(item)
        self.addTopLevelItem(root)
        root.setExpanded(True)
        step.setExpanded(True)

    @staticmethod
    def _families(data: ResultData) -> tuple[tuple[str, str], ...]:
        fields = data.fields
        families: list[tuple[str, str]] = []
        displacement = "U" if "U" in fields else next((key for key in fields if key.startswith("U")), None)
        reaction = "RF" if "RF" in fields else next((key for key in fields if key.startswith(("RF", "RM"))), None)
        stress_keys = [key for key in fields if not key.startswith(("U", "R3", "RF", "RM"))]
        stress = next((key for key in stress_keys if key.endswith("Mises")), stress_keys[0] if stress_keys else None)
        if displacement is not None:
            families.append(("位移 U", displacement))
        if reaction is not None:
            families.append(("反力 RF", reaction))
        if stress is not None:
            families.append(("应力 S", stress))
        return tuple(families)

    def _activate_item(self, item: QTreeWidgetItem) -> None:
        field = item.data(0, ROLE_FIELD)
        if field:
            self.fieldActivated.emit(str(field))
