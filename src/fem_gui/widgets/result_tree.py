"""只展示真实可用结果族的精简结果树。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

from ..visualization.result_adapter import ResultData, field_family


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
        labels = {
            "U": "位移 U",
            "R": "转角 R",
            "RF": "反力 RF",
            "RM": "反力矩 RM",
            "S": "应力 S",
        }
        for family in ("U", "R", "RF", "RM", "S"):
            keys = [key for key in fields if field_family(key) == family]
            if not keys:
                continue
            preferred = {
                "U": "U",
                "RF": "RF",
            }.get(family)
            selected = preferred if preferred in fields else keys[0]
            if family == "S":
                selected = next(
                    (key for key in keys if key.endswith(":S11AbsMax")),
                    next(
                        (key for key in keys if key.endswith(":Mises")),
                        keys[0],
                    ),
                )
            families.append((labels[family], selected))
        return tuple(families)

    def _activate_item(self, item: QTreeWidgetItem) -> None:
        field = item.data(0, ROLE_FIELD)
        if field:
            self.fieldActivated.emit(str(field))
