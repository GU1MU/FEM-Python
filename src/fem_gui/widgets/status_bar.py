"""单行 CAE 上下文状态栏。"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFrame, QLabel, QStatusBar


class CAEStatusBar(QStatusBar):
    """分别显示任务、选择、对象、坐标、分析步和结果状态。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("caeStatusBar")
        self.setSizeGripEnabled(False)
        self.setFixedHeight(22)
        self._field_count = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(lambda: self.set_state("就绪"))
        self.state_label = self._add_field("statusState", "状态：就绪", 175)
        self.selection_label = self._add_field("statusSelection", "选择：节点", 100)
        self.object_label = self._add_field("statusObject", "对象：—", 110)
        self.coordinate_label = self._add_field("statusCoordinate", "坐标：—", 245)
        self.step_label = self._add_field("statusStep", "Step：—", 145)
        self.result_label = self._add_field("statusResult", "结果：—", 180)

    def _add_field(self, name: str, text: str, minimum: int) -> QLabel:
        if self._field_count:
            separator = QFrame(self)
            separator.setObjectName("statusSeparator")
            separator.setFrameShape(QFrame.Shape.VLine)
            self.addWidget(separator)
        label = QLabel(text, self)
        label.setObjectName(name)
        label.setMinimumWidth(0)
        label.setMaximumWidth(minimum)
        self.addWidget(label, 1)
        self._field_count += 1
        return label

    def set_state(self, text: str, timeout: int = 0) -> None:
        self._timer.stop()
        self.state_label.setText(f"状态：{text}")
        if timeout > 0:
            self._timer.start(timeout)

    def set_selection_mode(self, mode: str) -> None:
        self.selection_label.setText(f"选择：{'单元' if mode == 'element' else '节点'}")

    def set_object(self, text: str = "—", coordinates: str = "—") -> None:
        self.object_label.setText(f"对象：{text}")
        self.coordinate_label.setText(f"坐标：{coordinates}")

    def set_step(self, step_name: str | None) -> None:
        self.step_label.setText(f"Step：{step_name or '—'}")

    def set_result(self, text: str = "—") -> None:
        self.result_label.setText(f"结果：{text}")

    def reset_document(self) -> None:
        self.set_state("就绪")
        self.set_object()
        self.set_step(None)
        self.set_result()
