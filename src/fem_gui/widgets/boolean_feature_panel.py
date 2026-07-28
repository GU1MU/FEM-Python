"""Non-modal panel for strict solid Body Boolean authoring."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..body_boolean import BodyBooleanController


class BooleanFeaturePanel(QWidget):
    """Expose explicit target/tool slots backed by a detached controller."""

    selectionRequested = Signal(str)
    operationChanged = Signal(str)
    finishRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("booleanFeaturePanel")
        self.setMinimumWidth(320)
        self.setMaximumWidth(480)
        self._controller: BodyBooleanController | None = None
        self._preview_running = False
        self._preview_valid = False
        self._build_ui()
        self.hide()

    @property
    def controller(self) -> BodyBooleanController | None:
        return self._controller

    def _build_ui(self) -> None:
        title = QLabel("实体布尔", self)
        title.setObjectName("booleanFeatureTitle")

        self.operation_combo = QComboBox(self)
        self.operation_combo.setObjectName("booleanOperationCombo")
        self.operation_combo.addItem("合并", "fuse")
        self.operation_combo.addItem("切除", "cut")
        self.operation_combo.currentIndexChanged.connect(
            self._operation_changed
        )

        self.target_label = QLabel("未选择", self)
        self.target_label.setObjectName("booleanTargetLabel")
        self.target_button = QPushButton("选择", self)
        self.target_button.setObjectName("booleanSelectTarget")
        self.target_button.clicked.connect(
            lambda: self.selectionRequested.emit("target")
        )
        target_row = QWidget(self)
        target_layout = QHBoxLayout(target_row)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.target_label, 1)
        target_layout.addWidget(self.target_button)

        self.tool_label = QLabel("未选择", self)
        self.tool_label.setObjectName("booleanToolLabel")
        self.tool_button = QPushButton("选择", self)
        self.tool_button.setObjectName("booleanSelectTool")
        self.tool_button.clicked.connect(
            lambda: self.selectionRequested.emit("tool")
        )
        tool_row = QWidget(self)
        tool_layout = QHBoxLayout(tool_row)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.addWidget(self.tool_label, 1)
        tool_layout.addWidget(self.tool_button)

        self.result_label = QLabel("未确定", self)
        self.result_label.setObjectName("booleanResultLabel")
        tool_policy = QLabel("操作成功后消耗", self)

        self.status_label = QLabel("请选择目标体和工具体", self)
        self.status_label.setObjectName("booleanPreviewStatus")
        self.status_label.setWordWrap(True)

        self.finish_button = QPushButton("完成", self)
        self.finish_button.setObjectName("booleanFinishButton")
        self.finish_button.setEnabled(False)
        self.finish_button.clicked.connect(self.finishRequested.emit)
        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.setObjectName("booleanCancelButton")
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.finish_button)
        buttons.addWidget(self.cancel_button)

        form = QFormLayout()
        form.addRow("操作", self.operation_combo)
        form.addRow("目标体", target_row)
        form.addRow("工具体", tool_row)
        form.addRow("结果体", self.result_label)
        form.addRow("工具体处理", tool_policy)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        layout.addLayout(buttons)

    def begin(self, controller: BodyBooleanController) -> None:
        if type(controller) is not BodyBooleanController:
            raise TypeError("controller must be BodyBooleanController")
        self._controller = controller
        self._preview_running = False
        self._preview_valid = False
        index = self.operation_combo.findData(controller.operation)
        self.operation_combo.setCurrentIndex(index)
        self.show_status("请选择目标体和工具体")
        self.refresh()
        self.show()

    def end(self) -> None:
        self._controller = None
        self._preview_running = False
        self._preview_valid = False
        self.hide()

    def refresh(self) -> None:
        controller = self._controller
        if controller is None:
            return
        self.target_label.setText(
            controller.body_label(controller.target_body_id)
        )
        self.tool_label.setText(
            controller.body_label(controller.tool_body_id)
        )
        self.result_label.setText(
            controller.body_label(controller.target_body_id)
        )
        self.finish_button.setEnabled(
            controller.ready
            and self._preview_valid
            and not self._preview_running
        )

    def set_preview_running(self, running: bool) -> None:
        self._preview_running = bool(running)
        enabled = not self._preview_running
        self.operation_combo.setEnabled(enabled)
        self.target_button.setEnabled(enabled)
        self.tool_button.setEnabled(enabled)
        self.refresh()

    def set_preview_valid(self, valid: bool) -> None:
        self._preview_valid = bool(valid)
        self.refresh()

    def show_status(self, message: str) -> None:
        self.status_label.setText(str(message))

    def _operation_changed(self) -> None:
        operation = self.operation_combo.currentData()
        if operation in {"fuse", "cut"}:
            self.operationChanged.emit(str(operation))


__all__ = ["BooleanFeaturePanel"]
