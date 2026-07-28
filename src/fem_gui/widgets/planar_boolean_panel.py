"""Non-modal panel for strict planar Boolean authoring."""

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

from ..planar_boolean import PlanarBooleanController


class PlanarBooleanPanel(QWidget):
    """Expose one committed target Face and one detached tool sketch."""

    targetSelectionRequested = Signal()
    toolSketchRequested = Signal()
    operationChanged = Signal(str)
    finishRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("planarBooleanPanel")
        self.setMinimumWidth(320)
        self.setMaximumWidth(480)
        self._controller: PlanarBooleanController | None = None
        self._preview_running = False
        self._preview_valid = False
        self._build_ui()
        self.hide()

    @property
    def controller(self) -> PlanarBooleanController | None:
        return self._controller

    def _build_ui(self) -> None:
        title = QLabel("二维布尔", self)
        title.setObjectName("planarBooleanTitle")
        self.operation_combo = QComboBox(self)
        self.operation_combo.setObjectName("planarBooleanOperation")
        self.operation_combo.addItem("合并", "fuse")
        self.operation_combo.addItem("切除", "cut")
        self.operation_combo.currentIndexChanged.connect(self._operation_changed)

        self.target_label = QLabel("未选择", self)
        self.target_label.setObjectName("planarBooleanTargetLabel")
        self.target_button = QPushButton("选择", self)
        self.target_button.setObjectName("planarBooleanSelectTarget")
        self.target_button.clicked.connect(self.targetSelectionRequested.emit)
        target_row = QWidget(self)
        target_layout = QHBoxLayout(target_row)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.target_label, 1)
        target_layout.addWidget(self.target_button)

        self.tool_label = QLabel("未绘制", self)
        self.tool_label.setObjectName("planarBooleanToolLabel")
        self.tool_button = QPushButton("绘制", self)
        self.tool_button.setObjectName("planarBooleanDrawTool")
        self.tool_button.clicked.connect(self.toolSketchRequested.emit)
        tool_row = QWidget(self)
        tool_layout = QHBoxLayout(tool_row)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.addWidget(self.tool_label, 1)
        tool_layout.addWidget(self.tool_button)

        self.status_label = QLabel("请选择目标面并绘制工具轮廓", self)
        self.status_label.setObjectName("planarBooleanPreviewStatus")
        self.status_label.setWordWrap(True)
        self.finish_button = QPushButton("完成", self)
        self.finish_button.setObjectName("planarBooleanFinish")
        self.finish_button.setEnabled(False)
        self.finish_button.clicked.connect(self.finishRequested.emit)
        cancel = QPushButton("取消", self)
        cancel.setObjectName("planarBooleanCancel")
        cancel.clicked.connect(self.cancelRequested.emit)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.finish_button)
        buttons.addWidget(cancel)

        form = QFormLayout()
        form.addRow("操作", self.operation_combo)
        form.addRow("目标面", target_row)
        form.addRow("工具轮廓", tool_row)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(QLabel("预览与诊断", self))
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        layout.addLayout(buttons)

    def begin(self, controller: PlanarBooleanController) -> None:
        if type(controller) is not PlanarBooleanController:
            raise TypeError("controller must be PlanarBooleanController")
        self._controller = controller
        self._preview_running = False
        self._preview_valid = False
        self._set_inputs_enabled(True)
        self.operation_combo.setCurrentIndex(
            self.operation_combo.findData(controller.operation)
        )
        self.refresh()
        self.show()

    def end(self) -> None:
        self._controller = None
        self._preview_running = False
        self._preview_valid = False
        self._set_inputs_enabled(True)
        self.hide()

    def refresh(self) -> None:
        controller = self._controller
        if controller is None:
            return
        self.target_label.setText(controller.target_label())
        self.tool_label.setText(controller.tool_label())
        self.tool_button.setText(
            "编辑" if controller.tool_geometry is not None else "绘制"
        )
        self.finish_button.setEnabled(
            controller.ready and self._preview_valid and not self._preview_running
        )

    def set_preview_running(self, running: bool) -> None:
        self._preview_running = bool(running)
        self._set_inputs_enabled(not self._preview_running)
        self.refresh()

    def set_preview_valid(self, valid: bool) -> None:
        self._preview_valid = bool(valid)
        self.refresh()

    def show_status(self, message: str) -> None:
        self.status_label.setText(str(message))

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for widget in (
            self.operation_combo,
            self.target_button,
            self.tool_button,
        ):
            widget.setEnabled(bool(enabled))

    def _operation_changed(self) -> None:
        operation = self.operation_combo.currentData()
        if operation in {"fuse", "cut"}:
            self.operationChanged.emit(str(operation))


__all__ = ["PlanarBooleanPanel"]
