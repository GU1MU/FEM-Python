"""Non-modal Chinese parameters for one detached face-sketch Boolean."""

from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from fem.geometry import (
    FaceSketchBooleanDirection,
    FaceSketchBooleanOperation,
)


@dataclass(frozen=True, slots=True)
class FaceSketchBooleanParameters:
    """Validated dialog values; enum implementation names never reach the UI."""

    operation: FaceSketchBooleanOperation
    direction: FaceSketchBooleanDirection
    distance: float
    participating_profile_ids: tuple[str, ...]


class FaceSketchBooleanDialog(QDialog):
    """Own parameters and exact-preview validity without touching Session."""

    parametersChanged = Signal(object)
    createFeatureRequested = Signal(object, int)
    returnSketchRequested = Signal(object)
    cancelRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("faceSketchBooleanDialog")
        self.setWindowTitle("拉伸布尔")
        self.setModal(False)
        self.setMinimumWidth(390)
        self._preview_generation = 0
        self._valid_generation: int | None = None
        self._closing_workflow = False
        self._profile_items: dict[str, QTreeWidgetItem] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self.operation_combo = QComboBox(self)
        for value in FaceSketchBooleanOperation:
            self.operation_combo.addItem(value.chinese_name, value.value)
        self.direction_combo = QComboBox(self)
        for value in FaceSketchBooleanDirection:
            self.direction_combo.addItem(value.chinese_name, value.value)
        self.distance_edit = QLineEdit("10", self)
        validator = QDoubleValidator(0.0, 1.0e100, 12, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.distance_edit.setValidator(validator)
        self.distance_edit.setAccessibleName("拉伸距离")

        form = QFormLayout()
        form.addRow("操作：", self.operation_combo)
        form.addRow("方向：", self.direction_combo)
        form.addRow("距离：", self.distance_edit)

        self.profile_tree = QTreeWidget(self)
        self.profile_tree.setObjectName("faceSketchBooleanProfiles")
        self.profile_tree.setHeaderLabels(("参与轮廓", "状态"))
        self.profile_tree.setRootIsDecorated(False)
        self.profile_tree.setMinimumHeight(135)

        self.preview_status = QLabel("等待精确预览", self)
        self.preview_status.setObjectName("faceSketchBooleanPreviewStatus")
        self.preview_status.setWordWrap(True)

        self.create_button = QPushButton("创建特征", self)
        self.create_button.setEnabled(False)
        self.return_button = QPushButton("返回草图", self)
        self.cancel_button = QPushButton("取消", self)
        buttons = QDialogButtonBox(self)
        buttons.addButton(
            self.create_button,
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        buttons.addButton(
            self.return_button,
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        buttons.addButton(
            self.cancel_button,
            QDialogButtonBox.ButtonRole.RejectRole,
        )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("材料轮廓", self))
        layout.addWidget(self.profile_tree)
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("预览状态：", self))
        status_row.addWidget(self.preview_status, 1)
        layout.addLayout(status_row)
        layout.addWidget(buttons)

        self.operation_combo.currentIndexChanged.connect(self._parameters_changed)
        self.direction_combo.currentIndexChanged.connect(self._parameters_changed)
        self.distance_edit.textChanged.connect(self._parameters_changed)
        self.profile_tree.itemChanged.connect(self._parameters_changed)
        self.create_button.clicked.connect(self._request_create)
        self.return_button.clicked.connect(self._request_return)
        self.cancel_button.clicked.connect(self.cancelRequested.emit)

    @property
    def preview_generation(self) -> int:
        return self._preview_generation

    @property
    def preview_is_valid(self) -> bool:
        return self._valid_generation == self._preview_generation

    def set_profiles(
        self,
        profile_ids: tuple[str, ...],
        *,
        selected_ids: tuple[str, ...] | None = None,
    ) -> None:
        profile_ids = tuple(profile_ids)
        selected = set(profile_ids if selected_ids is None else selected_ids)
        self.profile_tree.blockSignals(True)
        self.profile_tree.clear()
        self._profile_items.clear()
        for index, profile_id in enumerate(profile_ids, start=1):
            item = QTreeWidgetItem((f"材料轮廓 {index}", "参与"))
            item.setData(0, Qt.ItemDataRole.UserRole, profile_id)
            item.setCheckState(
                0,
                Qt.CheckState.Checked
                if profile_id in selected
                else Qt.CheckState.Unchecked,
            )
            self.profile_tree.addTopLevelItem(item)
            self._profile_items[profile_id] = item
        self.profile_tree.blockSignals(False)
        self._parameters_changed()

    def set_parameters(self, parameters: FaceSketchBooleanParameters) -> None:
        if type(parameters) is not FaceSketchBooleanParameters:
            raise TypeError("parameters must be FaceSketchBooleanParameters")
        widgets = (self.operation_combo, self.direction_combo, self.distance_edit)
        for widget in widgets:
            widget.blockSignals(True)
        self.operation_combo.setCurrentIndex(
            self.operation_combo.findData(parameters.operation.value)
        )
        self.direction_combo.setCurrentIndex(
            self.direction_combo.findData(parameters.direction.value)
        )
        self.distance_edit.setText(f"{parameters.distance:g}")
        for widget in widgets:
            widget.blockSignals(False)
        for profile_id, item in self._profile_items.items():
            item.setCheckState(
                0,
                Qt.CheckState.Checked
                if profile_id in parameters.participating_profile_ids
                else Qt.CheckState.Unchecked,
            )
        self._parameters_changed()

    def parameters(self) -> FaceSketchBooleanParameters | None:
        text = self.distance_edit.text().strip()
        try:
            distance = float(text)
        except ValueError:
            return None
        selected = tuple(
            profile_id
            for profile_id, item in self._profile_items.items()
            if item.checkState(0).value == 2
        )
        if not math.isfinite(distance) or distance <= 0.0 or not selected:
            return None
        try:
            operation = FaceSketchBooleanOperation(
                str(self.operation_combo.currentData())
            )
            direction = FaceSketchBooleanDirection(
                str(self.direction_combo.currentData())
            )
        except ValueError:
            return None
        return FaceSketchBooleanParameters(
            operation,
            direction,
            distance,
            selected,
        )

    def validation_reason(self) -> str:
        selected = any(
            item.checkState(0) is Qt.CheckState.Checked
            for item in self._profile_items.values()
        )
        if not selected:
            return "至少选择一个参与轮廓"
        try:
            distance = float(self.distance_edit.text().strip())
        except ValueError:
            return "距离必须为有限正值"
        if not math.isfinite(distance) or distance <= 0.0:
            return "距离必须为有限正值"
        return "参数无效"

    def set_preview_running(self, generation: int) -> None:
        self._preview_generation = int(generation)
        self._valid_generation = None
        self.create_button.setEnabled(False)
        self.preview_status.setText("正在计算精确预览…")

    def set_preview_valid(self, generation: int) -> None:
        if int(generation) != self._preview_generation:
            return
        self._valid_generation = int(generation)
        self.create_button.setEnabled(True)
        self.preview_status.setText("精确预览有效，可以创建特征")

    def set_preview_invalid(self, generation: int, reason: str) -> None:
        if int(generation) != self._preview_generation:
            return
        self._valid_generation = None
        self.create_button.setEnabled(False)
        self.preview_status.setText(str(reason).strip() or "精确预览无效")

    def close_for_workflow(self) -> None:
        self._closing_workflow = True
        self.close()

    def _parameters_changed(self, *_args) -> None:
        self._valid_generation = None
        self.create_button.setEnabled(False)
        parameters = self.parameters()
        for item in self._profile_items.values():
            item.setText(
                1,
                "参与"
                if item.checkState(0) is Qt.CheckState.Checked
                else "未参与",
            )
        if parameters is None:
            self.preview_status.setText(self.validation_reason())
        else:
            self.preview_status.setText("参数已变化，等待精确预览")
        self.parametersChanged.emit(parameters)

    def _request_create(self) -> None:
        parameters = self.parameters()
        if parameters is None or not self.preview_is_valid:
            return
        self.createFeatureRequested.emit(parameters, self._preview_generation)

    def _request_return(self) -> None:
        parameters = self.parameters()
        if parameters is not None:
            self.returnSketchRequested.emit(parameters)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._closing_workflow:
            self.cancelRequested.emit()
            event.ignore()
            return
        super().closeEvent(event)


__all__ = [
    "FaceSketchBooleanDialog",
    "FaceSketchBooleanParameters",
]
