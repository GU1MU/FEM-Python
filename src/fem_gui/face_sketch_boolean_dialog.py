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
        self.setWindowTitle("Extrude Boolean")
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
            self.operation_combo.addItem(value.display_name, value.value)
        self.direction_combo = QComboBox(self)
        for value in FaceSketchBooleanDirection:
            self.direction_combo.addItem(value.display_name, value.value)
        self.distance_edit = QLineEdit("10", self)
        validator = QDoubleValidator(0.0, 1.0e100, 12, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.distance_edit.setValidator(validator)
        self.distance_edit.setAccessibleName("Extrude distance")
        self.distance_label = QLabel("Distance:", self)

        form = QFormLayout()
        form.addRow("Operation:", self.operation_combo)
        form.addRow("Direction:", self.direction_combo)
        form.addRow(self.distance_label, self.distance_edit)

        self.profile_tree = QTreeWidget(self)
        self.profile_tree.setObjectName("faceSketchBooleanProfiles")
        self.profile_tree.setHeaderLabels(("Included profiles", "Status"))
        self.profile_tree.setRootIsDecorated(False)
        self.profile_tree.setMinimumHeight(135)

        self.preview_status = QLabel("Awaiting exact preview", self)
        self.preview_status.setObjectName("faceSketchBooleanPreviewStatus")
        self.preview_status.setWordWrap(True)

        self.create_button = QPushButton("Create Feature", self)
        self.create_button.setEnabled(False)
        self.return_button = QPushButton("Back to Sketch", self)
        self.cancel_button = QPushButton("Cancel", self)
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
        layout.addWidget(QLabel("Material profiles", self))
        layout.addWidget(self.profile_tree)
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Preview status:", self))
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
            item = QTreeWidgetItem((f"Material profile {index}", "Included"))
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

    def fix_operation(self, operation: FaceSketchBooleanOperation) -> None:
        if type(operation) is not FaceSketchBooleanOperation:
            raise TypeError("operation must be a FaceSketchBooleanOperation")
        index = self.operation_combo.findData(operation.value)
        if index < 0:
            raise ValueError("Extrude Boolean operation is unavailable")
        self.operation_combo.setCurrentIndex(index)
        self.operation_combo.setEnabled(False)
        self.distance_label.setText(
            "Fuse height:"
            if operation is FaceSketchBooleanOperation.FUSE
            else "Cut depth:"
        )
        self.distance_edit.setAccessibleName(
            "Fuse height"
            if operation is FaceSketchBooleanOperation.FUSE
            else "Cut depth"
        )
        self.setWindowTitle(
            "Extrude Fuse"
            if operation is FaceSketchBooleanOperation.FUSE
            else "Extrude Cut"
        )

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
            return "Select at least one profile"
        try:
            distance = float(self.distance_edit.text().strip())
        except ValueError:
            return "Distance must be finite and positive"
        if not math.isfinite(distance) or distance <= 0.0:
            return "Distance must be finite and positive"
        return "Invalid parameters"

    def set_preview_running(self, generation: int) -> None:
        self._preview_generation = int(generation)
        self._valid_generation = None
        self.create_button.setEnabled(False)
        self.preview_status.setText("Computing exact preview...")

    def set_preview_valid(self, generation: int) -> None:
        if int(generation) != self._preview_generation:
            return
        self._valid_generation = int(generation)
        self.create_button.setEnabled(True)
        self.preview_status.setText("Exact preview valid; ready to create")

    def set_preview_invalid(self, generation: int, reason: str) -> None:
        if int(generation) != self._preview_generation:
            return
        self._valid_generation = None
        self.create_button.setEnabled(False)
        self.preview_status.setText(str(reason).strip() or "Exact preview invalid")

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
                "Included"
                if item.checkState(0) is Qt.CheckState.Checked
                else "Excluded",
            )
        if parameters is None:
            self.preview_status.setText(self.validation_reason())
        else:
            self.preview_status.setText("Parameters changed; awaiting exact preview")
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
