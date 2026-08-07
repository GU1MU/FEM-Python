"""CSV result export dialog with an independent scalar field selection."""

from __future__ import annotations

from pathlib import Path

from fem.application.results import (
    FieldAvailability,
    FieldPosition,
    FieldState,
    ResultCatalog,
    ResultVariable,
    ScalarFieldSelection,
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .dialogs import configure_form_layout
from .result_presentation import (
    result_position_label,
    result_variable_label,
    visible_result_fields,
)


class _ExactDataComboBox(QComboBox):
    """Keep typed result identities from being coerced by QVariant."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._exact_user_data: list[object] = []

    def addItem(self, text: str, user_data: object = None) -> None:
        super().addItem(text)
        self._exact_user_data.append(user_data)

    def clear(self) -> None:
        super().clear()
        self._exact_user_data.clear()

    def itemData(
        self,
        index: int,
        role: int = Qt.ItemDataRole.UserRole,
    ) -> object:
        if role == Qt.ItemDataRole.UserRole:
            if 0 <= index < len(self._exact_user_data):
                return self._exact_user_data[index]
            return None
        return super().itemData(index, role)

    def currentData(
        self,
        role: int = Qt.ItemDataRole.UserRole,
    ) -> object:
        return self.itemData(self.currentIndex(), role)

    def findData(
        self,
        data: object,
        role: int = Qt.ItemDataRole.UserRole,
        flags: Qt.MatchFlag = (
            Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchCaseSensitive
        ),
    ) -> int:
        if role == Qt.ItemDataRole.UserRole:
            return next(
                (
                    index
                    for index, candidate in enumerate(self._exact_user_data)
                    if type(candidate) is type(data) and candidate == data
                ),
                -1,
            )
        return super().findData(data, role, flags)


class ResultCsvExportDialog(QDialog):
    """Choose one ready scalar result and its CSV destination."""

    def __init__(
        self,
        catalog: ResultCatalog,
        *,
        current_selection: ScalarFieldSelection,
        parent=None,
    ) -> None:
        if type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be ResultCatalog")
        if type(current_selection) is not ScalarFieldSelection:
            raise TypeError(
                "current_selection must be ScalarFieldSelection"
            )
        fields = tuple(
            availability
            for availability in visible_result_fields(catalog.fields)
            if availability.state is FieldState.READY
        )
        if not fields:
            raise ValueError("catalog has no ready result field to export")

        super().__init__(parent)
        self.setWindowTitle("导出 CSV")
        self.setMinimumWidth(520)
        self._catalog = catalog
        self._fields = fields

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        self.variable_combo = _ExactDataComboBox(self)
        self.position_combo = _ExactDataComboBox(self)
        self.component_combo = _ExactDataComboBox(self)
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("请选择 CSV 保存路径")
        self.browse_button = QPushButton("浏览…", self)

        path_host = QWidget(self)
        path_layout = QHBoxLayout(path_host)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(self.browse_button)

        form.addRow("场变量：", self.variable_combo)
        form.addRow("结果位置：", self.position_combo)
        form.addRow("分量：", self.component_combo)
        form.addRow("保存到：", path_host)
        layout.addLayout(form)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.export_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Ok
        )
        self.export_button.setText("导出")
        self.button_box.button(
            QDialogButtonBox.StandardButton.Cancel
        ).setText("取消")
        layout.addWidget(self.button_box)

        initial = self._initial_selection(current_selection)
        self._populate_variables(initial)
        self._populate_positions(initial)
        self._populate_components(initial)

        self.variable_combo.currentIndexChanged.connect(
            self._variable_changed
        )
        self.position_combo.currentIndexChanged.connect(
            self._position_changed
        )
        self.path_edit.textChanged.connect(self._refresh_export_enabled)
        self.browse_button.clicked.connect(self._browse)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self._refresh_export_enabled()

    @property
    def catalog(self) -> ResultCatalog:
        return self._catalog

    def current_selection(self) -> ScalarFieldSelection:
        selection = self.component_combo.currentData()
        if type(selection) is not ScalarFieldSelection:
            raise RuntimeError("no scalar result component is selected")
        return selection

    def target_path(self) -> Path:
        text = self.path_edit.text().strip()
        if not text:
            raise ValueError("CSV 保存路径不能为空")
        return Path(text).with_suffix(".csv")

    def accept(self) -> None:
        try:
            target = self.target_path()
        except ValueError:
            return
        self.path_edit.setText(str(target))
        super().accept()

    def _initial_selection(
        self,
        current_selection: ScalarFieldSelection,
    ) -> ScalarFieldSelection:
        for availability in self._fields:
            if (
                availability.key == current_selection.field_key
                and current_selection.component
                in availability.descriptor.columns
            ):
                return current_selection
        first = self._fields[0]
        return ScalarFieldSelection(
            first.key,
            first.descriptor.default_component,
        )

    def _populate_variables(
        self,
        preferred: ScalarFieldSelection,
    ) -> None:
        variables: list[ResultVariable] = []
        for availability in self._fields:
            variable = availability.descriptor.field_id.variable
            if variable not in variables:
                variables.append(variable)
                self.variable_combo.addItem(
                    result_variable_label(variable),
                    variable,
                )
        index = self.variable_combo.findData(
            preferred.field_key.request.field_id.variable
        )
        self.variable_combo.setCurrentIndex(index if index >= 0 else 0)

    def _populate_positions(
        self,
        preferred: ScalarFieldSelection | None = None,
    ) -> None:
        self.position_combo.blockSignals(True)
        self.position_combo.clear()
        variable = self.variable_combo.currentData()
        positions: list[FieldPosition] = []
        for availability in self._fields:
            field_id = availability.descriptor.field_id
            if (
                field_id.variable is variable
                and field_id.position not in positions
            ):
                positions.append(field_id.position)
                self.position_combo.addItem(
                    result_position_label(field_id.position),
                    field_id.position,
                )
        preferred_position = (
            preferred.field_key.request.field_id.position
            if preferred is not None
            and preferred.field_key.request.field_id.variable is variable
            else None
        )
        index = self.position_combo.findData(preferred_position)
        self.position_combo.setCurrentIndex(index if index >= 0 else 0)
        self.position_combo.blockSignals(False)

    def _populate_components(
        self,
        preferred: ScalarFieldSelection | None = None,
    ) -> None:
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        availabilities = self._matching_fields()
        component_counts: dict[str, int] = {}
        for availability in availabilities:
            for component in availability.descriptor.columns:
                component_counts[component] = (
                    component_counts.get(component, 0) + 1
                )
        for availability in availabilities:
            for component in availability.descriptor.columns:
                label = component
                if component_counts[component] > 1:
                    label = (
                        f"{component} · contract "
                        f"{availability.key.recovery_contract}"
                    )
                self.component_combo.addItem(
                    label,
                    ScalarFieldSelection(availability.key, component),
                )
        index = self.component_combo.findData(preferred)
        if index < 0 and availabilities:
            default = ScalarFieldSelection(
                availabilities[0].key,
                availabilities[0].descriptor.default_component,
            )
            index = self.component_combo.findData(default)
        self.component_combo.setCurrentIndex(index if index >= 0 else 0)
        self.component_combo.blockSignals(False)

    def _matching_fields(self) -> tuple[FieldAvailability, ...]:
        variable = self.variable_combo.currentData()
        position = self.position_combo.currentData()
        return tuple(
            availability
            for availability in self._fields
            if (
                availability.descriptor.field_id.variable is variable
                and availability.descriptor.field_id.position is position
            )
        )

    def _variable_changed(self, _index: int) -> None:
        self._populate_positions()
        self._populate_components()

    def _position_changed(self, _index: int) -> None:
        self._populate_components()

    def _browse(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "选择 CSV 保存位置",
            self.path_edit.text(),
            "CSV 文件 (*.csv)",
        )
        if not path:
            return
        self.path_edit.setText(str(Path(path).with_suffix(".csv")))

    def _refresh_export_enabled(self, *_args: object) -> None:
        self.export_button.setEnabled(bool(self.path_edit.text().strip()))
