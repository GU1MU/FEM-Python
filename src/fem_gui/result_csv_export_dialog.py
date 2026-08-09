"""CSV result export dialog with an independent scalar field selection."""

from __future__ import annotations

from pathlib import Path

from fem.application.results import (
    FieldAvailability,
    FieldState,
    ResultCatalog,
    ResultFieldId,
    ResultVariable,
    ScalarFieldSelection,
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .dialogs import configure_form_layout
from .result_presentation import (
    result_field_position_label,
    result_variable_label,
    visible_result_fields,
)


_ICON_ROOT = Path(__file__).with_name("resources") / "icons"
_SCROLL_UP_ARROW = (_ICON_ROOT / "agent_chat_scroll_up.svg").resolve().as_posix()
_SCROLL_DOWN_ARROW = (
    _ICON_ROOT / "agent_chat_scroll_down.svg"
).resolve().as_posix()

_COMPONENT_LIST_STYLESHEET = f"""
QListWidget#resultCsvComponentList {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget#resultCsvComponentList::item {{
    min-height: 30px;
    padding: 2px 5px;
    border-radius: 4px;
}}
QListWidget#resultCsvComponentList QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 12px 0 12px 0;
}}
QListWidget#resultCsvComponentList QScrollBar::handle:vertical {{
    background: rgba(76, 88, 98, 92);
    min-height: 34px;
    border-radius: 4px;
    margin: 1px 2px;
}}
QListWidget#resultCsvComponentList QScrollBar::handle:vertical:hover {{
    background: rgba(76, 88, 98, 138);
}}
QListWidget#resultCsvComponentList QScrollBar::add-line:vertical,
QListWidget#resultCsvComponentList QScrollBar::sub-line:vertical {{
    background: transparent;
    border: none;
    height: 12px;
    subcontrol-origin: margin;
}}
QListWidget#resultCsvComponentList QScrollBar::sub-line:vertical {{
    subcontrol-position: top;
}}
QListWidget#resultCsvComponentList QScrollBar::add-line:vertical {{
    subcontrol-position: bottom;
}}
QListWidget#resultCsvComponentList QScrollBar::up-arrow:vertical {{
    image: url("{_SCROLL_UP_ARROW}");
    width: 8px;
    height: 6px;
}}
QListWidget#resultCsvComponentList QScrollBar::down-arrow:vertical {{
    image: url("{_SCROLL_DOWN_ARROW}");
    width: 8px;
    height: 6px;
}}
QListWidget#resultCsvComponentList QScrollBar::add-page:vertical,
QListWidget#resultCsvComponentList QScrollBar::sub-page:vertical {{
    background: transparent;
}}
"""


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
        self.component_list = QListWidget(self)
        self.component_list.setObjectName("resultCsvComponentList")
        self.component_list.setMinimumHeight(120)
        self.component_list.setAlternatingRowColors(False)
        self.component_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.component_list.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.component_list.verticalScrollBar().setSingleStep(12)
        self.component_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.component_list.setStyleSheet(_COMPONENT_LIST_STYLESHEET)
        self._component_selections: list[ScalarFieldSelection] = []
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
        form.addRow("分量：", self.component_list)
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
        self.cancel_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.cancel_button.setText("取消")
        button_width = max(
            self.browse_button.sizeHint().width(),
            self.export_button.sizeHint().width(),
            self.cancel_button.sizeHint().width(),
        )
        button_height = max(
            self.browse_button.sizeHint().height(),
            self.export_button.sizeHint().height(),
            self.cancel_button.sizeHint().height(),
        )
        for button in (
            self.browse_button,
            self.export_button,
            self.cancel_button,
        ):
            button.setFixedSize(button_width, button_height)
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
        self.component_list.itemChanged.connect(
            self._component_item_changed
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
        selections = self.current_selections()
        if not selections:
            raise RuntimeError("no scalar result component is selected")
        return selections[0]

    def current_selections(self) -> tuple[ScalarFieldSelection, ...]:
        return tuple(
            selection
            for index, selection in enumerate(self._component_selections)
            if self.component_list.item(index).checkState()
            is Qt.CheckState.Checked
        )

    def target_path(self) -> Path:
        text = self.path_edit.text().strip()
        if not text:
            raise ValueError("CSV 保存路径不能为空")
        return Path(text).with_suffix(".csv")

    def accept(self) -> None:
        if not self.current_selections():
            return
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
        field_ids: list[ResultFieldId] = []
        for availability in self._fields:
            field_id = availability.descriptor.field_id
            if (
                field_id.variable is variable
                and field_id not in field_ids
            ):
                field_ids.append(field_id)
                self.position_combo.addItem(
                    result_field_position_label(field_id),
                    field_id,
                )
        preferred_field_id = (
            preferred.field_key.request.field_id
            if preferred is not None
            and preferred.field_key.request.field_id.variable is variable
            else None
        )
        index = self.position_combo.findData(preferred_field_id)
        self.position_combo.setCurrentIndex(index if index >= 0 else 0)
        self.position_combo.blockSignals(False)

    def _populate_components(
        self,
        preferred: ScalarFieldSelection | None = None,
    ) -> None:
        self.component_list.blockSignals(True)
        self.component_list.clear()
        self._component_selections.clear()
        availabilities = self._matching_fields()
        preferred_selection = next(
            (
                ScalarFieldSelection(availability.key, preferred.component)
                for availability in availabilities
                if preferred is not None
                and availability.key == preferred.field_key
                and preferred.component in availability.descriptor.columns
            ),
            None,
        )
        if preferred_selection is None and availabilities:
            preferred_selection = ScalarFieldSelection(
                availabilities[0].key,
                availabilities[0].descriptor.default_component,
            )
        component_counts: dict[str, int] = {}
        for availability in availabilities:
            for component in availability.descriptor.columns:
                component_counts[component] = (
                    component_counts.get(component, 0) + 1
                )
        for availability in availabilities:
            for component in availability.descriptor.columns:
                selection = ScalarFieldSelection(
                    availability.key,
                    component,
                )
                self._component_selections.append(selection)
                label = component
                if component_counts[component] > 1:
                    label = (
                        f"{component} · contract "
                        f"{availability.key.recovery_contract}"
                    )
                item = QListWidgetItem(label, self.component_list)
                item.setFlags(
                    (
                        item.flags()
                        | Qt.ItemFlag.ItemIsUserCheckable
                        | Qt.ItemFlag.ItemIsEnabled
                    )
                    & ~Qt.ItemFlag.ItemIsSelectable
                )
                item.setCheckState(
                    Qt.CheckState.Checked
                    if selection == preferred_selection
                    else Qt.CheckState.Unchecked
                )
        self.component_list.blockSignals(False)
        self._refresh_export_enabled()

    def _matching_fields(self) -> tuple[FieldAvailability, ...]:
        variable = self.variable_combo.currentData()
        field_id = self.position_combo.currentData()
        return tuple(
            availability
            for availability in self._fields
            if (
                availability.descriptor.field_id.variable is variable
                and availability.descriptor.field_id == field_id
            )
        )

    def _variable_changed(self, _index: int) -> None:
        self._populate_positions()
        self._populate_components()

    def _position_changed(self, _index: int) -> None:
        self._populate_components()

    def _component_item_changed(self, item: QListWidgetItem) -> None:
        if item.checkState() is Qt.CheckState.Checked:
            selection = self._component_selections[
                self.component_list.row(item)
            ]
            self.component_list.blockSignals(True)
            for index, candidate in enumerate(self._component_selections):
                if candidate.field_key != selection.field_key:
                    self.component_list.item(index).setCheckState(
                        Qt.CheckState.Unchecked
                    )
            self.component_list.blockSignals(False)
        self._refresh_export_enabled()

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
        self.export_button.setEnabled(
            bool(self.path_edit.text().strip())
            and bool(self.current_selections())
        )
