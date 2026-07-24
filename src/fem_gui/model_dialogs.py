"""Modal model-definition dialogs following the existing GUI style."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from fem.core.model import MaterialDefinition

from .dialogs import CompactDoubleSpinBox, configure_form_layout
from .document import RegionAssignment, SectionDefinition


def _number(parent: QDialog, value: float, *, minimum: float = 0.0) -> QDoubleSpinBox:
    box = CompactDoubleSpinBox(parent)
    box.setRange(minimum, 1.0e15)
    box.setDecimals(8)
    box.setValue(float(value))
    return box


class ElasticBehaviorDialog(QDialog):
    """Parameters for the currently supported isotropic linear elasticity."""

    def __init__(self, properties: dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("线弹性")
        self.elastic_spin = _number(
            self,
            float(properties.get("E", 210000.0)),
            minimum=1.0e-12,
        )
        self.poisson_spin = _number(
            self,
            float(properties.get("nu", 0.3)),
            minimum=-0.999999,
        )
        self.poisson_spin.setMaximum(0.499999)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("弹性模量 E", self.elastic_spin)
        form.addRow("泊松比 ν", self.poisson_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setMinimumWidth(330)

    def values(self) -> dict[str, float]:
        return {
            "E": self.elastic_spin.value(),
            "nu": self.poisson_spin.value(),
        }


class DensityBehaviorDialog(QDialog):
    def __init__(self, properties: dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("密度")
        self.density_spin = _number(
            self,
            float(properties.get("rho", 7850.0)),
            minimum=1.0e-12,
        )
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("密度 ρ", self.density_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setMinimumWidth(330)

    def value(self) -> float:
        return self.density_spin.value()


class MaterialEditDialog(QDialog):
    def __init__(self, material: MaterialDefinition | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑材料" if material else "新建材料")
        current = material or MaterialDefinition("Material-1", {"E": 210000.0, "nu": 0.3})
        self._properties = dict(current.properties)
        self._row_kinds: list[str] = []
        self.name_edit = QLineEdit(current.name, self)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("名称", self.name_edit)
        self.behavior_table = QTableWidget(0, 2, self)
        self.behavior_table.setHorizontalHeaderLabels(("材料行为", "状态"))
        self.behavior_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.behavior_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.behavior_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.behavior_table.setAlternatingRowColors(True)
        self.behavior_table.verticalHeader().setVisible(False)
        header = self.behavior_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.behavior_combo = QComboBox(self)
        self.behavior_combo.addItem("线弹性", "elastic")
        self.behavior_combo.addItem("密度", "density")
        self.add_behavior_button = QPushButton("添加", self)
        self.edit_behavior_button = QPushButton("编辑参数", self)
        self.delete_behavior_button = QPushButton("删除", self)
        self.add_behavior_button.clicked.connect(self._add_behavior)
        self.edit_behavior_button.clicked.connect(self._edit_behavior)
        self.delete_behavior_button.clicked.connect(self._delete_behavior)
        self.behavior_table.itemDoubleClicked.connect(
            lambda _item: self._edit_behavior()
        )
        self.behavior_table.itemSelectionChanged.connect(self._update_buttons)
        self.behavior_combo.currentIndexChanged.connect(self._update_buttons)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("添加行为", self))
        controls.addWidget(self.behavior_combo)
        controls.addWidget(self.add_behavior_button)
        controls.addSpacing(12)
        controls.addWidget(self.edit_behavior_button)
        controls.addWidget(self.delete_behavior_button)
        controls.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.behavior_table)
        layout.addLayout(controls)
        hint = QLabel("双击材料行为，或选中后点击“编辑参数”。", self)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        self.resize(420, 300)
        self._refresh_behaviors()

    def material(self) -> MaterialDefinition:
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("材料名称不能为空")
        return MaterialDefinition(name, dict(self._properties))

    def _behavior_rows(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        if "E" in self._properties or "nu" in self._properties:
            rows.append((
                "elastic",
                "线弹性",
                "已定义",
            ))
        if "rho" in self._properties:
            rows.append((
                "density",
                "密度",
                "已定义",
            ))
        known = {"E", "nu", "rho"}
        unknown = tuple(
            key for key in self._properties if key not in known
        )
        if unknown:
            rows.append((
                "preserved",
                "其他属性（来自 INP）",
                f"已保留 {len(unknown)} 项",
            ))
        return rows

    def _refresh_behaviors(self) -> None:
        selected = self.behavior_table.currentRow()
        rows = self._behavior_rows()
        self._row_kinds = [row[0] for row in rows]
        self.behavior_table.setRowCount(len(rows))
        for row, (_kind, behavior, summary) in enumerate(rows):
            self.behavior_table.setItem(row, 0, QTableWidgetItem(behavior))
            self.behavior_table.setItem(row, 1, QTableWidgetItem(summary))
        if rows:
            self.behavior_table.selectRow(
                max(0, min(selected, len(rows) - 1))
            )
        self._update_buttons()

    def _add_behavior(self) -> None:
        self._edit_kind(str(self.behavior_combo.currentData()))

    def _edit_behavior(self) -> None:
        row = self.behavior_table.currentRow()
        if 0 <= row < len(self._row_kinds):
            self._edit_kind(self._row_kinds[row])

    def _edit_kind(self, kind: str) -> None:
        if kind == "elastic":
            dialog = ElasticBehaviorDialog(self._properties, self)
            if not dialog.exec():
                return
            self._properties.update(dialog.values())
        elif kind == "density":
            dialog = DensityBehaviorDialog(self._properties, self)
            if not dialog.exec():
                return
            self._properties["rho"] = dialog.value()
        else:
            return
        self._refresh_behaviors()

    def _delete_behavior(self) -> None:
        row = self.behavior_table.currentRow()
        if not 0 <= row < len(self._row_kinds):
            return
        kind = self._row_kinds[row]
        if kind == "elastic":
            self._properties.pop("E", None)
            self._properties.pop("nu", None)
        elif kind == "density":
            self._properties.pop("rho", None)
        else:
            return
        self._refresh_behaviors()

    def _update_buttons(self) -> None:
        existing = set(self._row_kinds)
        selected = self.behavior_table.currentRow()
        selected_kind = (
            self._row_kinds[selected]
            if 0 <= selected < len(self._row_kinds)
            else None
        )
        self.add_behavior_button.setEnabled(
            str(self.behavior_combo.currentData()) not in existing
        )
        editable = selected_kind in {"elastic", "density"}
        self.edit_behavior_button.setEnabled(editable)
        self.delete_behavior_button.setEnabled(editable)


class MaterialManagerDialog(QDialog):
    def __init__(self, materials: list[MaterialDefinition], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("材料管理")
        self.materials = deepcopy(materials)
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(("名称", "材料行为"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.add_button = QPushButton("新建", self)
        self.edit_button = QPushButton("编辑", self)
        self.delete_button = QPushButton("删除", self)
        self.add_button.clicked.connect(self._add)
        self.edit_button.clicked.connect(self._edit)
        self.delete_button.clicked.connect(self._delete)
        self.table.itemDoubleClicked.connect(lambda _item: self._edit())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        controls = QHBoxLayout()
        for button in (self.add_button, self.edit_button, self.delete_button):
            controls.addWidget(button)
        controls.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(controls)
        layout.addWidget(buttons)
        self.resize(520, 330)
        self._refresh()

    def _refresh(self) -> None:
        self.table.setRowCount(len(self.materials))
        for row, material in enumerate(self.materials):
            values = (
                material.name,
                self._behavior_summary(material),
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        if self.materials:
            row = max(0, min(self.table.currentRow(), len(self.materials) - 1))
            self.table.selectRow(row)
        self._update_buttons()

    @staticmethod
    def _behavior_summary(material: MaterialDefinition) -> str:
        properties = material.properties
        behaviors: list[str] = []
        if "E" in properties or "nu" in properties:
            behaviors.append("线弹性")
        if "rho" in properties:
            behaviors.append("密度")
        if any(key not in {"E", "nu", "rho"} for key in properties):
            behaviors.append("其他属性")
        return "、".join(behaviors) or "未定义"

    def _selected_row(self) -> int:
        return self.table.currentRow()

    def _store(self, value: MaterialDefinition, row: int | None = None) -> None:
        duplicate = next((index for index, item in enumerate(self.materials) if item.name == value.name and index != row), None)
        if duplicate is not None:
            raise ValueError(f"材料名称已存在：{value.name}")
        if row is None:
            self.materials.append(value)
        else:
            self.materials[row] = value
        self._refresh()

    def _add(self) -> None:
        dialog = MaterialEditDialog(parent=self)
        if dialog.exec():
            try:
                self._store(dialog.material())
            except ValueError as error:
                QMessageBox.warning(self, "材料", str(error))

    def _edit(self) -> None:
        row = self._selected_row()
        if row < 0:
            return
        dialog = MaterialEditDialog(self.materials[row], self)
        if dialog.exec():
            try:
                self._store(dialog.material(), row)
            except ValueError as error:
                QMessageBox.warning(self, "材料", str(error))

    def _delete(self) -> None:
        row = self._selected_row()
        if row >= 0:
            del self.materials[row]
            self._refresh()

    def _update_buttons(self) -> None:
        selected = self._selected_row() >= 0
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def values(self) -> list[MaterialDefinition]:
        return deepcopy(self.materials)


class SectionEditDialog(QDialog):
    def __init__(
        self,
        materials: list[MaterialDefinition],
        section: SectionDefinition | None = None,
        parent=None,
        *,
        model_dimension: int = 2,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑截面" if section else "新建截面")
        self.model_dimension = int(model_dimension)
        current = section or SectionDefinition(
            "Section-1",
            materials[0].name if materials else "",
            "solid",
            {"plane_type": "stress", "thickness": 1.0}
            if self.model_dimension == 2
            else {},
        )
        self._properties = dict(current.properties)
        self._section_type = current.section_type
        self._preserve_section_type = (
            self.model_dimension == 1
            or current.section_type != "solid"
        )
        self.name_edit = QLineEdit(current.name, self)
        self.material_combo = QComboBox(self)
        for material in materials:
            self.material_combo.addItem(material.name)
        self.material_combo.setCurrentText(current.material)
        self.type_combo = QComboBox(self)
        if self._preserve_section_type:
            self.type_combo.addItem(
                f"{self._section_type}（来自 INP）",
                self._section_type,
            )
            self.type_combo.setEnabled(False)
        elif self.model_dimension == 2:
            self.type_combo.addItem("平面应力", "stress")
            self.type_combo.addItem("平面应变", "strain")
            self.type_combo.setCurrentIndex(
                max(
                    0,
                    self.type_combo.findData(
                        str(current.properties.get("plane_type", "stress")).casefold()
                    ),
                )
            )
        elif self.model_dimension == 3:
            self.type_combo.addItem("三维实体", "solid")
        self.thickness_spin = _number(self, current.properties.get("thickness", 1.0), minimum=1.0e-12)
        self.form = QFormLayout()
        configure_form_layout(self.form)
        self.form.addRow("名称", self.name_edit)
        self.form.addRow("材料", self.material_combo)
        self.form.addRow("类型", self.type_combo)
        self.form.addRow("厚度", self.thickness_spin)
        self.form.setRowVisible(
            self.thickness_spin,
            self.model_dimension == 2
            and not self._preserve_section_type,
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(self.form)
        layout.addWidget(buttons)
        self.setMinimumWidth(340)

    def section(self) -> SectionDefinition:
        name = self.name_edit.text().strip()
        material = self.material_combo.currentText().strip()
        if not name or not material:
            raise ValueError("截面名称和材料不能为空")
        properties = dict(self._properties)
        if self._preserve_section_type:
            section_type = self._section_type
        elif self.model_dimension == 2:
            properties["plane_type"] = str(self.type_combo.currentData())
            properties["thickness"] = self.thickness_spin.value()
            section_type = "solid"
        elif self.model_dimension == 3:
            properties.pop("plane_type", None)
            properties.pop("thickness", None)
            section_type = "solid"
        return SectionDefinition(name, material, section_type, properties)


class SectionManagerDialog(QDialog):
    def __init__(
        self,
        materials: list[MaterialDefinition],
        sections: list[SectionDefinition],
        parent=None,
        *,
        model_dimension: int = 2,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("截面管理")
        self.materials, self.sections = deepcopy(materials), deepcopy(sections)
        self.model_dimension = int(model_dimension)
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(("名称", "材料", "类型"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.add_button, self.edit_button, self.delete_button = (QPushButton(text, self) for text in ("新建", "编辑", "删除"))
        self.add_button.clicked.connect(self._add)
        self.edit_button.clicked.connect(self._edit)
        self.delete_button.clicked.connect(self._delete)
        self.table.itemDoubleClicked.connect(lambda _item: self._edit())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        controls = QHBoxLayout()
        for button in (self.add_button, self.edit_button, self.delete_button): controls.addWidget(button)
        controls.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self); layout.addWidget(self.table); layout.addLayout(controls); layout.addWidget(buttons)
        if self.model_dimension == 1:
            self.add_button.setEnabled(False)
            self.add_button.setToolTip(
                "当前界面尚未提供梁/杆截面参数创建；可保留并编辑 INP 中已有截面。"
            )
        self.resize(500, 320); self._refresh()

    def _refresh(self) -> None:
        self.table.setRowCount(len(self.sections))
        for row, section in enumerate(self.sections):
            for column, value in enumerate(
                (
                    section.name,
                    section.material,
                    self._section_label(section),
                )
            ):
                self.table.setItem(row, column, QTableWidgetItem(value))
        if self.sections:
            row = max(0, min(self.table.currentRow(), len(self.sections) - 1))
            self.table.selectRow(row)
        self._update_buttons()

    def _section_label(self, section: SectionDefinition) -> str:
        if self.model_dimension == 2 and section.section_type == "solid":
            return {
                "stress": "平面应力",
                "strain": "平面应变",
            }.get(
                str(section.properties.get("plane_type", "stress")).casefold(),
                "二维实体",
            )
        if self.model_dimension == 3 and section.section_type == "solid":
            return "三维实体"
        return section.section_type

    def _store(self, value: SectionDefinition, row: int | None = None) -> None:
        if any(item.name == value.name and index != row for index, item in enumerate(self.sections)):
            raise ValueError(f"截面名称已存在：{value.name}")
        if row is None: self.sections.append(value)
        else: self.sections[row] = value
        self._refresh()

    def _add(self) -> None:
        if not self.materials:
            QMessageBox.warning(self, "截面", "请先创建材料")
            return
        dialog = SectionEditDialog(
            self.materials,
            parent=self,
            model_dimension=self.model_dimension,
        )
        if dialog.exec():
            try: self._store(dialog.section())
            except ValueError as error: QMessageBox.warning(self, "截面", str(error))

    def _edit(self) -> None:
        row = self.table.currentRow()
        if row < 0: return
        dialog = SectionEditDialog(
            self.materials,
            self.sections[row],
            self,
            model_dimension=self.model_dimension,
        )
        if dialog.exec():
            try: self._store(dialog.section(), row)
            except ValueError as error: QMessageBox.warning(self, "截面", str(error))

    def _delete(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            del self.sections[row]; self._refresh()

    def _update_buttons(self) -> None:
        selected = self.table.currentRow() >= 0
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def values(self) -> list[SectionDefinition]:
        return deepcopy(self.sections)


class RegionAssignmentDialog(QDialog):
    def __init__(self, sections: list[SectionDefinition], regions: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("截面分配")
        self.section_combo, self.region_combo = QComboBox(self), QComboBox(self)
        self.section_combo.addItems([section.name for section in sections])
        self.region_combo.addItems(regions)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("截面", self.section_combo)
        form.addRow("单元区域", self.region_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定"); buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(buttons)
        self.setMinimumWidth(300)

    def assignment(self) -> RegionAssignment:
        return RegionAssignment(self.section_combo.currentText(), self.region_combo.currentText())
