"""Modal model-definition dialogs following the existing GUI style."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

from fem.application import RegionRef, require_region_kind
from fem.core.model import MaterialDefinition
from fem.materials import (
    resolve_section_preset_properties,
    section_type_for_preset,
)

from .dialogs import CompactDoubleSpinBox, configure_form_layout
from .document import RegionAssignment, SectionDefinition


def _number(parent: QDialog, value: float, *, minimum: float = 0.0) -> QDoubleSpinBox:
    box = CompactDoubleSpinBox(parent)
    box.setRange(minimum, 1.0e15)
    box.setDecimals(8)
    box.setValue(float(value))
    return box


_SECTION_PRESET_LABELS = {
    "solid_plane_stress": "平面应力",
    "solid_plane_strain": "平面应变",
    "solid": "三维实体",
    "truss": "桁架（面积）",
    "rectangle": "梁（矩形）",
    "solid_circle": "梁（实心圆）",
    "hollow_circle": "梁（空心圆）",
}
_SECTION_PROPERTY_FIELDS = frozenset(
    {
        "section_type",
        "plane_type",
        "thickness",
        "area",
        "height",
        "width",
        "radius",
        "outer_radius",
        "inner_radius",
        "I11",
        "I22",
        "Iyy",
        "Izz",
        "J",
    }
)
_REGION_KIND_LABELS = {
    "node_set": "节点集",
    "element_set": "单元集",
    "edge": "边",
    "surface": "面",
}


def _section_presets(
    values: Sequence[str] | None,
    model_dimension: int,
) -> tuple[str, ...]:
    if values is None:
        if model_dimension == 2:
            return ("solid_plane_stress", "solid_plane_strain")
        if model_dimension == 3:
            return ("solid",)
        return ()
    normalized: list[str] = []
    for value in values:
        preset = str(value).strip().casefold()
        if preset in _SECTION_PRESET_LABELS and preset not in normalized:
            normalized.append(preset)
    return tuple(normalized)


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
            minimum=0.0,
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
        section_presets: Sequence[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑截面" if section else "新建截面")
        self.model_dimension = int(model_dimension)
        self._legacy_dimension_mode = section_presets is None
        self.section_presets = _section_presets(
            section_presets,
            self.model_dimension,
        )
        self._materials = tuple(deepcopy(materials))
        self._original_section = deepcopy(section)
        default_preset = (
            self.section_presets[0]
            if self.section_presets
            else "solid"
        )
        current = section or SectionDefinition(
            "Section-1",
            materials[0].name if materials else "",
            section_type_for_preset(default_preset),
            self._default_properties(default_preset),
        )
        self._properties = dict(current.properties)
        self._section_type = current.section_type
        current_preset = self._preset_for_section(current)
        self._read_only = (
            section is not None
            and (
                current_preset is None
                or current_preset not in self.section_presets
            )
        )
        self._unsupported_new_section = section is None and not self.section_presets
        self._preserve_section_type = (
            self._read_only or self._unsupported_new_section
        )
        self.name_edit = QLineEdit(current.name, self)
        self.material_combo = QComboBox(self)
        for material in materials:
            self.material_combo.addItem(material.name)
        if (
            current.material
            and self.material_combo.findText(current.material) < 0
        ):
            self.material_combo.addItem(current.material)
        self.material_combo.setCurrentText(current.material)
        self.type_combo = QComboBox(self)
        if self._read_only:
            self.type_combo.addItem(
                f"{self._section_type}（来自 INP）",
                self._section_type,
            )
            self.type_combo.setEnabled(False)
        elif self._unsupported_new_section:
            self.type_combo.addItem("当前模型不支持新建截面")
            self.type_combo.setEnabled(False)
        else:
            for preset in self.section_presets:
                combo_data = self._combo_data_for_preset(preset)
                self.type_combo.addItem(
                    _SECTION_PRESET_LABELS[preset],
                    combo_data,
                )
            selected_data = self._combo_data_for_preset(
                current_preset or self.section_presets[0]
            )
            self.type_combo.setCurrentIndex(
                max(0, self.type_combo.findData(selected_data))
            )
        self.thickness_spin = _number(
            self,
            current.properties.get("thickness", 1.0),
            minimum=1.0e-12,
        )
        self.area_spin = _number(
            self,
            current.properties.get("area", 1.0),
            minimum=1.0e-12,
        )
        self.height_spin = _number(
            self,
            current.properties.get("height", 1.0),
            minimum=1.0e-12,
        )
        self.width_spin = _number(
            self,
            current.properties.get("width", 1.0),
            minimum=1.0e-12,
        )
        self.radius_spin = _number(
            self,
            current.properties.get("radius", 1.0),
            minimum=1.0e-12,
        )
        self.outer_radius_spin = _number(
            self,
            current.properties.get("outer_radius", 1.0),
            minimum=1.0e-12,
        )
        self.inner_radius_spin = _number(
            self,
            current.properties.get("inner_radius", 0.5),
            minimum=1.0e-12,
        )
        self.limitation_label = QLabel(self)
        self.limitation_label.setWordWrap(True)
        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)
        self.form = QFormLayout()
        configure_form_layout(self.form)
        self.form.addRow("名称", self.name_edit)
        self.form.addRow("材料", self.material_combo)
        self.form.addRow("类型", self.type_combo)
        self.form.addRow("厚度", self.thickness_spin)
        self.form.addRow("面积", self.area_spin)
        self.form.addRow("高度", self.height_spin)
        self.form.addRow("宽度", self.width_spin)
        self.form.addRow("半径", self.radius_spin)
        self.form.addRow("外半径", self.outer_radius_spin)
        self.form.addRow("内半径", self.inner_radius_spin)
        self.form.addRow("限制", self.limitation_label)
        self.form.addRow("", self.validation_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(self.form)
        layout.addWidget(self.buttons)
        self.type_combo.currentIndexChanged.connect(
            self._update_section_fields
        )
        self.outer_radius_spin.valueChanged.connect(
            self._update_validation_hint
        )
        self.inner_radius_spin.valueChanged.connect(
            self._update_validation_hint
        )
        self.setMinimumWidth(340)
        self._update_section_fields()

    def section(self) -> SectionDefinition:
        if self._read_only and self._original_section is not None:
            return deepcopy(self._original_section)
        if self._unsupported_new_section:
            raise ValueError("当前模型能力不支持新建截面")
        name = self.name_edit.text().strip()
        material = self.material_combo.currentText().strip()
        if not name or not material:
            raise ValueError("截面名称和材料不能为空")
        preset = self._current_preset()
        properties = {
            key: deepcopy(value)
            for key, value in self._properties.items()
            if key not in _SECTION_PROPERTY_FIELDS
        }
        if preset == "solid_plane_stress":
            properties["plane_type"] = "stress"
            properties["thickness"] = self.thickness_spin.value()
        elif preset == "solid_plane_strain":
            properties["plane_type"] = "strain"
            properties["thickness"] = self.thickness_spin.value()
        elif preset == "truss":
            properties["area"] = self.area_spin.value()
        elif preset == "rectangle":
            properties["height"] = self.height_spin.value()
            properties["width"] = self.width_spin.value()
        elif preset == "solid_circle":
            properties["radius"] = self.radius_spin.value()
        elif preset == "hollow_circle":
            properties["outer_radius"] = self.outer_radius_spin.value()
            properties["inner_radius"] = self.inner_radius_spin.value()
        selected_material = next(
            (
                item
                for item in self._materials
                if item.name == material
            ),
            None,
        )
        if selected_material is None:
            raise ValueError(f"截面引用的材料不存在：{material}")
        resolved = resolve_section_preset_properties(
            preset,
            selected_material.properties,
            properties,
        )
        return SectionDefinition(
            name,
            material,
            resolved.section_type,
            properties,
        )

    @staticmethod
    def _default_properties(preset: str) -> dict[str, object]:
        if preset == "solid_plane_stress":
            return {"plane_type": "stress", "thickness": 1.0}
        if preset == "solid_plane_strain":
            return {"plane_type": "strain", "thickness": 1.0}
        if preset == "truss":
            return {"area": 1.0}
        if preset == "rectangle":
            return {"height": 1.0, "width": 1.0}
        if preset == "solid_circle":
            return {"radius": 1.0}
        if preset == "hollow_circle":
            return {"outer_radius": 1.0, "inner_radius": 0.5}
        return {}

    def _combo_data_for_preset(self, preset: str) -> str:
        if not self._legacy_dimension_mode:
            return preset
        return {
            "solid_plane_stress": "stress",
            "solid_plane_strain": "strain",
        }.get(preset, preset)

    def _current_preset(self) -> str:
        value = str(self.type_combo.currentData() or "").casefold()
        return {
            "stress": "solid_plane_stress",
            "strain": "solid_plane_strain",
        }.get(value, value)

    def _preset_for_section(
        self,
        section: SectionDefinition,
    ) -> str | None:
        section_type = str(section.section_type).strip().casefold()
        if section_type == "solid":
            plane_type = str(
                section.properties.get("plane_type", "")
            ).strip().casefold()
            if plane_type == "stress":
                return "solid_plane_stress"
            if plane_type == "strain":
                return "solid_plane_strain"
            if self.model_dimension == 2:
                return "solid_plane_stress"
            return "solid"
        if section_type in {
            "truss",
            "rectangle",
            "solid_circle",
            "hollow_circle",
        }:
            return section_type
        return None

    def _update_section_fields(self) -> None:
        preset = (
            self._current_preset()
            if not self._preserve_section_type
            else ""
        )
        visibility = {
            self.thickness_spin: preset.startswith("solid_plane_"),
            self.area_spin: preset == "truss",
            self.height_spin: preset == "rectangle",
            self.width_spin: preset == "rectangle",
            self.radius_spin: preset == "solid_circle",
            self.outer_radius_spin: preset == "hollow_circle",
            self.inner_radius_spin: preset == "hollow_circle",
        }
        for widget, visible in visibility.items():
            self.form.setRowVisible(widget, visible)
        if self._read_only:
            self.name_edit.setEnabled(False)
            self.material_combo.setEnabled(False)
            self.limitation_label.setText(
                "该导入截面类型暂不支持编辑，保存时将原样保留。"
            )
            self.form.setRowVisible(self.limitation_label, True)
        elif self._unsupported_new_section:
            self.limitation_label.setText(
                "当前模型能力未提供可创建的截面预设。"
            )
            self.form.setRowVisible(self.limitation_label, True)
        elif preset == "rectangle":
            self.limitation_label.setText(
                "beam.orientation.assumed：局部坐标由内核根据单元几何"
                "自动确定；当前版本不提供方向编辑。"
            )
            self.form.setRowVisible(self.limitation_label, True)
        else:
            self.limitation_label.clear()
            self.form.setRowVisible(self.limitation_label, False)
        self._update_validation_hint()

    def _update_validation_hint(self) -> None:
        invalid_hollow = (
            not self._preserve_section_type
            and self._current_preset() == "hollow_circle"
            and self.outer_radius_spin.value()
            <= self.inner_radius_spin.value()
        )
        if invalid_hollow:
            self.validation_label.setText("外半径必须大于内半径。")
        else:
            self.validation_label.clear()
        self.form.setRowVisible(self.validation_label, invalid_hollow)
        self.buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setEnabled(not invalid_hollow and not self._unsupported_new_section)


class SectionManagerDialog(QDialog):
    def __init__(
        self,
        materials: list[MaterialDefinition],
        sections: list[SectionDefinition],
        parent=None,
        *,
        model_dimension: int = 2,
        section_presets: Sequence[str] | None = None,
        authoring_enabled: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("截面管理")
        self.materials, self.sections = deepcopy(materials), deepcopy(sections)
        self.model_dimension = int(model_dimension)
        self._section_presets_arg = (
            None
            if section_presets is None
            else tuple(section_presets)
        )
        self.section_presets = _section_presets(
            section_presets,
            self.model_dimension,
        )
        self._can_create = bool(
            authoring_enabled
            and self.materials
            and self.section_presets
        )
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
        for button in (
            self.add_button,
            self.edit_button,
            self.delete_button,
        ):
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
        self.add_button.setEnabled(self._can_create)
        if not authoring_enabled:
            self.add_button.setToolTip("当前模型策略不允许新建截面。")
        elif not self.materials:
            self.add_button.setToolTip("请先创建材料。")
        elif not self.section_presets:
            self.add_button.setToolTip("当前模型能力没有可用的截面预设。")
        self.resize(500, 320)
        self._refresh()

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
        return _SECTION_PRESET_LABELS.get(
            section.section_type,
            section.section_type,
        )

    def _store(self, value: SectionDefinition, row: int | None = None) -> None:
        if any(item.name == value.name and index != row for index, item in enumerate(self.sections)):
            raise ValueError(f"截面名称已存在：{value.name}")
        if row is None:
            self.sections.append(value)
        else:
            self.sections[row] = value
        self._refresh()

    def _add(self) -> None:
        if not self.materials:
            QMessageBox.warning(self, "截面", "请先创建材料")
            return
        if not self._can_create:
            return
        dialog = SectionEditDialog(
            self.materials,
            parent=self,
            model_dimension=self.model_dimension,
            section_presets=self._section_presets_arg,
        )
        if dialog.exec():
            try:
                self._store(dialog.section())
            except ValueError as error:
                QMessageBox.warning(self, "截面", str(error))

    def _edit(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        dialog = SectionEditDialog(
            self.materials,
            self.sections[row],
            self,
            model_dimension=self.model_dimension,
            section_presets=self._section_presets_arg,
        )
        if dialog.exec():
            try:
                self._store(dialog.section(), row)
            except ValueError as error:
                QMessageBox.warning(self, "截面", str(error))

    def _delete(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            del self.sections[row]
            self._refresh()

    def _update_buttons(self) -> None:
        selected = self.table.currentRow() >= 0
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def values(self) -> list[SectionDefinition]:
        return deepcopy(self.sections)


class RegionAssignmentDialog(QDialog):
    def __init__(
        self,
        sections: Sequence[SectionDefinition],
        regions: Sequence[RegionRef | str] = (),
        parent=None,
        *,
        compatible_targets: Mapping[
            str,
            Sequence[RegionRef | str],
        ]
        | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("截面分配")
        self._sections = tuple(deepcopy(sections))
        self._regions = self._normalize_regions(regions)
        self._compatible_targets = (
            None
            if compatible_targets is None
            else {
                str(key): self._normalize_regions(values)
                for key, values in compatible_targets.items()
            }
        )
        self.section_combo, self.region_combo = QComboBox(self), QComboBox(self)
        self.section_combo.addItems(
            [section.name for section in self._sections]
        )
        self.section_combo.currentIndexChanged.connect(
            self._refresh_regions
        )
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("截面", self.section_combo)
        form.addRow("单元区域", self.region_combo)
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
        self.setMinimumWidth(300)
        self._refresh_regions()

    def assignment(self) -> RegionAssignment:
        section_name = self.section_combo.currentText().strip()
        if not section_name:
            raise ValueError("截面不能为空")
        region = self.region_combo.currentData()
        if not isinstance(region, RegionRef):
            raise ValueError("没有可分配的兼容单元区域")
        region_name = require_region_kind(region, "element_set")
        return RegionAssignment(section_name, region_name)

    @staticmethod
    def _normalize_regions(
        values: Sequence[RegionRef | str],
    ) -> tuple[RegionRef, ...]:
        normalized: list[RegionRef] = []
        for value in values:
            reference = (
                value
                if isinstance(value, RegionRef)
                else RegionRef("element_set", str(value))
            )
            if reference not in normalized:
                normalized.append(reference)
        return tuple(normalized)

    def _refresh_regions(self) -> None:
        section_index = self.section_combo.currentIndex()
        if section_index < 0 or section_index >= len(self._sections):
            targets: tuple[RegionRef, ...] = ()
        elif self._compatible_targets is None:
            targets = self._regions
        else:
            section = self._sections[section_index]
            targets = self._compatible_targets.get(
                section.name,
                self._compatible_targets.get(section.section_type, ()),
            )
        self.region_combo.clear()
        name_counts = {
            region.name: sum(
                candidate.name == region.name for candidate in targets
            )
            for region in targets
        }
        for region in targets:
            label = region.name
            if name_counts[region.name] > 1 or region.kind != "element_set":
                kind_label = _REGION_KIND_LABELS.get(
                    region.kind,
                    region.kind,
                )
                label = f"{region.name}（{kind_label}）"
            self.region_combo.addItem(label, region)
