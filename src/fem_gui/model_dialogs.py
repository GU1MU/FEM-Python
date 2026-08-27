"""Modal model-definition dialogs following the existing GUI style."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from math import isfinite

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fem.application import (
    AuthoringCapability,
    BeamOrientation,
    DeleteIntent,
    RegionAssignment,
    RegionRef,
    RenameIntent,
    SectionDefinition,
    require_region_kind,
)
from fem.model import MaterialBehavior, MaterialDefinition
from fem.analysis import (
    resolve_section_preset_properties,
    section_type_for_preset,
)
from fem.materials import (
    DENSITY_BEHAVIOR_ID,
    ELASTIC_BEHAVIOR_ID,
    NEO_HOOKEAN_BEHAVIOR_ID,
    PLASTIC_BEHAVIOR_ID,
    behavior_summary,
    material_behavior_diagnostics,
    material_behavior_spec,
    material_behavior_specs,
)

from .dialogs import AdaptivePrecisionDoubleSpinBox, configure_form_layout


def _number(parent: QDialog, value: float, *, minimum: float = 0.0) -> QDoubleSpinBox:
    box = AdaptivePrecisionDoubleSpinBox(parent)
    box.setRange(minimum, 1.0e15)
    box.setValue(float(value))
    return box


def _signed_number(parent: QDialog, value: float = 0.0) -> QDoubleSpinBox:
    box = AdaptivePrecisionDoubleSpinBox(parent)
    box.setRange(-1.0e15, 1.0e15)
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
    values: Sequence[str],
    model_dimension: int,
) -> tuple[str, ...]:
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
        has_elastic_modulus = "E" in properties
        self.elastic_spin = _number(
            self,
            float(properties["E"]) if has_elastic_modulus else 1.0e-12,
            minimum=1.0e-12,
        )
        if not has_elastic_modulus:
            self.elastic_spin.clear()
        has_poisson_ratio = "nu" in properties
        self.poisson_spin = _number(
            self,
            float(properties["nu"]) if has_poisson_ratio else -0.999999,
            minimum=-0.999999,
        )
        if not has_poisson_ratio:
            self.poisson_spin.clear()
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
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.elastic_spin.textChanged.connect(self._update_ok_button)
        self.poisson_spin.textChanged.connect(self._update_ok_button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setMinimumWidth(330)
        self._update_ok_button()

    def _update_ok_button(self) -> None:
        self.ok_button.setEnabled(
            bool(self.elastic_spin.text().strip())
            and bool(self.poisson_spin.text().strip())
        )

    def values(self) -> dict[str, float]:
        if not self.ok_button.isEnabled():
            raise ValueError("弹性模量和泊松比不能为空")
        return {
            "E": self.elastic_spin.value(),
            "nu": self.poisson_spin.value(),
        }


class PlasticBehaviorDialog(QDialog):
    """Parameters for the public plastic material behavior.

    The current constitutive implementation is J2 plasticity, but that is an
    implementation detail rather than a user-facing material choice.
    """

    def __init__(self, properties: dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("塑性")
        has_yield_stress = "yield_stress" in properties
        self.yield_stress_spin = _number(
            self,
            float(properties["yield_stress"])
            if has_yield_stress
            else 1.0,
            minimum=0.0,
        )
        if not has_yield_stress:
            self.yield_stress_spin.clear()
        self.hardening_spin = _number(
            self,
            float(properties.get("hardening_modulus", 0.0)),
            minimum=0.0,
        )
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("屈服应力 σᵧ", self.yield_stress_spin)
        form.addRow("各向同性硬化模量 H", self.hardening_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.yield_stress_spin.textChanged.connect(self._update_ok_button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setMinimumWidth(360)
        self._update_ok_button()

    def _update_ok_button(self) -> None:
        self.ok_button.setEnabled(bool(self.yield_stress_spin.text().strip()))

    def values(self) -> dict[str, float]:
        if not self.ok_button.isEnabled():
            raise ValueError("屈服应力不能为空")
        return {
            "yield_stress": self.yield_stress_spin.value(),
            "hardening_modulus": self.hardening_spin.value(),
        }


# Compatibility name for callers written before the public behavior was
# renamed to simply "塑性".  The persisted material properties are unchanged.
J2PlasticityBehaviorDialog = PlasticBehaviorDialog


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


class NeoHookeanBehaviorDialog(QDialog):
    """Abaqus-style scalar editor for the first hyperelastic model."""

    def __init__(self, properties: dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("超弹性（Neo Hooke）")
        self.c10_spin = _number(
            self,
            float(properties.get("C10", 1.0)),
            minimum=1.0e-15,
        )
        self.d1_spin = _number(
            self,
            float(properties.get("D1", 1.0)),
            minimum=1.0e-15,
        )
        if "C10" not in properties:
            self.c10_spin.clear()
        if "D1" not in properties:
            self.d1_spin.clear()
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("材料类型", QLabel("各向同性"))
        form.addRow("应变能势", QLabel("Neo-Hookean"))
        form.addRow("C10", self.c10_spin)
        form.addRow("D1", self.d1_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.c10_spin.textChanged.connect(self._update_ok_button)
        self.d1_spin.textChanged.connect(self._update_ok_button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setMinimumWidth(380)
        self._update_ok_button()

    def _update_ok_button(self) -> None:
        self.ok_button.setEnabled(
            bool(self.c10_spin.text().strip())
            and bool(self.d1_spin.text().strip())
        )

    def values(self) -> dict[str, float]:
        if not self.ok_button.isEnabled():
            raise ValueError("C10 和 D1 不能为空")
        return {
            "C10": self.c10_spin.value(),
            "D1": self.d1_spin.value(),
        }


class MaterialEditDialog(QDialog):
    """A compact Abaqus-style material behavior editor."""

    _ALIASES = {
        "elastic": ELASTIC_BEHAVIOR_ID,
        "plastic": PLASTIC_BEHAVIOR_ID,
        "j2": PLASTIC_BEHAVIOR_ID,
        "density": DENSITY_BEHAVIOR_ID,
        "neo_hookean": NEO_HOOKEAN_BEHAVIOR_ID,
    }
    _PARAMETER_KEYS = {
        ELASTIC_BEHAVIOR_ID: ("E", "nu"),
        PLASTIC_BEHAVIOR_ID: ("yield_stress", "hardening_modulus"),
        DENSITY_BEHAVIOR_ID: ("rho",),
        NEO_HOOKEAN_BEHAVIOR_ID: ("C10", "D1"),
    }
    _REQUIRED_PARAMETER_KEYS = {
        ELASTIC_BEHAVIOR_ID: ("E", "nu"),
        # Perfect plasticity is valid; hardening_modulus defaults to zero in
        # the constitutive compiler and therefore remains optional here.
        PLASTIC_BEHAVIOR_ID: ("yield_stress",),
        DENSITY_BEHAVIOR_ID: ("rho",),
        NEO_HOOKEAN_BEHAVIOR_ID: ("C10", "D1"),
    }
    _KNOWN_PROPERTY_KEYS = frozenset(
        key for values in _PARAMETER_KEYS.values() for key in values
    )

    def __init__(self, material: MaterialDefinition | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑材料" if material else "新建材料")
        current = material or MaterialDefinition("Material-1", {})
        self._original_material = current
        self._properties = dict(current.properties)
        self._behaviors: dict[str, dict[str, object]] = {
            item.behavior_id: dict(item.parameters)
            for item in current.behaviors
        }
        self._row_kinds: list[str] = []
        self._active_behavior_id: str | None = None
        self._editor_widgets: dict[str, AdaptivePrecisionDoubleSpinBox] = {}

        self.name_edit = QLineEdit(current.name, self)
        self.description_edit = QLineEdit(current.description, self)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("名称", self.name_edit)
        form.addRow("描述", self.description_edit)

        # Kept as a non-visible data proxy for older callers.  The user-facing
        # editor no longer exposes a second constitutive-model selector.
        self.model_combo = QComboBox(self)
        self.model_combo.addItem("线弹性", "linear_elastic")
        self.model_combo.addItem("J2 塑性", "j2_plasticity")
        self.model_combo.addItem("Neo-Hookean 超弹性", "neo_hookean")
        model_index = self.model_combo.findData(current.constitutive_model)
        if model_index < 0:
            self.model_combo.addItem(
                f"{current.constitutive_model}（当前模型）",
                current.constitutive_model,
            )
            model_index = self.model_combo.findData(current.constitutive_model)
        self.model_combo.setCurrentIndex(model_index)
        self.model_combo.setVisible(False)

        self.menu_bar = QMenuBar(self)
        self._build_behavior_menus()

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
        self.behavior_table.itemDoubleClicked.connect(
            lambda _item: self._edit_behavior()
        )
        self.behavior_table.itemSelectionChanged.connect(
            self._show_selected_behavior
        )

        # Compatibility controls remain available to programmatic callers but
        # are not part of the Abaqus-style visible workflow.
        self.behavior_combo = QComboBox(self)
        self.behavior_combo.addItem("线弹性", "elastic")
        self.behavior_combo.addItem("塑性", "plastic")
        self.behavior_combo.addItem("密度", "density")
        self.behavior_combo.addItem("超弹性（Neo Hooke）", "neo_hookean")
        self.behavior_combo.setVisible(False)
        self.add_behavior_button = QPushButton("添加", self)
        self.edit_behavior_button = QPushButton("编辑参数", self)
        self.delete_behavior_button = QPushButton("删除行为", self)
        self.add_behavior_button.setVisible(False)
        self.add_behavior_button.clicked.connect(self._add_behavior)
        self.edit_behavior_button.clicked.connect(self._edit_behavior)
        self.delete_behavior_button.clicked.connect(self._delete_behavior)

        self.behavior_editor = QWidget(self)
        self.behavior_editor_layout = QVBoxLayout(self.behavior_editor)
        self.behavior_editor_layout.setContentsMargins(12, 8, 12, 8)
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        editor_row = QVBoxLayout()
        editor_row.addWidget(self.behavior_table, 1)
        behavior_actions = QHBoxLayout()
        behavior_actions.addStretch(1)
        behavior_actions.addWidget(self.edit_behavior_button)
        behavior_actions.addWidget(self.delete_behavior_button)
        editor_row.addLayout(behavior_actions)
        editor_row.addWidget(self.behavior_editor, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.menu_bar)
        layout.addLayout(editor_row, 1)
        layout.addWidget(self.status_label)
        layout.addWidget(buttons)
        self.resize(420, 520)
        self._refresh_behaviors()

    def _build_behavior_menus(self) -> None:
        menus = {}
        for spec in material_behavior_specs():
            parent = self.menu_bar
            parent_path = ()
            for part in spec.menu_path[:-1]:
                parent_path = (*parent_path, part)
                menu = menus.get(parent_path)
                if menu is None:
                    menu = parent.addMenu(part)
                    menus[parent_path] = menu
                parent = menu
            action = parent.addAction(spec.menu_path[-1])
            action.triggered.connect(
                lambda _checked=False, behavior_id=spec.behavior_id:
                self._add_behavior_kind(behavior_id)
            )

    @classmethod
    def _canonical_kind(cls, kind: str) -> str:
        normalized = str(kind).strip().casefold()
        return cls._ALIASES.get(normalized, normalized)

    def material(self) -> MaterialDefinition:
        self._sync_editor_to_behavior()
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("材料名称不能为空")
        behaviors = tuple(
            MaterialBehavior(kind, parameters)
            for kind, parameters in self._behaviors.items()
        )
        diagnostics = material_behavior_diagnostics(behaviors)
        blocking_diagnostics = tuple(
            item
            for item in diagnostics
            if not item.startswith("包含未注册的材料行为：")
        )
        if blocking_diagnostics:
            raise ValueError("；".join(blocking_diagnostics))
        incomplete = tuple(
            kind
            for kind, parameters in self._behaviors.items()
            if self._behavior_status(kind, parameters) == "未完成"
        )
        if incomplete:
            labels = tuple(
                material_behavior_spec(kind).label
                for kind in incomplete
                if kind in self._PARAMETER_KEYS
            )
            raise ValueError(
                "材料行为参数未填写完整：" + "、".join(labels)
            )
        if behaviors:
            properties = {
                key: value
                for key, value in self._properties.items()
                if key not in self._KNOWN_PROPERTY_KEYS
            }
            for behavior in behaviors:
                properties.update(dict(behavior.parameters))
        else:
            # An unknown/imported model has no registered behavior editor yet.
            # Preserve every property exactly instead of assuming that E/nu or
            # another familiar key belongs to the new material schema.
            properties = dict(self._properties)
        model = self._original_material.constitutive_model
        ids = {item.behavior_id for item in behaviors}
        if NEO_HOOKEAN_BEHAVIOR_ID in ids:
            model = "neo_hookean"
        elif PLASTIC_BEHAVIOR_ID in ids:
            model = "j2_plasticity"
        elif ELASTIC_BEHAVIOR_ID in ids:
            model = "linear_elastic"
        return MaterialDefinition(
            name,
            properties,
            constitutive_model=model,
            algorithm=self._original_material.algorithm,
            behaviors=behaviors,
            description=self.description_edit.text(),
        )

    def _behavior_rows(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for behavior_id, parameters in self._behaviors.items():
            try:
                label = material_behavior_spec(behavior_id).label
            except KeyError:
                label = f"其他行为（{behavior_id}）"
            rows.append((behavior_id, label, self._behavior_status(behavior_id, parameters)))
        known = self._KNOWN_PROPERTY_KEYS
        unknown = tuple(key for key in self._properties if key not in known)
        if unknown:
            rows.append(("preserved", "其他属性（来自 INP）", f"已保留 {len(unknown)} 项"))
        return rows

    def _behavior_status(self, behavior_id: str, parameters: Mapping[str, object]) -> str:
        required = self._REQUIRED_PARAMETER_KEYS.get(behavior_id, ())
        if required and not all(
            key in parameters and str(parameters[key]).strip() for key in required
        ):
            return "未完成"
        return "已定义"

    def _refresh_behaviors(self) -> None:
        selected_kind = self._selected_kind()
        rows = self._behavior_rows()
        self._row_kinds = [row[0] for row in rows]
        self.behavior_table.blockSignals(True)
        self.behavior_table.setRowCount(len(rows))
        for row, (_kind, behavior, summary) in enumerate(rows):
            self.behavior_table.setItem(row, 0, QTableWidgetItem(behavior))
            self.behavior_table.setItem(row, 1, QTableWidgetItem(summary))
        if rows:
            target = selected_kind if selected_kind in self._row_kinds else rows[0][0]
            self.behavior_table.selectRow(self._row_kinds.index(target))
        else:
            self.behavior_table.clearSelection()
        self.behavior_table.blockSignals(False)
        self._show_selected_behavior()
        self._update_buttons()
        self._update_status()

    def _selected_kind(self) -> str | None:
        row = self.behavior_table.currentRow()
        if 0 <= row < len(self._row_kinds):
            return self._row_kinds[row]
        return None

    def _add_behavior(self) -> None:
        self._add_behavior_kind(str(self.behavior_combo.currentData()))

    def _add_behavior_kind(self, kind: str) -> None:
        canonical = self._canonical_kind(kind)
        if canonical not in self._PARAMETER_KEYS:
            return
        if canonical not in self._behaviors:
            self._behaviors[canonical] = {}
        self._refresh_behaviors()
        if canonical in self._row_kinds:
            self.behavior_table.selectRow(self._row_kinds.index(canonical))
        self._show_selected_behavior()

    def _edit_behavior(self) -> None:
        kind = self._selected_kind()
        if kind is not None:
            self._edit_kind(kind)

    def _edit_kind(self, kind: str) -> None:
        canonical = self._canonical_kind(kind)
        properties = dict(self._behaviors.get(canonical, {}))
        if canonical == ELASTIC_BEHAVIOR_ID:
            dialog = ElasticBehaviorDialog(properties, self)
        elif canonical == PLASTIC_BEHAVIOR_ID:
            dialog = PlasticBehaviorDialog(properties, self)
        elif canonical == DENSITY_BEHAVIOR_ID:
            dialog = DensityBehaviorDialog(properties, self)
        elif canonical == NEO_HOOKEAN_BEHAVIOR_ID:
            dialog = NeoHookeanBehaviorDialog(properties, self)
        else:
            return
        if not dialog.exec():
            return
        values = dialog.values()
        self._behaviors[canonical] = dict(values)
        self._refresh_behaviors()
        if canonical in self._row_kinds:
            self.behavior_table.selectRow(self._row_kinds.index(canonical))

    def _delete_behavior(self) -> None:
        kind = self._selected_kind()
        if kind in self._behaviors:
            self._sync_editor_to_behavior()
            del self._behaviors[kind]
            self._refresh_behaviors()

    def _show_selected_behavior(self) -> None:
        self._sync_editor_to_behavior()
        kind = self._selected_kind()
        self._clear_behavior_editor()
        self._active_behavior_id = kind
        if kind is None:
            self.behavior_editor_layout.addWidget(QLabel("请选择或添加材料行为"))
            self._update_buttons()
            return
        if kind == "preserved":
            self.behavior_editor_layout.addWidget(
                QLabel("该材料行为来自 INP，当前版本仅保留其原始属性。")
            )
            self._update_buttons()
            return
        if kind not in self._PARAMETER_KEYS:
            self.behavior_editor_layout.addWidget(QLabel("当前材料行为暂不支持编辑。"))
            self._update_buttons()
            return
        self._build_inline_editor(kind)
        self._update_buttons()

    def _clear_behavior_editor(self) -> None:
        self._editor_widgets = {}
        def clear_layout(layout) -> None:
            while layout.count():
                item = layout.takeAt(0)
                nested = item.layout()
                if nested is not None:
                    clear_layout(nested)
                    nested.deleteLater()
                    continue
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        clear_layout(self.behavior_editor_layout)

    def _build_inline_editor(self, kind: str) -> None:
        title = QLabel(material_behavior_spec(kind).label)
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        self.behavior_editor_layout.addWidget(title)
        form = QFormLayout()
        configure_form_layout(form)
        parameters = self._behaviors[kind]
        if kind == ELASTIC_BEHAVIOR_ID:
            type_combo = QComboBox(self)
            type_combo.addItem("各向同性", "isotropic")
            type_combo.setToolTip("当前求解器暂支持各向同性线弹性")
            type_combo.setEnabled(False)
            form.addRow("类型", type_combo)
            fields = (("E", "弹性模量 E", 1.0e-15, 1.0e15), ("nu", "泊松比 ν", -0.999999, 0.499999))
        elif kind == PLASTIC_BEHAVIOR_ID:
            form.addRow("硬化方式", QLabel("各向同性"))
            fields = (("yield_stress", "屈服应力", 0.0, 1.0e15), ("hardening_modulus", "强化模量", 0.0, 1.0e15))
        elif kind == DENSITY_BEHAVIOR_ID:
            fields = (("rho", "密度 ρ", 0.0, 1.0e15),)
        else:
            form.addRow("材料类型", QLabel("各向同性"))
            form.addRow("应变能势", QLabel("Neo Hooke"))
            fields = (("C10", "C10", 1.0e-15, 1.0e15), ("D1", "D1", 1.0e-15, 1.0e15))
        for key, label, minimum, maximum in fields:
            editor = AdaptivePrecisionDoubleSpinBox(self)
            editor.setRange(minimum, maximum)
            editor.setMaximumWidth(140)
            if key in parameters:
                editor.setValue(float(parameters[key]))
            else:
                editor.clear()
            editor.textChanged.connect(self._update_status)
            self._editor_widgets[key] = editor
            form.addRow(label, editor)
        self.behavior_editor_layout.addLayout(form)
        self.behavior_editor_layout.addStretch(1)

    def _sync_editor_to_behavior(self) -> None:
        if self._active_behavior_id not in self._behaviors:
            return
        parameters = self._behaviors[self._active_behavior_id]
        for key, editor in self._editor_widgets.items():
            if editor.text().strip():
                parameters[key] = editor.value()
            else:
                parameters.pop(key, None)

    def _update_status(self) -> None:
        self._sync_editor_to_behavior()
        behaviors = tuple(
            MaterialBehavior(kind, parameters)
            for kind, parameters in self._behaviors.items()
        )
        diagnostics = material_behavior_diagnostics(behaviors)
        incomplete = any(
            self._behavior_status(kind, parameters) == "未完成"
            for kind, parameters in self._behaviors.items()
        )
        if diagnostics:
            self.status_label.setText("状态：" + "；".join(diagnostics))
        elif incomplete:
            self.status_label.setText("状态：材料行为参数未填写完整")
        else:
            self.status_label.setText("状态：材料定义完整")

    def _update_buttons(self) -> None:
        selected = self._selected_kind()
        editable = selected in self._behaviors
        self.edit_behavior_button.setEnabled(editable)
        self.delete_behavior_button.setEnabled(editable)


class MaterialManagerDialog(QDialog):
    def __init__(
        self,
        materials: Sequence[MaterialDefinition],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("材料管理")
        self.materials: list[MaterialDefinition] = deepcopy(list(materials))
        self._original_names: tuple[str, ...] = tuple(
            material.name for material in self.materials
        )
        self._origins: list[str | None] = list(self._original_names)
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
        self.copy_button = QPushButton("复制", self)
        self.rename_button = QPushButton("重命名", self)
        self.delete_button = QPushButton("删除", self)
        self.add_button.clicked.connect(self._add)
        self.edit_button.clicked.connect(self._edit)
        self.copy_button.clicked.connect(self._copy)
        self.rename_button.clicked.connect(self._rename)
        self.delete_button.clicked.connect(self._delete)
        self.table.itemDoubleClicked.connect(lambda _item: self._edit())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        controls = QVBoxLayout()
        for button in (
            self.add_button,
            self.edit_button,
            self.copy_button,
            self.rename_button,
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
        content = QHBoxLayout()
        content.addWidget(self.table, 1)
        content.addLayout(controls)
        layout.addLayout(content, 1)
        layout.addWidget(buttons)
        self.resize(520, 350)
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
        summary = behavior_summary(material)
        if summary != "未定义":
            return summary
        if material.properties:
            return "其他属性"
        return "未定义"

    def _selected_row(self) -> int:
        return self.table.currentRow()

    def _store(self, value: MaterialDefinition, row: int | None = None) -> None:
        value = deepcopy(value)
        duplicate = next(
            (
                index
                for index, item in enumerate(self.materials)
                if item.name.casefold() == value.name.casefold()
                and index != row
            ),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"材料名称已存在：{value.name}")
        if row is None:
            self.materials.append(value)
            self._origins.append(None)
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

    def _copy(self) -> None:
        row = self._selected_row()
        if row < 0:
            return
        source = self.materials[row]
        name = self._unique_copy_name(source.name)
        self._store(replace(deepcopy(source), name=name))
        self.table.selectRow(len(self.materials) - 1)

    def _unique_copy_name(self, source_name: str) -> str:
        candidate = f"{source_name}-副本"
        suffix = 2
        existing = {material.name for material in self.materials}
        while candidate in existing:
            candidate = f"{source_name}-副本{suffix}"
            suffix += 1
        return candidate

    def _rename(self) -> None:
        row = self._selected_row()
        if row < 0:
            return
        current = self.materials[row]
        name, accepted = QInputDialog.getText(
            self,
            "重命名材料",
            "材料名称：",
            text=current.name,
        )
        name = name.strip()
        if not accepted or not name or name == current.name:
            return
        try:
            self._store(replace(current, name=name), row)
        except ValueError as error:
            QMessageBox.warning(self, "材料", str(error))

    def _delete(self) -> None:
        row = self._selected_row()
        if row >= 0:
            del self.materials[row]
            del self._origins[row]
            self._refresh()

    def _update_buttons(self) -> None:
        selected = self._selected_row() >= 0
        self.edit_button.setEnabled(selected)
        self.copy_button.setEnabled(selected)
        self.rename_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def values(self) -> list[MaterialDefinition]:
        return deepcopy(self.materials)

    def rename_intents(self) -> tuple[RenameIntent, ...]:
        return tuple(
            RenameIntent(origin, material.name)
            for origin, material in zip(self._origins, self.materials)
            if origin is not None and origin != material.name
        )

    def delete_intents(self) -> tuple[DeleteIntent, ...]:
        retained = {origin for origin in self._origins if origin is not None}
        return tuple(
            DeleteIntent(name)
            for name in self._original_names
            if name not in retained
        )


class SectionEditDialog(QDialog):
    def __init__(
        self,
        materials: list[MaterialDefinition],
        section: SectionDefinition | None = None,
        parent=None,
        *,
        model_dimension: int = 2,
        section_presets: Sequence[str] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑截面" if section else "新建截面")
        self.model_dimension = int(model_dimension)
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
        self.form.addRow("高度（局部 z）", self.height_spin)
        self.form.addRow("宽度（局部 y）", self.width_spin)
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
            constitutive_model=selected_material.constitutive_model,
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
        return preset

    def _current_preset(self) -> str:
        return str(self.type_combo.currentData() or "").casefold()

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
        materials: Sequence[MaterialDefinition],
        sections: Sequence[SectionDefinition],
        parent=None,
        *,
        model_dimension: int = 2,
        section_presets: Sequence[str] = (),
        authoring_enabled: bool = True,
        assignment_usage: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("截面管理")
        self.materials = deepcopy(materials)
        self.sections: list[SectionDefinition] = deepcopy(list(sections))
        self._original_names: tuple[str, ...] = tuple(
            section.name for section in sections
        )
        self._origins: list[str | None] = list(self._original_names)
        self.model_dimension = int(model_dimension)
        self._section_presets_arg = tuple(section_presets)
        self._assignment_usage = {
            str(name): int(count)
            for name, count in (assignment_usage or {}).items()
        }
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
        self.add_button = QPushButton("新建", self)
        self.edit_button = QPushButton("编辑", self)
        self.copy_button = QPushButton("复制", self)
        self.rename_button = QPushButton("重命名", self)
        self.delete_button = QPushButton("删除", self)
        self.add_button.clicked.connect(self._add)
        self.edit_button.clicked.connect(self._edit)
        self.copy_button.clicked.connect(self._copy)
        self.rename_button.clicked.connect(self._rename)
        self.delete_button.clicked.connect(self._delete)
        self.table.itemDoubleClicked.connect(lambda _item: self._edit())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        controls = QHBoxLayout()
        for button in (
            self.add_button,
            self.edit_button,
            self.copy_button,
            self.rename_button,
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
                item = QTableWidgetItem(value)
                if column == 0:
                    count = self._assignment_usage.get(section.name, 0)
                    item.setToolTip(f"当前截面分配引用：{count} 个")
                self.table.setItem(row, column, item)
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
        value = deepcopy(value)
        if any(
            item.name.casefold() == value.name.casefold()
            and index != row
            for index, item in enumerate(self.sections)
        ):
            raise ValueError(f"截面名称已存在：{value.name}")
        if row is None:
            self.sections.append(value)
            self._origins.append(None)
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

    def _copy(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        source = self.sections[row]
        names = {section.name.casefold() for section in self.sections}
        candidate = f"{source.name}-副本"
        suffix = 2
        while candidate.casefold() in names:
            candidate = f"{source.name}-副本{suffix}"
            suffix += 1
        self._store(replace(deepcopy(source), name=candidate))

    def _rename(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        current = self.sections[row].name
        value, accepted = QInputDialog.getText(
            self,
            "重命名截面",
            "截面名称：",
            text=current,
        )
        if not accepted:
            return
        name = str(value).strip()
        if not name or name == current:
            return
        try:
            self._store(replace(self.sections[row], name=name), row)
        except ValueError as error:
            QMessageBox.warning(self, "截面", str(error))

    def _delete(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            del self.sections[row]
            del self._origins[row]
            self._refresh()

    def _update_buttons(self) -> None:
        selected = self.table.currentRow() >= 0
        self.edit_button.setEnabled(selected)
        self.copy_button.setEnabled(selected)
        self.rename_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def values(self) -> list[SectionDefinition]:
        return deepcopy(self.sections)

    def rename_intents(self) -> tuple[RenameIntent, ...]:
        return tuple(
            RenameIntent(origin, section.name)
            for origin, section in zip(self._origins, self.sections)
            if origin is not None and origin != section.name
        )

    def delete_intents(self) -> tuple[DeleteIntent, ...]:
        retained = {origin for origin in self._origins if origin is not None}
        return tuple(
            DeleteIntent(name)
            for name in self._original_names
            if name not in retained
        )


class RegionAssignmentDialog(QDialog):
    def __init__(
        self,
        sections: Sequence[SectionDefinition],
        regions: Sequence[RegionRef] = (),
        parent=None,
        *,
        compatible_targets: Mapping[
            str,
            Sequence[RegionRef],
        ]
        | None = None,
        current: RegionAssignment | None = None,
        candidate_evaluator: (
            Callable[[RegionAssignment], AuthoringCapability] | None
        ) = None,
        explicit_reference: Sequence[float] | None = None,
        orientation_suggester: Callable[[RegionRef], object] | None = None,
        allow_scope_selection: bool = False,
        selected_region_name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "编辑截面分配" if current is not None else "截面分配"
        )
        self._sections = tuple(deepcopy(sections))
        self._regions = self._normalize_regions(regions)
        self._current = deepcopy(current)
        self._candidate_evaluator = candidate_evaluator
        self._orientation_suggester = orientation_suggester
        self._conversion_message = ""
        self._last_candidate: RegionAssignment | None = None
        self._last_candidate_decision: AuthoringCapability | None = None
        self._scope_selection_request: str | None = None
        self._compatible_targets = (
            None
            if compatible_targets is None
            else {
                str(key): self._normalize_regions(values)
                for key, values in compatible_targets.items()
            }
        )
        self.section_combo, self.region_combo = QComboBox(self), QComboBox(self)
        self.scope_pick_button = QPushButton("创建", self)
        self.scope_pick_button.setEnabled(bool(allow_scope_selection))
        self.scope_pick_button.clicked.connect(
            self._request_scope_selection
        )
        region_widget = QWidget(self)
        region_layout = QHBoxLayout(region_widget)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.addWidget(self.region_combo, 1)
        region_layout.addWidget(self.scope_pick_button)
        self.orientation_mode_combo = QComboBox(self)
        self.orientation_mode_combo.addItem("自动", "automatic")
        self.orientation_mode_combo.addItem("参考方向", "explicit")
        authored_orientation = (
            None
            if current is None
            else getattr(current, "beam_orientation", None)
        )
        authored_reference = getattr(
            authored_orientation,
            "local_y_reference",
            None,
        )
        initial_reference = (
            authored_reference
            if authored_reference is not None
            else explicit_reference
        )
        if initial_reference is None:
            initial_reference = (0.0, 0.0, 0.0)
        try:
            reference_values = tuple(float(value) for value in initial_reference)
        except (TypeError, ValueError):
            reference_values = (0.0, 0.0, 0.0)
        if len(reference_values) != 3:
            reference_values = (0.0, 0.0, 0.0)
        self.orientation_x_spin = _signed_number(self, reference_values[0])
        self.orientation_y_spin = _signed_number(self, reference_values[1])
        self.orientation_z_spin = _signed_number(self, reference_values[2])
        self.orientation_diagnostic_label = QLabel(self)
        self.orientation_diagnostic_label.setWordWrap(True)
        self.section_combo.addItems(
            [section.name for section in self._sections]
        )
        self.section_combo.currentIndexChanged.connect(
            self._refresh_regions
        )
        self.section_combo.currentIndexChanged.connect(
            self._update_orientation_fields
        )
        self.orientation_mode_combo.currentIndexChanged.connect(
            self._orientation_mode_changed
        )
        self.region_combo.currentIndexChanged.connect(
            self._orientation_target_changed
        )
        for spin in (
            self.orientation_x_spin,
            self.orientation_y_spin,
            self.orientation_z_spin,
        ):
            spin.valueChanged.connect(self._update_orientation_fields)
        self.form = QFormLayout()
        configure_form_layout(self.form)
        self.form.addRow("截面", self.section_combo)
        self.form.addRow("单元作用域", region_widget)
        self.form.addRow("梁截面方向", self.orientation_mode_combo)
        self.form.addRow("参考方向 X", self.orientation_x_spin)
        self.form.addRow("参考方向 Y", self.orientation_y_spin)
        self.form.addRow("参考方向 Z", self.orientation_z_spin)
        self.form.addRow(self.orientation_diagnostic_label)
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
        self.setMinimumWidth(300)
        self._refresh_regions()
        if current is not None:
            section_index = self.section_combo.findText(current.section_name)
            if section_index >= 0:
                self.section_combo.setCurrentIndex(section_index)
            self._select_region(current.region_name)
        elif selected_region_name is not None:
            self._select_region(selected_region_name)
        if authored_orientation is not None:
            self.orientation_mode_combo.setCurrentIndex(
                self.orientation_mode_combo.findData("explicit")
            )
        self._update_orientation_fields()

    def _request_scope_selection(self) -> None:
        if not self.scope_pick_button.isEnabled():
            return
        self._scope_selection_request = "element_set"
        self.reject()

    def requested_scope_kind(self) -> str | None:
        return self._scope_selection_request

    def assignment(self) -> RegionAssignment:
        section_name = self.section_combo.currentText().strip()
        if not section_name:
            raise ValueError("截面不能为空")
        region = self.region_combo.currentData()
        if not isinstance(region, RegionRef):
            raise ValueError("没有可分配的兼容单元作用域")
        region_name = require_region_kind(region, "element_set")
        orientation = self.beam_orientation()
        if orientation is None:
            return RegionAssignment(section_name, region_name)
        return RegionAssignment(
            section_name,
            region_name,
            beam_orientation=orientation,
        )

    def beam_orientation(self) -> BeamOrientation | None:
        if (
            not self._selected_section_is_beam()
            or self.orientation_mode_combo.currentData() != "explicit"
        ):
            return None
        reference = self.reference_vector()
        if not all(isfinite(value) for value in reference):
            raise ValueError("梁截面参考方向必须包含三个有限分量")
        if not any(value != 0.0 for value in reference):
            raise ValueError("梁截面参考方向不能为零向量")
        return BeamOrientation(reference)

    def reference_vector(self) -> tuple[float, float, float]:
        return tuple(
            float(spin.value())
            for spin in (
                self.orientation_x_spin,
                self.orientation_y_spin,
                self.orientation_z_spin,
            )
        )

    def candidate_decision(
        self,
        candidate: RegionAssignment | None = None,
    ) -> AuthoringCapability:
        value = self.assignment() if candidate is None else candidate
        if (
            self._last_candidate is not None
            and value == self._last_candidate
        ):
            return self._last_candidate_decision
        if self._candidate_evaluator is None:
            raise RuntimeError("region assignment candidate evaluator is required")
        decision = self._candidate_evaluator(value)
        if type(decision) is not AuthoringCapability:
            raise TypeError(
                "region assignment candidate evaluator must return "
                "AuthoringCapability"
            )
        self._last_candidate = deepcopy(value)
        self._last_candidate_decision = decision
        return decision

    def accept(self) -> None:
        try:
            candidate = self.assignment()
            decision = self.candidate_decision(candidate)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "截面分配", str(error))
            return
        if not self._decision_enabled(decision):
            self.orientation_diagnostic_label.setText(
                self._decision_text(decision)
            )
            self.form.setRowVisible(
                self.orientation_diagnostic_label,
                True,
            )
            return
        super().accept()

    @staticmethod
    def _normalize_regions(
        values: Sequence[RegionRef],
    ) -> tuple[RegionRef, ...]:
        normalized: list[RegionRef] = []
        for value in values:
            if type(value) is not RegionRef:
                raise TypeError("assignment regions must contain RegionRef values")
            if value.kind != "element_set":
                raise ValueError("assignment regions must use element_set namespace")
            if value not in normalized:
                normalized.append(value)
        return tuple(normalized)

    def _refresh_regions(self) -> None:
        current_region = self.region_combo.currentData()
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
        if (
            self._current is not None
            and 0 <= section_index < len(self._sections)
            and self._sections[section_index].name
            == self._current.section_name
        ):
            existing = RegionRef(
                "element_set",
                self._current.region_name,
            )
            if existing not in targets:
                targets = (*targets, existing)
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
        if isinstance(current_region, RegionRef):
            index = self.region_combo.findData(current_region)
            if index >= 0:
                self.region_combo.setCurrentIndex(index)
        self._update_orientation_fields()

    def _select_region(self, region_name: str) -> None:
        index = self.region_combo.findData(
            RegionRef("element_set", str(region_name))
        )
        if index >= 0:
            self.region_combo.setCurrentIndex(index)

    def _orientation_mode_changed(self) -> None:
        if self.orientation_mode_combo.currentData() == "explicit":
            self._maybe_prefill_orientation()
        self._update_orientation_fields()

    def _orientation_target_changed(self) -> None:
        self._conversion_message = ""
        if self.orientation_mode_combo.currentData() == "explicit":
            self._maybe_prefill_orientation()
        self._update_orientation_fields()

    def _maybe_prefill_orientation(self) -> None:
        if (
            self._orientation_suggester is None
            or any(value != 0.0 for value in self.reference_vector())
        ):
            return
        region = self.region_combo.currentData()
        if not isinstance(region, RegionRef):
            return
        try:
            report = self._orientation_suggester(region)
        except (KeyError, TypeError, ValueError):
            report = None
        suggested = getattr(report, "suggested_orientation", report)
        reference = getattr(suggested, "local_y_reference", None)
        if reference is not None:
            values = tuple(float(value) for value in reference)
            if len(values) == 3 and all(isfinite(value) for value in values):
                for spin, value in zip(
                    (
                        self.orientation_x_spin,
                        self.orientation_y_spin,
                        self.orientation_z_spin,
                    ),
                    values,
                ):
                    spin.setValue(value)
                self._conversion_message = ""
                return
        self._conversion_message = (
            "当前区域的 compatibility frame 无法无损转换为一个统一的显式参考"
            "方向；切换可能改变截面方向，请输入并预览后提交。"
        )

    def _selected_section_is_beam(self) -> bool:
        section_index = self.section_combo.currentIndex()
        if section_index < 0 or section_index >= len(self._sections):
            return False
        section_type = str(
            self._sections[section_index].section_type
        ).strip().casefold()
        return section_type in {
            "beam",
            "rectangle",
            "solid_circle",
            "hollow_circle",
        }

    def _update_orientation_fields(self) -> None:
        beam_section = self._selected_section_is_beam()
        if (
            not beam_section
            and self.orientation_mode_combo.currentData() != "automatic"
        ):
            self.orientation_mode_combo.setCurrentIndex(
                self.orientation_mode_combo.findData("automatic")
            )
        explicit = (
            beam_section
            and self.orientation_mode_combo.currentData() == "explicit"
        )
        self.form.setRowVisible(self.orientation_mode_combo, beam_section)
        for spin in (
            self.orientation_x_spin,
            self.orientation_y_spin,
            self.orientation_z_spin,
        ):
            self.form.setRowVisible(spin, explicit)
        reference_valid = (
            not explicit
            or (
                all(isfinite(value) for value in self.reference_vector())
                and any(value != 0.0 for value in self.reference_vector())
            )
        )
        if explicit and not reference_valid:
            self.orientation_diagnostic_label.setText(
                self._conversion_message
                or "显式参考方向必须是非零的三个有限分量。"
            )
        else:
            self.orientation_diagnostic_label.clear()
        self.form.setRowVisible(
            self.orientation_diagnostic_label,
            explicit and not reference_valid,
        )
        self.buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setEnabled(reference_valid and self.region_combo.count() > 0)
        self._last_candidate = None
        self._last_candidate_decision = None

    @staticmethod
    def _decision_enabled(decision: AuthoringCapability) -> bool:
        if type(decision) is not AuthoringCapability:
            raise TypeError("candidate decision must be AuthoringCapability")
        return decision.can_submit

    @staticmethod
    def _decision_text(decision: AuthoringCapability) -> str:
        if type(decision) is not AuthoringCapability:
            raise TypeError("candidate decision must be AuthoringCapability")
        diagnostics = decision.diagnostics
        lines = []
        for diagnostic in diagnostics:
            code = str(getattr(diagnostic, "code", "")).strip()
            message = str(getattr(diagnostic, "message", "")).strip()
            remediation = str(
                getattr(diagnostic, "remediation", "")
            ).strip()
            text = f"[{code}] {message}" if code else message
            if remediation:
                text += f"\n建议：{remediation}"
            if text:
                lines.append(text)
        return "\n".join(lines) or "当前截面分配不能提交。"


class RegionAssignmentManagerDialog(QDialog):
    """Manage all section-to-region associations in one small list."""

    locateRequested = Signal(str)

    def __init__(
        self,
        sections: Sequence[SectionDefinition],
        assignments: Sequence[RegionAssignment],
        editor_factory: Callable[
            [RegionAssignment | None, int | None],
            QDialog | None,
        ],
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not callable(editor_factory):
            raise TypeError("editor_factory must be callable")
        self.setWindowTitle("截面分配管理")
        self.sections = tuple(deepcopy(sections))
        self.assignments = list(deepcopy(assignments))
        self._editor_factory = editor_factory
        self._scope_selection_request: str | None = None
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(("截面", "单元作用域", "梁方向"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.add_button = QPushButton("新建", self)
        self.edit_button = QPushButton("编辑", self)
        self.replace_button = QPushButton("替换", self)
        self.delete_button = QPushButton("删除", self)
        self.locate_button = QPushButton("定位区域", self)
        self.add_button.clicked.connect(self._add)
        self.edit_button.clicked.connect(self._edit)
        self.replace_button.clicked.connect(self._replace)
        self.delete_button.clicked.connect(self._delete)
        self.locate_button.clicked.connect(self._locate)
        self.table.itemDoubleClicked.connect(lambda _item: self._edit())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        controls = QHBoxLayout()
        for button in (
            self.add_button,
            self.edit_button,
            self.replace_button,
            self.delete_button,
            self.locate_button,
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
        self.resize(620, 340)
        self._refresh()

    def _refresh(self, selected: int = 0) -> None:
        self.table.setRowCount(len(self.assignments))
        for row, assignment in enumerate(self.assignments):
            values = (
                str(assignment.section_name),
                str(assignment.region_name),
                "显式" if assignment.beam_orientation is not None else "自动",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.table.setItem(row, column, item)
        if self.assignments:
            self.table.selectRow(max(0, min(selected, len(self.assignments) - 1)))
        self._update_buttons()

    def _selected_index(self) -> int | None:
        row = self.table.currentRow()
        return row if 0 <= row < len(self.assignments) else None

    def _editor(self, index: int | None) -> QDialog | None:
        current = None if index is None else self.assignments[index]
        return self._editor_factory(current, index)

    def _add(self) -> None:
        dialog = self._editor(None)
        if dialog is None:
            return
        if not dialog.exec():
            requested_scope_kind = self._requested_scope_kind(dialog)
            if requested_scope_kind is not None:
                # The editor can only request a viewport selection after it
                # closes.  Propagate that intent to the owning main window;
                # otherwise the manager silently eats the request.
                self._scope_selection_request = requested_scope_kind
                self.reject()
            return
        try:
            self._store(dialog.assignment(), None)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "截面分配", str(error))

    @staticmethod
    def _requested_scope_kind(dialog: QDialog) -> str | None:
        getter = getattr(dialog, "requested_scope_kind", None)
        if not callable(getter):
            return None
        value = getter()
        return None if value is None else str(value)

    def requested_scope_kind(self) -> str | None:
        """Return a scope-picking request that interrupted the manager."""

        return self._scope_selection_request

    def _edit(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        dialog = self._editor(index)
        if dialog is None or not dialog.exec():
            return
        try:
            self._store(dialog.assignment(), index)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "截面分配", str(error))

    def _replace(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        dialog = self._editor(index)
        if dialog is None or not dialog.exec():
            return
        try:
            self._store(dialog.assignment(), index, force_replace=True)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "截面分配", str(error))

    def _store(
        self,
        value: RegionAssignment,
        index: int | None,
        *,
        force_replace: bool = False,
    ) -> None:
        if type(value) is not RegionAssignment:
            raise TypeError("assignment editor must return RegionAssignment")
        duplicate = next(
            (
                candidate_index
                for candidate_index, candidate in enumerate(self.assignments)
                if candidate.region_name == value.region_name
                and candidate_index != index
            ),
            None,
        )
        if duplicate is not None:
            if not force_replace:
                answer = QMessageBox.question(
                    self,
                    "替换截面分配",
                    f"作用域“{value.region_name}”已有截面分配，是否替换？",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            del self.assignments[duplicate]
            if index is not None and duplicate < index:
                index -= 1
        if index is None:
            self.assignments.append(deepcopy(value))
            index = len(self.assignments) - 1
        else:
            self.assignments[index] = deepcopy(value)
        self._refresh(index)

    def _delete(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        del self.assignments[index]
        self._refresh(max(0, index - 1))

    def _locate(self) -> None:
        index = self._selected_index()
        if index is not None:
            self.locateRequested.emit(str(self.assignments[index].region_name))

    def _update_buttons(self) -> None:
        selected = self._selected_index() is not None
        self.edit_button.setEnabled(selected)
        self.replace_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)
        self.locate_button.setEnabled(selected)

    def values(self) -> list[RegionAssignment]:
        return deepcopy(self.assignments)
