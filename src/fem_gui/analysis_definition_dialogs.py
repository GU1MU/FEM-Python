"""Modal dialogs for the currently supported linear-static analysis inputs."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)
from fem.steps.factory import static

from .dialogs import CompactDoubleSpinBox, configure_form_layout


def _value(parent: QDialog, value: float = 0.0) -> QDoubleSpinBox:
    box = CompactDoubleSpinBox(parent)
    box.setRange(-1.0e15, 1.0e15)
    box.setDecimals(8)
    box.setValue(float(value))
    return box


def _buttons(dialog: QDialog) -> QDialogButtonBox:
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
    buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    return buttons


class StaticStepDialog(QDialog):
    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("创建静力分析步")
        self.name_edit = QLineEdit(name, self)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("分析步名称", self.name_edit)
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(_buttons(self))

    def step(self):
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("分析步名称不能为空")
        return static(name)


class DisplacementDialog(QDialog):
    def __init__(
        self,
        step_names: list[str],
        regions: list[str],
        dimensions: int,
        parent=None,
        *,
        selected_region: str | None = None,
        current: DisplacementConstraint | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("位移边界条件")
        self.region_combo, self.step_combo = QComboBox(self), QComboBox(self)
        self.region_combo.addItems(regions); self.step_combo.addItems(step_names)
        if selected_region:
            self.region_combo.setCurrentText(selected_region)
        self.component_checks: dict[int, QCheckBox] = {}
        self.component_values: dict[int, QDoubleSpinBox] = {}
        component_widget = QWidget(self)
        component_layout = QGridLayout(component_widget)
        component_layout.setContentsMargins(0, 0, 0, 0)
        component_layout.addWidget(QLabel("自由度", component_widget), 0, 0)
        component_layout.addWidget(QLabel("位移值", component_widget), 0, 1)
        for component in range(1, dimensions + 1):
            check = QCheckBox(f"U{component}", component_widget)
            value = _value(self)
            value.setEnabled(False)
            check.toggled.connect(value.setEnabled)
            component_layout.addWidget(check, component, 0)
            component_layout.addWidget(value, component, 1)
            self.component_checks[component] = check
            self.component_values[component] = value
        if self.component_checks:
            self.component_checks[1].setChecked(True)
        if current is not None:
            self.region_combo.setCurrentText(str(current.target))
            for component, check in self.component_checks.items():
                selected = (
                    current.first_component
                    <= component
                    <= current.last_component
                )
                check.setChecked(selected)
                if selected:
                    self.component_values[component].setValue(current.value)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("选择区域", self.region_combo); form.addRow("分析步", self.step_combo)
        form.addRow("约束分量", component_widget)
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(_buttons(self)); self.setMinimumWidth(350)

    def definitions(self) -> tuple[str, tuple[DisplacementConstraint, ...]]:
        target = self.region_combo.currentText().strip()
        if not target:
            raise ValueError("请选择约束区域")
        step_name = self.step_combo.currentText().strip()
        if not step_name:
            raise ValueError("请选择分析步")
        values = tuple(
            DisplacementConstraint(
                target,
                component,
                component,
                self.component_values[component].value(),
            )
            for component, check in self.component_checks.items()
            if check.isChecked()
        )
        if not values:
            raise ValueError("至少勾选一个位移自由度")
        return step_name, values

    def definition(self) -> tuple[str, DisplacementConstraint]:
        """Return the compact legacy form when selected values can be combined."""
        step_name, values = self.definitions()
        components = [value.first_component for value in values]
        displacement_values = {value.value for value in values}
        if (
            components == list(range(components[0], components[-1] + 1))
            and len(displacement_values) == 1
        ):
            return step_name, DisplacementConstraint(
                values[0].target,
                components[0],
                components[-1],
                values[0].value,
            )
        raise ValueError("不同自由度的位移值不能合并为一个边界条件")


class LoadDialog(QDialog):
    _COMPONENT_LABELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

    def __init__(
        self,
        step_names: list[str],
        node_regions: list[str],
        edge_regions: list[str],
        face_regions: list[str],
        dimensions: int,
        parent=None,
        *,
        spatial_dimensions: int | None = None,
        selected_region: str | None = None,
        preferred_kind: str | None = None,
        current: NodalLoad | EdgeLoad | SurfaceLoad | GravityLoad | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑载荷" if current is not None else "创建载荷")
        self._regions = {"node": node_regions, "edge": edge_regions, "surface": face_regions}
        self.dimensions = dimensions
        self.spatial_dimensions = int(
            spatial_dimensions if spatial_dimensions is not None else min(dimensions, 3)
        )
        self.kind_combo, self.region_combo, self.step_combo = QComboBox(self), QComboBox(self), QComboBox(self)
        if node_regions: self.kind_combo.addItem("节点力", "node")
        if edge_regions: self.kind_combo.addItem("边载荷", "edge")
        if face_regions: self.kind_combo.addItem("面载荷", "surface")
        self.kind_combo.addItem("重力", "gravity")
        self._gravity_target = (
            current.target if isinstance(current, GravityLoad) else None
        )
        self.step_combo.addItems(step_names)
        self.load_type_combo = QComboBox(self); self.load_type_combo.addItem("牵引", "traction"); self.load_type_combo.addItem("压力", "pressure")
        self.component_combo = QComboBox(self)
        for component in range(1, dimensions + 1):
            label = (
                self._COMPONENT_LABELS[component - 1]
                if component <= len(self._COMPONENT_LABELS)
                else f"F{component}"
            )
            self.component_combo.addItem(label, component)
        self.value_spin, self.x_spin, self.y_spin, self.z_spin = _value(self), _value(self), _value(self), _value(self)
        zero_vector = (0.0,) * self.spatial_dimensions
        gravity_vector = list(zero_vector)
        gravity_vector[-1] = -9.81
        self._vector_values = {
            "edge": zero_vector,
            "surface": zero_vector,
            "gravity": tuple(gravity_vector),
        }
        self._active_vector_kind: str | None = None
        self.form = QFormLayout()
        configure_form_layout(self.form)
        self.form.addRow("载荷类别", self.kind_combo)
        self.form.addRow("选择区域", self.region_combo)
        self.form.addRow("分析步", self.step_combo)
        self.form.addRow("载荷形式", self.load_type_combo)
        self.form.addRow("分量", self.component_combo)
        self.form.addRow("载荷值", self.value_spin)
        self.form.addRow("Fx", self.x_spin)
        self.form.addRow("Fy", self.y_spin)
        self.form.addRow("Fz", self.z_spin)
        self.kind_combo.currentIndexChanged.connect(self._refresh); self.load_type_combo.currentIndexChanged.connect(self._refresh)
        if preferred_kind:
            index = self.kind_combo.findData(preferred_kind)
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        if isinstance(current, NodalLoad):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("node"))
            )
            self.component_combo.setCurrentIndex(
                max(0, self.component_combo.findData(current.component))
            )
            self.value_spin.setValue(current.value)
            selected_region = str(current.target)
        elif isinstance(current, EdgeLoad):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("edge"))
            )
            selected_region = current.edge
            self.load_type_combo.setCurrentIndex(
                max(0, self.load_type_combo.findData(current.load_type))
            )
            self._set_distributed_values(current.vector, current.magnitude)
            self._vector_values["edge"] = tuple(current.vector)
        elif isinstance(current, SurfaceLoad):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("surface"))
            )
            selected_region = current.surface
            self.load_type_combo.setCurrentIndex(
                max(0, self.load_type_combo.findData(current.load_type))
            )
            self._set_distributed_values(current.vector, current.magnitude)
            self._vector_values["surface"] = tuple(current.vector)
        elif isinstance(current, GravityLoad):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("gravity"))
            )
            self._set_distributed_values(
                tuple(current.acceleration),
                None,
            )
            self._vector_values["gravity"] = tuple(current.acceleration)
        layout = QVBoxLayout(self); layout.addLayout(self.form); layout.addWidget(_buttons(self)); self.setMinimumWidth(350); self._refresh()
        if selected_region:
            self.region_combo.setCurrentText(selected_region)

    def _set_distributed_values(
        self,
        vector: tuple[float, ...],
        magnitude: float | None,
    ) -> None:
        if magnitude is not None:
            self.value_spin.setValue(magnitude)
        for spin, value in zip(
            (self.x_spin, self.y_spin, self.z_spin),
            vector,
        ):
            spin.setValue(value)

    def _refresh(self) -> None:
        kind = str(self.kind_combo.currentData() or "node")
        if (
            self._active_vector_kind in self._vector_values
            and self._active_vector_kind != kind
        ):
            self._vector_values[self._active_vector_kind] = tuple(
                spin.value()
                for spin in (self.x_spin, self.y_spin, self.z_spin)[
                    :self.spatial_dimensions
                ]
            )
        if kind in self._vector_values and self._active_vector_kind != kind:
            self._set_distributed_values(
                self._vector_values[kind],
                None,
            )
        self._active_vector_kind = (
            kind if kind in self._vector_values else None
        )
        gravity = kind == "gravity"
        current = self.region_combo.currentText()
        self.region_combo.clear()
        if gravity:
            if self._gravity_target is not None:
                self.region_combo.addItem(
                    str(self._gravity_target),
                    self._gravity_target,
                )
        else:
            self.region_combo.addItems(self._regions[kind])
        self.region_combo.setCurrentText(current)
        distributed = kind in {"edge", "surface"}
        pressure = self.load_type_combo.currentData() == "pressure"
        self.form.setRowVisible(
            self.region_combo,
            not gravity or self._gravity_target is not None,
        )
        self.region_combo.setEnabled(not gravity)
        self.form.setRowVisible(self.load_type_combo, distributed)
        self.form.setRowVisible(self.component_combo, kind == "node")
        self.form.setRowVisible(
            self.value_spin,
            kind == "node" or (distributed and pressure),
        )
        self.form.labelForField(self.value_spin).setText(
            "压力值" if pressure and distributed else "载荷值"
        )
        show_vector = gravity or (distributed and not pressure)
        vector_labels = ("ax", "ay", "az") if gravity else ("Fx", "Fy", "Fz")
        for spin, label in zip(
            (self.x_spin, self.y_spin, self.z_spin),
            vector_labels,
        ):
            self.form.labelForField(spin).setText(label)
        self.form.setRowVisible(self.x_spin, show_vector)
        self.form.setRowVisible(
            self.y_spin,
            show_vector and self.spatial_dimensions >= 2,
        )
        self.form.setRowVisible(
            self.z_spin,
            show_vector and self.spatial_dimensions == 3,
        )

    def definition(self):
        kind = str(self.kind_combo.currentData())
        step = self.step_combo.currentText().strip()
        region = self.region_combo.currentText().strip()
        if not step:
            raise ValueError("请选择分析步")
        if kind == "gravity":
            acceleration = (self.x_spin.value(),)
            if self.spatial_dimensions >= 2:
                acceleration += (self.y_spin.value(),)
            if self.spatial_dimensions == 3:
                acceleration += (self.z_spin.value(),)
            return step, GravityLoad(
                acceleration,
                self._gravity_target,
            )
        if not region:
            raise ValueError("请选择载荷区域")
        if kind == "node":
            component = self.component_combo.currentData()
            if component is None:
                raise ValueError("请选择节点力分量")
            return step, NodalLoad(
                region,
                int(component),
                self.value_spin.value(),
            )
        if kind not in {"edge", "surface"}:
            raise ValueError("当前没有可用的载荷区域")
        load_type = str(self.load_type_combo.currentData())
        if load_type == "pressure":
            return step, (EdgeLoad(region, magnitude=self.value_spin.value(), load_type="pressure") if kind == "edge" else SurfaceLoad(region, magnitude=self.value_spin.value(), load_type="pressure"))
        vector = tuple(value.value() for value in (self.x_spin, self.y_spin))
        if self.spatial_dimensions == 3:
            vector += (self.z_spin.value(),)
        return step, (EdgeLoad(region, vector, load_type="traction") if kind == "edge" else SurfaceLoad(region, vector, load_type="traction"))


class OutputRequestDialog(QDialog):
    _VARIABLES = (
        ("U", "位移 U", "node"),
        ("RF", "反力 RF", "node"),
        ("S", "应力 S", "element"),
    )

    def __init__(
        self,
        step_names: list[str],
        parent=None,
        *,
        current: OutputRequest | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑输出请求" if current is not None else "创建输出请求")
        self.step_combo, self.kind_combo, self.target_combo = QComboBox(self), QComboBox(self), QComboBox(self)
        self.step_combo.addItems(step_names)
        self.kind_combo.addItem("字段输出", "field")
        if current is not None and current.kind != "field":
            self.kind_combo.addItem("历史输出（来自 INP）", current.kind)
        self.target_combo.addItem("节点", "node")
        self.target_combo.addItem("单元", "element")
        if (
            current is not None
            and self.target_combo.findData(current.target) < 0
        ):
            self.target_combo.addItem(
                f"{current.target}（来自 INP）",
                current.target,
            )
        self.variable_checks: dict[str, QCheckBox] = {}
        variable_widget = QWidget(self)
        variable_layout = QVBoxLayout(variable_widget)
        variable_layout.setContentsMargins(0, 0, 0, 0)
        for variable, label, _target in self._VARIABLES:
            check = QCheckBox(label, variable_widget)
            self.variable_checks[variable] = check
            variable_layout.addWidget(check)
        self.preserved_label = QLabel("", variable_widget)
        self.preserved_label.setWordWrap(True)
        variable_layout.addWidget(self.preserved_label)
        self._preserved_signature = (
            (current.kind, current.target) if current is not None else None
        )
        self._preserved_metadata = (
            dict(current.metadata) if current is not None else {}
        )
        known_variables = set(self.variable_checks)
        self._preserved_variables = (
            tuple(
                variable
                for variable in current.variables
                if variable.upper() not in known_variables
            )
            if current is not None
            else ()
        )
        if current is not None:
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData(current.kind))
            )
            self.target_combo.setCurrentIndex(
                max(0, self.target_combo.findData(current.target))
            )
            selected = {variable.upper() for variable in current.variables}
            for variable, check in self.variable_checks.items():
                check.setChecked(variable in selected)
        else:
            self.variable_checks["U"].setChecked(True)
            self.variable_checks["RF"].setChecked(True)
        self.kind_combo.setEnabled(
            current is None or current.kind == "field"
        )
        self.target_combo.setEnabled(
            current is None or current.target in {"node", "element"}
        )
        self.kind_combo.currentIndexChanged.connect(self._refresh_variables)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("分析步", self.step_combo)
        form.addRow("输出类型", self.kind_combo)
        form.addRow("输出位置", self.target_combo)
        form.addRow("输出变量", variable_widget)
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(_buttons(self)); self.setMinimumWidth(330)
        self._refresh_variables()

    def definition(self) -> tuple[str, OutputRequest]:
        step_name = self.step_combo.currentText().strip()
        if not step_name:
            raise ValueError("请选择分析步")
        kind = str(self.kind_combo.currentData())
        target = str(self.target_combo.currentData())
        variables = tuple(
            variable
            for variable, _label, variable_target in self._VARIABLES
            if variable_target == target
            and self.variable_checks[variable].isChecked()
        )
        if self._preserved_signature == (kind, target):
            variables += self._preserved_variables
            metadata = self._preserved_metadata
        else:
            metadata = {}
        if not variables:
            raise ValueError("至少选择一个输出变量")
        return step_name, OutputRequest(kind, target, variables, metadata)

    def _target_changed(self) -> None:
        target = str(self.target_combo.currentData())
        for variable, _label, variable_target in self._VARIABLES:
            self.variable_checks[variable].setChecked(
                variable_target == target
            )
        self._refresh_variables()

    def _refresh_variables(self) -> None:
        kind = str(self.kind_combo.currentData())
        target = str(self.target_combo.currentData())
        for variable, _label, variable_target in self._VARIABLES:
            self.variable_checks[variable].setVisible(
                kind == "field" and target == variable_target
            )
        preserved = (
            self._preserved_variables
            if self._preserved_signature == (kind, target)
            else ()
        )
        self.preserved_label.setText(
            "保留 INP 变量：" + "、".join(preserved)
            if preserved
            else ""
        )
        self.preserved_label.setVisible(bool(preserved))


class AnalysisDefinitionManagerDialog(QDialog):
    """Edit existing supported analysis definitions in one flat dialog."""

    def __init__(
        self,
        steps: list[AnalysisStep],
        node_regions: list[str],
        edge_regions: list[str],
        face_regions: list[str],
        dimensions: int,
        parent=None,
        *,
        spatial_dimensions: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("分析定义管理")
        self.steps = deepcopy(steps)
        self.node_regions = list(node_regions)
        self.edge_regions = list(edge_regions)
        self.face_regions = list(face_regions)
        self.dimensions = int(dimensions)
        self.spatial_dimensions = int(
            spatial_dimensions
            if spatial_dimensions is not None
            else min(dimensions, 3)
        )
        self._rows: list[tuple[str, int, int | None]] = []

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(
            ("类型", "分析步", "对象/区域", "参数")
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.edit_button = QPushButton("编辑", self)
        self.delete_button = QPushButton("删除", self)
        self.edit_button.clicked.connect(self._edit)
        self.delete_button.clicked.connect(self._delete)
        self.table.itemDoubleClicked.connect(lambda _item: self._edit())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        controls = QHBoxLayout()
        controls.addWidget(self.edit_button)
        controls.addWidget(self.delete_button)
        controls.addStretch(1)
        buttons = _buttons(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(controls)
        layout.addWidget(buttons)
        self.resize(640, 380)
        self._refresh()

    def _refresh(self, selected: int = 0) -> None:
        self.table.setRowCount(0)
        self._rows.clear()
        for step_index, step in enumerate(self.steps):
            self._append_row(
                (
                    "分析步",
                    step.name,
                    step.name,
                    "线性静力" if step.procedure == "static" else step.procedure,
                ),
                ("step", step_index, None),
            )
            for item_index, boundary in enumerate(step.boundaries):
                self._append_row(
                    (
                        "位移边界",
                        step.name,
                        str(boundary.target),
                        self._boundary_text(boundary),
                    ),
                    ("boundary", step_index, item_index),
                )
            for item_index, load in enumerate(step.cloads):
                self._append_row(
                    (
                        "节点力",
                        step.name,
                        str(load.target),
                        f"{dict(enumerate(('Fx', 'Fy', 'Fz'), 1)).get(load.component, f'F{load.component}')} = {load.value:g}",
                    ),
                    ("node_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.edge_loads):
                self._append_row(
                    (
                        "边载荷",
                        step.name,
                        load.edge,
                        self._distributed_text(load),
                    ),
                    ("edge_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.surface_loads):
                self._append_row(
                    (
                        "面载荷",
                        step.name,
                        load.surface,
                        self._distributed_text(load),
                    ),
                    ("surface_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.gravity_loads):
                self._append_row(
                    (
                        "重力",
                        step.name,
                        (
                            "整个模型"
                            if load.target is None
                            else str(load.target)
                        ),
                        self._gravity_text(load),
                    ),
                    ("gravity_load", step_index, item_index),
                )
            for item_index, output in enumerate(step.outputs):
                self._append_row(
                    (
                        {
                            "field": "字段输出",
                            "history": "历史输出",
                        }.get(output.kind, "输出请求"),
                        step.name,
                        {
                            "node": "节点",
                            "element": "单元",
                            "preselect": "INP 预选",
                        }.get(output.target, output.target),
                        "、".join(output.variables),
                    ),
                    ("output", step_index, item_index),
                )
        if self._rows:
            self.table.selectRow(min(selected, len(self._rows) - 1))
        self._update_buttons()

    def _append_row(
        self,
        values: tuple[str, str, str, str],
        key: tuple[str, int, int | None],
    ) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate(values):
            self.table.setItem(row, column, QTableWidgetItem(value))
        self._rows.append(key)

    @staticmethod
    def _distributed_text(load: EdgeLoad | SurfaceLoad) -> str:
        if load.load_type == "pressure":
            return f"压力 = {float(load.magnitude or 0.0):g}"
        return "牵引 = (" + ", ".join(f"{value:g}" for value in load.vector) + ")"

    @staticmethod
    def _gravity_text(load: GravityLoad) -> str:
        return "加速度 = (" + ", ".join(
            f"{value:g}" for value in load.acceleration
        ) + ")"

    @staticmethod
    def _boundary_text(boundary: DisplacementConstraint) -> str:
        component = (
            f"U{boundary.first_component}"
            if boundary.first_component == boundary.last_component
            else f"U{boundary.first_component}–U{boundary.last_component}"
        )
        return f"{component} = {boundary.value:g}"

    def _selected(self) -> tuple[str, int, int | None] | None:
        row = self.table.currentRow()
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def select_definition(
        self,
        key: tuple[str, int, int | None],
    ) -> bool:
        """Select one definition identified by the model-tree key."""
        try:
            row = self._rows.index(key)
        except ValueError:
            return False
        self.table.selectRow(row)
        return True

    def edit_definition(
        self,
        key: tuple[str, int, int | None],
    ) -> bool:
        """Open the existing parameter dialog and report a real change."""
        if not self.select_definition(key):
            return False
        previous = deepcopy(self.steps)
        self._edit()
        return self.steps != previous

    @staticmethod
    def _with_existing(values: list[str], existing: str) -> list[str]:
        return (
            list(values)
            if not existing or existing in values
            else [*values, existing]
        )

    def _edit(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        kind, step_index, item_index = selected
        step = self.steps[step_index]
        row = self.table.currentRow()
        if kind == "step":
            dialog = StaticStepDialog(step.name, self)
            if not dialog.exec():
                return
            try:
                updated = dialog.step()
            except ValueError as error:
                QMessageBox.warning(self, "分析定义", str(error))
                return
            if any(
                index != step_index
                and item.name.casefold() == updated.name.casefold()
                for index, item in enumerate(self.steps)
            ):
                QMessageBox.warning(
                    self,
                    "分析定义",
                    f"分析步名称已存在：{updated.name}",
                )
                return
            step.name = updated.name
        elif kind == "boundary":
            current = step.boundaries[int(item_index)]
            dialog = DisplacementDialog(
                [item.name for item in self.steps],
                self._with_existing(self.node_regions, str(current.target)),
                self.dimensions,
                self,
                current=current,
            )
            dialog.step_combo.setCurrentText(step.name)
            if not dialog.exec():
                return
            try:
                target_step, values = dialog.definitions()
            except ValueError as error:
                QMessageBox.warning(self, "分析定义", str(error))
                return
            step.boundaries = tuple(
                item
                for index, item in enumerate(step.boundaries)
                if index != item_index
            )
            self._step(target_step).boundaries = tuple(
                self._step(target_step).boundaries
            ) + values
        elif kind in {
            "node_load",
            "edge_load",
            "surface_load",
            "gravity_load",
        }:
            collection_name = {
                "node_load": "cloads",
                "edge_load": "edge_loads",
                "surface_load": "surface_loads",
                "gravity_load": "gravity_loads",
            }[kind]
            collection = tuple(getattr(step, collection_name))
            current = collection[int(item_index)]
            dialog = LoadDialog(
                [item.name for item in self.steps],
                self._with_existing(
                    self.node_regions,
                    str(getattr(current, "target", "")),
                ),
                self._with_existing(
                    self.edge_regions,
                    str(getattr(current, "edge", "")),
                ),
                self._with_existing(
                    self.face_regions,
                    str(getattr(current, "surface", "")),
                ),
                self.dimensions,
                self,
                spatial_dimensions=self.spatial_dimensions,
                current=current,
            )
            dialog.step_combo.setCurrentText(step.name)
            if not dialog.exec():
                return
            try:
                target_step, value = dialog.definition()
            except ValueError as error:
                QMessageBox.warning(self, "分析定义", str(error))
                return
            setattr(
                step,
                collection_name,
                tuple(
                    item
                    for index, item in enumerate(collection)
                    if index != item_index
                ),
            )
            self._append_load(self._step(target_step), value)
        else:
            current = step.outputs[int(item_index)]
            dialog = OutputRequestDialog(
                [item.name for item in self.steps],
                self,
                current=current,
            )
            dialog.step_combo.setCurrentText(step.name)
            if not dialog.exec():
                return
            try:
                target_step, value = dialog.definition()
            except ValueError as error:
                QMessageBox.warning(self, "分析定义", str(error))
                return
            step.outputs = tuple(
                item
                for index, item in enumerate(step.outputs)
                if index != item_index
            )
            target = self._step(target_step)
            target.outputs = tuple(target.outputs) + (value,)
        self._refresh(row)

    def _delete(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        kind, step_index, item_index = selected
        step = self.steps[step_index]
        if kind == "step":
            del self.steps[step_index]
        else:
            collection_name = {
                "boundary": "boundaries",
                "node_load": "cloads",
                "edge_load": "edge_loads",
                "surface_load": "surface_loads",
                "gravity_load": "gravity_loads",
                "output": "outputs",
            }[kind]
            collection = tuple(getattr(step, collection_name))
            setattr(
                step,
                collection_name,
                tuple(
                    item
                    for index, item in enumerate(collection)
                    if index != item_index
                ),
            )
        self._refresh(max(0, self.table.currentRow() - 1))

    def _step(self, name: str) -> AnalysisStep:
        return next(step for step in self.steps if step.name == name)

    @staticmethod
    def _append_load(
        step: AnalysisStep,
        load: NodalLoad | EdgeLoad | SurfaceLoad | GravityLoad,
    ) -> None:
        if isinstance(load, NodalLoad):
            step.cloads = tuple(step.cloads) + (load,)
        elif isinstance(load, EdgeLoad):
            step.edge_loads = tuple(step.edge_loads) + (load,)
        elif isinstance(load, SurfaceLoad):
            step.surface_loads = tuple(step.surface_loads) + (load,)
        else:
            step.gravity_loads = tuple(step.gravity_loads) + (load,)

    def _update_buttons(self) -> None:
        selected = self._selected() is not None
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def values(self) -> list[AnalysisStep]:
        return deepcopy(self.steps)
