"""Modal dialogs for the currently supported linear-static analysis inputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from math import isfinite

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

from fem.application import (
    AuthoringCapability,
    AuthoringStatus,
    RegionRef,
    require_region_kind,
)
from fem.application.results import OutputRequestProjection
from fem.core.immutable_json import thaw_json_mapping
from fem.core.model import (
    AnalysisStep,
    BodyForce,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    LineLoad,
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
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel,
        dialog,
    )
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
    buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    return buttons


def _select_region(
    combo: QComboBox,
    value: RegionRef | None,
) -> None:
    if value is None:
        return
    if type(value) is not RegionRef:
        raise TypeError("selected region must be RegionRef")
    for index in range(combo.count()):
        reference = combo.itemData(index)
        if reference == value:
            combo.setCurrentIndex(index)
            return


def _typed_regions(
    values: Sequence[RegionRef],
    expected_kind: str,
) -> tuple[RegionRef, ...]:
    result: list[RegionRef] = []
    for value in values:
        if type(value) is not RegionRef:
            raise TypeError("dialog regions must contain RegionRef values")
        require_region_kind(value, expected_kind)
        if value not in result:
            result.append(value)
    return tuple(result)


def _authoring_candidate_enabled(decision: AuthoringCapability) -> bool:
    """Allow candidate writes only for an explicit, non-blocking ENABLED result."""

    if type(decision) is not AuthoringCapability:
        raise TypeError("candidate decision must be AuthoringCapability")
    return decision.can_submit


def _authoring_candidate_message(decision: AuthoringCapability) -> str:
    """Render an application decision without reimplementing its policy."""

    if type(decision) is not AuthoringCapability:
        raise TypeError("candidate decision must be AuthoringCapability")
    diagnostics = decision.diagnostics
    if diagnostics:
        return "\n".join(
            (
                f"[{getattr(item, 'code', 'authoring.unavailable')}] "
                f"{getattr(item, 'message', str(item))}"
            )
            for item in diagnostics
        )
    return f"当前候选状态为 {decision.status.value}；只有 ENABLED 才可保存。"


class StaticStepDialog(QDialog):
    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("创建静力分析步")
        self.name_edit = QLineEdit(name, self)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("分析步名称", self.name_edit)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(_buttons(self))

    def step(self):
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("分析步名称不能为空")
        return static(name)


class DisplacementDialog(QDialog):
    _COMPONENT_LABELS = ("U1", "U2", "U3", "UR1", "UR2", "UR3")

    def __init__(
        self,
        step_names: list[str],
        regions: Sequence[RegionRef],
        dimensions: int,
        parent=None,
        *,
        selected_region: RegionRef | None = None,
        current: DisplacementConstraint | None = None,
        labels: Sequence[str] | None = None,
        allow_scope_selection: bool = False,
        scope_selection_kinds: Sequence[str] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("位移边界条件")
        supported_scope_kinds = frozenset(
            str(kind)
            for kind in scope_selection_kinds
            if str(kind) in {"node", "edge", "surface"}
        )
        if allow_scope_selection and not supported_scope_kinds:
            supported_scope_kinds = frozenset({"node"})
        self._scope_selection_kinds = supported_scope_kinds
        self._scope_selection_request: str | None = None
        self._regions = {
            kind: [
                reference
                for reference in regions
                if type(reference) is RegionRef and reference.kind == kind
            ]
            for kind in ("node_set", "edge", "surface")
        }
        if any(type(reference) is not RegionRef for reference in regions):
            raise TypeError("dialog regions must contain RegionRef values")
        if any(
            reference.kind not in self._regions
            for reference in regions
        ):
            raise ValueError(
                "displacement regions must be node_set, edge, or surface"
            )
        self.kind_combo = QComboBox(self)
        self.region_combo = QComboBox(self)
        self.scope_pick_button = QPushButton("创建", self)
        self.scope_pick_button.clicked.connect(
            self._request_scope_selection
        )
        region_widget = QWidget(self)
        region_layout = QHBoxLayout(region_widget)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.addWidget(self.region_combo, 1)
        region_layout.addWidget(self.scope_pick_button)
        self.step_combo = QComboBox(self)
        for kind, label, selection_kind in (
            ("node_set", "节点集", "node"),
            ("edge", "边", "edge"),
            ("surface", "面", "surface"),
        ):
            if self._regions[kind] or selection_kind in supported_scope_kinds:
                self.kind_combo.addItem(label, kind)
        self.step_combo.addItems(step_names)
        self.component_checks: dict[int, QCheckBox] = {}
        self.component_values: dict[int, QDoubleSpinBox] = {}
        component_labels = tuple(str(label) for label in labels or ())
        component_widget = QWidget(self)
        component_layout = QGridLayout(component_widget)
        component_layout.setContentsMargins(0, 0, 0, 0)
        component_layout.addWidget(QLabel("自由度", component_widget), 0, 0)
        component_layout.addWidget(QLabel("位移值", component_widget), 0, 1)
        for component in range(1, dimensions + 1):
            if component <= len(component_labels):
                label = component_labels[component - 1]
            elif component <= len(self._COMPONENT_LABELS):
                label = self._COMPONENT_LABELS[component - 1]
            else:
                label = f"U{component}"
            check = QCheckBox(label, component_widget)
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
            selected_region = RegionRef(
                getattr(current, "target_kind", "node_set"),
                str(current.target),
            )
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
        form.addRow("作用域类型", self.kind_combo)
        form.addRow("选择作用域", region_widget)
        form.addRow("分析步", self.step_combo)
        form.addRow("约束分量", component_widget)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        self.buttons = _buttons(self)
        self.buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setEnabled(False)
        layout.addWidget(self.buttons)
        self.setMinimumWidth(350)
        self.kind_combo.currentIndexChanged.connect(self._refresh_regions)
        self.region_combo.currentIndexChanged.connect(
            self._update_accept_state
        )
        if selected_region is not None:
            index = self.kind_combo.findData(selected_region.kind)
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        self._refresh_regions()
        _select_region(self.region_combo, selected_region)
        self._update_accept_state()

    def _request_scope_selection(self) -> None:
        kind = str(self.kind_combo.currentData() or "")
        request_kind = {
            "node_set": "node",
            "edge": "edge",
            "surface": "surface",
        }.get(kind)
        if request_kind not in self._scope_selection_kinds:
            return
        self._scope_selection_request = request_kind
        self.reject()

    def requested_scope_kind(self) -> str | None:
        return self._scope_selection_request

    def _refresh_regions(self) -> None:
        current = self.region_combo.currentData()
        kind = str(self.kind_combo.currentData() or "")
        self.region_combo.clear()
        for reference in self._regions.get(kind, ()):
            self.region_combo.addItem(reference.name, reference)
        if isinstance(current, RegionRef) and current.kind == kind:
            _select_region(self.region_combo, current)
        request_kind = {
            "node_set": "node",
            "edge": "edge",
            "surface": "surface",
        }.get(kind)
        self.scope_pick_button.setEnabled(
            request_kind in self._scope_selection_kinds
        )
        self._update_accept_state()

    def _update_accept_state(self) -> None:
        buttons = getattr(self, "buttons", None)
        if buttons is None:
            return
        buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setEnabled(isinstance(self.region_combo.currentData(), RegionRef))

    def definitions(self) -> tuple[str, tuple[DisplacementConstraint, ...]]:
        region = self.region_combo.currentData()
        if not isinstance(region, RegionRef):
            raise ValueError("请选择约束作用域")
        if region.kind not in {"node_set", "edge", "surface"}:
            raise ValueError("位移边界作用域必须是节点集、边或面")
        step_name = self.step_combo.currentText().strip()
        if not step_name:
            raise ValueError("请选择分析步")
        values = tuple(
            DisplacementConstraint(
                region.name,
                component,
                component,
                self.component_values[component].value(),
                target_kind=region.kind,
            )
            for component, check in self.component_checks.items()
            if check.isChecked()
        )
        if not values:
            raise ValueError("至少勾选一个位移自由度")
        return step_name, values

class LoadDialog(QDialog):
    _COMPONENT_LABELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

    def __init__(
        self,
        step_names: list[str],
        node_regions: Sequence[RegionRef],
        edge_regions: Sequence[RegionRef],
        face_regions: Sequence[RegionRef],
        dimensions: int,
        parent=None,
        *,
        spatial_dimensions: int | None = None,
        line_regions: Sequence[RegionRef] | None = None,
        body_regions: Sequence[RegionRef] | None = None,
        selected_region: RegionRef | None = None,
        preferred_kind: str | None = None,
        current: (
            NodalLoad
            | EdgeLoad
            | SurfaceLoad
            | LineLoad
            | BodyForce
            | GravityLoad
            | None
        ) = None,
        labels: Sequence[str] | None = None,
        candidate_evaluator: (
            Callable[[LineLoad, str], AuthoringCapability] | None
        ) = None,
        scope_selection_kinds: Sequence[str] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑载荷" if current is not None else "创建载荷")
        supported_scope_kinds = frozenset(
            str(kind)
            for kind in scope_selection_kinds
            if str(kind) in {"node", "edge", "surface", "line", "body"}
        )
        self._scope_selection_kinds = supported_scope_kinds
        self._scope_selection_request: str | None = None
        self._candidate_evaluator = candidate_evaluator
        self._candidate_signature: tuple[str, LineLoad] | None = None
        self._candidate_result: AuthoringCapability | None = None
        resolved_line_regions = list(
            _typed_regions(line_regions or (), "element_set")
        )
        if (
            isinstance(current, LineLoad)
            and str(current.target)
            not in {region.name for region in resolved_line_regions}
        ):
            resolved_line_regions.append(
                RegionRef("element_set", str(current.target))
            )
        self._regions = {
            "node": list(_typed_regions(node_regions, "node_set")),
            "edge": list(_typed_regions(edge_regions, "edge")),
            "surface": list(_typed_regions(face_regions, "surface")),
            "line": resolved_line_regions,
            "body": list(_typed_regions(body_regions or (), "element_set")),
        }
        self.dimensions = dimensions
        self.spatial_dimensions = int(
            spatial_dimensions if spatial_dimensions is not None else min(dimensions, 3)
        )
        self.kind_combo = QComboBox(self)
        self.region_combo = QComboBox(self)
        self.scope_pick_button = QPushButton("创建", self)
        self.scope_pick_button.clicked.connect(
            self._request_scope_selection
        )
        self.region_widget = QWidget(self)
        region_layout = QHBoxLayout(self.region_widget)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.addWidget(self.region_combo, 1)
        region_layout.addWidget(self.scope_pick_button)
        self.step_combo = QComboBox(self)
        if self._regions["node"] or "node" in supported_scope_kinds:
            self.kind_combo.addItem("节点力", "node")
        if self._regions["edge"] or "edge" in supported_scope_kinds:
            self.kind_combo.addItem("边力", "edge")
        if self._regions["surface"] or "surface" in supported_scope_kinds:
            self.kind_combo.addItem("面力", "surface")
        if resolved_line_regions or "line" in supported_scope_kinds:
            self.kind_combo.addItem("边力", "line")
        if self._regions["body"] or "body" in supported_scope_kinds:
            self.kind_combo.addItem("体力", "body")
        self.kind_combo.addItem("重力", "gravity")
        self._gravity_target = (
            current.target if isinstance(current, GravityLoad) else None
        )
        self.step_combo.addItems(step_names)
        self.load_type_combo = QComboBox(self)
        self.load_type_combo.addItem("牵引", "traction")
        self.load_type_combo.addItem("压力", "pressure")
        self.coordinate_system_combo = QComboBox(self)
        self.coordinate_system_combo.addItem("全局坐标系", "global")
        self.coordinate_system_combo.addItem(
            "局部（Beam 已解析局部坐标）",
            "local",
        )
        self.component_combo = QComboBox(self)
        component_labels = tuple(str(label) for label in labels or ())
        for component in range(1, dimensions + 1):
            label = (
                component_labels[component - 1]
                if component <= len(component_labels)
                else self._COMPONENT_LABELS[component - 1]
                if component <= len(self._COMPONENT_LABELS)
                else f"F{component}"
            )
            self.component_combo.addItem(label, component)
        self.value_spin = _value(self)
        self.x_spin = _value(self)
        self.y_spin = _value(self)
        self.z_spin = _value(self)
        zero_vector = (0.0,) * self.spatial_dimensions
        gravity_vector = list(zero_vector)
        gravity_vector[-1] = -9.81
        self._vector_values = {
            "edge": zero_vector,
            "surface": zero_vector,
            "line": (0.0, 0.0, 0.0),
            "body": zero_vector,
            "gravity": tuple(gravity_vector),
        }
        self._active_vector_kind: str | None = None
        self.form = QFormLayout()
        configure_form_layout(self.form)
        self.form.addRow("载荷类别", self.kind_combo)
        self.form.addRow("选择作用域", self.region_widget)
        self.form.addRow("分析步", self.step_combo)
        self.form.addRow("载荷形式", self.load_type_combo)
        self.form.addRow("坐标系", self.coordinate_system_combo)
        self.local_axis_label = QLabel(
            "局部（Beam 已解析局部坐标）",
            self,
        )
        self.local_axis_label.setWordWrap(True)
        self.form.addRow(self.local_axis_label)
        self.candidate_diagnostic_label = QLabel("", self)
        self.candidate_diagnostic_label.setWordWrap(True)
        self.form.addRow(self.candidate_diagnostic_label)
        self.form.addRow("分量", self.component_combo)
        self.form.addRow("载荷值", self.value_spin)
        self.form.addRow("Fx", self.x_spin)
        self.form.addRow("Fy", self.y_spin)
        self.form.addRow("Fz", self.z_spin)
        self.kind_combo.currentIndexChanged.connect(self._refresh)
        self.load_type_combo.currentIndexChanged.connect(self._refresh)
        self.coordinate_system_combo.currentIndexChanged.connect(self._refresh)
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
            selected_region = RegionRef("node_set", str(current.target))
        elif isinstance(current, EdgeLoad):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("edge"))
            )
            selected_region = RegionRef("edge", current.edge)
            self.load_type_combo.setCurrentIndex(
                max(0, self.load_type_combo.findData(current.load_type))
            )
            self._set_distributed_values(current.vector, current.magnitude)
            self._vector_values["edge"] = tuple(current.vector)
        elif isinstance(current, SurfaceLoad):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("surface"))
            )
            selected_region = RegionRef("surface", current.surface)
            self.load_type_combo.setCurrentIndex(
                max(0, self.load_type_combo.findData(current.load_type))
            )
            self._set_distributed_values(current.vector, current.magnitude)
            self._vector_values["surface"] = tuple(current.vector)
        elif isinstance(current, LineLoad):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("line"))
            )
            selected_region = RegionRef("element_set", str(current.target))
            coordinate_index = self.coordinate_system_combo.findData(
                current.coordinate_system
            )
            if coordinate_index >= 0:
                self.coordinate_system_combo.setCurrentIndex(coordinate_index)
            line_vector = tuple(current.vector[:3])
            line_vector += (0.0,) * (3 - len(line_vector))
            self._set_distributed_values(line_vector, None)
            self._vector_values["line"] = line_vector
        elif isinstance(current, BodyForce):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("body"))
            )
            selected_region = RegionRef(
                "element_set",
                str(current.target),
            )
            self._set_distributed_values(tuple(current.vector), None)
            self._vector_values["body"] = tuple(current.vector)
        elif isinstance(current, GravityLoad):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("gravity"))
            )
            self._set_distributed_values(
                tuple(current.acceleration),
                None,
            )
            self._vector_values["gravity"] = tuple(current.acceleration)
        layout = QVBoxLayout(self)
        layout.addLayout(self.form)
        self.buttons = _buttons(self)
        layout.addWidget(self.buttons)
        self.setMinimumWidth(350)
        self.region_combo.currentIndexChanged.connect(
            self._update_candidate_state
        )
        self.step_combo.currentIndexChanged.connect(
            self._update_candidate_state
        )
        for spin in (self.x_spin, self.y_spin, self.z_spin):
            spin.valueChanged.connect(self._update_candidate_state)
        self._refresh()
        _select_region(self.region_combo, selected_region)
        self._update_candidate_state()

    def _request_scope_selection(self) -> None:
        kind = str(self.kind_combo.currentData() or "")
        if kind not in self._scope_selection_kinds:
            return
        self._scope_selection_request = kind
        self.reject()

    def requested_scope_kind(self) -> str | None:
        return self._scope_selection_request

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
            vector_dimensions = (
                3
                if self._active_vector_kind == "line"
                else self.spatial_dimensions
            )
            self._vector_values[self._active_vector_kind] = tuple(
                spin.value()
                for spin in (self.x_spin, self.y_spin, self.z_spin)[
                    :vector_dimensions
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
        current = self.region_combo.currentData()
        self.region_combo.clear()
        if gravity:
            if self._gravity_target is not None:
                self.region_combo.addItem(
                    str(self._gravity_target),
                    self._gravity_target,
                )
        else:
            for reference in self._regions[kind]:
                self.region_combo.addItem(reference.name, reference)
        _select_region(self.region_combo, current)
        distributed = kind in {"edge", "surface"}
        pressure = self.load_type_combo.currentData() == "pressure"
        self.form.setRowVisible(
            self.region_widget,
            not gravity or self._gravity_target is not None,
        )
        self.region_combo.setEnabled(not gravity)
        self.scope_pick_button.setVisible(not gravity)
        self.scope_pick_button.setEnabled(
            not gravity and kind in self._scope_selection_kinds
        )
        self.form.setRowVisible(self.load_type_combo, distributed)
        line_load = kind == "line"
        self.form.setRowVisible(self.coordinate_system_combo, line_load)
        local_coordinates = (
            line_load
            and self.coordinate_system_combo.currentData() == "local"
        )
        self.form.setRowVisible(self.local_axis_label, local_coordinates)
        self.form.setRowVisible(
            self.candidate_diagnostic_label,
            local_coordinates,
        )
        self.form.setRowVisible(self.component_combo, kind == "node")
        self.form.setRowVisible(
            self.value_spin,
            kind == "node" or (distributed and pressure),
        )
        self.form.labelForField(self.value_spin).setText(
            "压力值" if pressure and distributed else "载荷值"
        )
        body_load = kind == "body"
        show_vector = (
            gravity
            or body_load
            or line_load
            or (distributed and not pressure)
        )
        if gravity:
            vector_labels = ("ax", "ay", "az")
        elif body_load:
            vector_labels = ("bx", "by", "bz")
        elif line_load:
            vector_labels = ("q1", "q2", "q3")
        else:
            vector_labels = ("Fx", "Fy", "Fz")
        for spin, label in zip(
            (self.x_spin, self.y_spin, self.z_spin),
            vector_labels,
        ):
            self.form.labelForField(spin).setText(label)
        self.form.setRowVisible(self.x_spin, show_vector)
        vector_dimensions = 3 if line_load else self.spatial_dimensions
        self.form.setRowVisible(
            self.y_spin,
            show_vector and vector_dimensions >= 2,
        )
        self.form.setRowVisible(
            self.z_spin,
            show_vector and vector_dimensions == 3,
        )
        self._update_candidate_state()

    def candidate_decision(
        self,
        candidate: LineLoad | None = None,
        step_name: str | None = None,
    ) -> AuthoringCapability:
        """Return the cached application decision for one local LineLoad."""

        if candidate is None:
            selected_step, selected = self.definition()
            if not isinstance(selected, LineLoad):
                raise ValueError("candidate is not a LineLoad")
            candidate = selected
            step_name = selected_step
        if candidate.coordinate_system != "local":
            raise ValueError("candidate is not a local LineLoad")
        step = str(
            self.step_combo.currentText()
            if step_name is None
            else step_name
        ).strip()
        signature = (step, candidate)
        if signature == self._candidate_signature:
            return self._candidate_result
        self._candidate_signature = signature
        if self._candidate_evaluator is None:
            raise RuntimeError("line-load candidate evaluator is required")
        self._candidate_result = self._candidate_evaluator(candidate, step)
        if type(self._candidate_result) is not AuthoringCapability:
            raise TypeError(
                "line-load candidate evaluator must return AuthoringCapability"
            )
        return self._candidate_result

    def _update_candidate_state(self) -> None:
        if not hasattr(self, "buttons"):
            return
        local_line_load = (
            self.kind_combo.currentData() == "line"
            and self.coordinate_system_combo.currentData() == "local"
        )
        ok_button = self.buttons.button(
            QDialogButtonBox.StandardButton.Ok
        )
        target_available = (
            self.kind_combo.currentData() == "gravity"
            or isinstance(self.region_combo.currentData(), RegionRef)
        )
        if not local_line_load:
            self.candidate_diagnostic_label.clear()
            self.candidate_diagnostic_label.setVisible(False)
            ok_button.setEnabled(target_available)
            return
        self.candidate_diagnostic_label.setVisible(True)
        try:
            step_name, candidate = self.definition()
            decision = self.candidate_decision(candidate, step_name)
        except (KeyError, TypeError, ValueError) as error:
            self.candidate_diagnostic_label.setText(str(error))
            ok_button.setEnabled(False)
            return
        enabled = _authoring_candidate_enabled(decision)
        self.candidate_diagnostic_label.setText(
            ""
            if enabled
            else _authoring_candidate_message(decision)
        )
        ok_button.setEnabled(target_available and enabled)

    def accept(self) -> None:
        try:
            step_name, candidate = self.definition()
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "载荷", str(error))
            return
        if (
            isinstance(candidate, LineLoad)
            and candidate.coordinate_system == "local"
        ):
            decision = self.candidate_decision(candidate, step_name)
            if not _authoring_candidate_enabled(decision):
                QMessageBox.warning(
                    self,
                    "边力",
                    _authoring_candidate_message(decision),
                )
                return
        super().accept()

    def definition(self):
        kind = str(self.kind_combo.currentData())
        step = self.step_combo.currentText().strip()
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
        region = self.region_combo.currentData()
        if not isinstance(region, RegionRef):
            raise ValueError("请选择载荷作用域")
        expected_kind = {
            "node": "node_set",
            "edge": "edge",
            "surface": "surface",
            "line": "element_set",
            "body": "element_set",
        }.get(kind)
        if expected_kind is None:
            raise ValueError("当前没有可用的载荷作用域")
        target = require_region_kind(region, expected_kind)
        if kind == "node":
            component = self.component_combo.currentData()
            if component is None:
                raise ValueError("请选择节点力分量")
            return step, NodalLoad(
                target,
                int(component),
                self.value_spin.value(),
            )
        if kind == "line":
            vector = tuple(
                spin.value()
                for spin in (self.x_spin, self.y_spin, self.z_spin)
            )
            if len(vector) != 3 or not all(isfinite(value) for value in vector):
                raise ValueError("梁单元边力必须包含三个有限分量")
            coordinate_system = str(
                self.coordinate_system_combo.currentData() or ""
            )
            if coordinate_system not in {"global", "local"}:
                raise ValueError("梁单元边力坐标系只能为 global 或 local")
            return step, LineLoad(
                target,
                vector,
                coordinate_system=coordinate_system,
            )
        if kind == "body":
            vector = tuple(
                spin.value()
                for spin in (self.x_spin, self.y_spin, self.z_spin)[
                    :self.spatial_dimensions
                ]
            )
            if len(vector) != self.spatial_dimensions or not all(
                isfinite(value) for value in vector
            ):
                raise ValueError(
                    "体力必须包含与空间维数一致的有限分量"
                )
            return step, BodyForce(target, vector)
        load_type = str(self.load_type_combo.currentData())
        if load_type == "pressure":
            return step, (EdgeLoad(target, magnitude=self.value_spin.value(), load_type="pressure") if kind == "edge" else SurfaceLoad(target, magnitude=self.value_spin.value(), load_type="pressure"))
        vector = tuple(value.value() for value in (self.x_spin, self.y_spin))
        if self.spatial_dimensions == 3:
            vector += (self.z_spin.value(),)
        return step, (EdgeLoad(target, vector, load_type="traction") if kind == "edge" else SurfaceLoad(target, vector, load_type="traction"))


class OutputRequestDialog(QDialog):
    def __init__(
        self,
        step_names: list[str],
        parent=None,
        *,
        candidates: Sequence[OutputRequestProjection] = (),
        current: OutputRequest | None = None,
    ) -> None:
        super().__init__(parent)
        if any(type(name) is not str or not name.strip() for name in step_names):
            raise TypeError("step_names must contain nonblank strings")
        candidate_values = tuple(candidates)
        if any(
            type(candidate) is not OutputRequestProjection
            for candidate in candidate_values
        ):
            raise TypeError(
                "candidates must contain OutputRequestProjection values"
            )
        if any(not candidate.executable for candidate in candidate_values):
            raise ValueError("output request candidates must be executable")
        if current is not None and type(current) is not OutputRequest:
            raise TypeError("current must be exactly OutputRequest or None")
        if current is not None and candidate_values:
            raise ValueError(
                "read-only output request views cannot accept candidates"
            )

        self._candidates = deepcopy(candidate_values)
        self._current = deepcopy(current)
        self.setWindowTitle(
            "查看输出请求" if current is not None else "创建输出请求"
        )
        self.step_combo = QComboBox(self)
        self.step_combo.addItems(step_names)
        self.candidate_combo = QComboBox(self)
        for index, candidate in enumerate(self._candidates):
            self.candidate_combo.addItem(
                _output_request_summary(candidate.authoring_request),
                index,
            )
        self.kind_value = QLabel("", self)
        self.target_value = QLabel("", self)
        self.variables_value = QLabel("", self)
        self.metadata_value = QLabel("", self)
        self.source_evidence_value = QLabel("", self)
        for label in (
            self.variables_value,
            self.metadata_value,
            self.source_evidence_value,
        ):
            label.setWordWrap(True)

        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("分析步", self.step_combo)
        if self._current is None:
            form.addRow("候选输出", self.candidate_combo)
        form.addRow("输出类型", self.kind_value)
        form.addRow("输出位置", self.target_value)
        form.addRow("输出变量", self.variables_value)
        form.addRow("元数据", self.metadata_value)
        form.addRow("源证据", self.source_evidence_value)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        if self._current is None:
            layout.addWidget(_buttons(self))
            self.candidate_combo.currentIndexChanged.connect(
                self._refresh_request
            )
        else:
            self.step_combo.setEnabled(False)
            self.candidate_combo.setVisible(False)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Close,
                self,
            )
            buttons.button(
                QDialogButtonBox.StandardButton.Close
            ).setText("关闭")
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)
        self.setMinimumWidth(330)
        self._refresh_request()

    def definition(self) -> tuple[str, OutputRequest]:
        step_name = self.step_combo.currentText().strip()
        if not step_name:
            raise ValueError("请选择分析步")
        if self._current is not None:
            return step_name, deepcopy(self._current)
        index = self.candidate_combo.currentData()
        if type(index) is not int or not 0 <= index < len(self._candidates):
            raise ValueError("请选择受支持的输出请求")
        request = deepcopy(self._candidates[index].authoring_request)
        if type(request) is not OutputRequest:
            raise TypeError(
                "candidate authoring_request must be exactly OutputRequest"
            )
        return step_name, request

    def _refresh_request(self) -> None:
        request = self._selected_request()
        self.kind_value.setText("" if request is None else request.kind)
        self.target_value.setText("" if request is None else request.target)
        self.variables_value.setText(
            ""
            if request is None
            else "、".join(request.variables)
        )
        self.metadata_value.setText(
            ""
            if request is None
            else _output_metadata_text(request)
        )
        self.source_evidence_value.setText(
            ""
            if request is None
            else (
                "—"
                if request.source_evidence is None
                else repr(request.source_evidence)
            )
        )

    def _selected_request(self) -> OutputRequest | None:
        if self._current is not None:
            return self._current
        index = self.candidate_combo.currentData()
        if type(index) is not int or not 0 <= index < len(self._candidates):
            return None
        return self._candidates[index].authoring_request


def _output_request_summary(request: OutputRequest) -> str:
    if type(request) is not OutputRequest:
        raise TypeError("request must be exactly OutputRequest")
    values = [request.kind, request.target, "、".join(request.variables)]
    metadata = _output_metadata_text(request)
    if metadata != "—":
        values.append(metadata)
    return " · ".join(values)


def _output_metadata_text(request: OutputRequest) -> str:
    metadata = thaw_json_mapping(request.metadata)
    if not metadata:
        return "—"
    return "；".join(
        f"{key}={value}"
        for key, value in metadata.items()
    )


class AnalysisDefinitionManagerDialog(QDialog):
    """Edit existing supported analysis definitions in one flat dialog."""

    def __init__(
        self,
        steps: list[AnalysisStep],
        node_regions: Sequence[RegionRef],
        edge_regions: Sequence[RegionRef],
        face_regions: Sequence[RegionRef],
        dimensions: int,
        parent=None,
        *,
        spatial_dimensions: int | None = None,
        line_regions: Sequence[RegionRef] | None = None,
        body_regions: Sequence[RegionRef] | None = None,
        boundary_regions: Sequence[RegionRef] | None = None,
        dof_labels: Sequence[str] | None = None,
        force_labels: Sequence[str] | None = None,
        candidate_evaluator: Callable[..., AuthoringCapability] | None = None,
        output_view_capability: AuthoringCapability | None = None,
        output_delete_capability: AuthoringCapability | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("分析定义管理")
        self.steps = deepcopy(steps)
        self.node_regions = list(_typed_regions(node_regions, "node_set"))
        self.edge_regions = list(_typed_regions(edge_regions, "edge"))
        self.face_regions = list(_typed_regions(face_regions, "surface"))
        self.line_regions = list(
            _typed_regions(line_regions or (), "element_set")
        )
        self.body_regions = list(
            _typed_regions(body_regions or (), "element_set")
        )
        raw_boundary_regions = (
            boundary_regions
            if boundary_regions is not None
            else (*self.node_regions, *self.edge_regions, *self.face_regions)
        )
        self.boundary_regions: list[RegionRef] = []
        for reference in raw_boundary_regions:
            if type(reference) is not RegionRef:
                raise TypeError(
                    "boundary regions must contain RegionRef values"
                )
            if reference.kind not in {"node_set", "edge", "surface"}:
                raise ValueError(
                    "boundary regions must be node_set, edge, or surface"
                )
            if reference not in self.boundary_regions:
                self.boundary_regions.append(reference)
        self.dimensions = int(dimensions)
        self.dof_labels = tuple(str(label) for label in dof_labels or ())
        self.force_labels = tuple(
            str(label) for label in force_labels or ()
        )
        self._candidate_evaluator = candidate_evaluator
        self._output_view_capability = _manager_output_capability(
            output_view_capability,
            operation="output_request.view",
            default_status=AuthoringStatus.READ_ONLY,
        )
        self._output_delete_capability = _manager_output_capability(
            output_delete_capability,
            operation="output_request.delete",
            default_status=AuthoringStatus.ENABLED,
        )
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
                        f"{self._force_label(load.component)} = {load.value:g}",
                    ),
                    ("node_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.edge_loads):
                self._append_row(
                    (
                        "边力",
                        step.name,
                        load.edge,
                        self._distributed_text(load),
                    ),
                    ("edge_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.surface_loads):
                self._append_row(
                    (
                        "面力",
                        step.name,
                        load.surface,
                        self._distributed_text(load),
                    ),
                    ("surface_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.line_loads):
                self._append_row(
                    (
                        "边力",
                        step.name,
                        str(load.target),
                        self._line_load_text(load),
                    ),
                    ("line_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.body_loads):
                self._append_row(
                    (
                        "体力",
                        step.name,
                        str(load.target),
                        self._body_force_text(load),
                    ),
                    ("body_load", step_index, item_index),
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
    def _body_force_text(load: BodyForce) -> str:
        return "力密度 = (" + ", ".join(
            f"{value:g}" for value in load.vector
        ) + ")"

    @staticmethod
    def _line_load_text(load: LineLoad) -> str:
        coordinate_system = {
            "global": "全局",
            "local": "局部（Beam 已解析局部坐标）",
        }.get(load.coordinate_system, load.coordinate_system)
        return coordinate_system + " = (" + ", ".join(
            f"{value:g}" for value in load.vector
        ) + ")"

    @staticmethod
    def _label(
        component: int,
        labels: Sequence[str],
        defaults: Sequence[str],
        fallback_prefix: str,
    ) -> str:
        if 1 <= component <= len(labels):
            return str(labels[component - 1])
        if 1 <= component <= len(defaults):
            return str(defaults[component - 1])
        return f"{fallback_prefix}{component}"

    def _dof_label(self, component: int) -> str:
        return self._label(
            component,
            self.dof_labels,
            DisplacementDialog._COMPONENT_LABELS,
            "U",
        )

    def _force_label(self, component: int) -> str:
        return self._label(
            component,
            self.force_labels,
            LoadDialog._COMPONENT_LABELS,
            "F",
        )

    def _boundary_text(self, boundary: DisplacementConstraint) -> str:
        component = (
            self._dof_label(boundary.first_component)
            if boundary.first_component == boundary.last_component
            else (
                f"{self._dof_label(boundary.first_component)}–"
                f"{self._dof_label(boundary.last_component)}"
            )
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
    def _with_existing(
        values: Sequence[RegionRef],
        existing: RegionRef | None,
    ) -> list[RegionRef]:
        if any(type(value) is not RegionRef for value in values):
            raise TypeError("manager regions must contain RegionRef values")
        if existing is not None and type(existing) is not RegionRef:
            raise TypeError("existing manager region must be RegionRef")
        return (
            list(values)
            if existing is None or existing in values
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
            current_region = RegionRef(
                getattr(current, "target_kind", "node_set"),
                str(current.target),
            )
            dialog = DisplacementDialog(
                [item.name for item in self.steps],
                self._with_existing(
                    self.boundary_regions,
                    current_region,
                ),
                self.dimensions,
                self,
                selected_region=current_region,
                current=current,
                labels=self.dof_labels,
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
            "line_load",
            "body_load",
            "gravity_load",
        }:
            collection_name = {
                "node_load": "cloads",
                "edge_load": "edge_loads",
                "surface_load": "surface_loads",
                "line_load": "line_loads",
                "body_load": "body_loads",
                "gravity_load": "gravity_loads",
            }[kind]
            collection = tuple(getattr(step, collection_name))
            current = collection[int(item_index)]
            dialog = LoadDialog(
                [item.name for item in self.steps],
                self._with_existing(
                    self.node_regions,
                    RegionRef("node_set", str(current.target))
                    if kind == "node_load"
                    else None,
                ),
                self._with_existing(
                    self.edge_regions,
                    RegionRef("edge", str(current.edge))
                    if kind == "edge_load"
                    else None,
                ),
                self._with_existing(
                    self.face_regions,
                    RegionRef("surface", str(current.surface))
                    if kind == "surface_load"
                    else None,
                ),
                self.dimensions,
                self,
                spatial_dimensions=self.spatial_dimensions,
                line_regions=self._with_existing(
                    self.line_regions,
                    RegionRef("element_set", str(current.target))
                    if kind == "line_load"
                    else None,
                ),
                body_regions=self._with_existing(
                    self.body_regions,
                    RegionRef("element_set", str(current.target))
                    if kind == "body_load"
                    else None,
                ),
                current=current,
                labels=self.force_labels,
                candidate_evaluator=(
                    None
                    if self._candidate_evaluator is None
                    else (
                        lambda candidate, target_step,
                        source_step=step.name,
                        source_index=int(item_index):
                        self._candidate_evaluator(
                            candidate,
                            target_step,
                            candidate_index=(
                                source_index
                                if target_step == source_step
                                else None
                            ),
                        )
                    )
                ),
            )
            dialog.step_combo.setCurrentText(step.name)
            if not dialog.exec():
                return
            try:
                target_step, value = dialog.definition()
            except ValueError as error:
                QMessageBox.warning(self, "分析定义", str(error))
                return
            if (
                isinstance(value, LineLoad)
                and value.coordinate_system == "local"
            ):
                decision = dialog.candidate_decision(value, target_step)
                if not _authoring_candidate_enabled(decision):
                    QMessageBox.warning(
                        self,
                        "分析定义",
                        _authoring_candidate_message(decision),
                    )
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
            if not self._output_view_capability.can_enter:
                return
            dialog = OutputRequestDialog(
                [item.name for item in self.steps],
                self,
                current=current,
            )
            dialog.step_combo.setCurrentText(step.name)
            dialog.exec()
            return
        self._refresh(row)

    def _delete(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        kind, step_index, item_index = selected
        step = self.steps[step_index]
        initial_output_owner = (
            step.name.strip().casefold() == "initial"
            and bool(step.outputs)
        )
        if (
            kind == "output"
            and not self._output_delete_capability.can_submit
        ) or (
            kind in {"step", "output"}
            and initial_output_owner
        ):
            return
        if kind == "step":
            del self.steps[step_index]
        else:
            collection_name = {
                "boundary": "boundaries",
                "node_load": "cloads",
                "edge_load": "edge_loads",
                "surface_load": "surface_loads",
                "line_load": "line_loads",
                "body_load": "body_loads",
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
        load: (
            NodalLoad
            | EdgeLoad
            | SurfaceLoad
            | LineLoad
            | BodyForce
            | GravityLoad
        ),
    ) -> None:
        if isinstance(load, NodalLoad):
            step.cloads = tuple(step.cloads) + (load,)
        elif isinstance(load, EdgeLoad):
            step.edge_loads = tuple(step.edge_loads) + (load,)
        elif isinstance(load, SurfaceLoad):
            step.surface_loads = tuple(step.surface_loads) + (load,)
        elif isinstance(load, LineLoad):
            step.line_loads = tuple(step.line_loads) + (load,)
        elif isinstance(load, BodyForce):
            step.body_loads = tuple(step.body_loads) + (load,)
        else:
            step.gravity_loads = tuple(step.gravity_loads) + (load,)

    def _update_buttons(self) -> None:
        selected = self._selected()
        is_output = selected is not None and selected[0] == "output"
        deletes_initial_outputs = (
            selected is not None
            and selected[0] in {"step", "output"}
            and self.steps[selected[1]].name.strip().casefold()
            == "initial"
            and bool(self.steps[selected[1]].outputs)
        )
        output_step_is_editable = (
            is_output
            and self.steps[selected[1]].name.strip().casefold()
            != "initial"
        )
        self.edit_button.setEnabled(
            selected is not None
            and (
                not is_output
                or self._output_view_capability.can_enter
            )
        )
        self.edit_button.setText(
            "查看"
            if is_output
            else "编辑"
        )
        self.delete_button.setEnabled(
            selected is not None
            and not deletes_initial_outputs
            and (
                not is_output
                or (
                    output_step_is_editable
                    and self._output_delete_capability.can_submit
                )
            )
        )

    def values(self) -> list[AnalysisStep]:
        return deepcopy(self.steps)


def _manager_output_capability(
    value: AuthoringCapability | None,
    *,
    operation: str,
    default_status: AuthoringStatus,
) -> AuthoringCapability:
    capability = (
        AuthoringCapability(operation, default_status)
        if value is None
        else value
    )
    if type(capability) is not AuthoringCapability:
        raise TypeError(
            f"{operation} capability must be AuthoringCapability"
        )
    if capability.operation != operation:
        raise ValueError(
            f"expected {operation} capability, got {capability.operation}"
        )
    return capability
