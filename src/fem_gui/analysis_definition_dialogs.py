"""Modal dialogs for the supported analysis inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

from PySide6.QtCore import Qt, Signal
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
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fem.application import (
    AuthoringCapability,
    AuthoringStatus,
    RegionRef,
    require_region_kind,
)
from fem.results import OutputRequestProjection
from fem.analysis import is_nlgeom_enabled
from fem.model import (
    AnalysisStep,
    BodyForce,
    DisplacementConstraint,
    DampingModel,
    DynamicIntegrationMethod,
    DynamicProcedureKind,
    DynamicStepControls,
    GeometryMode,
    EdgeLoad,
    GravityLoad,
    InitialConditionSet,
    LineLoad,
    MassMatrixPolicy,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
    TimeAmplitude,
)
from fem.model.authoring import (
    static,
    transient_dynamic,
    update_dynamic_step,
    update_static_step,
)
from fem.model import (
    NewtonStrategy,
    StaticControlMode,
    StaticFormulation,
    StaticStepControls,
)

from .dialogs import AdaptivePrecisionDoubleSpinBox, configure_form_layout
from .analysis_presentation import analysis_step_label


_SCOPE_NOT_REPORTED = object()


def _value(parent: QDialog, value: float = 0.0) -> QDoubleSpinBox:
    box = AdaptivePrecisionDoubleSpinBox(parent)
    box.setRange(-1.0e15, 1.0e15)
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
    def __init__(
        self,
        name: str,
        parent=None,
        *,
        current: AnalysisStep | None = None,
        nlgeom: bool = False,
        controls: StaticStepControls | None = None,
    ) -> None:
        """Edit the public static-step settings.

        The constitutive material is selected by the model definition.  This
        dialog owns only the analysis-step controls and never exposes an
        implementation-specific return-mapping name.
        """
        super().__init__(parent)
        self._procedure = "static"
        self._original_step: AnalysisStep | None = None
        self._initial_conditions = InitialConditionSet()
        self._dynamic_controls = DynamicStepControls()
        self._dynamic_procedure_kind = DynamicProcedureKind.IMPLICIT
        if current is not None:
            if type(current) is not AnalysisStep:
                raise TypeError("current must be exactly AnalysisStep or None")
            self._original_step = deepcopy(current)
            self._procedure = (
                "dynamic"
                if str(current.procedure).strip().casefold() == "dynamic"
                else "static"
            )
            if self._procedure == "dynamic":
                dynamic_controls = current.controls
                if not isinstance(dynamic_controls, DynamicStepControls):
                    dynamic_controls = DynamicStepControls.from_metadata(
                        current.metadata
                    )
                self._dynamic_controls = dynamic_controls
                self._dynamic_procedure_kind = dynamic_controls.procedure_kind
                self._initial_conditions = current.initial_conditions
                nlgeom = (
                    current.formulation is StaticFormulation.NONLINEAR
                    or current.geometry_mode is GeometryMode.FINITE_STRAIN
                )
                controls = None
            else:
                nlgeom = is_nlgeom_enabled(current)
                controls = current.controls
                if controls is None and nlgeom:
                    controls = StaticStepControls.from_metadata(current.metadata)
        else:
            self._original_step = None
        if controls is None:
            controls = StaticStepControls()
        if type(controls) is not StaticStepControls:
            raise TypeError("controls must be StaticStepControls or None")
        self.controls = controls
        self.name_edit = QLineEdit(name, self)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("分析步名称", self.name_edit)
        self.procedure_combo = QComboBox(self)
        self.procedure_combo.addItem("线性静力", "static")
        # Keep the historical ``dynamic`` data value for implicit dynamics so
        # project/UI callers that only know the original dynamic entry still
        # select the same procedure.  The visible choices now mirror the
        # Abaqus split between implicit and explicit direct integration.
        self.procedure_combo.addItem("动力学-隐式", "dynamic")
        self.procedure_combo.addItem("动力学-显式", "dynamic_explicit")
        selected_procedure = self._procedure
        if self._procedure == "dynamic" and (
            self._dynamic_procedure_kind is DynamicProcedureKind.EXPLICIT
        ):
            selected_procedure = "dynamic_explicit"
        self.procedure_combo.setCurrentIndex(
            self.procedure_combo.findData(selected_procedure)
        )
        form.addRow("分析类型", self.procedure_combo)
        self.nlgeom_check = QCheckBox("启用几何非线性（NLGEOM）", self)
        self.nlgeom_check.setChecked(bool(nlgeom))
        form.addRow("几何非线性", self.nlgeom_check)
        self.initial_increment_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.initial_increment_spin.setRange(1.0e-12, 1.0)
        self.initial_increment_spin.setValue(controls.initial_increment)
        self.initial_increment_spin.setToolTip(
            "从当前步起始状态开始的载荷因子增量。"
        )
        self.maximum_increment_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.maximum_increment_spin.setRange(1.0e-12, 1.0)
        self.maximum_increment_spin.setValue(controls.maximum_increment)
        self.maximum_increment_spin.setToolTip(
            "自动增长时允许使用的最大载荷因子增量。"
        )
        self.maximum_increments_spin = QSpinBox(self)
        self.maximum_increments_spin.setRange(1, 1_000_000)
        self.maximum_increments_spin.setValue(controls.maximum_increments)
        self.maximum_increments_spin.setToolTip(
            "整个分析步允许的最大增量数。"
        )
        self.minimum_increment_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.minimum_increment_spin.setRange(1.0e-16, 1.0)
        self.minimum_increment_spin.setValue(controls.minimum_increment)
        self.minimum_increment_spin.setToolTip(
            "自动切步允许的最小载荷因子增量。"
        )
        self.growth_factor_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.growth_factor_spin.setRange(1.0000001, 10.0)
        self.growth_factor_spin.setValue(controls.growth_factor)
        self.growth_factor_spin.setToolTip(
            "每次满足增长条件时，下一增量相对于当前增量的精确倍数。"
        )
        self.growth_iteration_threshold_spin = QSpinBox(self)
        self.growth_iteration_threshold_spin.setRange(1, 100_000)
        self.growth_iteration_threshold_spin.setValue(
            controls.growth_iteration_threshold
        )
        self.incrementation_mode_combo = QComboBox(self)
        self.incrementation_mode_combo.addItem("固定增量", "fixed")
        self.incrementation_mode_combo.addItem("自动增量", "automatic")
        mode = (
            "automatic"
            if controls.automatic_cutback or controls.adaptive_growth
            else "fixed"
        )
        self.incrementation_mode_combo.setCurrentIndex(
            self.incrementation_mode_combo.findData(mode)
        )
        self.incrementation_mode_combo.setToolTip(
            "固定增量只按初始增量推进；自动增量允许失败切回和收敛增长。"
        )
        self.control_mode_combo = QComboBox(self)
        for mode, label in (
            (StaticControlMode.LOAD, "载荷控制"),
            (StaticControlMode.DISPLACEMENT, "位移控制"),
            (StaticControlMode.MIXED, "混合控制"),
        ):
            self.control_mode_combo.addItem(label, mode.value)
        mode_index = self.control_mode_combo.findData(
            controls.control_mode.value
        )
        self.control_mode_combo.setCurrentIndex(max(0, mode_index))
        self.control_mode_combo.setToolTip(
            "决定增量因子作用于外载、位移约束或两者。"
        )
        self.newton_strategy_combo = QComboBox(self)
        for strategy, label in (
            (NewtonStrategy.FULL, "Full Newton（每次更新切线）"),
            (NewtonStrategy.MODIFIED, "Modified Newton（增量内复用切线）"),
        ):
            self.newton_strategy_combo.addItem(label, strategy.value)
        strategy_index = self.newton_strategy_combo.findData(
            controls.newton_strategy.value
        )
        self.newton_strategy_combo.setCurrentIndex(max(0, strategy_index))
        self.line_search_check = QCheckBox("启用 Newton 线搜索", self)
        self.line_search_check.setChecked(controls.line_search)
        self.predictor_check = QCheckBox("启用增量预测", self)
        self.predictor_check.setChecked(controls.predictor)
        self.automatic_cutback_check = QCheckBox("失败时自动切步", self)
        self.automatic_cutback_check.setChecked(controls.automatic_cutback)
        self.adaptive_growth_check = QCheckBox("收敛较快时自动放大增量", self)
        self.adaptive_growth_check.setChecked(controls.adaptive_growth)
        self.newton_max_iterations_spin = QSpinBox(self)
        self.newton_max_iterations_spin.setRange(1, 100_000)
        self.newton_max_iterations_spin.setValue(
            controls.newton_max_iterations
        )
        self.residual_tolerance_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.residual_tolerance_spin.setRange(1.0e-16, 1.0e15)
        self.residual_tolerance_spin.setValue(controls.residual_tolerance)
        self.residual_tolerance_spin.setToolTip(
            "Newton 收敛的绝对残差容差。"
        )
        self.relative_residual_tolerance_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.relative_residual_tolerance_spin.setRange(1.0e-16, 1.0e15)
        self.relative_residual_tolerance_spin.setValue(
            controls.relative_residual_tolerance
        )
        self.displacement_tolerance_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.displacement_tolerance_spin.setRange(1.0e-16, 1.0e15)
        self.displacement_tolerance_spin.setValue(
            controls.displacement_tolerance
        )
        self.energy_tolerance_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.energy_tolerance_spin.setRange(1.0e-16, 1.0e15)
        self.energy_tolerance_spin.setValue(controls.energy_tolerance)
        self.constraint_tolerance_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.constraint_tolerance_spin.setRange(1.0e-16, 1.0e15)
        self.constraint_tolerance_spin.setValue(controls.constraint_tolerance)

        basic_form = QFormLayout()
        configure_form_layout(basic_form)
        basic_form.addRow("控制方式", self.control_mode_combo)
        basic_form.addRow("增量方式", self.incrementation_mode_combo)
        basic_page = QWidget(self)
        basic_page.setLayout(basic_form)

        increment_form = QFormLayout()
        configure_form_layout(increment_form)
        increment_form.addRow("初始增量", self.initial_increment_spin)
        increment_form.addRow("最小增量", self.minimum_increment_spin)
        increment_form.addRow("最大增量", self.maximum_increment_spin)
        increment_form.addRow("最大增量数", self.maximum_increments_spin)
        increment_form.addRow("失败时自动切步", self.automatic_cutback_check)
        increment_form.addRow("收敛后自动增长", self.adaptive_growth_check)
        increment_form.addRow("增长倍数", self.growth_factor_spin)
        increment_form.addRow(
            "增长触发迭代数",
            self.growth_iteration_threshold_spin,
        )
        increment_page = QWidget(self)
        increment_page.setLayout(increment_form)

        newton_form = QFormLayout()
        configure_form_layout(newton_form)
        newton_form.addRow("最大迭代次数", self.newton_max_iterations_spin)
        newton_form.addRow("Newton 策略", self.newton_strategy_combo)
        newton_form.addRow("线搜索", self.line_search_check)
        newton_form.addRow("增量预测", self.predictor_check)
        newton_form.addRow("绝对残差容差", self.residual_tolerance_spin)
        newton_form.addRow(
            "相对残差容差",
            self.relative_residual_tolerance_spin,
        )
        newton_form.addRow("位移修正容差", self.displacement_tolerance_spin)
        newton_form.addRow("能量容差", self.energy_tolerance_spin)
        newton_form.addRow("约束容差", self.constraint_tolerance_spin)
        newton_page = QWidget(self)
        newton_page.setLayout(newton_form)

        settings_tabs = QTabWidget(self)
        settings_tabs.addTab(basic_page, "基本")
        settings_tabs.addTab(increment_page, "增量控制")
        settings_tabs.addTab(newton_page, "Newton 与收敛")

        dynamic_controls = self._dynamic_controls
        dynamic_form = QFormLayout()
        configure_form_layout(dynamic_form)
        self.dynamic_time_period_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.dynamic_time_period_spin.setRange(1.0e-12, 1.0e15)
        self.dynamic_time_period_spin.setValue(dynamic_controls.time_period)
        self.dynamic_time_period_spin.setToolTip("当前动力学步的总时间。")
        self.dynamic_initial_increment_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.dynamic_initial_increment_spin.setRange(1.0e-12, 1.0e15)
        self.dynamic_initial_increment_spin.setValue(
            dynamic_controls.initial_time_increment
        )
        self.dynamic_minimum_increment_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.dynamic_minimum_increment_spin.setRange(1.0e-16, 1.0e15)
        self.dynamic_minimum_increment_spin.setValue(
            dynamic_controls.minimum_time_increment
        )
        self.dynamic_maximum_increment_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.dynamic_maximum_increment_spin.setRange(1.0e-12, 1.0e15)
        self.dynamic_maximum_increment_spin.setValue(
            dynamic_controls.maximum_time_increment
        )
        self.dynamic_maximum_increments_spin = QSpinBox(self)
        self.dynamic_maximum_increments_spin.setRange(1, 1_000_000)
        self.dynamic_maximum_increments_spin.setValue(
            dynamic_controls.maximum_increments
        )
        self.dynamic_mass_matrix_combo = QComboBox(self)
        self.dynamic_mass_matrix_combo.addItem("一致质量", MassMatrixPolicy.CONSISTENT.value)
        self.dynamic_mass_matrix_combo.addItem("集中质量", MassMatrixPolicy.LUMPED.value)
        self.dynamic_mass_matrix_combo.setCurrentIndex(
            self.dynamic_mass_matrix_combo.findData(
                dynamic_controls.mass_matrix.value
            )
        )
        self.dynamic_damping_combo = QComboBox(self)
        self.dynamic_damping_combo.addItem("无阻尼", DampingModel.NONE.value)
        self.dynamic_damping_combo.addItem(
            "Rayleigh 阻尼",
            DampingModel.RAYLEIGH.value,
        )
        self.dynamic_damping_combo.setCurrentIndex(
            self.dynamic_damping_combo.findData(
                dynamic_controls.damping_model.value
            )
        )
        self.dynamic_rayleigh_mass_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.dynamic_rayleigh_mass_spin.setRange(0.0, 1.0e15)
        self.dynamic_rayleigh_mass_spin.setValue(dynamic_controls.rayleigh_mass)
        self.dynamic_rayleigh_stiffness_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.dynamic_rayleigh_stiffness_spin.setRange(0.0, 1.0e15)
        self.dynamic_rayleigh_stiffness_spin.setValue(
            dynamic_controls.rayleigh_stiffness
        )
        amplitude_points = dynamic_controls.amplitude.points
        amplitude_start = amplitude_points[0][1]
        amplitude_end = amplitude_points[-1][1]
        self.dynamic_amplitude_start_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.dynamic_amplitude_start_spin.setRange(-1.0e15, 1.0e15)
        self.dynamic_amplitude_start_spin.setValue(amplitude_start)
        self.dynamic_amplitude_end_spin = AdaptivePrecisionDoubleSpinBox(
            self,
            input_decimals=16,
        )
        self.dynamic_amplitude_end_spin.setRange(-1.0e15, 1.0e15)
        self.dynamic_amplitude_end_spin.setValue(amplitude_end)
        initial_condition_text = (
            "已定义（编辑时保持不变）"
            if not self._initial_conditions.is_empty
            else "默认零初始条件"
        )
        self.dynamic_initial_condition_label = QLabel(
            initial_condition_text,
            self,
        )
        self.dynamic_initial_condition_label.setToolTip(
            "当前阶段沿用模型中的初始位移、速度和加速度；不会因编辑分析步而丢失。"
        )
        dynamic_form.addRow("总时间", self.dynamic_time_period_spin)
        dynamic_form.addRow("初始时间增量", self.dynamic_initial_increment_spin)
        dynamic_form.addRow("最小时间增量", self.dynamic_minimum_increment_spin)
        dynamic_form.addRow("最大时间增量", self.dynamic_maximum_increment_spin)
        dynamic_form.addRow("最大增量数", self.dynamic_maximum_increments_spin)
        dynamic_form.addRow("质量矩阵", self.dynamic_mass_matrix_combo)
        dynamic_form.addRow("阻尼模型", self.dynamic_damping_combo)
        dynamic_form.addRow("Rayleigh 质量系数", self.dynamic_rayleigh_mass_spin)
        dynamic_form.addRow(
            "Rayleigh 刚度系数",
            self.dynamic_rayleigh_stiffness_spin,
        )
        dynamic_form.addRow("幅值（起点）", self.dynamic_amplitude_start_spin)
        dynamic_form.addRow("幅值（终点）", self.dynamic_amplitude_end_spin)
        dynamic_form.addRow("初始条件", self.dynamic_initial_condition_label)
        dynamic_page = QWidget(self)
        dynamic_page.setLayout(dynamic_form)

        settings_stack = QStackedWidget(self)
        settings_stack.addWidget(settings_tabs)
        settings_stack.addWidget(dynamic_page)
        self.settings_tabs = settings_tabs
        self.settings_stack = settings_stack
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(settings_stack)
        layout.addWidget(_buttons(self))
        self.procedure_combo.currentIndexChanged.connect(
            self._sync_procedure_controls
        )
        self.procedure_combo.currentIndexChanged.connect(
            self._sync_dynamic_procedure_controls
        )
        self.incrementation_mode_combo.currentIndexChanged.connect(
            self._sync_nonlinear_controls
        )
        self.automatic_cutback_check.toggled.connect(
            self._sync_nonlinear_controls
        )
        self.adaptive_growth_check.toggled.connect(
            self._sync_nonlinear_controls
        )
        self.dynamic_damping_combo.currentIndexChanged.connect(
            self._sync_dynamic_damping_controls
        )
        self._sync_nonlinear_controls()
        self._sync_procedure_controls()
        self._sync_dynamic_procedure_controls()
        self._sync_dynamic_damping_controls()

    def _sync_nonlinear_controls(self, *_args: object) -> None:
        automatic = (
            self.incrementation_mode_combo.currentData() == "automatic"
        )
        self.automatic_cutback_check.setEnabled(automatic)
        self.adaptive_growth_check.setEnabled(automatic)
        self.minimum_increment_spin.setEnabled(
            automatic and self.automatic_cutback_check.isChecked()
        )
        self.maximum_increment_spin.setEnabled(automatic)
        self.growth_factor_spin.setEnabled(
            automatic and self.adaptive_growth_check.isChecked()
        )
        self.growth_iteration_threshold_spin.setEnabled(
            automatic and self.adaptive_growth_check.isChecked()
        )

    def _sync_procedure_controls(self, *_args: object) -> None:
        procedure = self.procedure_combo.currentData()
        dynamic = procedure in {"dynamic", "dynamic_explicit"}
        self._procedure = "dynamic" if dynamic else "static"
        self.settings_stack.setCurrentIndex(1 if dynamic else 0)
        self.nlgeom_check.setEnabled(True)
        if dynamic:
            self._dynamic_procedure_kind = (
                DynamicProcedureKind.EXPLICIT
                if procedure == "dynamic_explicit"
                else DynamicProcedureKind.IMPLICIT
            )
            self.setWindowTitle(
                "创建显式动力学分析步"
                if self._dynamic_procedure_kind is DynamicProcedureKind.EXPLICIT
                else "创建隐式动力学分析步"
            )
        else:
            self.setWindowTitle("创建静力分析步")

    def _sync_dynamic_procedure_controls(self, *_args: object) -> None:
        explicit = (
            self.procedure_combo.currentData() == "dynamic_explicit"
        )
        if explicit:
            self.dynamic_mass_matrix_combo.setCurrentIndex(
                self.dynamic_mass_matrix_combo.findData(
                    MassMatrixPolicy.LUMPED.value
                )
            )
        self.dynamic_mass_matrix_combo.setEnabled(not explicit)
        self.dynamic_mass_matrix_combo.setToolTip(
            "显式动力学固定使用集中质量。"
            if explicit
            else "隐式动力学可使用一致质量或集中质量。"
        )

    def _sync_dynamic_damping_controls(self, *_args: object) -> None:
        rayleigh = (
            self.dynamic_damping_combo.currentData()
            == DampingModel.RAYLEIGH.value
        )
        self.dynamic_rayleigh_mass_spin.setEnabled(rayleigh)
        self.dynamic_rayleigh_stiffness_spin.setEnabled(rayleigh)

    def step(self):
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("分析步名称不能为空")
        if self.procedure_combo.currentData() in {
            "dynamic",
            "dynamic_explicit",
        }:
            base = self._dynamic_controls
            time_period = self.dynamic_time_period_spin.value()
            explicit = (
                self.procedure_combo.currentData() == "dynamic_explicit"
            )
            dynamic_controls = DynamicStepControls(
                time_period=time_period,
                initial_time_increment=self.dynamic_initial_increment_spin.value(),
                minimum_time_increment=self.dynamic_minimum_increment_spin.value(),
                maximum_time_increment=self.dynamic_maximum_increment_spin.value(),
                maximum_increments=self.dynamic_maximum_increments_spin.value(),
                # The visible analysis type is the source of truth.  Do not
                # reuse the previous step's hidden integration method when
                # an explicit step is switched to implicit (or vice versa).
                integration_method=(
                    DynamicIntegrationMethod.CENTRAL_DIFFERENCE
                    if explicit
                    else DynamicIntegrationMethod.NEWMARK
                ),
                beta=base.beta,
                gamma=base.gamma,
                mass_matrix=(
                    MassMatrixPolicy.LUMPED
                    if explicit
                    else MassMatrixPolicy(
                        self.dynamic_mass_matrix_combo.currentData()
                    )
                ),
                damping_model=DampingModel(
                    self.dynamic_damping_combo.currentData()
                ),
                rayleigh_mass=self.dynamic_rayleigh_mass_spin.value(),
                rayleigh_stiffness=self.dynamic_rayleigh_stiffness_spin.value(),
                amplitude=TimeAmplitude(
                    (
                        (0.0, self.dynamic_amplitude_start_spin.value()),
                        (time_period, self.dynamic_amplitude_end_spin.value()),
                    )
                ),
                procedure_kind=(
                    DynamicProcedureKind.EXPLICIT
                    if explicit
                    else DynamicProcedureKind.IMPLICIT
                ),
            )
            formulation = (
                StaticFormulation.NONLINEAR
                if self.nlgeom_check.isChecked()
                else StaticFormulation.LINEAR
            )
            if self._original_step is not None:
                return update_dynamic_step(
                    self._original_step,
                    name=name,
                    controls=dynamic_controls,
                    initial_conditions=self._initial_conditions,
                    formulation=formulation,
                )
            return transient_dynamic(
                name,
                controls=dynamic_controls,
                initial_conditions=self._initial_conditions,
                formulation=formulation,
            )
        automatic = (
            self.incrementation_mode_combo.currentData() == "automatic"
        )
        controls = StaticStepControls(
            initial_increment=self.initial_increment_spin.value(),
            maximum_increments=self.maximum_increments_spin.value(),
            maximum_increment=self.maximum_increment_spin.value(),
            newton_max_iterations=self.newton_max_iterations_spin.value(),
            residual_tolerance=self.residual_tolerance_spin.value(),
            control_mode=StaticControlMode(
                self.control_mode_combo.currentData()
            ),
            relative_residual_tolerance=(
                self.relative_residual_tolerance_spin.value()
            ),
            displacement_tolerance=self.displacement_tolerance_spin.value(),
            energy_tolerance=self.energy_tolerance_spin.value(),
            constraint_tolerance=self.constraint_tolerance_spin.value(),
            newton_strategy=NewtonStrategy(
                self.newton_strategy_combo.currentData()
            ),
            line_search=self.line_search_check.isChecked(),
            predictor=self.predictor_check.isChecked(),
            automatic_cutback=(
                automatic and self.automatic_cutback_check.isChecked()
            ),
            minimum_increment=self.minimum_increment_spin.value(),
            adaptive_growth=(
                automatic and self.adaptive_growth_check.isChecked()
            ),
            growth_factor=self.growth_factor_spin.value(),
            growth_iteration_threshold=(
                self.growth_iteration_threshold_spin.value()
            ),
        )
        formulation = (
            StaticFormulation.NONLINEAR
            if self.nlgeom_check.isChecked()
            else StaticFormulation.LINEAR
        )
        if self._original_step is not None:
            return update_static_step(
                self._original_step,
                name=name,
                formulation=formulation,
                controls=controls,
            )
        return static(
            name,
            controls=controls,
            formulation=formulation,
        )


@dataclass(frozen=True)
class DisplacementDialogState:
    """Uncommitted displacement-form values retained during scope creation."""

    scope_kind: str
    step_name: str
    components: tuple[tuple[int, bool, float], ...]
    name: str = ""


class DisplacementDialog(QDialog):
    scopeChanged = Signal(object)

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
        form_state: DisplacementDialogState | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("位移边界条件")
        self._reported_scope: object = _SCOPE_NOT_REPORTED
        supported_scope_kinds = frozenset(
            str(kind)
            for kind in scope_selection_kinds
            if str(kind) in {"node", "edge", "surface"}
        )
        if allow_scope_selection and not supported_scope_kinds:
            supported_scope_kinds = frozenset({"node"})
        self._scope_selection_kinds = supported_scope_kinds
        self._scope_selection_request: str | None = None
        self.name_edit = QLineEdit(
            "" if current is None or current.name is None else current.name,
            self,
        )
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
            current_region = RegionRef(
                getattr(current, "target_kind", "node_set"),
                str(current.target),
            )
            if selected_region is None:
                selected_region = current_region
            for component, check in self.component_checks.items():
                selected = (
                    current.first_component
                    <= component
                    <= current.last_component
                )
                check.setChecked(selected)
                if selected:
                    self.component_values[component].setValue(current.value)
        if form_state is not None:
            self.name_edit.setText(form_state.name)
            kind = {
                "node": "node_set",
                "edge": "edge",
                "surface": "surface",
            }.get(form_state.scope_kind)
            index = self.kind_combo.findData(kind)
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
            self.step_combo.setCurrentText(form_state.step_name)
            for component, checked, value in form_state.components:
                if component not in self.component_checks:
                    continue
                self.component_checks[component].setChecked(checked)
                self.component_values[component].setValue(value)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("名称", self.name_edit)
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
            self._on_scope_changed
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

    def form_state(self) -> DisplacementDialogState:
        """Capture the current form without requiring a valid region."""

        scope_kind = {
            "node_set": "node",
            "edge": "edge",
            "surface": "surface",
        }.get(str(self.kind_combo.currentData() or ""), "node")
        return DisplacementDialogState(
            scope_kind,
            self.step_combo.currentText(),
            tuple(
                (
                    component,
                    check.isChecked(),
                    self.component_values[component].value(),
                )
                for component, check in self.component_checks.items()
            ),
            self.name_edit.text(),
        )

    def _refresh_regions(self) -> None:
        current = self.region_combo.currentData()
        kind = str(self.kind_combo.currentData() or "")
        signals_blocked = self.region_combo.blockSignals(True)
        try:
            self.region_combo.clear()
            for reference in self._regions.get(kind, ()):
                self.region_combo.addItem(reference.name, reference)
            if isinstance(current, RegionRef) and current.kind == kind:
                _select_region(self.region_combo, current)
        finally:
            self.region_combo.blockSignals(signals_blocked)
        request_kind = {
            "node_set": "node",
            "edge": "edge",
            "surface": "surface",
        }.get(kind)
        self.scope_pick_button.setEnabled(
            request_kind in self._scope_selection_kinds
        )
        self._on_scope_changed()

    def selected_scope(self) -> RegionRef | None:
        region = self.region_combo.currentData()
        return region if isinstance(region, RegionRef) else None

    def _on_scope_changed(self) -> None:
        self._update_accept_state()
        scope = self.selected_scope()
        if scope == self._reported_scope:
            return
        self._reported_scope = scope
        self.scopeChanged.emit(scope)

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
        selected = [
            (component, self.component_values[component].value())
            for component, check in self.component_checks.items()
            if check.isChecked()
        ]
        if not selected:
            raise ValueError("至少勾选一个位移自由度")
        ranges: list[tuple[int, int, float]] = []
        for component, value in selected:
            if (
                ranges
                and component == ranges[-1][1] + 1
                and value == ranges[-1][2]
            ):
                first, _last, previous_value = ranges[-1]
                ranges[-1] = (first, component, previous_value)
            else:
                ranges.append((component, component, value))
        name = self.name_edit.text().strip() or None
        return step_name, tuple(
            DisplacementConstraint(
                region.name,
                first,
                last,
                value,
                target_kind=region.kind,
                name=(
                    None
                    if name is None
                    else name
                    if index == 0
                    else f"{name}-{index + 1}"
                ),
            )
            for index, (first, last, value) in enumerate(ranges)
        )


@dataclass(frozen=True)
class LoadDialogState:
    """Uncommitted load-form values retained during scope creation."""

    scope_kind: str
    step_name: str
    load_type: str
    coordinate_system: str
    component: int | None
    value: float
    vectors: tuple[tuple[str, tuple[float, ...]], ...]
    name: str = ""


class LoadDialog(QDialog):
    scopeChanged = Signal(object)

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
        form_state: LoadDialogState | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑载荷" if current is not None else "创建载荷")
        self._reported_scope: object = _SCOPE_NOT_REPORTED
        supported_scope_kinds = frozenset(
            str(kind)
            for kind in scope_selection_kinds
            if str(kind) in {"node", "edge", "surface", "line", "body"}
        )
        self._scope_selection_kinds = supported_scope_kinds
        self._scope_selection_request: str | None = None
        self.name_edit = QLineEdit(
            "" if current is None or current.name is None else current.name,
            self,
        )
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
        self.form.addRow("名称", self.name_edit)
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
            if selected_region is None:
                selected_region = RegionRef("node_set", str(current.target))
        elif isinstance(current, EdgeLoad):
            self.kind_combo.setCurrentIndex(
                max(0, self.kind_combo.findData("edge"))
            )
            if selected_region is None:
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
            if selected_region is None:
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
            if selected_region is None:
                selected_region = RegionRef(
                    "element_set",
                    str(current.target),
                )
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
            if selected_region is None:
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
        if form_state is not None:
            self.name_edit.setText(form_state.name)
            for kind, vector in form_state.vectors:
                if kind in self._vector_values:
                    self._vector_values[kind] = tuple(vector)
            kind_index = self.kind_combo.findData(form_state.scope_kind)
            if kind_index >= 0:
                self.kind_combo.setCurrentIndex(kind_index)
            self.step_combo.setCurrentText(form_state.step_name)
            load_type_index = self.load_type_combo.findData(
                form_state.load_type
            )
            if load_type_index >= 0:
                self.load_type_combo.setCurrentIndex(load_type_index)
            coordinate_index = self.coordinate_system_combo.findData(
                form_state.coordinate_system
            )
            if coordinate_index >= 0:
                self.coordinate_system_combo.setCurrentIndex(coordinate_index)
            component_index = self.component_combo.findData(
                form_state.component
            )
            if component_index >= 0:
                self.component_combo.setCurrentIndex(component_index)
            self.value_spin.setValue(form_state.value)
        layout = QVBoxLayout(self)
        layout.addLayout(self.form)
        self.buttons = _buttons(self)
        layout.addWidget(self.buttons)
        self.setMinimumWidth(350)
        self.region_combo.currentIndexChanged.connect(
            self._on_scope_changed
        )
        self.step_combo.currentIndexChanged.connect(
            self._update_candidate_state
        )
        self.kind_combo.currentIndexChanged.connect(self._refresh)
        self.load_type_combo.currentIndexChanged.connect(self._refresh)
        self.coordinate_system_combo.currentIndexChanged.connect(self._refresh)
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

    def form_state(self) -> LoadDialogState:
        """Capture every load-kind value before leaving for scope creation."""

        vectors = dict(self._vector_values)
        kind = str(self.kind_combo.currentData() or "node")
        if kind in vectors:
            dimensions = 3 if kind == "line" else self.spatial_dimensions
            vectors[kind] = tuple(
                spin.value()
                for spin in (self.x_spin, self.y_spin, self.z_spin)[
                    :dimensions
                ]
            )
        component = self.component_combo.currentData()
        return LoadDialogState(
            kind,
            self.step_combo.currentText(),
            str(self.load_type_combo.currentData() or "traction"),
            str(self.coordinate_system_combo.currentData() or "global"),
            None if component is None else int(component),
            self.value_spin.value(),
            tuple((name, tuple(vector)) for name, vector in vectors.items()),
            self.name_edit.text(),
        )

    def selected_scope(self) -> RegionRef | int | None:
        kind = str(self.kind_combo.currentData() or "")
        if kind == "gravity":
            if isinstance(self._gravity_target, int):
                return int(self._gravity_target)
            if self._gravity_target is not None:
                return RegionRef(
                    "element_set",
                    str(self._gravity_target),
                )
            return None
        region = self.region_combo.currentData()
        return region if isinstance(region, RegionRef) else None

    def _on_scope_changed(self) -> None:
        self._update_candidate_state()
        scope = self.selected_scope()
        if scope == self._reported_scope:
            return
        self._reported_scope = scope
        self.scopeChanged.emit(scope)

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
        signals_blocked = self.region_combo.blockSignals(True)
        try:
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
        finally:
            self.region_combo.blockSignals(signals_blocked)
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
        self._on_scope_changed()

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
        name = self.name_edit.text().strip() or None
        if kind == "gravity":
            acceleration = (self.x_spin.value(),)
            if self.spatial_dimensions >= 2:
                acceleration += (self.y_spin.value(),)
            if self.spatial_dimensions == 3:
                acceleration += (self.z_spin.value(),)
            return step, GravityLoad(
                acceleration,
                self._gravity_target,
                name=name,
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
                name=name,
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
                name=name,
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
            return step, BodyForce(target, vector, name=name)
        load_type = str(self.load_type_combo.currentData())
        if load_type == "pressure":
            magnitude = self.value_spin.value()
            if magnitude == 0.0:
                raise ValueError("压力值不能为 0")
            load_class = EdgeLoad if kind == "edge" else SurfaceLoad
            return step, load_class(
                target,
                magnitude=magnitude,
                load_type="pressure",
                name=name,
            )
        vector = tuple(value.value() for value in (self.x_spin, self.y_spin))
        if self.spatial_dimensions == 3:
            vector += (self.z_spin.value(),)
        if not any(value != 0.0 for value in vector):
            raise ValueError("牵引载荷至少需要一个非零分量")
        load_class = EdgeLoad if kind == "edge" else SurfaceLoad
        return step, load_class(
            target,
            vector,
            load_type="traction",
            name=name,
        )


class OutputRequestDialog(QDialog):
    def __init__(
        self,
        step_names: list[str],
        parent=None,
        *,
        candidates: Sequence[OutputRequestProjection] = (),
        current: OutputRequest | None = None,
        existing_requests_by_step: Mapping[
            str,
            Sequence[OutputRequest],
        ] | None = None,
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
        existing_values = (
            {}
            if existing_requests_by_step is None
            else dict(existing_requests_by_step)
        )
        if any(
            type(name) is not str
            or type(requests) not in {tuple, list}
            or any(type(request) is not OutputRequest for request in requests)
            for name, requests in existing_values.items()
        ):
            raise TypeError(
                "existing_requests_by_step must map step names to OutputRequest sequences"
            )

        self._candidates = deepcopy(
            _visible_output_request_candidates(candidate_values)
        )
        self._existing_requests_by_step = {
            name: tuple(deepcopy(requests))
            for name, requests in existing_values.items()
        }
        self._current = (
            None
            if current is None
            else _compact_output_request(current)
        )
        self.setWindowTitle("输出请求")
        self.step_combo = QComboBox(self)
        self.step_combo.addItems(step_names)
        self.candidate_list = QListWidget(self)
        self.candidate_list.setObjectName("outputRequestCandidateList")
        self.candidate_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.candidate_list.setMinimumHeight(120)
        if self._current is None:
            for index, candidate in enumerate(self._candidates):
                item = QListWidgetItem(
                    candidate.authoring_request.variables[0],
                    self.candidate_list,
                )
                item.setData(Qt.ItemDataRole.UserRole, index)
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
        else:
            for variable in self._current.variables:
                item = QListWidgetItem(variable, self.candidate_list)
                item.setCheckState(Qt.CheckState.Checked)
                item.setFlags(
                    (item.flags() | Qt.ItemFlag.ItemIsEnabled)
                    & ~Qt.ItemFlag.ItemIsUserCheckable
                )

        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("分析步", self.step_combo)
        form.addRow(self.candidate_list)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        if self._current is None:
            layout.addWidget(_buttons(self))
            self.step_combo.currentTextChanged.connect(
                lambda _text: self._sync_existing_selection()
            )
        else:
            self.step_combo.setEnabled(False)
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
        if self._current is None:
            self._sync_existing_selection()

    def definitions(self) -> tuple[str, tuple[OutputRequest, ...]]:
        step_name = self.step_combo.currentText().strip()
        if not step_name:
            raise ValueError("请选择分析步")
        if self._current is not None:
            return step_name, (deepcopy(self._current),)
        requests = tuple(
            _compact_output_request(candidate.authoring_request)
            for candidate in self._selected_candidates()
        )
        if not requests:
            raise ValueError("请至少选择一个受支持的输出请求")
        if any(type(request) is not OutputRequest for request in requests):
            raise TypeError(
                "candidate authoring_requests must be exactly OutputRequest"
            )
        return step_name, requests

    def definition(self) -> tuple[str, OutputRequest]:
        step_name, requests = self.definitions()
        if len(requests) != 1:
            raise ValueError("当前选择包含多个输出请求，请使用 definitions()")
        return step_name, requests[0]

    def _sync_existing_selection(self) -> None:
        existing_requests = self._existing_requests_by_step.get(
            self.step_combo.currentText(),
            (),
        )
        self.candidate_list.blockSignals(True)
        for row, candidate in enumerate(self._candidates):
            candidate_request = candidate.authoring_request
            variable = candidate_request.variables[0]
            checked = any(
                request.kind.casefold() == candidate_request.kind.casefold()
                and request.target.casefold()
                == candidate_request.target.casefold()
                and any(
                    existing.strip().casefold() == variable.casefold()
                    for existing in request.variables
                )
                for request in existing_requests
            )
            self.candidate_list.item(row).setCheckState(
                Qt.CheckState.Checked
                if checked
                else Qt.CheckState.Unchecked
            )
        self.candidate_list.blockSignals(False)

    def _selected_candidates(self) -> tuple[OutputRequestProjection, ...]:
        if self._current is not None:
            return ()
        selected: list[OutputRequestProjection] = []
        for row in range(self.candidate_list.count()):
            item = self.candidate_list.item(row)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            index = item.data(Qt.ItemDataRole.UserRole)
            if type(index) is not int or not 0 <= index < len(self._candidates):
                raise RuntimeError("输出请求列表包含无效候选索引")
            selected.append(self._candidates[index])
        return tuple(selected)

    def _selected_requests(self) -> tuple[OutputRequest, ...]:
        if self._current is not None:
            return (self._current,)
        return tuple(
            candidate.authoring_request
            for candidate in self._selected_candidates()
        )


def _is_required_displacement_output(request: OutputRequest) -> bool:
    return (
        type(request) is OutputRequest
        and request.kind.casefold() == "field"
        and request.target.casefold() == "node"
        and any(
            variable.strip().casefold() == "u"
            for variable in request.variables
        )
    )


def _compact_output_request(request: OutputRequest) -> OutputRequest:
    if type(request) is not OutputRequest:
        raise TypeError("request must be exactly OutputRequest")
    # A view/edit dialog must never rebuild an output request from only the
    # visible variables.  Imported metadata, source evidence, and the name
    # are part of the persisted model state even when this UI cannot edit
    # them yet.
    return deepcopy(request)


def _visible_output_request_candidates(
    candidates: Sequence[OutputRequestProjection],
) -> tuple[OutputRequestProjection, ...]:
    allowed = ("U", "RF", "S")
    first_by_variable: dict[str, OutputRequestProjection] = {}
    for candidate in candidates:
        variables = candidate.authoring_request.variables
        if len(variables) != 1:
            continue
        variable = variables[0]
        if variable in allowed and variable not in first_by_variable:
            first_by_variable[variable] = candidate
    return tuple(
        first_by_variable[variable]
        for variable in allowed
        if variable in first_by_variable
    )


def _unique_analysis_name(base: str, existing: Sequence[object]) -> str:
    """Return a readable copy name without colliding in one manager scope."""

    names = {
        str(value).strip().casefold()
        for value in existing
        if value is not None and str(value).strip()
    }
    candidate = f"{str(base).strip()}-副本"
    suffix = 2
    while candidate.casefold() in names:
        candidate = f"{str(base).strip()}-副本{suffix}"
        suffix += 1
    return candidate


class AnalysisDefinitionManagerDialog(QDialog):
    """Edit existing supported analysis definitions in one flat dialog."""

    scopeChanged = Signal(object)

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
        boundary_scope_selection_kinds: Sequence[str] = (),
        load_scope_selection_kinds: Sequence[str] = (),
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
        self.boundary_scope_selection_kinds = tuple(
            str(kind) for kind in boundary_scope_selection_kinds
        )
        self.load_scope_selection_kinds = tuple(
            str(kind) for kind in load_scope_selection_kinds
        )
        self._scope_selection_request: (
            tuple[str, tuple[str, int, int | None]] | None
        ) = None
        self._scope_selection_dialog_state: (
            DisplacementDialogState | LoadDialogState | None
        ) = None
        self._selected_region_override: RegionRef | None = None
        self._dialog_state_override: (
            DisplacementDialogState | LoadDialogState | None
        ) = None
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
        self.copy_button = QPushButton("复制", self)
        self.rename_button = QPushButton("重命名", self)
        self.delete_button = QPushButton("删除", self)
        self.edit_button.clicked.connect(self._edit)
        self.copy_button.clicked.connect(self._copy)
        self.rename_button.clicked.connect(self._rename)
        self.delete_button.clicked.connect(self._delete)
        self.table.itemDoubleClicked.connect(lambda _item: self._edit())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        controls = QHBoxLayout()
        controls.addWidget(self.edit_button)
        controls.addWidget(self.copy_button)
        controls.addWidget(self.rename_button)
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
                    analysis_step_label(step),
                ),
                ("step", step_index, None),
            )
            for item_index, boundary in enumerate(step.boundaries):
                self._append_row(
                    (
                        "位移边界",
                        step.name,
                        (
                            f"{boundary.name} · {boundary.target}"
                            if boundary.name is not None
                            else str(boundary.target)
                        ),
                        self._boundary_text(boundary),
                    ),
                    ("boundary", step_index, item_index),
                )
            for item_index, load in enumerate(step.cloads):
                self._append_row(
                    (
                        "节点力",
                        step.name,
                        (
                            f"{load.name} · {load.target}"
                            if load.name is not None
                            else str(load.target)
                        ),
                        f"{self._force_label(load.component)} = {load.value:g}",
                    ),
                    ("node_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.edge_loads):
                self._append_row(
                    (
                        "边力",
                        step.name,
                        (
                            f"{load.name} · {load.edge}"
                            if load.name is not None
                            else load.edge
                        ),
                        self._distributed_text(load),
                    ),
                    ("edge_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.surface_loads):
                self._append_row(
                    (
                        "面力",
                        step.name,
                        (
                            f"{load.name} · {load.surface}"
                            if load.name is not None
                            else load.surface
                        ),
                        self._distributed_text(load),
                    ),
                    ("surface_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.line_loads):
                self._append_row(
                    (
                        "边力",
                        step.name,
                        (
                            f"{load.name} · {load.target}"
                            if load.name is not None
                            else str(load.target)
                        ),
                        self._line_load_text(load),
                    ),
                    ("line_load", step_index, item_index),
                )
            for item_index, load in enumerate(step.body_loads):
                self._append_row(
                    (
                        "体力",
                        step.name,
                        (
                            f"{load.name} · {load.target}"
                            if load.name is not None
                            else str(load.target)
                        ),
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
                        )
                        if load.name is None
                        else f"{load.name} · "
                        + (
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
                            "field": "输出",
                            "history": "历史输出",
                        }.get(output.kind, "输出请求"),
                        step.name,
                        (
                            f"{output.name} · "
                            if output.name is not None
                            else ""
                        )
                        + {
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
        *,
        selected_region: RegionRef | None = None,
        dialog_state: DisplacementDialogState | LoadDialogState | None = None,
    ) -> bool:
        """Open the existing parameter dialog and report a real change."""
        if not self.select_definition(key):
            return False
        if selected_region is not None and type(selected_region) is not RegionRef:
            raise TypeError("selected_region must be a RegionRef")
        self._scope_selection_request = None
        self._scope_selection_dialog_state = None
        self._selected_region_override = selected_region
        self._dialog_state_override = dialog_state
        previous = deepcopy(self.steps)
        try:
            self._edit()
        finally:
            self._selected_region_override = None
            self._dialog_state_override = None
        return self.steps != previous

    def requested_scope_selection(
        self,
    ) -> tuple[str, tuple[str, int, int | None]] | None:
        """Return a viewport-scope request raised by the current editor."""

        return self._scope_selection_request

    def requested_scope_dialog_state(
        self,
    ) -> DisplacementDialogState | LoadDialogState | None:
        """Return the nested editor values saved before scope selection."""

        return self._scope_selection_dialog_state

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

    def _exec_scope_dialog(
        self,
        dialog: DisplacementDialog | LoadDialog,
    ) -> int:
        dialog.scopeChanged.connect(self.scopeChanged.emit)
        self.scopeChanged.emit(dialog.selected_scope())
        try:
            return int(dialog.exec())
        finally:
            self.scopeChanged.emit(None)

    def _edit(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        kind, step_index, item_index = selected
        step = self.steps[step_index]
        row = self.table.currentRow()
        if kind == "step":
            dialog = StaticStepDialog(
                step.name,
                self,
                current=step,
            )
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
            self.steps[step_index] = updated
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
                selected_region=(
                    self._selected_region_override or current_region
                ),
                current=current,
                labels=self.dof_labels,
                scope_selection_kinds=self.boundary_scope_selection_kinds,
                form_state=(
                    self._dialog_state_override
                    if isinstance(
                        self._dialog_state_override,
                        DisplacementDialogState,
                    )
                    else None
                ),
            )
            if not isinstance(
                self._dialog_state_override,
                DisplacementDialogState,
            ):
                dialog.step_combo.setCurrentText(step.name)
            if not self._exec_scope_dialog(dialog):
                requested_kind = dialog.requested_scope_kind()
                if requested_kind is not None:
                    self._scope_selection_request = (
                        requested_kind,
                        selected,
                    )
                    self._scope_selection_dialog_state = dialog.form_state()
                    self.reject()
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
                selected_region=self._selected_region_override,
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
                scope_selection_kinds=self.load_scope_selection_kinds,
                form_state=(
                    self._dialog_state_override
                    if isinstance(
                        self._dialog_state_override,
                        LoadDialogState,
                    )
                    else None
                ),
            )
            if not isinstance(
                self._dialog_state_override,
                LoadDialogState,
            ):
                dialog.step_combo.setCurrentText(step.name)
            if not self._exec_scope_dialog(dialog):
                requested_kind = dialog.requested_scope_kind()
                if requested_kind is not None:
                    self._scope_selection_request = (
                        requested_kind,
                        selected,
                    )
                    self._scope_selection_dialog_state = dialog.form_state()
                    self.reject()
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
        required_output = (
            kind == "output"
            and item_index is not None
            and _is_required_displacement_output(
                step.outputs[int(item_index)]
            )
        )
        if (
            kind == "output"
            and not self._output_delete_capability.can_submit
        ) or (
            kind in {"step", "output"}
            and initial_output_owner
        ) or (
            required_output
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

    def _copy(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        kind, step_index, item_index = selected
        step = self.steps[step_index]
        if kind == "step":
            if step.name.strip().casefold() == "initial":
                return
            clone = deepcopy(step)
            clone.name = _unique_analysis_name(
                clone.name,
                (candidate.name for candidate in self.steps),
            )
            self.steps.insert(step_index + 1, clone)
            self._refresh(self.table.currentRow() + 1)
            return
        collection_name = {
            "boundary": "boundaries",
            "node_load": "cloads",
            "edge_load": "edge_loads",
            "surface_load": "surface_loads",
            "line_load": "line_loads",
            "body_load": "body_loads",
            "gravity_load": "gravity_loads",
            "output": "outputs",
        }.get(kind)
        if collection_name is None or item_index is None:
            return
        collection = list(getattr(step, collection_name))
        source = collection[int(item_index)]
        base_name = getattr(source, "name", None) or {
            "boundary": "位移边界",
            "node_load": "节点力",
            "edge_load": "边力",
            "surface_load": "面力",
            "line_load": "线力",
            "body_load": "体力",
            "gravity_load": "重力",
            "output": "输出请求",
        }.get(kind, "对象")
        clone = replace(
            deepcopy(source),
            name=_unique_analysis_name(
                str(base_name),
                (getattr(candidate, "name", None) for candidate in collection),
            ),
        )
        collection.insert(int(item_index) + 1, clone)
        setattr(step, collection_name, tuple(collection))
        self._refresh(self.table.currentRow() + 1)

    def _rename(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        kind, step_index, item_index = selected
        step = self.steps[step_index]
        if kind == "step":
            if step.name.strip().casefold() == "initial":
                return
            current = step.name
            values = (candidate.name for candidate in self.steps)
            target = "分析步"
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
            }.get(kind)
            if collection_name is None or item_index is None:
                return
            collection = tuple(getattr(step, collection_name))
            current = getattr(collection[int(item_index)], "name", None) or ""
            values = (
                getattr(candidate, "name", None)
                for candidate in collection
            )
            target = "分析对象"
        value, accepted = QInputDialog.getText(
            self,
            f"重命名{target}",
            f"{target}名称：",
            text=str(current),
        )
        if not accepted:
            return
        name = str(value).strip()
        if not name or name == current:
            return
        if any(
            existing is not None and str(existing).casefold() == name.casefold()
            and str(existing) != str(current)
            for existing in values
        ):
            QMessageBox.warning(self, "分析定义", f"名称已存在：{name}")
            return
        if kind == "step":
            step.name = name
            self._refresh(self.table.currentRow())
            return
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
        collection = list(getattr(step, collection_name))
        collection[int(item_index)] = replace(
            collection[int(item_index)],
            name=name,
        )
        setattr(step, collection_name, tuple(collection))
        self._refresh(self.table.currentRow())

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
        protects_initial_output = (
            selected is not None
            and selected[0] == "output"
            and self.steps[selected[1]].name.strip().casefold()
            == "initial"
        )
        output_step_is_editable = (
            is_output
            and self.steps[selected[1]].name.strip().casefold()
            != "initial"
        )
        required_output = (
            is_output
            and selected[2] is not None
            and _is_required_displacement_output(
                self.steps[selected[1]].outputs[int(selected[2])]
            )
        )
        self.edit_button.setEnabled(
            selected is not None
            and (
                not is_output
                or self._output_view_capability.can_enter
            )
        )
        self.copy_button.setEnabled(
            selected is not None
            and not (
                selected[0] == "step"
                and self.steps[selected[1]].name.strip().casefold()
                == "initial"
            )
            and not protects_initial_output
            and (
                not is_output
                or self._output_view_capability.can_enter
            )
        )
        self.rename_button.setEnabled(
            selected is not None
            and not (
                selected[0] == "step"
                and self.steps[selected[1]].name.strip().casefold()
                == "initial"
            )
            and not protects_initial_output
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
            and not required_output
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
