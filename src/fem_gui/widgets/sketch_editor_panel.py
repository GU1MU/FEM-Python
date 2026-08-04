"""Non-modal editor panel for strict planar sketch drafts."""

from __future__ import annotations

import math
import re

from PySide6.QtCore import QEvent, QItemSelectionModel, QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fem.geometry import (
    SketchArc,
    SketchCircle,
    SketchCoincidentConstraint,
    SketchDistanceDimension,
    SketchFixedConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchPlane,
    SketchPointOnCurveConstraint,
    SketchRadiusDimension,
    SketchReferencePoint,
    SketchVerticalConstraint,
)
from fem.geometry.recipes import sketch_constraint_entity_ids

from ..geometry_preview import build_strict_sketch_draft_preview
from ..sketch_preferences import (
    SketchPreferences,
    load_sketch_preferences,
    save_sketch_preferences,
)
from ..sketch_editor import (
    SketchDraftController,
    SketchDraftSnapshot,
    SketchDraftValidationError,
)
from ..sketch_constraint_ui import (
    build_constraint_overlays,
    constraint_text,
    constraints_for_entities,
    infer_line_preview,
    measured_dimension_value,
    solve_status_text,
)
from .viewport import SketchDraftRenderData


def _stable_id_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", str(value))
        if part
    )


class _PointTableItem(QTableWidgetItem):
    """Sort table values predictably and use the point ID as a tie-breaker."""

    def __init__(self, text: str, point_id: str, *, numeric: float | None = None):
        super().__init__(text)
        self._point_id = point_id
        self._numeric = numeric

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, _PointTableItem):
            if self.column() == 0 and other.column() == 0:
                return _stable_id_key(self.text()) < _stable_id_key(other.text())
            left = self._numeric if self._numeric is not None else self.text().casefold()
            right = (
                other._numeric
                if other._numeric is not None
                else other.text().casefold()
            )
            if left == right:
                return _stable_id_key(self._point_id) < _stable_id_key(other._point_id)
            return left < right
        return super().__lt__(other)


class SketchEditorPanel(QWidget):
    """Synchronize one detached strict sketch draft with the main viewport."""

    finishRequested = Signal()
    cancelRequested = Signal()
    draftChanged = Signal(object)
    statusChanged = Signal(str)
    entityFocusRequested = Signal(str, str)

    def __init__(
        self,
        controller: SketchDraftController | None = None,
        parent=None,
        *,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sketchEditorPanel")
        self.setMinimumWidth(330)
        self.setMaximumWidth(500)
        self._controller: SketchDraftController | None = None
        self._settings = settings
        self._preferences = (
            SketchPreferences()
            if settings is None
            else load_sketch_preferences(settings)
        )
        self._viewport = None
        self._base_snapshot: SketchDraftSnapshot | None = None
        self._refreshing = False
        self._pending_points: list[tuple[float, float]] = []
        self._pending_references: list[SketchReferencePoint | None] = []
        self._incoming_reference: SketchReferencePoint | None = None
        self._incoming_snap: dict[str, object] = {}
        self._inference_preview: tuple[str, ...] = ()
        self._drag_preview = None
        self._reference_points: tuple[SketchReferencePoint, ...] = ()
        self._polyline_start_id: str | None = None
        self._polyline_first_id: str | None = None
        self._authoring_purpose = "geometry"
        self._selection_anchor_id: str | None = None
        self._build_ui()
        if controller is not None:
            self.set_controller(controller)

    @property
    def controller(self) -> SketchDraftController | None:
        return self._controller

    @property
    def dirty(self) -> bool:
        return bool(self._controller is not None and self._controller.dirty)

    @property
    def base_snapshot(self) -> SketchDraftSnapshot | None:
        return self._base_snapshot

    def _build_ui(self) -> None:
        self.name_edit = QLineEdit(self)
        self.name_edit.setObjectName("sketchNameEdit")
        self.name_edit.editingFinished.connect(self._name_changed)

        modes = (
            ("select", "选择"),
            ("polyline", "折线"),
            ("rectangle", "矩形"),
            ("circle", "圆"),
            ("arc", "三点圆弧"),
            ("trim", "修剪"),
        )
        self._mode_buttons: dict[str, QPushButton] = {}
        first_modes = QHBoxLayout()
        second_modes = QHBoxLayout()
        for index, (mode, text) in enumerate(modes):
            button = QPushButton(text, self)
            button.setCheckable(True)
            button.setObjectName(f"sketch{mode.title()}ModeButton")
            button.clicked.connect(
                lambda _checked=False, selected=mode: self.set_mode(selected)
            )
            self._mode_buttons[mode] = button
            (first_modes if index < 3 else second_modes).addWidget(button)
        self._mode_buttons["polyline"].setChecked(True)
        self._mode_buttons["trim"].setToolTip(
            "单击曲线：有交点时删除点击段；无可用交点时删除整条曲线"
        )

        def preference_check(
            text: str,
            object_name: str,
            checked: bool,
        ) -> QCheckBox:
            control = QCheckBox(text, self)
            control.setObjectName(object_name)
            control.setChecked(checked)
            control.toggled.connect(self._preferences_changed)
            return control

        preferences = self._preferences
        self.grid_visible_check = preference_check(
            "显示网格", "sketchGridVisible", preferences.grid_visible
        )
        self.snap_check = preference_check(
            "捕捉网格点", "sketchGridSnap", preferences.grid_snap
        )
        self.spacing_spin = QDoubleSpinBox(self)
        self.spacing_spin.setObjectName("sketchGridSpacing")
        self.spacing_spin.setDecimals(3)
        self.spacing_spin.setRange(0.001, 1.0e12)
        self.spacing_spin.setSingleStep(0.1)
        self.spacing_spin.setValue(preferences.grid_spacing)
        self.spacing_spin.valueChanged.connect(self._preferences_changed)

        self.snap_sketch_points_check = preference_check(
            "已有草图点",
            "sketchSnapSketchPoints",
            preferences.snap_sketch_points,
        )
        self.snap_external_points_check = preference_check(
            "外部参考点",
            "sketchSnapExternalPoints",
            preferences.snap_external_points,
        )
        self.snap_midpoints_check = preference_check(
            "中点", "sketchSnapMidpoints", preferences.snap_midpoints
        )
        self.snap_centers_check = preference_check(
            "圆心", "sketchSnapCenters", preferences.snap_centers
        )
        self.snap_intersections_check = preference_check(
            "交点", "sketchSnapIntersections", preferences.snap_intersections
        )
        self.screen_snap_tolerance_spin = QDoubleSpinBox(self)
        self.screen_snap_tolerance_spin.setObjectName("sketchScreenSnapTolerance")
        self.screen_snap_tolerance_spin.setDecimals(1)
        self.screen_snap_tolerance_spin.setRange(0.0, 100.0)
        self.screen_snap_tolerance_spin.setSuffix(" px")
        self.screen_snap_tolerance_spin.setValue(preferences.screen_snap_tolerance)
        self.screen_snap_tolerance_spin.valueChanged.connect(
            self._preferences_changed
        )
        self.auto_merge_tolerance_spin = QDoubleSpinBox(self)
        self.auto_merge_tolerance_spin.setObjectName("sketchAutoMergeTolerance")
        self.auto_merge_tolerance_spin.setDecimals(9)
        self.auto_merge_tolerance_spin.setRange(0.0, 1.0e6)
        self.auto_merge_tolerance_spin.setValue(preferences.auto_merge_tolerance)
        self.auto_merge_tolerance_spin.valueChanged.connect(
            self._preferences_changed
        )

        self.show_point_ids_check = preference_check(
            "点数字 ID", "sketchShowPointIds", preferences.show_point_ids
        )
        self.show_external_labels_check = preference_check(
            "外部参考标签",
            "sketchShowExternalLabels",
            preferences.show_external_labels,
        )
        self.show_profile_fill_check = preference_check(
            "轮廓填充", "sketchShowProfileFill", preferences.show_profile_fill
        )
        self.show_work_plane_axes_check = preference_check(
            "工作平面坐标轴",
            "sketchShowWorkPlaneAxes",
            preferences.show_work_plane_axes,
        )
        self.continuous_polyline_check = preference_check(
            "连续折线",
            "sketchContinuousPolyline",
            preferences.continuous_polyline,
        )
        self.end_polyline_on_close_check = preference_check(
            "闭合后结束",
            "sketchEndPolylineOnClose",
            preferences.end_polyline_on_close,
        )
        self.keep_tool_after_completion_check = preference_check(
            "完成后保持工具",
            "sketchKeepToolAfterCompletion",
            preferences.keep_tool_after_completion,
        )
        self.confirm_cascade_delete_check = preference_check(
            "级联删除确认",
            "sketchConfirmCascadeDelete",
            preferences.confirm_cascade_delete,
        )
        self.auto_constraints_check = preference_check(
            "自动推断约束", "sketchAutoConstraints", preferences.auto_constraints
        )

        self.point_search_edit = QLineEdit(self)
        self.point_search_edit.setObjectName("sketchPointSearch")
        self.point_search_edit.setPlaceholderText("按点 ID 搜索")
        self.point_search_edit.setClearButtonEnabled(True)
        self.point_filter_combo = QComboBox(self)
        self.point_filter_combo.setObjectName("sketchPointFilter")
        for label, value in (
            ("全部", "all"),
            ("自由", "free"),
            ("已关联", "associated"),
            ("未解析", "unresolved"),
        ):
            self.point_filter_combo.addItem(label, value)
        self.point_search_edit.textChanged.connect(self._point_filter_changed)
        self.point_filter_combo.currentIndexChanged.connect(
            self._point_filter_changed
        )

        self.points_table = QTableWidget(0, 6, self)
        self.points_table.setObjectName("sketchPointsTable")
        self.points_table.setHorizontalHeaderLabels(
            ("ID", "U", "V", "关联", "类型/用途", "依赖曲线")
        )
        self.points_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.points_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.points_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.setSortingEnabled(True)
        self.points_table.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.points_table.installEventFilter(self)
        self.points_table.itemChanged.connect(self._point_item_changed)
        self.points_table.itemSelectionChanged.connect(self._point_row_selected)
        self.points_table.itemDoubleClicked.connect(self._focus_point_item)

        def parameter_spin(object_name: str) -> QDoubleSpinBox:
            editor = QDoubleSpinBox(self)
            editor.setObjectName(object_name)
            editor.setDecimals(6)
            editor.setRange(-1.0e12, 1.0e12)
            return editor

        self.line_parameter_group = QGroupBox("直线参数", self)
        self.line_parameter_group.setObjectName("sketchLineParameters")
        self.line_start_combo = QComboBox(self.line_parameter_group)
        self.line_start_combo.setObjectName("sketchLineStartPoint")
        self.line_end_combo = QComboBox(self.line_parameter_group)
        self.line_end_combo.setObjectName("sketchLineEndPoint")
        self.line_length_spin = parameter_spin("sketchLineLength")
        self.line_length_spin.setMinimum(1.0e-9)
        line_form = QFormLayout(self.line_parameter_group)
        line_form.addRow("起点", self.line_start_combo)
        line_form.addRow("终点", self.line_end_combo)
        line_form.addRow("长度", self.line_length_spin)
        self.line_start_combo.activated.connect(self._line_endpoints_changed)
        self.line_end_combo.activated.connect(self._line_endpoints_changed)
        self.line_length_spin.editingFinished.connect(
            self._line_length_changed
        )

        self.circle_parameter_group = QGroupBox("圆参数", self)
        self.circle_parameter_group.setObjectName("sketchCircleParameters")
        self.circle_radius_spin = parameter_spin("sketchCircleRadius")
        self.circle_radius_spin.setMinimum(1.0e-9)
        circle_form = QFormLayout(self.circle_parameter_group)
        circle_form.addRow("半径", self.circle_radius_spin)
        self.circle_radius_spin.editingFinished.connect(
            self._circle_radius_changed
        )

        self.arc_parameter_group = QGroupBox("圆弧参数", self)
        self.arc_parameter_group.setObjectName("sketchArcParameters")
        self.arc_radius_spin = parameter_spin("sketchArcRadius")
        self.arc_radius_spin.setMinimum(1.0e-9)
        self.arc_start_angle_spin = parameter_spin("sketchArcStartAngle")
        self.arc_start_angle_spin.setRange(-360.0, 360.0)
        self.arc_end_angle_spin = parameter_spin("sketchArcEndAngle")
        self.arc_end_angle_spin.setRange(-360.0, 360.0)
        self.arc_orientation_combo = QComboBox(self.arc_parameter_group)
        self.arc_orientation_combo.setObjectName("sketchArcOrientation")
        self.arc_orientation_combo.addItem("逆时针", "ccw")
        self.arc_orientation_combo.addItem("顺时针", "cw")
        arc_form = QFormLayout(self.arc_parameter_group)
        arc_form.addRow("半径", self.arc_radius_spin)
        arc_form.addRow("起始角 (°)", self.arc_start_angle_spin)
        arc_form.addRow("终止角 (°)", self.arc_end_angle_spin)
        arc_form.addRow("方向", self.arc_orientation_combo)
        self.arc_radius_spin.editingFinished.connect(
            self._arc_radius_changed
        )
        self.arc_start_angle_spin.editingFinished.connect(
            self._arc_start_angle_changed
        )
        self.arc_end_angle_spin.editingFinished.connect(
            self._arc_end_angle_changed
        )
        self.arc_orientation_combo.activated.connect(
            self._arc_orientation_changed
        )
        for group in (
            self.line_parameter_group,
            self.circle_parameter_group,
            self.arc_parameter_group,
        ):
            group.hide()

        self.delete_button = QPushButton("删除所选", self)
        self.delete_button.clicked.connect(self.delete_selected)
        self.release_association_button = QPushButton("解除关联", self)
        self.release_association_button.setObjectName("sketchReleaseAssociationButton")
        self.release_association_button.clicked.connect(self.release_selected_association)
        self.undo_button = QPushButton("撤销", self)
        self.redo_button = QPushButton("重做", self)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        self.constraint_type_combo = QComboBox(self)
        self.constraint_type_combo.setObjectName("sketchConstraintType")
        for text, value in (
            ("重合", "coincident"), ("点在曲线上", "point_on_curve"),
            ("水平", "horizontal"), ("垂直", "vertical"), ("固定", "fixed"),
            ("两点距离 / 直线长度", "distance"), ("圆 / 圆弧半径", "radius"),
        ):
            self.constraint_type_combo.addItem(text, value)
        self.constraint_targets_edit = QLineEdit(self)
        self.constraint_targets_edit.setObjectName("sketchConstraintTargets")
        self.constraint_targets_edit.setPlaceholderText(
            "目标稳定 ID（逗号分隔；留空使用当前选择）"
        )
        self.constraint_driving_check = QCheckBox("驱动尺寸", self)
        self.constraint_driving_check.setChecked(True)
        self.constraint_value_spin = QDoubleSpinBox(self)
        self.constraint_value_spin.setObjectName("sketchConstraintValue")
        self.constraint_value_spin.setDecimals(6)
        self.constraint_value_spin.setRange(1.0e-12, 1.0e12)
        self.add_constraint_button = QPushButton("添加约束", self)
        self.add_constraint_button.clicked.connect(self._add_selected_constraint)
        self.constraints_table = QTableWidget(0, 3, self)
        self.constraints_table.setObjectName("sketchConstraintsTable")
        self.constraints_table.setHorizontalHeaderLabels(("ID", "类型", "对象"))
        self.constraints_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.delete_constraint_button = QPushButton("删除约束", self)
        self.delete_constraint_button.clicked.connect(self.delete_selected_constraint)
        self.edit_constraint_button = QPushButton("编辑尺寸", self)
        self.edit_constraint_button.setObjectName("sketchEditDimensionButton")
        self.edit_constraint_button.clicked.connect(self._edit_selected_constraint)
        self.solve_status_label = QLabel(self)
        self.solve_status_label.setObjectName("sketchSolveStatus")
        self.solve_status_label.setWordWrap(True)

        self.diagnostic_label = QLabel(self)
        self.diagnostic_label.setObjectName("sketchDiagnostics")
        self.diagnostic_label.setWordWrap(True)
        self.diagnostic_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.diagnostic_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.diagnostic_scroll = QScrollArea(self)
        self.diagnostic_scroll.setObjectName("sketchDiagnosticsScroll")
        self.diagnostic_scroll.setWidgetResizable(True)
        self.diagnostic_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.diagnostic_scroll.setMinimumHeight(64)
        self.diagnostic_scroll.setMaximumHeight(150)
        self.diagnostic_scroll.setWidget(self.diagnostic_label)

        self.finish_button = QPushButton("完成草图", self)
        self.finish_button.setObjectName("sketchFinishButton")
        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.setObjectName("sketchCancelButton")
        self.finish_button.clicked.connect(self.try_finish)
        self.cancel_button.clicked.connect(self.cancelRequested.emit)

        form = QFormLayout()
        form.addRow("草图名称", self.name_edit)
        self.work_plane_label = QLabel("全局 XY", self)
        self.work_plane_label.setObjectName("sketchWorkPlaneLabel")
        form.addRow("工作平面", self.work_plane_label)

        grid_group = QGroupBox("网格", self)
        grid_layout = QFormLayout(grid_group)
        grid_switches = QHBoxLayout()
        grid_switches.addWidget(self.grid_visible_check)
        grid_switches.addWidget(self.snap_check)
        grid_layout.addRow(grid_switches)
        grid_layout.addRow("间距", self.spacing_spin)

        snap_group = QGroupBox("捕捉", self)
        snap_layout = QFormLayout(snap_group)
        snap_categories = QHBoxLayout()
        for control in (
            self.snap_sketch_points_check,
            self.snap_external_points_check,
            self.snap_midpoints_check,
        ):
            snap_categories.addWidget(control)
        snap_more_categories = QHBoxLayout()
        for control in (
            self.snap_centers_check,
            self.snap_intersections_check,
        ):
            snap_more_categories.addWidget(control)
        snap_layout.addRow(snap_categories)
        snap_layout.addRow(snap_more_categories)
        snap_layout.addRow("屏幕容差", self.screen_snap_tolerance_spin)
        snap_layout.addRow("自动合并容差", self.auto_merge_tolerance_spin)

        display_group = QGroupBox("显示", self)
        display_layout = QVBoxLayout(display_group)
        display_first = QHBoxLayout()
        display_first.addWidget(self.show_point_ids_check)
        display_first.addWidget(self.show_external_labels_check)
        display_second = QHBoxLayout()
        display_second.addWidget(self.show_profile_fill_check)
        display_second.addWidget(self.show_work_plane_axes_check)
        display_layout.addLayout(display_first)
        display_layout.addLayout(display_second)

        behavior_group = QGroupBox("绘图行为", self)
        behavior_layout = QVBoxLayout(behavior_group)
        behavior_first = QHBoxLayout()
        behavior_first.addWidget(self.continuous_polyline_check)
        behavior_first.addWidget(self.end_polyline_on_close_check)
        behavior_second = QHBoxLayout()
        behavior_second.addWidget(self.keep_tool_after_completion_check)
        behavior_second.addWidget(self.confirm_cascade_delete_check)
        behavior_layout.addWidget(self.auto_constraints_check)
        behavior_layout.addLayout(behavior_first)
        behavior_layout.addLayout(behavior_second)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.delete_button)
        edit_row.addWidget(self.release_association_button)
        edit_row.addWidget(self.undo_button)
        edit_row.addWidget(self.redo_button)
        bottom = QHBoxLayout()
        bottom.addWidget(self.finish_button)
        bottom.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(grid_group)
        layout.addWidget(snap_group)
        layout.addWidget(display_group)
        layout.addWidget(behavior_group)
        layout.addWidget(QLabel("绘图工具", self))
        layout.addLayout(first_modes)
        layout.addLayout(second_modes)
        layout.addWidget(QLabel("点坐标", self))
        point_tools = QHBoxLayout()
        point_tools.addWidget(self.point_search_edit, 2)
        point_tools.addWidget(self.point_filter_combo, 1)
        layout.addLayout(point_tools)
        layout.addWidget(self.points_table, 1)
        layout.addWidget(self.line_parameter_group)
        layout.addWidget(self.circle_parameter_group)
        layout.addWidget(self.arc_parameter_group)
        constraint_group = QGroupBox("草图约束与尺寸", self)
        constraint_layout = QVBoxLayout(constraint_group)
        constraint_add = QHBoxLayout()
        constraint_add.addWidget(self.constraint_type_combo, 2)
        constraint_add.addWidget(self.constraint_value_spin, 1)
        constraint_add.addWidget(self.constraint_driving_check)
        constraint_layout.addLayout(constraint_add)
        constraint_layout.addWidget(self.constraint_targets_edit)
        constraint_layout.addWidget(self.add_constraint_button)
        constraint_layout.addWidget(self.constraints_table)
        constraint_layout.addWidget(self.delete_constraint_button)
        constraint_layout.addWidget(self.edit_constraint_button)
        constraint_layout.addWidget(self.solve_status_label)
        layout.addWidget(constraint_group)
        layout.addLayout(edit_row)
        layout.addWidget(self.diagnostic_scroll)
        layout.addLayout(bottom)

    def set_controller(
        self,
        controller: SketchDraftController,
        *,
        base_snapshot: SketchDraftSnapshot | None = None,
    ) -> None:
        if type(controller) is not SketchDraftController:
            raise TypeError("controller must be a SketchDraftController")
        self._controller = controller
        self._base_snapshot = base_snapshot or controller.snapshot()
        self._clear_pending()
        self._refresh()

    def set_reference_points(
        self,
        reference_points: tuple[SketchReferencePoint, ...],
    ) -> None:
        values = tuple(reference_points)
        if any(type(item) is not SketchReferencePoint for item in values):
            raise TypeError("reference_points must contain SketchReferencePoint values")
        self._reference_points = values
        if self._controller is not None:
            self._controller.refresh_external_references(values)
        if self._viewport is not None:
            self._viewport.set_sketch_reference_points(values)
        self._refresh()

    def attach_viewport(self, viewport) -> None:
        if self._viewport is viewport:
            return
        if self._viewport is not None:
            self._disconnect_viewport(self._viewport)
        self._viewport = viewport
        if viewport is None:
            return
        viewport.sketchWorkPlanePointSelected.connect(self._point_from_viewport)
        viewport.sketchReferencePointSelected.connect(self._reference_from_viewport)
        viewport.sketchDraftPointSelectionRequested.connect(self._select_point)
        viewport.sketchDraftCurveSelectionRequested.connect(self._select_curve)
        viewport.sketchDraftProfileSelectionRequested.connect(self._select_profile)
        viewport.sketchTrimRequested.connect(self._trim_from_viewport)
        viewport.sketchAuthoringMissed.connect(self._authoring_missed)
        viewport.sketchPendingInteractionCancelled.connect(
            self._pending_cancelled
        )
        viewport.sketchAuthoringFinishRequested.connect(self.try_finish)
        viewport.sketchAuthoringCancelled.connect(self.cancelRequested.emit)
        viewport.sketchSnapConfirmed.connect(self._snap_confirmed)
        viewport.sketchInferencePreviewChanged.connect(self._inference_hovered)
        viewport.sketchPointDragPreviewRequested.connect(self._drag_preview_requested)
        viewport.sketchPointDragCommitRequested.connect(self._drag_commit_requested)

    def _disconnect_viewport(self, viewport) -> None:
        connections = (
            (viewport.sketchWorkPlanePointSelected, self._point_from_viewport),
            (viewport.sketchReferencePointSelected, self._reference_from_viewport),
            (viewport.sketchDraftPointSelectionRequested, self._select_point),
            (viewport.sketchDraftCurveSelectionRequested, self._select_curve),
            (viewport.sketchDraftProfileSelectionRequested, self._select_profile),
            (viewport.sketchTrimRequested, self._trim_from_viewport),
            (viewport.sketchAuthoringMissed, self._authoring_missed),
            (
                viewport.sketchPendingInteractionCancelled,
                self._pending_cancelled,
            ),
            (viewport.sketchAuthoringFinishRequested, self.try_finish),
            (viewport.sketchAuthoringCancelled, self.cancelRequested.emit),
            (viewport.sketchSnapConfirmed, self._snap_confirmed),
            (viewport.sketchInferencePreviewChanged, self._inference_hovered),
            (viewport.sketchPointDragPreviewRequested, self._drag_preview_requested),
            (viewport.sketchPointDragCommitRequested, self._drag_commit_requested),
        )
        for signal, slot in connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    @property
    def authoring_purpose(self) -> str:
        return self._authoring_purpose

    def begin(self, viewport, *, purpose: str = "geometry") -> None:
        if self._controller is None:
            raise RuntimeError("sketch editor requires a draft controller")
        normalized = str(purpose).strip().casefold()
        if normalized not in {"geometry", "planar_boolean_tool", "face_sketch"}:
            raise ValueError("unsupported sketch authoring purpose")
        self._authoring_purpose = normalized
        self.finish_button.setText(
            {
                "geometry": "完成草图",
                "planar_boolean_tool": "完成工具草图",
                "face_sketch": "创建",
            }[normalized]
        )
        self.attach_viewport(viewport)
        self.show()
        self._apply_preferences_to_viewport()
        viewport.start_sketch_authoring(
            self.render_data(),
            snap=self.snap_check.isChecked(),
            spacing=self.spacing_spin.value(),
            reference_points=self._reference_points,
        )
        self.set_mode("polyline")

    def end(self) -> None:
        if self._viewport is not None:
            self._viewport.stop_sketch_authoring()
        self.hide()
        self._clear_pending()
        self._authoring_purpose = "geometry"
        self.finish_button.setText("完成草图")

    def set_mode(self, mode: str) -> None:
        normalized = str(mode).strip().casefold()
        if normalized not in self._mode_buttons:
            raise ValueError("unsupported sketch editor mode")
        for name, button in self._mode_buttons.items():
            button.blockSignals(True)
            button.setChecked(name == normalized)
            button.blockSignals(False)
        self._clear_pending()
        if self._viewport is not None:
            self._viewport.set_sketch_authoring_mode(normalized)
        self._set_status(f"草图工具：{self._mode_buttons[normalized].text()}")

    @property
    def mode(self) -> str:
        return next(
            name
            for name, button in self._mode_buttons.items()
            if button.isChecked()
        )

    def render_data(self) -> SketchDraftRenderData:
        controller = self._require_controller()
        snapshot = controller.snapshot()
        solve_result = controller.current_solve_result()
        overlays = build_constraint_overlays(
            snapshot.points,
            snapshot.curves,
            snapshot.constraints,
            snapshot.plane,
            warning_ids=(
                *solve_result.redundant_constraint_ids,
                *solve_result.conflicting_constraint_ids,
            ),
        )
        snap_midpoints, snap_centers = self._derived_snap_points(snapshot)
        selected_kind = snapshot.selected_kind
        selected_id = snapshot.selected_ids[0] if snapshot.selected_ids else None
        if controller.profiles:
            preview = build_strict_sketch_draft_preview(
                SketchGeometry(
                    snapshot.name or "草图预览",
                    snapshot.plane,
                    snapshot.points,
                    snapshot.curves,
                )
            )
            point_ids = tuple(
                (
                    logical_id.removeprefix("point:")
                    if logical_id is not None
                    and logical_id.startswith("point:")
                    and logical_id.removeprefix("point:")
                    in {point.id for point in snapshot.points}
                    else None
                )
                for logical_id in preview.point_logical_ids
            )
            curve_lookup = {curve.id for curve in snapshot.curves}
            curves: list[tuple[int, ...]] = []
            curve_ids: list[str] = []
            for cell, logical_id in zip(
                preview.edges,
                preview.edge_logical_ids,
                strict=True,
            ):
                if logical_id is None or not logical_id.startswith("edge:"):
                    continue
                curve_id = logical_id.removeprefix("edge:")
                if curve_id in curve_lookup:
                    curves.append(cell)
                    curve_ids.append(curve_id)
            faces: list[tuple[int, ...]] = []
            face_ids: list[str] = []
            for cell, logical_id in zip(
                preview.faces,
                preview.face_logical_ids,
                strict=True,
            ):
                if logical_id is None or not logical_id.startswith("face:profile/"):
                    continue
                faces.append(cell)
                face_ids.append(logical_id.removeprefix("face:"))
            return SketchDraftRenderData(
                preview.points,
                point_ids,
                tuple(curves),
                tuple(curve_ids),
                tuple(faces),
                tuple(face_ids),
                selected_kind=(
                    "curve" if selected_kind == "edge" else selected_kind
                ),
                selected_id=selected_id,
                plane=snapshot.plane,
                selected_ids=snapshot.selected_ids,
                snap_midpoints=snap_midpoints,
                snap_centers=snap_centers,
                geometry_revision=snapshot.revision,
                analytic_points=snapshot.points,
                analytic_curves=snapshot.curves,
                constraint_overlays=overlays,
                inference_preview=self._inference_preview,
            )
        return self._incomplete_render_data(snapshot, selected_kind, selected_id)

    @staticmethod
    def _derived_snap_points(
        snapshot: SketchDraftSnapshot,
    ) -> tuple[
        tuple[tuple[float, float, float], ...],
        tuple[tuple[float, float, float], ...],
    ]:
        point_map = {point.id: point for point in snapshot.points}
        midpoints = tuple(
            snapshot.plane.to_global(
                0.5
                * (
                    point_map[curve.start_point_id].u
                    + point_map[curve.end_point_id].u
                ),
                0.5
                * (
                    point_map[curve.start_point_id].v
                    + point_map[curve.end_point_id].v
                ),
            )
            for curve in snapshot.curves
            if isinstance(curve, SketchLine)
        )
        centers = tuple(
            snapshot.plane.to_global(
                point_map[curve.center_point_id].u,
                point_map[curve.center_point_id].v,
            )
            for curve in snapshot.curves
            if isinstance(curve, (SketchCircle, SketchArc))
        )
        return midpoints, centers

    def _incomplete_render_data(
        self,
        snapshot: SketchDraftSnapshot,
        selected_kind: str | None,
        selected_id: str | None,
    ) -> SketchDraftRenderData:
        snap_midpoints, snap_centers = self._derived_snap_points(snapshot)
        points = [
            snapshot.plane.to_global(point.u, point.v)
            for point in snapshot.points
        ]
        point_ids: list[str | None] = [point.id for point in snapshot.points]
        point_index = {
            point.id: index for index, point in enumerate(snapshot.points)
        }

        def add_point(u: float, v: float) -> int:
            index = len(points)
            points.append(snapshot.plane.to_global(u, v))
            point_ids.append(None)
            return index

        curves: list[tuple[int, ...]] = []
        curve_ids: list[str] = []
        point_map = {point.id: point for point in snapshot.points}
        for curve in snapshot.curves:
            if isinstance(curve, SketchLine):
                cell = (
                    point_index[curve.start_point_id],
                    point_index[curve.end_point_id],
                )
            elif isinstance(curve, SketchCircle):
                center = point_map[curve.center_point_id]
                ring = tuple(
                    add_point(
                        center.u + curve.radius * math.cos(2.0 * math.pi * i / 48),
                        center.v + curve.radius * math.sin(2.0 * math.pi * i / 48),
                    )
                    for i in range(48)
                )
                cell = ring + (ring[0],)
            elif isinstance(curve, SketchArc):
                start = point_map[curve.start_point_id]
                center = point_map[curve.center_point_id]
                end = point_map[curve.end_point_id]
                start_angle = math.atan2(start.v - center.v, start.u - center.u)
                end_angle = math.atan2(end.v - center.v, end.u - center.u)
                sweep = (
                    (end_angle - start_angle) % (2.0 * math.pi)
                    if curve.orientation == "ccw"
                    else -((start_angle - end_angle) % (2.0 * math.pi))
                )
                radius = math.hypot(start.u - center.u, start.v - center.v)
                count = max(8, int(math.ceil(48 * abs(sweep) / (2.0 * math.pi))))
                values = [point_index[curve.start_point_id]]
                values.extend(
                    add_point(
                        center.u + radius * math.cos(start_angle + sweep * i / count),
                        center.v + radius * math.sin(start_angle + sweep * i / count),
                    )
                    for i in range(1, count)
                )
                values.append(point_index[curve.end_point_id])
                cell = tuple(values)
            else:  # pragma: no cover - strict snapshot validates curve types
                continue
            curves.append(cell)
            curve_ids.append(curve.id)
        return SketchDraftRenderData(
            tuple(points),
            tuple(point_ids),
            tuple(curves),
            tuple(curve_ids),
            selected_kind=(
                "curve" if selected_kind == "edge" else selected_kind
            ),
            selected_id=selected_id,
            plane=snapshot.plane,
            selected_ids=snapshot.selected_ids,
            snap_midpoints=snap_midpoints,
            snap_centers=snap_centers,
            geometry_revision=snapshot.revision,
            analytic_points=snapshot.points,
            analytic_curves=snapshot.curves,
            constraint_overlays=build_constraint_overlays(
                snapshot.points, snapshot.curves, snapshot.constraints, snapshot.plane
            ),
            inference_preview=self._inference_preview,
        )

    def _point_from_viewport(
        self,
        global_point: tuple[float, float, float],
    ) -> None:
        controller = self._require_controller()
        u, v = controller.plane.to_local(tuple(global_point))
        reference_point = self._incoming_reference
        self._incoming_reference = None
        self._inference_preview = ()
        try:
            if self.mode == "polyline":
                closed = False
                segment_completed = False
                point_id = self._point_id_at(u, v)
                if self._polyline_start_id is None and point_id is None:
                    point_id = controller.add_point(
                        u,
                        v,
                        external_reference=reference_point,
                    ).id
                if self._polyline_start_id is None:
                    if point_id is None:  # pragma: no cover - created above
                        raise RuntimeError("polyline start point was not created")
                    self._polyline_start_id = point_id
                    self._polyline_first_id = point_id
                elif point_id is None:
                    start = next(
                        point for point in controller.snapshot().points
                        if point.id == self._polyline_start_id
                    )
                    inference = infer_line_preview(
                        (start.u, start.v), (u, v),
                        auto_constraints=self._preferences.auto_constraints,
                        snap_kind=str(self._incoming_snap.get("kind") or "") or None,
                        snapped_point_id=self._incoming_snap.get("point_id") if isinstance(self._incoming_snap.get("point_id"), str) else None,
                        intersection_curve_ids=tuple(self._incoming_snap.get("curve_ids", ())),
                    )
                    if "horizontal" in inference.kinds:
                        v = start.v
                    elif "vertical" in inference.kinds:
                        u = start.u
                    line, result = controller.add_inferred_line(
                        self._polyline_start_id,
                        (u, v),
                        horizontal="horizontal" in inference.kinds,
                        vertical="vertical" in inference.kinds,
                        intersection_curve_ids=inference.intersection_curve_ids,
                    )
                    if line is None:
                        self._set_status(solve_status_text(result))
                        self._clear_pending()
                        self._refresh()
                        return
                    point_id = line.end_point_id
                    self._polyline_start_id = point_id
                    segment_completed = True
                elif point_id != self._polyline_start_id:
                    start = next(point for point in controller.snapshot().points if point.id == self._polyline_start_id)
                    end = next(point for point in controller.snapshot().points if point.id == point_id)
                    inference = infer_line_preview(
                        (start.u, start.v), (end.u, end.v),
                        auto_constraints=self._preferences.auto_constraints,
                        snap_kind="sketch_point",
                        snapped_point_id=point_id,
                    )
                    _line, result = controller.add_inferred_line(
                        self._polyline_start_id,
                        point_id,
                        horizontal="horizontal" in inference.kinds,
                        vertical="vertical" in inference.kinds,
                    )
                    if _line is None:
                        self._set_status(solve_status_text(result))
                        self._clear_pending()
                        self._refresh()
                        return
                    segment_completed = True
                    closed = point_id == self._polyline_first_id
                    self._polyline_start_id = point_id
                chain_ended = segment_completed and (
                    not self._preferences.continuous_polyline
                    or (closed and self._preferences.end_polyline_on_close)
                )
                if chain_ended:
                    self._clear_pending()
                    self._finish_shape_tool()
                else:
                    self._pending_points = [(u, v)]
            elif self.mode == "rectangle":
                self._pending_points.append((u, v))
                self._pending_references.append(reference_point)
                if len(self._pending_points) == 2:
                    first, second = self._pending_points
                    left, right = sorted((first[0], second[0]))
                    bottom, top = sorted((first[1], second[1]))
                    corners = (
                        (left, bottom),
                        (right, bottom),
                        (right, top),
                        (left, top),
                    )
                    clicked = tuple(
                        zip(
                            self._pending_points,
                            self._pending_references,
                            strict=True,
                        )
                    )
                    references = tuple(
                        next(
                            (
                                item
                                for coordinate, item in clicked
                                if coordinate == corner
                            ),
                            None,
                        )
                        for corner in corners
                    )
                    controller.add_rectangle(
                        *self._pending_points,
                        external_references=references,
                    )
                    self._clear_pending()
                    self._finish_shape_tool()
            elif self.mode == "circle":
                self._pending_points.append((u, v))
                self._pending_references.append(reference_point)
                if len(self._pending_points) == 2:
                    center, rim = self._pending_points
                    radius = math.hypot(rim[0] - center[0], rim[1] - center[1])
                    controller.add_circle(
                        center,
                        radius,
                        external_reference=self._pending_references[0],
                    )
                    self._clear_pending()
                    self._finish_shape_tool()
            elif self.mode == "arc":
                self._pending_points.append((u, v))
                self._pending_references.append(reference_point)
                if len(self._pending_points) == 3:
                    controller.add_arc(
                        *self._pending_points,
                        start_external_reference=self._pending_references[0],
                        center_external_reference=self._pending_references[1],
                        end_external_reference=self._pending_references[2],
                    )
                    self._clear_pending()
                    self._finish_shape_tool()
        except (TypeError, ValueError) as error:
            self._set_status(str(error))
            self._clear_pending()
        self._incoming_snap = {}
        self._refresh()

    def _reference_from_viewport(
        self,
        reference_point: SketchReferencePoint | None,
    ) -> None:
        self._incoming_reference = reference_point

    def _point_id_at(self, u: float, v: float) -> str | None:
        controller = self._require_controller()
        tolerance = self._preferences.auto_merge_tolerance
        for point in controller.snapshot().points:
            if math.hypot(point.u - u, point.v - v) <= tolerance:
                return point.id
        return None

    def _finish_shape_tool(self) -> None:
        if not self._preferences.keep_tool_after_completion:
            self.set_mode("select")

    def _select_point(
        self,
        point_id: str,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        controller = self._require_controller()
        self._ensure_point_visible(point_id)
        visible_ids = self._visible_point_ids()
        current_ids = (
            controller.selected_ids
            if controller.snapshot().selected_kind == "point"
            else ()
        )
        control = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if shift and self._selection_anchor_id in visible_ids:
            start = visible_ids.index(self._selection_anchor_id)
            end = visible_ids.index(point_id)
            span = visible_ids[min(start, end) : max(start, end) + 1]
            selected_ids = tuple(dict.fromkeys((*current_ids, *span))) if control else span
            controller.select_many(list(selected_ids))
        elif control:
            selected_ids = tuple(item for item in current_ids if item != point_id)
            if point_id not in current_ids:
                selected_ids = (*selected_ids, point_id)
            controller.select_many(list(selected_ids))
            self._selection_anchor_id = point_id
        else:
            controller.select_point(point_id)
            self._selection_anchor_id = point_id
        self._sync_point_table_selection(scroll_to_id=point_id)
        self._selection_changed_lightweight()

    def _select_curve(
        self,
        curve_id: str,
        _modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        self._require_controller().select_curve(curve_id)
        self._selection_anchor_id = None
        self._sync_point_table_selection()
        self._selection_changed_lightweight()

    def _select_profile(
        self,
        profile_id: str,
        _modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        self._require_controller().select_profile(profile_id)
        self._selection_anchor_id = None
        self._sync_point_table_selection()
        self._selection_changed_lightweight()

    def _trim_from_viewport(
        self,
        curve_id: str,
        global_point: tuple[float, float, float],
    ) -> None:
        controller = self._require_controller()
        try:
            replacements = controller.trim_curve(
                curve_id,
                controller.plane.to_local(tuple(global_point)),
            )
        except (TypeError, ValueError) as error:
            self._set_status(str(error))
        else:
            self._set_status(
                "没有可用的分割交点，已删除整条曲线"
                if not replacements
                else "已修剪鼠标所在的曲线段"
            )
        self._refresh()

    def delete_selected(self) -> None:
        controller = self._require_controller()
        if not controller.selected_ids:
            self._set_status("请先用“选择”工具选中要删除的点或曲线")
            return
        entity_ids = controller.selected_ids
        if controller.snapshot().selected_kind == "point":
            dependent_ids = tuple(
                dict.fromkeys(
                    curve_id
                    for entity_id in entity_ids
                    for curve_id in controller.dependent_curve_ids(entity_id)
                )
            )
            if dependent_ids and self._preferences.confirm_cascade_delete:
                answer = QMessageBox.question(
                    self,
                    "删除草图点" if len(entity_ids) == 1 else "删除多个草图点",
                    (
                        f"删除点 {', '.join(entity_ids)} 将同时删除依赖曲线：\n"
                        + "\n".join(dependent_ids)
                        + "\n\n是否继续？"
                    ),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    self._set_status("已取消删除")
                    return
        try:
            controller.delete_many(list(entity_ids))
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))
        else:
            self._set_status(f"已删除草图实体 {', '.join(entity_ids)}")
        self._refresh()

    def create_constraint(
        self,
        kind: str,
        entity_ids: tuple[str, ...],
        *,
        value: float | None = None,
        driving: bool = True,
    ):
        """Create any first-release relation through the controller solver path."""

        controller = self._require_controller()
        snapshot = controller.snapshot()
        point_map = {point.id: point for point in snapshot.points}
        curve_map = {curve.id: curve for curve in snapshot.curves}
        targets = tuple(str(item).strip() for item in entity_ids)
        if any(not item for item in targets):
            raise ValueError("约束目标无效：实体 ID 不能为空")
        used = {item.id.casefold() for item in snapshot.constraints}
        index = 1
        while f"C{index}".casefold() in used:
            index += 1
        constraint_id = f"C{index}"
        normalized = str(kind).strip().casefold()
        if normalized == "coincident":
            if len(targets) != 2 or any(item not in point_map for item in targets):
                raise ValueError("约束目标无效：重合约束需要两个草图点")
            constraint = SketchCoincidentConstraint(constraint_id, *targets)
        elif normalized == "point_on_curve":
            if (
                len(targets) != 2
                or targets[0] not in point_map
                or targets[1] not in curve_map
            ):
                raise ValueError("约束目标无效：点在曲线上需要一个点 ID 和一个曲线 ID")
            constraint = SketchPointOnCurveConstraint(constraint_id, *targets)
        elif normalized == "horizontal":
            if len(targets) != 1 or not isinstance(curve_map.get(targets[0]), SketchLine):
                raise ValueError("约束目标无效：水平约束需要一条直线")
            constraint = SketchHorizontalConstraint(constraint_id, targets[0])
        elif normalized == "vertical":
            if len(targets) != 1 or not isinstance(curve_map.get(targets[0]), SketchLine):
                raise ValueError("约束目标无效：垂直约束需要一条直线")
            constraint = SketchVerticalConstraint(constraint_id, targets[0])
        elif normalized == "fixed":
            if len(targets) != 1 or targets[0] not in point_map:
                raise ValueError("约束目标无效：固定约束需要一个草图点")
            point = point_map[targets[0]]
            constraint = SketchFixedConstraint(
                constraint_id, point.id, point.u, point.v
            )
        elif normalized == "distance":
            ids = targets
            if len(ids) == 1 and isinstance(curve_map.get(ids[0]), SketchLine):
                line = curve_map[ids[0]]
                ids = (line.start_point_id, line.end_point_id)
            if len(ids) != 2 or any(item not in point_map for item in ids):
                raise ValueError("约束目标无效：距离尺寸需要两个点或一条直线")
            measured = math.hypot(
                point_map[ids[1]].u - point_map[ids[0]].u,
                point_map[ids[1]].v - point_map[ids[0]].v,
            )
            constraint = SketchDistanceDimension(
                constraint_id, *ids, measured if value is None or not driving else value,
                driving=driving,
            )
        elif normalized == "radius":
            if len(targets) != 1:
                raise ValueError("约束目标无效：半径尺寸需要一个圆或圆弧")
            curve = curve_map.get(targets[0])
            if isinstance(curve, SketchCircle):
                measured = curve.radius
            elif isinstance(curve, SketchArc):
                center = point_map[curve.center_point_id]
                start = point_map[curve.start_point_id]
                measured = math.hypot(start.u - center.u, start.v - center.v)
            else:
                raise ValueError("约束目标无效：半径尺寸只适用于圆或圆弧")
            constraint = SketchRadiusDimension(
                constraint_id, curve.id,
                measured if value is None or not driving else value,
                driving=driving,
            )
        else:
            raise ValueError("不支持的草图约束类型")
        result = controller.add_constraint_and_solve(constraint)
        self._set_status(solve_status_text(result))
        if result.succeeded:
            self._refresh()
            return constraint
        return None

    def edit_dimension(
        self, constraint_id: str, *, value: float, driving: bool
    ) -> bool:
        controller = self._require_controller()
        constraint = next(
            item for item in controller.constraints if item.id == constraint_id
        )
        if not isinstance(constraint, (SketchDistanceDimension, SketchRadiusDimension)):
            raise ValueError("所选约束不是尺寸")
        snapshot = controller.snapshot()
        measured = measured_dimension_value(
            constraint,
            {point.id: point for point in snapshot.points},
            {curve.id: curve for curve in snapshot.curves},
        )
        replacement = type(constraint)(
            **{
                field: getattr(constraint, field)
                for field in constraint.__dataclass_fields__
                if field not in {"value", "driving"}
            },
            value=value if driving else measured,
            driving=driving,
        )
        result = controller.replace_constraint_and_solve(constraint_id, replacement)
        self._set_status(solve_status_text(result))
        if result.succeeded:
            self._refresh()
        return result.succeeded

    def edit_constraint(self, constraint_id: str, replacement) -> bool:
        """Replace any first-release constraint atomically after a trial solve."""

        result = self._require_controller().replace_constraint_and_solve(
            constraint_id, replacement
        )
        self._set_status(solve_status_text(result))
        if result.succeeded:
            self._refresh()
        return result.succeeded

    def delete_constraint(self, constraint_id: str) -> None:
        self._require_controller().delete_constraint(constraint_id)
        self._set_status(f"已删除草图约束 {constraint_id}")
        self._refresh()

    def preview_constrained_drag(
        self, point_id: str, u: float, v: float
    ):
        """Continuously solve drag candidates without touching draft history."""

        result = self._require_controller().solve_constraints_temporary(
            point_coordinates={point_id: (u, v)}
        )
        self._drag_preview = result
        self._set_status(solve_status_text(result))
        return result

    def commit_constrained_drag(self, point_id: str, u: float, v: float):
        """Commit a successful point drag as exactly one undo record."""

        preview = self.preview_constrained_drag(point_id, u, v)
        if not preview.succeeded:
            self._drag_preview = None
            return preview
        result = self._require_controller().move_points_constrained(
            {point_id: (u, v)}
        )
        self._drag_preview = None
        self._set_status(solve_status_text(result))
        self._refresh(selected_id=point_id)
        return result

    def _drag_preview_requested(self, point_id: str, point: object) -> None:
        u, v = self._require_controller().plane.to_local(tuple(point))
        self.preview_constrained_drag(point_id, u, v)

    def _drag_commit_requested(self, point_id: str, point: object) -> None:
        u, v = self._require_controller().plane.to_local(tuple(point))
        self.commit_constrained_drag(point_id, u, v)

    def delete_selected_constraint(self) -> None:
        row = self.constraints_table.currentRow()
        item = self.constraints_table.item(row, 0) if row >= 0 else None
        if item is None:
            self._set_status("请先选择要删除的草图约束")
            return
        self.delete_constraint(item.text())

    def _edit_selected_constraint(self) -> None:
        row = self.constraints_table.currentRow()
        item = self.constraints_table.item(row, 0) if row >= 0 else None
        if item is None:
            self._set_status("请先选择要编辑的草图约束")
            return
        constraint = next(
            value for value in self._require_controller().constraints
            if value.id == item.text()
        )
        if not isinstance(constraint, (SketchDistanceDimension, SketchRadiusDimension)):
            self._set_status("几何约束可通过 API 更换引用；当前控件用于编辑尺寸")
            return
        try:
            self.edit_dimension(
                constraint.id,
                value=self.constraint_value_spin.value(),
                driving=self.constraint_driving_check.isChecked(),
            )
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))

    def _add_selected_constraint(self) -> None:
        snapshot = self._require_controller().snapshot()
        kind = str(self.constraint_type_combo.currentData())
        typed_targets = tuple(
            item
            for item in re.split(r"[,，;；\s]+", self.constraint_targets_edit.text().strip())
            if item
        )
        entity_ids = typed_targets or snapshot.selected_ids
        try:
            self.create_constraint(
                kind,
                entity_ids,
                value=self.constraint_value_spin.value(),
                driving=self.constraint_driving_check.isChecked(),
            )
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))

    def _snap_confirmed(self, context: object) -> None:
        self._incoming_snap = dict(context) if isinstance(context, dict) else {}

    def _inference_hovered(self, context: object) -> None:
        if not isinstance(context, dict) or self.mode != "polyline" or not self._pending_points:
            self._set_inference_preview(())
            return
        point = context.get("point")
        if point is None:
            self._set_inference_preview(())
            return
        end = self._require_controller().plane.to_local(tuple(point))
        preview = infer_line_preview(
            self._pending_points[-1], end,
            auto_constraints=self._preferences.auto_constraints,
            snap_kind=context.get("snap_kind") if isinstance(context.get("snap_kind"), str) else None,
            snapped_point_id=context.get("point_id") if isinstance(context.get("point_id"), str) else None,
            intersection_curve_ids=tuple(context.get("curve_ids", ())),
        )
        self._set_inference_preview(preview.kinds)

    def _set_inference_preview(self, kinds: tuple[str, ...]) -> None:
        self._inference_preview = tuple(kinds)
        if self._viewport is not None:
            self._viewport.set_sketch_inference_preview(self._inference_preview)

    def release_selected_association(self) -> None:
        controller = self._require_controller()
        snapshot = controller.snapshot()
        point_id = (
            snapshot.selected_ids[0]
            if snapshot.selected_ids and snapshot.selected_kind == "point"
            else None
        )
        if point_id is None:
            self._set_status("请先选择一个关联草图点")
            return
        if controller.external_reference_for_point(point_id) is None:
            self._set_status("所选草图点没有外部关联")
            return
        controller.release_point_association(point_id)
        self._set_status(f"已解除草图点 {point_id} 的关联，当前位置保持不变")
        self._refresh(selected_id=point_id)

    def undo(self) -> None:
        self._require_controller().undo()
        self._clear_pending()
        self._refresh()

    def redo(self) -> None:
        self._require_controller().redo()
        self._clear_pending()
        self._refresh()

    def try_finish(self) -> None:
        controller = self._require_controller()
        try:
            controller.to_sketch_geometry()
        except SketchDraftValidationError as error:
            self._set_status(str(error))
            self._refresh()
            return
        self.finishRequested.emit()

    def show_status(self, message: str) -> None:
        self._set_status(message)

    def _name_changed(self) -> None:
        if self._refreshing:
            return
        self._require_controller().set_sketch_name(self.name_edit.text())
        self._refresh()

    def _grid_changed(self, *_args) -> None:
        self._preferences_changed()

    def _current_preferences(self) -> SketchPreferences:
        return SketchPreferences(
            grid_visible=self.grid_visible_check.isChecked(),
            grid_snap=self.snap_check.isChecked(),
            grid_spacing=self.spacing_spin.value(),
            snap_sketch_points=self.snap_sketch_points_check.isChecked(),
            snap_external_points=self.snap_external_points_check.isChecked(),
            snap_midpoints=self.snap_midpoints_check.isChecked(),
            snap_centers=self.snap_centers_check.isChecked(),
            snap_intersections=self.snap_intersections_check.isChecked(),
            screen_snap_tolerance=self.screen_snap_tolerance_spin.value(),
            auto_merge_tolerance=self.auto_merge_tolerance_spin.value(),
            show_point_ids=self.show_point_ids_check.isChecked(),
            show_external_labels=self.show_external_labels_check.isChecked(),
            show_profile_fill=self.show_profile_fill_check.isChecked(),
            show_work_plane_axes=self.show_work_plane_axes_check.isChecked(),
            continuous_polyline=self.continuous_polyline_check.isChecked(),
            end_polyline_on_close=self.end_polyline_on_close_check.isChecked(),
            keep_tool_after_completion=(
                self.keep_tool_after_completion_check.isChecked()
            ),
            confirm_cascade_delete=self.confirm_cascade_delete_check.isChecked(),
            auto_constraints=self.auto_constraints_check.isChecked(),
        ).normalized()

    def _preferences_changed(self, *_args) -> None:
        self._preferences = self._current_preferences()
        if self._settings is not None:
            save_sketch_preferences(self._settings, self._preferences)
        self._apply_preferences_to_viewport()

    def _apply_preferences_to_viewport(self) -> None:
        if self._viewport is None:
            return
        preferences = self._preferences
        self._viewport.set_sketch_grid(
            visible=preferences.grid_visible,
            snap=preferences.grid_snap,
            spacing=preferences.grid_spacing,
        )
        self._viewport.set_sketch_preferences(
            snap_sketch_points=preferences.snap_sketch_points,
            snap_external_points=preferences.snap_external_points,
            snap_midpoints=preferences.snap_midpoints,
            snap_centers=preferences.snap_centers,
            snap_intersections=preferences.snap_intersections,
            screen_snap_tolerance=preferences.screen_snap_tolerance,
            show_point_ids=preferences.show_point_ids,
            show_external_labels=preferences.show_external_labels,
            show_profile_fill=preferences.show_profile_fill,
            show_work_plane_axes=preferences.show_work_plane_axes,
        )

    def _point_item_changed(self, item: QTableWidgetItem) -> None:
        if self._refreshing or item.column() not in {1, 2}:
            return
        controller = self._require_controller()
        id_item = self.points_table.item(item.row(), 0)
        if id_item is None:
            return
        point_id = id_item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(point_id, str):
            return
        try:
            controller.move_point(
                point_id,
                u=float(self.points_table.item(item.row(), 1).text()),
                v=float(self.points_table.item(item.row(), 2).text()),
            )
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh(selected_id=point_id)

    def _point_row_selected(self) -> None:
        if self._refreshing:
            return
        selected_ids = tuple(
            point_id
            for index in self.points_table.selectionModel().selectedRows()
            if (point_id := self._point_id_for_row(index.row())) is not None
        )
        controller = self._require_controller()
        if selected_ids:
            controller.select_many(list(selected_ids))
            current_id = self._point_id_for_row(self.points_table.currentRow())
            if current_id is not None:
                self._selection_anchor_id = current_id
        else:
            controller.clear_selection()
            self._selection_anchor_id = None
        self._selection_changed_lightweight()

    def _focus_point_item(self, item: QTableWidgetItem) -> None:
        point_id = self._point_id_for_row(item.row())
        if point_id is None:
            return
        self._require_controller().select_point(point_id)
        self._selection_anchor_id = point_id
        self._sync_point_table_selection(scroll_to_id=point_id)
        self._selection_changed_lightweight()
        self.entityFocusRequested.emit("point", point_id)

    def eventFilter(self, watched: object, event: object) -> bool:
        if watched is self.points_table and event.type() == QEvent.Type.KeyPress:
            editing = (
                self.points_table.state()
                == QAbstractItemView.State.EditingState
            )
            if event.key() == Qt.Key.Key_Delete and not editing:
                self.delete_selected()
                return True
        return super().eventFilter(watched, event)

    def _selected_curve(self):
        snapshot = self._require_controller().snapshot()
        selected_id = (
            snapshot.selected_ids[0]
            if snapshot.selected_ids and snapshot.selected_kind == "edge"
            else None
        )
        return next(
            (
                curve
                for curve in snapshot.curves
                if curve.id == selected_id
            ),
            None,
        )

    def _line_endpoints_changed(self, _index: int = -1) -> None:
        if self._refreshing:
            return
        curve = self._selected_curve()
        if not isinstance(curve, SketchLine):
            return
        try:
            self._require_controller().update_curve_parameters(
                curve.id,
                start_point_id=str(self.line_start_combo.currentData()),
                end_point_id=str(self.line_end_combo.currentData()),
            )
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh(selected_id=curve.id)

    def _line_length_changed(self) -> None:
        if self._refreshing:
            return
        curve = self._selected_curve()
        if not isinstance(curve, SketchLine):
            return
        points = {
            point.id: point
            for point in self._require_controller().snapshot().points
        }
        start = points[curve.start_point_id]
        end = points[curve.end_point_id]
        delta_u = end.u - start.u
        delta_v = end.v - start.v
        current_length = math.hypot(delta_u, delta_v)
        if math.isclose(current_length, 0.0):
            self._set_status("重合端点无法定义直线长度")
            self._refresh(selected_id=curve.id)
            return
        target = self.line_length_spin.value()
        scale = target / current_length
        try:
            self._require_controller().move_point(
                curve.end_point_id,
                start.u + delta_u * scale,
                start.v + delta_v * scale,
            )
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh(selected_id=curve.id)

    def _circle_radius_changed(self) -> None:
        if self._refreshing:
            return
        curve = self._selected_curve()
        if not isinstance(curve, SketchCircle):
            return
        try:
            self._require_controller().update_curve_parameters(
                curve.id,
                radius=self.circle_radius_spin.value(),
            )
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh(selected_id=curve.id)

    def _arc_radius_changed(self) -> None:
        if self._refreshing:
            return
        curve = self._selected_curve()
        if not isinstance(curve, SketchArc):
            return
        controller = self._require_controller()
        points = {point.id: point for point in controller.snapshot().points}
        center = points[curve.center_point_id]
        target = self.arc_radius_spin.value()
        coordinates: dict[str, tuple[float, float]] = {}
        try:
            for point_id in (
                curve.start_point_id,
                curve.end_point_id,
            ):
                point = points[point_id]
                angle = math.atan2(
                    point.v - center.v,
                    point.u - center.u,
                )
                coordinates[point_id] = (
                    center.u + target * math.cos(angle),
                    center.v + target * math.sin(angle),
                )
            controller.move_points(coordinates)
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh(selected_id=curve.id)

    def _arc_start_angle_changed(self) -> None:
        self._move_arc_endpoint_to_angle(
            "start",
            self.arc_start_angle_spin.value(),
        )

    def _arc_end_angle_changed(self) -> None:
        self._move_arc_endpoint_to_angle(
            "end",
            self.arc_end_angle_spin.value(),
        )

    def _move_arc_endpoint_to_angle(
        self,
        endpoint: str,
        angle_degrees: float,
    ) -> None:
        if self._refreshing:
            return
        curve = self._selected_curve()
        if not isinstance(curve, SketchArc):
            return
        controller = self._require_controller()
        points = {point.id: point for point in controller.snapshot().points}
        center = points[curve.center_point_id]
        point_id = (
            curve.start_point_id
            if endpoint == "start"
            else curve.end_point_id
        )
        point = points[point_id]
        radius = math.hypot(point.u - center.u, point.v - center.v)
        angle = math.radians(angle_degrees)
        try:
            controller.move_point(
                point_id,
                center.u + radius * math.cos(angle),
                center.v + radius * math.sin(angle),
            )
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh(selected_id=curve.id)

    def _arc_orientation_changed(self, _index: int = -1) -> None:
        if self._refreshing:
            return
        curve = self._selected_curve()
        if not isinstance(curve, SketchArc):
            return
        try:
            self._require_controller().update_curve_parameters(
                curve.id,
                orientation=str(self.arc_orientation_combo.currentData()),
            )
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh(selected_id=curve.id)

    def _refresh_curve_parameters(
        self,
        snapshot: SketchDraftSnapshot,
    ) -> None:
        selected_id = (
            snapshot.selected_ids[0]
            if snapshot.selected_ids and snapshot.selected_kind == "edge"
            else None
        )
        curve = next(
            (
                item
                for item in snapshot.curves
                if item.id == selected_id
            ),
            None,
        )
        point_map = {point.id: point for point in snapshot.points}
        widgets = (
            self.line_start_combo,
            self.line_end_combo,
            self.line_length_spin,
            self.circle_radius_spin,
            self.arc_radius_spin,
            self.arc_start_angle_spin,
            self.arc_end_angle_spin,
            self.arc_orientation_combo,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.line_parameter_group.setVisible(
                isinstance(curve, SketchLine)
            )
            self.circle_parameter_group.setVisible(
                isinstance(curve, SketchCircle)
            )
            self.arc_parameter_group.setVisible(
                isinstance(curve, SketchArc)
            )
            if isinstance(curve, SketchLine):
                self.line_start_combo.clear()
                self.line_end_combo.clear()
                for point in snapshot.points:
                    self.line_start_combo.addItem(point.id, point.id)
                    self.line_end_combo.addItem(point.id, point.id)
                self.line_start_combo.setCurrentIndex(
                    self.line_start_combo.findData(curve.start_point_id)
                )
                self.line_end_combo.setCurrentIndex(
                    self.line_end_combo.findData(curve.end_point_id)
                )
                start = point_map[curve.start_point_id]
                end = point_map[curve.end_point_id]
                self.line_length_spin.setValue(
                    math.hypot(end.u - start.u, end.v - start.v)
                )
            elif isinstance(curve, SketchCircle):
                self.circle_radius_spin.setValue(curve.radius)
            elif isinstance(curve, SketchArc):
                center = point_map[curve.center_point_id]
                start = point_map[curve.start_point_id]
                end = point_map[curve.end_point_id]
                self.arc_radius_spin.setValue(
                    math.hypot(start.u - center.u, start.v - center.v)
                )
                self.arc_start_angle_spin.setValue(
                    math.degrees(
                        math.atan2(start.v - center.v, start.u - center.u)
                    )
                )
                self.arc_end_angle_spin.setValue(
                    math.degrees(
                        math.atan2(end.v - center.v, end.u - center.u)
                    )
                )
                self.arc_orientation_combo.setCurrentIndex(
                    self.arc_orientation_combo.findData(curve.orientation)
                )
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def _point_filter_changed(self, *_args) -> None:
        if self._refreshing or self._controller is None:
            return
        self._refreshing = True
        try:
            self._refresh_point_table(self._controller.snapshot())
        finally:
            self._refreshing = False

    def _filtered_points(self, snapshot: SketchDraftSnapshot):
        controller = self._require_controller()
        query = self.point_search_edit.text().strip().casefold()
        status_filter = self.point_filter_combo.currentData()
        expected_status = {
            "free": "自由",
            "associated": "已关联",
            "unresolved": "未解析",
        }.get(status_filter)
        return tuple(
            point
            for point in snapshot.points
            if (not query or query in point.id.casefold())
            and (
                expected_status is None
                or controller.association_status(point.id) == expected_status
            )
        )

    def _refresh_point_table(
        self,
        snapshot: SketchDraftSnapshot,
        *,
        selected_id: str | None = None,
    ) -> None:
        controller = self._require_controller()
        vertical_scroll = self.points_table.verticalScrollBar().value()
        horizontal_scroll = self.points_table.horizontalScrollBar().value()
        current_id = self._point_id_for_row(self.points_table.currentRow())
        current_column = self.points_table.currentColumn()
        editing = (
            self.points_table.state() == QAbstractItemView.State.EditingState
        )
        sort_column = self.points_table.horizontalHeader().sortIndicatorSection()
        sort_order = self.points_table.horizontalHeader().sortIndicatorOrder()
        sorting_enabled = self.points_table.isSortingEnabled()
        self.points_table.setSortingEnabled(False)
        self.points_table.setRowCount(0)
        for row, point in enumerate(self._filtered_points(snapshot)):
            self.points_table.insertRow(row)
            association_status = controller.association_status(point.id)
            usage = controller.point_usage(point.id)
            dependency_count = len(controller.dependent_curve_ids(point.id))
            values = (
                point.id,
                f"{point.u:.6g}",
                f"{point.v:.6g}",
                association_status,
                "、".join(usage) if usage else "普通点",
                str(dependency_count),
            )
            for column, value in enumerate(values):
                item = _PointTableItem(
                    value,
                    point.id,
                    numeric=(
                        point.u
                        if column == 1
                        else point.v
                        if column == 2
                        else float(dependency_count)
                        if column == 5
                        else None
                    ),
                )
                item.setData(Qt.ItemDataRole.UserRole, point.id)
                if column in {0, 3, 4, 5} or (
                    column in {1, 2} and association_status != "自由"
                ):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column in {1, 2} and association_status != "自由":
                    item.setToolTip("关联点坐标只读；解除关联后可编辑")
                self.points_table.setItem(row, column, item)
        self.points_table.setSortingEnabled(sorting_enabled)
        if sorting_enabled and sort_column >= 0:
            self.points_table.sortItems(sort_column, sort_order)

        requested_id = selected_id or current_id
        self._sync_point_table_selection(scroll_to_id=selected_id)
        if requested_id is not None:
            row = self._row_for_point_id(requested_id)
            if row is not None:
                item = self.points_table.item(row, max(0, current_column))
                if item is not None:
                    self.points_table.setCurrentItem(
                        item,
                        QItemSelectionModel.SelectionFlag.NoUpdate,
                    )
                    if editing and item.flags() & Qt.ItemFlag.ItemIsEditable:
                        self.points_table.editItem(item)
        self.points_table.verticalScrollBar().setValue(vertical_scroll)
        self.points_table.horizontalScrollBar().setValue(horizontal_scroll)

    def _visible_point_ids(self) -> tuple[str, ...]:
        return tuple(
            point_id
            for row in range(self.points_table.rowCount())
            if (point_id := self._point_id_for_row(row)) is not None
        )

    def _point_id_for_row(self, row: int) -> str | None:
        item = self.points_table.item(row, 0)
        if item is None:
            return None
        point_id = item.data(Qt.ItemDataRole.UserRole)
        return point_id if isinstance(point_id, str) else None

    def _row_for_point_id(self, point_id: str) -> int | None:
        return next(
            (
                row
                for row in range(self.points_table.rowCount())
                if self._point_id_for_row(row) == point_id
            ),
            None,
        )

    def _ensure_point_visible(self, point_id: str) -> None:
        if self._row_for_point_id(point_id) is not None:
            return
        self._refreshing = True
        try:
            self.point_search_edit.clear()
            self.point_filter_combo.setCurrentIndex(
                self.point_filter_combo.findData("all")
            )
        finally:
            self._refreshing = False
        self._refreshing = True
        try:
            self._refresh_point_table(self._require_controller().snapshot())
        finally:
            self._refreshing = False

    def _sync_point_table_selection(
        self,
        *,
        scroll_to_id: str | None = None,
    ) -> None:
        if self._controller is None:
            return
        snapshot = self._controller.snapshot()
        selected_ids = (
            set(snapshot.selected_ids) if snapshot.selected_kind == "point" else set()
        )
        previous = self._refreshing
        self._refreshing = True
        try:
            self.points_table.clearSelection()
            for row in range(self.points_table.rowCount()):
                point_id = self._point_id_for_row(row)
                if point_id in selected_ids:
                    for column in range(self.points_table.columnCount()):
                        item = self.points_table.item(row, column)
                        if item is not None:
                            item.setSelected(True)
            target = scroll_to_id or next(iter(snapshot.selected_ids), None)
            if target is not None:
                row = self._row_for_point_id(target)
                if row is not None:
                    item = self.points_table.item(row, 0)
                    self.points_table.setCurrentItem(
                        item,
                        QItemSelectionModel.SelectionFlag.NoUpdate,
                    )
                    self.points_table.scrollToItem(
                        item,
                        QAbstractItemView.ScrollHint.PositionAtCenter,
                    )
        finally:
            self._refreshing = previous

    def _selection_changed_lightweight(self) -> None:
        controller = self._require_controller()
        snapshot = controller.snapshot()
        associated_selection = (
            snapshot.selected_kind == "point"
            and len(snapshot.selected_ids) == 1
            and controller.external_reference_for_point(snapshot.selected_ids[0])
            is not None
        )
        self.release_association_button.setEnabled(associated_selection)
        self._refresh_curve_parameters(snapshot)
        self._refresh_constraints(snapshot)
        if self._viewport is not None:
            kind = "curve" if snapshot.selected_kind == "edge" else snapshot.selected_kind
            self._viewport.update_sketch_selection(kind, snapshot.selected_ids)
        self.draftChanged.emit(snapshot)

    def _refresh_constraints(self, snapshot: SketchDraftSnapshot) -> None:
        related = list(
            constraints_for_entities(snapshot.constraints, snapshot.selected_ids)
            if snapshot.selected_ids
            else snapshot.constraints
        )
        selected_curves = {
            curve.id: curve
            for curve in snapshot.curves
            if curve.id in snapshot.selected_ids
        }
        for constraint in snapshot.constraints:
            if not isinstance(constraint, SketchDistanceDimension):
                continue
            if any(
                isinstance(curve, SketchLine)
                and {curve.start_point_id, curve.end_point_id}
                == {constraint.first_point_id, constraint.second_point_id}
                for curve in selected_curves.values()
            ) and constraint not in related:
                related.append(constraint)
        self.constraints_table.setRowCount(len(related))
        point_map = {point.id: point for point in snapshot.points}
        curve_map = {curve.id: curve for curve in snapshot.curves}
        for row, constraint in enumerate(related):
            text = constraint_text(constraint)
            if isinstance(constraint, (SketchDistanceDimension, SketchRadiusDimension)) and not constraint.driving:
                measured = measured_dimension_value(constraint, point_map, curve_map)
                if measured is not None:
                    text = text.replace(f"{constraint.value:g}", f"{measured:g}")
            self.constraints_table.setItem(row, 0, QTableWidgetItem(constraint.id))
            self.constraints_table.setItem(row, 1, QTableWidgetItem(text))
            self.constraints_table.setItem(
                row, 2,
                QTableWidgetItem("、".join(sketch_constraint_entity_ids(constraint))),
            )

    def _refresh(self, *, selected_id: str | None = None) -> None:
        controller = self._controller
        if controller is None:
            return
        snapshot = controller.snapshot()
        self._refreshing = True
        try:
            self.name_edit.setText(snapshot.name)
            self.work_plane_label.setText(
                "全局 XY"
                if snapshot.plane == SketchPlane.xy()
                else "实体平面面（U/V）"
            )
            self._refresh_point_table(snapshot, selected_id=selected_id)
            diagnostics = controller.finish_diagnostics
            self.diagnostic_label.setText(
                "\n".join(item.message for item in diagnostics)
                if diagnostics
                else "草图已形成有效闭合轮廓"
            )
            self.finish_button.setEnabled(controller.can_finish)
            self.finish_button.setToolTip(
                "完成草图"
                if controller.can_finish
                else "请先处理草图诊断"
            )
            self.undo_button.setEnabled(controller.can_undo)
            self.redo_button.setEnabled(controller.can_redo)
            associated_selection = (
                snapshot.selected_kind == "point"
                and bool(snapshot.selected_ids)
                and controller.external_reference_for_point(
                    snapshot.selected_ids[0]
                )
                is not None
            )
            self.release_association_button.setEnabled(associated_selection)
            self._refresh_curve_parameters(snapshot)
            self._refresh_constraints(snapshot)
            self.solve_status_label.setText(
                solve_status_text(controller.current_solve_result())
            )
        finally:
            self._refreshing = False
        self._send_render_data()

    def _send_render_data(self) -> None:
        if self._controller is None:
            return
        if self._viewport is not None:
            self._viewport.update_sketch_draft(self.render_data())
            self._viewport.set_sketch_pending_points(
                tuple(
                    self._controller.plane.to_global(u, v)
                    for u, v in self._pending_points
                )
            )
        self.draftChanged.emit(self._controller.snapshot())

    def _clear_pending(self) -> None:
        self._pending_points.clear()
        self._pending_references.clear()
        self._incoming_reference = None
        self._incoming_snap = {}
        self._inference_preview = ()
        self._polyline_start_id = None
        self._polyline_first_id = None
        if self._viewport is not None:
            self._viewport.set_sketch_pending_points(())
            self._viewport.set_sketch_inference_preview(())

    def _pending_cancelled(self) -> None:
        self._clear_pending()
        self._set_status("已取消当前草图操作")

    def _authoring_missed(self, reason: str) -> None:
        if reason == "select" and self._controller is not None:
            self._controller.clear_selection()
            self._selection_anchor_id = None
            self._sync_point_table_selection()
            self._selection_changed_lightweight()
        messages = {
            "select": "当前位置没有可选择的草图实体",
            "trim": "请单击需要修剪的曲线",
            "point.ray": "无法将单击位置投影到当前工作平面",
            "point.parallel": "当前视线与工作平面平行",
        }
        self._set_status(messages.get(reason, f"草图操作未完成：{reason}"))

    def _set_status(self, message: str) -> None:
        self.statusChanged.emit(str(message))

    def _require_controller(self) -> SketchDraftController:
        if self._controller is None:
            raise RuntimeError("sketch editor has no controller")
        return self._controller


__all__ = ["SketchEditorPanel"]
