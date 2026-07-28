"""Non-modal Phase 1 editor panel for strict planar sketch drafts."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
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
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fem.geometry import SketchArc, SketchCircle, SketchGeometry, SketchLine

from ..geometry_preview import build_strict_sketch_draft_preview
from ..sketch_editor import (
    SketchDraftController,
    SketchDraftSnapshot,
    SketchDraftValidationError,
)
from .viewport import SketchDraftRenderData


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
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sketchEditorPanel")
        self.setMinimumWidth(330)
        self.setMaximumWidth(500)
        self._controller: SketchDraftController | None = None
        self._viewport = None
        self._base_snapshot: SketchDraftSnapshot | None = None
        self._refreshing = False
        self._pending_points: list[tuple[float, float]] = []
        self._polyline_start_id: str | None = None
        self._polyline_first_id: str | None = None
        self._authoring_purpose = "geometry"
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

        self.snap_check = QCheckBox("启用网格捕捉", self)
        self.snap_check.setChecked(True)
        self.snap_check.toggled.connect(self._grid_changed)
        self.spacing_spin = QDoubleSpinBox(self)
        self.spacing_spin.setObjectName("sketchGridSpacing")
        self.spacing_spin.setDecimals(3)
        self.spacing_spin.setRange(0.001, 1.0e12)
        self.spacing_spin.setSingleStep(0.1)
        self.spacing_spin.setValue(0.1)
        self.spacing_spin.valueChanged.connect(self._grid_changed)

        self.points_table = QTableWidget(0, 3, self)
        self.points_table.setObjectName("sketchPointsTable")
        self.points_table.setHorizontalHeaderLabels(("ID", "U", "V"))
        self.points_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.points_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.points_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.itemChanged.connect(self._point_item_changed)
        self.points_table.itemSelectionChanged.connect(self._point_row_selected)

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
        self.undo_button = QPushButton("撤销", self)
        self.redo_button = QPushButton("重做", self)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        self.diagnostic_label = QLabel(self)
        self.diagnostic_label.setObjectName("sketchDiagnostics")
        self.diagnostic_label.setWordWrap(True)

        self.finish_button = QPushButton("完成草图", self)
        self.finish_button.setObjectName("sketchFinishButton")
        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.setObjectName("sketchCancelButton")
        self.finish_button.clicked.connect(self.try_finish)
        self.cancel_button.clicked.connect(self.cancelRequested.emit)

        form = QFormLayout()
        form.addRow("草图名称", self.name_edit)
        form.addRow("工作平面", QLabel("全局 XY", self))
        form.addRow(self.snap_check)
        form.addRow("网格间距", self.spacing_spin)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.delete_button)
        edit_row.addWidget(self.undo_button)
        edit_row.addWidget(self.redo_button)
        bottom = QHBoxLayout()
        bottom.addWidget(self.finish_button)
        bottom.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("绘图工具", self))
        layout.addLayout(first_modes)
        layout.addLayout(second_modes)
        layout.addWidget(QLabel("点坐标", self))
        layout.addWidget(self.points_table, 1)
        layout.addWidget(self.line_parameter_group)
        layout.addWidget(self.circle_parameter_group)
        layout.addWidget(self.arc_parameter_group)
        layout.addLayout(edit_row)
        layout.addWidget(self.diagnostic_label)
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

    def attach_viewport(self, viewport) -> None:
        if self._viewport is viewport:
            return
        if self._viewport is not None:
            self._disconnect_viewport(self._viewport)
        self._viewport = viewport
        if viewport is None:
            return
        viewport.sketchWorkPlanePointSelected.connect(self._point_from_viewport)
        viewport.sketchDraftPointSelected.connect(self._select_point)
        viewport.sketchDraftCurveSelected.connect(self._select_curve)
        viewport.sketchDraftProfileSelected.connect(self._select_profile)
        viewport.sketchTrimRequested.connect(self._trim_from_viewport)
        viewport.sketchAuthoringMissed.connect(self._authoring_missed)
        viewport.sketchPendingInteractionCancelled.connect(
            self._pending_cancelled
        )
        viewport.sketchAuthoringFinishRequested.connect(self.try_finish)
        viewport.sketchAuthoringCancelled.connect(self.cancelRequested.emit)

    def _disconnect_viewport(self, viewport) -> None:
        connections = (
            (viewport.sketchWorkPlanePointSelected, self._point_from_viewport),
            (viewport.sketchDraftPointSelected, self._select_point),
            (viewport.sketchDraftCurveSelected, self._select_curve),
            (viewport.sketchDraftProfileSelected, self._select_profile),
            (viewport.sketchTrimRequested, self._trim_from_viewport),
            (viewport.sketchAuthoringMissed, self._authoring_missed),
            (
                viewport.sketchPendingInteractionCancelled,
                self._pending_cancelled,
            ),
            (viewport.sketchAuthoringFinishRequested, self.try_finish),
            (viewport.sketchAuthoringCancelled, self.cancelRequested.emit),
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
        if normalized not in {"geometry", "planar_boolean_tool"}:
            raise ValueError("unsupported sketch authoring purpose")
        self._authoring_purpose = normalized
        self.finish_button.setText(
            "完成工具草图"
            if normalized == "planar_boolean_tool"
            else "完成草图"
        )
        self.attach_viewport(viewport)
        self.show()
        viewport.start_sketch_authoring(
            self.render_data(),
            snap=self.snap_check.isChecked(),
            spacing=self.spacing_spin.value(),
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
            )
        return self._incomplete_render_data(snapshot, selected_kind, selected_id)

    def _incomplete_render_data(
        self,
        snapshot: SketchDraftSnapshot,
        selected_kind: str | None,
        selected_id: str | None,
    ) -> SketchDraftRenderData:
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
        )

    def _point_from_viewport(
        self,
        global_point: tuple[float, float, float],
    ) -> None:
        controller = self._require_controller()
        u, v = controller.plane.to_local(tuple(global_point))
        try:
            if self.mode == "polyline":
                closed = False
                point_id = self._point_id_at(u, v)
                if point_id is None:
                    point_id = controller.add_point(u, v).id
                if self._polyline_start_id is None:
                    self._polyline_start_id = point_id
                    self._polyline_first_id = point_id
                elif point_id != self._polyline_start_id:
                    controller.add_line(self._polyline_start_id, point_id)
                    if point_id == self._polyline_first_id:
                        self._clear_pending()
                        closed = True
                    else:
                        self._polyline_start_id = point_id
                if not closed:
                    self._pending_points = [(u, v)]
            elif self.mode == "rectangle":
                self._pending_points.append((u, v))
                if len(self._pending_points) == 2:
                    controller.add_rectangle(*self._pending_points)
                    self._clear_pending()
            elif self.mode == "circle":
                self._pending_points.append((u, v))
                if len(self._pending_points) == 2:
                    center, rim = self._pending_points
                    radius = math.hypot(rim[0] - center[0], rim[1] - center[1])
                    controller.add_circle(center, radius)
                    self._clear_pending()
            elif self.mode == "arc":
                self._pending_points.append((u, v))
                if len(self._pending_points) == 3:
                    controller.add_arc(*self._pending_points)
                    self._clear_pending()
        except (TypeError, ValueError) as error:
            self._set_status(str(error))
            self._clear_pending()
        self._refresh()

    def _point_id_at(self, u: float, v: float) -> str | None:
        controller = self._require_controller()
        tolerance = max(1.0e-8, self.spacing_spin.value() * 1.0e-6)
        for point in controller.snapshot().points:
            if math.hypot(point.u - u, point.v - v) <= tolerance:
                return point.id
        return None

    def _select_point(self, point_id: str) -> None:
        self._require_controller().select_point(point_id)
        self._refresh(selected_id=point_id)

    def _select_curve(self, curve_id: str) -> None:
        self._require_controller().select_curve(curve_id)
        self._refresh(selected_id=curve_id)

    def _select_profile(self, profile_id: str) -> None:
        self._require_controller().select_profile(profile_id)
        self._refresh(selected_id=profile_id)

    def _trim_from_viewport(
        self,
        curve_id: str,
        global_point: tuple[float, float, float],
    ) -> None:
        controller = self._require_controller()
        try:
            controller.trim_curve(
                curve_id,
                controller.plane.to_local(tuple(global_point)),
            )
        except (TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh()

    def delete_selected(self) -> None:
        controller = self._require_controller()
        if not controller.selected_ids:
            return
        entity_id = controller.selected_ids[0]
        try:
            controller.delete(entity_id)
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh()

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
        if self._viewport is None:
            return
        self._viewport.set_sketch_grid(
            snap=self.snap_check.isChecked(),
            spacing=self.spacing_spin.value(),
        )

    def _point_item_changed(self, item: QTableWidgetItem) -> None:
        if self._refreshing or item.column() not in {1, 2}:
            return
        controller = self._require_controller()
        snapshot = controller.snapshot()
        if item.row() >= len(snapshot.points):
            return
        point = snapshot.points[item.row()]
        try:
            controller.move_point(
                point.id,
                u=float(self.points_table.item(item.row(), 1).text()),
                v=float(self.points_table.item(item.row(), 2).text()),
            )
        except (TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh(selected_id=point.id)

    def _point_row_selected(self) -> None:
        if self._refreshing:
            return
        row = self.points_table.currentRow()
        snapshot = self._require_controller().snapshot()
        if 0 <= row < len(snapshot.points):
            self._select_point(snapshot.points[row].id)

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
                controller.move_point(
                    point_id,
                    center.u + target * math.cos(angle),
                    center.v + target * math.sin(angle),
                )
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

    def _refresh(self, *, selected_id: str | None = None) -> None:
        controller = self._controller
        if controller is None:
            return
        snapshot = controller.snapshot()
        self._refreshing = True
        try:
            self.name_edit.setText(snapshot.name)
            self.points_table.setRowCount(0)
            for row, point in enumerate(snapshot.points):
                self.points_table.insertRow(row)
                for column, value in enumerate(
                    (point.id, f"{point.u:.6g}", f"{point.v:.6g}")
                ):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setFlags(
                            item.flags() & ~Qt.ItemFlag.ItemIsEditable
                        )
                    self.points_table.setItem(row, column, item)
            diagnostics = controller.finish_diagnostics
            self.diagnostic_label.setText(
                "\n".join(item.message for item in diagnostics)
                if diagnostics
                else "草图已形成有效闭合 Profile"
            )
            self.finish_button.setEnabled(controller.can_finish)
            self.finish_button.setToolTip(
                "完成草图"
                if controller.can_finish
                else "请先处理草图诊断"
            )
            self.undo_button.setEnabled(controller.can_undo)
            self.redo_button.setEnabled(controller.can_redo)
            self._refresh_curve_parameters(snapshot)
            if selected_id is not None:
                for row, point in enumerate(snapshot.points):
                    if point.id == selected_id:
                        self.points_table.selectRow(row)
        finally:
            self._refreshing = False
        self._send_render_data()

    def _send_render_data(self) -> None:
        if self._controller is None:
            return
        if self._viewport is not None:
            self._viewport.update_sketch_draft(self.render_data())
            self._viewport.set_sketch_pending_points(
                tuple((u, v, 0.0) for u, v in self._pending_points)
            )
        self.draftChanged.emit(self._controller.snapshot())

    def _clear_pending(self) -> None:
        self._pending_points.clear()
        self._polyline_start_id = None
        self._polyline_first_id = None
        if self._viewport is not None:
            self._viewport.set_sketch_pending_points(())

    def _pending_cancelled(self) -> None:
        self._clear_pending()
        self._set_status("已取消当前草图操作")

    def _authoring_missed(self, reason: str) -> None:
        messages = {
            "select": "当前位置没有可选择的草图实体",
            "trim": "请单击需要修剪的曲线",
            "point.ray": "无法将单击位置投影到 XY 工作平面",
            "point.parallel": "当前视线与 XY 工作平面平行",
        }
        self._set_status(messages.get(reason, f"草图操作未完成：{reason}"))

    def _set_status(self, message: str) -> None:
        self.statusChanged.emit(str(message))

    def _require_controller(self) -> SketchDraftController:
        if self._controller is None:
            raise RuntimeError("sketch editor has no controller")
        return self._controller


__all__ = ["SketchEditorPanel"]
