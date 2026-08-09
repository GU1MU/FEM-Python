"""Non-modal editor panel for native one-dimensional wire drafts."""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
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

from ..wire_editor import (
    WORK_PLANES,
    WireDraftController,
    WireDraftSnapshot,
    snap_work_plane_point,
)
from .viewport import WireDraftRenderData


def _finite_spin_box(parent: QWidget, value: float = 0.0) -> QDoubleSpinBox:
    editor = QDoubleSpinBox(parent)
    editor.setRange(-1.0e12, 1.0e12)
    editor.setDecimals(2)
    editor.setSingleStep(1.0)
    editor.setValue(float(value))
    return editor


class WireEditorPanel(QWidget):
    """Keep one detached wire draft synchronized with the main viewport."""

    finishRequested = Signal()
    cancelRequested = Signal()
    draftChanged = Signal(object)
    workPlaneChanged = Signal(str)
    statusChanged = Signal(str)
    entityFocusRequested = Signal(str, str)

    def __init__(
        self,
        controller: WireDraftController | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("wireEditorPanel")
        self.setMinimumWidth(320)
        self.setMaximumWidth(480)
        self._controller: WireDraftController | None = None
        self._viewport = None
        self._refreshing = False
        self._base_snapshot: WireDraftSnapshot | None = None
        self._build_ui()
        if controller is not None:
            self.set_controller(controller)

    @property
    def controller(self) -> WireDraftController | None:
        return self._controller

    @property
    def dirty(self) -> bool:
        return bool(self._controller is not None and self._controller.dirty)

    @property
    def base_snapshot(self) -> WireDraftSnapshot | None:
        return self._base_snapshot

    def _build_ui(self) -> None:
        self.name_edit = QLineEdit(self)
        self.name_edit.setObjectName("wireNameEdit")
        self.name_edit.editingFinished.connect(self._wire_name_changed)

        self.point_mode_button = QPushButton("添加点", self)
        self.member_mode_button = QPushButton("连接杆件", self)
        self.select_mode_button = QPushButton("选择对象", self)
        self.point_mode_button.setToolTip("在视图区的工作平面上单击以添加点")
        self.member_mode_button.setToolTip("在视图区依次单击两个已有点以连接杆件")
        self.select_mode_button.setToolTip("在视图区单击已有点或杆件以选中对象")
        self._mode_buttons = {
            "point": self.point_mode_button,
            "member": self.member_mode_button,
            "select": self.select_mode_button,
        }
        for mode, button in self._mode_buttons.items():
            button.setCheckable(True)
            button.setObjectName(f"wire{mode.title()}ModeButton")
            button.clicked.connect(
                lambda _checked=False, selected=mode: self.set_mode(selected)
            )
        self.point_mode_button.setChecked(True)

        self.work_plane_combo = QComboBox(self)
        self.work_plane_combo.setObjectName("wireWorkPlaneCombo")
        for plane in WORK_PLANES:
            self.work_plane_combo.addItem(plane, plane)
        self.work_plane_combo.currentIndexChanged.connect(self._work_plane_changed)
        self.offset_spin = _finite_spin_box(self)
        self.offset_spin.setObjectName("wireWorkPlaneOffset")
        self.offset_spin.valueChanged.connect(self._work_plane_offset_changed)
        self.spacing_spin = QDoubleSpinBox(self)
        self.spacing_spin.setObjectName("wireGridSpacing")
        self.spacing_spin.setDecimals(2)
        self.spacing_spin.setRange(0.01, 1.0e12)
        self.spacing_spin.setSingleStep(0.1)
        self.spacing_spin.setValue(0.1)
        self.spacing_spin.valueChanged.connect(self._grid_settings_changed)

        self.points_table = QTableWidget(0, 4, self)
        self.points_table.setObjectName("wirePointsTable")
        self.points_table.setHorizontalHeaderLabels(("名称", "X", "Y", "Z"))
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
        self.points_table.cellDoubleClicked.connect(
            lambda row, _column: self._focus_point_row(row)
        )
        self.add_point_button = QPushButton("新增", self)
        self.delete_point_button = QPushButton("删除", self)
        self.add_point_button.setToolTip("新增一个坐标默认为零的点，可在表格中修改坐标")
        self.delete_point_button.setToolTip("删除表格中当前选中的点")
        self.add_point_button.clicked.connect(self.add_point)
        self.delete_point_button.clicked.connect(self.delete_point)

        self.members_table = QTableWidget(0, 3, self)
        self.members_table.setObjectName("wireMembersTable")
        self.members_table.setHorizontalHeaderLabels(("名称", "起点", "终点"))
        self.members_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.members_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.members_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.members_table.verticalHeader().setVisible(False)
        self.members_table.itemChanged.connect(self._member_item_changed)
        self.members_table.itemSelectionChanged.connect(self._member_row_selected)
        self.members_table.cellDoubleClicked.connect(
            lambda row, _column: self._focus_member_row(row)
        )
        self.add_member_button = QPushButton("新增", self)
        self.delete_member_button = QPushButton("删除", self)
        self.add_member_button.setToolTip("新增一根杆件，并在表格中选择它的起点和终点")
        self.delete_member_button.setToolTip("删除表格中当前选中的杆件")
        self.add_member_button.clicked.connect(self.add_member)
        self.delete_member_button.clicked.connect(self.delete_member)

        self.finish_button = QPushButton("完成创建", self)
        self.cancel_button = QPushButton("取消", self)
        self.finish_button.setObjectName("wireFinishButton")
        self.cancel_button.setObjectName("wireCancelButton")
        self.finish_button.clicked.connect(self.try_finish)
        self.cancel_button.clicked.connect(self.cancelRequested.emit)

        form = QFormLayout()
        form.addRow("线体名称", self.name_edit)
        form.addRow("工作平面", self.work_plane_combo)
        form.addRow("平面偏移", self.offset_spin)
        form.addRow("吸附间距", self.spacing_spin)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.point_mode_button)
        mode_row.addWidget(self.member_mode_button)
        mode_row.addWidget(self.select_mode_button)

        points_buttons = QHBoxLayout()
        points_buttons.addWidget(self.add_point_button)
        points_buttons.addWidget(self.delete_point_button)
        members_buttons = QHBoxLayout()
        members_buttons.addWidget(self.add_member_button)
        members_buttons.addWidget(self.delete_member_button)
        bottom = QHBoxLayout()
        bottom.addWidget(self.finish_button)
        bottom.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("视图区操作", self))
        layout.addLayout(mode_row)
        layout.addWidget(QLabel("点坐标", self))
        layout.addWidget(self.points_table, 1)
        layout.addLayout(points_buttons)
        layout.addWidget(QLabel("杆件连接", self))
        layout.addWidget(self.members_table, 1)
        layout.addLayout(members_buttons)
        layout.addLayout(bottom)

    def set_controller(
        self,
        controller: WireDraftController,
        *,
        base_snapshot: WireDraftSnapshot | None = None,
    ) -> None:
        if type(controller) is not WireDraftController:
            raise TypeError("controller must be a WireDraftController")
        self._controller = controller
        self._base_snapshot = base_snapshot or controller.snapshot()
        self._refresh()

    def attach_viewport(self, viewport) -> None:
        """Connect one viewport instance; repeated attachment is harmless."""

        if self._viewport is viewport:
            return
        if self._viewport is not None:
            self._disconnect_viewport(self._viewport)
        self._viewport = viewport
        if viewport is None:
            return
        viewport.wireWorkPlanePointSelected.connect(self._point_from_viewport)
        viewport.wireDraftPointSelected.connect(self._select_point)
        viewport.wireDraftMemberSelected.connect(self._select_member)
        viewport.wireMemberEndpointsSelected.connect(self._member_from_viewport)
        viewport.wireMemberStartSelected.connect(self._pending_member_start)
        viewport.wireAuthoringMissed.connect(self._authoring_missed)
        viewport.wirePendingInteractionCancelled.connect(self._pending_cancelled)
        viewport.wireAuthoringFinishRequested.connect(self.try_finish)
        viewport.wireAuthoringCancelled.connect(self._authoring_cancelled)

    def _disconnect_viewport(self, viewport) -> None:
        if viewport is None:
            return
        connections = (
            (viewport.wireWorkPlanePointSelected, self._point_from_viewport),
            (viewport.wireDraftPointSelected, self._select_point),
            (viewport.wireDraftMemberSelected, self._select_member),
            (viewport.wireMemberEndpointsSelected, self._member_from_viewport),
            (viewport.wireMemberStartSelected, self._pending_member_start),
            (viewport.wireAuthoringMissed, self._authoring_missed),
            (viewport.wirePendingInteractionCancelled, self._pending_cancelled),
            (viewport.wireAuthoringFinishRequested, self.try_finish),
            (viewport.wireAuthoringCancelled, self._authoring_cancelled),
        )
        for signal, slot in connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def begin(self, viewport) -> None:
        if self._controller is None:
            raise RuntimeError("wire editor requires a draft controller")
        self.attach_viewport(viewport)
        self.show()
        self._send_render_data(start=True)

    def end(self) -> None:
        if self._viewport is not None:
            self._viewport.stop_wire_authoring()
        self.hide()

    def set_mode(self, mode: str) -> None:
        normalized = str(mode).strip().casefold()
        if normalized not in self._mode_buttons:
            raise ValueError("wire mode must be point, member, or select")
        for name, button in self._mode_buttons.items():
            button.blockSignals(True)
            button.setChecked(name == normalized)
            button.blockSignals(False)
        if self._viewport is not None:
            self._viewport.set_wire_authoring_mode(normalized)
        labels = {
            "point": "添加点",
            "member": "连接杆件",
            "select": "选择对象",
        }
        self._set_status(f"视图区操作：{labels[normalized]}")

    def render_data(self) -> WireDraftRenderData:
        if self._controller is None:
            raise RuntimeError("wire editor has no controller")
        snapshot = self._controller.snapshot()
        point_index = {point.name: index for index, point in enumerate(snapshot.points)}
        members: list[tuple[int, int]] = []
        member_names: list[str] = []
        for member in snapshot.members:
            if member.start not in point_index or member.end not in point_index:
                continue
            members.append((point_index[member.start], point_index[member.end]))
            member_names.append(member.name)
        pending = None
        if self._viewport is not None:
            pending = self._viewport._wire_pending_member_start
        return WireDraftRenderData(
            points=tuple((point.x, point.y, point.z) for point in snapshot.points),
            point_names=tuple(point.name for point in snapshot.points),
            members=tuple(members),
            member_names=tuple(member_names),
            pending_member_start=pending,
        )

    def _send_render_data(self, *, start: bool = False) -> None:
        if self._viewport is None or self._controller is None:
            return
        data = self.render_data()
        if start:
            self._viewport.start_wire_authoring(
                data,
                work_plane=self.work_plane_combo.currentData(),
                offset=self.offset_spin.value(),
                snap=True,
                spacing=self.spacing_spin.value(),
            )
        else:
            self._viewport.update_wire_draft(data)
        self.draftChanged.emit(self._controller.snapshot())

    def _refresh(
        self,
        *,
        selected_point: str | None = None,
        selected_member: str | None = None,
    ) -> None:
        controller = self._controller
        if controller is None:
            self.finish_button.setEnabled(False)
            self.finish_button.setToolTip("完成线体创建")
            return
        if selected_point is None and selected_member is None:
            selection = controller.selection
            if selection is not None:
                kind, name = selection
                if kind == "point":
                    selected_point = name
                elif kind == "member":
                    selected_member = name
        snapshot = controller.snapshot()
        self._refreshing = True
        try:
            self.name_edit.setText(snapshot.name)
            self.points_table.setRowCount(0)
            for row, point in enumerate(snapshot.points):
                self.points_table.insertRow(row)
                values = (
                    point.name,
                    f"{point.x:.2f}",
                    f"{point.y:.2f}",
                    f"{point.z:.2f}",
                )
                for column, value in enumerate(values):
                    self.points_table.setItem(row, column, QTableWidgetItem(value))
            self.members_table.setRowCount(0)
            point_names = tuple(point.name for point in snapshot.points)
            for row, member in enumerate(snapshot.members):
                self.members_table.insertRow(row)
                self.members_table.setItem(row, 0, QTableWidgetItem(member.name))
                for column, value in ((1, member.start), (2, member.end)):
                    combo = QComboBox(self.members_table)
                    for name in point_names:
                        combo.addItem(name, name)
                    if value not in point_names:
                        combo.addItem(value, value)
                    combo.setCurrentIndex(max(0, combo.findData(value)))
                    combo.currentIndexChanged.connect(
                        lambda _index, selected_row=row, selected_column=column:
                        self._member_endpoint_changed(
                            selected_row, selected_column
                        )
                    )
                    self.members_table.setCellWidget(row, column, combo)
            self._update_validation()
            if selected_point is not None:
                self._select_table_row(self.points_table, snapshot.points, selected_point)
            elif selected_member is not None:
                self._select_table_row(
                    self.members_table,
                    snapshot.members,
                    selected_member,
                )
        finally:
            self._refreshing = False
        self._send_render_data()
        if selected_point is not None:
            self.entityFocusRequested.emit("point", selected_point)
        elif selected_member is not None:
            self.entityFocusRequested.emit("member", selected_member)

    def _update_validation(self) -> None:
        controller = self._controller
        if controller is None:
            return
        self.finish_button.setEnabled(controller.can_finish)
        self.finish_button.setToolTip("完成线体创建")

    def _set_status(self, message: str) -> None:
        self.statusChanged.emit(str(message))

    def _pending_member_start(self, name: str) -> None:
        self._set_status(f"已选择杆件起点：{name}，请再选择终点")

    def _authoring_missed(self, reason: str) -> None:
        self._set_status(self._miss_message(reason))

    def _pending_cancelled(self) -> None:
        self._set_status("已取消当前视图区操作")

    def _authoring_cancelled(self) -> None:
        self.cancelRequested.emit()

    def show_status(self, message: str) -> None:
        """Display a command or validation diagnostic without changing the draft."""

        self._set_status(message)

    @staticmethod
    def _miss_message(reason: str) -> str:
        return {
            "point.ray": "无法将单击位置投影到工作平面",
            "point.parallel": "当前视线与工作平面平行，无法添加点",
            "member": "请单击一个已有点来连接杆件",
            "member.same_endpoint": "杆件的起点和终点必须是两个不同的点",
            "select": "当前位置没有可选择的点或杆件",
        }.get(str(reason), f"视图区操作未完成：{reason}")

    def _wire_name_changed(self) -> None:
        if self._refreshing or self._controller is None:
            return
        try:
            self._controller.set_wire_name(self.name_edit.text())
        except (TypeError, ValueError) as error:
            self._set_status(str(error))
        self._refresh()

    def _work_plane_changed(self, _index: int) -> None:
        if self._refreshing:
            return
        plane = str(self.work_plane_combo.currentData())
        self.workPlaneChanged.emit(plane)
        if self._viewport is not None:
            self._viewport.set_wire_work_plane(
                plane,
                self.offset_spin.value(),
                snap=True,
                spacing=self.spacing_spin.value(),
            )

    def _work_plane_offset_changed(self, value: float) -> None:
        if self._refreshing or self._viewport is None:
            return
        self._viewport.set_wire_work_plane(
            str(self.work_plane_combo.currentData()),
            value,
            snap=True,
            spacing=self.spacing_spin.value(),
        )

    def _grid_settings_changed(self, _value: object = None) -> None:
        if self._refreshing or self._viewport is None:
            return
        spacing = max(self.spacing_spin.minimum(), self.spacing_spin.value())
        try:
            self._viewport.set_wire_work_plane(
                str(self.work_plane_combo.currentData()),
                self.offset_spin.value(),
                snap=True,
                spacing=spacing,
            )
        except ValueError:
            self._set_status("吸附间距必须大于零")

    def _point_from_viewport(self, point: object) -> None:
        if self._controller is None:
            return
        try:
            coordinates = list(float(value) for value in point)
            if len(coordinates) != 3:
                raise ValueError("点坐标必须包含三个分量")
            plane = str(self.work_plane_combo.currentData())
            fixed_axis = {"XY": 2, "XZ": 1, "YZ": 0}[plane]
            coordinates[fixed_axis] = float(self.offset_spin.value())
            coordinates = list(
                snap_work_plane_point(
                    coordinates,
                    plane,
                    self.spacing_spin.value(),
                )
            )
            x, y, z = coordinates
            created = self._controller.add_point(None, x, y, z)
        except (TypeError, ValueError) as error:
            self._set_status(str(error))
            return
        self._refresh(selected_point=created.name)

    def _select_point(self, name: str) -> None:
        if self._controller is None:
            return
        try:
            self._controller.select_point(name)
        except KeyError:
            return
        self._select_table_row(
            self.points_table,
            self._controller.snapshot().points,
            name,
        )

    def _select_member(self, name: str) -> None:
        if self._controller is None:
            return
        try:
            self._controller.select_member(name)
        except KeyError:
            return
        self._select_table_row(
            self.members_table,
            self._controller.snapshot().members,
            name,
        )

    def _member_from_viewport(self, start: str, end: str) -> None:
        if self._controller is None:
            return
        try:
            member = self._controller.add_member(None, start, end)
        except (TypeError, ValueError) as error:
            self._set_status(str(error))
            return
        self._refresh(selected_member=member.name)

    def add_point(self) -> None:
        if self._controller is None:
            return
        point = self._controller.add_point()
        self._refresh(selected_point=point.name)

    def delete_point(self) -> None:
        if self._controller is None:
            return
        row = self.points_table.currentRow()
        snapshot = self._controller.snapshot()
        if not 0 <= row < len(snapshot.points):
            return
        try:
            self._controller.delete_point(snapshot.points[row].name)
        except (KeyError, ValueError) as error:
            self._set_status(str(error))
            return
        self._refresh()

    def add_member(self) -> None:
        if self._controller is None:
            return
        member = self._controller.add_member()
        self._refresh(selected_member=member.name)

    def delete_member(self) -> None:
        if self._controller is None:
            return
        row = self.members_table.currentRow()
        snapshot = self._controller.snapshot()
        if not 0 <= row < len(snapshot.members):
            return
        try:
            self._controller.delete_member(snapshot.members[row].name)
        except KeyError as error:
            self._set_status(str(error))
            return
        self._refresh()

    def _point_item_changed(self, item: QTableWidgetItem) -> None:
        if self._refreshing or self._controller is None:
            return
        row, column = item.row(), item.column()
        snapshot = self._controller.snapshot()
        if not 0 <= row < len(snapshot.points):
            return
        point = snapshot.points[row]
        try:
            if column == 0:
                self._controller.rename_point(point.name, item.text())
            else:
                value = round(float(item.text()), 2)
                if not math.isfinite(value):
                    raise ValueError("点坐标必须是有限数值")
                self._controller.update_point(
                    point.name,
                    x=value if column == 1 else None,
                    y=value if column == 2 else None,
                    z=value if column == 3 else None,
                )
        except (TypeError, ValueError, KeyError) as error:
            self._set_status(str(error))
        self._refresh()

    def _member_item_changed(self, item: QTableWidgetItem) -> None:
        if self._refreshing or self._controller is None or item.column() != 0:
            return
        snapshot = self._controller.snapshot()
        row = item.row()
        if not 0 <= row < len(snapshot.members):
            return
        try:
            self._controller.rename_member(snapshot.members[row].name, item.text())
        except (TypeError, ValueError, KeyError) as error:
            self._set_status(str(error))
        self._refresh()

    def _member_endpoint_changed(self, row: int, column: int) -> None:
        if self._refreshing or self._controller is None:
            return
        snapshot = self._controller.snapshot()
        if not 0 <= row < len(snapshot.members):
            return
        combo = self.members_table.cellWidget(row, column)
        if not isinstance(combo, QComboBox):
            return
        member = snapshot.members[row]
        try:
            self._controller.update_member(
                member.name,
                start=str(combo.currentData()) if column == 1 else None,
                end=str(combo.currentData()) if column == 2 else None,
            )
        except (TypeError, ValueError, KeyError) as error:
            self._set_status(str(error))
        self._refresh()

    def _point_row_selected(self) -> None:
        if self._refreshing or self._controller is None:
            return
        row = self.points_table.currentRow()
        snapshot = self._controller.snapshot()
        if 0 <= row < len(snapshot.points):
            try:
                name = snapshot.points[row].name
                self._controller.select_point(name)
            except KeyError:
                pass
            else:
                self.entityFocusRequested.emit("point", name)

    def _member_row_selected(self) -> None:
        if self._refreshing or self._controller is None:
            return
        row = self.members_table.currentRow()
        snapshot = self._controller.snapshot()
        if 0 <= row < len(snapshot.members):
            try:
                name = snapshot.members[row].name
                self._controller.select_member(name)
            except KeyError:
                pass
            else:
                self.entityFocusRequested.emit("member", name)

    def _focus_point_row(self, row: int) -> None:
        if self._controller is None:
            return
        points = self._controller.snapshot().points
        if 0 <= row < len(points):
            self.entityFocusRequested.emit("point", points[row].name)

    def _focus_member_row(self, row: int) -> None:
        if self._controller is None:
            return
        members = self._controller.snapshot().members
        if 0 <= row < len(members):
            self.entityFocusRequested.emit("member", members[row].name)

    @staticmethod
    def _select_table_row(table, values, name: str) -> None:
        for row, value in enumerate(values):
            if getattr(value, "name", None) == name:
                table.selectRow(row)
                return

    def try_finish(self) -> None:
        if self._controller is None:
            return
        self._update_validation()
        if not self._controller.can_finish:
            self._set_status("请先补全点和杆件，并处理草图中的无效数据")
            return
        try:
            self._controller.to_geometry()
        except ValueError as error:
            self._set_status(str(error))
            return
        self.finishRequested.emit()


__all__ = ["WireEditorPanel"]
