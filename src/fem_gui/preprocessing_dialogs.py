"""Dialogs for native geometry and mesh inputs."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QLineEdit,
    QLabel,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QVBoxLayout,
)

from .dialogs import CompactDoubleSpinBox, configure_form_layout
from .document import NamedRegion
from .preprocessing import (
    BoxGeometry,
    BooleanGeometry,
    CylinderGeometry,
    DiskGeometry,
    MeshSettings,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    ExtrudedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    geometry_feature_rows,
    LocalMeshControl,
)


def _positive_spin_box(parent: QDialog, value: float) -> QDoubleSpinBox:
    editor = CompactDoubleSpinBox(parent)
    editor.setRange(1.0e-9, 1.0e12)
    editor.setDecimals(6)
    editor.setValue(float(value))
    return editor


def _signed_spin_box(parent: QDialog, value: float) -> QDoubleSpinBox:
    editor = CompactDoubleSpinBox(parent)
    editor.setRange(-1.0e12, 1.0e12)
    editor.setDecimals(6)
    editor.setValue(float(value))
    return editor


class SketchContourDialog(QDialog):
    """Edit one rectangle or circle contour without exposing CAD internals."""

    def __init__(self, contour=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("草图轮廓")
        current = contour or SketchRectangle("material", 0.0, 0.0, 100.0, 50.0)

        self.operation_combo = QComboBox(self)
        self.operation_combo.addItem("添加材料", "material")
        self.operation_combo.addItem("切除材料", "cut")
        self.operation_combo.setCurrentIndex(
            0 if current.operation == "material" else 1
        )
        self.shape_combo = QComboBox(self)
        self.shape_combo.addItem("矩形", "rectangle")
        self.shape_combo.addItem("圆", "circle")
        self.shape_combo.setCurrentIndex(
            0 if isinstance(current, SketchRectangle) else 1
        )
        self.x_spin = _signed_spin_box(self, current.x)
        self.y_spin = _signed_spin_box(self, current.y)
        self.width_spin = _positive_spin_box(
            self,
            current.width if isinstance(current, SketchRectangle) else 100.0,
        )
        self.height_spin = _positive_spin_box(
            self,
            current.height if isinstance(current, SketchRectangle) else 50.0,
        )
        self.radius_spin = _positive_spin_box(
            self,
            current.radius if isinstance(current, SketchCircle) else 25.0,
        )

        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("用途", self.operation_combo)
        form.addRow("形状", self.shape_combo)
        form.addRow("X", self.x_spin)
        form.addRow("Y", self.y_spin)
        form.addRow("宽度", self.width_spin)
        form.addRow("高度", self.height_spin)
        form.addRow("半径", self.radius_spin)
        self.width_label = form.labelForField(self.width_spin)
        self.height_label = form.labelForField(self.height_spin)
        self.radius_label = form.labelForField(self.radius_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.shape_combo.currentIndexChanged.connect(self._update_shape_fields)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setMinimumWidth(340)
        self._update_shape_fields()

    def _update_shape_fields(self) -> None:
        rectangle = self.shape_combo.currentData() == "rectangle"
        self.width_spin.setEnabled(rectangle)
        self.height_spin.setEnabled(rectangle)
        self.radius_spin.setEnabled(not rectangle)
        self.width_label.setVisible(rectangle)
        self.width_spin.setVisible(rectangle)
        self.height_label.setVisible(rectangle)
        self.height_spin.setVisible(rectangle)
        self.radius_label.setVisible(not rectangle)
        self.radius_spin.setVisible(not rectangle)
        self.adjustSize()

    def contour(self) -> SketchRectangle | SketchCircle:
        operation = self.operation_combo.currentData()
        if self.shape_combo.currentData() == "rectangle":
            return SketchRectangle(
                operation,
                self.x_spin.value(),
                self.y_spin.value(),
                self.width_spin.value(),
                self.height_spin.value(),
            )
        return SketchCircle(
            operation,
            self.x_spin.value(),
            self.y_spin.value(),
            self.radius_spin.value(),
        )


class SketchGeometryDialog(QDialog):
    """A compact Abaqus-style modal editor for a planar sketch."""

    def __init__(
        self,
        recipe: SketchGeometry | None = None,
        parent=None,
        *,
        new_contour_operation: str = "material",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建草图" if recipe is None else "编辑草图")
        current = recipe or SketchGeometry(
            "Sketch-1",
            (SketchRectangle("material", 0.0, 0.0, 100.0, 50.0),),
        )
        self.name_edit = QLineEdit(current.name, self)
        self.contours = list(current.contours)
        self.contour_list = QListWidget(self)

        self.add_rectangle_button = QPushButton("添加矩形", self)
        self.add_circle_button = QPushButton("添加圆", self)
        self.edit_button = QPushButton("编辑", self)
        self.delete_button = QPushButton("删除", self)
        self.add_rectangle_button.clicked.connect(
            lambda: self._add_contour(
                SketchRectangle(
                    new_contour_operation,
                    0.0,
                    0.0,
                    100.0,
                    50.0,
                )
            )
        )
        self.add_circle_button.clicked.connect(
            lambda: self._add_contour(
                SketchCircle(new_contour_operation, 0.0, 0.0, 25.0)
            )
        )
        self.edit_button.clicked.connect(self._edit_contour)
        self.delete_button.clicked.connect(self._delete_contour)
        self.contour_list.itemDoubleClicked.connect(self._edit_contour)
        self.contour_list.currentRowChanged.connect(self._update_buttons)
        self.name_edit.textChanged.connect(self._update_validity)

        name_form = QFormLayout()
        configure_form_layout(name_form)
        name_form.addRow("名称", self.name_edit)
        contour_buttons = QHBoxLayout()
        for button in (
            self.add_rectangle_button,
            self.add_circle_button,
            self.edit_button,
            self.delete_button,
        ):
            contour_buttons.addWidget(button)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)

        layout = QVBoxLayout(self)
        layout.addLayout(name_form)
        layout.addWidget(QLabel("轮廓（双击可编辑）", self))
        layout.addWidget(self.contour_list)
        layout.addLayout(contour_buttons)
        layout.addWidget(buttons)
        self.resize(440, 330)
        self._refresh_contours()

    def _contour_text(self, contour) -> str:
        operation = "材料" if contour.operation == "material" else "切除"
        if isinstance(contour, SketchRectangle):
            return (
                f"{operation} · 矩形  X={contour.x:g}, Y={contour.y:g}, "
                f"{contour.width:g} × {contour.height:g}"
            )
        return (
            f"{operation} · 圆  X={contour.x:g}, Y={contour.y:g}, "
            f"R={contour.radius:g}"
        )

    def _refresh_contours(self, selected: int = 0) -> None:
        self.contour_list.clear()
        self.contour_list.addItems(
            [self._contour_text(contour) for contour in self.contours]
        )
        if self.contours:
            self.contour_list.setCurrentRow(min(selected, len(self.contours) - 1))
        self._update_validity()
        self._update_buttons()

    def _update_validity(self, *_args) -> None:
        self.ok_button.setEnabled(
            bool(self.name_edit.text().strip())
            and any(contour.operation == "material" for contour in self.contours)
        )

    def _add_contour(self, default) -> None:
        dialog = SketchContourDialog(default, self)
        if dialog.exec():
            self.contours.append(dialog.contour())
            self._refresh_contours(len(self.contours) - 1)

    def _edit_contour(self, *_args) -> None:
        row = self.contour_list.currentRow()
        if row < 0:
            return
        dialog = SketchContourDialog(self.contours[row], self)
        if dialog.exec():
            self.contours[row] = dialog.contour()
            self._refresh_contours(row)

    def _delete_contour(self) -> None:
        row = self.contour_list.currentRow()
        if row >= 0:
            del self.contours[row]
            self._refresh_contours(row)

    def _update_buttons(self, *_args) -> None:
        selected = self.contour_list.currentRow() >= 0
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def recipe(self) -> SketchGeometry:
        return SketchGeometry(self.name_edit.text(), tuple(self.contours))


class RectangleGeometryDialog(QDialog):
    """Collect the first supported native geometry definition."""

    def __init__(
        self,
        recipe: RectangleGeometry | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("创建矩形几何")
        current = recipe or RectangleGeometry("Rectangle-1", 100.0, 50.0)
        self.name_edit = QLineEdit(current.name, self)
        self.width_spin = _positive_spin_box(self, current.width)
        self.height_spin = _positive_spin_box(self, current.height)

        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("名称", self.name_edit)
        form.addRow("宽度", self.width_spin)
        form.addRow("高度", self.height_spin)
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

    def recipe(self) -> RectangleGeometry:
        return RectangleGeometry(
            self.name_edit.text(),
            self.width_spin.value(),
            self.height_spin.value(),
        )


class DiskGeometryDialog(QDialog):
    """Collect one circular planar geometry definition."""

    def __init__(self, recipe: DiskGeometry | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("创建圆盘几何")
        current = recipe or DiskGeometry("Disk-1", 25.0)
        self.name_edit = QLineEdit(current.name, self)
        self.radius_spin = _positive_spin_box(self, current.radius)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("名称", self.name_edit)
        form.addRow("半径", self.radius_spin)
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

    def recipe(self) -> DiskGeometry:
        return DiskGeometry(self.name_edit.text(), self.radius_spin.value())


class BoxGeometryDialog(QDialog):
    """Collect one axis-aligned box definition."""

    def __init__(self, recipe: BoxGeometry | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("创建长方体几何")
        current = recipe or BoxGeometry("Box-1", 100.0, 50.0, 20.0)
        self.name_edit = QLineEdit(current.name, self)
        self.width_spin = _positive_spin_box(self, current.width)
        self.depth_spin = _positive_spin_box(self, current.depth)
        self.height_spin = _positive_spin_box(self, current.height)
        form = QFormLayout()
        configure_form_layout(form)
        for label, editor in (
            ("名称", self.name_edit),
            ("宽度 X", self.width_spin),
            ("深度 Y", self.depth_spin),
            ("高度 Z", self.height_spin),
        ):
            form.addRow(label, editor)
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

    def recipe(self) -> BoxGeometry:
        return BoxGeometry(
            self.name_edit.text(),
            self.width_spin.value(),
            self.depth_spin.value(),
            self.height_spin.value(),
        )


class CylinderGeometryDialog(QDialog):
    """Collect one positive-Z cylinder definition."""

    def __init__(self, recipe: CylinderGeometry | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("创建圆柱几何")
        current = recipe or CylinderGeometry("Cylinder-1", 25.0, 50.0)
        self.name_edit = QLineEdit(current.name, self)
        self.radius_spin = _positive_spin_box(self, current.radius)
        self.height_spin = _positive_spin_box(self, current.height)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("名称", self.name_edit)
        form.addRow("半径", self.radius_spin)
        form.addRow("高度 Z", self.height_spin)
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

    def recipe(self) -> CylinderGeometry:
        return CylinderGeometry(
            self.name_edit.text(),
            self.radius_spin.value(),
            self.height_spin.value(),
        )


class MoveGeometryDialog(QDialog):
    """Collect a global translation for the current geometry."""

    def __init__(self, base: object, parent=None, *, is_3d: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("移动几何")
        self._base = base
        self.dx_spin = _signed_spin_box(self, 0.0)
        self.dy_spin = _signed_spin_box(self, 0.0)
        self.dz_spin = _signed_spin_box(self, 0.0)
        self.dz_spin.setEnabled(is_3d)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("X 方向距离", self.dx_spin)
        form.addRow("Y 方向距离", self.dy_spin)
        form.addRow("Z 方向距离", self.dz_spin)
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

    def recipe(self) -> MovedGeometry:
        return MovedGeometry(
            self._base,
            self.dx_spin.value(),
            self.dy_spin.value(),
            self.dz_spin.value(),
        )


class RotateGeometryDialog(QDialog):
    """Collect a global-axis rotation for the current geometry."""

    def __init__(self, base: object, parent=None, *, is_3d: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("旋转几何")
        self._base = base
        self.axis_combo = QComboBox(self)
        if is_3d:
            self.axis_combo.addItem("X 轴", "x")
            self.axis_combo.addItem("Y 轴", "y")
        self.axis_combo.addItem("Z 轴", "z")
        self.angle_spin = _signed_spin_box(self, 90.0)
        self.angle_spin.setRange(-360000.0, 360000.0)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("旋转轴", self.axis_combo)
        form.addRow("角度（度）", self.angle_spin)
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

    def recipe(self) -> RotatedGeometry:
        return RotatedGeometry(
            self._base,
            str(self.axis_combo.currentData()),
            self.angle_spin.value(),
        )


class ExtrudeGeometryDialog(QDialog):
    """Collect a positive-Z extrusion height for a planar geometry."""

    def __init__(self, base: object, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("拉伸几何")
        self._base = base
        self.height_spin = _positive_spin_box(self, 10.0)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("拉伸高度 Z", self.height_spin)
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

    def recipe(self) -> ExtrudedGeometry:
        return ExtrudedGeometry(self._base, self.height_spin.value())


class GeometryManagerDialog(QDialog):
    """Show the current feature history as a flat list, not a model tree."""

    def __init__(
        self,
        recipe: object,
        parent=None,
        *,
        can_edit_base: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("几何管理")
        self.setMinimumWidth(400)
        self.operation: str | None = None
        self.feature_list = QListWidget(self)
        self.feature_list.addItems(geometry_feature_rows(recipe))
        self.feature_list.setCurrentRow(self.feature_list.count() - 1)
        self._can_edit_base = bool(can_edit_base)
        self.selected_row = self.feature_list.currentRow()
        self.edit_button = QPushButton("编辑基础草图", self)
        self.delete_button = QPushButton("删除最后特征", self)
        self.clear_button = QPushButton("清空几何", self)
        self.close_button = QPushButton("关闭", self)
        self.edit_button.clicked.connect(lambda: self._finish("edit"))
        self.delete_button.clicked.connect(lambda: self._finish("delete"))
        self.clear_button.clicked.connect(lambda: self._finish("clear"))
        self.close_button.clicked.connect(self.reject)
        self.feature_list.currentRowChanged.connect(self._update_buttons)
        buttons = QHBoxLayout()
        buttons.addWidget(self.edit_button)
        buttons.addWidget(self.delete_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.feature_list)
        layout.addLayout(buttons)
        self._update_buttons()

    def _finish(self, operation: str) -> None:
        self.selected_row = self.feature_list.currentRow()
        self.operation = operation
        self.accept()

    def _update_buttons(self) -> None:
        row = self.feature_list.currentRow()
        self.edit_button.setEnabled(self._can_edit_base and row == 0)
        self.delete_button.setEnabled(
            self.feature_list.count() > 1
            and row == self.feature_list.count() - 1
        )


class MeshControlsDialog(QDialog):
    """Show active mesh controls in one flat Abaqus-style dialog."""

    def __init__(self, settings: MeshSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("网格控制管理")
        self.resize(480, 330)
        self._settings = settings
        self.local_controls = list(settings.local_controls)
        self.control_list = QListWidget(self)
        self.edit_button = QPushButton("编辑", self)
        self.delete_button = QPushButton("删除", self)
        self.clear_button = QPushButton("清除全部局部控制", self)
        self.edit_button.clicked.connect(self._edit)
        self.delete_button.clicked.connect(self._delete)
        self.clear_button.clicked.connect(self._clear)
        self.control_list.itemSelectionChanged.connect(self._update_buttons)
        buttons = QHBoxLayout()
        buttons.addWidget(self.edit_button)
        buttons.addWidget(self.delete_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        standard = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        standard.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        standard.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        standard.accepted.connect(self.accept)
        standard.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.control_list)
        layout.addLayout(buttons)
        layout.addWidget(standard)
        self._refresh()

    def _refresh(self, selected: int = 0) -> None:
        self.control_list.clear()
        shape_names = {
            "triangle": "三角形",
            "quadrilateral": "四边形",
            "tetrahedron": "四面体",
            "hexahedron": "六面体（结构化）",
        }
        self.control_list.addItem(f"全局尺寸  {self._settings.size:g}")
        self.control_list.addItem(f"单元阶次  {self._settings.order} 阶")
        self.control_list.addItem(
            f"网格方法  {shape_names[self._settings.cell_shape]}"
        )
        kind_names = {"point": "点", "edge": "边", "face": "面"}
        for index, control in enumerate(self.local_controls, start=1):
            self.control_list.addItem(
                f"局部控制 {index}  类型={kind_names[control.entity_kind]}"
                f"  尺寸={control.size:g}"
            )
        if self.local_controls:
            self.control_list.setCurrentRow(
                3 + min(selected, len(self.local_controls) - 1)
            )
        self._update_buttons()

    def _selected_local_index(self) -> int | None:
        index = self.control_list.currentRow() - 3
        return index if 0 <= index < len(self.local_controls) else None

    def _edit(self) -> None:
        index = self._selected_local_index()
        if index is None:
            return
        current = self.local_controls[index]
        dialog = LocalMeshControlDialog(
            current.entity_kind,
            current.entity_id,
            self._settings.size,
            self,
            current_size=current.size,
        )
        if dialog.exec():
            self.local_controls[index] = dialog.control()
            self._refresh(index)

    def _delete(self) -> None:
        index = self._selected_local_index()
        if index is None:
            return
        del self.local_controls[index]
        self._refresh(max(0, index - 1))

    def _clear(self) -> None:
        self.local_controls.clear()
        self._refresh()

    def _update_buttons(self) -> None:
        selected = self._selected_local_index() is not None
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)
        self.clear_button.setEnabled(bool(self.local_controls))

    def settings(self) -> MeshSettings:
        return MeshSettings(
            self._settings.size,
            self._settings.order,
            self._settings.cell_shape,
            local_size=None,
            local_controls=tuple(self.local_controls),
        )


class LocalMeshControlDialog(QDialog):
    """Collect a size for the geometry entity selected in the viewport."""

    def __init__(
        self,
        entity_kind: str,
        entity_id: int,
        global_size: float,
        parent=None,
        *,
        current_size: float | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置局部网格")
        self._entity_kind = str(entity_kind)
        self._entity_id = int(entity_id)
        names = {"point": "点", "edge": "边", "face": "面"}
        self.size_spin = _positive_spin_box(
            self,
            float(current_size)
            if current_size is not None
            else float(global_size) / 2.0,
        )
        self.size_spin.setMaximum(max(1.0e-9, float(global_size) - 1.0e-9))
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow(
            "选择对象",
            QLabel(
                f"已选择 1 个{names.get(entity_kind, entity_kind)}",
                self,
            ),
        )
        form.addRow("局部尺寸", self.size_spin)
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

    def control(self) -> LocalMeshControl:
        return LocalMeshControl(
            self._entity_kind,
            self._entity_id,
            self.size_spin.value(),
        )


class NamedRegionDialog(QDialog):
    """Name a selected geometry entity without a permanent property panel."""

    def __init__(
        self,
        entity_kind: str,
        entity_id: int | tuple[int, ...],
        parent=None,
        *,
        suggested_name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("创建命名区域")
        names = {"point": "点", "edge": "边", "face": "面", "body": "体"}
        entity_ids = (
            (int(entity_id),)
            if isinstance(entity_id, int)
            else tuple(sorted({int(value) for value in entity_id}))
        )
        default_names = {
            "point": "PointSet-1",
            "edge": "EdgeSet-1",
            "face": "Surface-1",
            "body": "BodySet-1",
        }
        default_name = suggested_name or default_names.get(
            entity_kind,
            "Region-1",
        )
        self.name_edit = QLineEdit(
            default_name, self
        )
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow(
            "选择对象",
            QLabel(
                f"已选择 {len(entity_ids)} 个"
                f"{names.get(entity_kind, entity_kind)}",
                self,
            ),
        )
        form.addRow("区域名称", self.name_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def region_name(self) -> str:
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("区域名称不能为空")
        return name


class NamedRegionManagerDialog(QDialog):
    """Rename or delete geometry regions without exposing a second tree."""

    def __init__(
        self,
        regions: dict[str, NamedRegion],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("命名区域管理")
        self.regions = deepcopy(regions)
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(("名称", "类型", "实体数量"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.name_edit = QLineEdit(self)
        self.rename_button = QPushButton("改名", self)
        self.delete_button = QPushButton("删除", self)
        self.rename_button.clicked.connect(self._rename)
        self.delete_button.clicked.connect(self._delete)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("区域名称", self))
        controls.addWidget(self.name_edit, 1)
        controls.addWidget(self.rename_button)
        controls.addWidget(self.delete_button)
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

    def _refresh(self, selected: int = 0) -> None:
        self.table.setRowCount(0)
        type_names = {
            "point": "点",
            "edge": "边",
            "face": "面",
            "body": "体",
        }
        for row, region in enumerate(self.regions.values()):
            self.table.insertRow(row)
            values = (
                region.name,
                type_names.get(region.entity_kind, region.entity_kind),
                f"{len(region.entity_ids)} 个",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        if self.regions:
            self.table.selectRow(min(selected, len(self.regions) - 1))
        self._selection_changed()

    def _selected_name(self) -> str | None:
        row = self.table.currentRow()
        names = tuple(self.regions)
        return names[row] if 0 <= row < len(names) else None

    def _selection_changed(self) -> None:
        name = self._selected_name()
        self.name_edit.setText(name or "")
        enabled = name is not None
        self.name_edit.setEnabled(enabled)
        self.rename_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)

    def _rename(self) -> None:
        old_name = self._selected_name()
        new_name = self.name_edit.text().strip()
        if old_name is None or not new_name or new_name == old_name:
            return
        if new_name in self.regions:
            return
        items = list(self.regions.items())
        row = self.table.currentRow()
        region = self.regions[old_name]
        items[row] = (
            new_name,
            NamedRegion(
                new_name,
                region.entity_kind,
                region.entity_ids,
            ),
        )
        self.regions = dict(items)
        self._refresh(row)

    def _delete(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        row = self.table.currentRow()
        del self.regions[name]
        self._refresh(max(0, row - 1))

    def values(self) -> dict[str, NamedRegion]:
        return deepcopy(self.regions)


class BooleanGeometryDialog(QDialog):
    """Create a compatible primitive tool for one boolean feature."""

    def __init__(
        self,
        object_geometry: object,
        operation: str,
        parent=None,
        *,
        is_3d: bool,
    ) -> None:
        super().__init__(parent)
        operation_names = {"fuse": "合并", "cut": "切除", "fragment": "分割"}
        self.setWindowTitle(f"几何{operation_names[operation]}")
        self._object_geometry = object_geometry
        self._operation = operation
        self._is_3d = is_3d
        self.name_edit = QLineEdit(
            f"{getattr(object_geometry, 'name', 'Geometry')}-{operation}",
            self,
        )
        self.tool_name_edit = QLineEdit("Tool-1", self)
        self.tool_combo = QComboBox(self)
        if is_3d:
            self.tool_combo.addItem("长方体", "box")
            self.tool_combo.addItem("圆柱", "cylinder")
        else:
            self.tool_combo.addItem("矩形", "rectangle")
            self.tool_combo.addItem("圆盘", "disk")
        self.x_spin = _signed_spin_box(self, 0.0)
        self.y_spin = _signed_spin_box(self, 0.0)
        self.z_spin = _signed_spin_box(self, 0.0)
        self.z_spin.setEnabled(is_3d)
        self.size_a_spin = _positive_spin_box(self, 10.0)
        self.size_b_spin = _positive_spin_box(self, 10.0)
        self.size_c_spin = _positive_spin_box(self, 10.0)
        self.size_a_label = QLabel(self)
        self.size_b_label = QLabel(self)
        self.size_c_label = QLabel(self)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("结果名称", self.name_edit)
        form.addRow("工具体名称", self.tool_name_edit)
        form.addRow("工具体类型", self.tool_combo)
        form.addRow("位置 X", self.x_spin)
        form.addRow("位置 Y", self.y_spin)
        form.addRow("位置 Z", self.z_spin)
        form.addRow(self.size_a_label, self.size_a_spin)
        form.addRow(self.size_b_label, self.size_b_spin)
        form.addRow(self.size_c_label, self.size_c_spin)
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
        self.tool_combo.currentIndexChanged.connect(self._refresh_dimension_labels)
        self._refresh_dimension_labels()
        self._set_contextual_defaults()

    def _set_contextual_defaults(self) -> None:
        """Start common 2-D cuts with a valid, centered circular tool."""
        target = self._object_geometry
        offset_x = offset_y = 0.0
        while isinstance(target, MovedGeometry):
            offset_x += target.dx
            offset_y += target.dy
            target = target.base
        if (
            not self._is_3d
            and isinstance(target, RectangleGeometry)
            and self.tool_combo.currentData() == "disk"
        ):
            radius = min(target.width, target.height) / 6.0
            self.size_a_spin.setValue(radius)
            self.x_spin.setValue(offset_x + target.width / 2.0)
            self.y_spin.setValue(offset_y + target.height / 2.0)

    def _refresh_dimension_labels(self) -> None:
        tool_type = str(self.tool_combo.currentData())
        if tool_type in {"disk", "cylinder"}:
            labels = ("半径", "高度 Z", "")
            visible = (True, tool_type == "cylinder", False)
        elif tool_type == "box":
            labels = ("宽度 X", "深度 Y", "高度 Z")
            visible = (True, True, True)
        else:
            labels = ("宽度 X", "高度 Y", "")
            visible = (True, True, False)
        for label, text, is_visible, editor in zip(
            (self.size_a_label, self.size_b_label, self.size_c_label),
            labels,
            visible,
            (self.size_a_spin, self.size_b_spin, self.size_c_spin),
        ):
            label.setText(text)
            label.setVisible(is_visible)
            editor.setVisible(is_visible)

    def recipe(self) -> BooleanGeometry:
        tool_type = str(self.tool_combo.currentData())
        tool_name = self.tool_name_edit.text()
        if tool_type == "disk":
            tool = DiskGeometry(tool_name, self.size_a_spin.value())
        elif tool_type == "rectangle":
            tool = RectangleGeometry(
                tool_name,
                self.size_a_spin.value(),
                self.size_b_spin.value(),
            )
        elif tool_type == "cylinder":
            tool = CylinderGeometry(
                tool_name,
                self.size_a_spin.value(),
                self.size_b_spin.value(),
            )
        else:
            tool = BoxGeometry(
                tool_name,
                self.size_a_spin.value(),
                self.size_b_spin.value(),
                self.size_c_spin.value(),
            )
        if any(
            value != 0.0
            for value in (self.x_spin.value(), self.y_spin.value(), self.z_spin.value())
        ):
            tool = MovedGeometry(
                tool,
                self.x_spin.value(),
                self.y_spin.value(),
                self.z_spin.value(),
            )
        return BooleanGeometry(
            self.name_edit.text(),
            self._operation,
            self._object_geometry,
            tool,
        )


class PlateWithHoleGeometryDialog(QDialog):
    """Collect a rectangular plate and one internal circular hole."""

    def __init__(
        self,
        recipe: PlateWithHoleGeometry | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("创建带圆孔矩形板")
        current = recipe or PlateWithHoleGeometry(
            "Plate-With-Hole-1",
            100.0,
            50.0,
            50.0,
            25.0,
            5.0,
        )
        self.name_edit = QLineEdit(current.name, self)
        self.width_spin = _positive_spin_box(self, current.width)
        self.height_spin = _positive_spin_box(self, current.height)
        self.hole_x_spin = _positive_spin_box(self, current.hole_x)
        self.hole_y_spin = _positive_spin_box(self, current.hole_y)
        self.radius_spin = _positive_spin_box(self, current.hole_radius)
        form = QFormLayout()
        configure_form_layout(form)
        for label, editor in (
            ("名称", self.name_edit),
            ("板宽度", self.width_spin),
            ("板高度", self.height_spin),
            ("孔中心 X", self.hole_x_spin),
            ("孔中心 Y", self.hole_y_spin),
            ("孔半径", self.radius_spin),
        ):
            form.addRow(label, editor)
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

    def recipe(self) -> PlateWithHoleGeometry:
        return PlateWithHoleGeometry(
            self.name_edit.text(),
            self.width_spin.value(),
            self.height_spin.value(),
            self.hole_x_spin.value(),
            self.hole_y_spin.value(),
            self.radius_spin.value(),
        )


class MeshSettingsDialog(QDialog):
    """Collect global mesh settings shared by native model generation."""

    def __init__(
        self,
        settings: MeshSettings | None = None,
        parent=None,
        *,
        mesh_dimension: int = 2,
        allow_hexahedron: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("网格设置")
        current = settings or MeshSettings(5.0)
        self._mesh_dimension = int(mesh_dimension)
        self._allow_hexahedron = bool(allow_hexahedron)
        self.size_spin = _positive_spin_box(self, current.size)
        self._local_controls = current.local_controls
        if self._local_controls:
            self.size_spin.setMinimum(
                max(control.size for control in self._local_controls) + 1.0e-9
            )
        self.order_combo = QComboBox(self)
        self.order_combo.addItem("一阶", 1)
        self.order_combo.addItem("二阶", 2)
        self.order_combo.setCurrentIndex(0 if current.order == 1 else 1)
        self.method_combo = QComboBox(self)
        self.method_combo.addItem("自由网格", "free")
        if self._mesh_dimension == 2:
            self.method_combo.addItem("四边形重组", "recombine")
        elif self._allow_hexahedron:
            self.method_combo.addItem("结构化网格", "structured")
        self.shape_combo = QComboBox(self)
        current_method = {
            "quadrilateral": "recombine",
            "hexahedron": "structured",
        }.get(current.cell_shape, "free")
        method_index = self.method_combo.findData(current_method)
        self.method_combo.setCurrentIndex(max(0, method_index))
        self.method_combo.currentIndexChanged.connect(self._refresh_shape_options)
        self._refresh_shape_options()
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("网格方法", self.method_combo)
        form.addRow("单元类型", self.shape_combo)
        form.addRow("单元阶次", self.order_combo)
        form.addRow("全局尺寸", self.size_spin)
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

    def _refresh_shape_options(self) -> None:
        """Expose only element shapes implemented by the selected backend method."""
        method = str(self.method_combo.currentData())
        self.shape_combo.clear()
        if self._mesh_dimension == 3:
            if method == "structured" and self._allow_hexahedron:
                self.shape_combo.addItem("六面体", "hexahedron")
            else:
                self.shape_combo.addItem("四面体", "tetrahedron")
        elif method == "recombine":
            self.shape_combo.addItem("四边形", "quadrilateral")
        else:
            self.shape_combo.addItem("三角形", "triangle")

    def settings(self) -> MeshSettings:
        return MeshSettings(
            size=self.size_spin.value(),
            order=int(self.order_combo.currentData()),
            cell_shape=str(self.shape_combo.currentData()),
            local_size=None,
            local_controls=self._local_controls,
        )
