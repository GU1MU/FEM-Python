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
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QVBoxLayout,
)

from fem.application import (
    DeleteIntent,
    MeshEntityRef,
    NamedRegion,
    RenameIntent,
    derive_geometry_feature_rows,
)
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    resolve_extrusion_source_faces,
)
from fem.mesh import settings as mesh_settings_api
from fem.mesh.settings import LocalMeshControl, MeshSettings

from .dialogs import (
    AdaptivePrecisionDoubleSpinBox,
    CompactDoubleSpinBox,
    configure_form_layout,
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


def _mesh_size_spin_box(parent: QDialog, value: float) -> QDoubleSpinBox:
    editor = AdaptivePrecisionDoubleSpinBox(parent)
    editor.setRange(1.0e-9, 1.0e12)
    editor.setValue(float(value))
    return editor


class GeometryCreationDialog(QDialog):
    """Route the unified geometry command to a 1D, 2D, or 3D workflow."""

    def __init__(
        self,
        parent=None,
        *,
        default_part_name: str = "Part-1",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("geometryCreationDialog")
        self.setWindowTitle("New Part")
        self.part_name_edit = QLineEdit(default_part_name, self)
        self.part_name_edit.setObjectName("nativePartNameEdit")
        self.sketch_size_spin = _positive_spin_box(self, 50.0)
        self.sketch_size_spin.setObjectName("sketchDisplaySizeSpin")
        self.sketch_size_spin.setDecimals(0)
        self.sketch_size_spin.setRange(1.0, 1.0e12)

        self.dimension_list = QListWidget(self)
        self.dimension_list.setObjectName("geometryDimensionList")
        self.dimension_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        for label, value in (
            ("1D Wire Sketch", "1d"),
            ("2D Planar Sketch", "2d"),
            ("3D Primitive", "3d"),
        ):
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, value)
            self.dimension_list.addItem(item)
        self.dimension_list.setCurrentRow(0)
        self.dimension_list.setMinimumHeight(116)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self.dimension_list.itemDoubleClicked.connect(
            lambda _item: self._accept()
        )

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("Part Name", self.part_name_edit)
        form.addRow("Sketch Size", self.sketch_size_spin)
        layout.addLayout(form)
        layout.addWidget(QLabel("Model Dimension", self))
        layout.addWidget(self.dimension_list)
        layout.addWidget(buttons)

    def creation_kind(self) -> str:
        item = self.dimension_list.currentItem()
        if item is None:
            raise RuntimeError("Select a model dimension")
        return str(item.data(Qt.ItemDataRole.UserRole))

    def part_name(self) -> str:
        return self.part_name_edit.text().strip()

    def sketch_size(self) -> float:
        return float(self.sketch_size_spin.value())

    def _accept(self) -> None:
        if not self.part_name():
            self.part_name_edit.setFocus()
            return
        self.accept()


class BasicSolidCreationDialog(QDialog):
    """Choose a 3D primitive before opening its parameter dialog."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("basicSolidCreationDialog")
        self.setWindowTitle("Create 3D Primitive")

        self.solid_combo = QComboBox(self)
        self.solid_combo.setObjectName("basicSolidTypeCombo")
        for label, value in (
            ("Box", "box"),
            ("Cylinder", "cylinder"),
        ):
            self.solid_combo.addItem(label, value)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Solid Type", self))
        layout.addWidget(self.solid_combo)
        layout.addWidget(buttons)

    def solid_kind(self) -> str:
        if self.solid_combo.currentIndex() < 0:
            raise RuntimeError("Select a 3D primitive")
        return str(self.solid_combo.currentData())


class SketchContourDialog(QDialog):
    """Edit one rectangle or circle contour without exposing CAD internals."""

    def __init__(self, contour=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sketch Contour")
        current = contour or SketchRectangle("material", 0.0, 0.0, 100.0, 50.0)

        self.operation_combo = QComboBox(self)
        self.operation_combo.addItem("Add Material", "material")
        self.operation_combo.addItem("Cut Material", "cut")
        self.operation_combo.setCurrentIndex(
            0 if current.operation == "material" else 1
        )
        self.shape_combo = QComboBox(self)
        self.shape_combo.addItem("Rectangle", "rectangle")
        self.shape_combo.addItem("Circle", "circle")
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
        form.addRow("Operation", self.operation_combo)
        form.addRow("Shape", self.shape_combo)
        form.addRow("X", self.x_spin)
        form.addRow("Y", self.y_spin)
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow("Radius", self.radius_spin)
        self.width_label = form.labelForField(self.width_spin)
        self.height_label = form.labelForField(self.height_spin)
        self.radius_label = form.labelForField(self.radius_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
        self.setWindowTitle("New Sketch" if recipe is None else "Edit Sketch")
        current = recipe or SketchGeometry(
            "Sketch-1",
            (SketchRectangle("material", 0.0, 0.0, 100.0, 50.0),),
        )
        self.name_edit = QLineEdit(current.name, self)
        self.contours = list(current.contours)
        self.contour_list = QListWidget(self)

        self.add_rectangle_button = QPushButton("Add Rectangle", self)
        self.add_circle_button = QPushButton("Add Circle", self)
        self.edit_button = QPushButton("Edit", self)
        self.delete_button = QPushButton("Delete", self)
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
        name_form.addRow("Name", self.name_edit)
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
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)

        layout = QVBoxLayout(self)
        layout.addLayout(name_form)
        layout.addWidget(QLabel("Contours (Double-click to Edit)", self))
        layout.addWidget(self.contour_list)
        layout.addLayout(contour_buttons)
        layout.addWidget(buttons)
        self.resize(440, 330)
        self._refresh_contours()

    def _contour_text(self, contour) -> str:
        operation = "Material" if contour.operation == "material" else "Cut"
        if isinstance(contour, SketchRectangle):
            return (
                f"{operation} - Rectangle  X={contour.x:g}, Y={contour.y:g}, "
                f"{contour.width:g} × {contour.height:g}"
            )
        return (
            f"{operation} - Circle  X={contour.x:g}, Y={contour.y:g}, "
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
        self.setWindowTitle("Create Rectangle")
        current = recipe or RectangleGeometry("Rectangle-1", 100.0, 50.0)
        self.name_edit = QLineEdit(current.name, self)
        self.width_spin = _positive_spin_box(self, current.width)
        self.height_spin = _positive_spin_box(self, current.height)

        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("Name", self.name_edit)
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
        self.setWindowTitle("Create Disk")
        current = recipe or DiskGeometry("Disk-1", 25.0)
        self.name_edit = QLineEdit(current.name, self)
        self.radius_spin = _positive_spin_box(self, current.radius)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("Name", self.name_edit)
        form.addRow("Radius", self.radius_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
        self.setWindowTitle("Create Box")
        current = recipe or BoxGeometry("Box-1", 100.0, 50.0, 20.0)
        self.name_edit = QLineEdit(current.name, self)
        self.width_spin = _positive_spin_box(self, current.width)
        self.depth_spin = _positive_spin_box(self, current.depth)
        self.height_spin = _positive_spin_box(self, current.height)
        form = QFormLayout()
        configure_form_layout(form)
        for label, editor in (
            ("Name", self.name_edit),
            ("Width X", self.width_spin),
            ("Depth Y", self.depth_spin),
            ("Height Z", self.height_spin),
        ):
            form.addRow(label, editor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
        self.setWindowTitle("Create Cylinder")
        current = recipe or CylinderGeometry("Cylinder-1", 25.0, 50.0)
        self.name_edit = QLineEdit(current.name, self)
        self.radius_spin = _positive_spin_box(self, current.radius)
        self.height_spin = _positive_spin_box(self, current.height)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("Name", self.name_edit)
        form.addRow("Radius", self.radius_spin)
        form.addRow("Height Z", self.height_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
        self.setWindowTitle("Move Geometry")
        self._base = base
        self.dx_spin = _signed_spin_box(self, 0.0)
        self.dy_spin = _signed_spin_box(self, 0.0)
        self.dz_spin = _signed_spin_box(self, 0.0)
        self.dz_spin.setEnabled(is_3d)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("Distance X", self.dx_spin)
        form.addRow("Distance Y", self.dy_spin)
        form.addRow("Distance Z", self.dz_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
        self.setWindowTitle("Rotate Geometry")
        self._base = base
        self.axis_combo = QComboBox(self)
        if is_3d:
            self.axis_combo.addItem("X Axis", "x")
            self.axis_combo.addItem("Y Axis", "y")
        self.axis_combo.addItem("Z Axis", "z")
        self.angle_spin = _signed_spin_box(self, 90.0)
        self.angle_spin.setRange(-360000.0, 360000.0)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("Rotation Axis", self.axis_combo)
        form.addRow("Angle (Degrees)", self.angle_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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

    def __init__(
        self,
        base: object,
        parent=None,
        *,
        source_face_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extrude Geometry")
        self._base = base
        self._source_face_ids = resolve_extrusion_source_faces(
            base,
            source_face_ids,
        ).face_ids
        self.height_spin = _positive_spin_box(self, 10.0)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("Direction", QLabel("+Z", self))
        form.addRow("Height", self.height_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def recipe(self) -> ExtrudedGeometry:
        return ExtrudedGeometry(
            self._base,
            self.height_spin.value(),
            self._source_face_ids,
        )


class SweepGeometryDialog(QDialog):
    """Collect a global axis and angle for a rotational Profile sweep."""

    def __init__(
        self,
        base: object,
        parent=None,
        *,
        source_face_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sweep Geometry")
        self._base = base
        self._source_face_ids = resolve_extrusion_source_faces(
            base,
            source_face_ids,
        ).face_ids
        self.axis_combo = QComboBox(self)
        for label, axis in (("X Axis", "x"), ("Y Axis", "y"), ("Z Axis", "z")):
            self.axis_combo.addItem(label, axis)
        self.angle_spin = CompactDoubleSpinBox(self)
        self.angle_spin.setRange(1.0e-6, 360.0)
        self.angle_spin.setDecimals(6)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setValue(360.0)
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("Sweep Axis", self.axis_combo)
        form.addRow("Angle", self.angle_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def recipe(self) -> RevolvedGeometry:
        return RevolvedGeometry(
            self._base,
            str(self.axis_combo.currentData()),
            self.angle_spin.value(),
            self._source_face_ids,
        )


class GeometryManagerDialog(QDialog):
    """Show the current feature history as a flat list, not a model tree."""

    def __init__(
        self,
        recipe: object,
        parent=None,
        *,
        can_edit_base: bool = False,
        base_label: str = "Edit Base Sketch",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Geometry Manager")
        self.setMinimumWidth(400)
        self.operation: str | None = None
        self.feature_list = QListWidget(self)
        self.feature_list.addItems(derive_geometry_feature_rows(recipe))
        self.feature_list.setCurrentRow(self.feature_list.count() - 1)
        self._can_edit_base = bool(can_edit_base)
        self.selected_row = self.feature_list.currentRow()
        self.edit_button = QPushButton(base_label, self)
        self.delete_button = QPushButton("Delete Last Feature", self)
        self.clear_button = QPushButton("Clear Geometry", self)
        self.close_button = QPushButton("Close", self)
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
        self.setWindowTitle("Mesh Control Manager")
        self.resize(480, 330)
        self._settings = settings
        self.local_controls = list(settings.local_controls)
        self.control_list = QListWidget(self)
        self.edit_button = QPushButton("Edit", self)
        self.delete_button = QPushButton("Delete", self)
        self.clear_button = QPushButton("Clear Local Controls", self)
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
        standard.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        standard.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
            "line": "Line Mesh",
            "triangle": "Triangle",
            "quadrilateral": "Quadrilateral",
            "tetrahedron": "Tetrahedron",
            "hexahedron": "Hexahedron (Structured)",
        }
        if self._settings.line_element_type == "Truss2":
            self.control_list.addItem(
                "Truss2 Mesh  1 element per line segment"
            )
        else:
            self.control_list.addItem(f"Global Size  {self._settings.size:g}")
        self.control_list.addItem(f"Element Order  {self._settings.order}")
        self.control_list.addItem(
            f"Mesh Method  {shape_names.get(self._settings.cell_shape, self._settings.cell_shape)}"
        )
        if self._settings.cell_shape == "line":
            self.control_list.addItem(
                "Element Formulation  "
                f"{self._settings.line_element_type}"
            )
        kind_names = {"point": "Point", "edge": "Edge", "face": "Face"}
        local_offset = 4 if self._settings.cell_shape == "line" else 3
        for index, control in enumerate(self.local_controls, start=1):
            self.control_list.addItem(
                f"Local Control {index}  Type={kind_names[control.target.kind]}"
                f"  Size={control.size:g}"
            )
        if self.local_controls:
            self.control_list.setCurrentRow(
                local_offset + min(selected, len(self.local_controls) - 1)
            )
        self._update_buttons()

    def _selected_local_index(self) -> int | None:
        offset = 4 if self._settings.cell_shape == "line" else 3
        index = self.control_list.currentRow() - offset
        return index if 0 <= index < len(self.local_controls) else None

    def _edit(self) -> None:
        index = self._selected_local_index()
        if index is None:
            return
        current = self.local_controls[index]
        dialog = LocalMeshControlDialog(
            current.target,
            self._settings.size,
            self,
            current_size=current.size,
            falloff=current.falloff,
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
            local_controls=tuple(self.local_controls),
            line_element_type=self._settings.line_element_type,
        )


class LocalMeshControlDialog(QDialog):
    """Collect a size for the geometry entity selected in the viewport."""

    def __init__(
        self,
        target: LogicalEntityRef,
        global_size: float,
        parent=None,
        *,
        current_size: float | None = None,
        falloff: object | None = None,
    ) -> None:
        super().__init__(parent)
        if type(target) is not LogicalEntityRef:
            raise TypeError(
                "Local mesh control dialog accepts only LogicalEntityRef"
            )
        self.setWindowTitle("Set Local Mesh")
        self._target = target
        self._falloff = (
            falloff
            if falloff is not None
            else mesh_settings_api.MeshSizeFalloff(
                "global_size",
                0.0,
                2.0,
            )
        )
        names = {"point": "Point", "edge": "Edge", "face": "Face"}
        self.size_spin = _mesh_size_spin_box(
            self,
            float(current_size)
            if current_size is not None
            else float(global_size) / 2.0,
        )
        self.size_spin.setMaximum(max(1.0e-9, float(global_size) - 1.0e-9))
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow(
            "Selection",
            QLabel(
                f"Selected: 1 {names.get(target.kind, target.kind)}",
                self,
            ),
        )
        form.addRow("Local Size", self.size_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def control(self) -> LocalMeshControl:
        return LocalMeshControl(
            self._target,
            self.size_spin.value(),
            self._falloff,
        )


class NamedRegionDialog(QDialog):
    """Name a selected scope without a permanent property panel."""

    def __init__(
        self,
        references: tuple[MeshEntityRef, ...],
        parent=None,
        *,
        suggested_name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Scope")
        names = {
            "node": "Node",
            "edge": "Edge",
            "face": "Face",
            "element": "Element",
        }
        if any(
            type(reference) is not MeshEntityRef
            for reference in references
        ):
            raise TypeError(
                "Scope dialog accepts only MeshEntityRef"
            )
        canonical_references = tuple(
            sorted(
                set(references),
                key=lambda reference: (
                    reference.kind,
                    reference.identity,
                    reference.node_ids,
                ),
            )
        )
        if not canonical_references:
            raise ValueError("A scope requires at least one mesh entity")
        kinds = {
            reference.kind
            for reference in canonical_references
        }
        if len(kinds) != 1:
            raise ValueError("A scope can contain only one mesh entity type")
        entity_kind = canonical_references[0].kind
        default_names = {
            "node": "NodeSet-1",
            "edge": "EdgeSet-1",
            "face": "Surface-1",
            "element": "ElementSet-1",
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
            "Selection",
            QLabel(
                f"{names.get(entity_kind, entity_kind)} Count: "
                f"{len(canonical_references)}",
                self,
            ),
        )
        form.addRow("Scope Name", self.name_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def region_name(self) -> str:
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("Scope name is required")
        return name


class NamedRegionManagerDialog(QDialog):
    """Rename or delete scopes without exposing a second tree."""

    def __init__(
        self,
        regions: dict[str, NamedRegion],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scope Manager")
        self.regions = deepcopy(regions)
        self._original_names: tuple[str, ...] = tuple(regions)
        self._origins: dict[str, str | None] = {
            name: name for name in regions
        }
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(("Name", "Type", "Entity Count"))
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
        self.rename_button = QPushButton("Rename", self)
        self.delete_button = QPushButton("Delete", self)
        self.rename_button.clicked.connect(self._rename)
        self.delete_button.clicked.connect(self._delete)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Scope Name", self))
        controls.addWidget(self.name_edit, 1)
        controls.addWidget(self.rename_button)
        controls.addWidget(self.delete_button)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
            "node": "Node",
            "edge": "Edge",
            "face": "Surface",
            "element": "Element",
        }
        for row, region in enumerate(self.regions.values()):
            self.table.insertRow(row)
            values = (
                region.name,
                type_names.get(region.entity_kind, region.entity_kind),
                f"{len(region.references)}",
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
                region.references,
            ),
        )
        self.regions = dict(items)
        origin = self._origins.pop(old_name)
        self._origins[new_name] = origin
        self._refresh(row)

    def _delete(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        row = self.table.currentRow()
        del self.regions[name]
        del self._origins[name]
        self._refresh(max(0, row - 1))

    def values(self) -> dict[str, NamedRegion]:
        return deepcopy(self.regions)

    def rename_intents(self) -> tuple[RenameIntent, ...]:
        return tuple(
            RenameIntent(origin, current_name)
            for current_name, origin in self._origins.items()
            if origin is not None and origin != current_name
        )

    def delete_intents(self) -> tuple[DeleteIntent, ...]:
        retained = {origin for origin in self._origins.values() if origin is not None}
        return tuple(
            DeleteIntent(name)
            for name in self._original_names
            if name not in retained
        )


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
        operation_names = {"fuse": "Fuse", "cut": "Cut", "fragment": "Fragment"}
        self.setWindowTitle(f"{operation_names[operation]} Geometry")
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
            self.tool_combo.addItem("Box", "box")
            self.tool_combo.addItem("Cylinder", "cylinder")
        else:
            self.tool_combo.addItem("Rectangle", "rectangle")
            self.tool_combo.addItem("Disk", "disk")
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
        form.addRow("Result Name", self.name_edit)
        form.addRow("Tool Body Name", self.tool_name_edit)
        form.addRow("Tool Body Type", self.tool_combo)
        form.addRow("Position X", self.x_spin)
        form.addRow("Position Y", self.y_spin)
        form.addRow("Position Z", self.z_spin)
        form.addRow(self.size_a_label, self.size_a_spin)
        form.addRow(self.size_b_label, self.size_b_spin)
        form.addRow(self.size_c_label, self.size_c_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
            labels = ("Radius", "Height Z", "")
            visible = (True, tool_type == "cylinder", False)
        elif tool_type == "box":
            labels = ("Width X", "Depth Y", "Height Z")
            visible = (True, True, True)
        else:
            labels = ("Width X", "Height Y", "")
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
        self.setWindowTitle("Rectangular Plate with Circular Hole")
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
            ("Name", self.name_edit),
            ("Plate Width", self.width_spin),
            ("Plate Height", self.height_spin),
            ("Hole Center X", self.hole_x_spin),
            ("Hole Center Y", self.hole_y_spin),
            ("Hole Radius", self.radius_spin),
        ):
            form.addRow(label, editor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
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
        suggested_size: float = 5.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mesh Settings")
        self._mesh_dimension = int(mesh_dimension)
        self._allow_hexahedron = bool(allow_hexahedron)
        current_size = (
            settings.size if settings is not None else float(suggested_size)
        )
        self.size_spin = _mesh_size_spin_box(self, current_size)
        self._local_controls = () if settings is None else settings.local_controls
        if self._local_controls:
            self.size_spin.setMinimum(
                max(control.size for control in self._local_controls) + 1.0e-9
            )
        self.order_combo = QComboBox(self)
        self.order_combo.addItem("First Order", 1)
        self.order_combo.addItem("Second Order", 2)
        self.order_combo.setCurrentIndex(
            0 if settings is None or settings.order == 1 else 1
        )
        self.shape_combo = QComboBox(self)
        if self._mesh_dimension == 1:
            self.shape_combo.addItem("Line Mesh", "line")
        elif self._mesh_dimension == 3:
            self.shape_combo.addItem("Tetrahedron", "tetrahedron")
            if self._allow_hexahedron:
                self.shape_combo.addItem("Hexahedron", "hexahedron")
        else:
            self.shape_combo.addItem("Triangle", "triangle")
            self.shape_combo.addItem("Quadrilateral", "quadrilateral")
        current_shape = settings.cell_shape if settings is not None else ""
        shape_index = self.shape_combo.findData(current_shape)
        self.shape_combo.setCurrentIndex(max(0, shape_index))
        self.formulation_combo = None
        if self._mesh_dimension == 1:
            self.formulation_combo = QComboBox(self)
            self.formulation_combo.setObjectName("lineElementFormulationCombo")
            self.formulation_combo.addItem("Select Element Formulation", None)
            self.formulation_combo.addItem("Truss2", "Truss2")
            self.formulation_combo.addItem("Beam2", "Beam2")
            if settings is not None:
                index = self.formulation_combo.findData(
                    settings.line_element_type
                )
                if index >= 0:
                    self.formulation_combo.setCurrentIndex(index)
            self.formulation_combo.currentIndexChanged.connect(
                self._refresh_line_acceptance
            )
        form = QFormLayout()
        configure_form_layout(form)
        form.addRow("Element Type", self.shape_combo)
        form.addRow("Element Order", self.order_combo)
        if self.formulation_combo is not None:
            form.addRow("Element Formulation", self.formulation_combo)
        form.addRow("Global Size", self.size_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        if self._mesh_dimension == 1:
            self.shape_combo.setEnabled(False)
            self.order_combo.setEnabled(False)
            self._refresh_line_acceptance()

    def settings(self) -> MeshSettings:
        if self._mesh_dimension == 1:
            formulation = (
                None
                if self.formulation_combo is None
                else self.formulation_combo.currentData()
            )
            if formulation not in {"Truss2", "Beam2"}:
                raise ValueError("select Truss2 or Beam2 for a line mesh")
            if formulation == "Truss2" and self._local_controls:
                raise ValueError(
                    "Truss2 uses one element per Wire member; remove local "
                    "mesh controls before switching formulations"
                )
            return MeshSettings(
                size=self.size_spin.value(),
                order=1,
                cell_shape="line",
                local_controls=self._local_controls,
                line_element_type=str(formulation),
            )
        return MeshSettings(
            size=self.size_spin.value(),
            order=int(self.order_combo.currentData()),
            cell_shape=str(self.shape_combo.currentData()),
            local_controls=self._local_controls,
        )

    def _refresh_line_acceptance(self) -> None:
        if self._mesh_dimension != 1:
            return
        formulation = self.formulation_combo.currentData()
        truss_policy = formulation == "Truss2"
        self.size_spin.setEnabled(not truss_policy)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            formulation in {"Truss2", "Beam2"}
            and not (truss_policy and self._local_controls)
        )

    def _accept(self) -> None:
        try:
            self.settings()
        except (TypeError, ValueError):
            self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return
        self.accept()
