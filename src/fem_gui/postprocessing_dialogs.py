"""结果查询、结果显示、视口显示和云图设置弹窗。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import floor, isfinite, log10
import re
from typing import Any

import numpy as np

from fem.results import (
    FieldAssociation,
    FieldAvailability,
    FieldMaterializationKey,
    FieldState,
    ResultCatalog,
    ResultProvider,
    ResultProbeKind,
    ResultProbeRequest,
    ResultProbeResult,
    ResultProbeTarget,
    ResultQuery,
    ResultQueryRecord,
    ResultQueryResult,
    ResultFrameKey,
    ResultPathRequest,
    ResultPathResult,
    ResultSourceKey,
    ScalarFieldSelection,
)
from fem.results import encode_result_region_key
from PySide6.QtCore import QSignalBlocker, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOptionSlider,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dialogs import CompactDoubleSpinBox, configure_form_layout
from .result_presentation import (
    result_field_has_section_points,
    result_field_is_beam_section,
    result_field_position_label,
    result_field_is_visible,
    result_region_display_labels,
    result_provider_section_point_labels,
    section_point_relative_position_label,
    visible_result_fields,
)
from .theme import COLORS
from .view_cut_state import (
    VIEW_CUT_AXES,
    default_view_cut_settings,
    normalize_view_cut_settings,
)
from .visualization.colormaps import (
    ABAQUS_RAINBOW,
    COLORMAP_CHOICES,
    CUSTOM_COLORMAP,
    DEFAULT_CUSTOM_COLOR_STOPS,
    interpolate_color_stops,
    normalize_color_stops,
    preview_contour_colors,
)
from .visualization.contour_rendering import (
    CONTOUR_EDGE_ALL,
    CONTOUR_EDGE_EXTERIOR,
    CONTOUR_EDGE_FEATURE,
    CONTOUR_EDGE_FREE,
    CONTOUR_EDGE_GEOMETRY,
    CONTOUR_EDGE_NONE,
    CONTOUR_RENDER_FILLED,
    CONTOUR_RENDER_HIDDEN_LINE,
    CONTOUR_RENDER_SHADED,
    CONTOUR_RENDER_WIREFRAME,
)


@dataclass(frozen=True, slots=True)
class TypedResultDisplaySettings:
    """Catalog-native result display state with a complete field selection."""

    shape_mode: str
    contour_enabled: bool
    selection: ScalarFieldSelection
    scale_mode: str
    scale_value: float
    overlay_undeformed: bool
    show_edges: bool

    def __post_init__(self) -> None:
        _validate_typed_display_options(
            shape_mode=self.shape_mode,
            contour_enabled=self.contour_enabled,
            selection=self.selection,
            scale_mode=self.scale_mode,
            scale_value=self.scale_value,
            overlay_undeformed=self.overlay_undeformed,
            show_edges=self.show_edges,
        )


@dataclass(frozen=True, slots=True)
class _TypedQueryMode:
    association: FieldAssociation

    def __post_init__(self) -> None:
        if type(self.association) is not FieldAssociation:
            raise TypeError("association must be FieldAssociation")


class TypedResultDisplayDialog(QDialog):
    """从 immutable result catalog 选择完整的 scalar field identity。"""

    applyRequested = Signal(TypedResultDisplaySettings)

    def __init__(
        self,
        catalog: ResultCatalog,
        *,
        current_selection: ScalarFieldSelection,
        section_point_labels: Mapping[int, str] | None = None,
        shape_mode: str,
        contour_enabled: bool,
        scale_mode: str,
        scale_value: float,
        overlay_undeformed: bool,
        show_edges: bool,
        parent=None,
    ) -> None:
        if type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be ResultCatalog")
        _validate_typed_display_selection(catalog, current_selection)
        initial = TypedResultDisplaySettings(
            shape_mode=shape_mode,
            contour_enabled=contour_enabled,
            selection=current_selection,
            scale_mode=scale_mode,
            scale_value=scale_value,
            overlay_undeformed=overlay_undeformed,
            show_edges=show_edges,
        )

        super().__init__(parent)
        self.setWindowTitle("结果显示")
        self.setMinimumWidth(420)
        self._catalog = catalog
        self._section_point_labels = dict(section_point_labels or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        self.step_combo = QComboBox(self)
        self.step_combo.addItem(catalog.source.step_name, catalog.source)
        self.shape_combo = QComboBox(self)
        self.shape_combo.addItem("未变形形状", "undeformed")
        self.shape_combo.addItem("变形形状", "deformed")
        self.shape_combo.setCurrentIndex(
            self.shape_combo.findData(initial.shape_mode)
        )
        self.contour_checkbox = QCheckBox("显示云图", self)
        self.contour_checkbox.setChecked(initial.contour_enabled)
        self.field_combo = QComboBox(self)
        self.component_combo = QComboBox(self)
        self.availability_label = QLabel(self)
        self.availability_label.setWordWrap(True)
        for availability in visible_result_fields(catalog.fields):
            self.field_combo.addItem(
                _typed_result_display_field_label(
                    availability,
                    self._section_point_labels,
                ),
                availability.key,
            )
        self.field_combo.setCurrentIndex(
            self.field_combo.findData(current_selection.field_key)
        )
        form.addRow("结果步：", self.step_combo)
        form.addRow("几何形状：", self.shape_combo)
        form.addRow(self.contour_checkbox)
        form.addRow("场变量：", self.field_combo)
        form.addRow("分量：", self.component_combo)
        form.addRow("字段状态：", self.availability_label)
        layout.addLayout(form)

        self.scale_group = QGroupBox("变形比例", self)
        scale_layout = QVBoxLayout(self.scale_group)
        self.auto_scale = QRadioButton("自动", self.scale_group)
        self.real_scale = QRadioButton("真实比例", self.scale_group)
        self.custom_scale = QRadioButton("指定比例", self.scale_group)
        scale_buttons = QButtonGroup(self.scale_group)
        for button in (
            self.auto_scale,
            self.real_scale,
            self.custom_scale,
        ):
            scale_buttons.addButton(button)
        self.scale_value = CompactDoubleSpinBox(self.scale_group)
        self.scale_value.setRange(0.0, 1.0e12)
        self.scale_value.setDecimals(6)
        self.scale_value.setValue(initial.scale_value)
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_scale)
        custom_row.addWidget(self.scale_value, 1)
        scale_layout.addWidget(self.auto_scale)
        scale_layout.addWidget(self.real_scale)
        scale_layout.addLayout(custom_row)
        {
            "auto": self.auto_scale,
            "real": self.real_scale,
            "custom": self.custom_scale,
        }[initial.scale_mode].setChecked(True)
        layout.addWidget(self.scale_group)

        self.overlay_checkbox = QCheckBox("叠加未变形轮廓", self)
        self.overlay_checkbox.setChecked(initial.overlay_undeformed)
        self.edges_checkbox = QCheckBox("显示单元边", self)
        self.edges_checkbox.setChecked(initial.show_edges)
        layout.addWidget(self.overlay_checkbox)
        layout.addWidget(self.edges_checkbox)

        self.button_box = _dialog_buttons(self)
        self.apply_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Apply
        )
        self.ok_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Ok
        )
        self.apply_button.clicked.connect(self.apply)
        self.button_box.accepted.connect(self.accept_with_apply)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._sync_components(
            preferred_component=current_selection.component,
        )
        self.field_combo.currentIndexChanged.connect(
            self._field_changed
        )
        self.shape_combo.currentIndexChanged.connect(
            self._refresh_mode_state
        )
        self.contour_checkbox.toggled.connect(
            self._refresh_mode_state
        )
        self._refresh_availability()
        self._refresh_mode_state()

    @property
    def catalog(self) -> ResultCatalog:
        """返回 dialog 绑定的 exact immutable catalog。"""

        return self._catalog

    @property
    def source(self) -> ResultSourceKey:
        """返回 catalog 所属的 exact result source。"""

        return self._catalog.source

    def current_availability(self) -> FieldAvailability:
        """返回当前字段的完整 catalog entry。"""

        key = self.field_combo.currentData()
        if type(key) is not FieldMaterializationKey:
            raise RuntimeError("no typed result field is selected")
        for availability in self._catalog.fields:
            if availability.key == key:
                return availability
        raise RuntimeError("field key is outside the dialog catalog")

    def current_selection(self) -> ScalarFieldSelection:
        """返回当前完整 materialization key 与 descriptor component。"""

        availability = self.current_availability()
        component = self.component_combo.currentData()
        if type(component) is not str:
            raise RuntimeError("no typed scalar component is selected")
        if component not in availability.descriptor.columns:
            raise RuntimeError(
                "selected component is outside the field descriptor"
            )
        return ScalarFieldSelection(availability.key, component)

    def settings(self) -> TypedResultDisplaySettings:
        """返回当前 catalog-native 显示设置。"""

        scale_mode = (
            "auto"
            if self.auto_scale.isChecked()
            else "real"
            if self.real_scale.isChecked()
            else "custom"
        )
        return TypedResultDisplaySettings(
            shape_mode=self.shape_combo.currentData(),
            contour_enabled=self.contour_checkbox.isChecked(),
            selection=self.current_selection(),
            scale_mode=scale_mode,
            scale_value=float(self.scale_value.value()),
            overlay_undeformed=self.overlay_checkbox.isChecked(),
            show_edges=self.edges_checkbox.isChecked(),
        )

    def apply(self) -> None:
        """仅为 READY/LAZY selection 发出 typed settings。"""

        if self.current_availability().state is FieldState.UNAVAILABLE:
            return
        self.applyRequested.emit(self.settings())

    def accept_with_apply(self) -> None:
        """提交可显示字段并关闭对话框。"""

        if self.current_availability().state is FieldState.UNAVAILABLE:
            return
        self.apply()
        self.accept()

    def _field_changed(self, *_args: object) -> None:
        self._sync_components()
        self._refresh_availability()

    def _sync_components(
        self,
        *,
        preferred_component: str | None = None,
    ) -> None:
        if preferred_component is None:
            candidate = self.component_combo.currentData()
            if type(candidate) is str:
                preferred_component = candidate
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        availability = self.current_availability()
        for component in availability.descriptor.columns:
            self.component_combo.addItem(component, component)
        selected_index = self.component_combo.findData(preferred_component)
        if selected_index < 0:
            selected_index = self.component_combo.findData(
                availability.descriptor.default_component
            )
        self.component_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        self.component_combo.blockSignals(False)

    def _refresh_availability(self) -> None:
        availability = self.current_availability()
        self.availability_label.setText(
            _typed_result_display_availability_text(availability)
        )
        can_submit = availability.state is not FieldState.UNAVAILABLE
        self.apply_button.setEnabled(can_submit)
        self.ok_button.setEnabled(can_submit)

    def _refresh_mode_state(self) -> None:
        contour_enabled = self.contour_checkbox.isChecked()
        self.field_combo.setEnabled(contour_enabled)
        self.component_combo.setEnabled(contour_enabled)
        deformed = self.shape_combo.currentData() == "deformed"
        self.scale_group.setEnabled(deformed)
        self.overlay_checkbox.setEnabled(deformed)


class _ThinHorizontalSlider(QSlider):
    """使用细轨道和紧凑滑块，避免原生样式在 Windows 上被拉高。"""

    _margin = 5
    _track_height = 2
    _handle_size = 10

    def __init__(self, parent=None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setFixedHeight(18)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        span = max(0, self.width() - 2 * self._margin)
        position = QStyle.sliderPositionFromValue(
            self.minimum(),
            self.maximum(),
            self.sliderPosition(),
            span,
            option.upsideDown,
        )
        center_y = self.height() / 2
        track = QRectF(
            self._margin,
            center_y - self._track_height / 2,
            span,
            self._track_height,
        )
        handle = QRectF(
            self._margin + position - self._handle_size / 2,
            center_y - self._handle_size / 2,
            self._handle_size,
            self._handle_size,
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#d4d9dd"))
        painter.drawRoundedRect(track, 1, 1)
        if self.isEnabled():
            handle_color = "#9fa9b1" if not self.underMouse() else "#7f8b94"
        else:
            handle_color = COLORS["disabled"]
        painter.setBrush(QColor(handle_color))
        painter.drawEllipse(handle)


class _ContourColorRampPreview(QWidget):
    """Compact colour-ramp preview used by the contour options page."""

    def __init__(self, colors: tuple[str, ...] | list[str], parent=None) -> None:
        super().__init__(parent)
        self._colors = tuple(str(color) for color in colors)
        self.setMinimumHeight(24)
        self.setMaximumHeight(28)
        self.setMinimumWidth(220)

    def set_colors(self, colors: tuple[str, ...] | list[str]) -> None:
        self._colors = tuple(str(color) for color in colors)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        gradient = QLinearGradient(
            rect.left(),
            rect.top(),
            rect.right(),
            rect.top(),
        )
        colors = self._colors or ("#0000ff", "#ff0000")
        for index, color in enumerate(colors):
            position = index / max(1, len(colors) - 1)
            gradient.setColorAt(position, QColor(color))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#8c959e")))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(rect, 2.0, 2.0)


class _ContourColorButton(QPushButton):
    """A small colour swatch button that emits a canonical hex colour."""

    colorChanged = Signal(str)

    def __init__(self, color: str, parent=None) -> None:
        super().__init__(parent)
        self._color = "#000000"
        self.setMinimumWidth(94)
        self.setToolTip("点击选择颜色")
        self.clicked.connect(self._choose_color)
        self.set_color(color)

    def color(self) -> str:
        return self._color

    def set_color(self, color: str, *, emit: bool = False) -> None:
        selected = QColor(str(color))
        if not selected.isValid():
            selected = QColor(self._color)
        canonical = selected.name().lower()
        changed = canonical != self._color
        self._color = canonical
        text_color = "#ffffff" if selected.lightness() < 145 else "#20262d"
        self.setText(canonical.upper())
        self.setStyleSheet(
            "QPushButton {"
            f"background-color: {canonical}; color: {text_color};"
            " border: 1px solid #8c959e; border-radius: 2px;"
            " padding: 2px 8px; min-height: 22px;"
            "}"
        )
        if emit and changed:
            self.colorChanged.emit(canonical)

    def _choose_color(self) -> None:
        selected = QColorDialog.getColor(
            QColor(self._color),
            self,
            "选择云图颜色",
        )
        if selected.isValid():
            self.set_color(selected.name(), emit=True)


class _DisplayColorControl(QWidget):
    """Choose an explicit display colour or defer to the viewport palette."""

    changed = Signal()

    def __init__(
        self,
        value: object = "auto",
        *,
        fallback: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.auto_checkbox = QCheckBox("自动", self)
        self.button = _ContourColorButton(fallback, self)
        self.button.setMinimumWidth(92)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.auto_checkbox)
        row.addWidget(self.button)
        row.addStretch(1)
        self.set_value(value, fallback=fallback)
        self.auto_checkbox.toggled.connect(self._auto_changed)
        self.button.colorChanged.connect(lambda _color: self.changed.emit())

    def value(self) -> str:
        return "auto" if self.auto_checkbox.isChecked() else self.button.color()

    def set_value(self, value: object, *, fallback: str | None = None) -> None:
        raw = str(value).strip() if value is not None else "auto"
        selected = QColor(raw)
        is_auto = raw.casefold() in {"", "auto", "默认"} or not selected.isValid()
        if fallback is not None:
            self.button.set_color(fallback)
        if not is_auto:
            self.button.set_color(selected.name())
        self.auto_checkbox.setChecked(is_auto)
        self.button.setEnabled(not is_auto)

    def _auto_changed(self, checked: bool) -> None:
        self.button.setEnabled(not checked)
        self.changed.emit()


def _normalize_display_groups(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        return {}
    groups: dict[str, dict[str, object]] = {}
    for raw_name, raw_group in value.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_group, Mapping):
            continue
        raw_ids = raw_group.get("element_ids", raw_group.get("elements", ()))
        try:
            ids = tuple(sorted({int(item) for item in raw_ids if int(item) > 0}))
        except (TypeError, ValueError):
            continue
        if not ids:
            continue
        groups[name] = {
            "element_ids": ids,
            "exclude": bool(raw_group.get("exclude", False)),
        }
    return groups


def _normalize_view_cut_ranges(value: object) -> dict[str, tuple[float, float]]:
    raw = value if isinstance(value, Mapping) else {}
    ranges: dict[str, tuple[float, float]] = {}
    for axis in VIEW_CUT_AXES:
        candidate = raw.get(axis)
        try:
            minimum, maximum = (float(item) for item in candidate)
        except (TypeError, ValueError, OverflowError):
            minimum, maximum = -1.0, 1.0
        if not isfinite(minimum) or not isfinite(maximum):
            minimum, maximum = -1.0, 1.0
        if minimum > maximum:
            minimum, maximum = maximum, minimum
        if minimum == maximum:
            padding = max(abs(minimum) * 0.01, 1.0)
            minimum -= padding
            maximum += padding
        ranges[axis] = (minimum, maximum)
    return ranges


class _ViewCutPositionControl(QWidget):
    """A compact Abaqus-style slider with a precise numeric readout."""

    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._slider_steps = 1000

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, self._slider_steps)
        self.slider.setTracking(True)
        self.value_edit = QDoubleSpinBox(self)
        self.value_edit.setRange(self._minimum, self._maximum)
        self.value_edit.setDecimals(8)
        self.value_edit.setSingleStep(
            max((self._maximum - self._minimum) / self._slider_steps, 1.0e-8)
        )
        self.value_edit.setFixedWidth(142)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_edit)

        self.slider.valueChanged.connect(self._slider_changed)
        self.value_edit.valueChanged.connect(self._value_changed)

    def set_value(self, value: float) -> None:
        value = max(self._minimum, min(self._maximum, float(value)))
        with QSignalBlocker(self.slider), QSignalBlocker(self.value_edit):
            self.slider.setValue(self._slider_position(value))
            self.value_edit.setValue(value)

    def value(self) -> float:
        return float(self.value_edit.value())

    def _slider_position(self, value: float) -> int:
        span = self._maximum - self._minimum
        if span <= 0.0:
            return 0
        ratio = (value - self._minimum) / span
        return int(round(max(0.0, min(1.0, ratio)) * self._slider_steps))

    def _slider_changed(self, position: int) -> None:
        span = self._maximum - self._minimum
        value = self._minimum + span * position / self._slider_steps
        with QSignalBlocker(self.value_edit):
            self.value_edit.setValue(value)
        self.valueChanged.emit(float(value))

    def _value_changed(self, value: float) -> None:
        with QSignalBlocker(self.slider):
            self.slider.setValue(self._slider_position(float(value)))
        self.valueChanged.emit(float(value))


class DisplayGroupViewCutDialog(QDialog):
    """管理结果显示组和轴向视图切割。"""

    previewRequested = Signal(object)
    applyRequested = Signal(object)

    def __init__(
        self,
        options: Mapping[str, object] | None = None,
        *,
        initial_page: str = "display_group",
        parent=None,
    ) -> None:
        super().__init__(parent)
        options = options if isinstance(options, Mapping) else {}
        self.setWindowTitle("显示组与视图切割")
        self.resize(720, 440)
        self.setMinimumSize(620, 360)
        self._display_groups = _normalize_display_groups(
            options.get("display_groups", {})
        )
        active_group = options.get("active_display_group")
        active_name = str(active_group).strip() if active_group is not None else ""
        self._active_display_group = (
            active_name if active_name in self._display_groups else None
        )
        self._view_cut = normalize_view_cut_settings(options.get("view_cut", {}))
        self._view_cut_ranges = _normalize_view_cut_ranges(
            options.get("view_cut_bounds", {})
        )
        self._selected_display_group: str | None = None
        self._creating_display_group = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("displayGroupViewCutTabs")
        self.tabs.addTab(self._build_display_group_page(), "显示组")
        self.tabs.addTab(self._build_view_cut_page(), "视图切割")
        self.tabs.setCurrentIndex(1 if initial_page == "view_cut" else 0)
        layout.addWidget(self.tabs, 1)

        self.button_box = _dialog_buttons(self)
        self.button_box.button(
            QDialogButtonBox.StandardButton.Apply
        ).clicked.connect(self.apply)
        self.button_box.accepted.connect(self.accept_with_apply)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _build_display_group_page(self) -> QWidget:
        page = QWidget(self.tabs)
        page.setObjectName("displayGroupManagerPage")
        layout = QHBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(QLabel("显示组"))
        self.display_group_list = QListWidget(page)
        self.display_group_list.setObjectName("displayGroupList")
        self.display_group_list.setFixedWidth(150)
        self.display_group_list.currentRowChanged.connect(
            self._load_display_group
        )
        left.addWidget(self.display_group_list, 1)
        layout.addLayout(left)

        right = QVBoxLayout()
        right.setSpacing(8)
        group = QGroupBox("显示组属性", page)
        form = QFormLayout(group)
        configure_form_layout(form)
        self.display_group_name_edit = QLineEdit(group)
        self.display_group_name_edit.setPlaceholderText("显示组名称")
        form.addRow("名称", self.display_group_name_edit)
        self.display_group_type = QLabel("单元", group)
        form.addRow("对象类型", self.display_group_type)
        self.display_group_ids_edit = QLineEdit(group)
        self.display_group_ids_edit.setPlaceholderText(
            "单元编号，例如 1, 2, 3；支持范围 1-5"
        )
        form.addRow("单元", self.display_group_ids_edit)
        self.display_group_exclude = QCheckBox("排除这些单元", group)
        form.addRow("方式", self.display_group_exclude)
        self.display_group_status = QLabel(
            "显示组只影响当前视口，不修改模型。",
            group,
        )
        self.display_group_status.setWordWrap(True)
        form.addRow("说明", self.display_group_status)
        right.addWidget(group)

        commands = QHBoxLayout()
        commands.setSpacing(6)
        self.display_group_new_button = QPushButton("新建", page)
        self.display_group_save_button = QPushButton("保存", page)
        self.display_group_copy_button = QPushButton("复制", page)
        self.display_group_rename_button = QPushButton("重命名", page)
        self.display_group_delete_button = QPushButton("删除", page)
        for button in (
            self.display_group_new_button,
            self.display_group_save_button,
            self.display_group_copy_button,
            self.display_group_rename_button,
            self.display_group_delete_button,
        ):
            commands.addWidget(button)
        commands.addStretch(1)
        right.addLayout(commands)

        apply_row = QHBoxLayout()
        self.display_group_apply_button = QPushButton("应用当前", page)
        self.display_group_all_button = QPushButton("显示全部", page)
        apply_row.addWidget(self.display_group_apply_button)
        apply_row.addWidget(self.display_group_all_button)
        apply_row.addStretch(1)
        right.addLayout(apply_row)
        right.addStretch(1)
        layout.addLayout(right, 1)

        self.display_group_new_button.clicked.connect(
            self._start_new_display_group
        )
        self.display_group_save_button.clicked.connect(self._save_display_group)
        self.display_group_copy_button.clicked.connect(self._copy_display_group)
        self.display_group_rename_button.clicked.connect(
            self._rename_display_group
        )
        self.display_group_delete_button.clicked.connect(
            self._delete_display_group
        )
        self.display_group_apply_button.clicked.connect(
            self._apply_selected_display_group
        )
        self.display_group_all_button.clicked.connect(self._show_all_display_group)
        self._populate_display_group_list(self._active_display_group)
        return page

    def _build_view_cut_page(self) -> QWidget:
        page = QWidget(self.tabs)
        page.setObjectName("viewCutManagerPage")
        layout = QHBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        list_layout = QVBoxLayout()
        list_layout.setSpacing(6)
        list_layout.addWidget(QLabel("切割面", page))
        self.view_cut_list = QListWidget(page)
        self.view_cut_list.setObjectName("viewCutList")
        self.view_cut_list.setFixedWidth(142)
        planes = self._view_cut["planes"]
        assert isinstance(planes, Mapping)
        for label, axis in (
            ("X-平面", "x"),
            ("Y-平面", "y"),
            ("Z-平面", "z"),
        ):
            item = QListWidgetItem(label, self.view_cut_list)
            item.setData(Qt.ItemDataRole.UserRole, axis)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if bool(planes[axis].get("enabled", False))
                else Qt.CheckState.Unchecked
            )
        list_layout.addWidget(self.view_cut_list, 1)
        layout.addLayout(list_layout)

        right = QVBoxLayout()
        right.setSpacing(8)
        group = QGroupBox("选定切割的运动", page)
        form = QFormLayout(group)
        configure_form_layout(form)

        self.view_cut_selected_label = QLabel(group)
        form.addRow("当前切割", self.view_cut_selected_label)
        self.view_cut_shape = QLabel("平面（轴向）", group)
        form.addRow("形状", self.view_cut_shape)
        self.view_cut_positions: dict[str, _ViewCutPositionControl] = {}
        for label, axis in (
            ("X 位置", "x"),
            ("Y 位置", "y"),
            ("Z 位置", "z"),
        ):
            control = _ViewCutPositionControl(
                *self._view_cut_ranges[axis],
                parent=group,
            )
            control.setObjectName(f"viewCut{axis.upper()}Position")
            control.set_value(float(planes[axis].get("offset", 0.0)))
            control.valueChanged.connect(
                lambda value, axis=axis: self._view_cut_position_changed(
                    axis,
                    value,
                )
            )
            self.view_cut_positions[axis] = control
            form.addRow(label, control)
        self.view_cut_invert = QCheckBox("保留法向反侧", group)
        form.addRow("保留侧", self.view_cut_invert)
        right.addWidget(group)

        command_row = QHBoxLayout()
        self.view_cut_reset_button = QPushButton("重置全部切割", page)
        self.view_cut_all_button = QPushButton("显示完整模型", page)
        command_row.addWidget(self.view_cut_reset_button)
        command_row.addWidget(self.view_cut_all_button)
        command_row.addStretch(1)
        right.addLayout(command_row)
        hint = QLabel(
            "拖动滑块可实时预览；数字框用于精确定位。"
            "X、Y、Z 切割面可独立启用。",
            page,
        )
        hint.setWordWrap(True)
        right.addWidget(hint)
        right.addStretch(1)
        layout.addLayout(right, 1)
        self.view_cut_list.currentRowChanged.connect(
            self._view_cut_selection_changed
        )
        self.view_cut_list.itemChanged.connect(self._view_cut_item_changed)
        self.view_cut_invert.toggled.connect(self._view_cut_invert_changed)
        self.view_cut_reset_button.clicked.connect(self._reset_view_cut)
        self.view_cut_all_button.clicked.connect(self._disable_view_cut)
        selected_row = next(
            (
                index
                for index, axis in enumerate(VIEW_CUT_AXES)
                if bool(planes[axis].get("enabled", False))
            ),
            0,
        )
        self.view_cut_list.setCurrentRow(selected_row)
        self._sync_view_cut_selection()
        return page

    @staticmethod
    def _next_name(prefix: str, existing: Mapping[str, object]) -> str:
        index = 1
        while f"{prefix}-{index}" in existing:
            index += 1
        return f"{prefix}-{index}"

    def _populate_display_group_list(self, selected: str | None) -> None:
        self.display_group_list.blockSignals(True)
        self.display_group_list.clear()
        self.display_group_list.addItem("全部模型")
        for name in self._display_groups:
            self.display_group_list.addItem(name)
        row = 0
        if selected is not None:
            selected_row = self.display_group_list.findItems(
                selected,
                Qt.MatchFlag.MatchExactly,
            )
            if selected_row:
                row = self.display_group_list.row(selected_row[0])
        self.display_group_list.setCurrentRow(row)
        self.display_group_list.blockSignals(False)
        self._load_display_group(row)

    def _load_display_group(self, row: int) -> None:
        item = self.display_group_list.item(row)
        name = item.text() if item is not None and row > 0 else None
        group = self._display_groups.get(name) if name is not None else None
        self._selected_display_group = name
        self._creating_display_group = False
        if group is None:
            self.display_group_name_edit.setText("全部模型")
            self.display_group_ids_edit.clear()
            self.display_group_exclude.setChecked(False)
        else:
            self.display_group_name_edit.setText(name)
            self.display_group_ids_edit.setText(
                ", ".join(str(item) for item in group["element_ids"])
            )
            self.display_group_exclude.setChecked(
                bool(group.get("exclude", False))
            )
        self._sync_display_group_controls()

    def _sync_display_group_controls(self) -> None:
        saved = self._selected_display_group in self._display_groups
        editable = saved or self._creating_display_group
        self.display_group_name_edit.setEnabled(editable)
        self.display_group_name_edit.setReadOnly(
            saved and not self._creating_display_group
        )
        self.display_group_ids_edit.setEnabled(editable)
        self.display_group_exclude.setEnabled(editable)
        self.display_group_save_button.setEnabled(editable)
        self.display_group_copy_button.setEnabled(saved)
        self.display_group_rename_button.setEnabled(saved)
        self.display_group_delete_button.setEnabled(saved)
        self.display_group_apply_button.setEnabled(saved)

    def _start_new_display_group(self) -> None:
        self._selected_display_group = None
        self._creating_display_group = True
        self.display_group_list.clearSelection()
        self.display_group_name_edit.setText(
            self._next_name("显示组", self._display_groups)
        )
        self.display_group_ids_edit.clear()
        self.display_group_exclude.setChecked(False)
        self.display_group_status.setText("请输入单元编号后保存显示组。")
        self._sync_display_group_controls()
        self.display_group_ids_edit.setFocus()

    @staticmethod
    def _parse_element_ids(text: str) -> tuple[int, ...]:
        tokens = [token for token in re.split(r"[,，;；\s]+", text) if token]
        values: set[int] = set()
        for token in tokens:
            if re.fullmatch(r"\d+", token):
                values.add(int(token))
                continue
            match = re.fullmatch(r"(\d+)\s*[-~～]\s*(\d+)", token)
            if match is None:
                raise ValueError
            start, end = (int(value) for value in match.groups())
            if start > end or end - start > 100000:
                raise ValueError
            values.update(range(start, end + 1))
        if not values or any(value <= 0 for value in values):
            raise ValueError
        return tuple(sorted(values))

    def _save_display_group(self) -> None:
        name = self.display_group_name_edit.text().strip()
        if not name or name == "全部模型":
            self.display_group_status.setText("请输入有效的显示组名称。")
            return
        try:
            element_ids = self._parse_element_ids(
                self.display_group_ids_edit.text().strip()
            )
        except ValueError:
            self.display_group_status.setText(
                "单元编号必须是正整数或编号范围。"
            )
            return
        if (
            not self._creating_display_group
            and self._selected_display_group is not None
            and name != self._selected_display_group
        ):
            self.display_group_status.setText("已有显示组请使用“重命名”。")
            return
        self._display_groups[name] = {
            "element_ids": element_ids,
            "exclude": self.display_group_exclude.isChecked(),
        }
        self._creating_display_group = False
        self._selected_display_group = name
        self._populate_display_group_list(name)
        self.display_group_status.setText(f"已保存显示组“{name}”。")

    def _copy_display_group(self) -> None:
        source = self._display_groups.get(self._selected_display_group)
        if source is None:
            return
        name = self._next_name("显示组", self._display_groups)
        self._display_groups[name] = dict(source)
        self._selected_display_group = name
        self._populate_display_group_list(name)
        self.display_group_status.setText(f"已复制为“{name}”。")

    def _rename_display_group(self) -> None:
        old_name = self._selected_display_group
        if old_name not in self._display_groups:
            return
        name, accepted = QInputDialog.getText(
            self,
            "重命名显示组",
            "名称：",
            text=old_name,
        )
        name = name.strip()
        if not accepted or not name or name == "全部模型":
            return
        if name != old_name and name in self._display_groups:
            self.display_group_status.setText("该显示组名称已经存在。")
            return
        self._display_groups[name] = self._display_groups.pop(old_name)
        if self._active_display_group == old_name:
            self._active_display_group = name
        self._selected_display_group = name
        self._populate_display_group_list(name)
        self.display_group_status.setText(f"已重命名为“{name}”。")

    def _delete_display_group(self) -> None:
        name = self._selected_display_group
        if name not in self._display_groups:
            return
        self._display_groups.pop(name, None)
        if self._active_display_group == name:
            self._active_display_group = None
        self._populate_display_group_list(None)
        self.display_group_status.setText(f"已删除显示组“{name}”。")

    def _apply_selected_display_group(self) -> None:
        if self._selected_display_group not in self._display_groups:
            return
        self._active_display_group = self._selected_display_group
        self.display_group_status.setText(
            f"已将“{self._active_display_group}”设为当前显示组。"
        )

    def _show_all_display_group(self) -> None:
        self._active_display_group = None
        self.display_group_list.setCurrentRow(0)
        self.display_group_status.setText("当前视口将显示全部模型。")

    def _selected_view_cut_axis(self) -> str:
        item = self.view_cut_list.currentItem()
        axis = item.data(Qt.ItemDataRole.UserRole) if item is not None else "x"
        return str(axis) if axis in VIEW_CUT_AXES else "x"

    def _view_cut_item(self, axis: str) -> QListWidgetItem:
        return self.view_cut_list.item(VIEW_CUT_AXES.index(axis))

    def _sync_view_cut_selection(self) -> None:
        axis = self._selected_view_cut_axis()
        planes = self._view_cut["planes"]
        assert isinstance(planes, Mapping)
        self.view_cut_selected_label.setText(f"{axis.upper()}-平面")
        with QSignalBlocker(self.view_cut_invert):
            self.view_cut_invert.setChecked(
                bool(planes[axis].get("invert", False))
            )

    def _view_cut_selection_changed(self, _row: int) -> None:
        self._sync_view_cut_selection()

    def _view_cut_item_changed(self, item: QListWidgetItem) -> None:
        axis = item.data(Qt.ItemDataRole.UserRole)
        if axis not in VIEW_CUT_AXES:
            return
        planes = self._view_cut["planes"]
        assert isinstance(planes, Mapping)
        planes[axis]["enabled"] = (
            item.checkState() == Qt.CheckState.Checked
        )
        self._emit_view_cut_preview()

    def _view_cut_position_changed(self, axis: str, value: float) -> None:
        planes = self._view_cut["planes"]
        assert isinstance(planes, Mapping)
        planes[axis]["offset"] = float(value)
        self._emit_view_cut_preview()

    def _view_cut_invert_changed(self, checked: bool) -> None:
        axis = self._selected_view_cut_axis()
        planes = self._view_cut["planes"]
        assert isinstance(planes, Mapping)
        planes[axis]["invert"] = bool(checked)
        self._emit_view_cut_preview()

    def _emit_view_cut_preview(self) -> None:
        self.previewRequested.emit(self.settings())

    def _reset_view_cut(self) -> None:
        self._view_cut = default_view_cut_settings()
        planes = self._view_cut["planes"]
        assert isinstance(planes, Mapping)
        with QSignalBlocker(self.view_cut_list):
            for axis in VIEW_CUT_AXES:
                self._view_cut_item(axis).setCheckState(
                    Qt.CheckState.Unchecked
                )
        for axis, control in self.view_cut_positions.items():
            control.set_value(float(planes[axis]["offset"]))
        self._sync_view_cut_selection()
        self._emit_view_cut_preview()

    def _disable_view_cut(self) -> None:
        planes = self._view_cut["planes"]
        assert isinstance(planes, Mapping)
        with QSignalBlocker(self.view_cut_list):
            for axis in VIEW_CUT_AXES:
                self._view_cut_item(axis).setCheckState(
                    Qt.CheckState.Unchecked
                )
                planes[axis]["enabled"] = False
        self._emit_view_cut_preview()

    def settings(self) -> dict[str, object]:
        return {
            "display_groups": {
                name: {
                    "element_ids": tuple(group["element_ids"]),
                    "exclude": bool(group.get("exclude", False)),
                }
                for name, group in self._display_groups.items()
            },
            "active_display_group": self._active_display_group,
            "view_cut": {
                "planes": {
                    axis: {
                        "enabled": self._view_cut_item(axis).checkState()
                        == Qt.CheckState.Checked,
                        "offset": float(
                            self.view_cut_positions[axis].value()
                        ),
                        "invert": bool(
                            self._view_cut["planes"][axis].get(
                                "invert",
                                False,
                            )
                        ),
                    }
                    for axis in VIEW_CUT_AXES
                }
            },
        }

    def apply(self) -> None:
        self.applyRequested.emit(self.settings())

    def accept_with_apply(self) -> None:
        self.apply()
        self.accept()


def _populate_contour_colormap_combo(
    combo: QComboBox,
    *,
    include_custom: bool = True,
) -> None:
    """Populate one contour colour selector from the shared palette registry."""

    for label, key in COLORMAP_CHOICES:
        if key == CUSTOM_COLORMAP and not include_custom:
            continue
        combo.addItem(label, key)


def _option_color_stops(options: Mapping[str, Any]) -> tuple[tuple[float, str], ...]:
    try:
        return normalize_color_stops(
            options.get("custom_color_stops", DEFAULT_CUSTOM_COLOR_STOPS)
        )
    except (TypeError, ValueError):
        return DEFAULT_CUSTOM_COLOR_STOPS


class _SignificantDigitsDoubleSpinBox(QDoubleSpinBox):
    """以固定有效数字显示数值，同时保留内部精度。"""

    def __init__(self, significant_digits: int, parent=None) -> None:
        super().__init__(parent)
        self._significant_digits = int(significant_digits)

    def textFromValue(self, value: float) -> str:
        if value == 0.0:
            return "0"
        decimal_places = (
            self._significant_digits
            - floor(log10(abs(value)))
            - 1
        )
        rounded = round(value, decimal_places)
        if decimal_places <= 0:
            return f"{rounded:.0f}"
        displayed_places = min(decimal_places, self.decimals())
        return f"{rounded:.{displayed_places}f}".rstrip("0").rstrip(".")


class DisplaySettingsDialog(QDialog):
    """控制结果视口中的边线、图例和注释显示。"""

    applyRequested = Signal(object)

    def __init__(self, options: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("显示设置")
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.outline_group = QGroupBox("边线显示", self)
        self.outline_group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        outline_layout = QHBoxLayout(self.outline_group)
        outline_layout.setContentsMargins(12, 10, 12, 10)
        outline_layout.setSpacing(12)
        self.edge_mode = QComboBox(self.outline_group)
        for label, key in (
            ("几何边", CONTOUR_EDGE_GEOMETRY),
            ("全部边", CONTOUR_EDGE_ALL),
            ("外部边", CONTOUR_EDGE_EXTERIOR),
            ("特征边", CONTOUR_EDGE_FEATURE),
            ("自由边", CONTOUR_EDGE_FREE),
            ("无边", CONTOUR_EDGE_NONE),
        ):
            self.edge_mode.addItem(label, key)
        selected_edge_mode = options.get("edge_mode")
        if selected_edge_mode is None:
            selected_edge_mode = (
                CONTOUR_EDGE_ALL
                if options.get("edges")
                else (
                    CONTOUR_EDGE_NONE
                    if "edges" in options
                    else CONTOUR_EDGE_GEOMETRY
                )
            )
        self.edge_mode.setCurrentIndex(
            max(0, self.edge_mode.findData(selected_edge_mode))
        )
        self.edge_mode.setFixedWidth(112)
        self.edge_style = QComboBox(self.outline_group)
        for label, key in (
            ("实线", "solid"),
            ("虚线", "dashed"),
            ("短划线", "short_dashed"),
            ("加粗线", "bold"),
        ):
            self.edge_style.addItem(label, key)
        self.edge_style.setCurrentIndex(
            max(0, self.edge_style.findData(options.get("edge_style", "solid")))
        )
        self.edge_style.setFixedWidth(112)
        self.edge_width = QDoubleSpinBox(self.outline_group)
        self.edge_width.setRange(0.1, 20.0)
        self.edge_width.setDecimals(1)
        self.edge_width.setSingleStep(0.5)
        self.edge_width.setValue(float(options.get("edge_width", 1.0)))
        self.edge_width.setFixedWidth(60)
        self.edge_width_unit = QLabel("pt", self.outline_group)
        outline_layout.addWidget(QLabel("线条", self.outline_group))
        outline_layout.addWidget(self.edge_mode)
        outline_layout.addSpacing(18)
        outline_layout.addWidget(QLabel("样式", self.outline_group))
        outline_layout.addWidget(self.edge_style)
        outline_layout.addSpacing(18)
        outline_layout.addWidget(QLabel("粗细", self.outline_group))
        outline_layout.addWidget(self.edge_width)
        outline_layout.addWidget(self.edge_width_unit)
        outline_layout.addStretch(1)
        layout.addWidget(self.outline_group)

        self.labels_group = QGroupBox("标签", self)
        self.labels_group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        labels_layout = QHBoxLayout(self.labels_group)
        labels_layout.setContentsMargins(12, 8, 12, 8)
        labels_layout.setSpacing(24)
        self.show_node_labels = QCheckBox("显示节点编号", self.labels_group)
        self.show_node_labels.setChecked(
            bool(options.get("show_node_labels", False))
        )
        self.show_element_labels = QCheckBox("显示单元编号", self.labels_group)
        self.show_element_labels.setChecked(
            bool(options.get("show_element_labels", False))
        )
        labels_layout.addWidget(self.show_node_labels)
        labels_layout.addWidget(self.show_element_labels)
        labels_layout.addStretch(1)
        layout.addWidget(self.labels_group)

        self.legend_group = QGroupBox("图例与注释", self)
        self.legend_group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        legend_layout = QGridLayout(self.legend_group)
        legend_layout.setContentsMargins(12, 10, 12, 10)
        legend_layout.setHorizontalSpacing(16)
        legend_layout.setVerticalSpacing(14)
        legend_layout.setColumnMinimumWidth(3, 110)
        legend_layout.setColumnStretch(6, 1)
        legend_layout.addWidget(QLabel("数值格式", self.legend_group), 0, 0)
        self.scientific_format = QRadioButton("科学计数", self.legend_group)
        self.engineering_format = QRadioButton("工程计数", self.legend_group)
        self.number_format_buttons = QButtonGroup(self.legend_group)
        self.number_format_buttons.addButton(self.scientific_format)
        self.number_format_buttons.addButton(self.engineering_format)
        if options.get("number_format", "scientific") == "engineering":
            self.engineering_format.setChecked(True)
        else:
            self.scientific_format.setChecked(True)
        self.number_format_host = QWidget(self.legend_group)
        number_format_layout = QHBoxLayout(self.number_format_host)
        number_format_layout.setContentsMargins(0, 0, 0, 0)
        number_format_layout.setSpacing(18)
        number_format_layout.addWidget(self.scientific_format)
        number_format_layout.addWidget(self.engineering_format)
        number_format_layout.addStretch(1)
        self.number_format_host.setFixedWidth(175)
        legend_layout.addWidget(self.number_format_host, 0, 1)
        legend_layout.addWidget(QLabel("小数位", self.legend_group), 0, 2)
        self.decimals = QSpinBox(self.legend_group)
        self.decimals.setRange(0, 12)
        self.decimals.setValue(int(options.get("decimals", 2)))
        self.decimals.setFixedWidth(60)
        legend_layout.addWidget(
            self.decimals,
            0,
            3,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        legend_layout.addWidget(QLabel("图例方向", self.legend_group), 1, 0)
        self.horizontal_orientation = QRadioButton("横向", self.legend_group)
        self.vertical_orientation = QRadioButton("纵向", self.legend_group)
        self.horizontal_orientation.setFixedWidth(
            self.scientific_format.sizeHint().width()
        )
        self.orientation_buttons = QButtonGroup(self.legend_group)
        self.orientation_buttons.addButton(self.horizontal_orientation)
        self.orientation_buttons.addButton(self.vertical_orientation)
        if options.get("orientation", "vertical") == "horizontal":
            self.horizontal_orientation.setChecked(True)
        else:
            self.vertical_orientation.setChecked(True)
        self.orientation_host = QWidget(self.legend_group)
        orientation_layout = QHBoxLayout(self.orientation_host)
        orientation_layout.setContentsMargins(0, 0, 0, 0)
        orientation_layout.setSpacing(18)
        orientation_layout.addWidget(self.horizontal_orientation)
        orientation_layout.addWidget(self.vertical_orientation)
        orientation_layout.addStretch(1)
        self.orientation_host.setFixedWidth(175)
        legend_layout.addWidget(self.orientation_host, 1, 1)
        legend_layout.addWidget(QLabel("字体", self.legend_group), 1, 2)
        self.legend_font = QComboBox(self.legend_group)
        for font in ("Arial", "Times New Roman", "Courier New"):
            self.legend_font.addItem(font, font)
        self.legend_font.setCurrentIndex(
            max(0, self.legend_font.findData(options.get("legend_font", "Arial")))
        )
        self.legend_font.setFixedWidth(110)
        legend_layout.addWidget(
            self.legend_font,
            1,
            3,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        legend_layout.addWidget(QLabel("大小", self.legend_group), 1, 4)
        self.legend_font_size = QSpinBox(self.legend_group)
        self.legend_font_size.setRange(6, 72)
        self.legend_font_size.setValue(int(options.get("legend_font_size", 14)))
        self.legend_font_size.setFixedWidth(60)
        self.legend_font_size_unit = QLabel("pt", self.legend_group)
        self.legend_font_size_host = QWidget(self.legend_group)
        font_size_layout = QHBoxLayout(self.legend_font_size_host)
        font_size_layout.setContentsMargins(0, 0, 0, 0)
        font_size_layout.setSpacing(6)
        font_size_layout.addWidget(self.legend_font_size)
        font_size_layout.addWidget(self.legend_font_size_unit)
        legend_layout.addWidget(self.legend_font_size_host, 1, 5)

        legend_layout.addWidget(QLabel("位置", self.legend_group), 2, 0)
        self.legend_position = QComboBox(self.legend_group)
        for label, key in (
            ("自动", "auto"),
            ("左侧", "left"),
            ("右侧", "right"),
            ("顶部", "top"),
            ("底部", "bottom"),
        ):
            self.legend_position.addItem(label, key)
        self.legend_position.setCurrentIndex(
            max(
                0,
                self.legend_position.findData(
                    options.get("legend_position", "auto")
                ),
            )
        )
        self.legend_position.setFixedWidth(82)
        self.legend_position.setToolTip(
            "自动会根据图例方向放在右侧或底部"
        )
        legend_layout.addWidget(self.legend_position, 2, 1)
        legend_layout.addWidget(QLabel("标题", self.legend_group), 2, 2)
        self.legend_title = QCheckBox("显示", self.legend_group)
        self.legend_title.setChecked(
            bool(options.get("legend_title", True))
        )
        self.legend_title.setToolTip("显示结果变量、分量和当前增量标题")
        legend_layout.addWidget(self.legend_title, 2, 3)

        legend_layout.addWidget(QLabel("刻度数", self.legend_group), 3, 0)
        self.legend_label_count = QComboBox(self.legend_group)
        for label, key in (
            ("自动", "auto"),
            ("3 个", "3"),
            ("5 个", "5"),
            ("7 个", "7"),
            ("9 个", "9"),
        ):
            self.legend_label_count.addItem(label, key)
        self.legend_label_count.setCurrentIndex(
            max(
                0,
                self.legend_label_count.findData(
                    str(options.get("legend_label_count", "auto"))
                ),
            )
        )
        self.legend_label_count.setFixedWidth(82)
        self.legend_label_count.setToolTip("控制图例上显示的数值刻度数量")
        legend_layout.addWidget(self.legend_label_count, 3, 1)

        self.legend = QCheckBox("显示图例", self)
        self.legend.setChecked(bool(options.get("legend", True)))
        self.show_ids = QCheckBox("显示极值标签", self)
        self.show_ids.setChecked(bool(options.get("show_ids", False)))
        self.show_ids.setToolTip("在最小值/最大值标注中显示节点、单元或结果区域编号")
        self.show_coordinate_system = QCheckBox("显示坐标系", self)
        self.show_coordinate_system.setChecked(
            bool(options.get("show_coordinate_system", True))
        )
        annotation_host = QWidget(self.legend_group)
        annotation_layout = QHBoxLayout(annotation_host)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        annotation_layout.setSpacing(24)
        for checkbox in (
            self.legend,
            self.show_ids,
            self.show_coordinate_system,
        ):
            checkbox.setParent(annotation_host)
            annotation_layout.addWidget(checkbox)
        annotation_layout.addStretch(1)
        legend_layout.addWidget(annotation_host, 4, 0, 1, 6)
        layout.addWidget(self.legend_group)
        layout.addStretch(1)

        self.button_box = _dialog_buttons(self)
        self.button_box.button(
            QDialogButtonBox.StandardButton.Apply
        ).clicked.connect(self.apply)
        self.button_box.accepted.connect(self.accept_with_apply)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def settings(self) -> dict[str, Any]:
        edge_mode = str(self.edge_mode.currentData())
        return {
            "edge_mode": edge_mode,
            "edge_style": str(self.edge_style.currentData()),
            "edge_width": float(self.edge_width.value()),
            "number_format": (
                "scientific"
                if self.scientific_format.isChecked()
                else "engineering"
            ),
            "decimals": int(self.decimals.value()),
            "orientation": (
                "horizontal"
                if self.horizontal_orientation.isChecked()
                else "vertical"
            ),
            "legend_font": str(self.legend_font.currentData()),
            "legend_font_size": int(self.legend_font_size.value()),
            "legend_position": str(self.legend_position.currentData()),
            "legend_label_count": str(self.legend_label_count.currentData()),
            "legend_title": self.legend_title.isChecked(),
            "legend": self.legend.isChecked(),
            "show_ids": self.show_ids.isChecked(),
            "show_coordinate_system": self.show_coordinate_system.isChecked(),
            "show_node_labels": self.show_node_labels.isChecked(),
            "show_element_labels": self.show_element_labels.isChecked(),
            "edges": edge_mode != CONTOUR_EDGE_NONE,
        }

    def apply(self) -> None:
        self.applyRequested.emit(self.settings())

    def accept_with_apply(self) -> None:
        self.apply()
        self.accept()


class ContourSettingsDialog(QDialog):
    """控制云图范围、色带、渲染方式和节点平均阈值。"""

    applyRequested = Signal(object)

    def __init__(self, options: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("云图设置")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)

        self.render_group = QGroupBox("渲染", self)
        render_layout = QVBoxLayout(self.render_group)
        render_controls_row = QHBoxLayout()
        render_controls_row.setSpacing(12)
        render_controls_row.addWidget(QLabel("模式", self.render_group))
        self.filled_mode = QRadioButton("填充", self.render_group)
        self.shaded_mode = QRadioButton("光影", self.render_group)
        self.render_mode_buttons = QButtonGroup(self.render_group)
        self.render_mode_buttons.addButton(self.filled_mode)
        self.render_mode_buttons.addButton(self.shaded_mode)
        if options.get("render_mode", CONTOUR_RENDER_SHADED) == CONTOUR_RENDER_FILLED:
            self.filled_mode.setChecked(True)
        else:
            self.shaded_mode.setChecked(True)
        self.render_mode_host = QWidget(self.render_group)
        render_mode_layout = QHBoxLayout(self.render_mode_host)
        render_mode_layout.setContentsMargins(
            _ThinHorizontalSlider._margin,
            0,
            0,
            0,
        )
        render_mode_layout.setSpacing(18)
        render_mode_layout.addWidget(self.filled_mode)
        render_mode_layout.addWidget(self.shaded_mode)
        render_mode_layout.addStretch(1)
        self.render_mode_host.setFixedWidth(150)
        render_controls_row.addWidget(self.render_mode_host)
        render_controls_row.addSpacing(20)
        render_controls_row.addWidget(QLabel("样式", self.render_group))
        self.style = QComboBox(self.render_group)
        self.style.addItem("分段", "segmented")
        self.style.addItem("连续", "continuous")
        self.style.setCurrentIndex(
            max(0, self.style.findData(options.get("style", "segmented")))
        )
        self.style.setFixedWidth(90)
        render_controls_row.addWidget(self.style)
        render_controls_row.addSpacing(20)
        render_controls_row.addWidget(QLabel("色带", self.render_group))
        self.colormap = QComboBox(self.render_group)
        _populate_contour_colormap_combo(self.colormap, include_custom=False)
        selected_colormap = options.get("colormap", ABAQUS_RAINBOW)
        if selected_colormap == "jet":
            selected_colormap = ABAQUS_RAINBOW
        self.colormap.setCurrentIndex(
            max(0, self.colormap.findData(selected_colormap))
        )
        self.colormap.setFixedWidth(120)
        render_controls_row.addWidget(self.colormap)
        self.colormap_reverse = QCheckBox("反向", self.render_group)
        self.colormap_reverse.setChecked(
            bool(options.get("colormap_reverse", False))
        )
        render_controls_row.addWidget(self.colormap_reverse)
        render_controls_row.addStretch(1)
        render_layout.addLayout(render_controls_row)

        self.levels = QSpinBox(self.render_group)
        self.levels.setRange(4, 48)
        self.levels.setValue(int(options.get("levels", 12)))
        self.levels_slider = _ThinHorizontalSlider(self.render_group)
        self.levels_slider.setObjectName("contourLevelsSlider")
        self.levels_slider.setRange(4, 48)
        self.levels_slider.setSingleStep(1)
        self.levels_slider.setPageStep(4)
        self.levels_slider.setValue(self.levels.value())
        self.levels_slider.valueChanged.connect(self.levels.setValue)
        self.levels.valueChanged.connect(self.levels_slider.setValue)
        self.levels_row = QWidget(self.render_group)
        levels_layout = QHBoxLayout(self.levels_row)
        levels_layout.setContentsMargins(0, 0, 0, 0)
        levels_layout.setSpacing(8)
        levels_layout.addWidget(self.levels_slider, 1)
        levels_layout.addWidget(self.levels)
        self.style.currentIndexChanged.connect(
            lambda: self.levels_row.setEnabled(
                self.style.currentData() == "segmented"
            )
        )
        self.levels_row.setEnabled(self.style.currentData() == "segmented")
        self.averaging_threshold = QDoubleSpinBox(self.render_group)
        self.averaging_threshold.setRange(0.0, 100.0)
        self.averaging_threshold.setDecimals(0)
        self.averaging_threshold.setSingleStep(1.0)
        self.averaging_threshold.setSuffix(" %")
        self.averaging_threshold.setValue(float(options.get("averaging_threshold", 75.0)))
        self.averaging_threshold.setToolTip(
            "仅用于节点平均应力；超过阈值的当前分量按单元侧分开显示"
        )
        self.averaging_threshold_slider = _ThinHorizontalSlider(
            self.render_group
        )
        self.averaging_threshold_slider.setObjectName(
            "contourThresholdSlider"
        )
        self.averaging_threshold_slider.setRange(0, 100)
        self.averaging_threshold_slider.setSingleStep(1)
        self.averaging_threshold_slider.setPageStep(5)
        self.averaging_threshold_slider.setValue(
            round(self.averaging_threshold.value())
        )
        self.averaging_threshold_slider.valueChanged.connect(
            self.averaging_threshold.setValue
        )
        self.averaging_threshold.valueChanged.connect(
            lambda value: self.averaging_threshold_slider.setValue(
                round(value)
            )
        )
        self.averaging_threshold_row = QWidget(self.render_group)
        threshold_layout = QHBoxLayout(self.averaging_threshold_row)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(8)
        threshold_layout.addWidget(self.averaging_threshold_slider, 1)
        threshold_layout.addWidget(self.averaging_threshold)
        value_box_width = 60
        value_box_height = max(
            self.levels.sizeHint().height(),
            self.averaging_threshold.sizeHint().height(),
        )
        for value_box in (self.levels, self.averaging_threshold):
            value_box.setFixedSize(value_box_width, value_box_height)
            value_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.form = QFormLayout()
        configure_form_layout(self.form)
        self.form.addRow("级数", self.levels_row)
        self.form.addRow("阈值", self.averaging_threshold_row)
        render_layout.addLayout(self.form)
        layout.addWidget(self.render_group)

        self.range_group = QGroupBox("范围", self)
        range_layout = QVBoxLayout(self.range_group)
        self.minimum = _SignificantDigitsDoubleSpinBox(5, self.range_group)
        self.maximum = _SignificantDigitsDoubleSpinBox(5, self.range_group)
        for spin in (self.minimum, self.maximum):
            spin.setRange(-1.0e30, 1.0e30)
            spin.setDecimals(12)
            spin.setFixedWidth(110)
        range_mode = str(options.get("range_mode", "")).strip().casefold()
        if range_mode not in {"per_frame", "global_step", "manual"}:
            range_mode = "manual" if options.get("manual") else "per_frame"
        self._range_mode = range_mode
        if range_mode == "manual":
            minimum = float(options.get("minimum", 0.0))
            maximum = float(options.get("maximum", 1.0))
        elif range_mode == "global_step":
            minimum = float(
                options.get(
                    "global_minimum",
                    options.get("automatic_minimum", options.get("minimum", 0.0)),
                )
            )
            maximum = float(
                options.get(
                    "global_maximum",
                    options.get("automatic_maximum", options.get("maximum", 1.0)),
                )
            )
        else:
            minimum = float(
                options.get("automatic_minimum", options.get("minimum", 0.0))
            )
            maximum = float(
                options.get("automatic_maximum", options.get("maximum", 1.0))
            )
        self.minimum.setValue(minimum)
        self.maximum.setValue(maximum)
        self.show_minimum = QCheckBox("显示", self.range_group)
        self.show_minimum.setChecked(
            bool(options.get("show_minimum", False))
        )
        self.show_maximum = QCheckBox("显示", self.range_group)
        self.show_maximum.setChecked(
            bool(options.get("show_maximum", False))
        )
        value_row = QHBoxLayout()
        value_row.setSpacing(8)
        value_row.addWidget(QLabel("最小值", self.range_group))
        value_row.addWidget(self.minimum)
        value_row.addWidget(self.show_minimum)
        value_row.addSpacing(20)
        value_row.addWidget(QLabel("最大值", self.range_group))
        value_row.addWidget(self.maximum)
        value_row.addWidget(self.show_maximum)
        value_row.addStretch(1)
        range_layout.addLayout(value_row)

        self.auto_range = QRadioButton("每帧自动", self.range_group)
        self.global_range = QRadioButton("全步固定", self.range_group)
        self.manual_range = QRadioButton("手动", self.range_group)
        selected_range_button = {
            "per_frame": self.auto_range,
            "global_step": self.global_range,
            "manual": self.manual_range,
        }[range_mode]
        selected_range_button.setChecked(True)
        self.range_buttons = QButtonGroup(self.range_group)
        self.range_buttons.addButton(self.auto_range)
        self.range_buttons.addButton(self.global_range)
        self.range_buttons.addButton(self.manual_range)
        controls_row = QHBoxLayout()
        controls_row.setSpacing(18)
        controls_row.addWidget(self.auto_range)
        controls_row.addWidget(self.global_range)
        controls_row.addWidget(self.manual_range)
        controls_row.addStretch(1)
        range_layout.addLayout(controls_row)
        self.auto_range.toggled.connect(self._sync_range_mode)
        self.global_range.toggled.connect(self._sync_range_mode)
        self.manual_range.toggled.connect(self._sync_range_mode)
        self._sync_range_mode()
        layout.addWidget(self.range_group)

        buttons = _dialog_buttons(self)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self.accept_with_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> dict[str, Any]:
        if self.manual_range.isChecked():
            range_mode = "manual"
        elif self.global_range.isChecked():
            range_mode = "global_step"
        else:
            range_mode = "per_frame"
        return {
            "manual": range_mode == "manual",
            "range_mode": range_mode,
            "minimum": float(self.minimum.value()),
            "maximum": float(self.maximum.value()),
            "colormap": str(self.colormap.currentData()),
            "colormap_reverse": self.colormap_reverse.isChecked(),
            "style": str(self.style.currentData()),
            "render_mode": (
                CONTOUR_RENDER_FILLED
                if self.filled_mode.isChecked()
                else CONTOUR_RENDER_SHADED
            ),
            "levels": int(self.levels.value()),
            "show_minimum": self.show_minimum.isChecked(),
            "show_maximum": self.show_maximum.isChecked(),
            "averaging_threshold": float(self.averaging_threshold.value()),
        }

    def _sync_range_mode(self) -> None:
        manual = self.manual_range.isChecked()
        self.minimum.setEnabled(manual)
        self.maximum.setEnabled(manual)

    def apply(self) -> None:
        if self.manual_range.isChecked() and self.minimum.value() >= self.maximum.value():
            QMessageBox.warning(self, "云图设置", "手动范围的最小值必须小于最大值。")
            return
        self.applyRequested.emit(self.settings())

    def accept_with_apply(self) -> None:
        if self.manual_range.isChecked() and self.minimum.value() >= self.maximum.value():
            self.apply()
            return
        self.apply()
        self.accept()


class VisualizationOptionsDialog(QDialog):
    """统一的 Abaqus 风格“显示与云图”选项窗口。

    这里的分类只组织现有绘图状态，不创建新的结果功能。每个分类页
    都由实际可应用到视口的控件组成；主窗口的公共绘图选项和云图选项
    只是打开这个窗口时选择不同的当前页。
    """

    applyRequested = Signal(object)

    _categories = (
        "通用显示",
        "变形显示",
        "云图显示",
        "实体显示",
        "注释",
    )

    def __init__(
        self,
        options: dict[str, Any],
        *,
        initial_category: str = "通用显示",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("显示与云图")
        self.resize(760, 620)
        self.setMinimumSize(680, 320)
        self._options = dict(options)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        content = QHBoxLayout()
        content.setSpacing(8)

        self.category_list = QListWidget(self)
        self.category_list.setObjectName("visualizationCategoryList")
        self.category_list.setFixedWidth(116)
        self.category_list.setSpacing(2)
        self.category_list.addItems(self._categories)
        self.category_list.setCurrentRow(
            max(0, self._categories.index(initial_category))
            if initial_category in self._categories
            else 0
        )
        content.addWidget(self.category_list)

        self.pages = QStackedWidget(self)
        self.pages.setObjectName("visualizationOptionsPages")
        self.pages.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.common_page = self._build_common_page()
        self.deformation_page = self._build_deformation_page()
        self.contour_page = self._build_contour_page()
        self.entity_page = self._build_entity_page()
        self.annotation_page = self._build_annotation_page()
        for page in (
            self.common_page,
            self.deformation_page,
            self.contour_page,
            self.entity_page,
            self.annotation_page,
        ):
            self.pages.addWidget(page)
        self.category_list.currentRowChanged.connect(self._category_changed)
        self.pages.setCurrentIndex(self.category_list.currentRow())
        content.addWidget(self.pages, 1)
        root.addLayout(content, 1)

        self.button_box = _dialog_buttons(self)
        self.button_box.button(
            QDialogButtonBox.StandardButton.Apply
        ).clicked.connect(self.apply)
        self.button_box.accepted.connect(self.accept_with_apply)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

    def _category_changed(self, index: int) -> None:
        self.pages.setCurrentIndex(index)

    def _page(self, object_name: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget(self.pages)
        page.setObjectName(object_name)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        return page, layout

    def _build_common_page(self) -> QWidget:
        page, layout = self._page("commonDisplayPage")
        group = QGroupBox("绘图状态", page)
        form = QFormLayout(group)
        configure_form_layout(form)
        current_render_mode = self._options.get(
            "render_mode",
            CONTOUR_RENDER_SHADED,
        )
        current_edge_mode = self._options.get("edge_mode")
        if current_edge_mode is None:
            current_edge_mode = (
                CONTOUR_EDGE_ALL
                if self._options.get("edges", True)
                else CONTOUR_EDGE_NONE
            )
        if current_render_mode == CONTOUR_RENDER_HIDDEN_LINE:
            display_style = "hidden_line"
        elif current_render_mode == CONTOUR_RENDER_WIREFRAME:
            display_style = "wireframe"
        elif current_edge_mode != CONTOUR_EDGE_NONE:
            display_style = "shaded_edges"
        else:
            display_style = "shaded"
        self.display_style = QComboBox(group)
        for label, key in (
            ("着色", "shaded"),
            ("着色 + 网格线", "shaded_edges"),
            ("线框", "wireframe"),
            ("消隐线", "hidden_line"),
        ):
            self.display_style.addItem(label, key)
        self.display_style.setCurrentIndex(
            max(0, self.display_style.findData(display_style))
        )
        self._display_style_changed = False
        self.display_style.currentIndexChanged.connect(
            lambda _index: setattr(self, "_display_style_changed", True)
        )
        form.addRow("显示方式", self.display_style)
        self.shape_mode = QComboBox(group)
        self.shape_mode.addItem("未变形", "undeformed")
        self.shape_mode.addItem("变形", "deformed")
        self.shape_mode.setCurrentIndex(
            max(
                0,
                self.shape_mode.findData(
                    self._options.get("shape_mode", "undeformed")
                ),
            )
        )
        form.addRow("模型形状", self.shape_mode)
        self.contour_checkbox = QCheckBox("显示云图", group)
        self.contour_checkbox.setChecked(
            bool(self._options.get("contour_enabled", True))
        )
        form.addRow("结果", self.contour_checkbox)

        appearance_group = QGroupBox("绘图外观", page)
        appearance_form = QFormLayout(appearance_group)
        configure_form_layout(appearance_form)
        palette = self._options.get("palette", {})
        if not isinstance(palette, Mapping):
            palette = {}
        self.face_color_control = _DisplayColorControl(
            self._options.get("face_color", "auto"),
            fallback=str(palette.get("mesh", "#BFDCEB")),
            parent=appearance_group,
        )
        self.edge_color_control = _DisplayColorControl(
            self._options.get("edge_color", "auto"),
            fallback=str(palette.get("element", "#3F6F8C")),
            parent=appearance_group,
        )
        self.model_opacity = QDoubleSpinBox(appearance_group)
        self.model_opacity.setRange(0.0, 100.0)
        self.model_opacity.setDecimals(1)
        self.model_opacity.setSingleStep(5.0)
        self.model_opacity.setSuffix(" %")
        self.model_opacity.setValue(
            float(self._options.get("model_opacity", 1.0)) * 100.0
        )
        appearance_form.addRow("面颜色", self.face_color_control)
        appearance_form.addRow("边线颜色", self.edge_color_control)
        appearance_form.addRow("模型透明度", self.model_opacity)
        layout.addWidget(appearance_group)

        hint = QLabel(
            "这里控制当前结果的基本绘图状态；显示组和视图切割在“显示组”中设置。",
            group,
        )
        hint.setWordWrap(True)
        hint.setObjectName("visualizationOptionsHint")
        form.addRow("说明", hint)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _build_deformation_page(self) -> QWidget:
        page, layout = self._page("deformationDisplayPage")
        group = QGroupBox("变形显示", page)
        form = QFormLayout(group)
        configure_form_layout(form)
        self.scale_mode = QComboBox(group)
        for label, key in (
            ("自动", "auto"),
            ("真实比例", "real"),
            ("自定义", "custom"),
        ):
            self.scale_mode.addItem(label, key)
        self.scale_mode.setCurrentIndex(
            max(
                0,
                self.scale_mode.findData(
                    self._options.get("scale_mode", "auto")
                ),
            )
        )
        form.addRow("变形比例", self.scale_mode)

        self.scale_value = QDoubleSpinBox(group)
        self.scale_value.setRange(0.0, 1.0e12)
        self.scale_value.setDecimals(12)
        self.scale_value.setValue(float(self._options.get("scale_value", 1.0)))
        self.scale_value.setToolTip("仅在选择自定义时使用；不会改变求解结果")
        form.addRow("自定义比例", self.scale_value)

        maximum = self._options.get("maximum_displacement")
        try:
            maximum_text = f"{float(maximum):.6e}"
        except (TypeError, ValueError):
            maximum_text = "—"
        self.maximum_displacement_label = QLabel(maximum_text, group)
        self.maximum_displacement_label.setObjectName(
            "maximumDisplacementValue"
        )
        form.addRow("真实最大位移", self.maximum_displacement_label)

        self.overlay_checkbox = QCheckBox("叠加未变形形状", group)
        self.overlay_checkbox.setChecked(
            bool(self._options.get("overlay_undeformed", False))
        )
        form.addRow("对比", self.overlay_checkbox)

        style_group = QGroupBox("形状样式", page)
        style_form = QFormLayout(style_group)
        configure_form_layout(style_form)
        palette = self._options.get("palette", {})
        if not isinstance(palette, Mapping):
            palette = {}
        self.deformed_color_control = _DisplayColorControl(
            self._options.get("deformed_color", "auto"),
            fallback=str(palette.get("result", "#b9c6d2")),
            parent=style_group,
        )
        self.undeformed_color_control = _DisplayColorControl(
            self._options.get("undeformed_color", "auto"),
            fallback=str(palette.get("overlay", "#7f8c8d")),
            parent=style_group,
        )
        self.deformed_line_style = QComboBox(style_group)
        self.undeformed_line_style = QComboBox(style_group)
        for combo in (self.deformed_line_style, self.undeformed_line_style):
            for label, key in (
                ("实线", "solid"),
                ("虚线", "dashed"),
                ("短划线", "short_dashed"),
                ("加粗线", "bold"),
            ):
                combo.addItem(label, key)
        self.deformed_line_style.setCurrentIndex(
            max(
                0,
                self.deformed_line_style.findData(
                    self._options.get("deformed_line_style", "solid")
                ),
            )
        )
        self.undeformed_line_style.setCurrentIndex(
            max(
                0,
                self.undeformed_line_style.findData(
                    self._options.get("undeformed_line_style", "solid")
                ),
            )
        )
        self.deformed_opacity = QDoubleSpinBox(style_group)
        self.undeformed_opacity = QDoubleSpinBox(style_group)
        for spin, key, default in (
            (self.deformed_opacity, "deformed_opacity", 1.0),
            (self.undeformed_opacity, "undeformed_opacity", 0.65),
        ):
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(5.0)
            spin.setSuffix(" %")
            spin.setValue(float(self._options.get(key, default)) * 100.0)
        style_form.addRow("变形面颜色", self.deformed_color_control)
        style_form.addRow("变形边线", self.deformed_line_style)
        style_form.addRow("变形透明度", self.deformed_opacity)
        style_form.addRow("未变形颜色", self.undeformed_color_control)
        style_form.addRow("未变形边线", self.undeformed_line_style)
        style_form.addRow("未变形透明度", self.undeformed_opacity)
        layout.addWidget(style_group)
        note = QLabel(
            "变形比例只影响视口中的显示坐标，不改变位移、应力或求解结果。",
            group,
        )
        note.setWordWrap(True)
        form.addRow("说明", note)
        layout.addWidget(group)
        layout.addStretch(1)
        self.scale_mode.currentIndexChanged.connect(self._sync_scale_mode)
        self._sync_scale_mode()
        return page

    def _build_contour_page(self) -> QWidget:
        page, layout = self._page("contourDisplayPage")
        render_group = QGroupBox("色带与渲染", page)
        form = QFormLayout(render_group)
        configure_form_layout(form)

        self.render_mode_buttons = QButtonGroup(render_group)
        self.filled_mode = QRadioButton("填充", render_group)
        self.shaded_mode = QRadioButton("光影", render_group)
        self.render_mode_buttons.addButton(self.filled_mode)
        self.render_mode_buttons.addButton(self.shaded_mode)
        render_mode = self._options.get("render_mode", CONTOUR_RENDER_SHADED)
        if render_mode == CONTOUR_RENDER_FILLED:
            self.filled_mode.setChecked(True)
        else:
            self.shaded_mode.setChecked(True)
        self._render_mode_changed = False
        self.filled_mode.toggled.connect(self._mark_render_mode_changed)
        self.shaded_mode.toggled.connect(self._mark_render_mode_changed)
        render_host = QWidget(render_group)
        render_layout = QHBoxLayout(render_host)
        render_layout.setContentsMargins(0, 0, 0, 0)
        render_layout.setSpacing(18)
        render_layout.addWidget(self.filled_mode)
        render_layout.addWidget(self.shaded_mode)
        render_layout.addStretch(1)
        form.addRow("渲染模式", render_host)

        self.style = QComboBox(render_group)
        self.style.addItem("分段色带", "segmented")
        self.style.addItem("连续色带", "continuous")
        self.style.setCurrentIndex(
            max(0, self.style.findData(self._options.get("style", "segmented")))
        )
        form.addRow("色带方式", self.style)

        self.colormap = QComboBox(render_group)
        _populate_contour_colormap_combo(self.colormap)
        selected_colormap = self._options.get("colormap", ABAQUS_RAINBOW)
        if selected_colormap == "jet":
            selected_colormap = ABAQUS_RAINBOW
        self.colormap.setCurrentIndex(
            max(0, self.colormap.findData(selected_colormap))
        )
        self.colormap.setMinimumWidth(150)
        self.colormap_reverse = QCheckBox("反向", render_group)
        self.colormap_reverse.setChecked(
            bool(self._options.get("colormap_reverse", False))
        )
        colormap_host = QWidget(render_group)
        colormap_layout = QHBoxLayout(colormap_host)
        colormap_layout.setContentsMargins(0, 0, 0, 0)
        colormap_layout.setSpacing(12)
        colormap_layout.addWidget(self.colormap)
        colormap_layout.addWidget(self.colormap_reverse)
        colormap_layout.addStretch(1)
        form.addRow("色谱", colormap_host)

        self.colormap_preview = _ContourColorRampPreview(
            preview_contour_colors(
                str(self.colormap.currentData() or ABAQUS_RAINBOW),
                reverse=self.colormap_reverse.isChecked(),
                custom_color_stops=_option_color_stops(self._options),
            ),
            render_group,
        )
        form.addRow("预览", self.colormap_preview)

        self.levels = QSpinBox(render_group)
        self.levels.setRange(4, 48)
        self.levels.setValue(int(self._options.get("levels", 12)))
        form.addRow("级数", self.levels)

        self.averaging_threshold = QDoubleSpinBox(render_group)
        self.averaging_threshold.setRange(0.0, 100.0)
        self.averaging_threshold.setDecimals(0)
        self.averaging_threshold.setSuffix(" %")
        self.averaging_threshold.setValue(
            float(self._options.get("averaging_threshold", 75.0))
        )
        self.averaging_threshold.setToolTip(
            "仅用于节点平均应力；超过阈值的当前分量按单元侧分开显示"
        )
        form.addRow("节点平均阈值", self.averaging_threshold)
        layout.addWidget(render_group)

        self.custom_color_group = QGroupBox("自定义色标", page)
        custom_color_layout = QVBoxLayout(self.custom_color_group)
        custom_color_layout.setContentsMargins(8, 8, 8, 8)
        custom_color_layout.setSpacing(6)
        self.custom_color_stops = QTableWidget(0, 2, self.custom_color_group)
        self.custom_color_stops.setHorizontalHeaderLabels(("位置 (%)", "颜色"))
        self.custom_color_stops.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.custom_color_stops.setMinimumHeight(98)
        self.custom_color_stops.setMaximumHeight(150)
        self.custom_color_stops.setColumnWidth(0, 110)
        self.custom_color_stops.horizontalHeader().setStretchLastSection(True)
        custom_color_layout.addWidget(self.custom_color_stops)
        custom_buttons = QHBoxLayout()
        custom_buttons.setSpacing(8)
        self.add_custom_color_button = QPushButton("添加色标", self.custom_color_group)
        self.remove_custom_color_button = QPushButton(
            "删除选中",
            self.custom_color_group,
        )
        custom_buttons.addWidget(self.add_custom_color_button)
        custom_buttons.addWidget(self.remove_custom_color_button)
        custom_buttons.addStretch(1)
        custom_color_layout.addLayout(custom_buttons)
        layout.addWidget(self.custom_color_group)
        self._set_custom_color_stops(_option_color_stops(self._options))
        self.add_custom_color_button.clicked.connect(self._add_custom_color_stop)
        self.remove_custom_color_button.clicked.connect(
            self._remove_custom_color_stop
        )
        self.colormap.currentIndexChanged.connect(self._sync_colormap_controls)
        self.colormap_reverse.toggled.connect(self._refresh_colormap_preview)
        self._sync_colormap_controls()

        range_group = QGroupBox("范围", page)
        range_layout = QVBoxLayout(range_group)
        self.minimum = _SignificantDigitsDoubleSpinBox(5, range_group)
        self.maximum = _SignificantDigitsDoubleSpinBox(5, range_group)
        for spin in (self.minimum, self.maximum):
            spin.setRange(-1.0e30, 1.0e30)
            spin.setDecimals(12)
            spin.setFixedWidth(120)
        range_mode = str(self._options.get("range_mode", "")).strip().casefold()
        if range_mode not in {"per_frame", "global_step", "manual"}:
            range_mode = "manual" if self._options.get("manual") else "per_frame"
        self._range_mode = range_mode
        if range_mode == "manual":
            minimum = float(self._options.get("minimum", 0.0))
            maximum = float(self._options.get("maximum", 1.0))
        elif range_mode == "global_step":
            minimum = float(
                self._options.get(
                    "global_minimum",
                    self._options.get("automatic_minimum", self._options.get("minimum", 0.0)),
                )
            )
            maximum = float(
                self._options.get(
                    "global_maximum",
                    self._options.get("automatic_maximum", self._options.get("maximum", 1.0)),
                )
            )
        else:
            minimum = float(
                self._options.get("automatic_minimum", self._options.get("minimum", 0.0))
            )
            maximum = float(
                self._options.get("automatic_maximum", self._options.get("maximum", 1.0))
            )
        self.minimum.setValue(minimum)
        self.maximum.setValue(maximum)

        self.auto_range = QRadioButton("每帧自动", range_group)
        self.global_range = QRadioButton("全步固定", range_group)
        self.manual_range = QRadioButton("手动", range_group)
        {
            "per_frame": self.auto_range,
            "global_step": self.global_range,
            "manual": self.manual_range,
        }[range_mode].setChecked(True)
        self.range_buttons = QButtonGroup(range_group)
        for button in (self.auto_range, self.global_range, self.manual_range):
            self.range_buttons.addButton(button)
        range_controls = QHBoxLayout()
        range_controls.setSpacing(18)
        for button in (self.auto_range, self.global_range, self.manual_range):
            range_controls.addWidget(button)
        range_controls.addStretch(1)
        range_layout.addLayout(range_controls)

        value_row = QHBoxLayout()
        value_row.setSpacing(8)
        value_row.addWidget(QLabel("最小值", range_group))
        value_row.addWidget(self.minimum)
        value_row.addSpacing(18)
        value_row.addWidget(QLabel("最大值", range_group))
        value_row.addWidget(self.maximum)
        value_row.addStretch(1)
        range_layout.addLayout(value_row)
        layout.addWidget(range_group)
        layout.addStretch(1)

        for button in (self.auto_range, self.global_range, self.manual_range):
            button.toggled.connect(self._sync_range_mode)
        self._sync_range_mode()
        return page

    def _build_entity_page(self) -> QWidget:
        page, layout = self._page("entityDisplayPage")
        edge_group = QGroupBox("边线显示", page)
        form = QFormLayout(edge_group)
        configure_form_layout(form)
        self.edge_mode = QComboBox(edge_group)
        for label, key in (
            ("几何边", CONTOUR_EDGE_GEOMETRY),
            ("全部边", CONTOUR_EDGE_ALL),
            ("外部边", CONTOUR_EDGE_EXTERIOR),
            ("特征边", CONTOUR_EDGE_FEATURE),
            ("自由边", CONTOUR_EDGE_FREE),
            ("无边", CONTOUR_EDGE_NONE),
        ):
            self.edge_mode.addItem(label, key)
        selected_edge_mode = self._options.get("edge_mode")
        if selected_edge_mode is None:
            selected_edge_mode = CONTOUR_EDGE_ALL if self._options.get("edges") else CONTOUR_EDGE_GEOMETRY
        self.edge_mode.setCurrentIndex(
            max(0, self.edge_mode.findData(selected_edge_mode))
        )
        self.edge_mode.setFixedWidth(120)
        form.addRow("线条", self.edge_mode)
        self.edge_style = QComboBox(edge_group)
        for label, key in (
            ("实线", "solid"),
            ("虚线", "dashed"),
            ("短划线", "short_dashed"),
            ("加粗线", "bold"),
        ):
            self.edge_style.addItem(label, key)
        self.edge_style.setCurrentIndex(
            max(0, self.edge_style.findData(self._options.get("edge_style", "solid")))
        )
        form.addRow("样式", self.edge_style)
        self.edge_width = QDoubleSpinBox(edge_group)
        self.edge_width.setRange(0.1, 20.0)
        self.edge_width.setDecimals(1)
        self.edge_width.setSingleStep(0.5)
        self.edge_width.setValue(float(self._options.get("edge_width", 1.0)))
        self.edge_width.setFixedWidth(70)
        form.addRow("粗细", self.edge_width)
        layout.addWidget(edge_group)

        label_group = QGroupBox("实体与符号", page)
        label_layout = QVBoxLayout(label_group)
        self.show_node_labels = QCheckBox("显示节点编号", label_group)
        self.show_node_labels.setChecked(bool(self._options.get("show_node_labels", False)))
        self.show_element_labels = QCheckBox("显示单元编号", label_group)
        self.show_element_labels.setChecked(bool(self._options.get("show_element_labels", False)))
        self.show_symbols = QCheckBox("显示边界条件和载荷符号", label_group)
        self.show_symbols.setChecked(bool(self._options.get("show_symbols", False)))
        for checkbox in (
            self.show_node_labels,
            self.show_element_labels,
            self.show_symbols,
        ):
            label_layout.addWidget(checkbox)
        layout.addWidget(label_group)
        layout.addStretch(1)
        return page

    def _build_annotation_page(self) -> QWidget:
        page, layout = self._page("annotationPage")
        group = QGroupBox("图例与标注", page)
        form = QFormLayout(group)
        configure_form_layout(form)
        self.legend = QCheckBox("显示图例", group)
        self.legend.setChecked(bool(self._options.get("legend", True)))
        form.addRow("图例", self.legend)
        self.state_info_label = QLabel(
            str(self._options.get("state_info", "当前分析步 / 当前帧")),
            group,
        )
        self.state_info_label.setWordWrap(True)
        self.state_info_label.setObjectName("visualizationStateInfo")
        form.addRow("当前状态", self.state_info_label)
        self.legend_title = QCheckBox("显示结果标题", group)
        self.legend_title.setChecked(bool(self._options.get("legend_title", True)))
        form.addRow("标题", self.legend_title)
        self.show_ids = QCheckBox("在极值标签中显示实体编号", group)
        self.show_ids.setChecked(bool(self._options.get("show_ids", False)))
        form.addRow("极值编号", self.show_ids)
        self.show_minimum = QCheckBox("显示最小值标签", group)
        self.show_minimum.setChecked(
            bool(self._options.get("show_minimum", False))
        )
        form.addRow("最小值", self.show_minimum)
        self.show_maximum = QCheckBox("显示最大值标签", group)
        self.show_maximum.setChecked(
            bool(self._options.get("show_maximum", False))
        )
        form.addRow("最大值", self.show_maximum)
        self.show_coordinate_system = QCheckBox("显示坐标三向标", group)
        self.show_coordinate_system.setChecked(
            bool(self._options.get("show_coordinate_system", True))
        )
        form.addRow("坐标系", self.show_coordinate_system)

        self.number_format_buttons = QButtonGroup(group)
        self.scientific_format = QRadioButton("科学计数", group)
        self.engineering_format = QRadioButton("工程计数", group)
        self.number_format_buttons.addButton(self.scientific_format)
        self.number_format_buttons.addButton(self.engineering_format)
        if self._options.get("number_format", "scientific") == "engineering":
            self.engineering_format.setChecked(True)
        else:
            self.scientific_format.setChecked(True)
        number_host = QWidget(group)
        number_layout = QHBoxLayout(number_host)
        number_layout.setContentsMargins(0, 0, 0, 0)
        number_layout.setSpacing(18)
        number_layout.addWidget(self.scientific_format)
        number_layout.addWidget(self.engineering_format)
        number_layout.addStretch(1)
        form.addRow("数值格式", number_host)

        self.decimals = QSpinBox(group)
        self.decimals.setRange(0, 12)
        self.decimals.setValue(int(self._options.get("decimals", 2)))
        self.decimals.setFixedWidth(70)
        form.addRow("小数位", self.decimals)

        self.orientation_buttons = QButtonGroup(group)
        self.horizontal_orientation = QRadioButton("横向", group)
        self.vertical_orientation = QRadioButton("纵向", group)
        self.orientation_buttons.addButton(self.horizontal_orientation)
        self.orientation_buttons.addButton(self.vertical_orientation)
        if self._options.get("orientation", "vertical") == "horizontal":
            self.horizontal_orientation.setChecked(True)
        else:
            self.vertical_orientation.setChecked(True)
        orientation_host = QWidget(group)
        orientation_layout = QHBoxLayout(orientation_host)
        orientation_layout.setContentsMargins(0, 0, 0, 0)
        orientation_layout.setSpacing(18)
        orientation_layout.addWidget(self.horizontal_orientation)
        orientation_layout.addWidget(self.vertical_orientation)
        orientation_layout.addStretch(1)
        form.addRow("图例方向", orientation_host)

        self.legend_position = QComboBox(group)
        for label, key in (
            ("自动", "auto"),
            ("左侧", "left"),
            ("右侧", "right"),
            ("顶部", "top"),
            ("底部", "bottom"),
        ):
            self.legend_position.addItem(label, key)
        self.legend_position.setCurrentIndex(
            max(0, self.legend_position.findData(self._options.get("legend_position", "auto")))
        )
        form.addRow("图例位置", self.legend_position)

        self.legend_font = QComboBox(group)
        for font in ("Arial", "Times New Roman", "Courier New"):
            self.legend_font.addItem(font, font)
        self.legend_font.setCurrentIndex(
            max(0, self.legend_font.findData(self._options.get("legend_font", "Arial")))
        )
        form.addRow("字体", self.legend_font)
        self.legend_font_size = QSpinBox(group)
        self.legend_font_size.setRange(6, 72)
        self.legend_font_size.setValue(int(self._options.get("legend_font_size", 14)))
        self.legend_font_size.setFixedWidth(70)
        form.addRow("字体大小", self.legend_font_size)
        self.legend_label_count = QComboBox(group)
        for label, key in (
            ("自动", "auto"),
            ("3 个", "3"),
            ("5 个", "5"),
            ("7 个", "7"),
            ("9 个", "9"),
        ):
            self.legend_label_count.addItem(label, key)
        self.legend_label_count.setCurrentIndex(
            max(0, self.legend_label_count.findData(str(self._options.get("legend_label_count", "auto"))))
        )
        form.addRow("刻度数", self.legend_label_count)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _set_custom_color_stops(self, stops: object) -> None:
        normalized = normalize_color_stops(stops)
        self.custom_color_stops.setRowCount(0)
        for position, color in normalized:
            row = self.custom_color_stops.rowCount()
            self.custom_color_stops.insertRow(row)
            position_box = QDoubleSpinBox(self.custom_color_stops)
            position_box.setRange(0.0, 100.0)
            position_box.setDecimals(1)
            position_box.setSingleStep(5.0)
            position_box.setValue(position * 100.0)
            position_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            color_button = _ContourColorButton(color, self.custom_color_stops)
            position_box.valueChanged.connect(
                lambda _value: self._refresh_colormap_preview()
            )
            color_button.colorChanged.connect(
                lambda _color: self._refresh_colormap_preview()
            )
            self.custom_color_stops.setCellWidget(row, 0, position_box)
            self.custom_color_stops.setCellWidget(row, 1, color_button)

    def _read_custom_color_stops(self) -> list[tuple[float, str]]:
        stops: list[tuple[float, str]] = []
        for row in range(self.custom_color_stops.rowCount()):
            position_box = self.custom_color_stops.cellWidget(row, 0)
            color_button = self.custom_color_stops.cellWidget(row, 1)
            if not isinstance(position_box, QDoubleSpinBox):
                continue
            if not isinstance(color_button, _ContourColorButton):
                continue
            stops.append((position_box.value() / 100.0, color_button.color()))
        return stops

    def _refresh_colormap_preview(self) -> None:
        selected = str(self.colormap.currentData() or ABAQUS_RAINBOW)
        try:
            stops = normalize_color_stops(self._read_custom_color_stops())
        except (TypeError, ValueError):
            stops = DEFAULT_CUSTOM_COLOR_STOPS
        self.colormap_preview.set_colors(
            preview_contour_colors(
                selected,
                reverse=self.colormap_reverse.isChecked(),
                custom_color_stops=stops,
            )
        )

    def _sync_colormap_controls(self) -> None:
        custom = self.colormap.currentData() == CUSTOM_COLORMAP
        self.custom_color_group.setVisible(custom)
        self._refresh_colormap_preview()

    def _add_custom_color_stop(self) -> None:
        try:
            stops = list(normalize_color_stops(self._read_custom_color_stops()))
        except (TypeError, ValueError):
            stops = list(DEFAULT_CUSTOM_COLOR_STOPS)
        if len(stops) >= 9:
            return
        left, right = max(
            zip(stops, stops[1:]),
            key=lambda pair: pair[1][0] - pair[0][0],
        )
        position = (left[0] + right[0]) / 2.0
        sampled = interpolate_color_stops(stops, 101)[round(position * 100.0)]
        stops.append((position, sampled))
        self._set_custom_color_stops(stops)
        self._refresh_colormap_preview()

    def _remove_custom_color_stop(self) -> None:
        if self.custom_color_stops.rowCount() <= 2:
            return
        row = self.custom_color_stops.currentRow()
        if row < 0:
            row = self.custom_color_stops.rowCount() - 1
        self.custom_color_stops.removeRow(row)
        self._refresh_colormap_preview()

    def _sync_scale_mode(self) -> None:
        self.scale_value.setEnabled(self.scale_mode.currentData() == "custom")

    def _mark_render_mode_changed(self, checked: bool) -> None:
        if checked:
            self._render_mode_changed = True

    def _sync_range_mode(self) -> None:
        manual = self.manual_range.isChecked()
        self.minimum.setEnabled(manual)
        self.maximum.setEnabled(manual)

    def select_category(self, category: str) -> None:
        if category not in self._categories:
            raise ValueError(f"unknown visualization category: {category}")
        self.category_list.setCurrentRow(self._categories.index(category))

    def settings(self) -> dict[str, Any]:
        if self.manual_range.isChecked():
            range_mode = "manual"
        elif self.global_range.isChecked():
            range_mode = "global_step"
        else:
            range_mode = "per_frame"
        raw_custom_color_stops = self._read_custom_color_stops()
        try:
            custom_color_stops: object = normalize_color_stops(
                raw_custom_color_stops
            )
        except (TypeError, ValueError):
            # Keep the raw values until apply() can show the user the precise
            # validation message; valid values are always emitted canonically.
            custom_color_stops = tuple(raw_custom_color_stops)
        edge_mode = str(self.edge_mode.currentData())
        render_mode = str(
            self._options.get("render_mode", CONTOUR_RENDER_SHADED)
        )
        if self._render_mode_changed:
            render_mode = (
                CONTOUR_RENDER_FILLED
                if self.filled_mode.isChecked()
                else CONTOUR_RENDER_SHADED
            )
        if self._display_style_changed:
            display_style = str(self.display_style.currentData())
            if display_style == "wireframe":
                render_mode = CONTOUR_RENDER_WIREFRAME
                edge_mode = CONTOUR_EDGE_ALL
            elif display_style == "hidden_line":
                render_mode = CONTOUR_RENDER_HIDDEN_LINE
                edge_mode = CONTOUR_EDGE_GEOMETRY
            elif display_style == "shaded_edges":
                render_mode = CONTOUR_RENDER_SHADED
                if edge_mode == CONTOUR_EDGE_NONE:
                    edge_mode = CONTOUR_EDGE_GEOMETRY
            else:
                render_mode = CONTOUR_RENDER_SHADED
                edge_mode = CONTOUR_EDGE_NONE
        return {
            "display_style": str(self.display_style.currentData()),
            "shape_mode": str(self.shape_mode.currentData()),
            "contour_enabled": self.contour_checkbox.isChecked(),
            "face_color": self.face_color_control.value(),
            "edge_color": self.edge_color_control.value(),
            "model_opacity": float(self.model_opacity.value()) / 100.0,
            "scale_mode": str(self.scale_mode.currentData()),
            "scale_value": float(self.scale_value.value()),
            "overlay_undeformed": self.overlay_checkbox.isChecked(),
            "deformed_color": self.deformed_color_control.value(),
            "deformed_line_style": str(self.deformed_line_style.currentData()),
            "deformed_opacity": float(self.deformed_opacity.value()) / 100.0,
            "undeformed_color": self.undeformed_color_control.value(),
            "undeformed_line_style": str(self.undeformed_line_style.currentData()),
            "undeformed_opacity": float(self.undeformed_opacity.value()) / 100.0,
            "show_edges": edge_mode != CONTOUR_EDGE_NONE,
            "edge_mode": edge_mode,
            "edge_style": str(self.edge_style.currentData()),
            "edge_width": float(self.edge_width.value()),
            "show_node_labels": self.show_node_labels.isChecked(),
            "show_element_labels": self.show_element_labels.isChecked(),
            "show_symbols": self.show_symbols.isChecked(),
            "render_mode": render_mode,
            "style": str(self.style.currentData()),
            "colormap": str(self.colormap.currentData()),
            "colormap_reverse": self.colormap_reverse.isChecked(),
            "custom_color_stops": custom_color_stops,
            "levels": int(self.levels.value()),
            "averaging_threshold": float(self.averaging_threshold.value()),
            "range_mode": range_mode,
            "manual": range_mode == "manual",
            "minimum": float(self.minimum.value()),
            "maximum": float(self.maximum.value()),
            "show_minimum": self.show_minimum.isChecked(),
            "show_maximum": self.show_maximum.isChecked(),
            "legend": self.legend.isChecked(),
            "legend_title": self.legend_title.isChecked(),
            "show_ids": self.show_ids.isChecked(),
            "show_coordinate_system": self.show_coordinate_system.isChecked(),
            "number_format": (
                "scientific"
                if self.scientific_format.isChecked()
                else "engineering"
            ),
            "decimals": int(self.decimals.value()),
            "orientation": (
                "horizontal"
                if self.horizontal_orientation.isChecked()
                else "vertical"
            ),
            "legend_position": str(self.legend_position.currentData()),
            "legend_font": str(self.legend_font.currentData()),
            "legend_font_size": int(self.legend_font_size.value()),
            "legend_label_count": str(self.legend_label_count.currentData()),
            "edges": edge_mode != CONTOUR_EDGE_NONE,
        }

    def apply(self) -> bool:
        if self.manual_range.isChecked() and self.minimum.value() >= self.maximum.value():
            QMessageBox.warning(self, "云图显示", "手动范围的最小值必须小于最大值。")
            self.select_category("云图显示")
            return False
        try:
            settings = self.settings()
            normalize_color_stops(settings["custom_color_stops"])
        except (TypeError, ValueError):
            QMessageBox.warning(
                self,
                "云图显示",
                "自定义色标必须从 0% 开始、到 100% 结束，且位置不能重复。",
            )
            self.select_category("云图显示")
            return False
        self.applyRequested.emit(settings)
        return True

    def accept_with_apply(self) -> None:
        if self.apply():
            self.accept()


class TypedResultQueryDialog(QDialog):
    """用 provider catalog 构造精确查询，数值工作由外层命令完成。"""

    selectionRequested = Signal(object)
    queryRequested = Signal(object)

    def __init__(
        self,
        provider: ResultProvider,
        catalog: ResultCatalog | None = None,
        *,
        parent=None,
    ) -> None:
        if type(provider) is not ResultProvider:
            raise TypeError("provider must be ResultProvider")
        provider_catalog = provider.catalog()
        if catalog is None:
            catalog = provider_catalog
        elif type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be ResultCatalog or None")
        elif catalog != provider_catalog:
            raise ValueError("catalog must exactly match provider.catalog()")

        super().__init__(parent)
        self.resize(900, 520)
        self._catalog = catalog
        self._source = provider.source
        self._frame_key = provider.frame_key
        self.setWindowTitle(
            "查询结果"
            if self._frame_key is None
            else f"查询结果 · 增量 {self._frame_key.frame_index}"
        )
        self._section_point_labels = result_provider_section_point_labels(
            provider
        )
        self._initial_generation = provider.snapshot.generation
        self._node_ids = provider.snapshot.topology.node_ids
        self._element_ids = provider.snapshot.topology.element_ids
        self._last_query: ResultQuery | None = None
        self._displayed_generation: int | None = None
        self._query_pending = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        self.step_combo = QComboBox(self)
        self.step_combo.addItem(self._source.step_name, self._source)
        self.association_combo = QComboBox(self)
        self.association_combo.addItem(
            "节点",
            _TypedQueryMode(FieldAssociation.NODE),
        )
        self.association_combo.addItem(
            "单元",
            _TypedQueryMode(FieldAssociation.ELEMENT),
        )
        self.field_combo = QComboBox(self)
        self.component_combo = QComboBox(self)
        self.ids_edit = QLineEdit(self)
        self.ids_edit.setPlaceholderText("留空查询全部；例如：1, 3, 5-8")
        form.addRow("结果步：", self.step_combo)
        form.addRow("对象类型：", self.association_combo)
        form.addRow("场变量：", self.field_combo)
        form.addRow("分量：", self.component_combo)
        form.addRow("对象编号：", self.ids_edit)
        layout.addLayout(form)

        command_row = QHBoxLayout()
        self.query_button = QPushButton("查询", self)
        copy_button = QPushButton("复制", self)
        self.query_button.clicked.connect(self.request_query)
        copy_button.clicked.connect(self.copy_table)
        command_row.addWidget(self.query_button)
        command_row.addWidget(copy_button)
        command_row.addStretch(1)
        layout.addLayout(command_row)

        self.result_summary = QLabel("尚未查询", self)
        layout.addWidget(self.result_summary)
        self.table = QTableWidget(self)
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.table, 1)
        self._prepare_table(include_section_points=False)

        default_availability = self._availability_for_key(
            self._catalog.default_selection.field_key
        )
        default_mode = (
            FieldAssociation.NODE
            if _typed_query_association_matches(
                FieldAssociation.NODE,
                default_availability.descriptor.association,
            )
            else FieldAssociation.ELEMENT
        )
        self.association_combo.setCurrentIndex(
            0 if default_mode is FieldAssociation.NODE else 1
        )
        self._sync_fields(
            preferred_key=self._catalog.default_selection.field_key,
            emit_selection=False,
        )
        self.association_combo.currentIndexChanged.connect(
            self._association_changed
        )
        self.field_combo.currentIndexChanged.connect(self._field_changed)
        self.component_combo.currentIndexChanged.connect(
            self._component_changed
        )

    @property
    def catalog(self) -> ResultCatalog:
        """返回 dialog 绑定的 exact immutable catalog。"""

        return self._catalog

    @property
    def source(self) -> ResultSourceKey:
        """返回 dialog 打开时绑定的完整结果来源。"""

        return self._source

    @property
    def frame_key(self) -> ResultFrameKey | None:
        """返回查询打开时绑定的结果帧；旧单帧结果返回 ``None``。"""

        return self._frame_key

    @property
    def query_pending(self) -> bool:
        """返回是否已有一个 typed query 正在执行。"""

        return self._query_pending

    def set_query_pending(self, pending: bool) -> None:
        """在一个查询生命周期内冻结所有 query intent 控件。"""

        if type(pending) is not bool:
            raise TypeError("pending must be a bool")
        self._query_pending = pending
        enabled = not pending
        self.association_combo.setEnabled(enabled)
        self.field_combo.setEnabled(enabled)
        self.component_combo.setEnabled(enabled)
        self.ids_edit.setEnabled(enabled)
        if pending:
            self.result_summary.setText("正在查询……")
        self._refresh_availability()

    def set_query_message(self, message: str) -> None:
        """显示不携带 records 的查询状态消息。"""

        if type(message) is not str:
            raise TypeError("message must be a string")
        self.result_summary.setText(message.strip() or "结果查询未完成")

    def current_availability(self) -> FieldAvailability:
        """返回当前字段的 typed catalog entry。"""

        key = self.field_combo.currentData()
        if type(key) is not FieldMaterializationKey:
            raise RuntimeError("no typed result field is selected")
        return self._availability_for_key(key)

    def current_selection(self) -> ScalarFieldSelection:
        """返回当前完整 field key 与 scalar component。"""

        availability = self.current_availability()
        component = self.component_combo.currentData()
        if type(component) is not str:
            raise RuntimeError("no typed scalar component is selected")
        if component not in availability.descriptor.columns:
            raise RuntimeError(
                "selected component is outside the field descriptor"
            )
        return ScalarFieldSelection(availability.key, component)

    def current_query(self) -> ResultQuery:
        """根据 typed association 与 FEM ID 输入构造精确查询。"""

        availability = self.current_availability()
        if availability.state is FieldState.UNAVAILABLE:
            raise ValueError("当前字段不可查询。")
        selection = self.current_selection()
        mode = self.association_combo.currentData()
        if type(mode) is not _TypedQueryMode:
            raise RuntimeError("query association must be typed")
        if mode.association is FieldAssociation.NODE:
            node_ids = _parse_typed_query_ids(
                self.ids_edit.text(),
                self._node_ids,
            )
            element_ids: tuple[int, ...] = ()
        elif mode.association is FieldAssociation.ELEMENT:
            node_ids = ()
            element_ids = _parse_typed_query_ids(
                self.ids_edit.text(),
                self._element_ids,
            )
        else:
            raise RuntimeError("query association must be typed")
        return ResultQuery(
            field_key=selection.field_key,
            component=selection.component,
            node_ids=node_ids,
            element_ids=element_ids,
        )

    def request_query(self, *_args: object) -> None:
        """把 selection/query 交给外层，不在 dialog 内恢复或读取字段。"""

        if self._query_pending:
            return
        try:
            selection = self.current_selection()
            query = self.current_query()
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "查询结果", str(error))
            return
        self._last_query = query
        self.selectionRequested.emit(selection)
        self.queryRequested.emit(query)

    def set_query_result(self, result: ResultQueryResult) -> None:
        """按 application 结果原序显示全部 query records。"""

        if type(result) is not ResultQueryResult:
            raise TypeError("result must be ResultQueryResult")
        if result.source != self._source:
            raise ValueError("query result source must match the dialog source")
        minimum_generation = self._initial_generation
        if self._displayed_generation is not None:
            minimum_generation = max(
                minimum_generation,
                self._displayed_generation,
            )
        if result.materialization_generation < minimum_generation:
            raise ValueError("query result generation is stale")
        expected_query = self._last_query
        if expected_query is None:
            expected_query = self.current_query()
        if result.query != expected_query:
            raise ValueError("query result must match the latest dialog query")

        availability = self._availability_for_key(result.query.field_key)
        include_section_points = result_field_has_section_points(
            availability.descriptor.field_id
        )
        self._prepare_table(
            include_section_points=include_section_points
        )
        region_labels = result_region_display_labels(
            record.location.region_key
            for record in result.records
            if record.location.region_key is not None
        )
        region_column = self.table.columnCount() - 6
        self.table.setRowCount(len(result.records))
        for row, record in enumerate(result.records):
            location = record.location
            section_values = (
                (
                    (
                        ""
                        if location.section_point is None
                        else section_point_relative_position_label(
                            location.section_point
                        )
                    ),
                    _optional_number_text(
                        None
                        if location.section_point is None
                        else location.section_point.local_y
                    ),
                    _optional_number_text(
                        None
                        if location.section_point is None
                        else location.section_point.local_z
                    ),
                )
                if include_section_points
                else ()
            )
            values = (
                _typed_association_text(location.association),
                _optional_identity_text(location.node_id),
                _optional_identity_text(location.element_id),
                _optional_identity_text(location.integration_point),
                _optional_identity_text(location.local_node),
                *section_values,
                (
                    ""
                    if location.region_key is None
                    else region_labels[location.region_key]
                ),
                _averaged_text(location.averaged),
                _number_text(location.coordinates[0]),
                _number_text(location.coordinates[1]),
                _number_text(location.coordinates[2]),
                _number_text(record.value),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record)
                if (
                    column == region_column
                    and location.region_key is not None
                ):
                    item.setToolTip(
                        encode_result_region_key(location.region_key)
                    )
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        if self.table.rowCount():
            self.table.selectRow(0)
        self._displayed_generation = result.materialization_generation
        self.result_summary.setText(
            f"共 {len(result.records)} 行 · "
            f"结果版本 {result.materialization_generation}"
        )

    def show_result(self, result: ResultQueryResult) -> None:
        """兼容 Qt slot 风格的显式展示入口。"""

        self.set_query_result(result)

    def record_at(self, row: int) -> ResultQueryRecord:
        """返回显示行绑定的 exact ResultQueryRecord。"""

        if type(row) is not int:
            raise TypeError("row must be an integer")
        if row < 0 or row >= self.table.rowCount():
            raise IndexError(row)
        item = self.table.item(row, 0)
        if item is None:
            raise RuntimeError("query table row is incomplete")
        record = item.data(Qt.ItemDataRole.UserRole)
        if type(record) is not ResultQueryRecord:
            raise RuntimeError("query table row lost its typed record")
        return record

    def copy_table(self) -> None:
        """复制当前 typed query 表格。"""

        QApplication.clipboard().setText(self._table_text("\t"))

    def _association_changed(self, *_args: object) -> None:
        if self._query_pending:
            return
        self._last_query = None
        self._sync_fields(emit_selection=True)

    def _field_changed(self, *_args: object) -> None:
        if self._query_pending:
            return
        self._last_query = None
        self._sync_components()
        self._refresh_availability()
        self._emit_selection_requested()

    def _component_changed(self, *_args: object) -> None:
        if self._query_pending:
            return
        self._last_query = None
        self._emit_selection_requested()

    def _sync_fields(
        self,
        *,
        preferred_key: FieldMaterializationKey | None = None,
        emit_selection: bool,
    ) -> None:
        if preferred_key is None:
            candidate = self.field_combo.currentData()
            if type(candidate) is FieldMaterializationKey:
                preferred_key = candidate
        mode = self.association_combo.currentData()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        if type(mode) is _TypedQueryMode:
            for availability in visible_result_fields(
                self._catalog.fields
            ):
                if _typed_query_association_matches(
                    mode.association,
                    availability.descriptor.association,
                ):
                    self.field_combo.addItem(
                        _typed_field_label(
                            availability,
                            self._section_point_labels,
                        ),
                        availability.key,
                    )
        selected_index = self.field_combo.findData(preferred_key)
        if selected_index < 0:
            default_key = self._catalog.default_selection.field_key
            selected_index = self.field_combo.findData(default_key)
        self.field_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        self.field_combo.blockSignals(False)
        self._sync_components()
        self._refresh_availability()
        if emit_selection:
            self._emit_selection_requested()

    def _sync_components(self) -> None:
        current = self.component_combo.currentData()
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        try:
            availability = self.current_availability()
        except RuntimeError:
            self.component_combo.blockSignals(False)
            return
        for component in availability.descriptor.columns:
            self.component_combo.addItem(component, component)
        selected_component = current
        if (
            availability.key
            == self._catalog.default_selection.field_key
        ):
            selected_component = self._catalog.default_selection.component
        index = self.component_combo.findData(selected_component)
        if index < 0:
            index = self.component_combo.findData(
                availability.descriptor.default_component
            )
        self.component_combo.setCurrentIndex(index if index >= 0 else 0)
        self.component_combo.blockSignals(False)

    def _refresh_availability(self) -> None:
        try:
            availability = self.current_availability()
        except RuntimeError:
            self.query_button.setEnabled(False)
            return
        self.query_button.setEnabled(
            not self._query_pending
            and availability.state is not FieldState.UNAVAILABLE
        )

    def _emit_selection_requested(self) -> None:
        try:
            availability = self.current_availability()
            selection = self.current_selection()
        except RuntimeError:
            return
        if availability.state is not FieldState.UNAVAILABLE:
            self.selectionRequested.emit(selection)

    def _availability_for_key(
        self,
        key: FieldMaterializationKey,
    ) -> FieldAvailability:
        for availability in self._catalog.fields:
            if availability.key == key:
                return availability
        raise RuntimeError("field key is outside the dialog catalog")

    def _prepare_table(self, *, include_section_points: bool) -> None:
        section_headers = (
            ("截面位置", "截面局部 Y", "截面局部 Z")
            if include_section_points
            else ()
        )
        headers = (
            "关联",
            "节点",
            "单元",
            "积分点",
            "局部节点",
            *section_headers,
            "区域",
            "平均状态",
            "X",
            "Y",
            "Z",
            "值",
        )
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(0)

    def _table_rows(self) -> list[list[str]]:
        return [
            [
                self.table.horizontalHeaderItem(column).text()
                for column in range(self.table.columnCount())
            ],
            *[
                [
                    self.table.item(row, column).text()
                    for column in range(self.table.columnCount())
                ]
                for row in range(self.table.rowCount())
            ],
        ]

    def _table_text(self, separator: str) -> str:
        return "\n".join(
            separator.join(row) for row in self._table_rows()
        )


class ResultProbeDialog(QDialog):
    """绑定一个结果帧的精确位置探针。"""

    probeRequested = Signal(object)
    pickRequested = Signal(object)
    probeResultReady = Signal(object)

    def __init__(
        self,
        provider: ResultProvider,
        *,
        current_selection: ScalarFieldSelection | None = None,
        parent=None,
    ) -> None:
        if type(provider) is not ResultProvider:
            raise TypeError("provider must be ResultProvider")
        catalog = provider.catalog()
        if not catalog.fields:
            raise ValueError("probe requires a non-empty result catalog")

        super().__init__(parent)
        self.resize(860, 540)
        self._catalog = catalog
        self._source = provider.source
        self._frame_key = provider.frame_key
        self._initial_generation = provider.snapshot.generation
        self._node_ids = provider.snapshot.topology.node_ids
        self._element_ids = provider.snapshot.topology.element_ids
        self._section_point_labels = result_provider_section_point_labels(
            provider
        )
        self._last_request: ResultProbeRequest | None = None
        self._displayed_generation: int | None = None
        self._probe_pending = False
        self._probe_history: list[ResultProbeResult] = []

        self.setWindowTitle(
            "结果探针"
            if self._frame_key is None
            else f"结果探针 · 增量 {self._frame_key.frame_index}"
        )
        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)

        self.field_combo = QComboBox(self)
        self.component_combo = QComboBox(self)
        self.target_combo = QComboBox(self)
        self.target_combo.addItem("节点", ResultProbeKind.NODE)
        self.target_combo.addItem("单元", ResultProbeKind.ELEMENT)
        self.target_combo.addItem(
            "积分点",
            ResultProbeKind.INTEGRATION_POINT,
        )
        self.target_combo.addItem(
            "单元局部节点",
            ResultProbeKind.ELEMENT_NODE,
        )
        self.target_id_label = QLabel("编号", self)
        self.target_id_spin = QSpinBox(self)
        self.target_id_spin.setRange(1, 1_000_000_000)
        self.integration_point_label = QLabel("积分点编号", self)
        self.integration_point_spin = QSpinBox(self)
        self.integration_point_spin.setRange(1, 1_000_000_000)
        self.local_node_label = QLabel("局部节点编号", self)
        self.local_node_spin = QSpinBox(self)
        self.local_node_spin.setRange(1, 1_000_000_000)
        form.addRow("结果步", self._step_label())
        form.addRow("场变量", self.field_combo)
        form.addRow("分量", self.component_combo)
        form.addRow("定位类型", self.target_combo)
        form.addRow(self.target_id_label, self.target_id_spin)
        form.addRow(
            self.integration_point_label,
            self.integration_point_spin,
        )
        form.addRow(self.local_node_label, self.local_node_spin)
        layout.addLayout(form)

        self.availability_label = QLabel(self)
        self.availability_label.setWordWrap(True)
        layout.addWidget(self.availability_label)
        command_row = QHBoxLayout()
        self.probe_button = QPushButton("读取", self)
        copy_button = QPushButton("复制", self)
        self.pick_button = QPushButton("从视口拾取", self)
        self.probe_button.clicked.connect(self.request_probe)
        self.pick_button.clicked.connect(
            lambda: self.pickRequested.emit(self._current_probe_kind())
        )
        copy_button.clicked.connect(self.copy_table)
        command_row.addWidget(self.pick_button)
        command_row.addWidget(self.probe_button)
        command_row.addWidget(copy_button)
        command_row.addStretch(1)
        layout.addLayout(command_row)

        self.result_summary = QLabel("尚未定位", self)
        layout.addWidget(self.result_summary)
        self.probe_history_list = QListWidget(self)
        self.probe_history_list.setObjectName("probeHistoryList")
        self.probe_history_list.setMaximumHeight(90)
        self.probe_history_list.setToolTip("本次结果查询窗口内已定位的 Probe")
        layout.addWidget(QLabel("Probe 记录", self))
        layout.addWidget(self.probe_history_list)
        self.table = QTableWidget(self)
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.table, 1)

        entries = visible_result_fields(catalog.fields)
        for availability in entries:
            self.field_combo.addItem(
                _typed_result_display_field_label(
                    availability,
                    self._section_point_labels,
                ),
                availability.key,
            )
        preferred = current_selection
        if (
            type(preferred) is not ScalarFieldSelection
            or self.field_combo.findData(preferred.field_key) < 0
        ):
            preferred = catalog.default_selection
        if preferred is None:
            raise ValueError("probe catalog has no default field selection")
        field_index = self.field_combo.findData(preferred.field_key)
        self.field_combo.setCurrentIndex(field_index)
        self._sync_components(preferred_component=preferred.component)
        self._prepare_table(include_section_points=False)
        self._refresh_target_controls()
        self._refresh_availability()

        self.field_combo.currentIndexChanged.connect(self._field_changed)
        self.target_combo.currentIndexChanged.connect(
            self._target_changed
        )

    def _step_label(self) -> QLabel:
        label = QLabel(self._source.step_name, self)
        label.setToolTip(
            "探针只读取当前结果帧；切换增量后请重新打开探针。"
        )
        return label

    @property
    def source(self) -> ResultSourceKey:
        return self._source

    @property
    def frame_key(self) -> ResultFrameKey | None:
        return self._frame_key

    @property
    def probe_pending(self) -> bool:
        return self._probe_pending

    def current_availability(self) -> FieldAvailability:
        key = self.field_combo.currentData()
        if type(key) is not FieldMaterializationKey:
            raise RuntimeError("no typed result field is selected")
        for availability in self._catalog.fields:
            if availability.key == key:
                return availability
        raise RuntimeError("field key is outside the probe catalog")

    def current_selection(self) -> ScalarFieldSelection:
        availability = self.current_availability()
        component = self.component_combo.currentData()
        if type(component) is not str:
            raise RuntimeError("no typed scalar component is selected")
        if component not in availability.descriptor.columns:
            raise RuntimeError(
                "selected component is outside the field descriptor"
            )
        return ScalarFieldSelection(availability.key, component)

    def current_target(self) -> ResultProbeTarget:
        kind = self._current_probe_kind()
        if kind is ResultProbeKind.NODE:
            return ResultProbeTarget(
                kind,
                node_id=int(self.target_id_spin.value()),
            )
        if kind is ResultProbeKind.ELEMENT:
            return ResultProbeTarget(
                kind,
                element_id=int(self.target_id_spin.value()),
            )
        if kind is ResultProbeKind.INTEGRATION_POINT:
            return ResultProbeTarget(
                kind,
                element_id=int(self.target_id_spin.value()),
                integration_point=int(self.integration_point_spin.value()),
            )
        return ResultProbeTarget(
            kind,
            element_id=int(self.target_id_spin.value()),
            local_node=int(self.local_node_spin.value()),
        )

    def current_request(self) -> ResultProbeRequest:
        availability = self.current_availability()
        if availability.state is FieldState.UNAVAILABLE:
            raise ValueError("当前字段不可探针。")
        return ResultProbeRequest(
            selection=self.current_selection(),
            target=self.current_target(),
        )

    def set_probe_pending(self, pending: bool) -> None:
        if type(pending) is not bool:
            raise TypeError("pending must be a bool")
        self._probe_pending = pending
        enabled = not pending
        for widget in (
            self.field_combo,
            self.component_combo,
            self.target_combo,
            self.target_id_spin,
            self.integration_point_spin,
            self.local_node_spin,
            self.pick_button,
            self.probe_button,
        ):
            widget.setEnabled(enabled)
        if pending:
            self.result_summary.setText("正在读取当前结果位置……")
        self._refresh_target_controls()
        self._refresh_availability()

    def set_probe_message(self, message: str) -> None:
        if type(message) is not str:
            raise TypeError("message must be a string")
        self.result_summary.setText(message.strip() or "结果探针未完成")

    def request_probe(self, *_args: object) -> None:
        if self._probe_pending:
            return
        try:
            request = self.current_request()
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "结果探针", str(error))
            return
        self._last_request = request
        self.probeRequested.emit(request)

    def set_target_from_pick(self, kind: str, key: int) -> None:
        """将视口拾取到的节点/单元写入当前 Probe 目标。"""

        if type(kind) is not str or type(key) is not int or key <= 0:
            raise TypeError("picked result targets require a kind and positive integer")
        target_kind = (
            ResultProbeKind.NODE
            if kind == "node"
            else ResultProbeKind.ELEMENT
            if kind == "element"
            else None
        )
        if target_kind is None:
            raise ValueError("结果 Probe 只支持拾取节点或单元")
        index = self.target_combo.findData(target_kind)
        if index < 0:
            raise ValueError("当前 Probe 对话框不支持该拾取类型")
        self.target_combo.setCurrentIndex(index)
        self.target_id_spin.setValue(key)
        self.result_summary.setText(
            f"已从视口拾取{'节点' if target_kind is ResultProbeKind.NODE else '单元'} {key}；点击“读取”查看结果。"
        )

    def set_probe_result(self, result: ResultProbeResult) -> None:
        if type(result) is not ResultProbeResult:
            raise TypeError("result must be ResultProbeResult")
        if result.source != self._source:
            raise ValueError("probe result source must match dialog source")
        if result.frame_key != self._frame_key:
            raise ValueError("probe result frame must match dialog frame")
        minimum_generation = self._initial_generation
        if self._displayed_generation is not None:
            minimum_generation = max(
                minimum_generation,
                self._displayed_generation,
            )
        if result.materialization_generation < minimum_generation:
            raise ValueError("probe result generation is stale")
        expected_request = self._last_request
        if expected_request is None:
            expected_request = self.current_request()
        if result.request != expected_request:
            raise ValueError("probe result must match the latest probe")

        availability = self.current_availability()
        include_section_points = result_field_has_section_points(
            availability.descriptor.field_id
        )
        self._prepare_table(include_section_points=include_section_points)
        region_labels = result_region_display_labels(
            record.location.region_key
            for record in result.records
            if record.location.region_key is not None
        )
        region_column = self.table.columnCount() - 6
        self.table.setRowCount(len(result.records))
        for row, record in enumerate(result.records):
            location = record.location
            section_values = (
                (
                    ""
                    if location.section_point is None
                    else section_point_relative_position_label(
                        location.section_point
                    ),
                    _optional_number_text(
                        None
                        if location.section_point is None
                        else location.section_point.local_y
                    ),
                    _optional_number_text(
                        None
                        if location.section_point is None
                        else location.section_point.local_z
                    ),
                )
                if include_section_points
                else ()
            )
            values = (
                _typed_association_text(location.association),
                _optional_identity_text(location.node_id),
                _optional_identity_text(location.element_id),
                _optional_identity_text(location.integration_point),
                _optional_identity_text(location.local_node),
                *section_values,
                (
                    ""
                    if location.region_key is None
                    else region_labels[location.region_key]
                ),
                _averaged_text(location.averaged),
                _number_text(location.coordinates[0]),
                _number_text(location.coordinates[1]),
                _number_text(location.coordinates[2]),
                _number_text(record.value),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record)
                if (
                    column == region_column
                    and location.region_key is not None
                ):
                    item.setToolTip(
                        encode_result_region_key(location.region_key)
                    )
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        if self.table.rowCount():
            self.table.selectRow(0)
        self._displayed_generation = result.materialization_generation
        target = result.request.target
        self.result_summary.setText(
            f"{_probe_target_label(target)} · "
            f"命中 {len(result.records)} 个结果位置 · "
            f"结果版本 {result.materialization_generation}"
        )
        self._probe_history.append(result)
        self.probe_history_list.addItem(
            f"{len(self._probe_history)}. {_probe_target_label(target)} · "
            f"{len(result.records)} 个位置"
        )
        self.probeResultReady.emit(result)

    def record_at(self, row: int) -> ResultQueryRecord:
        if type(row) is not int:
            raise TypeError("row must be an integer")
        if row < 0 or row >= self.table.rowCount():
            raise IndexError(row)
        item = self.table.item(row, 0)
        if item is None:
            raise RuntimeError("probe table row is incomplete")
        record = item.data(Qt.ItemDataRole.UserRole)
        if type(record) is not ResultQueryRecord:
            raise RuntimeError("probe table row lost its typed record")
        return record

    def copy_table(self) -> None:
        rows = [
            [
                self.table.horizontalHeaderItem(column).text()
                for column in range(self.table.columnCount())
            ]
        ]
        rows.extend(
            [
                self.table.item(row, column).text()
                for column in range(self.table.columnCount())
            ]
            for row in range(self.table.rowCount())
        )
        QApplication.clipboard().setText(
            "\n".join("\t".join(row) for row in rows)
        )

    def _field_changed(self, *_args: object) -> None:
        if self._probe_pending:
            return
        self._sync_components()
        self._refresh_availability()

    def _target_changed(self, *_args: object) -> None:
        if self._probe_pending:
            return
        self._refresh_target_controls()

    def _sync_components(self, *, preferred_component: str | None = None) -> None:
        if preferred_component is None:
            current = self.component_combo.currentData()
            if type(current) is str:
                preferred_component = current
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        availability = self.current_availability()
        for component in availability.descriptor.columns:
            self.component_combo.addItem(component, component)
        index = self.component_combo.findData(preferred_component)
        if index < 0:
            index = self.component_combo.findData(
                availability.descriptor.default_component
            )
        self.component_combo.setCurrentIndex(index if index >= 0 else 0)
        self.component_combo.blockSignals(False)

    def _refresh_target_controls(self) -> None:
        kind = self._current_probe_kind()
        if kind is None:
            return
        is_ip = kind is ResultProbeKind.INTEGRATION_POINT
        is_local = kind is ResultProbeKind.ELEMENT_NODE
        self.target_id_label.setText(
            "节点编号" if kind is ResultProbeKind.NODE else "单元编号"
        )
        self.target_id_label.setVisible(True)
        self.target_id_spin.setVisible(True)
        self.integration_point_label.setVisible(is_ip)
        self.integration_point_spin.setVisible(is_ip)
        self.local_node_label.setVisible(is_local)
        self.local_node_spin.setVisible(is_local)
        if not self._probe_pending:
            if kind is ResultProbeKind.NODE and self._node_ids:
                self.target_id_spin.setValue(self._node_ids[0])
            elif kind is not ResultProbeKind.NODE and self._element_ids:
                self.target_id_spin.setValue(self._element_ids[0])

    def _current_probe_kind(self) -> ResultProbeKind | None:
        value = self.target_combo.currentData()
        if type(value) is ResultProbeKind:
            return value
        if type(value) is str:
            try:
                return ResultProbeKind(value)
            except ValueError:
                pass
        return None

    def _refresh_availability(self) -> None:
        try:
            availability = self.current_availability()
        except RuntimeError:
            self.probe_button.setEnabled(False)
            return
        self.availability_label.setText(
            _typed_result_display_availability_text(availability)
        )
        self.probe_button.setEnabled(
            not self._probe_pending
            and availability.state is not FieldState.UNAVAILABLE
        )

    def _prepare_table(self, *, include_section_points: bool) -> None:
        section_headers = (
            ("截面位置", "截面局部 Y", "截面局部 Z")
            if include_section_points
            else ()
        )
        headers = (
            "关联",
            "节点",
            "单元",
            "积分点",
            "局部节点",
            *section_headers,
            "区域",
            "平均状态",
            "X",
            "Y",
            "Z",
            "值",
        )
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(0)


class ResultQueryProbeDialog(QDialog):
    """统一结果查询入口。

    查询和探针是同一个结果查询工作流中的两种操作方式，而不是两个
    结果类别。使用紧凑的操作选择器切换工作区，避免把它们做成标签页。
    """

    queryRequested = Signal(object)
    selectionRequested = Signal(object)
    probeRequested = Signal(object)

    def __init__(
        self,
        query_dialog: TypedResultQueryDialog,
        probe_dialog: ResultProbeDialog,
        *,
        frame_provider: ResultProvider | None = None,
        parent=None,
    ) -> None:
        if type(query_dialog) is not TypedResultQueryDialog:
            raise TypeError("query_dialog must be TypedResultQueryDialog")
        if type(probe_dialog) is not ResultProbeDialog:
            raise TypeError("probe_dialog must be ResultProbeDialog")
        if frame_provider is not None and type(frame_provider) is not ResultProvider:
            raise TypeError("frame_provider must be ResultProvider or None")

        super().__init__(parent)
        self.setWindowTitle("结果查询")
        self.resize(980, 700)
        self.setMinimumWidth(900)
        self.query_dialog = query_dialog
        self.probe_dialog = probe_dialog
        self.frame_provider = frame_provider

        for page in (query_dialog, probe_dialog):
            page.setWindowFlags(Qt.WindowType.Widget)

        layout = QVBoxLayout(self)
        if frame_provider is not None:
            self.frame_summary = ResultFrameSummaryWidget(frame_provider, self)
            layout.addWidget(self.frame_summary)
        else:
            self.frame_summary = None

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("查询方式", self))
        self.mode_combo = QComboBox(self)
        self.mode_combo.setObjectName("resultQueryModeCombo")
        self.mode_combo.addItem("按编号查询", "query")
        self.mode_combo.addItem("定位探针", "probe")
        self.mode_combo.setToolTip(
            "按编号读取结果，或切换为当前结果帧的位置探针"
        )
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.mode_stack = QStackedWidget(self)
        self.mode_stack.setObjectName("resultQueryModeStack")
        self.mode_stack.addWidget(query_dialog)
        self.mode_stack.addWidget(probe_dialog)
        self.mode_combo.currentIndexChanged.connect(
            self.mode_stack.setCurrentIndex
        )
        layout.addWidget(self.mode_stack, 1)

        query_dialog.queryRequested.connect(self.queryRequested.emit)
        query_dialog.selectionRequested.connect(
            self.selectionRequested.emit
        )
        probe_dialog.probeRequested.connect(self.probeRequested.emit)


class ResultPathDialog(QDialog):
    """Extract a scalar result along two points or a mesh-node path."""

    pathRequested = Signal(object)

    def __init__(
        self,
        provider: ResultProvider,
        *,
        current_selection: ScalarFieldSelection | None = None,
        parent=None,
    ) -> None:
        if type(provider) is not ResultProvider:
            raise TypeError("provider must be ResultProvider")
        fields = visible_result_fields(provider.catalog().fields)
        if not fields:
            raise ValueError("path extraction requires a non-empty result catalog")
        super().__init__(parent)
        self.setWindowTitle("结果路径")
        self.resize(760, 620)
        self.setMinimumWidth(680)
        self._provider = provider
        self._fields = fields
        self._pending = False
        self._result: ResultPathResult | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        self.field_combo = QComboBox(self)
        self.component_combo = QComboBox(self)
        for availability in fields:
            self.field_combo.addItem(
                _typed_result_display_field_label(
                    availability,
                    result_provider_section_point_labels(provider),
                ),
                availability.key,
            )
        form.addRow("场变量：", self.field_combo)
        form.addRow("分量：", self.component_combo)
        layout.addLayout(form)

        path_form = QFormLayout()
        configure_form_layout(path_form)
        self.path_mode_combo = QComboBox(self)
        self.path_mode_combo.addItem("两点直线路径", "points")
        self.path_mode_combo.addItem("沿节点路径", "edge")
        path_form.addRow("路径类型：", self.path_mode_combo)
        self.node_path_edit = QLineEdit(self)
        self.node_path_edit.setPlaceholderText("例如：1, 2, 3, 4")
        path_form.addRow("节点路径：", self.node_path_edit)
        coordinate_host = QWidget(self)
        coordinate_grid = QGridLayout(coordinate_host)
        coordinate_grid.setContentsMargins(0, 0, 0, 0)
        self.start_spins = self._coordinate_spins(coordinate_host)
        self.end_spins = self._coordinate_spins(coordinate_host)
        coordinate_grid.addWidget(QLabel("起点", coordinate_host), 0, 0)
        coordinate_grid.addWidget(QLabel("终点", coordinate_host), 1, 0)
        for column, (start, end) in enumerate(
            zip(self.start_spins, self.end_spins, strict=True),
            start=1,
        ):
            coordinate_grid.addWidget(start, 0, column)
            coordinate_grid.addWidget(end, 1, column)
        path_form.addRow("坐标 XYZ：", coordinate_host)
        self.sample_count_spin = QSpinBox(self)
        self.sample_count_spin.setRange(2, 10000)
        self.sample_count_spin.setValue(20)
        path_form.addRow("采样点数：", self.sample_count_spin)
        layout.addLayout(path_form)

        command_row = QHBoxLayout()
        self.extract_button = QPushButton("提取路径", self)
        self.copy_button = QPushButton("复制数据", self)
        self.extract_button.clicked.connect(self.request_path)
        self.copy_button.clicked.connect(self.copy_table)
        command_row.addWidget(self.extract_button)
        command_row.addWidget(self.copy_button)
        command_row.addStretch(1)
        layout.addLayout(command_row)

        self.status_label = QLabel("尚未提取路径", self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(
            ("序号", "距离", "X", "Y", "Z", "值")
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)

        preferred = current_selection or provider.catalog().default_selection
        if preferred is not None:
            field_index = self.field_combo.findData(preferred.field_key)
            if field_index >= 0:
                self.field_combo.setCurrentIndex(field_index)
        self._sync_components(
            None if preferred is None else preferred.component
        )
        self._refresh_path_mode()
        self.field_combo.currentIndexChanged.connect(
            lambda _index: self._sync_components()
        )
        self.path_mode_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_path_mode()
        )

    def _coordinate_spins(self, parent: QWidget) -> tuple[QDoubleSpinBox, ...]:
        spins: list[QDoubleSpinBox] = []
        for _index in range(3):
            spin = QDoubleSpinBox(parent)
            spin.setRange(-1.0e12, 1.0e12)
            spin.setDecimals(8)
            spin.setSingleStep(0.1)
            spins.append(spin)
        return tuple(spins)

    def _sync_components(self, preferred: str | None = None) -> None:
        availability = self._fields[self.field_combo.currentIndex()]
        current = self.component_combo.currentData()
        candidate = preferred or (current if isinstance(current, str) else None)
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        for component in availability.descriptor.columns:
            self.component_combo.addItem(component, component)
        index = self.component_combo.findData(candidate)
        if index < 0:
            index = self.component_combo.findData(availability.descriptor.default_component)
        self.component_combo.setCurrentIndex(index if index >= 0 else 0)
        self.component_combo.blockSignals(False)

    def _refresh_path_mode(self) -> None:
        is_edge = self.path_mode_combo.currentData() == "edge"
        self.node_path_edit.setVisible(is_edge)
        for spin in (*self.start_spins, *self.end_spins):
            spin.setVisible(not is_edge)

    def current_request(self) -> ResultPathRequest:
        availability = self._fields[self.field_combo.currentIndex()]
        component = self.component_combo.currentData()
        if not isinstance(component, str) or not component:
            raise ValueError("请选择路径结果分量")
        if self.path_mode_combo.currentData() == "edge":
            raw_ids = self.node_path_edit.text().replace("，", ",")
            try:
                node_ids = tuple(
                    int(value.strip())
                    for value in raw_ids.split(",")
                    if value.strip()
                )
            except ValueError as error:
                raise ValueError("节点路径必须是逗号分隔的正整数") from error
            if len(node_ids) < 2 or any(value <= 0 for value in node_ids):
                raise ValueError("节点路径至少需要两个正整数节点编号")
            start = tuple(float(spin.value()) for spin in self.start_spins)
            end = tuple(float(spin.value()) for spin in self.end_spins)
        else:
            node_ids = ()
            start = tuple(float(spin.value()) for spin in self.start_spins)
            end = tuple(float(spin.value()) for spin in self.end_spins)
        return ResultPathRequest(
            ScalarFieldSelection(availability.key, component),
            start,
            end,
            int(self.sample_count_spin.value()),
            node_ids,
        )

    def request_path(self) -> None:
        if self._pending:
            return
        try:
            request = self.current_request()
        except (RuntimeError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "结果路径", str(error))
            return
        self.pathRequested.emit(request)

    def set_pending(self, pending: bool) -> None:
        if type(pending) is not bool:
            raise TypeError("pending must be a bool")
        self._pending = pending
        for widget in (
            self.field_combo,
            self.component_combo,
            self.path_mode_combo,
            self.node_path_edit,
            *self.start_spins,
            *self.end_spins,
            self.sample_count_spin,
            self.extract_button,
        ):
            widget.setEnabled(not pending)
        if pending:
            self.status_label.setText("正在提取结果路径……")

    def set_message(self, message: str) -> None:
        self.status_label.setText(str(message).strip() or "结果路径未完成")

    def set_path_result(self, result: ResultPathResult) -> None:
        if type(result) is not ResultPathResult:
            raise TypeError("result must be ResultPathResult")
        if result.source != self._provider.source:
            raise ValueError("path result source must match dialog provider")
        self._result = result
        self.table.setRowCount(len(result.samples))
        for row, sample in enumerate(result.samples):
            values = (
                str(row + 1),
                f"{sample.distance:.8g}",
                f"{sample.coordinates[0]:.8g}",
                f"{sample.coordinates[1]:.8g}",
                f"{sample.coordinates[2]:.8g}",
                f"{sample.value:.8g}",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.status_label.setText(
            f"已提取 {len(result.samples)} 个点 · "
            f"分量：{result.request.selection.component}"
        )
        self.copy_button.setEnabled(True)

    def copy_table(self) -> None:
        headers = [
            self.table.horizontalHeaderItem(column).text()
            for column in range(self.table.columnCount())
        ]
        rows = [headers]
        rows.extend(
            [
                self.table.item(row, column).text()
                for column in range(self.table.columnCount())
            ]
            for row in range(self.table.rowCount())
        )
        QApplication.clipboard().setText("\n".join("\t".join(row) for row in rows))


class ResultAnimationDialog(QDialog):
    """设置结果帧范围和播放方式。"""

    applyRequested = Signal(object)
    exportRequested = Signal(object)

    def __init__(
        self,
        frame_indices: tuple[int, ...],
        *,
        current_frame_index: int,
        start_frame: int | None = None,
        end_frame: int | None = None,
        interval_ms: int = 250,
        loop: bool = True,
        frame_step: int | None = None,
        sampling_mode: str | None = None,
        time_interval: float | None = None,
        playback_mode: str | None = None,
        playback_speed: float | None = None,
        first_hold_ms: int | None = None,
        last_hold_ms: int | None = None,
        lock_deformation_scale: bool | None = None,
        lock_contour_range: bool | None = None,
        sync_annotations: bool | None = None,
        parent=None,
    ) -> None:
        if not frame_indices:
            raise ValueError("animation requires at least one frame")
        if any(type(index) is not int for index in frame_indices):
            raise TypeError("frame_indices must contain integers")
        if len(set(frame_indices)) != len(frame_indices):
            raise ValueError("frame_indices must be unique")
        if current_frame_index not in frame_indices:
            raise ValueError("current_frame_index must be in frame_indices")

        super().__init__(parent)
        self.setWindowTitle("动画")
        self.setMinimumWidth(520)
        self._frame_indices = tuple(frame_indices)
        self._extended_settings = any(
            value is not None
            for value in (
                frame_step,
                sampling_mode,
                time_interval,
                playback_mode,
                playback_speed,
                first_hold_ms,
                last_hold_ms,
                lock_deformation_scale,
                lock_contour_range,
                sync_annotations,
            )
        )
        frame_step = 0 if frame_step is None else int(frame_step)
        sampling_mode = "frame" if sampling_mode is None else str(sampling_mode)
        time_interval = 0.0 if time_interval is None else float(time_interval)
        playback_mode = (
            ("loop" if loop else "once")
            if playback_mode is None
            else str(playback_mode)
        )
        playback_speed = 1.0 if playback_speed is None else float(playback_speed)
        first_hold_ms = 0 if first_hold_ms is None else int(first_hold_ms)
        last_hold_ms = 0 if last_hold_ms is None else int(last_hold_ms)
        lock_deformation_scale = (
            False if lock_deformation_scale is None else bool(lock_deformation_scale)
        )
        lock_contour_range = (
            False if lock_contour_range is None else bool(lock_contour_range)
        )
        sync_annotations = True if sync_annotations is None else bool(sync_annotations)

        start = (
            self._frame_indices[0]
            if start_frame is None
            else int(start_frame)
        )
        end = (
            self._frame_indices[-1]
            if end_frame is None
            else int(end_frame)
        )
        if start not in self._frame_indices or end not in self._frame_indices:
            raise ValueError("animation range must use existing frame indices")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        self.start_frame_combo = self._create_frame_combo()
        self.end_frame_combo = self._create_frame_combo()
        self._select_frame(self.start_frame_combo, start)
        self._select_frame(self.end_frame_combo, end)
        form.addRow("起始帧", self.start_frame_combo)
        form.addRow("结束帧", self.end_frame_combo)

        self.interval_spin = QSpinBox(self)
        self.interval_spin.setRange(20, 5000)
        self.interval_spin.setSingleStep(10)
        self.interval_spin.setSuffix(" ms")
        self.interval_spin.setValue(int(interval_ms))
        form.addRow("帧间隔", self.interval_spin)

        self.frame_step_spin = QSpinBox(self)
        self.frame_step_spin.setRange(0, 1000000)
        self.frame_step_spin.setSpecialValueText("自动")
        self.frame_step_spin.setValue(frame_step)
        self.frame_step_spin.setToolTip(
            "自动：按约 120 个显示帧播放；结果帧仍然全部保留"
        )
        form.addRow("帧步长", self.frame_step_spin)

        self.sampling_combo = QComboBox(self)
        self.sampling_combo.addItem("按帧采样", "frame")
        self.sampling_combo.addItem("按物理时间采样", "time")
        self.sampling_combo.setCurrentIndex(
            max(0, self.sampling_combo.findData(sampling_mode))
        )
        form.addRow("采样方式", self.sampling_combo)

        self.time_interval_spin = QDoubleSpinBox(self)
        self.time_interval_spin.setRange(0.0, 1.0e12)
        self.time_interval_spin.setDecimals(8)
        self.time_interval_spin.setValue(time_interval)
        self.time_interval_spin.setSuffix(" s")
        form.addRow("时间间隔", self.time_interval_spin)

        self.playback_mode_combo = QComboBox(self)
        for label, key in (
            ("单次播放", "once"),
            ("循环播放", "loop"),
            ("往返播放", "pingpong"),
            ("反向播放", "reverse"),
        ):
            self.playback_mode_combo.addItem(label, key)
        self.playback_mode_combo.setCurrentIndex(
            max(0, self.playback_mode_combo.findData(playback_mode))
        )
        form.addRow("播放方式", self.playback_mode_combo)

        self.playback_speed_spin = QDoubleSpinBox(self)
        self.playback_speed_spin.setRange(0.1, 10.0)
        self.playback_speed_spin.setDecimals(1)
        self.playback_speed_spin.setSingleStep(0.1)
        self.playback_speed_spin.setValue(playback_speed)
        self.playback_speed_spin.setSuffix(" ×")
        form.addRow("播放速度", self.playback_speed_spin)

        self.first_hold_spin = QSpinBox(self)
        self.last_hold_spin = QSpinBox(self)
        for spin, value in (
            (self.first_hold_spin, first_hold_ms),
            (self.last_hold_spin, last_hold_ms),
        ):
            spin.setRange(0, 60000)
            spin.setSingleStep(100)
            spin.setSuffix(" ms")
            spin.setValue(value)
        form.addRow("首帧停留", self.first_hold_spin)
        form.addRow("末帧停留", self.last_hold_spin)

        self.loop_checkbox = QCheckBox("播放到结束后循环", self)
        self.loop_checkbox.setChecked(bool(loop))
        layout.addLayout(form)
        layout.addWidget(self.loop_checkbox)

        options_group = QGroupBox("动画显示", self)
        options_layout = QVBoxLayout(options_group)
        self.lock_deformation_checkbox = QCheckBox("固定变形显示比例", options_group)
        self.lock_deformation_checkbox.setChecked(lock_deformation_scale)
        self.lock_contour_checkbox = QCheckBox("固定云图范围", options_group)
        self.lock_contour_checkbox.setChecked(lock_contour_range)
        self.sync_annotations_checkbox = QCheckBox("同步图例、标题和状态信息", options_group)
        self.sync_annotations_checkbox.setChecked(sync_annotations)
        for checkbox in (
            self.lock_deformation_checkbox,
            self.lock_contour_checkbox,
            self.sync_annotations_checkbox,
        ):
            options_layout.addWidget(checkbox)
        layout.addWidget(options_group)

        self.info_label = QLabel(
            f"当前帧：增量 {current_frame_index}；共 {len(frame_indices)} 帧",
            self,
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.button_box = _dialog_buttons(self)
        self.button_box.button(
            QDialogButtonBox.StandardButton.Apply
        ).clicked.connect(self.apply)
        self.button_box.accepted.connect(self.accept_with_apply)
        self.button_box.rejected.connect(self.reject)
        self.export_button = QPushButton("导出动画…", self)
        self.export_button.setToolTip("按当前帧范围和播放设置导出 GIF 或视频")
        self.export_button.clicked.connect(self._request_export)
        export_row = QHBoxLayout()
        export_row.addWidget(self.export_button)
        export_row.addStretch(1)
        layout.addLayout(export_row)
        layout.addWidget(self.button_box)

    def _create_frame_combo(self) -> QComboBox:
        combo = QComboBox(self)
        for frame_index in self._frame_indices:
            combo.addItem(f"增量 {frame_index}", frame_index)
        return combo

    @staticmethod
    def _select_frame(combo: QComboBox, frame_index: int) -> None:
        index = combo.findData(frame_index)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def settings(self) -> dict[str, Any]:
        start = self.start_frame_combo.currentData()
        end = self.end_frame_combo.currentData()
        if type(start) is not int or type(end) is not int:
            raise RuntimeError("animation frame selection is invalid")
        start_position = self._frame_indices.index(start)
        end_position = self._frame_indices.index(end)
        if start_position > end_position:
            raise ValueError("起始帧必须早于或等于结束帧")
        legacy = {
            "start_frame": start,
            "end_frame": end,
            "interval_ms": int(self.interval_spin.value()),
            "loop": self.loop_checkbox.isChecked(),
        }
        if not self._extended_settings:
            return legacy
        playback_mode = str(self.playback_mode_combo.currentData())
        return {
            **legacy,
            "frame_step": int(self.frame_step_spin.value()),
            "sampling_mode": str(self.sampling_combo.currentData()),
            "time_interval": float(self.time_interval_spin.value()),
            "playback_mode": playback_mode,
            "playback_speed": float(self.playback_speed_spin.value()),
            "first_hold_ms": int(self.first_hold_spin.value()),
            "last_hold_ms": int(self.last_hold_spin.value()),
            "lock_deformation_scale": self.lock_deformation_checkbox.isChecked(),
            "lock_contour_range": self.lock_contour_checkbox.isChecked(),
            "sync_annotations": self.sync_annotations_checkbox.isChecked(),
        }

    def apply(self) -> bool:
        try:
            settings = self.settings()
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "动画", str(error))
            return False
        self.applyRequested.emit(settings)
        return True

    def _request_export(self) -> None:
        try:
            settings = self.settings()
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "动画", str(error))
            return
        self.exportRequested.emit(settings)

    def accept_with_apply(self) -> None:
        if self.apply():
            self.accept()


def _dynamic_frame_values(result: Any) -> dict[str, Any]:
    """Project one result frame into the compact query-context summary."""

    outputs = dict(result.outputs)
    displacement = np.asarray(result.U, dtype=float)
    dynamic_data = getattr(result, "dynamic_data", None)
    velocity = _dynamic_vector_value(
        getattr(dynamic_data, "velocity", None),
        outputs,
        "velocity",
        displacement.size,
    )
    acceleration = _dynamic_vector_value(
        getattr(dynamic_data, "acceleration", None),
        outputs,
        "acceleration",
        displacement.size,
    )
    energy = getattr(dynamic_data, "energy", None)

    def energy_value(attribute: str, output_key: str) -> float | None:
        if energy is not None:
            return getattr(energy, attribute)
        value = outputs.get(output_key)
        if value is None and output_key == "internal_energy":
            value = outputs.get("strain_energy")
        return None if value is None else float(value)

    return {
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "time": (
            float(getattr(dynamic_data, "step_time"))
            if dynamic_data is not None
            else float(outputs.get("time", outputs.get("step_time", 0.0)))
        ),
        "time_increment": (
            float(getattr(dynamic_data, "time_increment"))
            if dynamic_data is not None
            else outputs.get("time_increment")
        ),
        "kinetic_energy": energy_value("kinetic", "kinetic_energy"),
        "internal_energy": energy_value("internal", "internal_energy"),
        "external_work": energy_value("external_work", "external_work"),
        "damping_dissipation": energy_value(
            "damping_dissipation",
            "damping_dissipation",
        ),
        "total_energy": energy_value("total", "total_energy"),
        "energy_balance_error": energy_value(
            "balance_error",
            "energy_balance_error",
        ),
    }


def _dynamic_vector_value(
    typed_value: Any,
    outputs: Mapping[str, Any],
    name: str,
    size: int,
) -> np.ndarray:
    raw = typed_value if typed_value is not None else outputs.get(name)
    if raw is None:
        return np.zeros(size, dtype=float)
    vector = np.asarray(raw, dtype=float)
    if vector.ndim != 1 or vector.size != size:
        raise ValueError(f"动力学结果 {name} 的自由度长度不匹配")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"动力学结果 {name} 含有非有限数")
    return vector


def _dynamic_optional_number(value: Any) -> str:
    return "—" if value is None else _dynamic_number(value)


def _dynamic_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.6e}"


class ResultFrameSummaryWidget(QWidget):
    """显示结果查询所绑定的当前步、帧和时间上下文。"""

    def __init__(self, provider: ResultProvider, parent=None) -> None:
        if type(provider) is not ResultProvider:
            raise TypeError("provider must be ResultProvider")
        result = provider.model_result
        if result is None:
            raise ValueError("当前结果没有可显示的结果帧")

        super().__init__(parent)
        self.setObjectName("resultFrameSummary")
        group = QGroupBox("当前结果", self)
        group_layout = QGridLayout(group)
        group_layout.setContentsMargins(12, 8, 12, 8)
        group_layout.setHorizontalSpacing(18)
        group_layout.setVerticalSpacing(4)

        frame_key = provider.frame_key
        frame_text = (
            "最终帧"
            if frame_key is None
            else f"增量 {frame_key.frame_index}"
        )
        values = _dynamic_frame_values(result)
        self.step_label = QLabel(provider.source.step_name, group)
        self.frame_label = QLabel(frame_text, group)
        self.time_label = QLabel(
            _dynamic_number(values["time"]),
            group,
        )
        self.time_increment_label = QLabel(
            _dynamic_optional_number(values["time_increment"]),
            group,
        )
        for label in (
            self.step_label,
            self.frame_label,
            self.time_label,
            self.time_increment_label,
        ):
            label.setObjectName("resultFrameSummaryValue")

        group_layout.addWidget(QLabel("分析步", group), 0, 0)
        group_layout.addWidget(self.step_label, 0, 1)
        group_layout.addWidget(QLabel("帧", group), 0, 2)
        group_layout.addWidget(self.frame_label, 0, 3)
        group_layout.addWidget(QLabel("步时间", group), 1, 0)
        group_layout.addWidget(self.time_label, 1, 1)
        group_layout.addWidget(QLabel("时间增量", group), 1, 2)
        group_layout.addWidget(self.time_increment_label, 1, 3)
        group_layout.setColumnStretch(1, 1)
        group_layout.setColumnStretch(3, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(group)

        if str(getattr(result.step, "procedure", "")).casefold() == "dynamic":
            dynamic_group = QGroupBox("当前帧摘要", self)
            dynamic_layout = QGridLayout(dynamic_group)
            dynamic_layout.setContentsMargins(12, 8, 12, 8)
            dynamic_layout.setHorizontalSpacing(18)
            dynamic_layout.setVerticalSpacing(4)
            summary_values = (
                ("最大位移", self._maximum(values["displacement"])),
                ("最大速度", self._maximum(values["velocity"])),
                ("最大加速度", self._maximum(values["acceleration"])),
                ("动能", _dynamic_optional_number(values["kinetic_energy"])),
                ("内能", _dynamic_optional_number(values["internal_energy"])),
                ("总能量", _dynamic_optional_number(values["total_energy"])),
            )
            for index, (label_text, value_text) in enumerate(summary_values):
                row = index // 3
                column = (index % 3) * 2
                dynamic_layout.addWidget(QLabel(label_text, dynamic_group), row, column)
                dynamic_layout.addWidget(QLabel(value_text, dynamic_group), row, column + 1)
            layout.addWidget(dynamic_group)

    @staticmethod
    def _maximum(value: Any) -> str:
        if value is None:
            return "—"
        array = np.asarray(value, dtype=float)
        if array.size == 0:
            return "—"
        return _dynamic_number(np.max(np.abs(array)))


def _typed_query_association_matches(
    mode: FieldAssociation,
    association: FieldAssociation,
) -> bool:
    if mode is FieldAssociation.NODE:
        if association is FieldAssociation.NODE:
            return True
        if association is FieldAssociation.ELEMENT_NODE:
            return True
        if association is FieldAssociation.NODE_REGION:
            return True
        return association is FieldAssociation.RESOLVED_NODAL
    if mode is FieldAssociation.ELEMENT:
        if association is FieldAssociation.ELEMENT:
            return True
        if association is FieldAssociation.INTEGRATION_POINT:
            return True
        return association is FieldAssociation.ELEMENT_NODE
    raise TypeError("mode must be a query association")


def _typed_association_text(association: FieldAssociation) -> str:
    labels = {
        FieldAssociation.NODE: "节点",
        FieldAssociation.ELEMENT_NODE: "单元节点",
        FieldAssociation.ELEMENT: "单元",
        FieldAssociation.INTEGRATION_POINT: "积分点",
        FieldAssociation.NODE_REGION: "节点区域",
        FieldAssociation.RESOLVED_NODAL: "节点平均值",
    }
    return labels[association]


def _probe_target_label(target: ResultProbeTarget) -> str:
    if target.kind is ResultProbeKind.NODE:
        return f"节点 {target.node_id}"
    if target.kind is ResultProbeKind.ELEMENT:
        return f"单元 {target.element_id}"
    if target.kind is ResultProbeKind.INTEGRATION_POINT:
        return (
            f"单元 {target.element_id} · "
            f"积分点 {target.integration_point}"
        )
    return f"单元 {target.element_id} · 局部节点 {target.local_node}"


def _parse_typed_query_ids(
    text: str,
    valid_ids: tuple[int, ...],
) -> tuple[int, ...]:
    if type(text) is not str:
        raise TypeError("text must be a string")
    if type(valid_ids) is not tuple or any(
        type(value) is not int for value in valid_ids
    ):
        raise TypeError("valid_ids must be a tuple of integers")
    if not text.strip():
        return ()

    valid = frozenset(valid_ids)
    parsed: list[int] = []
    for token in re.split(r"[\s,，;；]+", text.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(-?\d+)\s*[-~～]\s*(-?\d+)", token)
        if match is None:
            try:
                candidates = (int(token),)
            except ValueError as error:
                raise ValueError(
                    f"无法识别的有限元编号：{token}"
                ) from error
        else:
            first, last = (int(value) for value in match.groups())
            step = 1 if last >= first else -1
            candidates = range(first, last + step, step)
        for candidate in candidates:
            if candidate not in valid:
                raise ValueError(f"有限元编号不存在：{candidate}")
            if candidate not in parsed:
                parsed.append(candidate)
    return tuple(parsed)


def _typed_field_label(
    availability: FieldAvailability,
    section_point_labels: Mapping[int, str] | None = None,
) -> str:
    if availability.state is FieldState.READY:
        descriptor = availability.descriptor
        if result_field_is_beam_section(descriptor.field_id):
            position = result_field_position_label(
                descriptor.field_id,
                section_point_labels=section_point_labels,
            )
            return f"应力 S（{position}）"
        return _TYPED_RESULT_FIELD_LABELS.get(
            descriptor.label_key,
            descriptor.label_key,
        )
    return _typed_result_display_field_label(
        availability,
        section_point_labels,
    )


_TYPED_RESULT_FIELD_LABELS = {
    "result.field.u.node": "位移 U",
    "result.field.ur.node": "转角 UR",
    "result.field.rf.node": "反力 RF",
    "result.field.rm.node": "反力矩 RM",
    "result.field.le.centroid": "对数应变 LE（单元质心）",
    "result.field.s.integration_point": "应力 S（积分点）",
    "result.field.s.centroid": "应力 S（单元质心）",
    "result.field.s.element_nodal": "应力 S（单元节点，未平均）",
    "result.field.s.resolved_nodal": "应力 S（区域内节点平均）",
    "result.field.e.integration_point": "应变 E（积分点）",
    "result.field.e.centroid": "应变 E（单元质心）",
    "result.field.e.element_nodal": "应变 E（单元节点，未平均）",
    "result.field.e.resolved_nodal": "应变 E（区域内节点平均）",
    "result.field.peeq.integration_point": "塑性应变 PEEQ（积分点）",
    "result.field.peeq.centroid": "塑性应变 PEEQ（单元质心）",
    "result.field.peeq.element_nodal": "塑性应变 PEEQ（单元节点，未平均）",
    "result.field.peeq.resolved_nodal": "塑性应变 PEEQ（区域内节点平均）",
}
_TYPED_RESULT_FIELD_STATE_LABELS = {
    FieldState.READY: "就绪",
    FieldState.LAZY: "按需加载",
    FieldState.UNAVAILABLE: "不可用",
}


def _typed_result_display_field_label(
    availability: FieldAvailability,
    section_point_labels: Mapping[int, str] | None = None,
) -> str:
    descriptor = availability.descriptor
    if result_field_is_beam_section(descriptor.field_id):
        position = result_field_position_label(
            descriptor.field_id,
            section_point_labels=section_point_labels,
        )
        label = f"应力 S（{position}）"
    else:
        label = _TYPED_RESULT_FIELD_LABELS.get(
            descriptor.label_key,
            descriptor.label_key,
        )
    return (
        f"{label}"
        f"（{_TYPED_RESULT_FIELD_STATE_LABELS[availability.state]}）"
    )


def _typed_result_display_availability_text(
    availability: FieldAvailability,
) -> str:
    if availability.state is FieldState.READY:
        return "已就绪"
    if availability.state is FieldState.LAZY:
        return "待物化；应用后由外层命令加载"
    if availability.diagnostics:
        return "\n".join(
            diagnostic.message for diagnostic in availability.diagnostics
        )
    return "不可用"


def _validate_typed_display_selection(
    catalog: ResultCatalog,
    selection: ScalarFieldSelection,
) -> None:
    if type(selection) is not ScalarFieldSelection:
        raise TypeError("current_selection must be ScalarFieldSelection")
    matches = tuple(
        availability
        for availability in catalog.fields
        if (
            availability.key == selection.field_key
            and result_field_is_visible(availability)
        )
    )
    if len(matches) != 1:
        raise ValueError(
            "current_selection must reference exactly one catalog field"
        )
    if selection.component not in matches[0].descriptor.columns:
        raise ValueError(
            "current_selection component is outside the field descriptor"
        )


def _validate_typed_display_options(
    *,
    shape_mode: str,
    contour_enabled: bool,
    selection: ScalarFieldSelection,
    scale_mode: str,
    scale_value: float,
    overlay_undeformed: bool,
    show_edges: bool,
) -> None:
    if type(shape_mode) is not str:
        raise TypeError("shape_mode must be a string")
    if shape_mode not in {"undeformed", "deformed"}:
        raise ValueError("shape_mode must be undeformed or deformed")
    if type(contour_enabled) is not bool:
        raise TypeError("contour_enabled must be a boolean")
    if type(selection) is not ScalarFieldSelection:
        raise TypeError("selection must be ScalarFieldSelection")
    if type(scale_mode) is not str:
        raise TypeError("scale_mode must be a string")
    if scale_mode not in {"auto", "real", "custom"}:
        raise ValueError("scale_mode must be auto, real, or custom")
    if type(scale_value) is not float:
        raise TypeError("scale_value must be a float")
    if not isfinite(scale_value) or scale_value < 0.0:
        raise ValueError("scale_value must be finite and non-negative")
    if type(overlay_undeformed) is not bool:
        raise TypeError("overlay_undeformed must be a boolean")
    if type(show_edges) is not bool:
        raise TypeError("show_edges must be a boolean")


def _optional_identity_text(value: int | None) -> str:
    return "" if value is None else str(value)


def _optional_number_text(value: float | None) -> str:
    return "" if value is None else _number_text(value)


def _averaged_text(value: bool | None) -> str:
    if value is None:
        return "缺失"
    return "是" if value else "否"


def _number_text(value: float) -> str:
    return f"{value:.8g}"


def _dialog_buttons(parent: QWidget) -> QDialogButtonBox:
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Apply
        | QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel,
        parent=parent,
    )
    buttons.button(QDialogButtonBox.StandardButton.Apply).setText("应用")
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
    buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
    return buttons
