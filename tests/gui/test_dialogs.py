from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

from fem_gui.dialogs import CompactDoubleSpinBox
from fem_gui.postprocessing_dialogs import (
    ContourSettingsDialog,
    DisplaySettingsDialog,
)
from fem_gui.symbol_dialog import SymbolSettingsDialog
from fem_gui.viewport_background import ViewportBackgroundSettings
from fem_gui.viewport_background_dialog import ViewportBackgroundDialog
from fem_gui.visualization.colormaps import ABAQUS_RAINBOW
from fem_gui.visualization.contour_rendering import (
    CONTOUR_EDGE_FEATURE,
    CONTOUR_RENDER_FILLED,
)
from fem_gui.visualization.symbols import SymbolSettings


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_compact_number_input_hides_only_insignificant_trailing_zeroes():
    _application()
    editor = CompactDoubleSpinBox()
    editor.setDecimals(8)

    editor.setValue(1.0)
    assert editor.text() == "1.00"

    editor.setValue(0.0001)
    assert editor.text() == "0.0001"
    assert editor.value() == 0.0001


def test_contour_display_and_symbol_dialogs_round_trip_settings():
    _application()
    contour = ContourSettingsDialog(
        {
            "manual": False,
            "minimum": 0.0,
            "maximum": 1.0,
            "automatic_minimum": -2.5,
            "automatic_maximum": 8.75,
            "colormap": "viridis",
            "style": "continuous",
            "levels": 16,
            "show_minimum": True,
            "show_maximum": True,
        }
    )
    assert contour.settings()["levels"] == 16
    assert contour.settings()["style"] == "continuous"
    assert contour.layout().itemAt(0).widget() is contour.render_group
    assert contour.layout().itemAt(1).widget() is contour.range_group
    assert contour.render_group.title() == "渲染"
    assert contour.mode.currentText() == "连续"
    assert not contour.levels_row.isEnabled()
    assert contour.minimum.value() == -2.5
    assert contour.maximum.value() == 8.75
    assert not contour.minimum.isEnabled()
    assert not contour.maximum.isEnabled()
    assert contour.show_minimum.parent() is contour.range_group
    assert contour.show_maximum.parent() is contour.range_group

    display = DisplaySettingsDialog(
        {
            "edge_mode": CONTOUR_EDGE_FEATURE,
            "edges": True,
            "edge_style": "dashed",
            "edge_width": 2.5,
            "number_format": "engineering",
            "decimals": 4,
            "orientation": "horizontal",
            "legend_font": "Times New Roman",
            "legend_font_size": 16,
            "legend": True,
            "show_ids": True,
            "show_coordinate_system": False,
        }
    )
    display_settings = display.settings()
    assert display_settings["edge_mode"] == CONTOUR_EDGE_FEATURE
    assert display_settings["edge_style"] == "dashed"
    assert display_settings["edge_width"] == 2.5
    assert display_settings["number_format"] == "engineering"
    assert display_settings["decimals"] == 4
    assert display_settings["orientation"] == "horizontal"
    assert display_settings["legend_font"] == "Times New Roman"
    assert display_settings["legend_font_size"] == 16
    assert display_settings["show_ids"]
    assert not display_settings["show_coordinate_system"]

    labels = {
        label.text() for label in contour.findChildren(QLabel)
    }
    assert "阈值：" in labels
    assert "级数：" in labels
    assert "样式：" in labels
    assert "色带：" in labels
    assert "模式：" in labels
    assert "云图样式：" not in labels
    assert "渲染模式：" not in labels
    assert "色带级数：" not in labels
    assert "节点平均阈值：" not in labels
    assert contour.levels_slider.orientation() == Qt.Orientation.Horizontal
    assert contour.levels.minimum() == 4
    assert contour.levels.maximum() == 48
    assert contour.levels_slider.minimum() == 4
    assert contour.levels_slider.maximum() == 48
    assert contour.averaging_threshold_slider.orientation() == (
        Qt.Orientation.Horizontal
    )
    assert contour.averaging_threshold.decimals() == 0
    assert contour.averaging_threshold_slider.minimum() == 0
    assert contour.averaging_threshold_slider.maximum() == 100
    assert contour.levels.size() == contour.averaging_threshold.size()
    assert contour.levels.width() == 60
    assert contour.levels.alignment() == Qt.AlignmentFlag.AlignCenter
    assert contour.averaging_threshold.alignment() == (
        Qt.AlignmentFlag.AlignCenter
    )
    assert type(contour.levels_slider).__name__ == "_ThinHorizontalSlider"
    assert (
        type(contour.averaging_threshold_slider).__name__
        == "_ThinHorizontalSlider"
    )
    levels_row, _role = contour.form.getWidgetPosition(
        contour.levels_row
    )
    threshold_row, _role = contour.form.getWidgetPosition(
        contour.averaging_threshold_row
    )
    assert levels_row < threshold_row

    contour.levels_slider.setValue(24)
    contour.averaging_threshold_slider.setValue(83)
    assert contour.settings()["levels"] == 24
    assert contour.settings()["averaging_threshold"] == 83.0
    contour.manual_range.setChecked(True)
    assert contour.minimum.isEnabled()
    assert contour.maximum.isEnabled()
    contour.mode.setCurrentIndex(contour.mode.findData("segmented"))
    assert contour.levels_row.isEnabled()

    settings = SymbolSettings(step_name="Static-1", show_values=True, scale=1.5)
    symbols = SymbolSettingsDialog(settings, ("Static-1",))
    assert symbols.settings().step_name == "Static-1"
    assert symbols.settings().show_values
    assert symbols.settings().scale == 1.5


def test_contour_dialog_defaults_to_abaqus_rainbow():
    _application()
    contour = ContourSettingsDialog({})

    assert contour.colormap.currentText() == "Abaqus 彩虹"
    assert contour.settings()["colormap"] == ABAQUS_RAINBOW
    assert contour.auto_range.isChecked()
    assert contour.shaded_style.isChecked()


def test_contour_and_display_dialogs_split_render_and_edge_modes():
    _application()
    contour = ContourSettingsDialog(
        {
            "render_mode": CONTOUR_RENDER_FILLED,
        }
    )
    display = DisplaySettingsDialog(
        {
            "edge_mode": CONTOUR_EDGE_FEATURE,
            "edges": True,
        }
    )

    settings = contour.settings()
    assert settings["render_mode"] == CONTOUR_RENDER_FILLED
    assert "edge_mode" not in settings
    assert display.settings()["edge_mode"] == CONTOUR_EDGE_FEATURE
    assert display.settings()["edges"]


def test_viewport_background_dialog_supports_presets_and_live_preview():
    _application()
    dialog = ViewportBackgroundDialog(ViewportBackgroundSettings(), False)
    previews = []
    dialog.previewRequested.connect(previews.append)

    dialog.preset_combo.setCurrentText("白色")

    assert dialog.settings().style == "solid"
    assert dialog.settings().bottom_color == "#ffffff"
    assert not dialog.settings().is_dark
    assert previews[-1].bottom_color == "#ffffff"
    dialog.close()
