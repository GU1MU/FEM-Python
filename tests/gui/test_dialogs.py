from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from fem_gui.dialogs import (
    AdaptivePrecisionDoubleSpinBox,
    CompactDoubleSpinBox,
)
from fem_gui.postprocessing_dialogs import (
    ContourSettingsDialog,
    DisplaySettingsDialog,
    ResultAnimationDialog,
    DisplayGroupViewCutDialog,
    VisualizationOptionsDialog,
)
from fem_gui.symbol_dialog import SymbolSettingsDialog
from fem_gui.viewport_background import ViewportBackgroundSettings
from fem_gui.viewport_background_dialog import ViewportBackgroundDialog
from fem_gui.visualization.colormaps import ABAQUS_RAINBOW, CUSTOM_COLORMAP
from fem_gui.visualization.contour_rendering import (
    CONTOUR_EDGE_FEATURE,
    CONTOUR_EDGE_GEOMETRY,
    CONTOUR_EDGE_NONE,
    CONTOUR_RENDER_FILLED,
    CONTOUR_RENDER_WIREFRAME,
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


def test_adaptive_number_input_preserves_only_user_typed_precision():
    _application()
    editor = AdaptivePrecisionDoubleSpinBox()

    editor.setValue(2.3456789)
    assert editor.decimals() == 12
    assert editor.value() == 2.3456789
    assert editor.text() == "2.35"

    editor.selectAll()
    QTest.keyClicks(editor, "0.2500")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    assert editor.value() == 0.25
    assert editor.text() == "0.2500"

    editor.setValue(3.4567)
    assert editor.value() == 3.4567
    assert editor.text() == "3.46"


def test_adaptive_number_input_keeps_small_values_visible_and_accepts_exponents():
    _application()
    editor = AdaptivePrecisionDoubleSpinBox(input_decimals=16)
    editor.setRange(1.0e-16, 1.0)
    editor.setValue(1.0e-9)
    assert editor.value() == pytest.approx(1.0e-9)
    assert editor.text() != "0"
    assert "e-" in editor.text()

    editor.lineEdit().setText("2e-6")
    editor.interpretText()
    assert editor.value() == pytest.approx(2.0e-6)


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
    assert contour.style.currentText() == "连续"
    assert contour.render_group.layout().itemAt(0).layout().itemAt(0).widget().text() == "模式"
    assert contour.render_group.layout().itemAt(0).layout().itemAt(3).widget().text() == "样式"
    assert contour.render_group.layout().itemAt(0).layout().itemAt(6).widget().text() == "色带"
    assert contour.style.width() == 90
    assert contour.colormap.width() == 120
    assert not contour.levels_row.isEnabled()
    assert contour.minimum.value() == -2.5
    assert contour.maximum.value() == 8.75
    assert not contour.minimum.isEnabled()
    assert not contour.maximum.isEnabled()
    assert contour.show_minimum.parent() is contour.range_group
    assert contour.show_maximum.parent() is contour.range_group
    assert contour.show_minimum.text() == "显示"
    assert contour.show_maximum.text() == "显示"
    assert contour.minimum.width() == 110
    assert contour.maximum.width() == 110
    contour.maximum.setValue(0.001139735919)
    assert contour.maximum.text() == "0.0011397"

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
            "legend_position": "left",
            "legend_label_count": "9",
            "legend_title": False,
            "legend": True,
            "show_ids": True,
            "show_coordinate_system": False,
            "show_node_labels": True,
            "show_element_labels": True,
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
    assert display_settings["legend_position"] == "left"
    assert display_settings["legend_label_count"] == "9"
    assert not display_settings["legend_title"]
    assert display_settings["show_ids"]
    assert not display_settings["show_coordinate_system"]
    assert display_settings["show_node_labels"]
    assert display_settings["show_element_labels"]
    display_labels = {
        label.text() for label in display.findChildren(QLabel)
    }
    assert all("：" not in text and ":" not in text for text in display_labels)
    assert "线条" in display_labels
    assert "轮廓" not in display_labels
    assert display.edge_mode.width() == 112
    assert display.edge_style.width() == 112
    assert display.edge_width.width() == 60
    assert display.edge_width.suffix() == ""
    assert display.edge_width_unit.text() == "pt"
    assert display.decimals.width() == 60
    assert display.legend_font.width() == 110
    assert display.legend_font_size.width() == 60
    assert display.legend_font_size.suffix() == ""
    assert display.legend_font_size_unit.text() == "pt"
    assert display.legend_position.width() == 82
    assert display.legend_label_count.width() == 82
    assert display.legend_title.text() == "显示"
    assert display.show_node_labels.text() == "显示节点编号"
    assert display.show_element_labels.text() == "显示单元编号"
    display.engineering_format.click()
    display.vertical_orientation.click()
    assert display.engineering_format.isChecked()
    assert not display.scientific_format.isChecked()
    assert display.vertical_orientation.isChecked()
    assert not display.horizontal_orientation.isChecked()
    display.show()
    _application().processEvents()
    assert (
        display.decimals.geometry().left()
        == display.legend_font.geometry().left()
    )
    assert (
        display.vertical_orientation.geometry().left()
        == display.engineering_format.geometry().left()
    )

    labels = {
        label.text() for label in contour.findChildren(QLabel)
    }
    assert all("：" not in text and ":" not in text for text in labels)
    assert "阈值" in labels
    assert "级数" in labels
    assert "样式" in labels
    assert "色带" in labels
    assert "模式" in labels
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
    contour.show()
    _application().processEvents()
    filled_left = contour.filled_mode.mapTo(
        contour.render_group,
        contour.filled_mode.rect().topLeft(),
    ).x()
    levels_track_left = contour.levels_slider.mapTo(
        contour.render_group,
        contour.levels_slider.rect().topLeft(),
    ).x() + contour.levels_slider._margin
    assert filled_left == levels_track_left
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
    contour.global_range.setChecked(True)
    assert contour.settings()["range_mode"] == "global_step"
    assert not contour.settings()["manual"]
    assert not contour.minimum.isEnabled()
    assert not contour.maximum.isEnabled()
    contour.manual_range.setChecked(True)
    assert contour.settings()["range_mode"] == "manual"
    assert contour.minimum.isEnabled()
    assert contour.maximum.isEnabled()
    contour.style.setCurrentIndex(contour.style.findData("segmented"))
    assert contour.levels_row.isEnabled()
    contour.filled_mode.click()
    assert contour.filled_mode.isChecked()
    assert not contour.shaded_mode.isChecked()

    settings = SymbolSettings(step_name="Static-1", show_values=True, scale=1.5)
    symbols = SymbolSettingsDialog(settings, ("Static-1",))
    assert symbols.settings().step_name == "Static-1"
    assert symbols.settings().show_values
    assert symbols.settings().scale == 1.5


def test_contour_dialog_defaults_to_abaqus_rainbow():
    _application()
    contour = ContourSettingsDialog({})

    assert contour.colormap.currentText() == "彩虹"
    assert contour.settings()["colormap"] == ABAQUS_RAINBOW
    assert contour.auto_range.isChecked()
    assert contour.shaded_mode.isChecked()


def test_display_settings_defaults_to_geometry_edges():
    _application()
    display = DisplaySettingsDialog({})

    assert display.edge_mode.currentData() == CONTOUR_EDGE_GEOMETRY
    assert display.settings()["edges"]

    hidden = DisplaySettingsDialog(
        {
            "edge_mode": CONTOUR_EDGE_NONE,
            "edges": False,
        }
    )
    assert hidden.edge_mode.currentData() == CONTOUR_EDGE_NONE
    assert not hidden.settings()["edges"]


def test_visualization_options_dialog_uses_one_categorized_window():
    _application()
    dialog = VisualizationOptionsDialog(
        {
            "shape_mode": "deformed",
            "contour_enabled": True,
            "scale_mode": "custom",
            "scale_value": 2.5,
            "range_mode": "global_step",
            "global_minimum": -1.0,
            "global_maximum": 3.0,
            "show_node_labels": True,
            "show_symbols": True,
        },
        initial_category="云图显示",
    )

    assert [
        dialog.category_list.item(index).text()
        for index in range(dialog.category_list.count())
    ] == ["通用显示", "变形显示", "云图显示", "实体显示", "注释"]
    assert dialog.category_list.currentRow() == 2
    assert dialog.pages.currentWidget() is dialog.contour_page
    assert dialog.scale_mode.currentData() == "custom"
    assert dialog.scale_value.value() == 2.5
    assert dialog.auto_range.isChecked() is False
    assert dialog.global_range.isChecked()

    dialog.select_category("实体显示")
    dialog.show_node_labels.setChecked(True)
    dialog.show_element_labels.setChecked(True)
    settings = dialog.settings()
    assert settings["show_node_labels"]
    assert settings["show_element_labels"]
    assert settings["show_symbols"]
    assert settings["shape_mode"] == "deformed"
    assert settings["scale_mode"] == "custom"
    assert settings["range_mode"] == "global_step"
    dialog.deleteLater()


def test_visualization_options_dialog_combines_render_and_edge_modes():
    _application()
    dialog = VisualizationOptionsDialog(
        {
            "render_mode": CONTOUR_RENDER_FILLED,
            "edge_mode": CONTOUR_EDGE_FEATURE,
            "edges": True,
        },
        initial_category="云图显示",
    )

    settings = dialog.settings()
    assert settings["render_mode"] == CONTOUR_RENDER_FILLED
    assert settings["edge_mode"] == CONTOUR_EDGE_FEATURE
    assert settings["edges"]
    assert dialog.pages.count() == 5
    assert dialog.category_list.currentItem().text() == "云图显示"
    dialog.display_style.setCurrentIndex(
        dialog.display_style.findData("wireframe")
    )
    settings = dialog.settings()
    assert settings["render_mode"] == CONTOUR_RENDER_WIREFRAME
    assert settings["edge_mode"] == "all"
    dialog.select_category("注释")
    assert dialog.annotation_page.isAncestorOf(dialog.show_minimum.parent())
    assert not dialog.contour_page.isAncestorOf(dialog.show_minimum.parent())
    dialog.deleteLater()


def test_visualization_options_dialog_keeps_fixed_height_across_pages():
    application = _application()
    dialog = VisualizationOptionsDialog({}, initial_category="通用显示")
    initial_height = dialog.height()
    assert initial_height == 620
    assert not hasattr(dialog, "display_group_combo")
    assert not hasattr(dialog, "view_cut_list")
    assert dialog.common_page.sizeHint().height() <= 300
    for row in range(dialog.category_list.count()):
        dialog.category_list.setCurrentRow(row)
        application.processEvents()
        assert dialog.height() == initial_height
    dialog.deleteLater()


def test_display_group_view_cut_dialog_combines_manager_controls():
    _application()
    dialog = DisplayGroupViewCutDialog(
        {
            "display_groups": {
                "外壳": {"element_ids": (1, 3), "exclude": False},
            },
            "active_display_group": "外壳",
            "view_cut": {
                "planes": {
                    "x": {"enabled": False, "offset": 0.0, "invert": False},
                    "y": {"enabled": True, "offset": 0.25, "invert": True},
                    "z": {"enabled": False, "offset": 0.0, "invert": False},
                },
            },
        },
        initial_page="view_cut",
    )

    assert dialog.windowTitle() == "显示组与视图切割"
    assert [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())] == [
        "显示组",
        "视图切割",
    ]
    assert dialog.tabs.currentIndex() == 1
    assert dialog.display_group_list.count() == 2
    assert dialog.display_group_list.currentItem().text() == "外壳"
    assert dialog.view_cut_list.item(1).checkState() == Qt.CheckState.Checked
    assert dialog.view_cut_list.currentItem().text() == "Y-平面"
    assert dialog.view_cut_positions["y"].value() == pytest.approx(0.25)
    assert dialog.view_cut_invert.isChecked()

    dialog.tabs.setCurrentIndex(0)
    dialog.display_group_ids_edit.setText("2-4")
    dialog.display_group_save_button.click()
    dialog.tabs.setCurrentIndex(1)
    dialog.view_cut_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    settings = dialog.settings()
    assert settings["display_groups"]["外壳"]["element_ids"] == (2, 3, 4)
    assert settings["active_display_group"] == "外壳"
    assert not settings["view_cut"]["planes"]["y"]["enabled"]
    dialog.deleteLater()


def test_visualization_options_dialog_exposes_palette_preview_and_custom_stops():
    _application()
    dialog = VisualizationOptionsDialog({}, initial_category="云图显示")

    assert dialog.colormap.count() >= 14
    assert dialog.custom_color_group.isHidden()
    dialog.colormap.setCurrentIndex(dialog.colormap.findData(CUSTOM_COLORMAP))
    assert not dialog.custom_color_group.isHidden()
    assert dialog.custom_color_stops.rowCount() == 3

    middle_color = dialog.custom_color_stops.cellWidget(1, 1)
    middle_color.set_color("#ffff00")
    dialog.custom_color_stops.cellWidget(1, 0).setValue(40.0)
    dialog.colormap_reverse.setChecked(True)
    settings = dialog.settings()

    assert settings["colormap"] == CUSTOM_COLORMAP
    assert settings["colormap_reverse"]
    assert settings["custom_color_stops"] == (
        (0.0, "#0000ff"),
        (0.4, "#ffff00"),
        (1.0, "#ff0000"),
    )
    dialog.add_custom_color_button.click()
    assert dialog.custom_color_stops.rowCount() == 4
    dialog.deleteLater()


def test_animation_dialog_preserves_noncontiguous_frame_indices():
    _application()
    dialog = ResultAnimationDialog(
        (1, 3, 5),
        current_frame_index=3,
        start_frame=1,
        end_frame=5,
        interval_ms=250,
        loop=True,
    )
    assert dialog.windowTitle() == "动画"
    assert [
        dialog.start_frame_combo.itemData(index)
        for index in range(dialog.start_frame_combo.count())
    ] == [1, 3, 5]
    assert dialog.settings() == {
        "start_frame": 1,
        "end_frame": 5,
        "interval_ms": 250,
        "loop": True,
    }

    applied = []
    dialog.applyRequested.connect(applied.append)
    dialog.start_frame_combo.setCurrentIndex(1)
    dialog.end_frame_combo.setCurrentIndex(2)
    dialog.interval_spin.setValue(400)
    dialog.loop_checkbox.setChecked(False)
    assert dialog.apply()
    assert applied == [
        {
            "start_frame": 3,
            "end_frame": 5,
            "interval_ms": 400,
            "loop": False,
        }
    ]


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
