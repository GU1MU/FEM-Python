from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem.solvers.static_linear import solve
from fem_gui.dialogs import CompactDoubleSpinBox
from fem_gui.postprocessing_dialogs import (
    ContourSettingsDialog,
    ResultDisplayDialog,
    ResultQueryDialog,
)
from fem_gui.symbol_dialog import SymbolSettingsDialog
from fem_gui.viewport_background import ViewportBackgroundSettings
from fem_gui.viewport_background_dialog import ViewportBackgroundDialog
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data
from fem_gui.visualization.symbols import SymbolSettings


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _result_data(gui_inp_path):
    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    return model, geometry, build_result_data(solve(model), geometry)


def test_compact_number_input_hides_only_insignificant_trailing_zeroes():
    _application()
    editor = CompactDoubleSpinBox()
    editor.setDecimals(8)

    editor.setValue(1.0)
    assert editor.text() == "1.00"

    editor.setValue(0.0001)
    assert editor.text() == "0.0001"
    assert editor.value() == 0.0001


def test_result_display_dialog_exposes_real_families_positions_and_components(gui_inp_path):
    _application()
    _model, _geometry, data = _result_data(gui_inp_path)
    dialog = ResultDisplayDialog(
        data.fields,
        step_name="Static-1",
        current_field="U",
        shape_mode="undeformed",
        contour_enabled=True,
        scale_mode="auto",
        scale_value=1.0,
        overlay_undeformed=True,
        show_edges=True,
    )

    families = [dialog.family_combo.itemText(index) for index in range(dialog.family_combo.count())]
    assert families == ["位移", "反力", "应力"]
    assert dialog.family_combo.currentText() == "位移"
    assert dialog.component_combo.currentText() == "总位移"
    assert dialog.position_combo.currentText() == "节点"
    assert dialog.settings().field_key == "U"
    assert dialog.settings().shape_mode == "undeformed"
    assert dialog.settings().contour_enabled
    assert dialog.settings().overlay_undeformed


def test_query_dialog_uses_current_selection_and_builds_table(gui_inp_path):
    _application()
    _model, geometry, data = _result_data(gui_inp_path)
    dialog = ResultQueryDialog(
        data,
        step_name="Static-1",
        node_ids=tuple(geometry.node_id_to_point_index),
        element_ids=tuple(geometry.element_id_to_cell_index),
        selected_kind="node",
        selected_id=2,
    )

    assert dialog.ids_edit.text() == "2"
    dialog.run_query()
    assert dialog.table.rowCount() == 1
    assert dialog.table.horizontalHeaderItem(0).text() == "编号"
    assert dialog.table.item(0, 0).text() == "2"


def test_contour_and_symbol_dialogs_round_trip_settings():
    _application()
    contour = ContourSettingsDialog(
        {
            "manual": False,
            "minimum": 0.0,
            "maximum": 1.0,
            "colormap": "viridis",
            "style": "continuous",
            "levels": 16,
            "number_format": "scientific",
            "decimals": 4,
            "orientation": "vertical",
            "legend": True,
            "show_minimum": True,
            "show_maximum": True,
            "show_ids": True,
        }
    )
    assert contour.settings()["levels"] == 16
    assert contour.settings()["style"] == "continuous"
    assert contour.settings()["orientation"] == "vertical"
    assert contour.settings()["show_ids"]

    settings = SymbolSettings(step_name="Static-1", show_values=True, scale=1.5)
    symbols = SymbolSettingsDialog(settings, ("Static-1",))
    assert symbols.settings().step_name == "Static-1"
    assert symbols.settings().show_values
    assert symbols.settings().scale == 1.5


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
