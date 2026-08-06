from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from fem.application.results import (
    FieldState,
    ResultSourceKey,
    ScalarFieldSelection,
    build_result_provider,
)
from fem.solvers.static_linear import solve
from fem_gui import result_csv_export_dialog as dialog_module
from fem_gui.result_csv_export_dialog import ResultCsvExportDialog
from fem_gui.result_presentation import visible_result_fields
from tests.helpers.model_builders import make_static_pull_truss_model


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _catalog():
    model = make_static_pull_truss_model()
    result = solve(model, "pull", name="result-csv-dialog")
    provider = build_result_provider(
        ResultSourceKey(
            result_id="result-csv-dialog",
            session_id="session-csv-dialog",
            artifact_id="artifact-csv-dialog",
            model_revision=1,
            step_name="pull",
            run_id="run-csv-dialog",
        ),
        result,
    )
    return provider.catalog()


def _find_data(combo, value: object) -> int:
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            return index
    return -1


def test_dialog_selects_ready_fields_without_showing_field_status(
    tmp_path: Path,
) -> None:
    _application()
    catalog = _catalog()
    dialog = ResultCsvExportDialog(
        catalog,
        current_selection=catalog.default_selection,
        default_directory=tmp_path,
        filename_stem="pull",
    )

    ready_fields = tuple(
        availability
        for availability in visible_result_fields(catalog.fields)
        if availability.state is FieldState.READY
    )
    assert ready_fields
    assert dialog.current_selection() == catalog.default_selection
    assert dialog.target_path().parent == tmp_path
    assert dialog.target_path().suffix == ".csv"
    assert all(
        dialog.component_combo.itemData(index).field_key
        in {availability.key for availability in ready_fields}
        for index in range(dialog.component_combo.count())
    )

    visible_text = " ".join(
        label.text() for label in dialog.findChildren(QLabel)
    )
    assert "字段状态" not in visible_text
    assert "已就绪" not in visible_text
    assert "按需加载" not in visible_text
    assert "不可用" not in visible_text
    assert not hasattr(dialog, "availability_label")
    dialog.close()


def test_field_selection_updates_suggested_name_and_preserves_custom_path(
    tmp_path: Path,
) -> None:
    _application()
    catalog = _catalog()
    dialog = ResultCsvExportDialog(
        catalog,
        current_selection=catalog.default_selection,
        default_directory=tmp_path,
        filename_stem="pull",
    )
    selections = tuple(
        selection
        for availability in visible_result_fields(catalog.fields)
        if availability.state is FieldState.READY
        for component in availability.descriptor.columns
        for selection in (
            ScalarFieldSelection(availability.key, component),
        )
        if selection != catalog.default_selection
    )
    selected = selections[0]
    field_id = selected.field_key.request.field_id

    dialog.variable_combo.setCurrentIndex(
        _find_data(dialog.variable_combo, field_id.variable)
    )
    dialog.position_combo.setCurrentIndex(
        _find_data(dialog.position_combo, field_id.position)
    )
    dialog.component_combo.setCurrentIndex(
        _find_data(dialog.component_combo, selected)
    )

    assert dialog.current_selection() == selected
    assert selected.component in dialog.target_path().stem

    custom = tmp_path / "custom-name.csv"
    dialog.path_edit.setText(str(custom))
    dialog.path_edit.textEdited.emit(str(custom))
    alternative_index = (
        1 if dialog.component_combo.count() > 1 else 0
    )
    dialog.component_combo.setCurrentIndex(alternative_index)
    assert dialog.target_path() == custom
    dialog.close()


def test_browse_normalizes_csv_suffix_and_cancel_keeps_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    catalog = _catalog()
    dialog = ResultCsvExportDialog(
        catalog,
        current_selection=catalog.default_selection,
        default_directory=tmp_path,
        filename_stem="pull",
    )
    original = dialog.target_path()
    monkeypatch.setattr(
        dialog_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("", ""),
    )
    dialog.browse_button.click()
    assert dialog.target_path() == original

    picked = tmp_path / "picked.txt"
    monkeypatch.setattr(
        dialog_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(picked), "CSV 文件 (*.csv)"),
    )
    dialog.browse_button.click()
    assert dialog.target_path() == picked.with_suffix(".csv")

    dialog.path_edit.clear()
    assert not dialog.export_button.isEnabled()
    dialog.close()
