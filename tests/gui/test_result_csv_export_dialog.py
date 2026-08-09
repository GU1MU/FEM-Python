from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication, QLabel

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


def _check_components(
    dialog: ResultCsvExportDialog,
    selections: tuple[ScalarFieldSelection, ...],
) -> None:
    for index, candidate in enumerate(dialog._component_selections):
        dialog.component_list.item(index).setCheckState(
            Qt.CheckState.Checked
            if candidate in selections
            else Qt.CheckState.Unchecked
        )


def test_dialog_selects_ready_fields_without_showing_field_status() -> None:
    _application()
    catalog = _catalog()
    dialog = ResultCsvExportDialog(
        catalog,
        current_selection=catalog.default_selection,
    )

    ready_fields = tuple(
        availability
        for availability in visible_result_fields(catalog.fields)
        if availability.state is FieldState.READY
    )
    assert ready_fields
    assert dialog.current_selection() == catalog.default_selection
    assert dialog.path_edit.text() == ""
    assert dialog.path_edit.placeholderText() == "请选择 CSV 保存路径"
    assert not dialog.export_button.isEnabled()
    assert all(
        selection.field_key
        in {availability.key for availability in ready_fields}
        for selection in dialog._component_selections
    )
    assert dialog.current_selections() == (catalog.default_selection,)

    visible_text = " ".join(
        label.text() for label in dialog.findChildren(QLabel)
    )
    assert "字段状态" not in visible_text
    assert "已就绪" not in visible_text
    assert "按需加载" not in visible_text
    assert "不可用" not in visible_text
    assert not hasattr(dialog, "availability_label")
    assert "分量" in visible_text
    assert "：" not in visible_text
    assert "可多选" not in visible_text
    assert dialog.component_list.verticalScrollMode() == (
        QAbstractItemView.ScrollMode.ScrollPerPixel
    )
    assert dialog.component_list.selectionMode() == (
        QAbstractItemView.SelectionMode.NoSelection
    )
    assert all(
        not dialog.component_list.item(index).flags()
        & Qt.ItemFlag.ItemIsSelectable
        for index in range(dialog.component_list.count())
    )
    assert dialog.component_list.verticalScrollBar().singleStep() == 12
    assert dialog.component_list.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert "item:hover" not in dialog.component_list.styleSheet()
    assert "rgba(76, 88, 98, 92)" in dialog.component_list.styleSheet()
    assert dialog.browse_button.size() == dialog.cancel_button.size()
    assert dialog.export_button.size() == dialog.cancel_button.size()
    dialog.close()


def test_component_list_supports_multiple_checks_and_preserves_user_path(
    tmp_path: Path,
) -> None:
    _application()
    catalog = _catalog()
    dialog = ResultCsvExportDialog(
        catalog,
        current_selection=catalog.default_selection,
    )
    field_key = dialog._component_selections[0].field_key
    selected = tuple(
        selection
        for selection in dialog._component_selections
        if selection.field_key == field_key
    )[:2]
    assert len(selected) == 2
    _check_components(dialog, selected)

    assert dialog.current_selections() == selected
    assert dialog.current_selection() == selected[0]
    assert dialog.path_edit.text() == ""
    assert not dialog.export_button.isEnabled()

    custom = tmp_path / "custom-name.csv"
    dialog.path_edit.setText(str(custom))
    assert dialog.export_button.isEnabled()
    _check_components(dialog, ())
    assert not dialog.export_button.isEnabled()
    _check_components(dialog, (selected[1],))
    assert dialog.target_path() == custom
    assert dialog.export_button.isEnabled()
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
    )
    browse_calls = []
    monkeypatch.setattr(
        dialog_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **_kwargs: (
            browse_calls.append(args) or ("", "")
        ),
    )
    dialog.browse_button.click()
    assert browse_calls[0][2] == ""
    assert dialog.path_edit.text() == ""

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
