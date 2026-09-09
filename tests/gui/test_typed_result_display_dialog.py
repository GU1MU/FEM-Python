from __future__ import annotations

from tests.helpers.result_catalogs import make_result_catalog, make_result_field_key


import pytest
from PySide6.QtWidgets import QDialog

from fem.application.results import (
    FieldMaterializationKey,
    FieldPosition,
    ResultCatalog,
    ResultDiagnostic,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
)
from fem_gui.postprocessing_dialogs import (
    TypedResultDisplayDialog,
    TypedResultDisplaySettings,
)


def _catalog() -> ResultCatalog:
    unavailable_diagnostic = ResultDiagnostic(
        code="result.field.unavailable",
        severity="error",
        message="当前模型无法恢复节点应力。",
        path=("Step", "Static-1", "S"),
        remediation="选择受支持的场变量。",
        details={},
    )
    return make_result_catalog(
        source=ResultSourceKey(
            result_id="result-display",
            session_id="session-display",
            artifact_id="artifact-display",
            model_revision=8,
            step_name="Static-1",
            run_id="run-display",
        ),
        unavailable_diagnostic=unavailable_diagnostic,
    )


def _dialog(
    catalog: ResultCatalog,
    *,
    selection: ScalarFieldSelection | None = None,
) -> TypedResultDisplayDialog:
    return TypedResultDisplayDialog(
        catalog,
        current_selection=selection or catalog.default_selection,
        shape_mode="deformed",
        contour_enabled=True,
        scale_mode="custom",
        scale_value=2.5,
        overlay_undeformed=True,
        show_edges=False,
    )


def _combo_index(combo, value: object) -> int:
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            return index
    return -1


def test_settings_dto_enforces_exact_types_and_display_modes() -> None:
    catalog = _catalog()
    selection = catalog.default_selection
    settings = TypedResultDisplaySettings(
        shape_mode="undeformed",
        contour_enabled=True,
        selection=selection,
        scale_mode="auto",
        scale_value=1.0,
        overlay_undeformed=False,
        show_edges=True,
    )

    assert settings.selection is selection
    with pytest.raises(TypeError, match="contour_enabled"):
        TypedResultDisplaySettings(
            "undeformed",
            1,  # type: ignore[arg-type]
            selection,
            "auto",
            1.0,
            False,
            True,
        )
    with pytest.raises(TypeError, match="scale_value"):
        TypedResultDisplaySettings(
            "undeformed",
            True,
            selection,
            "auto",
            1,  # type: ignore[arg-type]
            False,
            True,
        )
    with pytest.raises(ValueError, match="shape_mode"):
        TypedResultDisplaySettings(
            "wireframe",
            True,
            selection,
            "auto",
            1.0,
            False,
            True,
        )
    with pytest.raises(ValueError, match="scale_mode"):
        TypedResultDisplaySettings(
            "undeformed",
            True,
            selection,
            "fit",
            1.0,
            False,
            True,
        )


def test_dialog_requires_catalog_selection_membership_and_component(gui_application) -> None:
    catalog = _catalog()

    with pytest.raises(TypeError, match="ResultCatalog"):
        TypedResultDisplayDialog(
            object(),  # type: ignore[arg-type]
            current_selection=catalog.default_selection,
            shape_mode="deformed",
            contour_enabled=True,
            scale_mode="custom",
            scale_value=2.5,
            overlay_undeformed=True,
            show_edges=False,
        )
    with pytest.raises(TypeError, match="ScalarFieldSelection"):
        _dialog(catalog, selection=object())  # type: ignore[arg-type]

    foreign_key = make_result_field_key(
        ResultVariable.RM,
        FieldPosition.NODE,
        contract=99,
    )
    with pytest.raises(ValueError, match="catalog field"):
        _dialog(
            catalog,
            selection=ScalarFieldSelection(foreign_key, "RM1"),
        )
    with pytest.raises(ValueError, match="field descriptor"):
        _dialog(
            catalog,
            selection=ScalarFieldSelection(
                catalog.fields[0].key,
                "U9",
            ),
        )

    dialog = _dialog(catalog)
    assert dialog.catalog is catalog
    assert dialog.source is catalog.source
    assert dialog.step_combo.currentData() is catalog.source
    dialog.close()


def test_catalog_and_descriptor_order_keep_complete_typed_identity(gui_application) -> None:
    catalog = _catalog()
    dialog = _dialog(catalog)

    assert tuple(
        dialog.field_combo.itemData(index)
        for index in range(dialog.field_combo.count())
    ) == tuple(availability.key for availability in catalog.fields)
    assert tuple(
        dialog.field_combo.itemText(index)
        for index in range(dialog.field_combo.count())
    ) == (
        "Displacement U (Ready)",
        "vendor.result.reaction (Load on demand)",
        "Stress S (Node) (Unavailable)",
    )
    assert all(
        type(dialog.field_combo.itemData(index)) is FieldMaterializationKey
        for index in range(dialog.field_combo.count())
    )
    assert dialog.current_selection() == catalog.default_selection
    assert (
        tuple(
            dialog.component_combo.itemData(index)
            for index in range(dialog.component_combo.count())
        )
        == catalog.fields[0].descriptor.columns
    )

    lazy = catalog.fields[1]
    dialog.field_combo.setCurrentIndex(_combo_index(dialog.field_combo, lazy.key))
    assert (
        tuple(
            dialog.component_combo.itemData(index)
            for index in range(dialog.component_combo.count())
        )
        == lazy.descriptor.columns
    )
    assert dialog.current_selection() == ScalarFieldSelection(
        lazy.key,
        lazy.descriptor.default_component,
    )
    dialog.close()


def test_ready_and_lazy_apply_emit_complete_typed_settings(gui_application) -> None:
    catalog = _catalog()
    dialog = _dialog(catalog)
    emitted: list[TypedResultDisplaySettings] = []
    dialog.applyRequested.connect(emitted.append)

    dialog.apply()
    assert emitted == [
        TypedResultDisplaySettings(
            shape_mode="deformed",
            contour_enabled=True,
            selection=catalog.default_selection,
            scale_mode="custom",
            scale_value=2.5,
            overlay_undeformed=True,
            show_edges=False,
        )
    ]
    assert type(emitted[0]) is TypedResultDisplaySettings

    lazy = catalog.fields[1]
    dialog.field_combo.setCurrentIndex(_combo_index(dialog.field_combo, lazy.key))
    dialog.component_combo.setCurrentIndex(_combo_index(dialog.component_combo, "RF1"))
    dialog.apply()

    assert emitted[-1].selection == ScalarFieldSelection(
        lazy.key,
        "RF1",
    )
    assert emitted[-1].selection.field_key is lazy.key
    assert dialog.apply_button.isEnabled()
    assert "loaded by the outer command" in dialog.availability_label.text()
    dialog.close()


def test_unavailable_field_shows_diagnostic_and_cannot_submit(gui_application) -> None:
    catalog = _catalog()
    dialog = _dialog(catalog)
    emitted: list[TypedResultDisplaySettings] = []
    dialog.applyRequested.connect(emitted.append)
    unavailable = catalog.fields[2]

    dialog.field_combo.setCurrentIndex(
        _combo_index(dialog.field_combo, unavailable.key)
    )

    assert (
        tuple(
            dialog.component_combo.itemData(index)
            for index in range(dialog.component_combo.count())
        )
        == unavailable.descriptor.columns
    )
    assert "无法恢复节点应力" in dialog.availability_label.text()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.ok_button.isEnabled()
    dialog.apply()
    dialog.accept_with_apply()
    assert emitted == []
    assert dialog.result() == QDialog.DialogCode.Rejected
    dialog.close()
