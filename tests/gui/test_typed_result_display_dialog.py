from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from fem.application.results import (
    FieldAssociation,
    FieldAvailability,
    FieldDescriptor,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    FieldState,
    PhysicalQuantity,
    ResultCatalog,
    ResultDiagnostic,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
)
from fem_gui.postprocessing_dialogs import (
    TypedResultDisplayDialog,
    TypedResultDisplaySettings,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _key(
    variable: ResultVariable,
    position: FieldPosition,
    *,
    contract: int,
) -> FieldMaterializationKey:
    return FieldMaterializationKey(
        FieldRequest(ResultFieldId(variable, position)),
        contract,
    )


def _descriptor(
    key: FieldMaterializationKey,
    *,
    association: FieldAssociation,
    quantity: PhysicalQuantity,
    components: tuple[str, ...],
    derived_components: tuple[str, ...],
    label_key: str,
    default_component: str,
    order: int,
) -> FieldDescriptor:
    return FieldDescriptor(
        field_id=key.request.field_id,
        association=association,
        quantity=quantity,
        components=components,
        derived_components=derived_components,
        label_key=label_key,
        unit_label=None,
        default_component=default_component,
        order=order,
    )


def _catalog() -> ResultCatalog:
    displacement = _key(
        ResultVariable.U,
        FieldPosition.NODE,
        contract=11,
    )
    reaction = _key(
        ResultVariable.RF,
        FieldPosition.NODE,
        contract=12,
    )
    stress = _key(
        ResultVariable.S,
        FieldPosition.ELEMENT_NODAL,
        contract=13,
    )
    unavailable_diagnostic = ResultDiagnostic(
        code="result.field.unavailable",
        severity="error",
        message="当前模型无法恢复节点应力。",
        path=("Step", "Static-1", "S"),
        remediation="选择受支持的场变量。",
        details={},
    )
    fields = (
        FieldAvailability(
            displacement,
            _descriptor(
                displacement,
                association=FieldAssociation.NODE,
                quantity=PhysicalQuantity.DISPLACEMENT,
                components=("U2", "U1"),
                derived_components=("Magnitude",),
                label_key="result.field.u.node",
                default_component="Magnitude",
                order=80,
            ),
            FieldState.READY,
        ),
        FieldAvailability(
            reaction,
            _descriptor(
                reaction,
                association=FieldAssociation.NODE,
                quantity=PhysicalQuantity.FORCE,
                components=("RF2", "RF1"),
                derived_components=("Magnitude",),
                label_key="vendor.result.reaction",
                default_component="Magnitude",
                order=2,
            ),
            FieldState.LAZY,
        ),
        FieldAvailability(
            stress,
            _descriptor(
                stress,
                association=FieldAssociation.ELEMENT_NODE,
                quantity=PhysicalQuantity.STRESS,
                components=("S22", "S11"),
                derived_components=("Mises",),
                label_key="result.field.s.element_nodal",
                default_component="Mises",
                order=1,
            ),
            FieldState.UNAVAILABLE,
            (unavailable_diagnostic,),
        ),
    )
    return ResultCatalog(
        source=ResultSourceKey(
            result_id="result-display",
            session_id="session-display",
            artifact_id="artifact-display",
            model_revision=8,
            step_name="Static-1",
            run_id="run-display",
        ),
        fields=fields,
        default_selection=ScalarFieldSelection(displacement, "U1"),
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


def test_dialog_requires_catalog_selection_membership_and_component() -> None:
    _application()
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

    foreign_key = _key(
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


def test_catalog_and_descriptor_order_keep_complete_typed_identity() -> None:
    _application()
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
        "位移 U（就绪）",
        "vendor.result.reaction（按需加载）",
        "应力 S（节点）（不可用）",
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


def test_ready_and_lazy_apply_emit_complete_typed_settings() -> None:
    _application()
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
    assert "外层命令加载" in dialog.availability_label.text()
    dialog.close()


def test_unavailable_field_shows_diagnostic_and_cannot_submit() -> None:
    _application()
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


def test_typed_display_path_has_no_legacy_identity_or_numerical_work() -> None:
    module_path = Path(inspect.getsourcefile(TypedResultDisplayDialog) or "")
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    typed_definitions = {
        "TypedResultDisplayDialog",
        "TypedResultDisplaySettings",
        "_typed_result_display_field_label",
        "_typed_result_display_availability_text",
        "_validate_typed_display_selection",
        "_validate_typed_display_options",
    }
    definitions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
        and node.name in typed_definitions
    }

    assert definitions.keys() == typed_definitions
    for definition in definitions.values():
        names = {node.id for node in ast.walk(definition) if isinstance(node, ast.Name)}
        attributes = {
            node.attr
            for node in ast.walk(definition)
            if isinstance(node, ast.Attribute)
        }
        string_literals = {
            node.value
            for node in ast.walk(definition)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert names.isdisjoint(
            {
                "ResultData",
                "ResultProvider",
                "ResultQuery",
                "ResultVariable",
                "FieldPosition",
                "_field_records",
                "field_family",
                "sorted",
            }
        )
        assert attributes.isdisjoint(
            {
                "query",
                "materialize",
                "field",
                "split",
                "partition",
                "startswith",
                "endswith",
            }
        )
        assert string_literals.isdisjoint(
            {
                "NODAL:",
                "EN:",
                "IP:",
                "CENTROID:",
            }
        )
