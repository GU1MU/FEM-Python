from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidgetItem
import pytest

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
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
)
from fem_gui.widgets.result_tree import (
    ROLE_FIELD_STATE,
    ROLE_MATERIALIZATION_KEY,
    ROLE_SELECTION,
    ResultTree,
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
        FieldPosition.CENTROID,
        contract=13,
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
                association=FieldAssociation.ELEMENT,
                quantity=PhysicalQuantity.STRESS,
                components=("S22", "S11"),
                derived_components=("Mises",),
                label_key="result.field.s.centroid",
                default_component="Mises",
                order=1,
            ),
            FieldState.UNAVAILABLE,
        ),
    )
    return ResultCatalog(
        source=ResultSourceKey(
            result_id="result-1",
            session_id="session-1",
            artifact_id="artifact-1",
            model_revision=4,
            step_name="Static-1",
            run_id="run-1",
        ),
        fields=fields,
        default_selection=ScalarFieldSelection(displacement, "U1"),
    )


def _step_item(tree: ResultTree) -> QTreeWidgetItem:
    return tree.topLevelItem(0).child(0)


def test_catalog_tree_preserves_published_field_and_component_order() -> None:
    _application()
    catalog = _catalog()
    tree = ResultTree()

    tree.set_catalog("Job-1 · Static-1", catalog)

    step = _step_item(tree)
    assert step.text(0) == "Job-1 · Static-1"
    assert [step.child(index).text(0) for index in range(3)] == [
        "位移 U（就绪）",
        "vendor.result.reaction（按需加载）",
        "应力 S（单元质心）（不可用）",
    ]
    assert [
        step.child(0).child(index).text(0)
        for index in range(step.child(0).childCount())
    ] == ["U2", "U1", "Magnitude"]
    assert [
        step.child(1).child(index).text(0)
        for index in range(step.child(1).childCount())
    ] == ["RF2", "RF1", "Magnitude"]
    assert [
        step.child(2).child(index).text(0)
        for index in range(step.child(2).childCount())
    ] == ["S22", "S11", "Mises"]


def test_catalog_items_keep_complete_typed_identity_and_default_selection() -> None:
    _application()
    catalog = _catalog()
    tree = ResultTree()

    tree.set_catalog("Static-1", catalog)

    step = _step_item(tree)
    for index, availability in enumerate(catalog.fields):
        field_item = step.child(index)
        assert field_item.data(0, ROLE_MATERIALIZATION_KEY) == availability.key
        assert field_item.data(0, ROLE_FIELD_STATE) == availability.state.value
        assert field_item.data(0, ROLE_SELECTION).field_key == availability.key
        for component_index, component in enumerate(availability.descriptor.columns):
            component_item = field_item.child(component_index)
            assert (
                component_item.data(
                    0,
                    ROLE_MATERIALIZATION_KEY,
                )
                == availability.key
            )
            assert component_item.data(
                0,
                ROLE_SELECTION,
            ) == ScalarFieldSelection(availability.key, component)
            assert (
                component_item.data(
                    0,
                    ROLE_FIELD_STATE,
                )
                == availability.state.value
            )

    assert tree.currentItem().data(0, ROLE_SELECTION) == catalog.default_selection
    assert tree.currentItem().text(0) == "U1"


def test_ready_and_lazy_items_emit_typed_selection_while_unavailable_does_not() -> None:
    _application()
    catalog = _catalog()
    tree = ResultTree()
    tree.set_catalog("Static-1", catalog)
    emitted: list[ScalarFieldSelection] = []
    legacy: list[str] = []
    tree.fieldSelectionActivated.connect(emitted.append)
    tree.fieldActivated.connect(legacy.append)
    step = _step_item(tree)

    ready_field = step.child(0)
    ready_component = step.child(0).child(0)
    lazy_field = step.child(1)
    unavailable_component = step.child(2).child(0)
    tree.itemDoubleClicked.emit(ready_field, 0)
    tree.itemDoubleClicked.emit(ready_component, 0)
    tree.itemDoubleClicked.emit(lazy_field, 0)
    tree._activate_item(unavailable_component)

    assert emitted == [
        ScalarFieldSelection(catalog.fields[0].key, "Magnitude"),
        ScalarFieldSelection(catalog.fields[0].key, "U2"),
        ScalarFieldSelection(catalog.fields[1].key, "Magnitude"),
    ]
    assert legacy == []
    unavailable_flags = unavailable_component.flags()
    assert not unavailable_flags & Qt.ItemFlag.ItemIsEnabled
    assert not unavailable_flags & Qt.ItemFlag.ItemIsSelectable


def test_set_catalog_requires_exact_typed_inputs() -> None:
    _application()
    tree = ResultTree()
    catalog = _catalog()

    with pytest.raises(TypeError, match="step_name"):
        tree.set_catalog(None, catalog)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ResultCatalog"):
        tree.set_catalog("Static-1", object())  # type: ignore[arg-type]


def test_typed_catalog_path_has_no_legacy_parsing_or_gui_field_order() -> None:
    module_path = Path(inspect.getsourcefile(ResultTree) or "")
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    typed_functions = {
        "set_catalog",
        "_catalog_field_item",
        "_set_typed_item_data",
    }
    definitions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in typed_functions
    }

    assert definitions.keys() == typed_functions
    for definition in definitions.values():
        names = {node.id for node in ast.walk(definition) if isinstance(node, ast.Name)}
        attributes = {
            node.attr
            for node in ast.walk(definition)
            if isinstance(node, ast.Attribute)
        }
        assert names.isdisjoint(
            {
                "ResultData",
                "ResultVariable",
                "FieldPosition",
                "field_family",
                "sorted",
            }
        )
        assert attributes.isdisjoint({"split", "partition", "startswith", "endswith"})
