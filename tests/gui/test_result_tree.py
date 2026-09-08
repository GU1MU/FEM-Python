from __future__ import annotations

from tests.helpers.result_catalogs import make_result_catalog


from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem
import pytest

from fem.application.results import ScalarFieldSelection
from fem_gui.widgets.result_tree import (
    ROLE_FIELD_STATE,
    ROLE_MATERIALIZATION_KEY,
    ROLE_SELECTION,
    ResultTree,
)


def _step_item(tree: ResultTree) -> QTreeWidgetItem:
    return tree.topLevelItem(0).child(0)


def test_catalog_tree_preserves_published_field_and_component_order() -> None:
    catalog = make_result_catalog()
    tree = ResultTree()

    tree.set_catalog("Job-1 · Static-1", catalog)

    assert tree.catalog is catalog
    step = _step_item(tree)
    assert step.text(0) == "Job-1 · Static-1"
    assert [step.child(index).text(0) for index in range(3)] == [
        "位移 U",
        "vendor.result.reaction",
        "应力 S",
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
    catalog = make_result_catalog()
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
    catalog = make_result_catalog()
    tree = ResultTree()
    tree.set_catalog("Static-1", catalog)
    emitted: list[ScalarFieldSelection] = []
    tree.fieldSelectionActivated.connect(emitted.append)
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
    unavailable_flags = unavailable_component.flags()
    assert not unavailable_flags & Qt.ItemFlag.ItemIsEnabled
    assert not unavailable_flags & Qt.ItemFlag.ItemIsSelectable


def test_set_catalog_requires_exact_typed_inputs() -> None:
    tree = ResultTree()
    catalog = make_result_catalog()

    with pytest.raises(TypeError, match="step_name"):
        tree.set_catalog(None, catalog)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ResultCatalog"):
        tree.set_catalog("Static-1", object())  # type: ignore[arg-type]
    assert tree.catalog is None


def test_select_selection_prefers_the_exact_component_leaf() -> None:
    tree = ResultTree()
    catalog = make_result_catalog()
    tree.set_catalog("Static-1", catalog)

    selection = ScalarFieldSelection(
        catalog.fields[0].key,
        catalog.fields[0].descriptor.default_component,
    )

    assert tree.select_selection(selection)
    assert tree.currentItem().childCount() == 0
    assert tree.currentItem().text(0) == "Magnitude"
    assert tree.currentItem().data(0, ROLE_SELECTION) == selection
