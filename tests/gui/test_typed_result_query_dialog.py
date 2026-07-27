from __future__ import annotations

import ast
from dataclasses import replace
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem.application.results import (
    FieldAssociation,
    FieldMaterializationKey,
    FieldPosition,
    FieldState,
    ResultProvider,
    ResultQuery,
    ResultSourceKey,
    ScalarFieldSelection,
    advance_materialization,
    build_result_provider,
    restore_result_provider,
)
from fem.post.fields import encode_result_region_key
from fem_gui.postprocessing_dialogs import TypedResultQueryDialog
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _source(result_id: str = "typed-query") -> ResultSourceKey:
    return ResultSourceKey(
        result_id=result_id,
        session_id="session-query-dialog",
        artifact_id="artifact-query-dialog",
        model_revision=8,
        step_name="Step-1",
        run_id="run-query-dialog",
    )


@pytest.fixture
def result_provider():
    result = make_continuum_nodal_semantics_result()
    provider = build_result_provider(_source(), result)
    return result, provider


def _availability_at(
    provider: ResultProvider,
    position: FieldPosition,
):
    matches = tuple(
        item
        for item in provider.catalog().fields
        if item.descriptor.field_id.position is position
    )
    assert len(matches) == 1
    return matches[0]


def _combo_index(combo, value: object) -> int:
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            return index
    return -1


def test_dialog_requires_exact_provider_and_matching_catalog(
    result_provider,
) -> None:
    _application()
    _result, provider = result_provider

    with pytest.raises(TypeError, match="ResultProvider"):
        TypedResultQueryDialog(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ResultCatalog"):
        TypedResultQueryDialog(
            provider,
            object(),  # type: ignore[arg-type]
        )

    foreign_catalog = replace(
        provider.catalog(),
        source=_source("foreign-query"),
    )
    with pytest.raises(ValueError, match="exactly match"):
        TypedResultQueryDialog(provider, foreign_catalog)

    equal_catalog = replace(provider.catalog())
    dialog = TypedResultQueryDialog(provider, equal_catalog)
    assert dialog.catalog is equal_catalog
    dialog.close()


def test_catalog_order_typed_association_and_descriptor_components_are_exact(
    result_provider,
) -> None:
    _application()
    _result, provider = result_provider
    catalog = provider.catalog()
    dialog = TypedResultQueryDialog(provider)

    mode = dialog.association_combo.currentData()
    assert type(mode).__name__ == "_TypedQueryMode"
    assert mode.association is FieldAssociation.NODE
    node_associations = (
        FieldAssociation.NODE,
        FieldAssociation.ELEMENT_NODE,
        FieldAssociation.NODE_REGION,
        FieldAssociation.RESOLVED_NODAL,
    )
    expected_node_keys = tuple(
        availability.key
        for availability in catalog.fields
        if availability.descriptor.association in node_associations
    )
    actual_node_keys = tuple(
        dialog.field_combo.itemData(index)
        for index in range(dialog.field_combo.count())
    )
    assert actual_node_keys == expected_node_keys
    assert all(
        type(key) is FieldMaterializationKey for key in actual_node_keys
    )
    assert dialog.current_selection() == catalog.default_selection

    default_availability = dialog.current_availability()
    assert tuple(
        dialog.component_combo.itemData(index)
        for index in range(dialog.component_combo.count())
    ) == default_availability.descriptor.columns

    dialog.association_combo.setCurrentIndex(1)
    mode = dialog.association_combo.currentData()
    assert mode.association is FieldAssociation.ELEMENT
    element_associations = (
        FieldAssociation.ELEMENT,
        FieldAssociation.INTEGRATION_POINT,
        FieldAssociation.ELEMENT_NODE,
    )
    assert tuple(
        dialog.field_combo.itemData(index)
        for index in range(dialog.field_combo.count())
    ) == tuple(
        availability.key
        for availability in catalog.fields
        if availability.descriptor.association in element_associations
    )
    dialog.close()


def test_ready_and_lazy_selections_emit_exact_typed_requests_without_recovery(
    result_provider,
) -> None:
    _application()
    _result, provider = result_provider
    original_snapshot = provider.snapshot
    dialog = TypedResultQueryDialog(provider)
    selections: list[object] = []
    queries: list[object] = []
    dialog.selectionRequested.connect(selections.append)
    dialog.queryRequested.connect(queries.append)

    dialog.ids_edit.clear()
    dialog.request_query()
    assert selections == [provider.catalog().default_selection]
    assert queries == [
        ResultQuery(
            field_key=provider.catalog().default_selection.field_key,
            component=provider.catalog().default_selection.component,
        )
    ]
    assert provider.snapshot is original_snapshot

    selections.clear()
    queries.clear()
    dialog.association_combo.setCurrentIndex(1)
    integration_point = _availability_at(
        provider,
        FieldPosition.INTEGRATION_POINT,
    )
    dialog.field_combo.setCurrentIndex(
        _combo_index(dialog.field_combo, integration_point.key)
    )
    dialog.component_combo.setCurrentIndex(
        _combo_index(dialog.component_combo, "S11")
    )
    selections.clear()
    queries.clear()
    dialog.ids_edit.setText("3, 1-2, 3")

    dialog.request_query()

    selection = ScalarFieldSelection(integration_point.key, "S11")
    query = ResultQuery(
        field_key=integration_point.key,
        component="S11",
        element_ids=(3, 1, 2),
    )
    assert integration_point.state is FieldState.LAZY
    assert selections == [selection]
    assert queries == [query]
    assert type(queries[0]) is ResultQuery
    assert provider.field_status(integration_point.key).state is FieldState.LAZY
    assert provider.snapshot is original_snapshot
    assert integration_point.key not in {
        field.key for field in provider.snapshot.fields
    }

    dialog.ids_edit.clear()
    assert dialog.current_query().element_ids == ()
    dialog.close()


def test_pending_query_freezes_intent_without_overwriting_latest_query(
    result_provider,
) -> None:
    _application()
    _result, provider = result_provider
    dialog = TypedResultQueryDialog(provider)
    queries: list[ResultQuery] = []
    dialog.queryRequested.connect(queries.append)
    dialog.request_query()
    submitted = queries[-1]

    dialog.set_query_pending(True)

    assert dialog.source == provider.source
    assert dialog.query_pending
    assert not dialog.association_combo.isEnabled()
    assert not dialog.field_combo.isEnabled()
    assert not dialog.component_combo.isEnabled()
    assert not dialog.ids_edit.isEnabled()
    assert not dialog.query_button.isEnabled()
    dialog.request_query()
    assert queries == [submitted]

    dialog.set_query_pending(False)

    assert not dialog.query_pending
    assert dialog.association_combo.isEnabled()
    assert dialog.field_combo.isEnabled()
    assert dialog.component_combo.isEnabled()
    assert dialog.ids_edit.isEnabled()
    assert dialog.query_button.isEnabled()
    dialog.set_query_message("查询已取消")
    assert dialog.result_summary.text() == "查询已取消"
    dialog.close()


def test_query_result_keeps_multi_region_and_provenance_rows_in_order(
    result_provider,
) -> None:
    _application()
    result, provider = result_provider
    dialog = TypedResultQueryDialog(provider)
    resolved = _availability_at(provider, FieldPosition.RESOLVED_NODAL)
    dialog.field_combo.setCurrentIndex(
        _combo_index(dialog.field_combo, resolved.key)
    )
    dialog.component_combo.setCurrentIndex(
        _combo_index(dialog.component_combo, "S11")
    )
    dialog.ids_edit.setText("1")
    emitted: list[ResultQuery] = []
    dialog.queryRequested.connect(emitted.append)
    dialog.request_query()
    query = emitted[-1]

    accepted = advance_materialization(
        provider.snapshot,
        provider.materialize((resolved.key,)),
    )
    ready_provider = restore_result_provider(result, accepted)
    query_result = ready_provider.query(query)
    assert query_result.materialization_generation == 1
    assert len(query_result.records) >= 2
    assert len(
        {
            record.location.region_key
            for record in query_result.records
        }
    ) >= 2
    assert any(
        record.location.element_id is not None
        for record in query_result.records
    )

    dialog.set_query_result(query_result)

    assert dialog.table.rowCount() == len(query_result.records)
    assert dialog.table.horizontalHeaderItem(5).text() == "区域"
    assert dialog.table.horizontalHeaderItem(6).text() == "平均状态"
    for row, record in enumerate(query_result.records):
        location = record.location
        assert dialog.record_at(row) == record
        assert dialog.table.item(row, 1).text() == str(location.node_id)
        assert dialog.table.item(row, 2).text() == (
            "" if location.element_id is None else str(location.element_id)
        )
        assert dialog.table.item(row, 3).text() == (
            ""
            if location.integration_point is None
            else str(location.integration_point)
        )
        assert dialog.table.item(row, 4).text() == (
            "" if location.local_node is None else str(location.local_node)
        )
        assert dialog.table.item(row, 5).text() == (
            ""
            if location.region_key is None
            else encode_result_region_key(location.region_key)
        )
        assert dialog.table.item(row, 6).text() == (
            "缺失"
            if location.averaged is None
            else "是"
            if location.averaged
            else "否"
        )
    assert "generation 1" in dialog.result_summary.text()

    stale = replace(query_result, materialization_generation=0)
    with pytest.raises(ValueError, match="stale"):
        dialog.show_result(stale)
    wrong_query = replace(
        query_result,
        query=replace(query, component="S22"),
    )
    with pytest.raises(ValueError, match="latest"):
        dialog.set_query_result(wrong_query)
    dialog.close()


def test_typed_dialog_ast_has_no_legacy_query_dependencies_or_support_map() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fem_gui"
        / "postprocessing_dialogs.py"
    )
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "TypedResultQueryDialog"
    )
    helper_names = {
        "_TypedQueryMode",
        "_typed_query_association_matches",
        "_parse_typed_query_ids",
        "_typed_field_label",
        "_typed_availability_text",
        "_optional_identity_text",
        "_averaged_text",
        "_number_text",
    }
    typed_nodes = [
        class_node,
        *[
            node
            for node in module.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in helper_names
        ],
    ]
    forbidden_names = {
        "ResultData",
        "QueryRecord",
        "available_components",
        "available_query_types",
        "parse_object_ids",
        "query_records",
    }
    names = {
        node.id
        for root in typed_nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Name)
    }
    assert names.isdisjoint(forbidden_names)
    assert not any(
        isinstance(node, (ast.Dict, ast.Set))
        for root in typed_nodes
        for node in ast.walk(root)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"materialize", "query"}
        for root in typed_nodes
        for node in ast.walk(root)
    )
    assert not any(
        isinstance(comparator, ast.Constant)
        and isinstance(comparator.value, str)
        for root in typed_nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Compare)
        for comparator in node.comparators
    )

    top_level_names = {
        node.name
        for node in module.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert top_level_names.isdisjoint(
        {
            "ResultQueryDialog",
            "ResultDisplayDialog",
            "ResultDisplaySettings",
            "_field_records",
            "_component_label",
        }
    )
    assert "visualization.query" not in source
    assert "result_adapter" not in source
