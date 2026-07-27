from __future__ import annotations

import ast
import os
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Callable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QComboBox

from fem.application.results import (
    FieldAvailability,
    FieldMaterializationKey,
    FieldPosition,
    FieldState,
    ResultCatalog,
    ResultProvider,
    ResultVariable,
    ScalarFieldSelection,
    build_solve_result_bundle,
    field_materialization_sort_key,
)
from fem.solvers.static_linear import solve
from fem_gui.commands import (
    GuiCommandOutcome,
    GuiCommandReceipt,
    GuiCommandStatus,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.task_controller import BackgroundTaskState
from fem_gui.visualization.model_adapter import build_model_geometry
from tests.helpers.model_builders import make_static_pull_truss_model
from tests.helpers.preflight_builders import passing_preflight_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_WINDOW_PATH = PROJECT_ROOT / "src" / "fem_gui" / "main_window.py"


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def solved_window() -> FEMMainWindow:
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    imported = window.session.prepare_import(Path("typed-result-ribbon.inp"))
    assert window._apply_session_delta(
        window.session.accept_imported_model(imported.token, model),
        model_geometry=build_model_geometry(model),
        source_label="typed-result-ribbon.inp",
    )

    validation = window.session.prepare_validation("pull")
    assert window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )
    solve_task = window.session.prepare_solve("pull", "Typed-Ribbon")
    assert solve_task.delta is not None
    assert window._apply_session_delta(solve_task.delta)
    assert window._apply_session_delta(
        window.session.begin_run(solve_task.token)
    )
    result = solve(
        solve_task.model,
        solve_task.step_name,
        name="Typed-Ribbon",
    )
    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            solve_task.token,
            build_solve_result_bundle(solve_task, result),
        )
    )

    yield window
    window.close()


def _combo_data(combo: QComboBox) -> tuple[object, ...]:
    return tuple(combo.itemData(index) for index in range(combo.count()))


def _selectable_fields(
    catalog: ResultCatalog,
) -> tuple[FieldAvailability, ...]:
    return tuple(
        availability
        for availability in catalog.fields
        if availability.state is not FieldState.UNAVAILABLE
    )


def _ordered_unique(values: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(dict.fromkeys(values))


def _catalog_variables(
    catalog: ResultCatalog,
) -> tuple[ResultVariable, ...]:
    return _ordered_unique(
        tuple(
            availability.descriptor.field_id.variable
            for availability in _selectable_fields(catalog)
        )
    )


def _catalog_positions(
    catalog: ResultCatalog,
    variable: ResultVariable,
) -> tuple[FieldPosition, ...]:
    return _ordered_unique(
        tuple(
            availability.descriptor.field_id.position
            for availability in _selectable_fields(catalog)
            if availability.descriptor.field_id.variable is variable
        )
    )


def _catalog_selections(
    catalog: ResultCatalog,
    variable: ResultVariable,
    position: FieldPosition,
) -> tuple[ScalarFieldSelection, ...]:
    return tuple(
        ScalarFieldSelection(availability.key, component)
        for availability in _selectable_fields(catalog)
        if availability.descriptor.field_id.variable is variable
        and availability.descriptor.field_id.position is position
        for component in availability.descriptor.columns
    )


def _accepted_without_projection(
    provider: ResultProvider,
) -> Callable[[ScalarFieldSelection], GuiCommandReceipt]:
    command_id = 10_000

    def submit(selection: ScalarFieldSelection) -> GuiCommandReceipt:
        nonlocal command_id
        command_id += 1
        return GuiCommandReceipt.accepted(
            command_id,
            outcome=GuiCommandOutcome(
                source=provider.source,
                materialization_generation=provider.snapshot.generation,
                selection=selection,
                record_count=0,
            ),
        )

    return submit


def _prepare_ribbon_selection(
    window: FEMMainWindow,
    selection: ScalarFieldSelection,
) -> int:
    provider = window._current_result_provider()
    assert provider is not None
    variable = selection.field_key.request.field_id.variable
    position = selection.field_key.request.field_id.position
    original_submit = window.select_result_field
    window.select_result_field = _accepted_without_projection(provider)
    try:
        variable_index = window.result_variable_combo.findData(variable)
        assert variable_index >= 0
        window.result_variable_combo.setCurrentIndex(variable_index)
        window._result_variable_changed(variable_index)

        position_index = window.result_position_combo.findData(position)
        assert position_index >= 0
        window.result_position_combo.setCurrentIndex(position_index)
        window._result_position_changed(position_index)

        component_index = window.result_component_combo.findData(selection)
        assert component_index >= 0
        window.result_component_combo.setCurrentIndex(component_index)
    finally:
        window.select_result_field = original_submit
    return component_index


def _process_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 10.0,
) -> None:
    application = _application()
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert predicate()


def test_ribbon_preserves_catalog_variable_position_and_selection_order(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    provider = window._current_result_provider()
    assert provider is not None
    catalog = provider.catalog()
    original_submit = window.select_result_field
    window.select_result_field = _accepted_without_projection(provider)
    try:
        window._refresh_result_controls()
        assert _combo_data(window.result_variable_combo) == (
            _catalog_variables(catalog)
        )
        assert all(
            type(value) is ResultVariable
            for value in _combo_data(window.result_variable_combo)
        )

        for variable in _catalog_variables(catalog):
            variable_index = window.result_variable_combo.findData(variable)
            window.result_variable_combo.setCurrentIndex(variable_index)
            window._result_variable_changed(variable_index)

            assert _combo_data(window.result_position_combo) == (
                _catalog_positions(catalog, variable)
            )
            assert all(
                type(value) is FieldPosition
                for value in _combo_data(window.result_position_combo)
            )

            for position in _catalog_positions(catalog, variable):
                position_index = window.result_position_combo.findData(
                    position
                )
                window.result_position_combo.setCurrentIndex(position_index)
                window._result_position_changed(position_index)

                assert _combo_data(window.result_component_combo) == (
                    _catalog_selections(catalog, variable, position)
                )
                assert all(
                    type(value) is ScalarFieldSelection
                    for value in _combo_data(window.result_component_combo)
                )
    finally:
        window.select_result_field = original_submit


def test_ribbon_keeps_complete_keys_distinct_for_one_request_and_position(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window._current_result_provider()
    assert provider is not None
    base_catalog = provider.catalog()
    base = next(
        availability
        for availability in base_catalog.fields
        if availability.state is FieldState.READY
    )
    historic_key = FieldMaterializationKey(
        base.key.request,
        base.key.recovery_contract + 100,
    )
    historic = FieldAvailability(
        historic_key,
        base.descriptor,
        base.state,
        base.diagnostics,
    )
    fields = tuple(
        sorted(
            (*base_catalog.fields, historic),
            key=lambda item: field_materialization_sort_key(item.key),
        )
    )
    augmented = ResultCatalog(
        source=base_catalog.source,
        fields=fields,
        default_selection=base_catalog.default_selection,
        diagnostics=base_catalog.diagnostics,
    )
    original_catalog = ResultProvider.catalog

    def catalog_for_provider(target: ResultProvider) -> ResultCatalog:
        if target is provider:
            return augmented
        return original_catalog(target)

    monkeypatch.setattr(ResultProvider, "catalog", catalog_for_provider)
    original_submit = window.select_result_field
    window.select_result_field = _accepted_without_projection(provider)
    try:
        window._refresh_result_controls()
        variable = base.descriptor.field_id.variable
        position = base.descriptor.field_id.position
        variable_index = window.result_variable_combo.findData(variable)
        window.result_variable_combo.setCurrentIndex(variable_index)
        window._result_variable_changed(variable_index)
        position_index = window.result_position_combo.findData(position)
        window.result_position_combo.setCurrentIndex(position_index)
        window._result_position_changed(position_index)

        selections = _combo_data(window.result_component_combo)
        expected = _catalog_selections(augmented, variable, position)
        assert selections == expected
        assert ScalarFieldSelection(
            base.key,
            base.descriptor.default_component,
        ) in selections
        assert ScalarFieldSelection(
            historic_key,
            base.descriptor.default_component,
        ) in selections
    finally:
        window.select_result_field = original_submit


def test_ready_ribbon_selection_uses_public_exact_selection_without_result_data(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    provider = window._current_result_provider()
    current = window.result_selection
    assert provider is not None
    assert type(current) is ScalarFieldSelection
    target = next(
        ScalarFieldSelection(availability.key, component)
        for availability in provider.catalog().fields
        if availability.state is FieldState.READY
        for component in availability.descriptor.columns
        if ScalarFieldSelection(availability.key, component) != current
    )

    window.result_data = None
    window._refresh_result_controls()
    component_index = _prepare_ribbon_selection(window, target)
    submitted: list[ScalarFieldSelection] = []
    original_submit = window.select_result_field

    def submit(selection: ScalarFieldSelection) -> GuiCommandReceipt:
        submitted.append(selection)
        return original_submit(selection)

    window.select_result_field = submit
    try:
        window._result_component_changed(component_index)
    finally:
        window.select_result_field = original_submit

    assert submitted == [target]
    assert window.result_selection == target
    assert window.result_component_combo.currentData() == target
    assert (
        window.viewport._result_render_payload.topology.selection
        == target
    )


def test_cancelled_lazy_ribbon_selection_restores_current_exact_selection(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window._current_result_provider()
    current = window.result_selection
    assert provider is not None
    assert type(current) is ScalarFieldSelection
    target = next(
        ScalarFieldSelection(
            availability.key,
            availability.descriptor.default_component,
        )
        for availability in provider.catalog().fields
        if availability.state is FieldState.LAZY
    )
    component_index = _prepare_ribbon_selection(window, target)

    worker_started = Event()
    worker_release = Event()
    original_materialize = ResultProvider.materialize

    def blocked_materialize(
        detached: ResultProvider,
        keys: tuple[FieldMaterializationKey, ...],
        *,
        cancellation: object | None = None,
    ):
        worker_started.set()
        while not worker_release.wait(0.001):
            if cancellation is not None:
                cancellation.checkpoint()
        if cancellation is not None:
            cancellation.checkpoint()
        return original_materialize(
            detached,
            keys,
            cancellation=cancellation,
        )

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        blocked_materialize,
    )
    submitted: list[ScalarFieldSelection] = []
    receipts: list[GuiCommandReceipt] = []
    original_submit = window.select_result_field

    def submit(selection: ScalarFieldSelection) -> GuiCommandReceipt:
        submitted.append(selection)
        receipt = original_submit(selection)
        receipts.append(receipt)
        return receipt

    window.select_result_field = submit
    try:
        window._result_component_changed(component_index)
        _process_until(worker_started.is_set)
        assert receipts[-1].status is GuiCommandStatus.PENDING
        assert window.result_selection == current
        window.cancel_current_task()
        worker_release.set()
        _process_until(lambda: not window.busy)
    finally:
        worker_release.set()
        window.select_result_field = original_submit

    completion = receipts[-1].completion
    assert completion is not None
    assert completion.terminal is not None
    assert completion.terminal.state is BackgroundTaskState.CANCELLED
    assert submitted == [target]
    assert window.result_selection == current
    assert window.result_component_combo.currentData() == current
    assert (
        window.viewport._result_render_payload.topology.selection
        == current
    )


def test_typed_ribbon_methods_do_not_read_legacy_result_maps_or_order() -> None:
    tree = ast.parse(
        MAIN_WINDOW_PATH.read_text(encoding="utf-8"),
        filename=str(MAIN_WINDOW_PATH),
    )
    window_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FEMMainWindow"
    )
    method_names = {
        "_refresh_result_controls",
        "_populate_result_positions",
        "_populate_result_components",
        "_result_variable_changed",
        "_result_position_changed",
        "_result_component_changed",
    }
    methods = tuple(
        node
        for node in window_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    )
    assert {method.name for method in methods} == method_names

    referenced_names = {
        name
        for method in methods
        for node in ast.walk(method)
        for name in (
            node.id
            if isinstance(node, ast.Name)
            else node.attr
            if isinstance(node, ast.Attribute)
            else None,
        )
        if name is not None
    }
    assert referenced_names.isdisjoint(
        {
            "ResultData",
            "result_data",
            "field_family",
            "_field_family",
            "_field_sort_key",
            "available_stress_prefixes",
            "stress_position_label",
        }
    )

    string_constants = {
        node.value
        for method in methods
        for node in ast.walk(method)
        if isinstance(node, ast.Constant)
        and type(node.value) is str
    }
    assert string_constants.isdisjoint(
        {
            "U",
            "U1",
            "U2",
            "U3",
            "R",
            "R1",
            "R2",
            "R3",
            "RF",
            "RF1",
            "RF2",
            "RF3",
            "RM1",
            "RM2",
            "RM3",
            "S",
            "S11",
            "S22",
            "S33",
            "S12",
            "S13",
            "S23",
            "Mises",
            "IP",
            "CENTROID",
            "EN",
            "NODAL",
        }
    )
