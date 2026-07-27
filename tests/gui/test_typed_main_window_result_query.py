from __future__ import annotations

import os
from pathlib import Path
from threading import Event
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.application import (
    ResultMaterializationTaskSnapshot,
    TokenStatus,
)
from fem.application.results import (
    FieldMaterializationKey,
    FieldState,
    ResultProvider,
    ResultQuery,
    ResultQueryResult,
    ScalarFieldSelection,
    restore_result_provider,
    build_solve_result_bundle,
)
from fem.solvers.static_linear import solve
from fem_gui.commands import (
    GuiCommandOutcome,
    GuiCommandReceipt,
    GuiCommandStatus,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.postprocessing_dialogs import TypedResultQueryDialog
from fem_gui.task_controller import (
    BackgroundTaskState,
    TaskApplyStatus,
    TaskCompletion,
)
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_renderer import ResultRenderPayload
from tests.helpers.model_builders import make_static_pull_truss_model
from tests.helpers.preflight_builders import passing_preflight_report


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def solved_window() -> FEMMainWindow:
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    imported = window.session.prepare_import(
        Path("typed-main-window-query.inp")
    )
    assert window._apply_session_delta(
        window.session.accept_imported_model(imported.token, model),
        model_geometry=build_model_geometry(model),
        source_label="typed-main-window-query.inp",
    )

    validation = window.session.prepare_validation("pull")
    assert window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )
    solve_task = window.session.prepare_solve("pull", "Typed-Query")
    assert solve_task.delta is not None
    assert window._apply_session_delta(solve_task.delta)
    assert window._apply_session_delta(
        window.session.begin_run(solve_task.token)
    )
    result = solve(
        solve_task.model,
        solve_task.step_name,
        name="Typed-Query",
    )
    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            solve_task.token,
            build_solve_result_bundle(solve_task, result),
        )
    )

    yield window
    window.close()


def _ready_selection(provider: ResultProvider) -> ScalarFieldSelection:
    selection = provider.catalog().default_selection
    assert (
        provider.field_status(selection.field_key).state
        is FieldState.READY
    )
    return selection


def _lazy_selection(provider: ResultProvider) -> ScalarFieldSelection:
    return _lazy_selections(provider)[0]


def _lazy_selections(
    provider: ResultProvider,
) -> tuple[ScalarFieldSelection, ...]:
    selections = tuple(
        ScalarFieldSelection(
            availability.key,
            availability.descriptor.default_component,
        )
        for availability in provider.catalog().fields
        if availability.state is FieldState.LAZY
    )
    if not selections:
        raise AssertionError("fixture must publish a LAZY result field")
    return selections


def _result_payload(window: FEMMainWindow) -> ResultRenderPayload:
    payload = window.viewport._result_render_payload
    assert type(payload) is ResultRenderPayload
    return payload


def _await_completion(
    window: FEMMainWindow,
    receipt: GuiCommandReceipt,
    *,
    timeout: float = 10.0,
) -> TaskCompletion:
    assert receipt.status is GuiCommandStatus.PENDING
    assert receipt.completion is not None
    deadline = monotonic() + timeout
    while (
        (not receipt.completion.done or window.busy)
        and monotonic() < deadline
    ):
        _application().processEvents()
        QThread.msleep(1)
    _application().processEvents()
    return receipt.completion.result(0.0)


def _succeed_additional_run(
    window: FEMMainWindow,
    run_name: str,
) -> str:
    task = window.session.prepare_solve("pull", run_name)
    assert task.delta is not None
    assert window._apply_session_delta(task.delta)
    assert window._apply_session_delta(window.session.begin_run(task.token))
    result = solve(task.model, task.step_name, name=run_name)
    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            task.token,
            build_solve_result_bundle(task, result),
        )
    )
    return task.run_id


def _capture_materialization_tasks(
    window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> list[ResultMaterializationTaskSnapshot]:
    tasks: list[ResultMaterializationTaskSnapshot] = []
    original = window.session.prepare_result_materialization

    def capture(run_id, field_keys):
        task = original(run_id, field_keys)
        tasks.append(task)
        return task

    monkeypatch.setattr(
        window.session,
        "prepare_result_materialization",
        capture,
    )
    return tasks


def _install_blocking_materializer(
    monkeypatch: pytest.MonkeyPatch,
):
    original = ResultProvider.materialize
    entered = Event()
    release = Event()
    calls: list[tuple[FieldMaterializationKey, ...]] = []

    def materialize(
        provider: ResultProvider,
        keys,
        *,
        cancellation=None,
    ):
        requested = tuple(keys)
        calls.append(requested)
        if cancellation is None:
            raise AssertionError(
                "LAZY query materialization requires cancellation"
            )
        patch = original(
            provider,
            requested,
            cancellation=cancellation,
        )
        entered.set()
        while not release.wait(0.001):
            cancellation.checkpoint()
        cancellation.checkpoint()
        return patch

    monkeypatch.setattr(ResultProvider, "materialize", materialize)
    return original, entered, release, calls


def test_ready_query_is_synchronous_filtered_and_revision_neutral(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    ready = _ready_selection(provider)
    query = ResultQuery(
        ready.field_key,
        ready.component,
        node_ids=(provider.snapshot.topology.node_ids[0],),
    )
    expected = provider.query(query)
    assert expected.records

    def fail_worker(*_args, **_kwargs):
        raise AssertionError("a READY query must not start a worker")

    monkeypatch.setattr(window, "_start_task", fail_worker)
    document = window.document
    revision = document.session_revision
    payload = _result_payload(window)
    delivered: list[ResultQueryResult] = []
    window.resultQueryCompleted.connect(delivered.append)

    receipt = window.query_result(query)

    assert receipt.status is GuiCommandStatus.ACCEPTED
    assert receipt.delta is None
    assert receipt.completion is None
    assert receipt.diagnostic is None
    assert type(receipt.outcome) is GuiCommandOutcome
    assert receipt.outcome.source == provider.source
    assert (
        receipt.outcome.materialization_generation
        == provider.snapshot.generation
    )
    assert receipt.outcome.selection == ready
    assert receipt.outcome.record_count == len(expected.records)
    assert window.document is document
    assert window.document.session_revision == revision
    assert window.session.snapshot().session_revision == revision
    assert window.result_provider is provider
    assert window.result_selection == selection
    assert _result_payload(window) is payload
    assert delivered == [expected]
    assert not window.busy


def test_ready_query_accepts_a_valid_zero_record_filter(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    ready = _ready_selection(provider)
    query = ResultQuery(
        ready.field_key,
        ready.component,
        element_ids=(provider.snapshot.topology.element_ids[0],),
    )
    expected = provider.query(query)
    assert expected.records == ()
    revision = window.document.session_revision

    receipt = window.query_result(query)

    assert receipt.status is GuiCommandStatus.ACCEPTED
    assert type(receipt.outcome) is GuiCommandOutcome
    assert receipt.outcome.record_count == 0
    assert receipt.outcome.source == provider.source
    assert receipt.outcome.selection == ready
    assert window.document.session_revision == revision
    assert window.result_provider is provider
    assert window.result_selection == selection


def test_lazy_query_materializes_only_its_field_and_sanitizes_completion(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    query = ResultQuery(lazy.field_key, lazy.component)
    revision = window.document.session_revision
    generation = provider.snapshot.generation
    payload = _result_payload(window)
    calls: list[tuple[FieldMaterializationKey, ...]] = []
    original_materialize = ResultProvider.materialize

    def capture_materialize(
        detached: ResultProvider,
        keys,
        *,
        cancellation=None,
    ):
        requested = tuple(keys)
        calls.append(requested)
        return original_materialize(
            detached,
            requested,
            cancellation=cancellation,
        )

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        capture_materialize,
    )
    delivered: list[ResultQueryResult] = []
    window.resultQueryCompleted.connect(delivered.append)

    receipt = window.query_result(query)

    assert receipt.status is GuiCommandStatus.PENDING
    assert receipt.delta is None
    assert receipt.outcome is None
    assert receipt.diagnostic is None
    assert receipt.completion is not None
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert terminal.apply_status is TaskApplyStatus.ACCEPTED
    assert terminal.value is None
    assert terminal.projection_error is None
    assert calls == [(lazy.field_key,)]
    outcome = receipt.completion.outcome
    assert type(outcome) is GuiCommandOutcome
    assert outcome.source == provider.source
    assert outcome.materialization_generation == generation + 1
    assert outcome.selection == lazy

    accepted_provider = window.result_provider
    assert type(accepted_provider) is ResultProvider
    assert accepted_provider is not provider
    assert (
        accepted_provider.field_status(lazy.field_key).state
        is FieldState.READY
    )
    assert (
        outcome.record_count
        == len(accepted_provider.query(query).records)
    )
    expected = accepted_provider.query(query)
    assert delivered == [expected]
    assert window.document.session_revision == revision + 1
    assert window.result_selection == selection
    accepted_payload = _result_payload(window)
    assert accepted_payload.topology.selection == selection
    assert accepted_payload.topology.source == provider.source
    assert (
        accepted_payload.topology.materialization_generation
        == generation + 1
    )
    assert payload.topology.selection == selection


def test_cancelled_lazy_query_consumes_token_and_preserves_projection(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    query = ResultQuery(lazy.field_key, lazy.component)
    tasks = _capture_materialization_tasks(window, monkeypatch)
    _original, entered, _release, calls = (
        _install_blocking_materializer(monkeypatch)
    )
    revision = window.document.session_revision
    generation = provider.snapshot.generation
    payload = _result_payload(window)
    delivered: list[ResultQueryResult] = []
    window.resultQueryCompleted.connect(delivered.append)

    receipt = window.query_result(query)

    assert receipt.status is GuiCommandStatus.PENDING
    assert entered.wait(5.0)
    assert window.cancel_current_task()
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.CANCELLED
    assert terminal.apply_status is None
    assert receipt.completion is not None
    assert receipt.completion.outcome is None
    assert calls == [(lazy.field_key,)]
    assert len(tasks) == 1
    assert (
        window.session.validate_task_token(tasks[0].token)
        is TokenStatus.ALREADY_COMPLETED
    )
    assert window.document.session_revision == revision
    record = window.session.current_result()
    assert record is not None
    assert record.materialization.generation == generation
    assert window.result_provider is provider
    assert window.result_selection == selection
    assert _result_payload(window) is payload
    assert delivered == []
    assert window._pending_result_query is None
    assert window._pending_result_query_source is None
    assert window._pending_result_query_generation is None


def test_hidden_run_lazy_query_updates_only_its_record_without_delivery(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    initial_a = window.result_provider
    assert type(initial_a) is ResultProvider
    run_a = initial_a.source.run_id
    run_b = _succeed_additional_run(window, "Typed-Query-Hidden-B")
    assert window.select_run_result(run_a).status is GuiCommandStatus.ACCEPTED
    provider_a = window.result_provider
    assert type(provider_a) is ResultProvider
    lazy_a = _lazy_selection(provider_a)
    query_a = ResultQuery(lazy_a.field_key, lazy_a.component)
    tasks = _capture_materialization_tasks(window, monkeypatch)
    _original, entered, release, calls = _install_blocking_materializer(
        monkeypatch
    )
    delivered: list[ResultQueryResult] = []
    window.resultQueryCompleted.connect(delivered.append)

    receipt = window.query_result(query_a)

    assert receipt.status is GuiCommandStatus.PENDING
    assert entered.wait(5.0)
    assert len(tasks) == 1
    task_a = tasks[0]
    assert task_a.run_id == run_a
    assert task_a.field_keys == (lazy_a.field_key,)
    assert window.select_run_result(run_b).status is GuiCommandStatus.ACCEPTED
    provider_b = window.result_provider
    selection_b = window.result_selection
    payload_b = _result_payload(window)
    revision_on_b = window.document.session_revision
    assert type(provider_b) is ResultProvider
    assert type(selection_b) is ScalarFieldSelection
    assert provider_b.source.run_id == run_b
    assert provider_b.snapshot.generation == 0

    release.set()
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert terminal.apply_status is TaskApplyStatus.ACCEPTED
    assert receipt.completion is not None
    outcome = receipt.completion.outcome
    assert type(outcome) is GuiCommandOutcome
    assert outcome.source == provider_a.source
    assert outcome.materialization_generation == 1
    assert outcome.selection == lazy_a
    assert calls == [(lazy_a.field_key,)]
    assert delivered == []
    assert window.document.session_revision == revision_on_b + 1
    assert window.document.displayed_result_run_id == run_b
    assert window.result_provider is provider_b
    assert window.result_selection == selection_b
    assert _result_payload(window) is payload_b
    current_b = window.session.current_result()
    assert current_b is not None
    assert current_b.provenance.run_id == run_b
    assert current_b.materialization.generation == 0

    projection_a = window.session.prepare_result_projection(run_a)
    assert projection_a.record.materialization.generation == 1
    accepted_a = restore_result_provider(
        projection_a.record.result,
        projection_a.record.materialization,
    )
    assert (
        accepted_a.field_status(lazy_a.field_key).state
        is FieldState.READY
    )
    assert outcome.record_count == len(accepted_a.query(query_a).records)
    assert window.session.accept_result_projection(
        projection_a.token
    ).accepted


def test_query_rejects_invalid_typed_filters_without_mutating_projection(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    ready = _ready_selection(provider)
    revision = window.document.session_revision
    payload = _result_payload(window)

    wrong_type = window.query_result(object())  # type: ignore[arg-type]
    unknown_node = window.query_result(
        ResultQuery(
            ready.field_key,
            ready.component,
            node_ids=(max(provider.snapshot.topology.node_ids) + 1,),
        )
    )
    unknown_component = window.query_result(
        ResultQuery(ready.field_key, "__missing_component__")
    )

    assert wrong_type.status is GuiCommandStatus.REJECTED
    assert wrong_type.diagnostic is not None
    assert wrong_type.diagnostic.code == "command.type.invalid"
    assert unknown_node.status is GuiCommandStatus.REJECTED
    assert unknown_node.diagnostic is not None
    assert (
        unknown_node.diagnostic.code
        == "result.query.unknown_node_ids"
    )
    assert unknown_component.status is GuiCommandStatus.REJECTED
    assert unknown_component.diagnostic is not None
    assert window.document.session_revision == revision
    assert window.result_provider is provider
    assert window.result_selection == selection
    assert _result_payload(window) is payload


def test_query_rejects_a_provider_outside_the_current_result_source(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    provider_a = window.result_provider
    assert type(provider_a) is ResultProvider
    ready_a = _ready_selection(provider_a)
    query_a = ResultQuery(ready_a.field_key, ready_a.component)
    run_b = _succeed_additional_run(window, "Typed-Query-B")
    provider_b = window.result_provider
    selection_b = window.result_selection
    assert type(provider_b) is ResultProvider
    assert type(selection_b) is ScalarFieldSelection
    assert provider_b.source.run_id == run_b
    payload_b = _result_payload(window)
    revision = window.document.session_revision

    window.result_provider = provider_a
    try:
        receipt = window.query_result(query_a)
    finally:
        window.result_provider = provider_b

    assert receipt.status is GuiCommandStatus.REJECTED
    assert receipt.diagnostic is not None
    assert receipt.diagnostic.code == "result.current.unavailable"
    assert window.document.session_revision == revision
    assert window.result_selection == selection_b
    assert _result_payload(window) is payload_b


def test_query_action_open_and_cancel_never_recovers_fields(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    assert type(provider) is ResultProvider
    revision = window.document.session_revision
    generation = provider.snapshot.generation
    opened: list[TypedResultQueryDialog] = []

    def fail_recovery(*_args, **_kwargs):
        raise AssertionError("opening a query dialog must not recover fields")

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        fail_recovery,
    )
    monkeypatch.setattr(
        TypedResultQueryDialog,
        "exec",
        lambda dialog: opened.append(dialog) or 0,
    )
    window.show_result_query_dialog()

    assert len(opened) == 1
    assert opened[0].source == provider.source
    assert opened[0].catalog is provider.catalog()
    assert window.document.session_revision == revision
    assert provider.snapshot.generation == generation


def test_query_dialog_source_switch_rejects_before_querying_new_run(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider_a = window.result_provider
    assert type(provider_a) is ResultProvider
    run_a = provider_a.source.run_id
    run_b = _succeed_additional_run(window, "Typed-Query-Dialog-B")
    assert window.select_run_result(run_a).status is GuiCommandStatus.ACCEPTED
    provider_a = window.result_provider
    assert type(provider_a) is ResultProvider
    ready_a = _ready_selection(provider_a)
    query_a = ResultQuery(ready_a.field_key, ready_a.component)
    opened: list[TypedResultQueryDialog] = []

    def fail_query(*_args, **_kwargs):
        raise AssertionError("a stale dialog must not query the new run")

    def switch_source_and_submit(dialog: TypedResultQueryDialog) -> int:
        opened.append(dialog)
        assert (
            window.select_run_result(run_b).status
            is GuiCommandStatus.ACCEPTED
        )
        dialog.queryRequested.emit(query_a)
        return 0

    monkeypatch.setattr(ResultProvider, "query", fail_query)
    monkeypatch.setattr(
        TypedResultQueryDialog,
        "exec",
        switch_source_and_submit,
    )

    window.show_result_query_dialog()

    provider_b = window.result_provider
    assert len(opened) == 1
    assert type(provider_b) is ResultProvider
    assert provider_b.source.run_id == run_b
    assert window.document.displayed_result_run_id == run_b
    assert not opened[0].query_pending
    assert "no longer current" in opened[0].result_summary.text()
