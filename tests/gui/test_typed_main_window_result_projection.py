from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
from threading import Event
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

import fem.application.runs as runs_module
from fem.application.results import (
    FieldMaterializationKey,
    FieldState,
    ResultProvider,
    ScalarFieldSelection,
    build_solve_result_bundle,
    restore_result_provider,
)
from fem.application.revisions import TokenStatus
from fem.solvers.static_linear import solve
from fem_gui.commands import (
    GuiCommandDiagnostic,
    GuiCommandOutcome,
    GuiCommandReceipt,
    GuiCommandStatus,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.postprocessing_dialogs import TypedResultDisplaySettings
from fem_gui.task_controller import (
    BackgroundTaskState,
    TaskApplyStatus,
    TaskCompletion,
)
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_renderer import ResultRenderPayload
from fem_gui.widgets.result_tree import (
    ROLE_MATERIALIZATION_KEY,
    ROLE_SELECTION,
)
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
        Path("typed-main-window-projection.inp")
    )
    assert window._apply_session_delta(
        window.session.accept_imported_model(imported.token, model),
        model_geometry=build_model_geometry(model),
        source_label="typed-main-window-projection.inp",
    )

    validation = window.session.prepare_validation("pull")
    assert window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )
    solve_task = window.session.prepare_solve(
        "pull",
        "Typed-Projection",
    )
    assert solve_task.delta is not None
    assert window._apply_session_delta(solve_task.delta)
    assert window._apply_session_delta(
        window.session.begin_run(solve_task.token)
    )
    result = solve(
        solve_task.model,
        solve_task.step_name,
        name="Typed-Projection",
    )
    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            solve_task.token,
            build_solve_result_bundle(solve_task, result),
        )
    )

    yield window
    window.close()


def _alternate_ready_selection(
    provider: ResultProvider,
    current: ScalarFieldSelection,
) -> ScalarFieldSelection:
    for availability in provider.catalog().fields:
        if availability.state is not FieldState.READY:
            continue
        for component in availability.descriptor.columns:
            candidate = ScalarFieldSelection(
                availability.key,
                component,
            )
            if candidate != current:
                return candidate
    raise AssertionError("fixture must publish an alternate READY component")


def _succeed_additional_run(
    window: FEMMainWindow,
    run_name: str,
) -> str:
    task = window.session.prepare_solve("pull", run_name)
    assert task.delta is not None
    assert window._apply_session_delta(task.delta)
    assert window._apply_session_delta(
        window.session.begin_run(task.token)
    )
    result = solve(
        task.model,
        task.step_name,
        name=run_name,
    )
    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            task.token,
            build_solve_result_bundle(task, result),
        )
    )
    return task.run_id


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
        raise AssertionError("fixture must publish at least one LAZY field")
    return selections


def _await_completion(
    window: FEMMainWindow,
    receipt: GuiCommandReceipt,
    *,
    timeout: float = 10.0,
) -> TaskCompletion:
    assert receipt.status is GuiCommandStatus.PENDING
    assert receipt.delta is None
    assert receipt.outcome is None
    assert receipt.diagnostic is None
    completion = receipt.completion
    assert completion is not None

    application = QApplication.instance() or QApplication([])
    deadline = monotonic() + timeout
    while (
        (not completion.done or window.busy)
        and monotonic() < deadline
    ):
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()

    terminal = completion.result(0.0)
    assert not window.busy
    return terminal


def _capture_materialization_tasks(
    window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> list[object]:
    tasks: list[object] = []
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
                "LAZY materialization must receive task cancellation"
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


def test_result_status_and_display_refresh_reuse_provider_without_detaching(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    assert type(provider) is ResultProvider
    identity = (provider.source, provider.snapshot.generation)
    assert window.session.current_result_identity() == identity

    def unexpected(*_args, **_kwargs):
        raise AssertionError(
            "result refresh must not detach or restore the accepted result"
        )

    monkeypatch.setattr(window.session, "current_result", unexpected)
    monkeypatch.setattr(
        runs_module,
        "restore_result_provider",
        unexpected,
    )
    monkeypatch.setattr(runs_module, "deep_owned_result", unexpected)
    monkeypatch.setattr(
        runs_module,
        "deep_owned_materialization",
        unexpected,
    )

    for _ in range(3):
        assert window._current_result_provider() is provider
        window._refresh_result_controls()
        window._update_action_states()
        window._apply_display()

    assert window.result_provider is provider
    assert window.session.current_result_identity() == identity


def _result_payload(window: FEMMainWindow) -> ResultRenderPayload:
    payload = window.viewport._result_render_payload
    assert type(payload) is ResultRenderPayload
    return payload


def _select_scale_mode(window: FEMMainWindow, mode: str) -> None:
    index = window.result_scale_combo.findData(mode)
    assert index >= 0
    window.result_scale_combo.setCurrentIndex(index)
    window.result_scale_combo.activated.emit(index)


def test_successful_solve_projects_one_typed_result_spine(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    record = window.session.current_result()
    assert record is not None

    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    assert provider.source == record.materialization.source
    assert provider.snapshot.generation == record.materialization.generation
    assert selection == provider.catalog().default_selection

    root = window.result_tree.topLevelItem(0)
    step = root.child(0)
    assert root.text(0) == "分析结果"
    assert [
        step.child(index).data(0, ROLE_MATERIALIZATION_KEY)
        for index in range(step.childCount())
    ] == [item.key for item in provider.catalog().fields]
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == selection
    )

    inspection_service = window.inspection_service
    assert inspection_service is not None
    assert inspection_service.result_provider is provider
    payload = _result_payload(window)
    assert payload.topology.source == provider.source
    assert (
        payload.topology.materialization_generation
        == provider.snapshot.generation
    )
    assert payload.topology.selection == selection

    assert not hasattr(window, "result_data")
    node_id = provider.snapshot.topology.node_ids[0]
    inspection = inspection_service.inspect("node", node_id)
    assert any(page.title == "结果" for page in inspection.pages)


def test_ready_selection_is_revision_neutral_and_idempotent(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    current = window.result_selection
    assert type(provider) is ResultProvider
    assert type(current) is ScalarFieldSelection
    selection = _alternate_ready_selection(provider, current)

    def fail_worker(*_args, **_kwargs):
        raise AssertionError("READY field selection must not start a worker")

    monkeypatch.setattr(window, "_start_task", fail_worker)
    revision = window.document.session_revision
    document = window.document
    original_payload = _result_payload(window)

    receipt = window.select_result_field(selection)

    assert receipt.status is GuiCommandStatus.ACCEPTED
    assert receipt.delta is None
    assert type(receipt.outcome) is GuiCommandOutcome
    assert receipt.outcome.source == provider.source
    assert (
        receipt.outcome.materialization_generation
        == provider.snapshot.generation
    )
    assert receipt.outcome.selection == selection
    assert receipt.diagnostic is None
    assert receipt.completion is None
    assert window.document is document
    assert window.document.session_revision == revision
    assert window.session.snapshot().session_revision == revision
    assert window.result_provider is provider
    assert window.result_selection == selection
    selected_payload = _result_payload(window)
    assert selected_payload is not original_payload
    assert selected_payload.topology.selection == selection
    assert not window.busy

    repeated = window.select_result_field(selection)

    assert repeated.status is GuiCommandStatus.ACCEPTED
    assert repeated.delta is None
    assert repeated.outcome == receipt.outcome
    assert window.document is document
    assert window.document.session_revision == revision
    assert window.result_provider is provider
    assert window.result_selection == selection
    assert _result_payload(window) is selected_payload
    assert not window.busy


def test_reopening_current_run_preserves_exact_typed_selection(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    provider = window.result_provider
    current = window.result_selection
    assert type(provider) is ResultProvider
    assert type(current) is ScalarFieldSelection
    selection = _alternate_ready_selection(provider, current)
    receipt = window.select_result_field(selection)
    assert receipt.status is GuiCommandStatus.ACCEPTED
    payload = _result_payload(window)

    job = window.session.find_run(provider.source.run_id)
    assert job is not None
    window._activate_job_result(job)

    assert window.result_provider is provider
    assert window.result_selection == selection
    assert _result_payload(window).topology.selection == selection
    assert _result_payload(window) is not payload
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == selection
    )
    assert window.result_tree.currentItem().childCount() == 0


def test_ready_renderer_failure_restores_previous_typed_scene(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    current = window.result_selection
    assert type(provider) is ResultProvider
    assert type(current) is ScalarFieldSelection
    selection = _alternate_ready_selection(provider, current)
    payload = _result_payload(window)
    tree_selection = window.result_tree.currentItem().data(
        0,
        ROLE_SELECTION,
    )
    original_set_display = window.viewport.set_display

    def fail_after_render(
        shape_mode: str,
        contour_enabled: bool,
    ) -> None:
        original_set_display(shape_mode, contour_enabled)
        raise RuntimeError("injected renderer failure")

    monkeypatch.setattr(
        window.viewport,
        "set_display",
        fail_after_render,
    )

    receipt = window.select_result_field(selection)

    assert receipt.status is GuiCommandStatus.REJECTED
    assert receipt.diagnostic is not None
    assert receipt.diagnostic.code == "result.field.projection.failed"
    assert window.result_provider is provider
    assert window.result_selection == current
    assert window.viewport._result_render_payload is payload
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == tree_selection
    )


def test_ready_activation_failure_restores_the_installed_tree_selection(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    current = window.result_selection
    assert type(provider) is ResultProvider
    assert type(current) is ScalarFieldSelection
    selection = _alternate_ready_selection(provider, current)
    payload = _result_payload(window)
    assert window.result_tree.select_selection(selection)
    original_set_display = window.viewport.set_display

    def fail_after_render(
        shape_mode: str,
        contour_enabled: bool,
    ) -> None:
        original_set_display(shape_mode, contour_enabled)
        raise RuntimeError("injected activation renderer failure")

    monkeypatch.setattr(
        window.viewport,
        "set_display",
        fail_after_render,
    )
    receipts: list[GuiCommandReceipt] = []
    original_select_result_field = window.select_result_field

    def capture_selection(
        requested: ScalarFieldSelection,
    ) -> GuiCommandReceipt:
        receipt = original_select_result_field(requested)
        receipts.append(receipt)
        return receipt

    monkeypatch.setattr(
        window,
        "select_result_field",
        capture_selection,
    )

    window._activate_result_selection(selection)

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.status is GuiCommandStatus.REJECTED
    assert receipt.diagnostic is not None
    assert receipt.diagnostic.code == "result.field.projection.failed"
    assert window.result_provider is provider
    assert window.result_selection == current
    assert _result_payload(window) is payload
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == current
    )


def test_first_provider_projection_failure_restores_the_model_only_scene(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    inspection = window.inspection_service
    assert inspection is not None
    window._clear_result_projection()
    assert window.result_provider is None
    assert window.result_selection is None
    assert window.result_tree.catalog is None
    assert inspection.result_provider is None
    assert window.viewport._result_render_payload is None
    assert window.viewport.run_id is None
    original_set_display = window.viewport.set_display

    def fail_after_render(
        shape_mode: str,
        contour_enabled: bool,
    ) -> None:
        original_set_display(shape_mode, contour_enabled)
        raise RuntimeError("injected first projection failure")

    monkeypatch.setattr(
        window.viewport,
        "set_display",
        fail_after_render,
    )

    with pytest.raises(
        RuntimeError,
        match="injected first projection failure",
    ):
        window._rebuild_full_projection()

    assert window.result_provider is None
    assert window.result_selection is None
    assert not hasattr(window, "result_data")
    assert window.result_tree.catalog is None
    assert inspection.result_provider is None
    assert window.viewport._result_render_payload is None


def test_typed_display_settings_reject_a_stale_dialog_source(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    provider = window.result_provider
    current = window.result_selection
    assert type(provider) is ResultProvider
    assert type(current) is ScalarFieldSelection
    selection = _alternate_ready_selection(provider, current)
    payload = _result_payload(window)
    settings = TypedResultDisplaySettings(
        shape_mode="deformed",
        contour_enabled=True,
        selection=selection,
        scale_mode="custom",
        scale_value=3.5,
        overlay_undeformed=False,
        show_edges=True,
    )

    window._apply_typed_result_display_settings(
        settings,
        expected_source=replace(
            provider.source,
            run_id="stale-dialog-run",
        ),
    )

    assert window.result_provider is provider
    assert window.result_selection == current
    assert window.viewport._result_render_payload is payload
    assert window._scale_mode != "custom"


def test_typed_display_rejects_a_second_apply_while_lazy_work_is_pending(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    ready = window.result_selection
    assert type(provider) is ResultProvider
    assert type(ready) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    pending_settings = TypedResultDisplaySettings(
        shape_mode="undeformed",
        contour_enabled=True,
        selection=lazy,
        scale_mode="auto",
        scale_value=1.0,
        overlay_undeformed=True,
        show_edges=False,
    )
    later_settings = TypedResultDisplaySettings(
        shape_mode="deformed",
        contour_enabled=True,
        selection=ready,
        scale_mode="custom",
        scale_value=7.0,
        overlay_undeformed=False,
        show_edges=True,
    )
    _original, entered, release, calls = _install_blocking_materializer(
        monkeypatch
    )
    payload = _result_payload(window)
    display = window._display
    receipts: list[GuiCommandReceipt] = []
    original_select_result_field = window.select_result_field

    def capture_selection(
        requested: ScalarFieldSelection,
    ) -> GuiCommandReceipt:
        receipt = original_select_result_field(requested)
        receipts.append(receipt)
        return receipt

    monkeypatch.setattr(
        window,
        "select_result_field",
        capture_selection,
    )

    window._apply_typed_result_display_settings(
        pending_settings,
        expected_source=provider.source,
    )

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.status is GuiCommandStatus.PENDING
    assert entered.wait(5.0)

    window._apply_typed_result_display_settings(
        later_settings,
        expected_source=provider.source,
    )

    assert len(receipts) == 1
    assert window.result_selection == ready
    assert _result_payload(window) is payload
    assert window._display == display

    release.set()
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert terminal.projection_error is None
    assert calls == [(lazy.field_key,)]
    assert window.result_selection == lazy
    assert _result_payload(window).topology.selection == lazy
    assert window._display.shape_mode == pending_settings.shape_mode
    assert (
        window._display.contour_enabled
        == pending_settings.contour_enabled
    )
    assert window._scale_mode == pending_settings.scale_mode
    assert window._overlay_undeformed == pending_settings.overlay_undeformed


def test_invalid_selection_rejections_preserve_projection(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    foreign_key = FieldMaterializationKey(
        selection.field_key.request,
        selection.field_key.recovery_contract + 10_000,
    )
    rejected_selections = (
        ScalarFieldSelection(foreign_key, selection.component),
        ScalarFieldSelection(selection.field_key, "NotAComponent"),
    )

    def fail_worker(*_args, **_kwargs):
        raise AssertionError("rejected selection must not start a worker")

    monkeypatch.setattr(window, "_start_task", fail_worker)
    revision = window.document.session_revision
    payload = _result_payload(window)

    for rejected_selection in rejected_selections:
        receipt = window.select_result_field(rejected_selection)

        assert receipt.status is GuiCommandStatus.REJECTED
        assert receipt.delta is None
        assert receipt.outcome is None
        assert type(receipt.diagnostic) is GuiCommandDiagnostic
        assert receipt.diagnostic.code.strip()
        assert receipt.diagnostic.message.strip()
        assert receipt.completion is None
        assert window.document.session_revision == revision
        assert window.session.snapshot().session_revision == revision
        assert window.result_provider is provider
        assert window.result_selection == selection
        assert _result_payload(window) is payload
        assert not window.busy


def test_lazy_selection_returns_pending_and_commits_exact_generation(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    tasks = _capture_materialization_tasks(window, monkeypatch)
    _original, entered, release, calls = _install_blocking_materializer(
        monkeypatch
    )
    revision = window.document.session_revision
    generation = provider.snapshot.generation
    payload = _result_payload(window)

    receipt = window.select_result_field(lazy)

    assert receipt.status is GuiCommandStatus.PENDING
    assert receipt.delta is None
    assert receipt.outcome is None
    assert receipt.diagnostic is None
    assert receipt.completion is not None
    assert receipt.completion.command_id == receipt.command_id
    assert receipt.completion.task_id is not None
    assert entered.wait(5.0)
    assert window.busy
    assert window.document.session_revision == revision
    assert window.result_provider is provider
    assert window.result_selection == selection
    assert _result_payload(window) is payload
    assert calls == [(lazy.field_key,)]
    assert len(tasks) == 1
    task = tasks[0]
    assert task.run_id == provider.source.run_id
    assert task.field_keys == (lazy.field_key,)
    assert dict(task.token.dependency_revisions) == {
        "materialization_generation": generation,
        "model_revision": window.document.model_revision,
    }

    release.set()
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert terminal.apply_status is TaskApplyStatus.ACCEPTED
    assert terminal.value is None
    outcome = receipt.completion.outcome
    assert type(outcome) is GuiCommandOutcome
    assert outcome.source == provider.source
    assert outcome.materialization_generation == generation + 1
    assert outcome.selection == lazy
    assert (
        window.session.validate_task_token(task.token)
        is TokenStatus.ALREADY_COMPLETED
    )
    assert window.document.session_revision == revision + 1
    assert window.session.snapshot().session_revision == revision + 1
    record = window.session.current_result()
    accepted_provider = window.result_provider
    assert record is not None
    assert type(accepted_provider) is ResultProvider
    assert accepted_provider is not provider
    assert (
        record.materialization.generation
        == accepted_provider.snapshot.generation
        == generation + 1
    )
    assert (
        accepted_provider.field_status(lazy.field_key).state
        is FieldState.READY
    )
    assert (
        outcome.record_count
        == len(accepted_provider.field(lazy.field_key).locations)
    )
    assert window.result_selection == lazy
    accepted_payload = _result_payload(window)
    assert accepted_payload is not payload
    assert accepted_payload.topology.source == provider.source
    assert accepted_payload.topology.materialization_generation == generation + 1
    assert accepted_payload.topology.selection == lazy
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == lazy
    )


def test_lazy_renderer_failure_keeps_selection_and_does_not_enable_contour(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    _original, entered, release, calls = _install_blocking_materializer(
        monkeypatch
    )
    display = window._display
    original_set_payload = window.viewport.set_result_render_payload

    def reject_lazy_payload(payload: ResultRenderPayload) -> None:
        if payload.topology.selection == lazy:
            raise RuntimeError("injected lazy renderer failure")
        original_set_payload(payload)

    monkeypatch.setattr(
        window.viewport,
        "set_result_render_payload",
        reject_lazy_payload,
    )
    receipts: list[GuiCommandReceipt] = []
    original_select_result_field = window.select_result_field

    def capture_selection(
        requested: ScalarFieldSelection,
    ) -> GuiCommandReceipt:
        receipt = original_select_result_field(requested)
        receipts.append(receipt)
        return receipt

    monkeypatch.setattr(
        window,
        "select_result_field",
        capture_selection,
    )

    window._activate_result_selection(lazy)

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.status is GuiCommandStatus.PENDING
    assert entered.wait(5.0)
    release.set()
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert terminal.apply_status is TaskApplyStatus.ACCEPTED
    assert terminal.projection_error is not None
    assert "injected lazy renderer failure" in terminal.projection_error
    assert receipt.completion is not None
    outcome = receipt.completion.outcome
    assert type(outcome) is GuiCommandOutcome
    assert outcome.source == provider.source
    assert outcome.selection == lazy
    assert calls == [(lazy.field_key,)]
    accepted_provider = window.result_provider
    assert type(accepted_provider) is ResultProvider
    assert accepted_provider.source == provider.source
    assert accepted_provider.snapshot.generation == outcome.materialization_generation
    assert (
        accepted_provider.field_status(lazy.field_key).state
        is FieldState.READY
    )
    assert window.result_selection == selection
    assert _result_payload(window).topology.selection == selection
    assert window._display == display
    assert window.actions["contour"].isChecked() == display.contour_enabled
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == selection
    )
    assert window._pending_result_selection is None
    assert window._pending_result_source is None
    assert window._pending_result_generation is None


def test_lazy_generation_refresh_failure_keeps_all_consumers_on_old_generation(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    inspection = window.inspection_service
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    assert inspection is not None
    generation = provider.snapshot.generation
    lazy = _lazy_selection(provider)
    payload = _result_payload(window)
    display = window._display
    _original, entered, release, calls = _install_blocking_materializer(
        monkeypatch
    )
    original_set_display = window.viewport.set_display

    def reject_new_generation(
        shape_mode: str,
        contour_enabled: bool,
    ) -> None:
        original_set_display(shape_mode, contour_enabled)
        candidate = window.viewport._result_render_payload
        if (
            candidate is not None
            and candidate.topology.materialization_generation
            == generation + 1
        ):
            raise RuntimeError("injected generation refresh failure")

    monkeypatch.setattr(
        window.viewport,
        "set_display",
        reject_new_generation,
    )
    assert window.result_tree.select_selection(lazy)
    receipts: list[GuiCommandReceipt] = []
    original_select_result_field = window.select_result_field

    def capture_selection(
        requested: ScalarFieldSelection,
    ) -> GuiCommandReceipt:
        receipt = original_select_result_field(requested)
        receipts.append(receipt)
        return receipt

    monkeypatch.setattr(
        window,
        "select_result_field",
        capture_selection,
    )

    window._activate_result_selection(lazy)

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.status is GuiCommandStatus.PENDING
    assert entered.wait(5.0)
    release.set()
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert terminal.apply_status is TaskApplyStatus.ACCEPTED
    assert terminal.projection_error is not None
    assert terminal.rebuild_error is not None
    assert "injected generation refresh failure" in terminal.projection_error
    assert "injected generation refresh failure" in terminal.rebuild_error
    assert calls == [(lazy.field_key,)]
    record = window.session.current_result()
    assert record is not None
    assert record.materialization.generation == generation + 1
    assert window.result_provider is provider
    assert window.result_selection == selection
    assert inspection.result_provider is provider
    assert _result_payload(window) is payload
    assert payload.topology.materialization_generation == generation
    assert window._display == display
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == selection
    )
    assert window._current_result_provider() is None
    assert window._pending_result_selection is None
    assert window._pending_result_source is None
    assert window._pending_result_generation is None

    monkeypatch.setattr(
        window.viewport,
        "set_display",
        original_set_display,
    )
    window._rebuild_full_projection()

    rebuilt = window._current_result_provider()
    assert type(rebuilt) is ResultProvider
    assert rebuilt.snapshot.generation == generation + 1
    assert window.result_selection == selection
    assert inspection.result_provider is rebuilt
    assert (
        _result_payload(window).topology.materialization_generation
        == generation + 1
    )
    retry = original_select_result_field(lazy)
    assert retry.status is GuiCommandStatus.ACCEPTED
    assert window.result_selection == lazy


def test_full_rebuild_repairs_same_run_stale_viewport_provenance(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    provider = window.result_provider
    assert type(provider) is ResultProvider
    generation = provider.snapshot.generation
    lazy = _lazy_selection(provider)
    stale_payload = _result_payload(window)

    receipt = window.select_result_field(lazy)
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    current = window._current_result_provider()
    assert type(current) is ResultProvider
    assert current.snapshot.generation == generation + 1
    assert window.result_selection == lazy
    assert window.result_tree.catalog is current.catalog()
    fresh_payload = _result_payload(window)
    assert fresh_payload.topology.selection == lazy
    assert (
        fresh_payload.topology.materialization_generation
        == generation + 1
    )

    window.viewport.set_result_render_payload(stale_payload)
    window.viewport.set_display(
        window._display.shape_mode,
        window._display.contour_enabled,
    )

    assert _result_payload(window) is stale_payload
    assert stale_payload.topology.source == current.source
    assert stale_payload.topology.materialization_generation == generation
    window._rebuild_full_projection()

    repaired_provider = window._current_result_provider()
    repaired_payload = _result_payload(window)
    assert type(repaired_provider) is ResultProvider
    assert repaired_provider.snapshot.generation == generation + 1
    assert repaired_provider.source == current.source
    assert window.result_selection == lazy
    assert window.result_tree.catalog is repaired_provider.catalog()
    assert repaired_payload is not stale_payload
    assert repaired_payload.topology.source == repaired_provider.source
    assert (
        repaired_payload.topology.materialization_generation
        == repaired_provider.snapshot.generation
    )
    assert repaired_payload.topology.selection == lazy


def test_duplicate_lazy_selection_and_ready_selection_share_the_busy_gate(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    ready = window.result_selection
    assert type(provider) is ResultProvider
    assert type(ready) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    _original, entered, release, calls = _install_blocking_materializer(
        monkeypatch
    )
    payload = _result_payload(window)
    screenshot_enabled = window.actions["screenshot"].isEnabled()

    pending = window.select_result_field(lazy)

    assert pending.status is GuiCommandStatus.PENDING
    assert entered.wait(5.0)
    duplicate = window.select_result_field(lazy)
    conflicting_ready = window.select_result_field(ready)

    for rejected in (duplicate, conflicting_ready):
        assert rejected.status is GuiCommandStatus.REJECTED
        assert rejected.delta is None
        assert rejected.outcome is None
        assert rejected.completion is None
        assert rejected.diagnostic is not None
        assert rejected.diagnostic.code == "task.busy"
    assert duplicate.command_id != pending.command_id
    assert conflicting_ready.command_id != duplicate.command_id
    assert calls == [(lazy.field_key,)]
    assert window.result_provider is provider
    assert window.result_selection == ready
    assert _result_payload(window) is payload
    pending_actions = {
        action_name: window.actions[action_name].isEnabled()
        for action_name in (
            "field",
            "query",
            "export_csv",
            "export_vtk",
            "screenshot",
        )
    }

    release.set()
    terminal = _await_completion(window, pending)

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert pending_actions == {
        "field": False,
        "query": False,
        "export_csv": False,
        "export_vtk": False,
        "screenshot": screenshot_enabled,
    }
    assert window.result_selection == lazy
    assert calls == [(lazy.field_key,)]
    for action_name in ("field", "query", "export_csv", "export_vtk"):
        assert window.actions[action_name].isEnabled()
    assert window.actions["screenshot"].isEnabled() == screenshot_enabled


def test_lazy_worker_start_failure_consumes_token_and_clears_pending_state(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    tasks = _capture_materialization_tasks(window, monkeypatch)
    payload = _result_payload(window)
    revision = window.document.session_revision
    assert window.result_tree.select_selection(lazy)
    receipts: list[GuiCommandReceipt] = []
    original_select_result_field = window.select_result_field

    def capture_selection(
        requested: ScalarFieldSelection,
    ) -> GuiCommandReceipt:
        receipt = original_select_result_field(requested)
        receipts.append(receipt)
        return receipt

    monkeypatch.setattr(
        window,
        "select_result_field",
        capture_selection,
    )
    monkeypatch.setattr(
        window,
        "_start_task",
        lambda *_args, **_kwargs: False,
    )

    window._activate_result_selection(lazy)

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.status is GuiCommandStatus.REJECTED
    assert receipt.diagnostic is not None
    assert receipt.diagnostic.code == "task.start.rejected"
    assert len(tasks) == 1
    assert (
        window.session.validate_task_token(tasks[0].token)
        is TokenStatus.ALREADY_COMPLETED
    )
    assert window.document.session_revision == revision
    assert window.result_provider is provider
    assert window.result_selection == selection
    assert _result_payload(window) is payload
    assert window._pending_result_selection is None
    assert window._pending_result_source is None
    assert window._pending_result_generation is None
    assert not window.busy
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == selection
    )


def test_cancelled_lazy_selection_consumes_token_and_preserves_projection(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    tasks = _capture_materialization_tasks(window, monkeypatch)
    _original, entered, _release, calls = _install_blocking_materializer(
        monkeypatch
    )
    revision = window.document.session_revision
    generation = provider.snapshot.generation
    payload = _result_payload(window)
    assert window.result_tree.select_selection(lazy)
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == lazy
    )

    receipt = window.select_result_field(lazy)

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
    assert provider.field_status(lazy.field_key).state is FieldState.LAZY
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == selection
    )


def test_failed_lazy_selection_consumes_token_and_preserves_projection(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    tasks = _capture_materialization_tasks(window, monkeypatch)
    calls: list[tuple[FieldMaterializationKey, ...]] = []

    def fail_materialization(
        _provider: ResultProvider,
        keys,
        *,
        cancellation=None,
    ):
        calls.append(tuple(keys))
        assert cancellation is not None
        raise RuntimeError("injected lazy materialization failure")

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        fail_materialization,
    )
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: shown.append((title, message)),
    )
    revision = window.document.session_revision
    generation = provider.snapshot.generation
    payload = _result_payload(window)
    assert window.result_tree.select_selection(lazy)
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == lazy
    )

    receipt = window.select_result_field(lazy)
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.FAILED
    assert terminal.apply_status is None
    assert "injected lazy materialization failure" in terminal.message
    assert receipt.completion is not None
    assert receipt.completion.outcome is None
    assert calls == [(lazy.field_key,)]
    assert len(tasks) == 1
    assert (
        window.session.validate_task_token(tasks[0].token)
        is TokenStatus.ALREADY_COMPLETED
    )
    assert shown
    assert "injected lazy materialization failure" in shown[0][1]
    assert window.document.session_revision == revision
    record = window.session.current_result()
    assert record is not None
    assert record.materialization.generation == generation
    assert window.result_provider is provider
    assert window.result_selection == selection
    assert _result_payload(window) is payload
    assert provider.field_status(lazy.field_key).state is FieldState.LAZY
    assert (
        window.result_tree.currentItem().data(0, ROLE_SELECTION)
        == selection
    )


def test_old_generation_lazy_completion_is_discarded_without_overwrite(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy_selections = _lazy_selections(provider)
    assert len(lazy_selections) >= 2
    target, competing = lazy_selections[:2]
    tasks = _capture_materialization_tasks(window, monkeypatch)
    original, entered, release, calls = _install_blocking_materializer(
        monkeypatch
    )
    revision = window.document.session_revision
    generation = provider.snapshot.generation

    receipt = window.select_result_field(target)

    assert receipt.status is GuiCommandStatus.PENDING
    assert entered.wait(5.0)
    assert len(tasks) == 1
    stale_task = tasks[0]

    competing_task = window.session.prepare_result_materialization(
        provider.source.run_id,
        (competing.field_key,),
    )
    competing_provider = restore_result_provider(
        competing_task.record.result,
        competing_task.record.materialization,
    )
    competing_patch = original(
        competing_provider,
        competing_task.field_keys,
    )
    accepted = window.session.accept_result_materialization(
        competing_task.token,
        competing_patch,
    )
    assert accepted.accepted
    assert window._apply_session_delta(accepted)
    accepted_provider = window.result_provider
    accepted_payload = _result_payload(window)
    assert type(accepted_provider) is ResultProvider
    assert accepted_provider.snapshot.generation == generation + 1
    assert window.result_selection == selection
    assert (
        accepted_provider.field_status(competing.field_key).state
        is FieldState.READY
    )
    assert (
        window.session.validate_task_token(stale_task.token)
        is TokenStatus.STALE_REVISION
    )

    release.set()
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.DISCARDED
    assert terminal.apply_status is TaskApplyStatus.STALE
    assert receipt.completion is not None
    assert receipt.completion.outcome is None
    assert calls == [(target.field_key,)]
    assert window.document.session_revision == revision + 1
    assert window.result_provider is accepted_provider
    assert window.result_selection == selection
    assert _result_payload(window) is accepted_payload
    assert (
        accepted_provider.field_status(target.field_key).state
        is FieldState.LAZY
    )


def test_hidden_run_lazy_completion_updates_only_its_bound_record(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    initial_a = window.result_provider
    assert type(initial_a) is ResultProvider
    run_a = initial_a.source.run_id
    run_b = _succeed_additional_run(window, "Typed-Hidden-B")
    assert run_b != run_a
    switched_to_a = window.select_run_result(run_a)
    assert switched_to_a.status is GuiCommandStatus.ACCEPTED
    provider_a = window.result_provider
    selection_a = window.result_selection
    assert type(provider_a) is ResultProvider
    assert type(selection_a) is ScalarFieldSelection
    assert provider_a.source.run_id == run_a
    lazy_a = _lazy_selection(provider_a)
    tasks = _capture_materialization_tasks(window, monkeypatch)
    _original, entered, release, calls = _install_blocking_materializer(
        monkeypatch
    )

    receipts: list[GuiCommandReceipt] = []
    original_select_result_field = window.select_result_field

    def capture_selection(selection: ScalarFieldSelection) -> GuiCommandReceipt:
        receipt = original_select_result_field(selection)
        receipts.append(receipt)
        return receipt

    monkeypatch.setattr(
        window,
        "select_result_field",
        capture_selection,
    )
    window._activate_result_selection(lazy_a)
    assert len(receipts) == 1
    receipt = receipts[0]

    assert receipt.status is GuiCommandStatus.PENDING
    assert entered.wait(5.0)
    assert len(tasks) == 1
    task_a = tasks[0]
    assert task_a.run_id == run_a
    assert task_a.field_keys == (lazy_a.field_key,)

    switched_to_b = window.select_run_result(run_b)
    assert switched_to_b.status is GuiCommandStatus.ACCEPTED
    provider_b = window.result_provider
    selection_b = window.result_selection
    payload_b = _result_payload(window)
    revision_on_b = window.document.session_revision
    assert type(provider_b) is ResultProvider
    assert type(selection_b) is ScalarFieldSelection
    assert provider_b.source.run_id == run_b
    assert provider_b.snapshot.generation == 0
    assert window.document.displayed_result_run_id == run_b
    assert (
        window.session.validate_task_token(task_a.token)
        is TokenStatus.CURRENT
    )
    display_calls: list[tuple[str, bool]] = []
    original_set_display = window.viewport.set_display

    def record_set_display(
        shape_mode: str,
        contour_enabled: bool,
    ) -> None:
        display_calls.append((shape_mode, contour_enabled))
        original_set_display(shape_mode, contour_enabled)

    monkeypatch.setattr(
        window.viewport,
        "set_display",
        record_set_display,
    )

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
    assert window.document.session_revision == revision_on_b + 1
    assert window.document.displayed_result_run_id == run_b
    assert window.result_provider is provider_b
    assert window.result_selection == selection_b
    assert _result_payload(window) is payload_b
    assert display_calls == []
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
    assert window.session.accept_result_projection(
        projection_a.token
    ).accepted


def test_invalidated_run_lazy_completion_cannot_restore_old_projection(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    tasks = _capture_materialization_tasks(window, monkeypatch)
    _original, entered, release, calls = _install_blocking_materializer(
        monkeypatch
    )

    receipt = window.select_result_field(lazy)

    assert receipt.status is GuiCommandStatus.PENDING
    assert entered.wait(5.0)
    assert len(tasks) == 1
    stale_task = tasks[0]
    closed = window.session.close(
        expected_session_revision=window.document.session_revision
    )
    assert window._apply_session_delta(closed)
    closed_revision = window.document.session_revision
    assert window.result_provider is None
    assert window.result_selection is None
    assert window.viewport._result_render_payload is None
    assert (
        window.session.validate_task_token(stale_task.token)
        is TokenStatus.STALE_SESSION
    )

    release.set()
    terminal = _await_completion(window, receipt)

    assert terminal.state is BackgroundTaskState.DISCARDED
    assert terminal.apply_status is TaskApplyStatus.STALE
    assert receipt.completion is not None
    assert receipt.completion.outcome is None
    assert calls == [(lazy.field_key,)]
    assert window.document.session_revision == closed_revision
    assert window.session.current_result() is None
    assert window.result_provider is None
    assert window.result_selection is None
    assert window.viewport._result_render_payload is None


def test_typed_render_uses_the_selected_display_deformation_scale(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    selection = window.result_selection
    assert type(selection) is ScalarFieldSelection

    window.set_shape_mode("undeformed")
    undeformed = _result_payload(window)
    assert undeformed.topology.selection == selection
    assert undeformed.topology.deformation_scale == 0.0

    window.set_shape_mode("deformed")
    _select_scale_mode(window, "real")
    real = _result_payload(window)
    assert real.topology.selection == selection
    assert real.topology.deformation_scale == 1.0

    _select_scale_mode(window, "custom")
    window.result_scale_value.setValue(2.75)
    custom = _result_payload(window)
    assert custom.topology.selection == selection
    assert custom.topology.deformation_scale == 2.75

    _select_scale_mode(window, "auto")
    automatic = _result_payload(window)
    assert automatic.topology.selection == selection
    assert math.isfinite(automatic.topology.deformation_scale)
    assert automatic.topology.deformation_scale > 0.0


def test_export_actions_follow_exact_selected_field_readiness(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    provider = window.result_provider
    ready = window.result_selection
    assert type(provider) is ResultProvider
    assert type(ready) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    foreign = ScalarFieldSelection(
        FieldMaterializationKey(
            ready.field_key.request,
            ready.field_key.recovery_contract + 10_000,
        ),
        ready.component,
    )

    window.result_selection = ready
    window._update_action_states()
    assert window.actions["export_csv"].isEnabled()
    assert window.actions["export_vtk"].isEnabled()

    for unavailable_selection in (None, lazy, foreign):
        window.result_selection = unavailable_selection
        window._update_action_states()
        assert not window.actions["export_csv"].isEnabled()
        assert not window.actions["export_vtk"].isEnabled()

    window.result_selection = ready
    window._update_action_states()
    assert window.actions["export_csv"].isEnabled()
    assert window.actions["export_vtk"].isEnabled()
