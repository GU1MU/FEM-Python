from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem.application.results import (
    FieldMaterializationKey,
    FieldState,
    ResultProvider,
    ScalarFieldSelection,
    build_solve_result_bundle,
)
from fem.solvers.static_linear import solve
from fem_gui.commands import (
    GuiCommandDiagnostic,
    GuiCommandOutcome,
    GuiCommandStatus,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.postprocessing_dialogs import TypedResultDisplaySettings
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


def _lazy_selection(provider: ResultProvider) -> ScalarFieldSelection:
    availability = next(
        item
        for item in provider.catalog().fields
        if item.state is FieldState.LAZY
    )
    return ScalarFieldSelection(
        availability.key,
        availability.descriptor.default_component,
    )


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

    # The canonical consumers must remain functional with the migration-only
    # engineering projection removed from the window.
    window.result_data = None

    def fail_legacy_inspection(*_args, **_kwargs):
        raise AssertionError("typed inspection must not read ResultData")

    monkeypatch.setattr(
        inspection_service,
        "_node_result_fields",
        fail_legacy_inspection,
    )
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
    window.result_data = None
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


def test_invalid_and_lazy_selection_rejections_preserve_projection(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    lazy = _lazy_selection(provider)
    foreign_key = FieldMaterializationKey(
        selection.field_key.request,
        selection.field_key.recovery_contract + 10_000,
    )
    rejected_selections = (
        ScalarFieldSelection(foreign_key, selection.component),
        ScalarFieldSelection(selection.field_key, "NotAComponent"),
        lazy,
    )

    def fail_worker(*_args, **_kwargs):
        raise AssertionError("rejected selection must not start a worker")

    monkeypatch.setattr(window, "_start_task", fail_worker)
    window.result_data = None
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


def test_typed_render_uses_the_selected_display_deformation_scale(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    selection = window.result_selection
    assert type(selection) is ScalarFieldSelection
    window.result_data = None

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
    window.result_data = None

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
