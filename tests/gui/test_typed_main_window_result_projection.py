from __future__ import annotations

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
    SolveResultBundle,
    build_result_provider,
    build_solve_result_bundle,
)
from fem.solvers.static_linear import solve
from fem_gui.commands import GuiCommandOutcome, GuiCommandStatus
from fem_gui.main_window import FEMMainWindow
from fem_gui.result_presentation import visible_result_fields
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


def _full_catalog_bundle(task, result) -> SolveResultBundle:
    lightweight = build_solve_result_bundle(task, result)
    provider = build_result_provider(lightweight.source, result)
    return SolveResultBundle._from_provider(
        source=lightweight.source,
        result=result,
        execution_report=lightweight.execution_report,
        provider=provider,
    )


@pytest.fixture
def solved_window() -> FEMMainWindow:
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    imported = window.session.prepare_import(Path("typed-projection.inp"))
    assert window._apply_session_delta(
        window.session.accept_imported_model(imported.token, model),
        model_geometry=build_model_geometry(model),
        source_label="typed-projection.inp",
    )

    validation = window.session.prepare_validation("pull")
    assert window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )
    task = window.session.prepare_solve("pull", "Typed-Projection")
    assert task.delta is not None
    assert window._apply_session_delta(task.delta)
    assert window._apply_session_delta(window.session.begin_run(task.token))
    result = solve(task.model, task.step_name, name="Typed-Projection")
    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            task.token,
            _full_catalog_bundle(task, result),
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
            candidate = ScalarFieldSelection(availability.key, component)
            if candidate != current:
                return candidate
    raise AssertionError("fixture must publish an alternate READY component")


def _lazy_selection(provider: ResultProvider) -> ScalarFieldSelection:
    for availability in provider.catalog().fields:
        if availability.state is FieldState.LAZY:
            return ScalarFieldSelection(
                availability.key,
                availability.descriptor.default_component,
            )
    raise AssertionError("fixture must publish at least one LAZY field")


def _result_payload(window: FEMMainWindow) -> ResultRenderPayload:
    payload = window.viewport._result_render_payload
    assert type(payload) is ResultRenderPayload
    return payload


def test_successful_solve_projects_one_typed_result_spine(
    solved_window: FEMMainWindow,
) -> None:
    window = solved_window
    record = window.session.current_result()
    provider = window.result_provider
    selection = window.result_selection

    assert record is not None
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    assert provider.source == record.materialization.source
    assert provider.snapshot.generation == record.materialization.generation
    assert selection == provider.catalog().default_selection

    root = window.result_tree.topLevelItem(0)
    step = root.child(0)
    assert [
        step.child(index).data(0, ROLE_MATERIALIZATION_KEY)
        for index in range(step.childCount())
    ] == [
        item.key for item in visible_result_fields(provider.catalog().fields)
    ]
    assert window.result_tree.currentItem().data(0, ROLE_SELECTION) == selection
    assert window.inspection_service.result_provider is provider
    payload = _result_payload(window)
    assert payload.topology.source == provider.source
    assert payload.topology.selection == selection


def test_ready_selection_is_synchronous_and_revision_neutral(
    solved_window: FEMMainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window
    provider = window.result_provider
    current = window.result_selection
    assert type(provider) is ResultProvider
    assert type(current) is ScalarFieldSelection
    selection = _alternate_ready_selection(provider, current)

    monkeypatch.setattr(
        window,
        "_start_task",
        lambda *_args, **_kwargs: pytest.fail(
            "READY field selection must not start a worker"
        ),
    )
    revision = window.document.session_revision
    receipt = window.select_result_field(selection)

    assert receipt.status is GuiCommandStatus.ACCEPTED
    assert type(receipt.outcome) is GuiCommandOutcome
    assert receipt.completion is None
    assert window.document.session_revision == revision
    assert window.result_provider is provider
    assert window.result_selection == selection
    assert _result_payload(window).topology.selection == selection
    assert not window.busy


def test_export_actions_follow_catalog_readiness(
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

    for selection, vtk_enabled in (
        (ready, True),
        (None, False),
        (lazy, False),
        (foreign, False),
    ):
        window.result_selection = selection
        window._update_action_states()
        assert window.actions["export_csv"].isEnabled()
        assert window.actions["export_vtk"].isEnabled() is vtk_enabled
