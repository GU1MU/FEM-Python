from __future__ import annotations
from fem.application.result_workflow import build_solve_result_bundle

import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import QApplication

from fem.model import authoring as steps
from fem.results import (
    FieldState,
    ResultQuery,
    ResultLegendMode,
    ResultVariable,
    probe_result_from_query_result,
    result_query_for_probe,
)
from fem.application import solve_analysis
from fem_gui.main_window import FEMMainWindow
from fem_gui.xy_data_dialog import XYDataDialog
from fem_gui.visualization.model_adapter import build_model_geometry
from tests.architecture.test_nlf30_gui_workflow import (
    _make_nonlinear_workflow_model,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_nonlinear_quad4_result_frames_drive_display_without_replacing_session_result():
    _application()
    model = _make_nonlinear_workflow_model(load=1.0e-4)
    controls = steps.StaticStepControls(
        initial_increment=0.5,
        maximum_increments=2,
        newton_max_iterations=20,
        residual_tolerance=1.0e-9,
    )
    model.steps[0].metadata.update(controls.to_metadata())
    window = FEMMainWindow()
    try:
        window._model_loaded(
            Path("nonlinear-frame-test.inp"),
            (model, build_model_geometry(model)),
        )
        assert window.check_current_model(show_success=False)
        task = window.session.prepare_solve("nonlinear", "Frame-Job")
        if task.delta is not None:
            assert window._apply_session_delta(task.delta)
        assert window._apply_session_delta(window.session.begin_run(task.token))
        result = solve_analysis(task.model, "nonlinear")
        window._job_succeeded(
            task.token,
            (build_solve_result_bundle(task, result), {}),
        )

        provider = window._current_result_provider()
        assert provider is not None
        assert provider.frame_indices == (1, 2)
        assert {
            availability.descriptor.field_id.variable
            for availability in provider.catalog().fields
        } >= {ResultVariable.E, ResultVariable.PEEQ}
        assert window.result_frame_combo is not None
        assert window.result_frame_combo.count() == 2
        assert window.result_frame_combo.itemData(0) == 1
        assert window.result_frame_combo.itemData(1) == 2
        assert window.result_frame_combo.itemText(1) == "增量 2/2"
        assert all(
            "λ=" not in window.result_frame_combo.itemText(index)
            and "Newton" not in window.result_frame_combo.itemText(index)
            for index in range(window.result_frame_combo.count())
        )
        assert "Newton" in window.result_frame_combo.itemData(
            1,
            Qt.ItemDataRole.ToolTipRole,
        )
        assert "最终结果" in window.result_frame_combo.itemData(
            1,
            Qt.ItemDataRole.ToolTipRole,
        )

        accepted_provider = window.result_provider
        final_displacement = provider.model_result.U.copy()
        final_payload = window.viewport._result_render_payload
        assert final_payload is not None
        default_query = final_payload.topology.display_query
        assert default_query is not None
        assert default_query.legend.mode is ResultLegendMode.GLOBAL_STEP
        default_global_range = window._result_global_legend_range(
            provider,
            window.result_selection,
        )
        assert default_global_range is not None
        assert window.viewport._contour["manual"]
        assert window.viewport._contour["minimum"] == default_global_range[0]
        assert window.viewport._contour["maximum"] == default_global_range[1]
        window._set_result_frame(1)

        assert window._result_frame_index == 1
        assert window.result_provider is accepted_provider
        assert np.array_equal(
            provider.model_result.U,
            final_displacement,
        )
        assert "增量 1" in window.status_panel.result_label.text()
        frame_payload = window.viewport._result_render_payload
        assert frame_payload is not None and final_payload is not None
        assert "增量 1" in window.viewport._contour_bar_args(
            frame_payload
        )["title"]
        assert final_payload.topology.display_query is not None
        assert final_payload.topology.display_query.frame.frame_index == 2
        assert frame_payload.topology.display_query is not None
        assert frame_payload.topology.display_query.frame.frame_index == 1
        assert not np.array_equal(
            frame_payload.topology._points,
            final_payload.topology._points,
        )
        assert (
            frame_payload.topology.deformation_scale
            == final_payload.topology.deformation_scale
        )

        window._set_contour_options({"range_mode": "global_step"})
        global_payload = window.viewport._result_render_payload
        assert global_payload is not None
        global_query = global_payload.topology.display_query
        assert global_query is not None
        assert global_query.legend.mode is ResultLegendMode.GLOBAL_STEP
        global_range = window._result_global_legend_range(
            provider,
            window.result_selection,
        )
        assert global_range is not None
        assert window.viewport._contour["manual"]
        assert window.viewport._contour["minimum"] == global_range[0]
        assert window.viewport._contour["maximum"] == global_range[1]

        frame_query_provider = window._result_frame_provider_for_query(
            provider
        )
        assert frame_query_provider.frame_key is not None
        assert frame_query_provider.frame_key.frame_index == 1
        frame_selection = frame_query_provider.catalog().default_selection
        assert frame_selection is not None
        frame_query = ResultQuery(
            frame_selection.field_key,
            frame_selection.component,
            node_ids=(
                frame_query_provider.snapshot.topology.node_ids[0],
            ),
        )
        expected_frame_query = frame_query_provider.query(frame_query)
        delivered: list[object] = []
        def collect_frame_query(result: object) -> None:
            delivered.append(result)

        window.resultQueryCompleted.connect(collect_frame_query)
        try:
            receipt = window._submit_result_query(
                frame_query,
                provider_override=frame_query_provider,
            )
        finally:
            window.resultQueryCompleted.disconnect(collect_frame_query)
        assert receipt.status.value == "accepted"
        assert delivered == [expected_frame_query]

        lazy_availability = next(
            (
                availability
                for availability in frame_query_provider.catalog().fields
                if availability.state is FieldState.LAZY
            ),
            None,
        )
        assert lazy_availability is not None
        lazy_query = ResultQuery(
            lazy_availability.key,
            lazy_availability.descriptor.default_component,
        )
        lazy_delivered: list[object] = []

        def collect_lazy_query(result: object) -> None:
            lazy_delivered.append(result)

        window.resultQueryCompleted.connect(collect_lazy_query)
        try:
            lazy_receipt = window._submit_result_query(
                lazy_query,
                provider_override=frame_query_provider,
            )
            assert lazy_receipt.completion is not None
            deadline = monotonic() + 1.9
            while (
                (not lazy_receipt.completion.done or window.busy)
                and monotonic() < deadline
            ):
                _application().processEvents()
                QThread.msleep(1)
            _application().processEvents()
            lazy_terminal = lazy_receipt.completion.result(0.0)
        finally:
            window.resultQueryCompleted.disconnect(collect_lazy_query)
        assert lazy_receipt.status.value == "pending"
        assert lazy_terminal.state.value == "succeeded"
        assert lazy_delivered

        window._set_result_frame(2)
        stale_receipt = window._submit_result_query(
            frame_query,
            provider_override=frame_query_provider,
        )
        assert stale_receipt.status.value == "rejected"
        assert stale_receipt.diagnostic is not None
        assert stale_receipt.diagnostic.code == "result.frame.query.stale"

        assert window.result_frame_speed_combo is not None
        window.result_frame_speed_combo.setCurrentIndex(0)
        window._result_frame_speed_changed(0)
        assert window._result_frame_timer.interval() == 500
        window._toggle_result_animation()
        assert window._result_frame_timer.isActive()
        window._toggle_result_animation()
        assert not window._result_frame_timer.isActive()

        window._apply_result_animation_settings(
            {
                "start_frame": 1,
                "end_frame": 2,
                "interval_ms": 400,
                "loop": False,
            }
        )
        assert window._result_animation_start_frame == 1
        assert window._result_animation_end_frame == 2
        assert not window._result_animation_loop
        window._set_result_frame(1)
        window._toggle_result_animation()
        assert window._result_frame_timer.isActive()
        window._advance_result_animation()
        assert window._result_frame_index == 2
        assert window._result_frame_timer.isActive()
        window._advance_result_animation()
        assert not window._result_frame_timer.isActive()

        window._set_result_frame(2)
        assert window._result_frame_index == 2
        assert "最终结果" in window.status_panel.result_label.text()
        assert "增量 2" in window.viewport._contour_bar_args(
            window.viewport._result_render_payload
        )["title"]
        assert window.viewport._contour["minimum"] == global_range[0]
        assert window.viewport._contour["maximum"] == global_range[1]
    finally:
        window.close()


def test_nonlinear_quad4_probe_and_xy_data_share_exact_frame_values():
    _application()
    model = _make_nonlinear_workflow_model(load=1.0e-4)
    controls = steps.StaticStepControls(
        initial_increment=0.5,
        maximum_increments=2,
        newton_max_iterations=20,
        residual_tolerance=1.0e-9,
    )
    model.steps[0].metadata.update(controls.to_metadata())
    window = FEMMainWindow()
    dialog = None
    try:
        window._model_loaded(
            Path("nonlinear-xy-test.inp"),
            (model, build_model_geometry(model)),
        )
        assert window.check_current_model(show_success=False)
        task = window.session.prepare_solve("nonlinear", "XY-Job")
        if task.delta is not None:
            assert window._apply_session_delta(task.delta)
        assert window._apply_session_delta(window.session.begin_run(task.token))
        result = solve_analysis(task.model, "nonlinear")
        window._job_succeeded(
            task.token,
            (build_solve_result_bundle(task, result), {}),
        )

        provider = window._current_result_provider()
        assert provider is not None
        selection = provider.catalog().default_selection
        assert selection is not None
        dialog = XYDataDialog(
            provider.catalog(),
            frame_catalog=provider.frame_catalog,
            current_selection=selection,
            node_ids=provider.snapshot.topology.node_ids,
            element_ids=provider.snapshot.topology.element_ids,
        )
        request = dialog.current_request()
        assert request.selection == selection
        window._start_result_xy_data(
            dialog,
            provider,
            request,
            is_open=lambda: True,
        )

        deadline = monotonic() + 2.0
        while window.busy and monotonic() < deadline:
            _application().processEvents()
            QThread.msleep(1)
        _application().processEvents()

        assert not window.busy
        assert dialog.table.rowCount() == len(provider.frame_indices)
        xy_values = [
            float(dialog.table.item(row, 2).text())
            for row in range(dialog.table.rowCount())
        ]
        assert xy_values[0] != xy_values[-1]

        expected_values = []
        for frame_index in provider.frame_indices:
            frame_provider = provider.frame_provider(frame_index)
            availability = frame_provider.field_status(
                request.selection.field_key
            )
            if availability.state is FieldState.LAZY:
                frame_provider = frame_provider.apply(
                    frame_provider.materialize(
                        (request.selection.field_key,)
                    )
                )
            probe_request = request.probe_request
            query_result = frame_provider.query(
                result_query_for_probe(probe_request)
            )
            probe = probe_result_from_query_result(
                query_result,
                probe_request,
                frame_key=frame_provider.frame_key,
            )
            assert len(probe.records) == 1
            expected_values.append(probe.records[0].value)
        assert xy_values == pytest.approx(expected_values, rel=1.0e-7)
    finally:
        if dialog is not None:
            dialog.close()
        window.close()
