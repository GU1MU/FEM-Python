from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.results import (
    ResultProbeKind,
    ResultProbeRequest,
    ResultProbeTarget,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
    probe_result_from_query_result,
    result_query_for_probe,
)
from fem_gui.postprocessing_dialogs import (
    ResultProbeDialog,
    ResultQueryProbeDialog,
    TypedResultQueryDialog,
)
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="probe-dialog",
        session_id="probe-dialog-session",
        artifact_id="probe-dialog-artifact",
        model_revision=1,
        step_name="Step-1",
        run_id="probe-dialog-run",
    )


def test_probe_dialog_emits_exact_node_request_and_displays_record() -> None:
    _application()
    provider = build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )
    dialog = ResultProbeDialog(provider)
    emitted: list[object] = []
    dialog.probeRequested.connect(emitted.append)

    dialog.target_id_spin.setValue(1)
    dialog.request_probe()

    assert len(emitted) == 1
    request = emitted[0]
    assert isinstance(request, ResultProbeRequest)
    assert request.target == ResultProbeTarget(
        ResultProbeKind.NODE,
        node_id=1,
    )
    assert request.selection.field_key.request.field_id.variable is ResultVariable.U

    query_result = provider.query(result_query_for_probe(request))
    dialog.set_probe_result(
        probe_result_from_query_result(query_result, request)
    )

    assert dialog.table.rowCount() == 1
    assert dialog.record_at(0).location.node_id == 1
    assert "节点 1" in dialog.result_summary.text()
    dialog.close()


def test_probe_dialog_switches_to_integration_point_identity_controls() -> None:
    _application()
    provider = build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )
    dialog = ResultProbeDialog(provider)

    dialog.target_combo.setCurrentIndex(
        dialog.target_combo.findData(ResultProbeKind.INTEGRATION_POINT)
    )

    assert dialog.target_id_label.text() == "单元编号"
    assert not dialog.integration_point_label.isHidden()
    assert not dialog.integration_point_spin.isHidden()
    assert dialog.local_node_label.isHidden()
    assert dialog.local_node_spin.isHidden()
    dialog.close()


def test_query_probe_dialog_uses_one_workflow_with_operation_selector() -> None:
    _application()
    provider = build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )
    query = TypedResultQueryDialog(provider)
    probe = ResultProbeDialog(provider)
    dialog = ResultQueryProbeDialog(query, probe, frame_provider=provider)

    assert not hasattr(dialog, "tabs")
    assert dialog.mode_combo.count() == 2
    assert [
        dialog.mode_combo.itemText(index)
        for index in range(dialog.mode_combo.count())
    ] == [
        "按编号查询",
        "定位探针",
    ]
    assert dialog.mode_stack.count() == 2
    assert dialog.frame_summary is not None
    assert dialog.frame_summary.step_label.text() == "Step-1"
    assert dialog.windowTitle() == "结果查询"
    query_events: list[object] = []
    probe_events: list[object] = []
    dialog.queryRequested.connect(query_events.append)
    dialog.probeRequested.connect(probe_events.append)

    query.request_query()
    probe.request_probe()

    assert len(query_events) == 1
    assert len(probe_events) == 1
    dialog.mode_combo.setCurrentIndex(1)
    assert dialog.mode_stack.currentWidget() is probe
    dialog.close()
