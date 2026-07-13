from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem.solvers.static_linear import solve
from fem_gui.main_window import FEMMainWindow
from fem_gui.analysis_jobs import AnalysisJob, JobStatus
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_actions_follow_document_and_result_context(gui_inp_path):
    _application()
    window = FEMMainWindow()
    assert window.actions["open"].isEnabled()
    for name in ("reload", "close", "submit_job", "fit", "select_node", "model_info", "deformed", "overlay", "query", "export"):
        assert not window.actions[name].isEnabled()
    assert not window.result_variable_combo.isEnabled()

    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    window._model_loaded(gui_inp_path, (model, geometry))
    window._update_action_states()
    for name in ("reload", "close", "submit_job", "fit", "select_node", "model_info", "symbols", "step_info", "check_model", "job_manager"):
        assert window.actions[name].isEnabled()
    for name in ("deformed", "contour", "query", "export"):
        assert not window.actions[name].isEnabled()

    result = solve(model)
    job = AnalysisJob("Job-1", "Static-1", JobStatus.RUNNING)
    window.document.add_job(job)
    window._job_succeeded(job, (result, build_result_data(result, geometry)))
    window._update_action_states()
    for name in ("undeformed", "deformed", "contour", "field", "query", "export"):
        assert window.actions[name].isEnabled()
    assert window.result_variable_combo.isEnabled()
    assert window.result_component_combo.isEnabled()

    window.close_model()
    assert "尚无分析结果" == window.result_tree.topLevelItem(0).text(0)
    assert window.status_panel.object_label.text() == "对象：—"
    assert window.status_panel.step_label.text() == "Step：—"
    assert window.status_panel.result_label.text() == "结果：—"
    assert not window.actions["submit_job"].isEnabled()
    window.close()
