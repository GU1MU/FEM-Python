from __future__ import annotations
from fem.application.result_workflow import build_solve_result_bundle

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from fem.application import solve_analysis
from fem.results import (
    FieldPosition,
    ResultFieldId,
    ResultVariable,
    build_result_provider,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry
from fem.model import authoring as steps
from tests.architecture.test_nlf30_gui_workflow import (
    _make_nonlinear_workflow_model,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_gui_linear_elastic_nlgeom_uses_consistent_result_source() -> None:
    _application()
    model = _make_nonlinear_workflow_model(
        yield_stress=None,
        load=1.0e-4,
    )
    steps.output(model.steps[0], "field", "node", ("U", "RF"))
    steps.output(model.steps[0], "field", "element", ("S",))
    window = FEMMainWindow()
    try:
        window._model_loaded(
            Path("linear-elastic-nlgeom.inp"),
            (model, build_model_geometry(model)),
        )
        assert window.check_current_model(show_success=False)
        task = window.session.prepare_solve("nonlinear", "Elastic-NLGEOM")
        if task.delta is not None:
            window._apply_session_delta(task.delta)
        window._apply_session_delta(window.session.begin_run(task.token))
        result = solve_analysis(task.model, "nonlinear")
        window._job_succeeded(
            task.token,
            (build_solve_result_bundle(task, result), {}),
        )

        provider = window._current_result_provider()
        assert provider is not None
        assert provider.frame_indices
        assert window.viewport._result_render_payload is not None

        stress_key = next(
                availability.key
                for availability in provider.catalog().fields
                if availability.descriptor.field_id
                == ResultFieldId(
                    ResultVariable.S,
                    FieldPosition.ELEMENT_NODAL,
                )
        )
        stress_field = provider.field(stress_key)
        assert np.all(np.isfinite(stress_field.values))
    finally:
        window.close()
