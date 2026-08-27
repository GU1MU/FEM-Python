from __future__ import annotations

import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel

from fem.application import RunStatus
from fem.model import (
    DampingModel,
    DynamicProcedureKind,
    DynamicIntegrationMethod,
    DynamicStepControls,
    Element3D,
    ElementSet,
    FEMModel,
    MassMatrixPolicy,
    Mesh3D,
    NodeSet,
    Node3D,
    StaticStepControls,
    TimeAmplitude,
    transient_dynamic,
)
from fem.model.authoring import static
from fem.model import authoring as step_authoring
from fem.results import (
    ModelResult,
    ResultFrame,
    ResultSourceKey,
    build_result_provider,
)
from fem_gui.analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    StaticStepDialog,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.postprocessing_dialogs import (
    ResultFrameSummaryWidget,
)
from fem_gui.visualization.model_adapter import build_model_geometry
import fem_gui.main_window as main_window_module
from tests.helpers.preflight_builders import passing_preflight_report


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow) -> None:
    controller = window.task_controller
    assert controller.busy
    deadline = monotonic() + 2.0
    application = QApplication.instance()
    while controller.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not controller.busy


def _dynamic_truss_model() -> FEMModel:
    model = FEMModel(
        mesh=Mesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
            ],
            elements=[
                Element3D(
                    1,
                    [1, 2],
                    "Truss2",
                    {
                        "E": 100.0,
                        "area": 1.0,
                        "rho": 1.0,
                        "constitutive_model": "linear_elastic",
                    },
                )
            ],
        ),
        node_sets={
            "fixed": NodeSet("fixed", (1,)),
            "tip": NodeSet("tip", (2,)),
        },
        element_sets={"bar": ElementSet("bar", (1,))},
    )
    step = transient_dynamic(
        "动力学步",
        controls=DynamicStepControls(
            time_period=0.1,
            initial_time_increment=0.1,
            maximum_increments=1,
        ),
    )
    step_authoring.displacement(step, "fixed", components=(1, 2, 3))
    step_authoring.displacement(step, "tip", components=(2, 3))
    step_authoring.nodal_load(step, "tip", component=1, value=1.0)
    step_authoring.add(model, step)
    return model


def test_analysis_step_dialog_creates_typed_dynamic_controls() -> None:
    _application()
    dialog = StaticStepDialog("动力学步")
    dialog.procedure_combo.setCurrentIndex(
        dialog.procedure_combo.findData("dynamic")
    )
    dialog.dynamic_time_period_spin.setValue(2.0)
    dialog.dynamic_initial_increment_spin.setValue(0.2)
    dialog.dynamic_minimum_increment_spin.setValue(0.05)
    dialog.dynamic_maximum_increment_spin.setValue(0.4)
    dialog.dynamic_maximum_increments_spin.setValue(10)
    dialog.dynamic_mass_matrix_combo.setCurrentIndex(
        dialog.dynamic_mass_matrix_combo.findData(MassMatrixPolicy.LUMPED.value)
    )
    dialog.dynamic_damping_combo.setCurrentIndex(
        dialog.dynamic_damping_combo.findData(DampingModel.RAYLEIGH.value)
    )
    dialog.dynamic_rayleigh_mass_spin.setValue(0.01)
    dialog.dynamic_rayleigh_stiffness_spin.setValue(0.02)
    dialog.dynamic_amplitude_start_spin.setValue(0.0)
    dialog.dynamic_amplitude_end_spin.setValue(1.5)

    step = dialog.step()

    assert step.procedure == "dynamic"
    assert isinstance(step.controls, DynamicStepControls)
    assert step.controls.mass_matrix is MassMatrixPolicy.LUMPED
    assert step.controls.damping_model is DampingModel.RAYLEIGH
    assert step.controls.time_period == 2.0
    assert step.controls.amplitude == TimeAmplitude(((0.0, 0.0), (2.0, 1.5)))
    assert "initial_increment" not in step.metadata
    assert step.metadata["mass_matrix"] == "lumped"
    dialog.deleteLater()


def test_analysis_step_dialog_creates_explicit_dynamic_controls() -> None:
    _application()
    dialog = StaticStepDialog("显式动力学步")
    dialog.procedure_combo.setCurrentIndex(
        dialog.procedure_combo.findData("dynamic_explicit")
    )

    step = dialog.step()

    assert step.procedure == "dynamic"
    assert isinstance(step.controls, DynamicStepControls)
    assert step.controls.procedure_kind is DynamicProcedureKind.EXPLICIT
    assert step.controls.mass_matrix is MassMatrixPolicy.LUMPED
    assert step.controls.integration_method.value == "central_difference"
    assert dialog.procedure_combo.currentText() == "动力学-显式"
    dialog.deleteLater()


def test_dynamic_step_edit_preserves_typed_state_and_manager_label() -> None:
    _application()
    original = transient_dynamic(
        "动力学步",
        controls=DynamicStepControls(
            time_period=1.0,
            initial_time_increment=0.1,
            maximum_increments=10,
            amplitude=TimeAmplitude(((0.0, 0.0), (1.0, 2.0))),
        ),
    )
    dialog = StaticStepDialog(original.name, current=original)
    assert dialog.procedure_combo.currentData() == "dynamic"
    assert dialog.dynamic_time_period_spin.value() == 1.0
    dialog.dynamic_time_period_spin.setValue(2.0)
    dialog.dynamic_maximum_increments_spin.setValue(20)
    updated = dialog.step()

    assert updated.procedure == "dynamic"
    assert isinstance(updated.controls, DynamicStepControls)
    assert updated.controls.time_period == 2.0
    assert updated.controls.amplitude.points == ((0.0, 0.0), (2.0, 2.0))

    manager = AnalysisDefinitionManagerDialog([updated], [], [], [], 3)
    assert manager.table.item(0, 3).text() == "动力学-隐式"
    manager.deleteLater()
    dialog.deleteLater()


def test_dynamic_step_switching_explicit_to_implicit_resets_integration_method() -> None:
    _application()
    original = transient_dynamic(
        "显式动力学步",
        controls=DynamicStepControls(
            time_period=1.0,
            initial_time_increment=0.1,
            maximum_increments=10,
            procedure_kind=DynamicProcedureKind.EXPLICIT,
        ),
    )
    dialog = StaticStepDialog(original.name, current=original)
    dialog.procedure_combo.setCurrentIndex(
        dialog.procedure_combo.findData("dynamic")
    )

    updated = dialog.step()

    assert updated.controls is not None
    assert updated.controls.procedure_kind is DynamicProcedureKind.IMPLICIT
    assert updated.controls.integration_method is DynamicIntegrationMethod.NEWMARK
    dialog.deleteLater()


def test_static_step_can_switch_to_dynamic_without_stale_static_controls() -> None:
    _application()
    original = static(
        "待切换",
        controls=StaticStepControls(initial_increment=0.25),
        custom_marker="keep",
    )
    dialog = StaticStepDialog(original.name, current=original)
    dialog.procedure_combo.setCurrentIndex(
        dialog.procedure_combo.findData("dynamic")
    )
    updated = dialog.step()

    assert updated.procedure == "dynamic"
    assert updated.metadata["custom_marker"] == "keep"
    assert "initial_increment" not in updated.metadata
    assert "newton_max_iterations" not in updated.metadata
    reverse_dialog = StaticStepDialog(updated.name, current=updated)
    reverse_dialog.procedure_combo.setCurrentIndex(
        reverse_dialog.procedure_combo.findData("static")
    )
    reversed_step = reverse_dialog.step()
    assert reversed_step.procedure == "static"
    assert isinstance(reversed_step.controls, StaticStepControls)
    assert "time_period" not in reversed_step.metadata
    dialog.deleteLater()
    reverse_dialog.deleteLater()


def test_dynamic_job_does_not_enter_linear_static_preparation(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    model = _dynamic_truss_model()
    window._model_loaded(
        Path("dynamic.inp"),
        (model, build_model_geometry(model)),
    )
    step_name = model.steps[0].name
    validation = window.session.prepare_validation(step_name)
    window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )

    def unexpected_static_preparation(*_args, **_kwargs):
        raise AssertionError(
            "linear-dynamic jobs must not call prepare_linear_analysis"
        )

    monkeypatch.setattr(
        main_window_module,
        "prepare_linear_analysis",
        unexpected_static_preparation,
    )

    started = window._submit_job("动力学作业", step_name)
    assert started is not None
    _wait_for_task(window)

    completed = window.session.find_run(started.run_id)
    assert completed is not None
    assert completed.status is RunStatus.SUCCEEDED
    result = window.session.current_result()
    assert result is not None
    assert len(result.result.frames) == 1
    assert window.result_frame_combo is not None
    assert window.result_frame_combo.isEnabled()
    assert window.result_frame_play_button is not None
    assert window.result_frame_play_button.isEnabled()
    assert window.actions["animation_settings"].isEnabled()
    window.close()


def _dynamic_result_provider(*, as_frame: bool = True):
    mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
        elements=[
            Element3D(
                1,
                [1, 2],
                "Truss2",
                {"E": 100.0, "area": 1.0, "rho": 1.0},
            )
        ],
    )
    step = transient_dynamic(
        "动力学步",
        controls=DynamicStepControls(
            time_period=0.1,
            initial_time_increment=0.1,
            maximum_increments=1,
        ),
    )
    model = FEMModel(mesh=mesh, steps=[step])
    size = mesh.num_dofs
    displacement = np.arange(size, dtype=float) * 0.1
    velocity = np.arange(size, dtype=float) * 0.2
    acceleration = np.arange(size, dtype=float) * 0.3
    frame_record = ResultFrame(
        model=model,
        step=step,
        U=displacement,
        reactions=np.zeros(size),
        frame_index=1,
        load_factor=1.0,
        outputs={
            "time": 0.1,
            "time_increment": 0.1,
            "velocity": velocity,
            "acceleration": acceleration,
            "kinetic_energy": 1.0,
            "strain_energy": 2.0,
            "external_work": 3.0,
            "total_energy": 3.0,
        },
    )
    result = ModelResult(
        model=model,
        step=step,
        U=displacement,
        reactions=np.zeros(size),
        outputs=frame_record.outputs,
        frames=(frame_record,),
    )
    source = ResultSourceKey("result", "session", "artifact", 0, step.name, "run")
    root_provider = build_result_provider(source, result)
    return root_provider.frame_provider(1) if as_frame else root_provider


def test_dynamic_frame_summary_belongs_to_the_unified_result_query() -> None:
    _application()
    root_provider = _dynamic_result_provider(as_frame=False)
    summary = ResultFrameSummaryWidget(root_provider.frame_provider(1))

    assert summary.frame_label.text() == "增量 1"
    assert summary.time_label.text() == "1.000000e-01"
    assert summary.time_increment_label.text() == "1.000000e-01"
    assert summary.findChild(QLabel, "resultFrameSummaryValue") is not None
    assert summary.findChildren(QGroupBox)
    summary.deleteLater()
