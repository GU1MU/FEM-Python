from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import QThread

from fem.io.inp import read
from fem.solvers import static_linear
from fem_gui.main_window import FEMMainWindow
import fem_gui.main_window as main_window_module
from fem_gui.visualization.model_adapter import build_model_geometry
from tests.helpers.gui_analysis import wait_for_analysis_task
from tests.helpers.preflight_builders import passing_preflight_report


def test_model_check_runs_the_shared_numerical_stiffness_preflight(
    monkeypatch,
    gui_inp_path,
):
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    original = static_linear.validate_stiffness
    calls: list[str] = []

    def tracked(model, step):
        calls.append(str(step.name))
        return original(model, step)

    monkeypatch.setattr(
        static_linear,
        "validate_stiffness",
        tracked,
    )

    assert window.check_current_model(show_success=False)
    assert calls == ["Static-1"]
    window.close()


def test_preflight_and_repeated_runs_assemble_one_artifact_once(
    gui_application,
    monkeypatch,
    gui_inp_path,
) -> None:
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    calls: list[tuple[str, bool]] = []
    original_apply = static_linear.materials.apply_sections
    original_assemble = static_linear.assemble_global_stiffness_sparse
    original_factor = static_linear.factorize_spd

    def apply_sections(candidate):
        calls.append(
            (
                "materials",
                QThread.currentThread() is window.thread(),
            )
        )
        return original_apply(candidate)

    def assemble(mesh):
        calls.append(
            (
                "stiffness",
                QThread.currentThread() is window.thread(),
            )
        )
        return original_assemble(mesh)

    def factor(stiffness):
        calls.append(
            (
                "factor",
                QThread.currentThread() is window.thread(),
            )
        )
        return original_factor(stiffness)

    monkeypatch.setattr(
        static_linear.materials,
        "apply_sections",
        apply_sections,
    )
    monkeypatch.setattr(
        static_linear,
        "assemble_global_stiffness_sparse",
        assemble,
    )
    monkeypatch.setattr(static_linear, "factorize_spd", factor)

    assert window.check_current_model(show_success=False)
    first = window._submit_job("Job-1", "Static-1")
    assert first is not None
    wait_for_analysis_task(window, gui_application)
    second = window._submit_job("Job-2", "Static-1")
    assert second is not None
    wait_for_analysis_task(window, gui_application)

    assert calls == [
        ("materials", False),
        ("stiffness", False),
        ("factor", False),
    ]
    previous_artifact = window.document.artifact.artifact_id
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            window.document.materials,
            window.document.sections,
            window.document.assignments,
            window.document.steps,
        )
    )
    assert window.document.artifact.artifact_id != previous_artifact
    assert window.check_current_model(show_success=False)
    assert calls == [
        ("materials", False),
        ("stiffness", False),
        ("factor", False),
        ("materials", False),
        ("stiffness", False),
        ("factor", False),
    ]
    assert errors == []
    window.close()


def test_quick_preflight_defers_prepare_until_first_run_and_then_reuses_it(
    gui_application,
    monkeypatch,
    gui_inp_path,
) -> None:
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    prepare_threads: list[bool] = []
    factor_threads: list[bool] = []
    original_prepare = static_linear.prepare
    original_factor = static_linear.factorize_spd

    def prepare(*args, **kwargs):
        prepare_threads.append(
            QThread.currentThread() is window.thread()
        )
        return original_prepare(*args, **kwargs)

    def factor(stiffness):
        factor_threads.append(
            QThread.currentThread() is window.thread()
        )
        return original_factor(stiffness)

    monkeypatch.setattr(
        main_window_module,
        "should_run_numerical_model_check",
        lambda _model: False,
    )
    monkeypatch.setattr(static_linear, "prepare", prepare)
    monkeypatch.setattr(static_linear, "factorize_spd", factor)

    assert window.check_current_model(show_success=False)
    assert prepare_threads == []
    assert factor_threads == []
    first = window._submit_job("Job-1", "Static-1")
    assert first is not None
    wait_for_analysis_task(window, gui_application)
    second = window._submit_job("Job-2", "Static-1")
    assert second is not None
    wait_for_analysis_task(window, gui_application)

    assert prepare_threads == [False]
    assert factor_threads == [False]
    assert errors == []
    window.close()


def test_large_model_check_policy_avoids_preflight_factorization() -> None:
    within_limit = SimpleNamespace(
        mesh=SimpleNamespace(
            elements=range(100_000),
            num_dofs=50_000,
        )
    )
    too_many_elements = SimpleNamespace(
        mesh=SimpleNamespace(
            elements=range(100_001),
            num_dofs=50_000,
        )
    )
    too_many_dofs = SimpleNamespace(
        mesh=SimpleNamespace(
            elements=range(100_000),
            num_dofs=50_001,
        )
    )

    assert main_window_module.should_run_numerical_model_check(
        within_limit
    )
    assert not main_window_module.should_run_numerical_model_check(
        too_many_elements
    )
    assert not main_window_module.should_run_numerical_model_check(
        too_many_dofs
    )


def test_gui_large_model_check_defers_copy_and_uses_quick_preflight(
    monkeypatch,
    gui_inp_path,
):
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    detach_options = []
    preflight_options = []
    original_prepare = window.session.prepare_validation
    original_preflight = main_window_module.safe_static_preflight

    def tracked_prepare(step_name=None, *, detach_model=True):
        detach_options.append(detach_model)
        return original_prepare(
            step_name,
            detach_model=detach_model,
        )

    def tracked_preflight(*args, **kwargs):
        preflight_options.append(
            (
                kwargs["check_numerical_stability"],
                kwargs["copy_model"],
                kwargs["quick_check"],
            )
        )
        return original_preflight(*args, **kwargs)

    monkeypatch.setattr(
        main_window_module,
        "should_run_numerical_model_check",
        lambda _model: False,
    )
    monkeypatch.setattr(
        window.session,
        "prepare_validation",
        tracked_prepare,
    )
    monkeypatch.setattr(
        main_window_module,
        "safe_static_preflight",
        tracked_preflight,
    )

    assert window.check_current_model(show_success=False)

    validation = window.session.validation_for("Static-1")
    assert detach_options == [False]
    assert preflight_options == [(False, False, True)]
    assert validation is not None and validation.passed
    assert {
        item.code for item in validation.report.warnings
    } == {
        "model.capability.sampled_large_model",
        "static.stiffness.skipped_large_model",
    }
    reported: list[tuple[str, list[tuple[str, object]]]] = []
    monkeypatch.setattr(
        window,
        "_show_information",
        lambda title, rows: reported.append((title, list(rows))),
    )
    window._show_model_check_report(validation.report)
    assert dict(reported[0][1])["数值稳定性"] == "已跳过"
    assert "大模型快速检查" not in str(reported[0][1])
    assert "model.capability.sampled_large_model" not in str(reported[0][1])
    window.close()


def test_validation_only_projection_reuses_detached_model_snapshot(
    monkeypatch,
    gui_inp_path,
):
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    detached_model = window.document.model
    validation = window.session.prepare_validation("Static-1")
    delta = window.session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )

    def unexpected_snapshot():
        raise AssertionError("validation projection must not copy the model")

    monkeypatch.setattr(window.session, "snapshot", unexpected_snapshot)

    assert window._apply_session_delta(delta)
    assert window.document.model is detached_model
    assert window.document.validation_current("Static-1")
    window.close()

