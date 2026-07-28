from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.application import (
    BeamOrientation,
    ModelSession,
    NativePart,
    RegionRef,
    resolve_effective_beam_frames,
)
from fem.application.results import ResultVariable
from fem.core.model import (
    ElementSet,
    LineLoad,
    MaterialDefinition,
    SectionAssignment,
)
from fem.geometry.recipes import RectangleGeometry
from fem.io.project import LoadedProject
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.widgets.model_tree import ROLE_KIND
from fem_gui.widgets.viewport import _effective_line_load_vector
from tests.helpers.model_builders import make_static_pull_truss_model


STANDARD_FIXTURES = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


@dataclass(frozen=True, slots=True)
class _ImportNotice:
    code: str
    message: str
    locations: tuple[object, ...] = ()


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow, timeout: float = 20.0) -> None:
    deadline = monotonic() + timeout
    application = _application()
    while window.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not window.busy


def _defined_truss_model(*, load: float = 100.0):
    model = make_static_pull_truss_model(load=load)
    model.element_sets["BAR"] = ElementSet("BAR", (1,))
    model.materials = {
        "STEEL": MaterialDefinition("STEEL", {"E": 100.0})
    }
    model.sections = [
        SectionAssignment(
            "BAR",
            "STEEL",
            "truss",
            {"area": 2.0},
        )
    ]
    return model


def _install_import(
    window: FEMMainWindow,
    path: Path,
    *,
    notices: tuple[object, ...],
    load: float = 100.0,
) -> None:
    model = _defined_truss_model(load=load)
    window._model_loaded(
        path,
        (
            model,
            build_model_geometry(model),
            {},
            notices,
        ),
    )


def _close_window(window: FEMMainWindow) -> None:
    if window.document.is_open:
        window.close_model(confirm=False)
    window.close()


def _tree_items(window: FEMMainWindow) -> tuple[object, ...]:
    pending = [
        window.model_tree.topLevelItem(index)
        for index in range(window.model_tree.topLevelItemCount())
    ]
    items: list[object] = []
    while pending:
        item = pending.pop()
        items.append(item)
        pending.extend(
            item.child(index)
            for index in range(item.childCount())
        )
    return tuple(items)


def _open_check_solve(
    window: FEMMainWindow,
    path: Path,
    step_name: str,
    run_name: str,
    monkeypatch,
) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(path), ""),
    )
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    window.open_inp()
    _wait_for_task(window)

    assert errors == []
    assert window.document.source_kind == "imported"
    assert window.document.source_path == path
    assert window.document.artifact is not None
    assert (
        window.geometry.artifact_id
        == window.document.artifact.artifact_id
    )
    assert (
        window.viewport.artifact_id
        == window.document.artifact.artifact_id
    )
    assert not window.document.can_save
    assert not window.actions["save_project"].isEnabled()
    assert window.check_current_model(show_success=False), errors
    assert window.session.can_submit(step_name)
    assert window.actions["submit_job"].isEnabled()

    run = window._submit_job(run_name, step_name)
    assert run is not None
    _wait_for_task(window)

    assert errors == []
    current = window.session.current_result()
    assert current is not None
    assert current.provenance.run_id == run.run_id
    assert (
        current.provenance.artifact_id
        == window.document.artifact.artifact_id
    )
    provider = window.result_provider
    selection = window.result_selection
    payload = window.viewport._result_render_payload
    assert provider is not None
    assert selection is not None
    assert payload is not None
    assert provider.source.run_id == run.run_id
    assert provider.source.artifact_id == current.provenance.artifact_id
    assert selection.field_key.request.field_id.variable is ResultVariable.U
    displacement = provider.field(selection.field_key)
    assert np.isfinite(displacement.values).all()
    assert np.max(np.abs(displacement.values)) > 0.0
    assert payload.topology.source == provider.source
    assert window.actions["deformed"].isEnabled()
    assert window.actions["query"].isEnabled()
    assert window.result_tree.topLevelItem(0).text(0) != "尚无分析结果"
    return errors


def test_literal_standard_t3d2_main_window_open_check_solve_result(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    path = STANDARD_FIXTURES / "t3d2_tension.inp"

    try:
        _open_check_solve(
            window,
            path,
            "TENSION",
            "T3D2-Job",
            monkeypatch,
        )

        assert {
            str(element.type)
            for element in window.document.model.mesh.elements
        } == {"Truss2"}
        section = window.document.sections[0]
        assert section.section_type == "truss"
        assert section.properties["area"] == pytest.approx(1.0e-4)
        assignment = window.document.assignments[0]
        assert assignment.region_name == "TRUSS"
        assert assignment.beam_orientation is None
        assert window.import_notices == ()
        kinds = {
            item.data(0, ROLE_KIND)
            for item in _tree_items(window)
        }
        assert {"section", "assignment"}.issubset(kinds)
    finally:
        _close_window(window)


def test_literal_standard_b31_main_window_open_check_solve_result(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    path = STANDARD_FIXTURES / "b31_rect_explicit_n1_loads.inp"

    try:
        _open_check_solve(
            window,
            path,
            "STANDARD_LINE_LOADS",
            "B31-Job",
            monkeypatch,
        )

        assert {
            str(element.type)
            for element in window.document.model.mesh.elements
        } == {"Beam2"}
        section = window.document.sections[0]
        assert section.section_type == "rectangle"
        assert section.properties == {
            "height": pytest.approx(0.20),
            "width": pytest.approx(0.10),
        }
        assignment = window.document.assignments[0]
        assert assignment.region_name == "BEAM"
        assert assignment.beam_orientation == BeamOrientation(
            (0.0, 1.0, 0.0)
        )
        assert len(window.import_notices) == 1
        assert window.import_notices[0].code == (
            "abaqus.b31.euler_bernoulli_approximation"
        )

        step = next(
            item
            for item in window.document.steps
            if item.name == "STANDARD_LINE_LOADS"
        )
        assert step.line_loads == (
            LineLoad("BEAM", (11.0, 0.0, 0.0), "global"),
            LineLoad("BEAM", (0.0, -12.0, 0.0), "global"),
            LineLoad("BEAM", (0.0, 0.0, 13.0), "global"),
            LineLoad("BEAM", (0.0, -14.0, 0.0), "local"),
            LineLoad("BEAM", (0.0, 0.0, 15.0), "local"),
            LineLoad("BEAM", (0.0, 1.5, 0.0), "local"),
        )
        target = RegionRef("element_set", "BEAM")
        frame_report = resolve_effective_beam_frames(
            window.document.model,
            target,
        )
        assert frame_report.passed
        assert len(frame_report.entries) == 2
        assert all(
            entry.frame.source == "explicit"
            for entry in frame_report.entries
        )
        viewport_report = window.viewport._effective_beam_frame_report(
            target
        )
        assert viewport_report is not None
        assert viewport_report.passed
        frame = frame_report.entries[0].frame
        for load in step.line_loads:
            arrow = _effective_line_load_vector(
                load.vector,
                load.coordinate_system,
                frame,
            )
            expected = np.asarray(load.vector, dtype=float)
            if load.coordinate_system == "local":
                expected = frame.rotation.T @ expected
            assert arrow == pytest.approx(expected)

        inspection = window.inspection_service.inspect("assignment", 0)
        fields = dict(inspection.pages[0].fields)
        assert fields["orientation source"] == "explicit"
        assert fields["effective frame source"] == "explicit"
        assert fields["validity"] == "valid"
        kinds = [
            item.data(0, ROLE_KIND)
            for item in _tree_items(window)
        ]
        assert "section" in kinds
        assert "assignment" in kinds
        assert kinds.count("line_load") == 6
    finally:
        _close_window(window)


def test_unpack_model_load_keeps_legacy_shapes_and_adds_notices() -> None:
    model = object()
    geometry = object()
    notice = _ImportNotice("source.notice", "source limitation")

    assert FEMMainWindow._unpack_model_load((model, geometry)) == (
        model,
        geometry,
        {},
        (),
    )
    assert FEMMainWindow._unpack_model_load(
        (model, geometry, {"parse": 1.0})
    ) == (
        model,
        geometry,
        {"parse": 1.0},
        (),
    )
    assert FEMMainWindow._unpack_model_load(
        (model, geometry, {"parse": 1.0}, (notice,))
    ) == (
        model,
        geometry,
        {"parse": 1.0},
        (notice,),
    )
    with pytest.raises(ValueError, match="model load result"):
        FEMMainWindow._unpack_model_load((model,))


def test_background_import_uses_report_and_installs_notice_after_accept(
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    path = tmp_path / "reported.inp"
    model = _defined_truss_model()
    notice = _ImportNotice(
        "source.formulation.approximation",
        "Current solver uses an approximation.",
    )
    parsed_deck = object()
    calls: list[tuple[str, object]] = []
    errors: list[tuple[str, str]] = []

    def parse(candidate):
        calls.append(("parse", candidate))
        return parsed_deck

    def build(deck):
        calls.append(("build", deck))
        return SimpleNamespace(model=model, notices=(notice,))

    monkeypatch.setattr(main_window_module, "parse_file", parse)
    monkeypatch.setattr(
        main_window_module,
        "build_abaqus_model_with_report",
        build,
    )
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    try:
        window._load_path(path)
        _wait_for_task(window)

        assert errors == []
        assert calls == [("parse", path), ("build", parsed_deck)]
        assert window.document.source_kind == "imported"
        assert window.document.source_path == path
        assert window.import_notices == (notice,)
        assert notice.message in window.status_panel.state_label.text()
        assert not window.document.can_save
        with pytest.raises(AttributeError):
            window.import_notices = ()
    finally:
        _close_window(window)


def test_notices_survive_edit_check_solve_stale_and_failed_import(
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    current_notice = _ImportNotice(
        "source.formulation.approximation",
        "Current document limitation.",
    )
    stale_notice = _ImportNotice(
        "source.stale",
        "This stale notice must not be installed.",
    )
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    try:
        _install_import(
            window,
            tmp_path / "current.inp",
            notices=(current_notice,),
        )
        assert window.check_current_model(show_success=False), errors
        assert window.import_notices == (current_notice,)
        first_run = window._submit_job("Job-1", "pull")
        assert first_run is not None
        _wait_for_task(window)
        assert window.session.current_result() is not None
        assert window.import_notices == (current_notice,)

        section = window.document.sections[0]
        properties = dict(section.properties)
        properties["area"] = 2.5
        assert window._apply_model_definition_changes(
            "截面已修改",
            sections=(replace(section, properties=properties),),
        )
        assert window.session.current_result() is None
        assert window.import_notices == (current_notice,)

        assert window.check_current_model(show_success=False), errors
        second_run = window._submit_job("Job-2", "pull")
        assert second_run is not None
        _wait_for_task(window)
        current_result = window.session.current_result()
        assert current_result is not None
        assert window.import_notices == (current_notice,)

        stale = window.session.prepare_import(tmp_path / "stale.inp")
        assert window._apply_session_delta(
            window.session.select_result(second_run.run_id)
        )
        artifact_id = window.document.artifact.artifact_id
        result_run_id = current_result.provenance.run_id
        viewport_artifact_id = window.viewport.artifact_id
        viewport_run_id = window.viewport.run_id
        model_tree_label = window.model_tree.topLevelItem(0).text(0)
        result_tree_label = window.result_tree.topLevelItem(0).text(0)

        stale_model = _defined_truss_model(load=999.0)
        window._model_loaded(
            tmp_path / "stale.inp",
            (
                stale_model,
                build_model_geometry(stale_model),
                {},
                (stale_notice,),
            ),
            token=stale.token,
        )

        assert window.document.artifact.artifact_id == artifact_id
        assert (
            window.session.current_result().provenance.run_id
            == result_run_id
        )
        assert window.import_notices == (current_notice,)

        def fail_parse(_path):
            raise ValueError("invalid imported input")

        monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
        monkeypatch.setattr(main_window_module, "parse_file", fail_parse)
        window._load_path(tmp_path / "invalid.inp")
        _wait_for_task(window)

        assert errors[-1] == (
            "模型加载失败",
            "invalid imported input",
        )
        assert window.document.artifact.artifact_id == artifact_id
        assert (
            window.session.current_result().provenance.run_id
            == result_run_id
        )
        assert window.viewport.artifact_id == viewport_artifact_id
        assert window.viewport.run_id == viewport_run_id
        assert window.model_tree.topLevelItem(0).text(0) == model_tree_label
        assert window.result_tree.topLevelItem(0).text(0) == result_tree_label
        assert window.import_notices == (current_notice,)
    finally:
        _close_window(window)


def test_successful_document_replacements_clear_import_notices(
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    notice = _ImportNotice(
        "source.formulation.approximation",
        "Current document limitation.",
    )

    try:
        _install_import(
            window,
            tmp_path / "first.inp",
            notices=(notice,),
        )
        _install_import(
            window,
            tmp_path / "no-notice.inp",
            notices=(),
        )
        assert window.import_notices == ()

        _install_import(
            window,
            tmp_path / "before-project.inp",
            notices=(notice,),
        )
        authoring = ModelSession()
        authoring.new_native_project()
        authoring.replace_geometry(
            (NativePart(),),
            RectangleGeometry("Plate", 2.0, 1.0),
        )
        project_path = tmp_path / "native.femproj"
        loaded = LoadedProject(
            snapshot=replace(
                authoring.prepare_project_save().snapshot,
                source_path=project_path,
            ),
            path=project_path,
            source_schema=5,
            notices=(),
        )
        monkeypatch.setattr(
            main_window_module.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(project_path), ""),
        )
        monkeypatch.setattr(
            main_window_module,
            "load_project",
            lambda _path: loaded,
        )
        window.open_native_project()
        _wait_for_task(window)
        assert window.document.source_kind == "native"
        assert window.import_notices == ()

        window._import_notices = (notice,)
        generated = _defined_truss_model()
        window._generated_model_loaded(
            (
                generated,
                build_model_geometry(generated),
                {},
            )
        )
        assert window.document.source_kind == "native"
        assert window.import_notices == ()

        _install_import(
            window,
            tmp_path / "before-new.inp",
            notices=(notice,),
        )
        window._create_native_model("Model-1")
        assert window.document.source_kind == "native"
        assert window.import_notices == ()

        _install_import(
            window,
            tmp_path / "before-close.inp",
            notices=(notice,),
        )
        assert window.close_model(confirm=False)
        assert window.document.source_kind is None
        assert window.import_notices == ()
    finally:
        _close_window(window)
