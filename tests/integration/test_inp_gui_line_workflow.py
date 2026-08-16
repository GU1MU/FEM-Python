from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fem.io.inp import read
from fem.application import (
    BeamOrientation,
    ModelSession,
    RegionRef,
    RunStatus,
    SessionStateError,
    resolve_effective_beam_frames,
)
from fem.application.preflight import run_static_preflight
from fem.application.results import build_solve_result_bundle
from fem.core.model import LineLoad
from fem.solvers.static_linear import solve
from fem_gui.inspection_service import InspectionService
from fem_gui.widgets.viewport import _effective_line_load_vector


FIXTURES = (
    Path(__file__).parents[1]
    / "helpers" / "fixtures"
    / "inp"
    / "abaqus_standard"
)


def _import_fixture(name: str) -> ModelSession:
    path = FIXTURES / name
    session = ModelSession()
    task = session.prepare_import(path)

    delta = session.accept_imported_model(task.token, read(path))

    assert delta.accepted
    snapshot = session.snapshot()
    assert snapshot.source_kind == "imported"
    assert snapshot.source_path == path
    assert snapshot.artifact is not None
    return session


def _check_submit_and_solve(
    session: ModelSession,
    step_name: str,
    run_name: str,
):
    validation = session.prepare_validation(step_name)
    report = run_static_preflight(
        validation.model,
        validation.step_name,
        token=validation.token,
    )

    assert report.passed, tuple(
        (diagnostic.code, diagnostic.message)
        for diagnostic in report.errors
    )
    assert report.numerical_stability_checked
    assert session.accept_validation(validation.token, report).accepted
    assert session.can_submit(step_name)

    solve_task = session.prepare_solve(step_name, run_name)
    assert session.begin_run(solve_task.token).accepted
    result = solve(solve_task.model, solve_task.step_name)
    assert session.accept_run_succeeded(
        solve_task.token,
        build_solve_result_bundle(solve_task, result),
    ).accepted

    run = session.find_run(solve_task.run_id)
    assert run is not None
    assert run.status is RunStatus.SUCCEEDED
    current_result = session.current_result()
    assert current_result is not None
    assert current_result.provenance.run_id == solve_task.run_id
    return result


def test_imported_truss_definition_edit_check_submit_and_solve() -> None:
    session = _import_fixture("truss2_tension.inp")
    before = session.snapshot()
    section = before.sections[0]
    edited_properties = dict(section.properties)
    edited_properties["area"] = 1.25 * float(edited_properties["area"])

    delta = session.replace_model_definitions(
        before.materials,
        (replace(section, properties=edited_properties),),
        before.assignments,
        before.steps,
    )

    assert delta.accepted
    after = session.snapshot()
    assert after.artifact is not None
    assert before.artifact is not None
    assert after.artifact.artifact_id != before.artifact.artifact_id
    assert after.model.sections[0].properties["area"] == pytest.approx(
        1.25e-4
    )

    result = _check_submit_and_solve(session, "Tension", "Truss-Edited")

    assert np.isfinite(result.U).all()
    assert np.max(np.abs(result.U)) > 0.0


@pytest.mark.parametrize("coordinate_system", ("local", "global"))
def test_imported_beam_definition_and_line_load_edit_check_submit_and_solve(
    coordinate_system: str,
) -> None:
    session = _import_fixture("beam2_rectangle_uniform_load.inp")
    before = session.snapshot()
    section = before.sections[0]
    edited_properties = dict(section.properties)
    edited_properties.update({"height": 0.12, "width": 0.025})
    edited_steps = tuple(
        replace(
            step,
            line_loads=(
                LineLoad(
                    "BEAM",
                    (0.0, -375.0, 0.0),
                    coordinate_system,
                ),
            ),
        )
        if step.name == "UniformLoad"
        else step
        for step in before.steps
    )

    delta = session.replace_model_definitions(
        before.materials,
        (replace(section, properties=edited_properties),),
        (
            replace(
                before.assignments[0],
                beam_orientation=BeamOrientation((0.0, 1.0, 0.0)),
            ),
        ),
        edited_steps,
    )

    assert delta.accepted
    after = session.snapshot()
    assert after.model.sections[0].section_type == "rectangle"
    assert after.model.sections[0].properties["height"] == pytest.approx(0.12)
    uniform_load = next(
        step for step in after.model.steps if step.name == "UniformLoad"
    )
    assert uniform_load.line_loads == (
        LineLoad(
            "BEAM",
            (0.0, -375.0, 0.0),
            coordinate_system,
        ),
    )
    frame_report = resolve_effective_beam_frames(
        after.model,
        RegionRef("element_set", "BEAM"),
    )
    assert frame_report.passed
    frame = frame_report.entries[0].frame
    assert frame.source == "explicit"
    assert frame.orientation == BeamOrientation((0.0, 1.0, 0.0))

    inspection = InspectionService(
        after.model,
        definitions=after,
        effective_frame_query=lambda target: (
            resolve_effective_beam_frames(after.model, target)
        ),
    ).inspect("assignment", 0)
    inspection_fields = dict(inspection.pages[0].fields)
    assert inspection_fields["orientation source"] == "explicit"
    assert inspection_fields["effective frame source"] == "explicit"
    assert inspection_fields["validity"] == "valid"

    line_load = uniform_load.line_loads[0]
    arrow_vector = _effective_line_load_vector(
        line_load.vector,
        line_load.coordinate_system,
        frame,
    )
    expected_arrow = np.asarray(line_load.vector, dtype=float)
    if coordinate_system == "local":
        expected_arrow = frame.rotation.T @ expected_arrow
    assert arrow_vector == pytest.approx(expected_arrow)

    result = _check_submit_and_solve(
        session,
        "UniformLoad",
        f"Beam-{coordinate_system}",
    )

    assert np.isfinite(result.U).all()
    assert np.max(np.abs(result.U)) > 0.0


def test_imported_truss_line_load_has_stable_error_and_cannot_submit() -> None:
    session = _import_fixture("truss2_tension.inp")
    before = session.snapshot()
    edited_steps = tuple(
        replace(
            step,
            line_loads=(
                LineLoad("TRUSS", (0.0, -10.0, 0.0), "global"),
            ),
        )
        if step.name == "Tension"
        else step
        for step in before.steps
    )
    session.replace_model_definitions(
        before.materials,
        before.sections,
        before.assignments,
        edited_steps,
    )

    validation = session.prepare_validation("Tension")
    report = run_static_preflight(
        validation.model,
        validation.step_name,
        token=validation.token,
    )

    assert not report.passed
    assert "step.reference.invalid" in {
        diagnostic.code for diagnostic in report.errors
    }
    assert session.accept_validation(validation.token, report).accepted
    record = session.validation_for("Tension")
    assert record is not None
    assert not record.passed
    assert not session.snapshot().validation_current("Tension")
    assert not session.can_submit("Tension")
    with pytest.raises(SessionStateError, match="passing validation"):
        session.prepare_solve("Tension", "Rejected-Truss")
