from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from fem.application.results import (
    FieldPosition,
    OutputExecutionStatus,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
    execute_output_requests,
)
from fem.io import inp
from fem.solvers import static_linear


def _write_b31_output_deck(tmp_path: Path) -> Path:
    path = tmp_path / "phase6-output.inp"
    path.write_text(
        "\n".join(
            (
                "*Heading",
                "Phase 6 output projection",
                "*Node",
                "1, 0.0, 0.0, 0.0",
                "2, 1.0, 0.0, 0.0",
                "*Nset, NSET=ALL",
                "1, 2",
                "*Element, TYPE=B31, ELSET=BEAM",
                "1, 1, 2",
                "*Material, NAME=STEEL",
                "*Elastic",
                "2.10E11, 0.30",
                "*Beam Section, ELSET=BEAM, MATERIAL=STEEL, SECTION=RECT",
                "0.20, 0.10",
                "0.0, 0.0, 1.0",
                "*Step, NAME=Load",
                "*Static",
                "1.0, 1.0",
                "*Boundary",
                "1, 1, 6, 0.0",
                "*Output, FIELD, VARIABLE=PRESELECT",
                "*Node Output, NSET=ALL",
                "U, RF",
                "*Element Output, ELSET=BEAM, DIRECTIONS=YES",
                "S, E",
                "*End Step",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-phase6",
        session_id="session-phase6",
        artifact_id="artifact-phase6",
        model_revision=1,
        step_name="Load",
        run_id="run-phase6",
    )


def test_b31_parent_child_output_projects_without_blocking_physics(
    tmp_path: Path,
) -> None:
    imported = inp.read_with_report(_write_b31_output_deck(tmp_path))
    model = imported.model
    step = next(item for item in model.steps if item.name == "Load")
    requests = tuple(step.outputs)
    preserved_requests = deepcopy(requests)

    assert tuple(request.variables for request in requests) == (
        ("PRESELECT",),
        ("U", "RF"),
        ("S", "E"),
    )
    assert requests[0].source_evidence is not None
    assert requests[0].source_evidence.parent_parameters == (
        ("variable", "PRESELECT"),
    )
    assert requests[0].source_evidence.parent_flags == ("field",)
    assert requests[1].source_evidence is not None
    assert requests[1].source_evidence.parent_flags == ("field",)
    assert requests[1].source_evidence.child_parameters == (
        ("nset", "ALL"),
    )
    assert requests[1].source_evidence.child_flags == ()
    assert requests[2].source_evidence is not None
    assert requests[2].source_evidence.parent_flags == ("field",)
    assert requests[2].source_evidence.child_parameters == (
        ("elset", "BEAM"),
        ("directions", "YES"),
    )
    assert requests[2].source_evidence.child_flags == ()

    result = static_linear.solve(model, step)
    provider = build_result_provider(_source(), result)
    outcome = execute_output_requests(provider, requests)

    assert tuple(
        request.status for request in outcome.report.requests
    ) == (
        OutputExecutionStatus.UNSUPPORTED,
        OutputExecutionStatus.EXECUTED,
        OutputExecutionStatus.UNSUPPORTED,
    )
    assert tuple(
        variable.status
        for variable in outcome.report.requests[1].variables
    ) == (
        OutputExecutionStatus.EXECUTED,
        OutputExecutionStatus.EXECUTED,
    )
    assert tuple(
        (variable.canonical_variable, variable.status)
        for variable in outcome.report.requests[2].variables
    ) == (
        (ResultVariable.S, OutputExecutionStatus.EXECUTED),
        (None, OutputExecutionStatus.UNSUPPORTED),
    )
    assert outcome.report.requests[0].diagnostics[0].code == (
        "output.request.target_unsupported"
    )
    assert outcome.report.requests[2].diagnostics[0].code == (
        "output.request.variable_unsupported"
    )
    assert tuple(
        field.key.request.field_id.variable
        for field in outcome.eager_patch.fields
    ) == (ResultVariable.S,) * 4
    assert tuple(
        field.key.request.field_id.section_point_number
        for field in outcome.eager_patch.fields
        if field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
    ) == (1, 2, 3, 4)
    assert requests == preserved_requests
