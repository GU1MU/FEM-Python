from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from fem.abaqus import AbaqusParseError, parse_file, read
from fem.application import (
    AuthoringStatus,
    PreflightSeverity,
    PreflightStage,
    describe_model_capabilities,
    run_static_preflight,
)
from fem.core.model import AnalysisStep, OutputRequest
from fem_gui.analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    OutputRequestDialog,
)
from tests.helpers.file_builders import write_inp


_STANDARD_INP_FIXTURES = (
    Path(__file__).parent / "fixtures" / "inp" / "abaqus_standard"
)


def _output_deck(tmp_path: Path, *output_lines: str):
    path = write_inp(
        tmp_path,
        "phase8_output_characterization.inp",
        [
            "*Step, name=Output-Step",
            "*Static",
            *output_lines,
            "*End Step",
        ],
    )
    return parse_file(path)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_programmatic_output_request_string_coercion_is_current_oracle() -> None:
    request = OutputRequest(
        23,
        17,
        ("u", 2, None, "u"),
    )

    assert request.kind == "23"
    assert request.target == "17"
    assert request.variables == ("u", "2", "None", "u")


def test_programmatic_variables_preserve_case_order_and_duplicates() -> None:
    request = OutputRequest(
        "FIELD",
        "NODE",
        ("rf", "U", "rf", "CustomVariable"),
    )

    assert request.kind == "field"
    assert request.target == "node"
    assert request.variables == ("rf", "U", "rf", "CustomVariable")


def test_programmatic_metadata_is_only_shallowly_detached() -> None:
    nested = {"thresholds": [0, 75, 100]}
    source = {"averaging": nested}

    request = OutputRequest("field", "element", ("S",), source)
    source["late_outer_value"] = True
    nested["thresholds"][1] = 80

    assert request.metadata == {
        "averaging": {"thresholds": [0, 80, 100]},
    }
    assert request.metadata is not source
    assert request.metadata["averaging"] is nested

    # The frozen dataclass currently exposes a mutable metadata dict.
    request.metadata["public_mutation"] = {"kept": True}
    assert request.metadata["public_mutation"] == {"kept": True}


def test_current_capability_publishes_create_and_existing_operations() -> None:
    model = read(_STANDARD_INP_FIXTURES / "truss2_tension.inp")
    report = describe_model_capabilities(model)
    output_operations = tuple(
        capability
        for capability in report.authoring
        if capability.operation.startswith("output_request.")
    )

    assert tuple(
        (capability.operation, capability.status)
        for capability in output_operations
    ) == (
        ("output_request.create", AuthoringStatus.UNAVAILABLE),
        ("output_request.existing", AuthoringStatus.READ_ONLY),
    )
    assert {
        diagnostic.code
        for capability in output_operations
        for diagnostic in capability.diagnostics
    } == {"output.request.not_executed"}


def test_preflight_uses_one_blanket_warning_for_all_output_requests() -> None:
    model = read(_STANDARD_INP_FIXTURES / "truss2_tension.inp")
    step = next(step for step in model.steps if step.name == "Tension")
    step.outputs = (
        OutputRequest("field", "node", ("U",)),
        OutputRequest(
            "history",
            "preselect",
            ("PRESELECT",),
            {"variable": "PRESELECT"},
        ),
        OutputRequest("field", "element", ("UNKNOWN",)),
    )

    report = run_static_preflight(model, "Tension")
    output_diagnostics = tuple(
        diagnostic
        for diagnostic in report.diagnostics
        if diagnostic.stage is PreflightStage.OUTPUT
    )

    assert report.passed
    assert len(output_diagnostics) == 1
    warning = output_diagnostics[0]
    assert warning.code == "output.request.not_executed"
    assert warning.severity is PreflightSeverity.WARNING
    assert warning.path == ("steps", "Tension", "outputs")
    assert warning.details == (("count", 3),)


def test_abaqus_parent_output_context_is_not_inherited_current_oracle(
    tmp_path: Path,
) -> None:
    deck = _output_deck(
        tmp_path,
        "*Output, FIELD, FREQUENCY=1, ParentOption=kept-only-in-source, ParentFlag",
        "*Node Output",
        "U",
    )

    request = deck.steps[0].output_requests[0]
    parent = next(
        occurrence
        for occurrence in deck.keyword_occurrences
        if occurrence.name == "output"
    )

    assert parent.params == (
        ("frequency", "1"),
        ("parentoption", "kept-only-in-source"),
    )
    assert parent.flags == ("field", "parentflag")
    assert request.kind == "field"
    assert request.target == "node"
    assert request.metadata == {}


def test_abaqus_child_options_and_variables_define_current_request_oracle(
    tmp_path: Path,
) -> None:
    deck = _output_deck(
        tmp_path,
        "*Output, FIELD, frequency=1, ParentFlag",
        (
            "*Node Output, FREQUENCY=2, NSET=Tip, "
            "FutureOption=ChildValue, ChildFlag, CHILDFLAG"
        ),
        "u, Rf, u, customVariable",
    )

    request = deck.steps[0].output_requests[0]
    child = next(
        occurrence
        for occurrence in deck.keyword_occurrences
        if occurrence.name == "node output"
    )

    assert child.params == (
        ("frequency", "2"),
        ("nset", "Tip"),
        ("futureoption", "ChildValue"),
    )
    assert child.flags == ("childflag",)
    assert request.variables == ("U", "RF", "U", "CUSTOMVARIABLE")
    assert request.metadata == {
        "frequency": "2",
        "nset": "Tip",
        "futureoption": "ChildValue",
    }
    assert "parentflag" not in request.metadata
    assert "childflag" not in request.metadata


@pytest.mark.parametrize(
    "keyword",
    (
        "*Output, FIELD, frequency=1, FREQUENCY=2",
        "*Node Output, nset=First, NSET=Second",
    ),
)
def test_abaqus_same_layer_parameter_collision_fails_closed_case_insensitively(
    tmp_path: Path,
    keyword: str,
) -> None:
    lines = (
        (keyword, "*Node Output", "U")
        if keyword.lower().startswith("*output,")
        else ("*Output, FIELD", keyword, "U")
    )

    with pytest.raises(AbaqusParseError) as caught:
        _output_deck(tmp_path, *lines)

    assert caught.value.code == "abaqus.keyword.parameter_duplicate"


def test_existing_output_view_acceptance_rebuilds_and_normalizes_dto_current_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _application()
    original = OutputRequest(
        "field",
        "node",
        ("rf", "U", "U", "custom", "Custom"),
        {"nested": {"value": 1}},
    )
    manager = AnalysisDefinitionManagerDialog(
        [AnalysisStep("Step-A", outputs=(original,))],
        [],
        [],
        [],
        3,
    )
    before_view = manager.steps[0].outputs[0]
    monkeypatch.setattr(
        OutputRequestDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )

    changed = manager.edit_definition(("output", 0, 0))
    after_view = manager.steps[0].outputs[0]

    assert changed
    assert after_view is not before_view
    assert before_view.variables == ("rf", "U", "U", "custom", "Custom")
    assert after_view.variables == ("U", "RF", "custom", "Custom")
    assert after_view.metadata == before_view.metadata

    manager.close()
    assert application is QApplication.instance()
