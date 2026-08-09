from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from fem.io import inp
from fem.application import (
    AuthoringStatus,
    PreflightSeverity,
    PreflightStage,
    describe_model_capabilities,
    run_static_preflight,
)
from fem.application.results import project_output_requests
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
            "*Heading",
            "Phase 8 output characterization",
            "*Node",
            "1, 0.0, 0.0, 0.0",
            "2, 1.0, 0.0, 0.0",
            "*Element, type=B31, elset=BEAMS",
            "1, 1, 2",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Beam Section, elset=BEAMS, material=STEEL, section=RECT",
            "0.2, 0.1",
            "0.0, 0.0, 1.0",
            "*Step, name=Output-Step",
            "*Static",
            *output_lines,
            "*End Step",
        ],
    )
    return inp.read_with_report(path)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ((23, "node", ("U",)), "kind"),
        (("field", 17, ("U",)), "target"),
        (("field", "node", ("u", 2, None, "u")), r"variables\[1\]"),
    ),
)
def test_programmatic_output_request_rejects_string_coercion(
    arguments: tuple[object, object, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        OutputRequest(*arguments)


def test_programmatic_variables_preserve_case_order_and_duplicates() -> None:
    request = OutputRequest(
        "FIELD",
        "NODE",
        ("rf", "U", "rf", "CustomVariable"),
    )

    assert request.kind == "field"
    assert request.target == "node"
    assert request.variables == ("rf", "U", "rf", "CustomVariable")


def test_programmatic_metadata_is_deeply_owned_and_immutable() -> None:
    nested = {"thresholds": [0, 75, 100]}
    source = {"averaging": nested}

    request = OutputRequest("field", "element", ("S",), source)
    source["late_outer_value"] = True
    nested["thresholds"][1] = 80

    assert request.metadata == {
        "averaging": {"thresholds": (0, 75, 100)},
    }
    assert request.metadata is not source
    assert request.metadata["averaging"] is not nested

    with pytest.raises(TypeError):
        request.metadata["public_mutation"] = {"kept": True}
    with pytest.raises(TypeError):
        request.metadata["averaging"]["late"] = True
    with pytest.raises(TypeError):
        request.metadata["averaging"]["thresholds"][0] = 1


def test_installed_capability_publishes_intrinsic_create_and_catalog() -> None:
    model = inp.read(_STANDARD_INP_FIXTURES / "truss2_tension.inp")
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
        ("output_request.create", AuthoringStatus.ENABLED),
    )
    assert all(not capability.diagnostics for capability in output_operations)
    assert report.output_request_catalog is not None
    assert report.output_request_candidates


def test_preflight_reports_each_unsupported_output_request() -> None:
    model = inp.read(_STANDARD_INP_FIXTURES / "truss2_tension.inp")
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
    assert tuple(item.code for item in output_diagnostics) == (
        "output.request.kind_unsupported",
        "output.request.target_unsupported",
        "output.request.variable_unsupported",
    )
    assert all(
        item.severity is PreflightSeverity.WARNING
        for item in output_diagnostics
    )
    assert tuple(item.path for item in output_diagnostics) == (
        ("steps", "Tension", "outputs", "1", "kind"),
        ("steps", "Tension", "outputs", "1", "target"),
        ("steps", "Tension", "outputs", "2", "variables", "0"),
    )
    assert tuple(
        dict(item.details)["request_index"]
        for item in output_diagnostics
    ) == (1, 1, 2)


def test_abaqus_parent_output_context_is_inherited_with_source_evidence(
    tmp_path: Path,
) -> None:
    deck = _output_deck(
        tmp_path,
        "*Output, FIELD, FREQUENCY=1, ParentOption=kept-only-in-source, ParentFlag",
        "*Node Output",
        "U",
    )

    request = deck.model.steps[0].outputs[0]
    parent = next(
        occurrence
        for occurrence in deck.source_summary.occurrences
        if occurrence.name == "output"
    )

    assert parent.params == (
        ("frequency", "1"),
        ("parentoption", "kept-only-in-source"),
    )
    assert parent.flags == ("field", "parentflag")
    assert request.kind == "field"
    assert request.target == "node"
    assert request.metadata == {
        "frequency": "1",
        "parentoption": "kept-only-in-source",
    }
    assert request.source_evidence is not None
    assert request.source_evidence.parent_parameters == parent.params
    assert request.source_evidence.parent_flags == parent.flags
    assert request.source_evidence.child_parameters == ()
    assert request.source_evidence.child_flags == ()


def test_executable_authoring_requests_exclude_history_and_unknown_variables(
    tmp_path: Path,
) -> None:
    imported = _output_deck(
        tmp_path,
        "*Output",
        "*Node Output",
        "U, RF",
        "*Element Output",
        "S, MISES",
    )
    model = imported.model
    catalog = describe_model_capabilities(model).output_request_catalog
    assert catalog is not None

    history = project_output_requests(tuple(model.steps[0].outputs), catalog)
    assert tuple(
        projection.executable_authoring_request for projection in history
    ) == (None, None)

    model.steps[0].outputs = (
        OutputRequest("field", "node", ("U", "RF")),
        OutputRequest("field", "element", ("S", "MISES")),
    )
    field = project_output_requests(tuple(model.steps[0].outputs), catalog)
    assert tuple(
        projection.executable_authoring_request for projection in field
    ) == (
        OutputRequest("field", "node", ("U", "RF")),
        OutputRequest("field", "element", ("S",)),
    )


def test_abaqus_child_options_override_parent_and_preserve_variables_and_flags(
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

    request = deck.model.steps[0].outputs[0]
    child = next(
        occurrence
        for occurrence in deck.source_summary.occurrences
        if occurrence.name == "node output"
    )

    assert child.params == (
        ("frequency", "2"),
        ("nset", "Tip"),
        ("futureoption", "ChildValue"),
    )
    assert child.flags == ("childflag",)
    assert request.variables == ("u", "Rf", "u", "customVariable")
    assert request.metadata == {
        "frequency": "2",
        "nset": "Tip",
        "futureoption": "ChildValue",
    }
    assert request.source_evidence is not None
    assert request.source_evidence.parent_parameters == (("frequency", "1"),)
    assert request.source_evidence.parent_flags == ("field", "parentflag")
    assert request.source_evidence.child_parameters == child.params
    assert request.source_evidence.child_flags == ("childflag",)
    assert "parentflag" not in request.metadata
    assert "childflag" not in request.metadata


def test_abaqus_output_parent_context_ends_at_unrelated_keyword(
    tmp_path: Path,
) -> None:
    deck = _output_deck(
        tmp_path,
        "*Output, field, frequency=1",
        "*Node Output",
        "U",
        "*Boundary",
        "1, 1, 1, 0",
        "*Element Output",
        "S",
    )

    inherited, standalone = deck.model.steps[0].outputs

    assert inherited.metadata == {"frequency": "1"}
    assert inherited.source_evidence is not None
    assert inherited.source_evidence.parent_parameters == (("frequency", "1"),)
    assert standalone.metadata == {}
    assert standalone.source_evidence is not None
    assert standalone.source_evidence.parent_parameters == ()
    assert standalone.source_evidence.parent_flags == ()


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

    with pytest.raises(inp.InpParseError) as caught:
        _output_deck(tmp_path, *lines)

    assert caught.value.code == "abaqus.keyword.parameter_duplicate"


def test_existing_output_view_acceptance_preserves_exact_dto_phase8_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Superseded by the Phase 8 typed read-only view contract: accepting a
    # viewer must not rebuild, normalize, or replace the saved request.
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

    assert not changed
    assert after_view is before_view
    assert before_view.variables == ("rf", "U", "U", "custom", "Custom")
    assert after_view == before_view

    manager.close()
    assert application is QApplication.instance()
