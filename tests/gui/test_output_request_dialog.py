from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from fem.application import AuthoringCapability, AuthoringStatus
from fem.application.results import (
    ElementResultProfile,
    ResultCapabilityCatalog,
    ResultModelFamily,
)
from fem.core.model import OutputRequest, OutputSourceEvidence
from fem.steps.factory import static
from fem_gui.analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    OutputRequestDialog,
)


def _output_candidates(
    family: ResultModelFamily = ResultModelFamily.PLANE_CONTINUUM,
):
    values = {
        ResultModelFamily.PLANE_CONTINUUM: (
            ("Quad4",),
            ("plane_continuum",),
            ("U1", "U2"),
            ("Fx", "Fy"),
        ),
        ResultModelFamily.BEAM: (
            ("Beam2",),
            ("beam",),
            ("U1", "U2", "U3", "UR1", "UR2", "UR3"),
            ("Fx", "Fy", "Fz", "Mx", "My", "Mz"),
        ),
        ResultModelFamily.TRUSS: (
            ("Truss2",),
            ("truss",),
            ("U1", "U2", "U3"),
            ("Fx", "Fy", "Fz"),
        ),
    }
    element_types, element_families, dofs, forces = values[family]
    profile = ElementResultProfile(
        family=family,
        canonical_element_types=element_types,
        element_families=element_families,
        dofs_per_node=len(dofs),
        dof_labels=dofs,
        force_labels=forces,
        primary_compatible=True,
        stress_compatible=True,
    )
    return ResultCapabilityCatalog.from_profile(profile).candidates


def _output_capability(
    operation: str,
    status: AuthoringStatus,
) -> AuthoringCapability:
    return AuthoringCapability(operation, status)


@pytest.mark.parametrize(
    "family",
    (
        ResultModelFamily.PLANE_CONTINUUM,
        ResultModelFamily.TRUSS,
        ResultModelFamily.BEAM,
    ),
)
def test_output_request_uses_only_published_candidate_order_and_dto(
    gui_application,
    family,
):
    candidates = _output_candidates(family)
    dialog = OutputRequestDialog(
        ["Load"],
        candidates=candidates,
    )

    expected = tuple(
        next(
            candidate
            for candidate in candidates
            if candidate.authoring_request.variables == (variable,)
        )
        for variable in ("U", "RF", "S")
    )
    assert dialog.candidate_list.count() == len(expected)
    assert tuple(
        dialog.candidate_list.item(index).text()
        for index in range(dialog.candidate_list.count())
    ) == ("U", "RF", "S")
    assert all(
        "position" not in dialog.candidate_list.item(index).text()
        for index in range(dialog.candidate_list.count())
    )
    displacement_item = dialog.candidate_list.item(0)
    assert displacement_item.checkState() == Qt.CheckState.Unchecked
    displacement_item.setCheckState(Qt.CheckState.Unchecked)
    assert displacement_item.checkState() == Qt.CheckState.Unchecked
    for index in range(dialog.candidate_list.count()):
        dialog.candidate_list.item(index).setCheckState(
            Qt.CheckState.Checked
        )
    step_name, outputs = dialog.definitions()
    assert step_name == "Load"
    assert outputs == tuple(
        candidate.authoring_request
        for candidate in expected
    )
    assert all(
        output is not candidate.authoring_request
        for output, candidate in zip(outputs, expected, strict=True)
    )


def test_output_request_discards_parsed_inp_details(gui_application):
    current = OutputRequest(
        "history",
        "preselect",
        ("PRESELECT", "PRESELECT", "Future"),
        {"variable": "PRESELECT"},
        OutputSourceEvidence(
            "abaqus",
            parent_parameters=(("frequency", "2"),),
            child_flags=("preselect",),
        ),
    )
    dialog = OutputRequestDialog(["Load"], current=current)

    step_name, output = dialog.definition()

    assert step_name == "Load"
    assert output != current
    assert output is not current
    assert output == OutputRequest(
        "history",
        "preselect",
        ("PRESELECT", "PRESELECT", "Future"),
    )
    assert not output.metadata
    assert output.source_evidence is None
    assert not dialog.step_combo.isEnabled()
    assert tuple(
        dialog.candidate_list.item(index).text()
        for index in range(dialog.candidate_list.count())
    ) == (
        "PRESELECT",
        "PRESELECT",
        "Future",
    )
    assert all(
        dialog.candidate_list.item(index).checkState()
        == Qt.CheckState.Checked
        for index in range(dialog.candidate_list.count())
    )


def test_output_request_dialog_shows_existing_imported_requests_by_step(gui_application):
    candidates = _output_candidates(ResultModelFamily.PLANE_CONTINUUM)
    dialog = OutputRequestDialog(
        ["Load", "Empty"],
        candidates=candidates,
        existing_requests_by_step={
            "Load": (
                OutputRequest("field", "node", ("RF", "U")),
                OutputRequest("field", "element", ("S",)),
            ),
            "Empty": (),
        },
    )

    assert tuple(
        dialog.candidate_list.item(index).text()
        for index in range(dialog.candidate_list.count())
    ) == ("U", "RF", "S")
    assert all(
        dialog.candidate_list.item(index).checkState()
        == Qt.CheckState.Checked
        for index in range(dialog.candidate_list.count())
    )
    dialog.step_combo.setCurrentText("Empty")
    assert tuple(
        dialog.candidate_list.item(index).checkState()
        for index in range(dialog.candidate_list.count())
    ) == (
        Qt.CheckState.Unchecked,
        Qt.CheckState.Unchecked,
        Qt.CheckState.Unchecked,
    )


def test_output_request_dialog_does_not_select_history_variables_as_fields(gui_application):
    candidates = _output_candidates(ResultModelFamily.PLANE_CONTINUUM)
    dialog = OutputRequestDialog(
        ["Load"],
        candidates=candidates,
        existing_requests_by_step={
            "Load": (
                OutputRequest("history", "node", ("U", "RF")),
                OutputRequest("history", "element", ("S", "MISES")),
            ),
        },
    )

    assert tuple(
        dialog.candidate_list.item(index).checkState()
        for index in range(dialog.candidate_list.count())
    ) == (
        Qt.CheckState.Unchecked,
        Qt.CheckState.Unchecked,
        Qt.CheckState.Unchecked,
    )


def test_output_view_is_read_only_and_preserves_unsupported_request(
    gui_application,
    monkeypatch,
) -> None:
    output = OutputRequest(
        "history",
        "preselect",
        ("Future", "Future", "PRESELECT"),
        {"future": {"mode": "opaque"}},
        OutputSourceEvidence(
            "abaqus",
            parent_flags=("history",),
            child_parameters=(("variable", "PRESELECT"),),
        ),
    )
    step = static("Load")
    step.outputs = (output,)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        [],
        [],
        [],
        2,
        output_view_capability=_output_capability(
            "output_request.view",
            AuthoringStatus.READ_ONLY,
        ),
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.UNAVAILABLE,
        ),
    )
    monkeypatch.setattr(
        OutputRequestDialog,
        "exec",
        lambda _dialog: QDialog.DialogCode.Accepted,
    )
    before = manager.values()

    assert manager.select_definition(("output", 0, 0))
    assert manager.edit_button.text() == "查看"
    assert manager.edit_button.isEnabled()
    assert not manager.delete_button.isEnabled()
    assert not manager.edit_definition(("output", 0, 0))
    assert manager.values() == before
    assert manager.values()[0].outputs[0] == output


def test_output_delete_uses_independent_capability_and_protects_initial(gui_application) -> None:
    output = OutputRequest("history", "preselect", ("Future",))
    load = static("Load")
    load.outputs = (output,)
    denied = AnalysisDefinitionManagerDialog(
        [load],
        [],
        [],
        [],
        2,
        output_view_capability=_output_capability(
            "output_request.view",
            AuthoringStatus.UNAVAILABLE,
        ),
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.UNAVAILABLE,
        ),
    )
    assert denied.select_definition(("output", 0, 0))
    assert not denied.edit_button.isEnabled()
    assert not denied.delete_button.isEnabled()
    denied._delete()
    assert denied.values()[0].outputs == (output,)

    allowed = AnalysisDefinitionManagerDialog(
        [load],
        [],
        [],
        [],
        2,
        output_view_capability=_output_capability(
            "output_request.view",
            AuthoringStatus.UNAVAILABLE,
        ),
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.ENABLED,
        ),
    )
    assert allowed.select_definition(("output", 0, 0))
    assert not allowed.edit_button.isEnabled()
    assert allowed.delete_button.isEnabled()
    allowed._delete()
    assert allowed.values()[0].outputs == ()

    required = OutputRequest("field", "node", ("U",))
    required_step = static("Load")
    required_step.outputs = (required,)
    required_manager = AnalysisDefinitionManagerDialog(
        [required_step],
        [],
        [],
        [],
        2,
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.ENABLED,
        ),
    )
    assert required_manager.select_definition(("output", 0, 0))
    assert not required_manager.delete_button.isEnabled()
    required_manager._delete()
    assert required_manager.values()[0].outputs == (required,)

    initial = static("Initial")
    initial.outputs = (output,)
    protected = AnalysisDefinitionManagerDialog(
        [initial],
        [],
        [],
        [],
        2,
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.ENABLED,
        ),
    )
    assert protected.select_definition(("output", 0, 0))
    assert not protected.delete_button.isEnabled()
    protected._delete()
    assert protected.values()[0].outputs == (output,)
    assert protected.select_definition(("step", 0, None))
    assert not protected.delete_button.isEnabled()
    protected._delete()
    assert len(protected.values()) == 1
    assert protected.values()[0].outputs == (output,)


def test_output_dialog_returns_selected_candidate_definition(gui_application):
    candidates = _output_candidates()
    output_dialog = OutputRequestDialog(
        ["Load"],
        candidates=candidates,
    )
    output_dialog.candidate_list.item(0).setCheckState(
        Qt.CheckState.Checked
    )
    step_name, output = output_dialog.definition()
    assert step_name == "Load"
    assert output == candidates[0].authoring_request
