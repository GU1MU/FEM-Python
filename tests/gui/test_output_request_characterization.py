from __future__ import annotations
import pytest
from PySide6.QtWidgets import QApplication, QDialog
from fem.core.model import AnalysisStep, OutputRequest
from fem_gui.analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    OutputRequestDialog,
)


def test_output_view_acceptance_preserves_exact_dto(
    monkeypatch: pytest.MonkeyPatch,
    gui_application,
) -> None:
    # The typed read-only view contract: accepting a
    # viewer must not rebuild, normalize, or replace the saved request.
    application = gui_application
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
