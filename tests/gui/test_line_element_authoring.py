from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPushButton,
)

from fem.application import (
    AuthoringCapability,
    AuthoringStatus,
    PreflightDiagnostic,
    PreflightSeverity,
    PreflightStage,
    RegionRef,
)
from fem.core.model import (
    DisplacementConstraint,
    LineLoad,
    NodalLoad,
    OutputRequest,
)
from fem.steps.factory import static
from fem_gui.analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    DisplacementDialog,
    LoadDialog,
    OutputRequestDialog,
)
from fem_gui.widgets.model_tree import ModelTree, ROLE_KIND


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _regions(kind: str, *names: str) -> list[RegionRef]:
    return [RegionRef(kind, name) for name in names]


def _line_candidate(
    status: AuthoringStatus,
    *,
    code: str | None = None,
    message: str = "",
    blocking: bool = False,
) -> AuthoringCapability:
    diagnostics = (
        (
            PreflightDiagnostic(
                code=code,
                severity=(
                    PreflightSeverity.ERROR
                    if blocking
                    else PreflightSeverity.WARNING
                ),
                stage=PreflightStage.CAPABILITY,
                message=message,
            ),
        )
        if code is not None
        else ()
    )
    return AuthoringCapability(
        operation="load.line.local",
        status=status,
        diagnostics=diagnostics,
    )


def _tree_items(tree: ModelTree) -> list[object]:
    root = tree.invisibleRootItem()
    stack = [
        root.child(index)
        for index in range(root.childCount())
    ]
    items = []
    while stack:
        item = stack.pop()
        items.append(item)
        stack.extend(
            item.child(index)
            for index in range(item.childCount())
        )
    return items


def test_line_load_dialog_uses_capability_regions_and_three_components() -> None:
    _application()
    dialog = LoadDialog(
        ["Step-A"],
        [],
        [],
        [],
        6,
        spatial_dimensions=1,
        line_regions=_regions("element_set", "BEAM-SET"),
        preferred_kind="line",
    )

    assert dialog.kind_combo.currentData() == "line"
    assert dialog.region_combo.currentText() == "BEAM-SET"
    assert [
        dialog.coordinate_system_combo.itemData(index)
        for index in range(dialog.coordinate_system_combo.count())
    ] == ["global", "local"]
    assert dialog.coordinate_system_combo.itemText(1) == (
        "局部（Beam 已解析局部坐标）"
    )
    assert dialog.form.isRowVisible(dialog.coordinate_system_combo)
    assert not dialog.form.isRowVisible(dialog.load_type_combo)
    assert not dialog.form.isRowVisible(dialog.component_combo)
    assert not dialog.form.isRowVisible(dialog.value_spin)
    assert dialog.form.isRowVisible(dialog.x_spin)
    assert dialog.form.isRowVisible(dialog.y_spin)
    assert dialog.form.isRowVisible(dialog.z_spin)
    assert not dialog.form.isRowVisible(dialog.local_axis_label)

    dialog.x_spin.setValue(1.25)
    dialog.y_spin.setValue(-2.5)
    dialog.z_spin.setValue(3.75)
    step_name, load = dialog.definition()

    assert step_name == "Step-A"
    assert load == LineLoad(
        "BEAM-SET",
        (1.25, -2.5, 3.75),
        coordinate_system="global",
    )


def test_line_load_dialog_edits_local_load_and_explains_local_axes() -> None:
    _application()
    current = LineLoad("IMPORTED-BEAM", (2.0, 3.0, 4.0), "local")
    dialog = LoadDialog(
        ["Step-A"],
        [],
        [],
        [],
        6,
        current=current,
        candidate_evaluator=lambda _candidate, _step: _line_candidate(
            AuthoringStatus.LIMITED,
            code="beam.orientation.assumed",
            message="旧载荷不可保存",
        ),
    )

    assert dialog.kind_combo.currentData() == "line"
    assert dialog.region_combo.currentText() == "IMPORTED-BEAM"
    assert dialog.coordinate_system_combo.currentData() == "local"
    assert dialog.form.isRowVisible(dialog.local_axis_label)
    assert (
        dialog.local_axis_label.text()
        == "局部（Beam 已解析局部坐标）"
    )
    assert not dialog.buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert "不可保存" in dialog.candidate_diagnostic_label.text()
    assert dialog.definition() == ("Step-A", current)


def test_local_line_load_requires_an_enabled_nonblocking_candidate() -> None:
    _application()
    decisions = []

    def evaluate(candidate, step_name):
        decisions.append((candidate, step_name))
        return _line_candidate(
            AuthoringStatus.LIMITED,
            code="beam.orientation.assumed",
            message="BEAM-SET 缺少显式方向",
        )

    dialog = LoadDialog(
        ["Step-A"],
        [],
        [],
        [],
        6,
        spatial_dimensions=3,
        line_regions=_regions("element_set", "BEAM-SET"),
        preferred_kind="line",
        candidate_evaluator=evaluate,
    )
    dialog.coordinate_system_combo.setCurrentIndex(
        dialog.coordinate_system_combo.findData("local")
    )

    assert not dialog.buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert "beam.orientation.assumed" in (
        dialog.candidate_diagnostic_label.text()
    )
    assert dialog.candidate_decision() is dialog.candidate_decision()
    assert len(decisions) == 1

    dialog.coordinate_system_combo.setCurrentIndex(
        dialog.coordinate_system_combo.findData("global")
    )
    assert dialog.buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()


def test_enabled_candidate_with_blocking_diagnostic_is_not_writable() -> None:
    _application()
    dialog = LoadDialog(
        ["Step-A"],
        [],
        [],
        [],
        6,
        line_regions=_regions("element_set", "BEAM-SET"),
        preferred_kind="line",
        candidate_evaluator=lambda _candidate, _step, **_context: _line_candidate(
            AuthoringStatus.ENABLED,
            code="beam.orientation.parallel",
            message="参考方向与单元平行",
            blocking=True,
        ),
    )
    dialog.coordinate_system_combo.setCurrentIndex(
        dialog.coordinate_system_combo.findData("local")
    )

    assert not dialog.buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()


def test_boolean_candidate_result_disables_local_line_load_submission() -> None:
    _application()
    dialog = LoadDialog(
        ["Step-A"],
        [],
        [],
        [],
        6,
        line_regions=_regions("element_set", "BEAM-SET"),
        preferred_kind="line",
        candidate_evaluator=lambda _candidate, _step: True,
    )

    dialog.coordinate_system_combo.setCurrentIndex(
        dialog.coordinate_system_combo.findData("local")
    )

    assert not dialog.buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert "AuthoringCapability" in dialog.candidate_diagnostic_label.text()


def test_beam_component_labels_support_capability_overrides() -> None:
    _application()
    displacement = DisplacementDialog(
        ["Step-A"],
        _regions("node_set", "NODES"),
        6,
        labels=("DX", "DY", "DZ", "RX", "RY", "RZ"),
    )
    load = LoadDialog(
        ["Step-A"],
        _regions("node_set", "NODES"),
        [],
        [],
        6,
        labels=("PX", "PY", "PZ", "TX", "TY", "TZ"),
    )
    default_displacement = DisplacementDialog(
        ["Step-A"],
        _regions("node_set", "NODES"),
        6,
    )
    default_load = LoadDialog(
        ["Step-A"],
        _regions("node_set", "NODES"),
        [],
        [],
        6,
    )

    assert [
        displacement.component_checks[index].text()
        for index in range(1, 7)
    ] == ["DX", "DY", "DZ", "RX", "RY", "RZ"]
    assert [
        load.component_combo.itemText(index)
        for index in range(load.component_combo.count())
    ] == ["PX", "PY", "PZ", "TX", "TY", "TZ"]
    assert [
        default_displacement.component_checks[index].text()
        for index in range(4, 7)
    ] == ["UR1", "UR2", "UR3"]
    assert [
        default_load.component_combo.itemText(index)
        for index in range(3, 6)
    ] == ["Mx", "My", "Mz"]


def test_manager_lists_moves_and_deletes_line_loads(monkeypatch) -> None:
    _application()
    step_a = static("Step-A")
    step_a.boundaries = (
        DisplacementConstraint("NODES", 4, 4, 0.0),
    )
    step_a.cloads = (NodalLoad("NODES", 4, 2.0),)
    step_a.line_loads = (
        LineLoad("BEAM-SET", (1.0, 2.0, 3.0), "global"),
    )
    step_b = static("Step-B")
    manager = AnalysisDefinitionManagerDialog(
        [step_a, step_b],
        _regions("node_set", "NODES"),
        [],
        [],
        6,
        spatial_dimensions=3,
        line_regions=_regions("element_set", "BEAM-SET"),
        candidate_evaluator=(
            lambda _candidate, _step, **_context: _line_candidate(
                AuthoringStatus.ENABLED,
            )
        ),
    )

    line_row = manager._rows.index(("line_load", 0, 0))
    boundary_row = manager._rows.index(("boundary", 0, 0))
    node_load_row = manager._rows.index(("node_load", 0, 0))
    assert manager.table.item(line_row, 0).text() == "边力"
    assert manager.table.item(line_row, 2).text() == "BEAM-SET"
    assert manager.table.item(line_row, 3).text() == "全局 = (1, 2, 3)"
    assert manager.table.item(boundary_row, 3).text() == "UR1 = 0"
    assert manager.table.item(node_load_row, 3).text() == "Mx = 2"
    assert all(
        "新建" not in button.text()
        for button in manager.findChildren(QPushButton)
    )

    def accept_line_edit(dialog: LoadDialog) -> QDialog.DialogCode:
        assert dialog.kind_combo.currentData() == "line"
        dialog.step_combo.setCurrentText("Step-B")
        dialog.coordinate_system_combo.setCurrentIndex(
            dialog.coordinate_system_combo.findData("local")
        )
        dialog.x_spin.setValue(-4.0)
        dialog.y_spin.setValue(5.0)
        dialog.z_spin.setValue(6.0)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(LoadDialog, "exec", accept_line_edit)

    assert manager.edit_definition(("line_load", 0, 0))
    values = manager.values()
    assert values[0].line_loads == ()
    assert values[1].line_loads == (
        LineLoad("BEAM-SET", (-4.0, 5.0, 6.0), "local"),
    )

    assert manager.select_definition(("line_load", 1, 0))
    manager._delete()
    assert manager.values()[1].line_loads == ()


def test_manager_does_not_resave_a_limited_legacy_local_load(
    monkeypatch,
) -> None:
    _application()
    step = static("Step-A")
    current = LineLoad(
        "BEAM-SET",
        (0.0, -1.0, 0.0),
        "local",
    )
    step.line_loads = (current,)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        [],
        [],
        [],
        6,
        line_regions=_regions("element_set", "BEAM-SET"),
        candidate_evaluator=lambda _candidate, _step, **_context: _line_candidate(
            AuthoringStatus.LIMITED,
            code="beam.orientation.assumed",
            message="legacy compatibility frame",
        ),
    )
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )
    monkeypatch.setattr(
        LoadDialog,
        "exec",
        lambda _dialog: QDialog.DialogCode.Accepted,
    )

    assert not manager.edit_definition(("line_load", 0, 0))
    assert manager.values()[0].line_loads == (current,)
    assert warnings


def test_limited_legacy_local_load_can_be_changed_to_global(
    monkeypatch,
) -> None:
    _application()
    step = static("Step-A")
    step.line_loads = (
        LineLoad("BEAM-SET", (0.0, -1.0, 0.0), "local"),
    )
    manager = AnalysisDefinitionManagerDialog(
        [step],
        [],
        [],
        [],
        6,
        line_regions=_regions("element_set", "BEAM-SET"),
        candidate_evaluator=lambda *_args, **_kwargs: _line_candidate(
            AuthoringStatus.LIMITED,
        ),
    )

    def accept_as_global(dialog: LoadDialog):
        dialog.coordinate_system_combo.setCurrentIndex(
            dialog.coordinate_system_combo.findData("global")
        )
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(LoadDialog, "exec", accept_as_global)

    assert manager.edit_definition(("line_load", 0, 0))
    assert manager.values()[0].line_loads == (
        LineLoad("BEAM-SET", (0.0, -1.0, 0.0), "global"),
    )


def test_existing_displacement_output_is_read_only_and_not_deletable() -> None:
    _application()
    output = OutputRequest("field", "node", ("U", "RF"))
    dialog = OutputRequestDialog(["Step-A"], current=output)
    step = static("Step-A")
    step.outputs = (output,)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        [],
        [],
        [],
        3,
    )

    assert not dialog.step_combo.isEnabled()
    assert dialog.kind_value.text() == "field"
    assert not hasattr(dialog, "target_value")
    assert dialog.variables_value.text() == "U、RF"
    assert manager.select_definition(("output", 0, 0))
    assert manager.edit_button.text() == "查看"
    assert not manager.delete_button.isEnabled()
    manager._delete()
    assert manager.values()[0].outputs == (output,)


def test_line_load_tree_item_routes_double_click_to_edit() -> None:
    _application()
    step = static("Step-A")
    step.line_loads = (
        LineLoad("BEAM-SET", (0.0, -1.0, 0.0)),
    )
    model = SimpleNamespace(
        name="Beam model",
        mesh=SimpleNamespace(nodes=[None, None], elements=[None]),
        node_sets={},
        element_sets={},
        surfaces={},
        edges={},
        materials={},
        sections=[],
        steps=[step],
    )
    tree = ModelTree()
    tree.set_model(model)
    line_item = next(
        item
        for item in _tree_items(tree)
        if item.data(0, ROLE_KIND) == "line_load"
    )
    edited: list[tuple[str, object]] = []
    informed: list[tuple[str, object]] = []
    tree.editRequested.connect(
        lambda kind, key: edited.append((kind, key))
    )
    tree.informationRequested.connect(
        lambda kind, key: informed.append((kind, key))
    )

    tree._on_double_clicked(line_item)

    assert edited == [("line_load", (0, 0))]
    assert informed == []
