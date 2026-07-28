"""Checked-in GUI vertical regression for native Truss2 and Beam2 authoring."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from fem.application import (
    AuthoringStatus,
    BeamOrientation,
    DefinitionEditBatch,
    MeshEntityRef,
    NamedRegion,
    NamedRegionEditBatch,
    NativePart,
    RegionAssignment,
    SectionDefinition,
    evaluate_native_assignment_candidate,
    evaluate_native_line_load_candidate,
)
from fem.application.results import ResultVariable
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
)
from fem.geometry.recipes import WireGeometry, WireMember, WirePoint
from fem.mesh.settings import MeshSettings
from fem_gui.commands import (
    CloseSessionCommand,
    MeshInputEdit,
    NativeGeometryEdit,
    NewNativeProjectCommand,
)
from fem_gui.main_window import FEMMainWindow
from tests.helpers.gui_command_receipts import (
    await_succeeded,
    require_accepted,
)


PUBLIC_GUI_WORKFLOW_ENTRYPOINTS = (
    "new_native_project",
    "apply_native_geometry_edit",
    "apply_named_region_edit",
    "apply_mesh_input_edit",
    "apply_definition_edit",
    "save_project_path",
    "close_session",
    "open_project_path",
    "generate_mesh",
    "check_step",
    "submit_run",
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wire() -> WireGeometry:
    return WireGeometry(
        "Member",
        (
            WirePoint("RootPoint", 0.0, 0.0, 0.0),
            WirePoint("TipPoint", 1.0, 0.0, 0.0),
        ),
        (WireMember("Member", "RootPoint", "TipPoint"),),
    )


def _regions(model) -> tuple[NamedRegion, ...]:
    nodes = tuple(model.mesh.nodes)
    elements = tuple(model.mesh.elements)
    root = min(nodes, key=lambda node: float(node.x))
    tip = max(nodes, key=lambda node: float(node.x))
    return (
        NamedRegion("Root", (MeshEntityRef.node(root.id),)),
        NamedRegion("Tip", (MeshEntityRef.node(tip.id),)),
        NamedRegion(
            "Member",
            tuple(
                MeshEntityRef.element(element.id)
                for element in elements
            ),
        ),
        NamedRegion(
            "DOMAIN",
            tuple(
                MeshEntityRef.element(element.id)
                for element in elements
            ),
        ),
    )


@pytest.mark.parametrize("formulation", ("Truss2", "Beam2"))
def test_native_1d_public_gui_workflow_persists_checks_solves_and_displays(
    formulation,
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    shown_errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: shown_errors.append((title, message)),
    )

    require_accepted(
        window.new_native_project(
            NewNativeProjectCommand(f"{formulation} Project")
        )
    )
    require_accepted(
        window.apply_native_geometry_edit(
            NativeGeometryEdit(
                base_session_revision=window.document.session_revision,
                parts=(NativePart("Member", "Wire"),),
                recipe=_wire(),
            )
        )
    )
    mesh_size = 0.05 if formulation == "Truss2" else 0.25
    require_accepted(
        window.apply_mesh_input_edit(
            MeshInputEdit(
                base_session_revision=window.document.session_revision,
                settings=MeshSettings(
                    mesh_size,
                    cell_shape="line",
                    line_element_type=formulation,
                ),
            )
        )
    )
    await_succeeded(window.generate_mesh())
    generated = window.document.model
    assert generated is not None
    require_accepted(
        window.apply_named_region_edit(
            NamedRegionEditBatch(
                base_session_revision=window.document.session_revision,
                regions=_regions(generated),
            )
        )
    )

    material = MaterialDefinition(
        "Steel",
        {"E": 210000.0, "nu": 0.3},
    )
    if formulation == "Truss2":
        section = SectionDefinition(
            "Section",
            "Steel",
            "truss",
            {"area": 0.01},
        )
        orientation = None
        step = AnalysisStep(
            "Load",
            boundaries=(
                DisplacementConstraint("Root", 1, 3),
                DisplacementConstraint("Tip", 2, 3),
            ),
            cloads=(NodalLoad("Tip", 1, 10.0),),
        )
    else:
        section = SectionDefinition(
            "Section",
            "Steel",
            "rectangle",
            {"width": 0.1, "height": 0.2},
        )
        orientation = BeamOrientation((0.0, 1.0, 0.0))
        step = AnalysisStep(
            "Load",
            boundaries=(DisplacementConstraint("Root", 1, 6),),
            cloads=(NodalLoad("Tip", 2, -10.0),),
        )

    require_accepted(
        window.apply_definition_edit(
            DefinitionEditBatch(
                base_session_revision=window.document.session_revision,
                materials=(material,),
                sections=(section,),
                assignments=(),
                steps=(step,),
            )
        )
    )
    assert window.document.model is not None
    assert window.actions["section_assign"].isEnabled()

    assignment = RegionAssignment("Section", "DOMAIN", orientation)
    if formulation == "Beam2":
        automatic = evaluate_native_assignment_candidate(
            window.document,
            RegionAssignment("Section", "DOMAIN")
        )
        parallel = evaluate_native_assignment_candidate(
            window.document,
            RegionAssignment(
                "Section",
                "DOMAIN",
                BeamOrientation((1.0, 0.0, 0.0)),
            )
        )
        assert automatic.status is AuthoringStatus.ENABLED
        assert automatic.diagnostics == ()
        assert parallel.status is AuthoringStatus.UNAVAILABLE
        assert parallel.diagnostics[0].code == "beam.orientation.parallel"
    assignment_decision = evaluate_native_assignment_candidate(
        window.document,
        assignment
    )
    assert assignment_decision.can_submit
    assert evaluate_native_assignment_candidate(
        window.document,
        RegionAssignment("Section", "Member", orientation)
    ).can_submit

    if formulation == "Beam2":
        local_load = LineLoad(
            "Member",
            (0.0, -1.0, 0.0),
            "local",
        )
        before_orientation = evaluate_native_line_load_candidate(
            window.document,
            local_load,
            "Load",
        )
        assert before_orientation.status is AuthoringStatus.LIMITED
    else:
        local_load = None

    require_accepted(
        window.apply_definition_edit(
            DefinitionEditBatch(
                base_session_revision=window.document.session_revision,
                materials=(material,),
                sections=(section,),
                assignments=(assignment,),
                steps=(step,),
            )
        )
    )
    if local_load is not None:
        local_decision = evaluate_native_line_load_candidate(
            window.document,
            local_load,
            "Load",
        )
        assert local_decision.can_submit
        step.line_loads = (local_load,)
        require_accepted(
            window.apply_definition_edit(
                DefinitionEditBatch(
                    base_session_revision=window.document.session_revision,
                    materials=(material,),
                    sections=(section,),
                    assignments=(assignment,),
                    steps=(step,),
                )
            )
        )

    project_path = tmp_path / f"{formulation.casefold()}-native.femproj"
    await_succeeded(window.save_project_path(project_path))
    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    await_succeeded(window.open_project_path(project_path))
    assert window.document.model is None
    assert window.document.assignments == (assignment,)
    if local_load is not None:
        assert window.document.steps[0].line_loads == (local_load,)

    await_succeeded(window.generate_mesh())
    model = window.document.model
    assert model is not None
    assert {element.type for element in model.mesh.elements} == {formulation}
    assert model.mesh.dofs_per_node == (3 if formulation == "Truss2" else 6)
    assert window.actions["nodes"].isChecked()
    assert window.viewport._show_nodes
    if formulation == "Truss2":
        assert len(model.mesh.nodes) == 2
        assert len(model.mesh.elements) == 1

    await_succeeded(window.check_step("Load"))
    assert window.document.validation_current("Load")
    await_succeeded(window.submit_run(f"{formulation}-Job", "Load"))

    record = window.session.current_result()
    provider = window.result_provider
    selection = window.result_selection
    payload = window.viewport._result_render_payload
    assert record is not None
    assert np.isfinite(record.result.U).all()
    assert provider is not None
    assert selection is not None
    assert payload is not None
    assert selection.field_key.request.field_id.variable is ResultVariable.U
    assert payload.topology.source == provider.source
    assert shown_errors == []

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()
