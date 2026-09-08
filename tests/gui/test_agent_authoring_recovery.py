from __future__ import annotations

from tests.helpers.agent_authoring_workflows import (
    STATIC_STEP_NAME,
    make_authoring_controller,
    dispatch_authoring_tool,
    apply_static_analysis_definitions,
    solve_and_read_displacement,
)

from copy import deepcopy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fem.application import ModelSession
from fem.io.project import load_project, save_project
from fem.mesh.settings import MeshSettings
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from tests.helpers.agent_session_fixtures import _a5_session


def test_production_entry_solves_and_reads_one_accepted_result() -> None:
    session = _a5_session()
    controller, bridge = make_authoring_controller(session)

    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    apply_static_analysis_definitions(controller, session)
    scalar = solve_and_read_displacement(controller, bridge, session)

    assert scalar["location"]["association"] == "node"


def test_save_reopen_remesh_resumes_preflight_solve_and_result(
    tmp_path,
) -> None:
    session = _a5_session()
    controller, _bridge = make_authoring_controller(session)
    apply_static_analysis_definitions(controller, session)
    remeshed_candidate = deepcopy(session.snapshot().artifact.model)

    prepared = session.prepare_project_save()
    target = save_project(tmp_path / "accepted.femproj", prepared)
    assert session.accept_project_saved(prepared.token, target).accepted

    reopened = ModelSession()
    reopened.replace_from_snapshot(load_project(target).snapshot)
    task = reopened.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(0.8),
        "b" * 64,
        expected_session_revision=reopened.session_revision,
    )
    assert reopened.accept_agent_generated_model(
        task.token,
        remeshed_candidate,
    ).accepted

    resumed, bridge = make_authoring_controller(reopened)

    assert resumed.stage is AuthoringWorkflowStage.PREFLIGHT_READY
    assert reopened.snapshot().steps[0].name == STATIC_STEP_NAME
    solve_and_read_displacement(resumed, bridge, reopened)


def test_direct_definitions_reject_nonconforming_visible_names() -> None:
    session = _a5_session()
    controller, _bridge = make_authoring_controller(session)
    before = session.snapshot()

    rejected = dispatch_authoring_tool(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_material",
            "parameters": {
                "name": "steel",
                "properties": {"E": 70000.0, "nu": 0.33},
            },
        },
        "bad-name",
    )

    assert not rejected.ok
    assert session.snapshot() == before
