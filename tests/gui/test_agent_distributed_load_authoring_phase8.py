from __future__ import annotations

from fem.application import (
    MeshEntityRef,
    NamedRegion,
    ScopedDefinitionBatch,
    UnitContext,
)
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from tests.gui.test_agent_authoring_recovery_phase_a8 import (
    _apply_analysis_definitions,
    _dispatch,
    _production_controller,
    _solve_and_read_displacement,
)
from tests.gui.test_agent_definition_actions_a4_a5 import _surface_session
from tests.integration.test_agent_beam2_authoring_phase4 import (
    STEP_NAME as BEAM_STEP_NAME,
    _base_scopes_and_material,
    _beam_definition_actions,
)
from tests.integration.test_agent_truss2_authoring_phase3 import (
    _meshed_session,
)
from tests.helpers.agent_session_fixtures import _a5_session as _plate_session


PLATE_STEP_NAME = "分析步-静力"


def _create_step(controller: object, session: object, name: str) -> None:
    result = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {"action": "create_static_step", "parameters": {"name": name}},
        "phase8-step",
    )
    assert result.ok, result.to_json()


def _add_all_element_region(session: object, name: str) -> None:
    snapshot = session.snapshot()
    model = snapshot.artifact.model
    session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            snapshot.session_revision,
            tuple(snapshot.named_regions.values())
            + (
                NamedRegion(
                    name,
                    tuple(
                        MeshEntityRef.element(element.id, part_id="P1")
                        for element in model.mesh.elements
                    ),
                ),
            ),
            tuple(snapshot.materials),
            tuple(snapshot.sections),
            tuple(snapshot.assignments),
            tuple(snapshot.steps),
        )
    )


def test_phase8_schema_publishes_closed_line_body_and_gravity_branches() -> None:
    session = _plate_session()
    controller, _bridge = _production_controller(session)
    tool = next(
        item for item in controller.definitions
        if item.name == "apply_model_definition"
    )
    load_action = next(
        branch
        for branch in tool.parameters["oneOf"]
        if branch["properties"]["action"].get("const") == "create_load"
    )
    branches = load_action["properties"]["parameters"]["oneOf"]
    extended = {
        branch["properties"]["load_type"]["const"]: branch
        for branch in branches
        if branch["properties"]["load_type"].get("const")
        in {"line", "body", "gravity"}
    }

    assert set(extended) == {"line", "body", "gravity"}
    assert all(branch["additionalProperties"] is False for branch in extended.values())
    assert extended["line"]["properties"]["vector"]["minItems"] == 3
    assert extended["line"]["properties"]["vector"]["maxItems"] == 3
    assert extended["gravity"]["properties"]["target_scope"]["oneOf"][0] == {
        "type": "null"
    }


def test_phase8_creates_body_force_and_fails_closed_on_wrong_dimension() -> None:
    session = _plate_session()
    controller, _bridge = _production_controller(session)
    _create_step(controller, session, PLATE_STEP_NAME)

    created = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-体力",
                "step_name": PLATE_STEP_NAME,
                "target_scope": "域-板体",
                "entity_type": "element",
                "load_type": "body",
                "vector": [1.0, -2.0],
                "direction": "global",
                "unit": "N/mm^3",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "body-force",
    )
    assert created.ok, created.to_json()
    assert session.snapshot().steps[0].body_loads[0].vector == (1.0, -2.0)

    revision = session.session_revision
    rejected = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-错误体力",
                "step_name": PLATE_STEP_NAME,
                "target_scope": "域-板体",
                "entity_type": "element",
                "load_type": "body",
                "vector": [1.0, 2.0, 3.0],
                "direction": "global",
                "unit": "N/mm^3",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "wrong-body-force",
    )
    assert not rejected.ok
    assert session.session_revision == revision

    invalid_edits = (
        ({"vector": [3.0, 4.0, 5.0]}, "wrong-body-edit-dimension"),
        ({"vector": [3.0, 4.0], "unit": "N/mm"}, "wrong-body-edit-unit"),
        (
            {"vector": [3.0, 4.0], "target_scope": "边-加载端"},
            "wrong-body-edit-scope",
        ),
    )
    for overrides, key in invalid_edits:
        changes = {
            "direction": "global",
            "unit": "N/mm^3",
            "distribution": "uniform",
            "confirmed": True,
            **overrides,
        }
        rejected_edit = _dispatch(
            controller,
            session,
            "edit_model_object",
            {
                "object_type": "load",
                "target_id": "载荷-体力",
                "step_name": PLATE_STEP_NAME,
                "changes": changes,
            },
            key,
        )
        assert not rejected_edit.ok
        assert session.session_revision == revision
        assert session.snapshot().steps[0].body_loads[0].vector == (1.0, -2.0)


def test_phase8_creates_global_and_local_beam_line_loads() -> None:
    session = _meshed_session("Beam2")
    controller, _bridge = _production_controller(session)
    _beam_definition_actions(controller, session)

    for coordinate_system in ("global", "local"):
        result = _dispatch(
            controller,
            session,
            "apply_model_definition",
            {
                "action": "create_load",
                "parameters": {
                    "name": f"载荷-梁-{coordinate_system}",
                    "step_name": BEAM_STEP_NAME,
                    "target_scope": "域-梁",
                    "entity_type": "element",
                    "load_type": "line",
                    "vector": [0.0, -5.0, 0.0],
                    "coordinate_system": coordinate_system,
                    "unit": "N/mm",
                    "distribution": "uniform",
                    "confirmed": True,
                },
            },
            f"line-{coordinate_system}",
        )
        assert result.ok, result.to_json()

    assert tuple(
        item.coordinate_system for item in session.snapshot().steps[0].line_loads
    ) == ("global", "local")

    edited = _dispatch(
        controller,
        session,
        "edit_model_object",
        {
            "object_type": "load",
            "target_id": "载荷-梁-global",
            "step_name": BEAM_STEP_NAME,
            "changes": {
                "vector": [1.0, -8.0, 2.0],
                "coordinate_system": "global",
                "unit": "N/mm",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "edit-global-line",
    )
    assert edited.ok, edited.to_json()
    assert session.snapshot().steps[0].line_loads[0].vector == (1.0, -8.0, 2.0)

    revision = session.session_revision
    rejected = _dispatch(
        controller,
        session,
        "edit_model_object",
        {
            "object_type": "load",
            "target_id": "载荷-梁-global",
            "step_name": BEAM_STEP_NAME,
            "changes": {
                "vector": [9.0, 9.0, 9.0],
                "coordinate_system": "global",
                "unit": "N",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "invalid-line-edit",
    )
    assert not rejected.ok
    assert session.session_revision == revision
    assert session.snapshot().steps[0].line_loads[0].vector == (1.0, -8.0, 2.0)

    invalid_creates = (
        ({"coordinate_system": "cylindrical"}, "invalid-line-coordinate"),
        ({"unit": "N"}, "invalid-line-unit"),
        ({"target_scope": "点-自由端"}, "invalid-line-scope"),
    )
    for index, (overrides, key) in enumerate(invalid_creates):
        parameters = {
            "name": f"载荷-无效梁-{index}",
            "step_name": BEAM_STEP_NAME,
            "target_scope": "域-梁",
            "entity_type": "element",
            "load_type": "line",
            "vector": [0.0, -5.0, 0.0],
            "coordinate_system": "global",
            "unit": "N/mm",
            "distribution": "uniform",
            "confirmed": True,
            **overrides,
        }
        invalid = _dispatch(
            controller,
            session,
            "apply_model_definition",
            {"action": "create_load", "parameters": parameters},
            key,
        )
        assert not invalid.ok
        assert session.session_revision == revision


def test_phase8_line_rejects_non_beam_element_region_atomically() -> None:
    session = _plate_session()
    controller, _bridge = _production_controller(session)
    _create_step(controller, session, PLATE_STEP_NAME)
    revision = session.session_revision

    rejected = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-非梁线载荷",
                "step_name": PLATE_STEP_NAME,
                "target_scope": "域-板体",
                "entity_type": "element",
                "load_type": "line",
                "vector": [0.0, -5.0, 0.0],
                "coordinate_system": "global",
                "unit": "N/mm",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "non-beam-line",
    )
    assert not rejected.ok
    assert session.session_revision == revision


def test_phase8_local_line_and_gravity_fail_closed_without_capability() -> None:
    session = _meshed_session("Beam2")
    controller, _bridge = _production_controller(session)
    _base_scopes_and_material(controller, session)
    _create_step(controller, session, BEAM_STEP_NAME)
    revision = session.session_revision

    local = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-局部未定向",
                "step_name": BEAM_STEP_NAME,
                "target_scope": "域-梁",
                "entity_type": "element",
                "load_type": "line",
                "vector": [0.0, -5.0, 0.0],
                "coordinate_system": "local",
                "unit": "N/mm",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "unresolved-local-line",
    )
    assert not local.ok
    assert session.session_revision == revision

    gravity = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-重力",
                "step_name": BEAM_STEP_NAME,
                "target_scope": None,
                "entity_type": "element",
                "load_type": "gravity",
                "acceleration": [0.0, -9810.0, 0.0],
                "direction": "global",
                "unit": "mm/s^2",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "missing-acceleration-unit",
    )
    assert not gravity.ok
    assert session.session_revision == revision
    assert controller.stage is AuthoringWorkflowStage.PREFLIGHT_READY


def test_phase8_gravity_accepts_global_target_with_explicit_unit() -> None:
    session = _plate_session(
        UnitContext("mm", "N", "MPa", acceleration="mm/s^2")
    )
    controller, _bridge = _production_controller(session)
    _create_step(controller, session, PLATE_STEP_NAME)

    created = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-重力",
                "step_name": PLATE_STEP_NAME,
                "target_scope": None,
                "entity_type": "element",
                "load_type": "gravity",
                "acceleration": [0.0, -9810.0],
                "direction": "global",
                "unit": "mm/s^2",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "gravity-global",
    )
    assert created.ok, created.to_json()
    stored = session.snapshot().steps[0].gravity_loads[0]
    assert stored.target is None
    assert stored.acceleration == (0.0, -9810.0)

    scoped = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-区域重力",
                "step_name": PLATE_STEP_NAME,
                "target_scope": "域-板体",
                "entity_type": "element",
                "load_type": "gravity",
                "acceleration": [9810.0, 0.0],
                "direction": "global",
                "unit": "mm/s^2",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "gravity-scoped",
    )
    assert scoped.ok, scoped.to_json()
    stored_scoped = session.snapshot().steps[0].gravity_loads[1]
    assert stored_scoped.target == "域-板体"
    assert stored_scoped.acceleration == (9810.0, 0.0)


def test_phase8_creates_three_dimensional_body_force() -> None:
    session = _surface_session()
    _add_all_element_region(session, "域-三维块")
    controller, _bridge = _production_controller(session)

    created = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-三维体力",
                "step_name": PLATE_STEP_NAME,
                "target_scope": "域-三维块",
                "entity_type": "element",
                "load_type": "body",
                "vector": [1.0, -2.0, 3.0],
                "direction": "global",
                "unit": "N/mm^3",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "body-force-3d",
    )
    assert created.ok, created.to_json()
    assert session.snapshot().steps[0].body_loads[0].vector == (1.0, -2.0, 3.0)


def test_phase8_new_load_retains_accepted_result_history() -> None:
    session = _plate_session()
    controller, bridge = _production_controller(session)
    _apply_analysis_definitions(controller, session)
    _solve_and_read_displacement(controller, bridge, session)
    before = session.snapshot()
    run_ids = tuple(run.run_id for run in before.runs)
    assert run_ids
    assert all(session.result_for(run_id) is not None for run_id in run_ids)

    created = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-求解后体力",
                "step_name": PLATE_STEP_NAME,
                "target_scope": "域-板体",
                "entity_type": "element",
                "load_type": "body",
                "vector": [0.0, -1.0],
                "direction": "global",
                "unit": "N/mm^3",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "post-result-body-force",
    )
    assert created.ok, created.to_json()

    after = session.snapshot()
    assert after.session_id == before.session_id
    assert after.source_kind == before.source_kind == "native"
    assert after.active_part_id == before.active_part_id
    assert after.artifact.model.name == before.artifact.model.name
    assert tuple(run.run_id for run in after.runs) == run_ids
    assert all(session.result_for(run_id) is not None for run_id in run_ids)
    assert after.displayed_result_run_id is None
    assert not after.validations
    assert controller.stage is AuthoringWorkflowStage.PREFLIGHT_READY
