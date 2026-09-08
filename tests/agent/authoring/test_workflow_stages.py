from __future__ import annotations

import threading

from fem_agent.authoring import (
    AuthoringContext,
    CapabilitySummary,
    LocalModelBinding,
    MeshSummary,
    PartSummary,
    ProposalState,
)
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
)

from tests.helpers.agent_authoring_workflow_fixtures import (
    _context,
    _requirements_for,
    _dispatch,
)


def test_geometry_uses_one_operation_confirmation_without_requirement_review() -> None:
    calls: list[str] = []

    def geometry(_arguments, _controller):
        calls.append("geometry")
        return AuthoringToolOutcome(
            "Geometry proposal registered.",
            {"state": "pending_confirmation"},
        )

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"prepare_geometry_proposal": geometry},
    )
    initial_names = {item.name for item in controller.definitions}
    assert "set_authoring_requirements" in initial_names
    assert "prepare_geometry_proposal" not in initial_names
    assert not any(
        fragment in name
        for name in initial_names
        for fragment in ("accept", "confirm", "reject", "cancel")
    )

    recorded = _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-geometry",
            "requirements": _requirements_for("geometry"),
        },
        1,
    )
    assert recorded.ok
    assert recorded.data["operation_confirmation_required"] is True
    assert recorded.data["next_action"] == "prepare_stage_proposal"
    names = {item.name for item in controller.definitions}
    assert "prepare_geometry_proposal" in names
    assert "set_authoring_requirements" in names
    prepared = _dispatch(controller, "prepare_geometry_proposal", {}, 2)
    assert prepared.ok
    assert calls == ["geometry"]
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING


def test_mesh_uses_one_confirmation_and_definitions_are_direct() -> None:
    calls: list[str] = []

    def handler(arguments, _controller):
        calls.append(str(arguments.get("action", "mesh")))
        return AuthoringToolOutcome(
            "Applied.",
            {
                "state": "succeeded",
                "definition_object_type": "named_region",
            },
        )

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {
            "prepare_mesh_proposal": handler,
            "apply_model_definition": handler,
            "run_native_preflight": handler,
        },
    )
    controller._stage = AuthoringWorkflowStage.MESH_READY
    assert "prepare_mesh_proposal" not in {
        item.name for item in controller.definitions
    }

    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-mesh",
            "requirements": _requirements_for("mesh"),
        },
        3,
    ).ok
    names = {item.name for item in controller.definitions}
    assert "prepare_mesh_proposal" in names

    controller._stage = AuthoringWorkflowStage.DEFINITIONS_READY
    names = {item.name for item in controller.definitions}
    assert {"apply_model_definition", "run_native_preflight"} <= names
    assert "set_authoring_requirements" in names
    assert "prepare_mesh_proposal" in names

    applied = _dispatch(
        controller,
        "apply_model_definition",
        {
            "action": "create_material",
            "parameters": {
                "name": "材料-铝合金",
                "properties": {"E": 70000.0, "nu": 0.33},
            },
        },
        4,
    )
    assert applied.ok
    assert calls == ["create_material"]
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY


def test_only_geometry_mesh_and_solve_publish_execution_proposals() -> None:
    def handler(_arguments, _controller) -> AuthoringToolOutcome:
        return AuthoringToolOutcome(
            "Registered.",
            {"state": "pending_confirmation"},
        )

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {
            "prepare_geometry_proposal": handler,
            "prepare_mesh_proposal": handler,
            "prepare_solve_proposal": handler,
            "apply_model_definition": handler,
            "edit_model_object": handler,
        },
    )

    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-geometry",
            "requirements": _requirements_for("geometry"),
        },
        5,
    ).ok
    assert "prepare_geometry_proposal" in {
        item.name for item in controller.definitions
    }

    controller._stage = AuthoringWorkflowStage.MESH_READY
    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-mesh",
            "requirements": _requirements_for("mesh"),
        },
        6,
    ).ok
    assert "prepare_mesh_proposal" in {
        item.name for item in controller.definitions
    }

    controller._stage = AuthoringWorkflowStage.SOLVE_READY
    names = {item.name for item in controller.definitions}
    assert "prepare_solve_proposal" in names
    assert "apply_model_definition" in names


def test_geometry_edit_is_available_after_creation_and_returns_to_mesh() -> None:
    context = AuthoringContext(
        binding=LocalModelBinding(
            "document:edit",
            "native-edit",
            4,
            "native",
            True,
        ),
        model_name="模型-双孔板",
        active_part_id="part-1",
        parts=(
            PartSummary(
                "part-1",
                "部件-孔板",
                "planar_sketch",
                2,
                False,
            ),
        ),
        capabilities=(CapabilitySummary("edit_native_geometry", True),),
    )
    calls: list[str] = []

    def handler(arguments, _controller):
        calls.append(str(arguments.get("part_id")))
        return AuthoringToolOutcome(
            "Geometry edit prepared.",
            {"state": "pending_confirmation"},
        )

    controller = AuthoringWorkflowController(
        lambda: context,
        {
            "read_geometry_edit_context": handler,
            "prepare_geometry_edit": handler,
        },
    )
    controller._stage = AuthoringWorkflowStage.MESH_READY

    names = {item.name for item in controller.definitions}
    assert {
        "read_geometry_edit_context",
        "prepare_geometry_edit",
    } <= names
    prepared = _dispatch(
        controller,
        "prepare_geometry_edit",
        {
            "part_id": "part-1",
            "edit": {
                "operation": "add_circle",
                "center_x": 50.0,
                "center_y": 130.0,
                "radius": 5.0,
            },
        },
        7,
    )

    assert prepared.ok
    assert calls == ["part-1"]
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING

    controller.record_proposal_state("geometry", ProposalState.SUCCEEDED)

    assert controller.stage is AuthoringWorkflowStage.MESH_READY


def test_existing_current_mesh_exposes_direct_definitions_immediately() -> None:
    context = AuthoringContext(
        binding=LocalModelBinding(
            "document:existing",
            "native-existing",
            7,
            "native",
            True,
        ),
        model_name="模型-既有",
        active_part_id="part-existing",
        mesh=MeshSummary(True, True, 20, 10),
    )
    controller = AuthoringWorkflowController(
        lambda: context,
        {
            "apply_model_definition": lambda _arguments, _controller: (
                AuthoringToolOutcome("Applied.", {"state": "succeeded"})
            ),
        },
    )
    controller.observe_binding(context)

    assert "apply_model_definition" in {
        item.name for item in controller.definitions
    }


def test_dynamic_dispatch_is_serial_and_thread_safe() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked(_arguments, _controller):
        entered.set()
        release.wait(timeout=2.0)
        return AuthoringToolOutcome(
            "Applied.",
            {
                "state": "succeeded",
                "definition_object_type": "named_region",
            },
        )

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"apply_model_definition": blocked},
    )
    controller._stage = AuthoringWorkflowStage.DEFINITIONS_READY
    results = []

    def invoke() -> None:
        results.append(
            _dispatch(
                controller,
                "apply_model_definition",
                {
                    "action": "create_material",
                    "parameters": {
                        "name": "材料-测试",
                        "properties": {"E": 1.0},
                    },
                },
                len(results) + 20,
            )
        )

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    assert entered.wait(timeout=1.0)
    second.start()
    release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert len(results) == 2
    assert all(item.ok for item in results)
