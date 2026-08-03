from __future__ import annotations

import pytest

from fem.geometry import LogicalEntityRef
from fem.geometry.recipes import WireGeometry, WireMember, WirePoint
from fem.mesh.settings import LocalMeshControl
from fem_agent.authoring import (
    AuthoringContext,
    LocalModelBinding,
    PartSummary,
)
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
)
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_agent.tools.registry import ToolExecutionContext


def _wire() -> WireGeometry:
    return WireGeometry(
        "Frame",
        (
            WirePoint("A", 0.0, 0.0, 0.0),
            WirePoint("B", 1.0, 0.0, 0.0),
        ),
        (WireMember("AB", "A", "B"),),
    )


def _context(*, dimension: int) -> AuthoringContext:
    return AuthoringContext(
        binding=LocalModelBinding(
            "document:line-phase2",
            "session-line-phase2",
            4,
            "native",
            True,
        ),
        model_name="Line model",
        active_part_id="P1",
        parts=(PartSummary("P1", "Wire", "wire", dimension, False),),
    )


@pytest.mark.parametrize("line_element_type", ["Truss2", "Beam2"])
def test_line_mesh_intent_uses_strict_schema_11_and_round_trips(
    line_element_type: str,
) -> None:
    intent = MeshIntent(
        "line",
        1,
        global_size=0.25,
        local_controls=(
            LocalMeshControl(LogicalEntityRef("edge:AB"), 0.1),
        ),
        line_element_type=line_element_type,
    )

    payload = intent.to_dict()
    restored = MeshIntent.from_dict(payload)
    settings = restored.to_mesh_settings(_wire())

    assert payload["schema_version"] == "1.1"
    assert payload["line_element_type"] == line_element_type
    assert restored == intent
    assert settings.cell_shape == "line"
    assert settings.order == 1
    assert settings.line_element_type == line_element_type
    automatic = MeshIntent(
        "line",
        1,
        auto_level=3,
        line_element_type=line_element_type,
    ).to_auto_mesh_spec()
    assert automatic is not None
    assert automatic.cell_shape is None


def test_legacy_planar_mesh_intent_schema_10_remains_exact() -> None:
    legacy = {
        "schema_version": "1.0",
        "mode": "explicit",
        "global_size": 0.5,
        "auto_level": None,
        "cell_shape": "triangle",
        "order": 1,
        "local_controls": [],
    }

    restored = MeshIntent.from_dict(legacy)

    assert restored.to_dict() == legacy
    with pytest.raises(ValueError, match="schema 1.0"):
        MeshIntent.from_dict({**legacy, "line_element_type": None})


@pytest.mark.parametrize(
    "factory, message",
    [
        (
            lambda: MeshIntent("line", 1, global_size=0.5),
            "line_element_type",
        ),
        (
            lambda: MeshIntent(
                "line",
                2,
                global_size=0.5,
                line_element_type="Truss2",
            ),
            "order must be 1",
        ),
        (
            lambda: MeshIntent(
                "line",
                1,
                global_size=0.5,
                line_element_type="Truss2/Beam2",
            ),
            "Truss2 or Beam2",
        ),
    ],
)
def test_line_mesh_intent_rejects_missing_second_order_and_mixed_formulations(
    factory,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_line_mesh_proposal_retains_formulation_in_summary_and_hash() -> None:
    intent = MeshIntent(
        "line",
        1,
        global_size=0.5,
        line_element_type="Beam2",
    )

    proposal = create_mesh_proposal(
        proposal_id="proposal-line-phase2",
        agent_session_id="agent-line-phase2",
        turn_id="turn-line-phase2",
        source_tool_call_ids=("call-line-phase2",),
        context=_context(dimension=1),
        draft_revision=1,
        part_id="P1",
        mesh_intent=intent,
    )

    assert proposal.display_summary["cell_shape"] == "line"
    assert proposal.display_summary["order"] == 1
    assert proposal.display_summary["line_element_type"] == "Beam2"
    assert proposal.operations[1].parameters["mesh_intent_hash"] == (
        intent.intent_hash
    )


def test_runtime_exposes_dimension_specific_line_mesh_requirements() -> None:
    context = _context(dimension=1)
    controller = AuthoringWorkflowController(
        lambda: context,
        {
            "prepare_mesh_proposal": lambda _arguments, _controller: (
                AuthoringToolOutcome("prepared", {})
            )
        },
    )
    controller.observe_binding(context)
    requirement_tool = next(
        item
        for item in controller.definitions
        if item.name == "set_authoring_requirements"
    )
    properties = requirement_tool.parameters["properties"]["requirements"][
        "properties"
    ]

    assert set(properties) == {
        "mesh_cell_shape",
        "mesh_order",
        "mesh_global_size",
        "line_element_type",
    }
    assert properties["mesh_cell_shape"]["enum"] == ["line"]
    assert properties["mesh_order"]["enum"] == [1]
    assert properties["line_element_type"]["enum"] == ["Truss2", "Beam2"]

    execution = ToolExecutionContext("agent-line-phase2", 0, "requirements")
    partial = controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-line-phase2",
            "requirements": {
                "mesh_cell_shape": "line",
                "mesh_order": 1,
                "mesh_global_size": 0.5,
            },
        },
        execution,
    )
    assert partial.ok
    assert partial.data["missing_requirements"] == ["line_element_type"]
    assert "prepare_mesh_proposal" not in {
        item.name for item in controller.definitions
    }

    complete = controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-line-phase2",
            "requirements": {"line_element_type": "Truss2"},
        },
        ToolExecutionContext("agent-line-phase2", 0, "formulation"),
    )
    assert complete.ok
    assert complete.data["missing_requirements"] == []
    assert "prepare_mesh_proposal" in {
        item.name for item in controller.definitions
    }


def test_runtime_keeps_planar_mesh_requirements_free_of_line_formulation() -> None:
    context = _context(dimension=2)
    controller = AuthoringWorkflowController(lambda: context, {})
    controller.observe_binding(context)
    requirement_tool = next(
        item
        for item in controller.definitions
        if item.name == "set_authoring_requirements"
    )
    properties = requirement_tool.parameters["properties"]["requirements"][
        "properties"
    ]

    assert set(properties) == {
        "mesh_cell_shape",
        "mesh_order",
        "mesh_global_size",
    }
    assert properties["mesh_cell_shape"]["enum"] == [
        "triangle",
        "quadrilateral",
    ]
