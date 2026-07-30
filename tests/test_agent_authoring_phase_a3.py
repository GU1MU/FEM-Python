from __future__ import annotations

import pytest

from fem.geometry import (
    LogicalEntityRef,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    legacy_sketch_to_strict,
)
from fem.mesh.settings import LocalMeshControl, MeshSizeFalloff
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_agent.tools.registry import AgentToolRegistry
from fem_gui.agent_authoring import authoring_context_from_snapshot
from fem.application import ModelSession, UnitContext


def _recipe() -> SketchGeometry:
    return legacy_sketch_to_strict(
        SketchGeometry(
            "草图-通用孔板",
            (
                SketchRectangle("material", 0.0, 0.0, 10.0, 6.0),
                SketchCircle("cut", 6.5, 2.0, 1.0),
            ),
        )
    )


def _local_control() -> LocalMeshControl:
    return LocalMeshControl(
        LogicalEntityRef("edge:C5"),
        0.2,
        MeshSizeFalloff("target_radius", 0.25, 2.0),
    )


def test_a3_mesh_intent_requires_exactly_one_density_mode() -> None:
    explicit = MeshIntent(
        "triangle",
        1,
        global_size=1.0,
    )
    automatic = MeshIntent(
        "quadrilateral",
        2,
        auto_level=4,
    )

    assert explicit.mode == "explicit"
    assert automatic.mode == "automatic"
    with pytest.raises(ValueError, match="exactly one"):
        MeshIntent("triangle", 1)
    with pytest.raises(ValueError, match="exactly one"):
        MeshIntent(
            "triangle",
            1,
            global_size=1.0,
            auto_level=3,
        )


def test_a3_mesh_intent_json_round_trip_keeps_generic_stable_ref_and_falloff() -> None:
    intent = MeshIntent(
        "quadrilateral",
        2,
        auto_level=4,
        local_controls=(_local_control(),),
    )

    restored = MeshIntent.from_dict(intent.to_dict())

    assert restored == intent
    assert restored.intent_hash == intent.intent_hash
    assert restored.local_controls[0].target == LogicalEntityRef(
        "edge:C5"
    )
    assert restored.local_controls[0].falloff == MeshSizeFalloff(
        "target_radius",
        0.25,
        2.0,
    )


def test_a3_mesh_intent_connects_mesh_settings_and_strict_auto_spec() -> None:
    explicit = MeshIntent(
        "quadrilateral",
        1,
        global_size=0.8,
        local_controls=(_local_control(),),
    )
    automatic = MeshIntent(
        "triangle",
        2,
        auto_level=4,
    )

    explicit_settings = explicit.to_mesh_settings(_recipe())
    automatic_settings = automatic.to_mesh_settings(_recipe())
    auto_spec = automatic.to_auto_mesh_spec()

    assert explicit_settings.size == 0.8
    assert explicit_settings.auto_level is None
    assert explicit_settings.strict_cell_shape is True
    assert automatic_settings.auto_level == 4
    assert automatic_settings.strict_cell_shape is True
    assert auto_spec is not None
    assert (auto_spec.level, auto_spec.cell_shape, auto_spec.order) == (
        4,
        "tri",
        2,
    )


def test_a3_mesh_proposal_is_revision_bound_and_uses_local_gui_summary() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        _recipe(),
        part_name="部件-偏心孔板",
    )
    context = authoring_context_from_snapshot(session.snapshot())
    intent = MeshIntent(
        "quadrilateral",
        1,
        global_size=0.8,
        local_controls=(_local_control(),),
    )

    proposal = create_mesh_proposal(
        proposal_id="proposal-mesh-a3",
        agent_session_id="agent-session-a3",
        turn_id="turn-a3",
        source_tool_call_ids=("call-mesh-a3",),
        context=context,
        draft_revision=2,
        part_id="P1",
        mesh_intent=intent,
    )

    assert [item.kind.value for item in proposal.operations] == [
        "set_part_mesh_intent",
        "request_mesh",
    ]
    assert proposal.base_session_revision == session.session_revision
    assert proposal.display_summary["confirm_label"] == "开始划分"
    assert proposal.display_summary["estimate_only"] is True
    assert proposal.display_summary["local_refinements"][0]["target"] == (
        "edge:C5"
    )
    assert proposal.operations[1].parameters["mesh_intent_hash"] == (
        intent.intent_hash
    )


def test_a3_provider_tool_catalog_exposes_no_confirmation_capability(
    tmp_path,
) -> None:
    names = {
        definition.name
        for definition in AgentToolRegistry(tmp_path / "workspace").definitions
    }

    assert "accept_proposal" not in names
    assert "confirm_mesh" not in names
    assert "confirm_solve" not in names
