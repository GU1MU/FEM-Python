from __future__ import annotations

from fem.application import ModelSession
from fem_agent.definition_authoring import create_scope_definition_change
from fem_gui.agent_authoring import authoring_context_from_snapshot


def build_plate_definition_patch(session: ModelSession):
    snapshot = session.snapshot()
    return create_scope_definition_change(
        patch_id="patch-a4",
        proposal_id="proposal-a4",
        agent_session_id="agent-a4",
        turn_id="turn-a4",
        source_tool_call_ids=("call-a4",),
        context=authoring_context_from_snapshot(snapshot),
        snapshot=snapshot,
        draft_revision=4,
        material_function="结构钢",
        material_properties={"E": 210000.0, "nu": 0.3},
        section_function="平面应力",
        plane_type="stress",
        thickness=2.0,
    )
