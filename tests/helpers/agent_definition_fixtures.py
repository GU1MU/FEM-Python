from __future__ import annotations

from fem.application import ModelSession
from fem_agent.incremental_authoring import create_incremental_definition_patch
from fem_gui.agent_authoring import authoring_context_from_snapshot


def build_material_definition_patch(session: ModelSession):
    snapshot = session.snapshot()
    return create_incremental_definition_patch(
        patch_id="patch-a4",
        agent_session_id="agent-a4",
        turn_id="turn-a4",
        source_tool_call_ids=("call-a4",),
        context=authoring_context_from_snapshot(snapshot),
        snapshot=snapshot,
        draft_revision=4,
        action="create_material",
        parameters={
            "name": "Material-结构钢",
            "properties": {"E": 210000.0, "nu": 0.3},
        },
    )
