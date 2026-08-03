"""Revision-bound Agent proposals for proven exact Part Booleans."""

from __future__ import annotations

from collections.abc import Sequence
import json

from fem.application.part_boolean import StrictPartBooleanResult
from fem.application.solid_boolean import StrictBodyBooleanResult

from .authoring import (
    AgentProposal,
    AuthoringContext,
    AuthoringContractError,
    ModelOperation,
    OperationKind,
    ProposalKind,
)
from .geometry_authoring import geometry_recipe_to_payload


PART_BOOLEAN_TOOL_HANDLING = "consume_tool_part"
BODY_BOOLEAN_TOOL_HANDLING = "consume_tool_body"


def create_part_boolean_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    target_part_id: str,
    tool_part_id: str,
    operation: str,
    result_name: str,
    tool_handling: str,
    prepared: StrictPartBooleanResult,
    summary: str,
) -> AgentProposal:
    """Bind one detached OCC proof to the current native Session revision."""

    if type(context) is not AuthoringContext:
        raise TypeError("context must be AuthoringContext")
    if type(prepared) is not StrictPartBooleanResult:
        raise TypeError("prepared must be StrictPartBooleanResult")
    if operation not in {"fuse", "cut"}:
        raise AuthoringContractError(
            "Agent exact Boolean supports fuse and cut only; "
            "intersect/fragment remain disabled until stable multi-result "
            "Body IDs and lineage replay are proven"
        )
    if tool_handling != PART_BOOLEAN_TOOL_HANDLING:
        raise AuthoringContractError(
            f"tool_handling must be {PART_BOOLEAN_TOOL_HANDLING!r}"
        )
    if (
        not context.binding.supported
        or context.binding.source_kind != "native"
    ):
        raise AuthoringContractError(
            "exact Part Boolean requires an editable native project"
        )
    by_id = {
        part.part_id: part
        for part in context.parts
        if not part.suppressed
    }
    target = by_id.get(str(target_part_id))
    tool = by_id.get(str(tool_part_id))
    if target is None or tool is None or target is tool:
        raise AuthoringContractError(
            "target_part_id and tool_part_id must identify different active Parts"
        )
    normalized_name = str(result_name).strip()
    if not normalized_name:
        raise AuthoringContractError("result_name must not be blank")
    if any(part.name == normalized_name for part in context.parts):
        raise AuthoringContractError(
            f"result Part name already exists: {normalized_name!r}"
        )
    proposal_summary = str(summary).strip()
    if not proposal_summary:
        raise AuthoringContractError("Part Boolean proposal summary is blank")
    recipe = prepared.recipe
    part_context = prepared.context
    if (
        not part_context.proven
        or part_context.target_part_id != target.part_id
        or part_context.tool_part_id != tool.part_id
        or recipe.operation != operation
        or recipe.name != normalized_name
    ):
        raise AuthoringContractError(
            "detached Part Boolean proof does not match the requested operands"
        )
    preview = prepared.preview
    parameters = {
        "target_part_id": target.part_id,
        "tool_part_id": tool.part_id,
        "operation": operation,
        "result_name": normalized_name,
        "tool_handling": tool_handling,
        "result_part_id": part_context.result_part_id,
        "feature_id": part_context.feature_id,
        "recipe_json": json.dumps(
            geometry_recipe_to_payload(recipe),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=context.binding.document_id,
        target_session_id=context.binding.session_id,
        base_session_revision=context.binding.session_revision,
        draft_revision=draft_revision,
        operations=(
            ModelOperation(OperationKind.APPLY_PART_BOOLEAN, parameters),
        ),
        preconditions={
            "source_kind": "native",
            "target_part_id": target.part_id,
            "tool_part_id": tool.part_id,
            "result_part_id": part_context.result_part_id,
            "feature_id": part_context.feature_id,
            "lineage_proven": True,
            "result_volume_count": 1,
        },
        expected_changes={
            "part_count_delta": 1,
            "suppressed_source_part_ids": [target.part_id, tool.part_id],
            "result_part_id": part_context.result_part_id,
            "projection_refresh_count": 1,
        },
        invalidation_impact={
            "mesh": True,
            "definitions": True,
            "results": True,
        },
        display_summary={
            "title": f"精确{('合并' if operation == 'fuse' else '切除')}部件",
            "summary": proposal_summary,
            "target_model": context.model_name,
            "operation": operation,
            "target_part_id": target.part_id,
            "target_part_name": target.name,
            "tool_part_id": tool.part_id,
            "tool_part_name": tool.name,
            "result_part_id": part_context.result_part_id,
            "result_part_name": normalized_name,
            "feature_id": part_context.feature_id,
            "target_handling": "suppress_source_part_and_create_result",
            "tool_handling": tool_handling,
            "result_body_count": 1,
            "lineage_entity_count": len(part_context.result_entities),
            "lineage_mapping_count": len(part_context.topology_mappings),
            "preview": {
                "kind": "detached_exact_boolean",
                "point_count": len(preview.points),
                "face_count": len(preview.faces),
                "edge_count": len(preview.edges),
            },
            "invalidated_objects": ["mesh", "definitions", "results"],
            "base_session_revision": context.binding.session_revision,
        },
    )


def create_body_boolean_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    part_id: str,
    target_body_id: str,
    tool_body_id: str,
    operation: str,
    result_name: str,
    tool_handling: str,
    prepared: StrictBodyBooleanResult,
    summary: str,
) -> AgentProposal:
    """Bind one exact same-Part MultiBody proof to the current revision."""

    if type(context) is not AuthoringContext:
        raise TypeError("context must be AuthoringContext")
    if type(prepared) is not StrictBodyBooleanResult:
        raise TypeError("prepared must be StrictBodyBooleanResult")
    if operation not in {"fuse", "cut"}:
        raise AuthoringContractError(
            "Agent exact Boolean supports fuse and cut only; "
            "intersect/fragment remain disabled until stable multi-result "
            "Body IDs and lineage replay are proven"
        )
    if tool_handling != BODY_BOOLEAN_TOOL_HANDLING:
        raise AuthoringContractError(
            f"tool_handling must be {BODY_BOOLEAN_TOOL_HANDLING!r}"
        )
    if (
        not context.binding.supported
        or context.binding.source_kind != "native"
    ):
        raise AuthoringContractError(
            "exact Body Boolean requires an editable native project"
        )
    target_part = next(
        (
            item
            for item in context.parts
            if item.part_id == str(part_id) and not item.suppressed
        ),
        None,
    )
    if target_part is None:
        raise AuthoringContractError(
            "part_id must identify one active canonical MultiBody Part"
        )
    normalized_name = str(result_name).strip()
    if not normalized_name:
        raise AuthoringContractError("result_name must not be blank")
    proposal_summary = str(summary).strip()
    if not proposal_summary:
        raise AuthoringContractError("Body Boolean proposal summary is blank")
    recipe = prepared.recipe
    body_context = recipe.body_context
    if (
        body_context is None
        or not body_context.proven
        or body_context.target_body_id != str(target_body_id)
        or body_context.tool_body_id != str(tool_body_id)
        or recipe.operation != operation
        or recipe.name != normalized_name
    ):
        raise AuthoringContractError(
            "detached Body Boolean proof does not match requested operands"
        )
    preview = prepared.preview
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=context.binding.document_id,
        target_session_id=context.binding.session_id,
        base_session_revision=context.binding.session_revision,
        draft_revision=draft_revision,
        operations=(
            ModelOperation(
                OperationKind.APPLY_BODY_BOOLEAN,
                {
                    "part_id": target_part.part_id,
                    "target_body_id": str(target_body_id),
                    "tool_body_id": str(tool_body_id),
                    "operation": operation,
                    "result_name": normalized_name,
                    "tool_handling": tool_handling,
                    "recipe_json": json.dumps(
                        geometry_recipe_to_payload(prepared.geometry),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ),
        ),
        preconditions={
            "source_kind": "native",
            "part_id": target_part.part_id,
            "target_body_id": str(target_body_id),
            "tool_body_id": str(tool_body_id),
            "lineage_proven": True,
            "result_volume_count": 1,
        },
        expected_changes={
            "part_count_delta": 0,
            "preserved_target_body_id": str(target_body_id),
            "consumed_tool_body_id": str(tool_body_id),
            "projection_refresh_count": 1,
        },
        invalidation_impact={
            "mesh": True,
            "definitions": True,
            "results": True,
        },
        display_summary={
            "title": f"精确{('合并' if operation == 'fuse' else '切除')}实体",
            "summary": proposal_summary,
            "target_model": context.model_name,
            "operation": operation,
            "part_id": target_part.part_id,
            "part_name": target_part.name,
            "target_body_id": str(target_body_id),
            "tool_body_id": str(tool_body_id),
            "result_name": normalized_name,
            "target_handling": "preserve_target_body_id",
            "tool_handling": tool_handling,
            "result_body_count": 1,
            "lineage_entity_count": len(body_context.result_entities),
            "lineage_mapping_count": len(body_context.topology_mappings),
            "preview": {
                "kind": "detached_exact_boolean",
                "point_count": len(preview.points),
                "face_count": len(preview.faces),
                "edge_count": len(preview.edges),
            },
            "invalidated_objects": ["mesh", "definitions", "results"],
            "base_session_revision": context.binding.session_revision,
        },
    )


__all__ = [
    "BODY_BOOLEAN_TOOL_HANDLING",
    "PART_BOOLEAN_TOOL_HANDLING",
    "create_body_boolean_proposal",
    "create_part_boolean_proposal",
]
