"""Conversions between recipe-local and Part-namespaced Boolean proof."""

from __future__ import annotations

from .part_namespace import namespace_part_logical_id, strip_part_logical_id
from .recipes import (
    BooleanBodyContext,
    BooleanLineageEntity,
    BooleanLineageMapping,
    PartBooleanContext,
)


def namespace_part_boolean_context(
    *,
    feature_id: str,
    target_part_id: str,
    tool_part_id: str,
    result_part_id: str,
    result_entities: tuple[BooleanLineageEntity, ...],
    topology_mappings: tuple[BooleanLineageMapping, ...],
) -> PartBooleanContext:
    """Install recipe-local lineage into target/tool/result Part namespaces."""

    entities = tuple(
        BooleanLineageEntity(
            item.kind,
            namespace_part_logical_id(result_part_id, item.logical_id),
            item.semantic_role,
            tuple(
                namespace_part_logical_id(result_part_id, link)
                for link in item.topology_links
            ),
        )
        for item in result_entities
    )
    source_parts = {"target": target_part_id, "tool": tool_part_id}
    mappings = tuple(
        BooleanLineageMapping(
            item.source,
            namespace_part_logical_id(
                source_parts[item.source],
                item.source_logical_id,
            ),
            namespace_part_logical_id(
                result_part_id,
                item.target_logical_id,
            ),
            item.relation,
        )
        for item in topology_mappings
    )
    return PartBooleanContext(
        feature_id,
        target_part_id,
        tool_part_id,
        result_part_id,
        entities,
        mappings,
    )


def localize_part_boolean_context(
    context: PartBooleanContext,
) -> BooleanBodyContext:
    """Project persisted Part proof into the local strict-Boolean resolver."""

    if type(context) is not PartBooleanContext:
        raise TypeError("context must be a PartBooleanContext")
    entities = tuple(
        BooleanLineageEntity(
            item.kind,
            strip_part_logical_id(
                context.result_part_id,
                item.logical_id,
            ),
            item.semantic_role,
            tuple(
                strip_part_logical_id(context.result_part_id, link)
                for link in item.topology_links
            ),
        )
        for item in context.result_entities
    )
    source_parts = {
        "target": context.target_part_id,
        "tool": context.tool_part_id,
    }
    mappings = tuple(
        BooleanLineageMapping(
            item.source,
            strip_part_logical_id(
                source_parts[item.source],
                item.source_logical_id,
            ),
            strip_part_logical_id(
                context.result_part_id,
                item.target_logical_id,
            ),
            item.relation,
        )
        for item in context.topology_mappings
    )
    return BooleanBodyContext(
        f"BF{context.feature_id[3:]}",
        "B1",
        "B2",
        "工具部件",
        entities,
        mappings,
    )


__all__ = [
    "localize_part_boolean_context",
    "namespace_part_boolean_context",
]
