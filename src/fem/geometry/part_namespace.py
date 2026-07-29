"""Stable Part namespace helpers for logical geometry references."""

from __future__ import annotations

from .references import LogicalEntityRef


def namespace_part_logical_id(part_id: str, logical_id: str) -> str:
    """Map one recipe-local logical ID into a stable Part namespace."""

    from fem.application.native_part import normalize_part_id

    normalized_part_id = normalize_part_id(part_id)
    reference = LogicalEntityRef(logical_id)
    if reference.kind == "part":
        if logical_id != f"part:{normalized_part_id}":
            raise ValueError("part logical ID does not match part_id")
        return logical_id
    _kind, semantic_name = logical_id.split(":", 1)
    if reference.kind == "body":
        if semantic_name != "domain":
            raise ValueError("recipe-local body reference must be body:domain")
        semantic_name = "domain"
    return f"{reference.kind}:{normalized_part_id}/{semantic_name}"


def strip_part_logical_id(part_id: str, logical_id: str) -> str:
    """Return the recipe-local form of one reference owned by *part_id*."""

    from fem.application.native_part import normalize_part_id

    normalized_part_id = normalize_part_id(part_id)
    reference = LogicalEntityRef(logical_id)
    if reference.kind == "part":
        if logical_id != f"part:{normalized_part_id}":
            raise ValueError("part logical ID does not match part_id")
        return "part:self"
    prefix = f"{reference.kind}:{normalized_part_id}/"
    if not logical_id.startswith(prefix):
        raise ValueError(
            f"logical reference {logical_id!r} is not owned by "
            f"{normalized_part_id}"
        )
    semantic_name = logical_id[len(prefix):]
    if not semantic_name:
        raise ValueError("Part-namespaced logical reference has no local name")
    if reference.kind == "body":
        if semantic_name != "domain":
            raise ValueError("Part body reference must use /domain")
        return "body:domain"
    return f"{reference.kind}:{semantic_name}"


def part_id_from_logical_id(logical_id: str) -> str | None:
    """Return a canonical owner Part ID, or ``None`` for a local reference."""

    from fem.application.native_part import normalize_part_id

    reference = LogicalEntityRef(logical_id)
    semantic_name = logical_id.split(":", 1)[1]
    if reference.kind == "part":
        return normalize_part_id(semantic_name)
    owner, separator, _local_name = semantic_name.partition("/")
    if separator != "/":
        return None
    try:
        return normalize_part_id(owner)
    except (TypeError, ValueError):
        return None


def part_logical_ref(part_id: str) -> LogicalEntityRef:
    """Create the selection-only reference for one Part."""

    from fem.application.native_part import normalize_part_id

    return LogicalEntityRef(f"part:{normalize_part_id(part_id)}")


def namespace_part_reference(
    part_id: str,
    reference: LogicalEntityRef,
) -> LogicalEntityRef:
    if type(reference) is not LogicalEntityRef:
        raise TypeError("reference must be a LogicalEntityRef")
    return LogicalEntityRef(
        namespace_part_logical_id(part_id, reference.logical_id)
    )


def strip_part_reference(
    part_id: str,
    reference: LogicalEntityRef,
) -> LogicalEntityRef:
    if type(reference) is not LogicalEntityRef:
        raise TypeError("reference must be a LogicalEntityRef")
    return LogicalEntityRef(
        strip_part_logical_id(part_id, reference.logical_id)
    )


__all__ = [
    "namespace_part_logical_id",
    "namespace_part_reference",
    "part_id_from_logical_id",
    "part_logical_ref",
    "strip_part_logical_id",
    "strip_part_reference",
]
