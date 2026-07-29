"""Stable backend-neutral references to native recipe topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast


EntityKind = Literal["point", "edge", "face", "body", "part"]

_ENTITY_KINDS: tuple[EntityKind, ...] = (
    "point",
    "edge",
    "face",
    "body",
    "part",
)
_KIND_ORDER = {kind: index for index, kind in enumerate(_ENTITY_KINDS)}


@dataclass(frozen=True, slots=True)
class LogicalEntityRef:
    """One immutable engineering reference identified only by a logical ID."""

    logical_id: str

    def __post_init__(self) -> None:
        if type(self.logical_id) is not str:
            raise TypeError("logical_id must be a string")
        if self.logical_id != self.logical_id.strip():
            raise ValueError("logical_id must not contain surrounding whitespace")
        kind, separator, semantic_name = self.logical_id.partition(":")
        if (
            separator != ":"
            or kind not in _KIND_ORDER
            or not semantic_name
            or not semantic_name.strip()
        ):
            raise ValueError(
                "logical_id must use '<kind>:<semantic-name>' with kind "
                "point, edge, face, body, or part"
            )

    @property
    def kind(self) -> EntityKind:
        """Return the entity kind encoded by the stable logical ID."""

        return cast(EntityKind, self.logical_id.partition(":")[0])


def logical_ref_sort_key(reference: LogicalEntityRef) -> tuple[int, str]:
    """Return the only canonical ordering key for logical references."""

    if type(reference) is not LogicalEntityRef:
        raise TypeError("reference must be a LogicalEntityRef")
    return _KIND_ORDER[reference.kind], reference.logical_id


__all__ = ["EntityKind", "LogicalEntityRef", "logical_ref_sort_key"]
