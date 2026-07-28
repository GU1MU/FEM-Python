"""Detached state controller for strict solid Body Boolean authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fem.geometry import LogicalEntityRef, MultiBodyGeometry

BooleanOperation = Literal["fuse", "cut"]
BooleanSelectionSlot = Literal["target", "tool"]


@dataclass(slots=True)
class BodyBooleanController:
    """Own target/tool roles without mutating the committed Session."""

    geometry: MultiBodyGeometry
    base_session_revision: int
    operation: BooleanOperation
    target_body_id: str | None = None
    tool_body_id: str | None = None
    pending_slot: BooleanSelectionSlot | None = None

    def __post_init__(self) -> None:
        if type(self.geometry) is not MultiBodyGeometry:
            raise TypeError("geometry must be MultiBodyGeometry")
        if type(self.base_session_revision) is not int:
            raise TypeError("base_session_revision must be an integer")
        self.set_operation(self.operation)
        if len(self.geometry.bodies) < 2:
            raise ValueError("strict Body Boolean requires at least two Bodies")

    @property
    def ready(self) -> bool:
        return (
            self.target_body_id is not None
            and self.tool_body_id is not None
            and self.target_body_id != self.tool_body_id
        )

    def set_operation(self, operation: str) -> None:
        if operation not in {"fuse", "cut"}:
            raise ValueError("Body Boolean operation must be fuse or cut")
        self.operation = operation

    def request_selection(self, slot: BooleanSelectionSlot) -> None:
        if slot not in {"target", "tool"}:
            raise ValueError("Boolean selection slot must be target or tool")
        self.pending_slot = slot

    def assign_reference(self, reference: LogicalEntityRef) -> BooleanSelectionSlot:
        if self.pending_slot is None:
            raise ValueError("no Boolean target/tool selection is pending")
        if type(reference) is not LogicalEntityRef or reference.kind != "body":
            raise ValueError("Boolean target/tool selection requires a Body")
        body_id = reference.logical_id.split(":", 1)[1]
        if body_id == "domain":
            raise ValueError("aggregate body:domain cannot be a Boolean operand")
        self.geometry.body(body_id)
        slot = self.pending_slot
        other = (
            self.tool_body_id
            if slot == "target"
            else self.target_body_id
        )
        if body_id == other:
            raise ValueError("target and tool must be different Bodies")
        if slot == "target":
            self.target_body_id = body_id
        else:
            self.tool_body_id = body_id
        self.pending_slot = None
        return slot

    def body_label(self, body_id: str | None) -> str:
        if body_id is None:
            return "未选择"
        body = self.geometry.body(body_id)
        return f"{body.name} ({body.id})"


__all__ = [
    "BodyBooleanController",
    "BooleanOperation",
    "BooleanSelectionSlot",
]
