"""Detached state controller for strict Boolean authoring between Parts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fem.application.native_part import NativePart
from fem.geometry import LogicalEntityRef


BooleanOperation = Literal["fuse", "cut"]
BooleanSelectionSlot = Literal["target", "tool"]


@dataclass(slots=True)
class PartBooleanController:
    """Own target/tool roles without mutating the committed Session."""

    parts: tuple[NativePart, ...]
    base_session_revision: int
    operation: BooleanOperation
    target_part_id: str | None = None
    tool_part_id: str | None = None
    pending_slot: BooleanSelectionSlot | None = None

    def __post_init__(self) -> None:
        self.parts = tuple(self.parts)
        if type(self.base_session_revision) is not int:
            raise TypeError("base_session_revision must be an integer")
        self.set_operation(self.operation)
        candidates = tuple(
            part
            for part in self.parts
            if not part.suppressed and part.dimension == 3
        )
        if len(candidates) < 2:
            raise ValueError("实体布尔需要至少两个未抑制三维部件")
        if self.target_part_id is not None:
            self._require_operand(self.target_part_id)

    @property
    def ready(self) -> bool:
        return (
            self.target_part_id is not None
            and self.tool_part_id is not None
            and self.target_part_id != self.tool_part_id
        )

    def set_operation(self, operation: str) -> None:
        if operation not in {"fuse", "cut"}:
            raise ValueError("实体布尔操作必须是合并或切除")
        self.operation = operation

    def request_selection(self, slot: BooleanSelectionSlot) -> None:
        if slot not in {"target", "tool"}:
            raise ValueError("选择槽必须是目标部件或工具部件")
        self.pending_slot = slot

    def assign_reference(
        self,
        reference: LogicalEntityRef,
    ) -> BooleanSelectionSlot:
        if self.pending_slot is None:
            raise ValueError("当前没有待选择的布尔操作对象")
        if type(reference) is not LogicalEntityRef:
            raise TypeError("实体布尔选择需要逻辑引用")
        if reference.kind != "part":
            raise ValueError("请选择一个稳定部件")
        part_id = reference.logical_id.split(":", 1)[1]
        self._require_operand(part_id)
        slot = self.pending_slot
        other = (
            self.tool_part_id if slot == "target" else self.target_part_id
        )
        if part_id == other:
            raise ValueError("目标部件和工具部件必须不同")
        if slot == "target":
            self.target_part_id = part_id
        else:
            self.tool_part_id = part_id
        self.pending_slot = None
        return slot

    def part(self, part_id: str) -> NativePart:
        return self._require_operand(part_id)

    def part_label(self, part_id: str | None) -> str:
        if part_id is None:
            return "未选择"
        part = self._require_operand(part_id)
        return f"{part.name} [{part.id}]"

    def _require_operand(self, part_id: str) -> NativePart:
        for part in self.parts:
            if part.id != part_id:
                continue
            if part.suppressed or part.dimension != 3:
                break
            return part
        raise ValueError(f"{part_id} 不是可用的三维部件")


__all__ = [
    "BooleanOperation",
    "BooleanSelectionSlot",
    "PartBooleanController",
]
