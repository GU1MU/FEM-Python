"""Bounded, document-owned state for committed native geometry previews."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from fem.application import ModelSession

from .geometry_preview import GeometryPreview

if TYPE_CHECKING:
    from .task_controller import BackgroundTaskController
    from .workspace import WorkspaceDocument


class NativePreviewState(str, Enum):
    COARSE = "coarse"
    PENDING = "pending"
    EXACT = "exact"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class NativePartPreviewKey:
    session_id: str
    part_id: str
    part_revision: int
    recipe: object


@dataclass(slots=True, eq=False)
class NativeGeometryReplayRequest:
    document: WorkspaceDocument
    session: ModelSession
    controller: BackgroundTaskController
    key: NativePartPreviewKey
    scheduled: bool = False


@dataclass(slots=True)
class NativePartPreview:
    key: NativePartPreviewKey
    preview: GeometryPreview
    state: NativePreviewState
    request: NativeGeometryReplayRequest | None = None


@dataclass(slots=True)
class NativeGeometryCache:
    session: ModelSession | None = None
    closing: bool = False
    parts: dict[str, NativePartPreview] = field(default_factory=dict)
    ghost_key: tuple[object, ...] | None = None
    ghost_preview: GeometryPreview | None = None

    def clear(self) -> None:
        self.session = None
        self.closing = False
        self.parts.clear()
        self.ghost_key = None
        self.ghost_preview = None
