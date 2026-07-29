"""Conversational orchestration for the local FEM package."""

from __future__ import annotations

from .authoring import (
    AUTHORING_SCHEMA_VERSION,
    AgentDraft,
    AgentProposal,
    AuthoringContext,
    FakeAuthoringPort,
    LocalModelBinding,
    ModelPatch,
    RequirementLedger,
    RequirementReview,
    UnitContextSummary,
)
from .geometry_authoring import (
    GeometryDraft,
    StaticGeometryPreview,
    box_geometry,
    create_geometry_proposal,
    cylinder_geometry,
    disk_geometry,
    plate_with_hole_geometry,
    rectangle_geometry,
    rotate_geometry,
    translate_geometry,
)
from .naming import NameAllocator, NamePolicy
from .schemas import SCHEMA_VERSION

__all__ = [
    "AUTHORING_SCHEMA_VERSION",
    "AgentDraft",
    "AgentProposal",
    "AuthoringContext",
    "FakeAuthoringPort",
    "LocalModelBinding",
    "NameAllocator",
    "NamePolicy",
    "ModelPatch",
    "RequirementLedger",
    "RequirementReview",
    "GeometryDraft",
    "StaticGeometryPreview",
    "UnitContextSummary",
    "box_geometry",
    "create_geometry_proposal",
    "cylinder_geometry",
    "disk_geometry",
    "plate_with_hole_geometry",
    "rectangle_geometry",
    "rotate_geometry",
    "translate_geometry",
    "SCHEMA_VERSION",
]

__version__ = "0.1.0"
