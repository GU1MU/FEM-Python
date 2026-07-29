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
from .analysis_authoring import (
    AnalysisAuthoringError,
    ConfirmedDisplacement,
    ConfirmedLoad,
    ConfirmedResultRequest,
    LinearStaticAnalysis,
    create_analysis_definition_change,
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
from .definition_authoring import (
    PlateScopeSet,
    ScopeSelectionError,
    ScopeSelectionEvidence,
    build_eccentric_plate_scopes,
    create_scope_definition_change,
)
from .naming import NameAllocator, NamePolicy
from .mesh_authoring import (
    MESH_INTENT_SCHEMA_VERSION,
    MeshIntent,
    create_mesh_proposal,
)
from .schemas import SCHEMA_VERSION

__all__ = [
    "AUTHORING_SCHEMA_VERSION",
    "AgentDraft",
    "AgentProposal",
    "AnalysisAuthoringError",
    "AuthoringContext",
    "ConfirmedDisplacement",
    "ConfirmedLoad",
    "ConfirmedResultRequest",
    "FakeAuthoringPort",
    "LocalModelBinding",
    "LinearStaticAnalysis",
    "NameAllocator",
    "NamePolicy",
    "MESH_INTENT_SCHEMA_VERSION",
    "MeshIntent",
    "ModelPatch",
    "PlateScopeSet",
    "RequirementLedger",
    "RequirementReview",
    "ScopeSelectionError",
    "ScopeSelectionEvidence",
    "GeometryDraft",
    "StaticGeometryPreview",
    "UnitContextSummary",
    "box_geometry",
    "build_eccentric_plate_scopes",
    "create_scope_definition_change",
    "create_analysis_definition_change",
    "create_mesh_proposal",
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
