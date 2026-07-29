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
)
from .schemas import SCHEMA_VERSION

__all__ = [
    "AUTHORING_SCHEMA_VERSION",
    "AgentDraft",
    "AgentProposal",
    "AuthoringContext",
    "FakeAuthoringPort",
    "LocalModelBinding",
    "ModelPatch",
    "RequirementLedger",
    "RequirementReview",
    "SCHEMA_VERSION",
]

__version__ = "0.1.0"
