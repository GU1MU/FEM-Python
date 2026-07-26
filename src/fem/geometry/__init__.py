"""Public scripted-geometry API."""

from __future__ import annotations

from .errors import (
    EntityOwnershipError,
    GeometryError,
    GeometryStateError,
    StaleEntityError,
)
from .measurements import (
    TargetRadiusResolutionError,
    resolve_legacy_hole_target,
    resolve_target_radius,
)
from .references import EntityKind, LogicalEntityRef, logical_ref_sort_key
from .types import (
    BooleanResult,
    CurveLoopRef,
    EntityRef,
    FeatureResult,
    LoftContinuity,
    LoftParametrization,
    LoftResult,
    OrientedCurveRef,
    SweepFrame,
    WireRef,
)
from ._gmsh.model import GeometryModel, model

__all__ = [
    "BooleanResult",
    "CurveLoopRef",
    "EntityOwnershipError",
    "EntityKind",
    "EntityRef",
    "FeatureResult",
    "GeometryError",
    "GeometryModel",
    "GeometryStateError",
    "LoftContinuity",
    "LoftParametrization",
    "LoftResult",
    "LogicalEntityRef",
    "OrientedCurveRef",
    "StaleEntityError",
    "SweepFrame",
    "TargetRadiusResolutionError",
    "WireRef",
    "logical_ref_sort_key",
    "model",
    "resolve_legacy_hole_target",
    "resolve_target_radius",
]
