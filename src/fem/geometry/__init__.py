"""Public scripted-geometry API."""

from __future__ import annotations

from .errors import (
    EntityOwnershipError,
    GeometryError,
    GeometryStateError,
    StaleEntityError,
)
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
    "EntityRef",
    "FeatureResult",
    "GeometryError",
    "GeometryModel",
    "GeometryStateError",
    "LoftContinuity",
    "LoftParametrization",
    "LoftResult",
    "OrientedCurveRef",
    "StaleEntityError",
    "SweepFrame",
    "WireRef",
    "model",
]
