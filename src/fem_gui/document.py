"""Read-only compatibility exports for the headless application session.

``ModelSession`` owns every project mutation.  GUI code that still imports
authoring contracts from this module receives the canonical headless types,
and the historical ``FEMDocument`` name resolves to an immutable
``SessionSnapshot`` rather than a second mutable project store.

Remove this module after all GUI imports target :mod:`fem.application`
directly.
"""

from __future__ import annotations

from fem.application import (
    FeatureRecord,
    ModelSession,
    NamedRegion,
    NativePart,
    ProjectSnapshot,
    RegionAssignment,
    SectionDefinition,
    SessionSnapshot,
)


# Compatibility-only type alias.  A snapshot is detached and frozen; callers
# must invoke ``ModelSession`` commands to change project state.
FEMDocument = SessionSnapshot


__all__ = [
    "FEMDocument",
    "FeatureRecord",
    "ModelSession",
    "NamedRegion",
    "NativePart",
    "ProjectSnapshot",
    "RegionAssignment",
    "SectionDefinition",
    "SessionSnapshot",
]
