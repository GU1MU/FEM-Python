from __future__ import annotations

from ..inp import (
    InpBuildError,
    InpInputError,
    InpParseError,
    InpSourceLocation,
    UnsupportedInpFeatureError,
)

# Keep the low-level names available during the migration.  The canonical
# classes and source value now belong to the public ``fem.io.inp`` facade.
AbaqusSourceLocation = InpSourceLocation
AbaqusInputError = InpInputError
AbaqusParseError = InpParseError
AbaqusBuildError = InpBuildError
UnsupportedAbaqusFeatureError = UnsupportedInpFeatureError


__all__ = [
    "AbaqusBuildError",
    "AbaqusInputError",
    "AbaqusParseError",
    "AbaqusSourceLocation",
    "UnsupportedAbaqusFeatureError",
]
