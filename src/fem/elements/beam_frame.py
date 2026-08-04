"""Canonical Beam2 orientation values and effective local-frame resolution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import hypot, isfinite
from numbers import Real
from typing import Any, Literal

import numpy as np


BEAM_LOCAL_Y_REFERENCE_KEY = "beam_local_y_reference"
BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY = "beam_default_local_y_reference"
BEAM_DEFAULT_LOCAL_Y_REFERENCE = (0.0, 0.0, -1.0)
BEAM_ORIENTATION_PARALLEL_TOLERANCE = 1e-8
_AUTOMATIC_FALLBACK_TOLERANCE = 1e-12


class BeamOrientationError(ValueError):
    """Base error for a Beam2 orientation contract violation."""

    code = "beam.orientation.invalid"

    def __init__(
        self,
        message: str,
        *,
        element_id: int | None = None,
        reference: object | None = None,
        tangent: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__(message)
        self.element_id = element_id
        self.reference = reference
        self.tangent = tangent


class BeamOrientationInvalidError(BeamOrientationError):
    """A reference value is malformed, non-finite, or zero."""

    code = "beam.orientation.invalid"


class BeamOrientationParallelError(BeamOrientationError):
    """A valid reference value is parallel to one Beam2 tangent."""

    code = "beam.orientation.parallel"


class BeamOrientationUnsupportedTargetError(BeamOrientationError):
    """A Beam orientation was attached to a non-Beam target."""

    code = "beam.orientation.unsupported_target"


@dataclass(frozen=True, slots=True)
class BeamOrientation:
    """An owned approximate local-y reference expressed in global coordinates."""

    local_y_reference: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_y_reference",
            _owned_reference(self.local_y_reference),
        )


@dataclass(frozen=True, slots=True, eq=False)
class BeamFrame:
    """One immutable effective Beam2 frame."""

    length: float
    rotation: np.ndarray
    source: Literal["explicit", "default", "automatic"]
    orientation: BeamOrientation | None

    def __post_init__(self) -> None:
        length = float(self.length)
        if not isfinite(length) or length <= 0.0:
            raise ValueError(
                f"Beam2 frame length must be finite and > 0, got {self.length!r}"
            )

        rotation = np.asarray(self.rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(
                "Beam2 frame rotation must have shape (3, 3), "
                f"got {rotation.shape}"
            )
        if not np.all(np.isfinite(rotation)):
            raise ValueError("Beam2 frame rotation must contain only finite values")
        if not np.allclose(
            rotation @ rotation.T,
            np.eye(3),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Beam2 frame rotation must be orthonormal")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Beam2 frame rotation must be right-handed with determinant 1"
            )

        if self.source == "explicit":
            if not isinstance(self.orientation, BeamOrientation):
                raise ValueError(
                    "an explicit Beam2 frame requires a BeamOrientation"
                )
        elif self.source in {"default", "automatic"}:
            if self.orientation is not None:
                raise ValueError(
                    f"a {self.source} Beam2 frame cannot carry a BeamOrientation"
                )
        else:
            raise ValueError(
                "Beam2 frame source must be 'explicit', 'default', or 'automatic', "
                f"got {self.source!r}"
            )

        # A bytes-backed view cannot be made writable again by a caller.
        immutable = np.frombuffer(
            np.array(rotation, dtype=float, copy=True, order="C").tobytes(),
            dtype=float,
        ).reshape(3, 3)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "rotation", immutable)

    @property
    def local_x(self) -> np.ndarray:
        """Return the global components of the effective local x axis."""

        return self.rotation[0]

    @property
    def local_y(self) -> np.ndarray:
        """Return the global components of the effective local y axis."""

        return self.rotation[1]

    @property
    def local_z(self) -> np.ndarray:
        """Return the global components of the effective local z axis."""

        return self.rotation[2]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BeamFrame):
            return NotImplemented
        return (
            self.length == other.length
            and self.source == other.source
            and self.orientation == other.orientation
            and np.array_equal(self.rotation, other.rotation)
        )

    def __copy__(self) -> BeamFrame:
        """Return this immutable value unchanged."""

        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> BeamFrame:
        """Preserve bytes-backed immutability across owned DTO copies."""

        memo[id(self)] = self
        return self


def parse_beam_orientation(value: object) -> BeamOrientation:
    """Return an immutable, validated Beam2 orientation value."""

    if isinstance(value, BeamOrientation):
        return value
    return BeamOrientation(value)  # type: ignore[arg-type]


def resolve_beam_frame(
    mesh: Any,
    elem: Any,
    node_lookup: dict[int, Any] | None = None,
    *,
    properties: Mapping[str, Any] | None = None,
) -> BeamFrame:
    """Resolve the only effective global-to-local Beam2 frame."""

    # The local import keeps the general line-geometry helper in its existing
    # module without creating a beam_frame <-> line import cycle.
    from .line import line3d_geometry

    length, local_x = line3d_geometry(mesh, elem, node_lookup)
    local_x = np.asarray(local_x, dtype=float)
    if (
        not isfinite(float(length))
        or local_x.shape != (3,)
        or not np.all(np.isfinite(local_x))
    ):
        raise ValueError(
            f"Beam2 element {_element_id(elem)!r} has non-finite geometry"
        )

    source_properties: object = (
        getattr(elem, "props", {}) if properties is None else properties
    )
    if not isinstance(source_properties, Mapping):
        raise TypeError("Beam2 frame properties must be a mapping")
    if BEAM_LOCAL_Y_REFERENCE_KEY not in source_properties:
        missing = object()
        raw_default: object = source_properties.get(
            BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY,
            missing,
        )
        if raw_default is not missing:
            try:
                default_orientation = parse_beam_orientation(raw_default)
            except BeamOrientationInvalidError as error:
                raise BeamOrientationInvalidError(
                    str(error),
                    element_id=_element_id(elem),
                    reference=error.reference,
                ) from error
            return _reference_frame(
                length,
                local_x,
                default_orientation,
                elem,
                source="default",
            )
        return _automatic_frame(length, local_x)

    raw_reference = source_properties[BEAM_LOCAL_Y_REFERENCE_KEY]
    try:
        orientation = parse_beam_orientation(raw_reference)
    except BeamOrientationInvalidError as error:
        raise BeamOrientationInvalidError(
            str(error),
            element_id=_element_id(elem),
            reference=error.reference,
        ) from error
    return _reference_frame(
        length,
        local_x,
        orientation,
        elem,
        source="explicit",
    )


def _automatic_frame(length: float, local_x: np.ndarray) -> BeamFrame:
    """Preserve the Phase 3 automatic local-z reference algorithm exactly."""

    reference = np.array([0.0, 0.0, 1.0])
    projected = reference - float(reference @ local_x) * local_x
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= _AUTOMATIC_FALLBACK_TOLERANCE:
        reference = np.array([0.0, 1.0, 0.0])
        projected = reference - float(reference @ local_x) * local_x
        projected_norm = float(np.linalg.norm(projected))
    local_z = projected / projected_norm
    local_y = np.cross(local_z, local_x)
    local_y /= np.linalg.norm(local_y)
    return BeamFrame(
        length,
        np.vstack([local_x, local_y, local_z]),
        "automatic",
        None,
    )


def _reference_frame(
    length: float,
    local_x: np.ndarray,
    orientation: BeamOrientation,
    elem: Any,
    *,
    source: Literal["explicit", "default"],
) -> BeamFrame:
    reference = np.asarray(
        _normalized_reference(orientation.local_y_reference),
        dtype=float,
    )
    projected = reference - float(reference @ local_x) * local_x
    projected_norm = hypot(*(float(value) for value in projected))
    if projected_norm <= BEAM_ORIENTATION_PARALLEL_TOLERANCE:
        tangent = tuple(float(value) for value in local_x)
        raise BeamOrientationParallelError(
            (
                f"Beam2 element {_element_id(elem)!r} local-y reference is "
                "parallel or nearly parallel to the element axis"
            ),
            element_id=_element_id(elem),
            reference=orientation.local_y_reference,
            tangent=tangent,
        )

    local_y = projected / projected_norm
    local_z = np.cross(local_x, local_y)
    local_z /= np.linalg.norm(local_z)
    local_y = np.cross(local_z, local_x)
    local_y /= np.linalg.norm(local_y)
    return BeamFrame(
        length,
        np.vstack([local_x, local_y, local_z]),
        source,
        orientation if source == "explicit" else None,
    )


def _owned_reference(value: object) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value,
        Iterable,
    ):
        raise BeamOrientationInvalidError(
            "Beam2 local-y reference must contain exactly three finite numbers",
            reference=value,
        )
    components = tuple(value)
    if len(components) != 3:
        raise BeamOrientationInvalidError(
            "Beam2 local-y reference must contain exactly three finite numbers",
            reference=value,
        )

    owned: list[float] = []
    for component in components:
        if isinstance(component, (bool, np.bool_)) or not isinstance(
            component,
            Real,
        ):
            raise BeamOrientationInvalidError(
                "Beam2 local-y reference must contain exactly three finite numbers",
                reference=value,
            )
        try:
            converted = float(component)
        except (OverflowError, TypeError, ValueError) as error:
            raise BeamOrientationInvalidError(
                "Beam2 local-y reference must contain representable numbers",
                reference=value,
            ) from error
        if not isfinite(converted):
            raise BeamOrientationInvalidError(
                "Beam2 local-y reference components must be finite",
                reference=value,
            )
        owned.append(converted)
    result = (owned[0], owned[1], owned[2])
    if max(abs(component) for component in result) == 0.0:
        raise BeamOrientationInvalidError(
            "Beam2 local-y reference must be non-zero",
            reference=result,
        )
    return result


def _normalized_reference(
    reference: tuple[float, float, float],
) -> tuple[float, float, float]:
    scale = max(abs(component) for component in reference)
    scaled = tuple(component / scale for component in reference)
    norm = hypot(*scaled)
    return tuple(component / norm for component in scaled)


def _element_id(elem: Any) -> int | None:
    try:
        return int(elem.id)
    except (AttributeError, TypeError, ValueError):
        return None


__all__ = [
    "BEAM_DEFAULT_LOCAL_Y_REFERENCE",
    "BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY",
    "BEAM_LOCAL_Y_REFERENCE_KEY",
    "BEAM_ORIENTATION_PARALLEL_TOLERANCE",
    "BeamFrame",
    "BeamOrientation",
    "BeamOrientationError",
    "BeamOrientationInvalidError",
    "BeamOrientationParallelError",
    "BeamOrientationUnsupportedTargetError",
    "parse_beam_orientation",
    "resolve_beam_frame",
]
