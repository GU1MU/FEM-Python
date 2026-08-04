"""Canonical Beam2 orientation values and effective local-frame resolution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, sin
from numbers import Real
from typing import Any, Literal

import numpy as np


BEAM_LOCAL_Y_REFERENCE_KEY = "beam_local_y_reference"
BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY = "beam_element_local_y_reference"
BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY = "beam_default_local_y_reference"
BEAM_FRAME_FIELD_KEY = "beam_frame_field"
# The longer spelling is useful at adapter boundaries and is intentionally an
# alias: the contract is still one generic Beam2 element property.
BEAM_ELEMENT_FRAME_FIELD_KEY = BEAM_FRAME_FIELD_KEY
BEAM_DEFAULT_LOCAL_Y_REFERENCE = (0.0, 0.0, -1.0)
BEAM_ORIENTATION_PARALLEL_TOLERANCE = 1e-8
BEAM_FRAME_COMPARISON_TOLERANCE = 1e-10
BEAM_FRAME_INTEGRATION_ORDER = 8
BEAM_FRAME_INTERPOLATION = "linear_twist"
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


class BeamFrameFieldError(BeamOrientationError):
    """Base error for an invalid verified two-end Beam2 frame field."""

    code = "beam.frame_field.invalid"


class BeamFrameFieldInvalidError(BeamFrameFieldError):
    """A frame field is malformed or does not match element geometry."""

    code = "beam.frame_field.invalid"


class BeamFrameVariationError(BeamFrameFieldError):
    """A constant-frame compatibility query received a varying field."""

    code = "beam.frame_field.variation"


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


@dataclass(frozen=True, slots=True, eq=False)
class BeamEndFrame:
    """One immutable, Abaqus-independent frame at a Beam2 element end.

    ``rotation`` is global-to-local and stores local ``(x, y, z)`` axes in
    rows.  The value deliberately carries no source-format or provenance
    object; adapters project their own records into this contract first.
    """

    rotation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _owned_rotation(self.rotation))

    @classmethod
    def from_axes(
        cls,
        tangent: Iterable[float],
        local_y: Iterable[float],
        local_z: Iterable[float],
    ) -> BeamEndFrame:
        return cls(np.vstack((tangent, local_y, local_z)))

    @property
    def local_x(self) -> np.ndarray:
        """Return global components of the local x axis."""

        return self.rotation[0]

    @property
    def local_y(self) -> np.ndarray:
        """Return global components of the local y axis."""

        return self.rotation[1]

    @property
    def local_z(self) -> np.ndarray:
        """Return global components of the local z axis."""

        return self.rotation[2]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BeamEndFrame):
            return NotImplemented
        return np.array_equal(self.rotation, other.rotation)

    def __copy__(self) -> BeamEndFrame:
        """Return this immutable value unchanged."""

        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> BeamEndFrame:
        """Preserve bytes-backed immutability across owned model copies."""

        memo[id(self)] = self
        return self


@dataclass(frozen=True, slots=True, eq=False)
class BeamFrameField:
    """Verified two-end frame and its single interpolation/integration owner.

    Beam2 geometry is a straight line, so both endpoint tangents must agree.
    The transverse axes are interpolated by the shortest signed rotation about
    that tangent.  The frame is sampled at a fixed eight-point Gauss rule;
    the same samples are used by stiffness, distributed-load, and recovery
    paths.  No frame derivative is introduced, keeping this a linear
    Euler--Bernoulli contract without warping or corotational terms.
    """

    length: float
    start: BeamEndFrame
    end: BeamEndFrame
    interpolation: Literal["linear_twist"] = BEAM_FRAME_INTERPOLATION
    comparison_tolerance: float = BEAM_FRAME_COMPARISON_TOLERANCE

    def __post_init__(self) -> None:
        try:
            length = float(self.length)
        except (TypeError, ValueError, OverflowError) as error:
            raise BeamFrameFieldInvalidError(
                f"Beam2 frame field length must be finite and > 0, got {self.length!r}"
            ) from error
        if not isfinite(length) or length <= 0.0:
            raise BeamFrameFieldInvalidError(
                f"Beam2 frame field length must be finite and > 0, got {self.length!r}"
            )
        if type(self.start) is not BeamEndFrame or type(self.end) is not BeamEndFrame:
            raise BeamFrameFieldInvalidError(
                "Beam2 frame field endpoints must be BeamEndFrame values"
            )
        interpolation = str(self.interpolation)
        if interpolation != BEAM_FRAME_INTERPOLATION:
            raise BeamFrameFieldInvalidError(
                "Beam2 frame field interpolation must be 'linear_twist'"
            )
        try:
            tolerance = float(self.comparison_tolerance)
        except (TypeError, ValueError, OverflowError) as error:
            raise BeamFrameFieldInvalidError(
                "Beam2 frame field comparison tolerance must be finite and positive"
            ) from error
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise BeamFrameFieldInvalidError(
                "Beam2 frame field comparison tolerance must be finite and positive"
            )
        if not np.allclose(
            self.start.local_x,
            self.end.local_x,
            rtol=tolerance,
            atol=tolerance,
        ):
            raise BeamFrameFieldInvalidError(
                "Beam2 frame field endpoint tangents must agree for straight geometry"
            )
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "comparison_tolerance", tolerance)

    @classmethod
    def from_rotations(
        cls,
        length: float,
        start_rotation: Any,
        end_rotation: Any,
        *,
        comparison_tolerance: float = BEAM_FRAME_COMPARISON_TOLERANCE,
    ) -> BeamFrameField:
        """Build a field from two global-to-local rotation matrices."""

        return cls(
            length,
            BeamEndFrame(start_rotation),
            BeamEndFrame(end_rotation),
            comparison_tolerance=comparison_tolerance,
        )

    @classmethod
    def from_axes(
        cls,
        length: float,
        start_tangent: Iterable[float],
        start_local_y: Iterable[float],
        start_local_z: Iterable[float],
        end_tangent: Iterable[float],
        end_local_y: Iterable[float],
        end_local_z: Iterable[float],
        *,
        comparison_tolerance: float = BEAM_FRAME_COMPARISON_TOLERANCE,
    ) -> BeamFrameField:
        """Build a field from its two right-handed endpoint triads."""

        return cls(
            length,
            BeamEndFrame.from_axes(start_tangent, start_local_y, start_local_z),
            BeamEndFrame.from_axes(end_tangent, end_local_y, end_local_z),
            comparison_tolerance=comparison_tolerance,
        )

    @classmethod
    def constant(cls, frame: BeamFrame) -> BeamFrameField:
        """Promote one existing constant Beam2 frame to a two-end field."""

        if type(frame) is not BeamFrame:
            raise TypeError("constant Beam2 frame field requires BeamFrame")
        return cls.from_rotations(frame.length, frame.rotation, frame.rotation)

    @property
    def start_frame(self) -> BeamEndFrame:
        """Compatibility spelling for the first element-end frame."""

        return self.start

    @property
    def end_frame(self) -> BeamEndFrame:
        """Compatibility spelling for the second element-end frame."""

        return self.end

    @property
    def is_constant(self) -> bool:
        """Whether both ends reduce to one numerical frame."""

        return bool(
            np.allclose(
                self.start.rotation,
                self.end.rotation,
                rtol=self.comparison_tolerance,
                atol=self.comparison_tolerance,
            )
        )

    def rotation_at_fraction(self, fraction: float) -> np.ndarray:
        """Return the interpolated global-to-local rotation at ``0 <= fraction <= 1``."""

        fraction = float(fraction)
        if not isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"Beam2 frame interpolation fraction must be in [0, 1], got {fraction!r}"
            )
        if fraction == 0.0:
            return self.start.rotation
        if fraction == 1.0:
            return self.end.rotation
        if self.is_constant:
            return self.start.rotation

        tangent = self.start.local_x
        start_y = self.start.local_y
        start_z = self.start.local_z
        cosine = float(start_y @ self.end.local_y)
        sine = float(tangent @ np.cross(start_y, self.end.local_y))
        angle = atan2(sine, cosine)
        current_angle = fraction * angle
        local_y = cos(current_angle) * start_y + sin(current_angle) * start_z
        local_z = np.cross(tangent, local_y)
        local_z /= np.linalg.norm(local_z)
        return np.vstack((tangent, local_y, local_z))

    def frame_at_fraction(self, fraction: float) -> BeamFrame:
        """Return the effective BeamFrame at one interpolation fraction."""

        return BeamFrame(
            self.length,
            self.rotation_at_fraction(fraction),
            "automatic",
            None,
        )

    def quadrature(self) -> tuple[tuple[float, float], ...]:
        """Return the fixed unit-interval Gauss samples owned by this field."""

        points, weights = np.polynomial.legendre.leggauss(
            BEAM_FRAME_INTEGRATION_ORDER
        )
        return tuple(
            ((float(point) + 1.0) / 2.0, float(weight) / 2.0)
            for point, weight in zip(points, weights, strict=True)
        )

    def integrate(
        self,
        integrand: Callable[[float, BeamFrame], Any],
    ) -> Any:
        """Integrate one frame-dependent quantity over physical length."""

        if not callable(integrand):
            raise TypeError("Beam2 frame-field integrand must be callable")
        result: Any = None
        for fraction, weight in self.quadrature():
            value = np.asarray(
                integrand(fraction, self.frame_at_fraction(fraction)),
                dtype=float,
            )
            contribution = self.length * weight * value
            result = contribution if result is None else result + contribution
        return result

    def as_constant_frame(self) -> BeamFrame:
        """Return a numerical frame for a field known to be constant."""

        if not self.is_constant:
            raise BeamFrameVariationError(
                "a varying Beam2 frame field cannot be represented by one BeamFrame"
            )
        return BeamFrame(self.length, self.start.rotation, "automatic", None)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BeamFrameField):
            return NotImplemented
        return (
            self.length == other.length
            and self.start == other.start
            and self.end == other.end
            and self.interpolation == other.interpolation
            and self.comparison_tolerance == other.comparison_tolerance
        )

    def __copy__(self) -> BeamFrameField:
        """Return this immutable value unchanged."""

        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> BeamFrameField:
        """Preserve field ownership across model/session copies."""

        memo[id(self)] = self
        return self


def parse_beam_orientation(value: object) -> BeamOrientation:
    """Return an immutable, validated Beam2 orientation value."""

    if isinstance(value, BeamOrientation):
        return value
    return BeamOrientation(value)  # type: ignore[arg-type]


def validate_beam_frame_field(
    field: BeamFrameField,
    *,
    length: float,
    tangent: Iterable[float],
    element_id: int | None = None,
) -> BeamFrameField:
    """Validate one field against the straight Beam2 geometry it serves."""

    if type(field) is not BeamFrameField:
        raise BeamFrameFieldInvalidError(
            "Beam2 element frame field must be a BeamFrameField",
            element_id=element_id,
        )
    try:
        geometry_length = float(length)
        geometry_tangent = np.asarray(tuple(tangent), dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise BeamFrameFieldInvalidError(
            "Beam2 element geometry is not finite",
            element_id=element_id,
        ) from error
    if (
        not isfinite(geometry_length)
        or geometry_length <= 0.0
        or geometry_tangent.shape != (3,)
        or not np.all(np.isfinite(geometry_tangent))
    ):
        raise BeamFrameFieldInvalidError(
            "Beam2 element geometry is not finite",
            element_id=element_id,
        )
    tangent_norm = float(np.linalg.norm(geometry_tangent))
    if not isfinite(tangent_norm) or tangent_norm <= 0.0:
        raise BeamFrameFieldInvalidError(
            "Beam2 element tangent must be finite and non-zero",
            element_id=element_id,
        )
    geometry_tangent = geometry_tangent / tangent_norm
    tolerance = field.comparison_tolerance
    if not np.isclose(
        field.length,
        geometry_length,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise BeamFrameFieldInvalidError(
            (
                f"Beam2 element {element_id!r} frame field length "
                f"{field.length!r} does not match geometry {geometry_length!r}"
            ),
            element_id=element_id,
            tangent=tuple(float(value) for value in geometry_tangent),
        )
    if not np.allclose(
        field.start.local_x,
        geometry_tangent,
        rtol=tolerance,
        atol=tolerance,
    ) or not np.allclose(
        field.end.local_x,
        geometry_tangent,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise BeamFrameFieldInvalidError(
            (
                f"Beam2 element {element_id!r} frame field tangent does not "
                "match connectivity"
            ),
            element_id=element_id,
            tangent=tuple(float(value) for value in geometry_tangent),
        )
    return field


def resolve_beam_frame_field(
    mesh: Any,
    elem: Any,
    node_lookup: dict[int, Any] | None = None,
    *,
    properties: Mapping[str, Any] | None = None,
) -> BeamFrameField:
    """Resolve the verified, source-independent two-end Beam2 contract."""

    from .line import line3d_geometry

    length, tangent = line3d_geometry(mesh, elem, node_lookup)
    source_properties = _frame_properties(elem, properties)
    if BEAM_FRAME_FIELD_KEY not in source_properties:
        return BeamFrameField.constant(
            resolve_beam_frame(
                mesh,
                elem,
                node_lookup,
                properties=source_properties,
            )
        )
    raw_field = source_properties[BEAM_FRAME_FIELD_KEY]
    if type(raw_field) is not BeamFrameField:
        raise BeamFrameFieldInvalidError(
            "Beam2 element frame field must be a BeamFrameField",
            element_id=_element_id(elem),
        )
    return validate_beam_frame_field(
        raw_field,
        length=length,
        tangent=tangent,
        element_id=_element_id(elem),
    )


def validate_beam_frame_fields(mesh: Any) -> None:
    """Validate every installed field before a model can enter a session."""

    for elem in getattr(mesh, "elements", ()):
        properties = getattr(elem, "props", {})
        if isinstance(properties, Mapping) and BEAM_FRAME_FIELD_KEY in properties:
            resolve_beam_frame_field(mesh, elem, properties=properties)


def _frame_properties(
    elem: Any,
    properties: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Keep an element-owned core field when callers pass section properties."""

    source: object = (
        getattr(elem, "props", {}) if properties is None else properties
    )
    if not isinstance(source, Mapping):
        raise TypeError("Beam2 frame properties must be a mapping")
    if properties is None:
        return source
    element_properties = getattr(elem, "props", {})
    if not isinstance(element_properties, Mapping):
        return source
    if BEAM_FRAME_FIELD_KEY not in element_properties:
        return source
    merged = dict(source)
    merged[BEAM_FRAME_FIELD_KEY] = element_properties[BEAM_FRAME_FIELD_KEY]
    return merged


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

    source_properties = _frame_properties(elem, properties)
    if BEAM_FRAME_FIELD_KEY in source_properties:
        raw_field = source_properties[BEAM_FRAME_FIELD_KEY]
        if type(raw_field) is not BeamFrameField:
            raise BeamFrameFieldInvalidError(
                "Beam2 element frame field must be a BeamFrameField",
                element_id=_element_id(elem),
            )
        validate_beam_frame_field(
            raw_field,
            length=length,
            tangent=local_x,
            element_id=_element_id(elem),
        )
        if not raw_field.is_constant:
            raise BeamFrameVariationError(
                (
                    f"Beam2 element {_element_id(elem)!r} has a varying frame "
                    "field and cannot be reduced to one BeamFrame"
                ),
                element_id=_element_id(elem),
                tangent=tuple(float(value) for value in local_x),
            )
        return _constant_field_frame(raw_field, source_properties, elem)
    element_reference = source_properties.get(
        BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY,
        None,
    )
    if element_reference is not None:
        raw_reference = element_reference
    elif BEAM_LOCAL_Y_REFERENCE_KEY not in source_properties:
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

    if element_reference is None:
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


def _constant_field_frame(
    field: BeamFrameField,
    properties: Mapping[str, Any],
    elem: Any,
) -> BeamFrame:
    """Attach legacy provenance without recalculating the canonical rotation."""

    element_reference = properties.get(
        BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY,
        None,
    )
    if element_reference is not None:
        raw_reference = element_reference
        source: Literal["explicit", "default", "automatic"] = "explicit"
    elif BEAM_LOCAL_Y_REFERENCE_KEY in properties:
        raw_reference = properties[BEAM_LOCAL_Y_REFERENCE_KEY]
        source = "explicit"
    elif BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY in properties:
        raw_reference = properties[BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY]
        source = "default"
    else:
        return BeamFrame(field.length, field.start.rotation, "automatic", None)

    try:
        orientation = parse_beam_orientation(raw_reference)
    except BeamOrientationInvalidError as error:
        raise BeamOrientationInvalidError(
            str(error),
            element_id=_element_id(elem),
            reference=error.reference,
        ) from error
    return BeamFrame(
        field.length,
        field.start.rotation,
        source,
        orientation if source == "explicit" else None,
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


def _owned_rotation(value: object) -> np.ndarray:
    try:
        rotation = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise BeamFrameFieldInvalidError(
            "Beam2 frame rotation must be a finite 3-by-3 array"
        ) from error
    if rotation.shape != (3, 3):
        raise BeamFrameFieldInvalidError(
            "Beam2 frame rotation must have shape (3, 3), "
            f"got {rotation.shape}"
        )
    if not np.all(np.isfinite(rotation)):
        raise BeamFrameFieldInvalidError(
            "Beam2 frame rotation must contain only finite values"
        )
    if not np.allclose(
        rotation @ rotation.T,
        np.eye(3),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise BeamFrameFieldInvalidError(
            "Beam2 frame rotation must be orthonormal"
        )
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, rtol=1e-10, atol=1e-12):
        raise BeamFrameFieldInvalidError(
            "Beam2 frame rotation must be right-handed with determinant 1"
        )
    return np.frombuffer(
        np.array(rotation, dtype=float, copy=True, order="C").tobytes(),
        dtype=float,
    ).reshape(3, 3)


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
    "BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY",
    "BEAM_ELEMENT_FRAME_FIELD_KEY",
    "BEAM_FRAME_FIELD_KEY",
    "BEAM_FRAME_COMPARISON_TOLERANCE",
    "BEAM_FRAME_INTEGRATION_ORDER",
    "BEAM_FRAME_INTERPOLATION",
    "BEAM_LOCAL_Y_REFERENCE_KEY",
    "BEAM_ORIENTATION_PARALLEL_TOLERANCE",
    "BeamFrame",
    "BeamEndFrame",
    "BeamFrameField",
    "BeamFrameFieldError",
    "BeamFrameFieldInvalidError",
    "BeamFrameVariationError",
    "BeamOrientation",
    "BeamOrientationError",
    "BeamOrientationInvalidError",
    "BeamOrientationParallelError",
    "BeamOrientationUnsupportedTargetError",
    "parse_beam_orientation",
    "resolve_beam_frame",
    "resolve_beam_frame_field",
    "validate_beam_frame_field",
    "validate_beam_frame_fields",
]
