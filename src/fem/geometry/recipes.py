"""Headless geometry recipes for native model authoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real
import re
from typing import Literal

from .references import LogicalEntityRef, logical_ref_sort_key


def _normalize_wire_name(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_wire_coordinate(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite real number")
    return normalized


_SKETCH_GEOMETRY_TOLERANCE = 1.0e-9


def _normalize_sketch_id(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_sketch_scalar(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite real number")
    return normalized


def _normalize_sketch_vector(
    value: object,
    field_name: str,
    *,
    length: int,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a {length}-component sequence")
    try:
        components = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(
            f"{field_name} must be a {length}-component sequence"
        ) from error
    if len(components) != length:
        raise ValueError(
            f"{field_name} must contain exactly {length} components"
        )
    return tuple(
        _normalize_sketch_scalar(component, f"{field_name}[{index}]")
        for index, component in enumerate(components)
    )


@dataclass(frozen=True, slots=True)
class RectangleGeometry:
    """One rectangular two-dimensional geometry definition."""

    name: str
    width: float
    height: float

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        if not normalized_name:
            raise ValueError("几何名称不能为空")
        if float(self.width) <= 0.0:
            raise ValueError("矩形宽度必须大于零")
        if float(self.height) <= 0.0:
            raise ValueError("矩形高度必须大于零")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", float(self.height))


@dataclass(frozen=True, slots=True)
class DiskGeometry:
    """One circular two-dimensional geometry definition."""

    name: str
    radius: float

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        if not normalized_name:
            raise ValueError("几何名称不能为空")
        if float(self.radius) <= 0.0:
            raise ValueError("圆盘半径必须大于零")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "radius", float(self.radius))


@dataclass(frozen=True, slots=True)
class BoxGeometry:
    """One axis-aligned three-dimensional box definition."""

    name: str
    width: float
    depth: float
    height: float

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        dimensions = tuple(float(value) for value in (self.width, self.depth, self.height))
        if not normalized_name:
            raise ValueError("几何名称不能为空")
        if any(value <= 0.0 for value in dimensions):
            raise ValueError("长方体尺寸必须大于零")
        object.__setattr__(self, "name", normalized_name)
        for field_name, value in zip(("width", "depth", "height"), dimensions):
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class CylinderGeometry:
    """One cylinder aligned with the positive Z axis."""

    name: str
    radius: float
    height: float

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        dimensions = float(self.radius), float(self.height)
        if not normalized_name:
            raise ValueError("几何名称不能为空")
        if any(value <= 0.0 for value in dimensions):
            raise ValueError("圆柱半径和高度必须大于零")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "radius", dimensions[0])
        object.__setattr__(self, "height", dimensions[1])


@dataclass(frozen=True, slots=True)
class WirePoint:
    """One named point in a spatial straight-member wire recipe."""

    name: str
    x: float
    y: float
    z: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_wire_name(self.name, "point name"))
        for field_name in ("x", "y", "z"):
            object.__setattr__(
                self,
                field_name,
                _normalize_wire_coordinate(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, slots=True)
class WireMember:
    """One named straight member between two named wire points."""

    name: str
    start: str
    end: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_wire_name(self.name, "member name"))
        object.__setattr__(
            self,
            "start",
            _normalize_wire_name(self.start, "member start"),
        )
        object.__setattr__(
            self,
            "end",
            _normalize_wire_name(self.end, "member end"),
        )


@dataclass(frozen=True, slots=True)
class WireGeometry:
    """A validated graph of named spatial points and straight members."""

    name: str
    points: tuple[WirePoint, ...]
    members: tuple[WireMember, ...]

    def __post_init__(self) -> None:
        normalized_name = _normalize_wire_name(self.name, "wire name")
        points = tuple(self.points)
        members = tuple(self.members)
        if len(points) < 2:
            raise ValueError("a wire requires at least two points")
        if not members:
            raise ValueError("a wire requires at least one member")
        if any(type(point) is not WirePoint for point in points):
            raise TypeError("wire points must contain only WirePoint values")
        if any(type(member) is not WireMember for member in members):
            raise TypeError("wire members must contain only WireMember values")

        point_names: dict[str, WirePoint] = {}
        point_name_keys: set[str] = set()
        for point in points:
            folded = point.name.casefold()
            if folded in point_name_keys:
                raise ValueError(f"duplicate wire point name: {point.name!r}")
            point_name_keys.add(folded)
            point_names[point.name] = point

        member_names: set[str] = set()
        endpoint_pairs: set[frozenset[str]] = set()
        used_points: set[str] = set()
        for member in members:
            folded = member.name.casefold()
            if folded in member_names:
                raise ValueError(f"duplicate wire member name: {member.name!r}")
            member_names.add(folded)
            if member.start not in point_names or member.end not in point_names:
                raise ValueError(
                    f"wire member {member.name!r} references an unknown point"
                )
            if member.start == member.end:
                raise ValueError(
                    f"wire member {member.name!r} cannot use the same point twice"
                )
            start = point_names[member.start]
            end = point_names[member.end]
            if (start.x, start.y, start.z) == (end.x, end.y, end.z):
                raise ValueError(f"wire member {member.name!r} has zero length")
            pair = frozenset((member.start, member.end))
            if pair in endpoint_pairs:
                raise ValueError(
                    f"wire members cannot duplicate endpoint pair: "
                    f"{member.start!r}, {member.end!r}"
                )
            endpoint_pairs.add(pair)
            used_points.update((member.start, member.end))

        unused = set(point_names) - used_points
        if unused:
            raise ValueError(
                "every wire point must participate in a member: "
                + ", ".join(sorted(unused))
            )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "members", members)


@dataclass(frozen=True, slots=True)
class PlateWithHoleGeometry:
    """A rectangular plate with one circular through-hole."""

    name: str
    width: float
    height: float
    hole_x: float
    hole_y: float
    hole_radius: float

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        values = tuple(
            float(value)
            for value in (
                self.width,
                self.height,
                self.hole_x,
                self.hole_y,
                self.hole_radius,
            )
        )
        width, height, hole_x, hole_y, radius = values
        if not normalized_name:
            raise ValueError("几何名称不能为空")
        if width <= 0.0 or height <= 0.0 or radius <= 0.0:
            raise ValueError("板尺寸和孔半径必须大于零")
        clearance = min(hole_x, width - hole_x, hole_y, height - hole_y)
        if clearance <= radius:
            raise ValueError("圆孔必须完整位于矩形板内部")
        object.__setattr__(self, "name", normalized_name)
        for field_name, value in zip(
            ("width", "height", "hole_x", "hole_y", "hole_radius"),
            values,
        ):
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class SketchRectangle:
    """One rectangular contour in a planar sketch."""

    operation: Literal["material", "cut"]
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.operation not in {"material", "cut"}:
            raise ValueError("草图轮廓只能用于添加材料或切除材料")
        values = tuple(float(value) for value in (self.x, self.y, self.width, self.height))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("草图轮廓参数必须是有限数值")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError("矩形宽度和高度必须大于零")
        for field_name, value in zip(("x", "y", "width", "height"), values):
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True, init=False)
class SketchCircle:
    """A circle curve with a compatibility constructor for legacy contours.

    The curve form is ``SketchCircle(id, center_point_id, radius)``.  The
    frozen contour form ``SketchCircle(operation, x, y, radius)`` remains
    readable so existing schema-v1/v2 projects and callers continue to work.
    ``operation``/``x``/``y`` are populated only for the legacy form.
    """

    id: str | None
    center_point_id: str | None
    radius: float
    operation: Literal["material", "cut"] | None
    x: float | None
    y: float | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        if args and kwargs:
            raise TypeError("SketchCircle accepts either positional or keyword arguments")
        if len(args) == 4:
            operation, x, y, radius = args
            normalized_operation = operation
            if normalized_operation not in {"material", "cut"}:
                raise ValueError("草图轮廓只能用于添加材料或切除材料")
            normalized_x = _normalize_sketch_scalar(x, "x")
            normalized_y = _normalize_sketch_scalar(y, "y")
            normalized_radius = _normalize_sketch_scalar(radius, "radius")
            if normalized_radius <= 0.0:
                raise ValueError("圆半径必须大于零")
            object.__setattr__(self, "id", None)
            object.__setattr__(self, "center_point_id", None)
            object.__setattr__(self, "radius", normalized_radius)
            object.__setattr__(self, "operation", normalized_operation)
            object.__setattr__(self, "x", normalized_x)
            object.__setattr__(self, "y", normalized_y)
            return
        if len(args) == 3:
            curve_id, center_point_id, radius = args
            normalized_id = _normalize_sketch_id(curve_id, "circle id")
            normalized_center = _normalize_sketch_id(
                center_point_id,
                "circle center_point_id",
            )
            normalized_radius = _normalize_sketch_scalar(radius, "radius")
            if normalized_radius <= 0.0:
                raise ValueError("圆半径必须大于零")
            object.__setattr__(self, "id", normalized_id)
            object.__setattr__(self, "center_point_id", normalized_center)
            object.__setattr__(self, "radius", normalized_radius)
            object.__setattr__(self, "operation", None)
            object.__setattr__(self, "x", None)
            object.__setattr__(self, "y", None)
            return
        if kwargs:
            legacy_keys = {"operation", "x", "y", "radius"}
            strict_keys = {"id", "center_point_id", "radius"}
            keys = set(kwargs)
            if keys == legacy_keys:
                self.__init__(
                    kwargs["operation"],
                    kwargs["x"],
                    kwargs["y"],
                    kwargs["radius"],
                )
                return
            if keys == strict_keys:
                self.__init__(
                    kwargs["id"],
                    kwargs["center_point_id"],
                    kwargs["radius"],
                )
                return
        raise TypeError(
            "SketchCircle requires (operation, x, y, radius) for a legacy "
            "contour or (id, center_point_id, radius) for a curve"
        )

    @property
    def is_legacy(self) -> bool:
        """Return whether this value is the compatibility contour form."""

        return self.operation is not None

    @property
    def is_curve(self) -> bool:
        """Return whether this value is the strict curve form."""

        return self.id is not None


@dataclass(frozen=True, slots=True)
class SketchPlane:
    """An immutable two-dimensional authoring frame in global coordinates."""

    origin: tuple[float, float, float]
    x_direction: tuple[float, float, float]
    y_direction: tuple[float, float, float]

    def __post_init__(self) -> None:
        origin = _normalize_sketch_vector(self.origin, "origin", length=3)
        x_direction = _normalize_sketch_vector(
            self.x_direction,
            "x_direction",
            length=3,
        )
        y_direction = _normalize_sketch_vector(
            self.y_direction,
            "y_direction",
            length=3,
        )
        x_norm = math.sqrt(sum(value * value for value in x_direction))
        y_norm = math.sqrt(sum(value * value for value in y_direction))
        if x_norm <= _SKETCH_GEOMETRY_TOLERANCE:
            raise ValueError("sketch plane x_direction must be non-zero")
        if y_norm <= _SKETCH_GEOMETRY_TOLERANCE:
            raise ValueError("sketch plane y_direction must be non-zero")
        dot = sum(
            left * right for left, right in zip(x_direction, y_direction, strict=True)
        )
        if not math.isclose(dot, 0.0, abs_tol=_SKETCH_GEOMETRY_TOLERANCE):
            raise ValueError("sketch plane directions must be orthogonal")
        x_unit = tuple(value / x_norm for value in x_direction)
        y_unit = tuple(value / y_norm for value in y_direction)
        normal = (
            x_unit[1] * y_unit[2] - x_unit[2] * y_unit[1],
            x_unit[2] * y_unit[0] - x_unit[0] * y_unit[2],
            x_unit[0] * y_unit[1] - x_unit[1] * y_unit[0],
        )
        normal_norm = math.sqrt(sum(value * value for value in normal))
        if normal_norm <= _SKETCH_GEOMETRY_TOLERANCE:
            raise ValueError("sketch plane directions must define a normal")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "x_direction", x_unit)
        object.__setattr__(self, "y_direction", y_unit)

    @classmethod
    def xy(cls) -> "SketchPlane":
        """Return the Phase 1 global XY authoring plane."""

        return cls((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    @property
    def normal(self) -> tuple[float, float, float]:
        """Return the normalized right-handed frame normal."""

        x = self.x_direction
        y = self.y_direction
        return (
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        )

    def to_global(self, u: float, v: float) -> tuple[float, float, float]:
        """Map one plane-local coordinate pair into global coordinates."""

        u_value = _normalize_sketch_scalar(u, "u")
        v_value = _normalize_sketch_scalar(v, "v")
        return tuple(
            origin
            + u_value * x_direction
            + v_value * y_direction
            for origin, x_direction, y_direction in zip(
                self.origin,
                self.x_direction,
                self.y_direction,
                strict=True,
            )
        )

    def to_local(self, point: tuple[float, float, float]) -> tuple[float, float]:
        """Project one global point into this orthonormal frame."""

        coordinates = _normalize_sketch_vector(point, "point", length=3)
        delta = tuple(
            value - origin for value, origin in zip(coordinates, self.origin, strict=True)
        )
        return (
            sum(value * axis for value, axis in zip(delta, self.x_direction, strict=True)),
            sum(value * axis for value, axis in zip(delta, self.y_direction, strict=True)),
        )


@dataclass(frozen=True, slots=True)
class SketchPoint:
    """A stable plane-local sketch point."""

    id: str
    u: float
    v: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_sketch_id(self.id, "point id"))
        object.__setattr__(self, "u", _normalize_sketch_scalar(self.u, "u"))
        object.__setattr__(self, "v", _normalize_sketch_scalar(self.v, "v"))

    @property
    def x(self) -> float:
        """Compatibility alias for the local horizontal coordinate."""

        return self.u

    @property
    def y(self) -> float:
        """Compatibility alias for the local vertical coordinate."""

        return self.v


@dataclass(frozen=True, slots=True)
class SketchLine:
    """A straight sketch curve between two stable point IDs."""

    id: str
    start_point_id: str
    end_point_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_sketch_id(self.id, "line id"))
        start = _normalize_sketch_id(self.start_point_id, "line start_point_id")
        end = _normalize_sketch_id(self.end_point_id, "line end_point_id")
        if start == end:
            raise ValueError("line endpoints must be different points")
        object.__setattr__(self, "start_point_id", start)
        object.__setattr__(self, "end_point_id", end)


@dataclass(frozen=True, slots=True)
class SketchArc:
    """A circular arc with an explicit center point and orientation."""

    id: str
    start_point_id: str
    center_point_id: str
    end_point_id: str
    orientation: Literal["ccw", "cw"] = "ccw"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_sketch_id(self.id, "arc id"))
        start = _normalize_sketch_id(self.start_point_id, "arc start_point_id")
        center = _normalize_sketch_id(self.center_point_id, "arc center_point_id")
        end = _normalize_sketch_id(self.end_point_id, "arc end_point_id")
        if len({start, center, end}) != 3:
            raise ValueError("arc start, center, and end points must differ")
        orientation = str(self.orientation).strip().lower()
        aliases = {
            "counterclockwise": "ccw",
            "counter-clockwise": "ccw",
            "clockwise": "cw",
        }
        orientation = aliases.get(orientation, orientation)
        if orientation not in {"ccw", "cw"}:
            raise ValueError("arc orientation must be 'ccw' or 'cw'")
        object.__setattr__(self, "start_point_id", start)
        object.__setattr__(self, "center_point_id", center)
        object.__setattr__(self, "end_point_id", end)
        object.__setattr__(self, "orientation", orientation)


SketchCurve = SketchLine | SketchArc | SketchCircle
STRICT_SKETCH_CURVE_TYPES = (SketchLine, SketchArc, SketchCircle)


SketchContour = SketchRectangle | SketchCircle
SKETCH_CONTOUR_TYPES = (SketchRectangle, SketchCircle)


@dataclass(frozen=True, slots=True, init=False)
class SketchGeometry:
    """A strict curve graph with a compatibility contour representation.

    Legacy callers use ``SketchGeometry(name, contours)``.  New authoring code
    uses ``SketchGeometry(name, plane, points, curves)`` or the equivalent
    keyword form.  The two representations never share mutable containers and
    the strict representation deliberately stores no material/cut guess.
    """

    name: str
    contours: tuple[SketchContour, ...]
    plane: SketchPlane | None
    points: tuple[SketchPoint, ...]
    curves: tuple[SketchCurve, ...]

    # Legacy contour sketches and their strict curve-graph migration describe
    # the same authoring intent.  Keep equality semantic across that boundary
    # so existing detached-project comparisons remain meaningful after a v3
    # save/reopen cycle.  SketchGeometry values are domain values rather than
    # dictionary keys, so the custom equality deliberately disables hashing.
    __hash__ = None

    def __init__(
        self,
        name: object,
        contours: object = (),
        plane: object | None = None,
        points: object = (),
        curves: object = (),
    ) -> None:
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("草图名称不能为空")

        # The positional strict form is (name, plane, points, curves).  The
        # keyword form naturally lands in the named ``plane``/``points``/
        # ``curves`` parameters and is handled by the same normalization path.
        strict_plane: object | None = plane
        strict_points: object = points
        strict_curves: object = curves
        legacy_contours: object = contours
        if isinstance(contours, SketchPlane):
            if plane is None:
                strict_points = ()
            else:
                strict_points = plane
            strict_curves = points
            strict_plane = contours
            legacy_contours = ()

        is_strict = isinstance(strict_plane, SketchPlane) or strict_plane is not None
        if is_strict:
            if not isinstance(strict_plane, SketchPlane):
                raise TypeError("strict sketch plane must be a SketchPlane")
            normalized_points = tuple(strict_points)
            normalized_curves = tuple(strict_curves)
            self._validate_strict(
                strict_plane,
                normalized_points,
                normalized_curves,
            )
            object.__setattr__(self, "name", normalized_name)
            object.__setattr__(self, "contours", ())
            object.__setattr__(self, "plane", strict_plane)
            object.__setattr__(self, "points", normalized_points)
            object.__setattr__(self, "curves", normalized_curves)
            return

        normalized_contours = tuple(legacy_contours)
        if not normalized_contours or not all(
            isinstance(item, SKETCH_CONTOUR_TYPES)
            and getattr(item, "is_legacy", True)
            for item in normalized_contours
        ):
            raise ValueError("草图至少需要一个有效轮廓")
        if not any(item.operation == "material" for item in normalized_contours):
            raise ValueError("草图至少需要一个添加材料轮廓")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "contours", normalized_contours)
        object.__setattr__(self, "plane", None)
        object.__setattr__(self, "points", ())
        object.__setattr__(self, "curves", ())

    def __eq__(self, other: object) -> bool:
        if type(other) is not SketchGeometry:
            return NotImplemented
        if self.is_strict == other.is_strict:
            return (
                self.name,
                self.contours,
                self.plane,
                self.points,
                self.curves,
            ) == (
                other.name,
                other.contours,
                other.plane,
                other.points,
                other.curves,
            )
        # Import lazily to avoid the recipes <-> analysis module cycle during
        # normal package initialization.
        from .recipe_analysis import legacy_sketch_to_strict

        left = self if self.is_strict else legacy_sketch_to_strict(self)
        right = other if other.is_strict else legacy_sketch_to_strict(other)
        return left == right

    @staticmethod
    def _validate_strict(
        plane: SketchPlane,
        points: tuple[object, ...],
        curves: tuple[object, ...],
    ) -> None:
        del plane  # The constructor has already validated the immutable frame.
        if not points:
            raise ValueError("严格草图至少需要一个点")
        if not curves:
            raise ValueError("严格草图至少需要一条曲线")
        if any(type(point) is not SketchPoint for point in points):
            raise TypeError("strict sketch points must contain only SketchPoint values")
        if any(type(curve) not in STRICT_SKETCH_CURVE_TYPES for curve in curves):
            raise TypeError(
                "strict sketch curves must contain only SketchLine, SketchArc, "
                "or strict SketchCircle values"
            )
        entity_ids: set[str] = set()
        point_map: dict[str, SketchPoint] = {}
        for point in points:
            folded = point.id.casefold()
            if folded in entity_ids:
                raise ValueError(f"duplicate sketch entity id: {point.id!r}")
            entity_ids.add(folded)
            point_map[point.id] = point
        for curve in curves:
            if isinstance(curve, SketchCircle) and not curve.is_curve:
                raise TypeError("strict sketches cannot contain legacy circle contours")
            folded = curve.id.casefold()
            if folded in entity_ids:
                raise ValueError(f"duplicate sketch entity id: {curve.id!r}")
            entity_ids.add(folded)
            references = _curve_point_ids(curve)
            if any(reference not in point_map for reference in references):
                missing = next(
                    reference for reference in references if reference not in point_map
                )
                raise ValueError(
                    f"sketch curve {curve.id!r} references unknown point {missing!r}"
                )
            if isinstance(curve, SketchLine):
                start = point_map[curve.start_point_id]
                end = point_map[curve.end_point_id]
                if _sketch_distance(start, end) <= _SKETCH_GEOMETRY_TOLERANCE:
                    raise ValueError(f"sketch line {curve.id!r} has zero length")
            elif isinstance(curve, SketchArc):
                start = point_map[curve.start_point_id]
                center = point_map[curve.center_point_id]
                end = point_map[curve.end_point_id]
                start_radius = _sketch_distance(start, center)
                end_radius = _sketch_distance(end, center)
                if min(start_radius, end_radius) <= _SKETCH_GEOMETRY_TOLERANCE:
                    raise ValueError(f"sketch arc {curve.id!r} has an invalid radius")
                if not math.isclose(
                    start_radius,
                    end_radius,
                    rel_tol=1.0e-8,
                    abs_tol=_SKETCH_GEOMETRY_TOLERANCE,
                ):
                    raise ValueError(
                        f"sketch arc {curve.id!r} start and end radii differ"
                    )
                if _sketch_distance(start, end) <= _SKETCH_GEOMETRY_TOLERANCE:
                    raise ValueError(f"sketch arc {curve.id!r} has identical endpoints")

    @property
    def is_strict(self) -> bool:
        """Return whether this value uses the curve-first representation."""

        return self.plane is not None

    @property
    def is_legacy(self) -> bool:
        """Return whether this value uses the contour compatibility form."""

        return self.plane is None

    def point(self, point_id: str) -> SketchPoint:
        """Return one strict sketch point by stable ID."""

        for point in self.points:
            if point.id == point_id:
                return point
        raise KeyError(point_id)

    def curve(self, curve_id: str) -> SketchCurve:
        """Return one strict sketch curve by stable ID."""

        for curve in self.curves:
            if curve.id == curve_id:
                return curve
        raise KeyError(curve_id)


def _curve_point_ids(curve: SketchCurve) -> tuple[str, ...]:
    if isinstance(curve, SketchLine):
        return curve.start_point_id, curve.end_point_id
    if isinstance(curve, SketchArc):
        return curve.start_point_id, curve.center_point_id, curve.end_point_id
    if isinstance(curve, SketchCircle):
        return (curve.center_point_id,) if curve.center_point_id is not None else ()
    raise TypeError(f"Unsupported sketch curve: {type(curve).__name__}")


def _sketch_distance(left: SketchPoint, right: SketchPoint) -> float:
    return math.hypot(left.u - right.u, left.v - right.v)


PrimitiveGeometry = (
    RectangleGeometry
    | DiskGeometry
    | PlateWithHoleGeometry
    | BoxGeometry
    | CylinderGeometry
)
PRIMITIVE_GEOMETRY_TYPES = (
    RectangleGeometry,
    DiskGeometry,
    PlateWithHoleGeometry,
    BoxGeometry,
    CylinderGeometry,
)
BASE_GEOMETRY_TYPES = (*PRIMITIVE_GEOMETRY_TYPES, SketchGeometry, WireGeometry)


_BODY_ID_PATTERN = re.compile(r"B([1-9][0-9]*)\Z")


def _normalize_body_id(value: object, field_name: str = "body id") -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if value != value.strip() or _BODY_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must use B1, B2, B3, ...")
    return value


def _normalize_body_name(value: object, field_name: str = "body name") -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class BooleanLineageEntity:
    """One persisted logical entity proven by a strict solid Boolean."""

    kind: Literal["point", "edge", "face", "body"]
    logical_id: str
    semantic_role: str
    topology_links: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reference = LogicalEntityRef(self.logical_id)
        if reference.kind != self.kind:
            raise ValueError("Boolean lineage kind must match logical_id")
        if type(self.semantic_role) is not str or not self.semantic_role.strip():
            raise ValueError("Boolean lineage semantic_role must not be empty")
        links = tuple(self.topology_links)
        for link in links:
            LogicalEntityRef(link)
        object.__setattr__(self, "topology_links", tuple(sorted(set(links))))


@dataclass(frozen=True, slots=True)
class BooleanLineageMapping:
    """Persisted source-to-result mapping for a strict solid Boolean."""

    source: Literal["target", "tool"]
    source_logical_id: str
    target_logical_id: str
    relation: Literal["preserved", "derived"]

    def __post_init__(self) -> None:
        if self.source not in {"target", "tool"}:
            raise ValueError("Boolean lineage source must be target or tool")
        if self.relation not in {"preserved", "derived"}:
            raise ValueError(
                "Boolean lineage relation must be preserved or derived"
            )
        source = LogicalEntityRef(self.source_logical_id)
        target = LogicalEntityRef(self.target_logical_id)
        if self.relation == "preserved" and (
            self.source_logical_id != self.target_logical_id
        ):
            raise ValueError("preserved Boolean lineage must keep its logical ID")
        dimensions = {"point": 0, "edge": 1, "face": 2, "body": 3}
        descent = dimensions[source.kind] - dimensions[target.kind]
        if descent not in {0, 1}:
            raise ValueError(
                "Boolean lineage may preserve dimension or derive one "
                "lower-dimensional intersection"
            )
        if descent == 1 and "/intersection/" not in target.logical_id:
            raise ValueError(
                "cross-dimensional Boolean lineage is only valid for "
                "generated intersections"
            )


@dataclass(frozen=True, slots=True)
class BooleanBodyContext:
    """Stable target/tool intent and optional proven Boolean lineage."""

    feature_id: str
    target_body_id: str
    tool_body_id: str
    tool_body_name: str
    result_entities: tuple[BooleanLineageEntity, ...] = ()
    topology_mappings: tuple[BooleanLineageMapping, ...] = ()

    def __post_init__(self) -> None:
        if type(self.feature_id) is not str:
            raise TypeError("feature_id must be a string")
        feature_id = self.feature_id.strip()
        if not feature_id or not re.fullmatch(r"BF[1-9][0-9]*", feature_id):
            raise ValueError("feature_id must use BF1, BF2, BF3, ...")
        target_id = _normalize_body_id(
            self.target_body_id,
            "target_body_id",
        )
        tool_id = _normalize_body_id(self.tool_body_id, "tool_body_id")
        if target_id == tool_id:
            raise ValueError("target_body_id and tool_body_id must differ")
        entities = tuple(self.result_entities)
        mappings = tuple(self.topology_mappings)
        if any(type(item) is not BooleanLineageEntity for item in entities):
            raise TypeError(
                "result_entities must contain BooleanLineageEntity values"
            )
        if any(type(item) is not BooleanLineageMapping for item in mappings):
            raise TypeError(
                "topology_mappings must contain BooleanLineageMapping values"
            )
        logical_ids = tuple(item.logical_id for item in entities)
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("Boolean lineage contains duplicate logical IDs")
        entity_ids = set(logical_ids)
        if any(item.target_logical_id not in entity_ids for item in mappings):
            raise ValueError("Boolean lineage mapping targets an unknown entity")
        if bool(entities) != bool(mappings):
            raise ValueError(
                "Boolean lineage entities and mappings must be present together"
            )
        if entities:
            body_ids = tuple(
                item.logical_id for item in entities if item.kind == "body"
            )
            if body_ids != ("body:domain",):
                raise ValueError(
                    "proven Boolean lineage must contain exactly body:domain"
                )
            if not any(item.kind == "face" for item in entities):
                raise ValueError(
                    "proven Boolean lineage must contain boundary faces"
                )
            mapped_ids = {item.target_logical_id for item in mappings}
            if mapped_ids != entity_ids:
                raise ValueError(
                    "proven Boolean lineage must map every result entity"
                )
            if {item.source for item in mappings} != {"target", "tool"}:
                raise ValueError(
                    "proven Boolean lineage must contain target and tool "
                    "source evidence"
                )
            if any(
                link not in entity_ids
                for item in entities
                for link in item.topology_links
            ):
                raise ValueError(
                    "Boolean lineage topology link targets an unknown entity"
                )
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "target_body_id", target_id)
        object.__setattr__(self, "tool_body_id", tool_id)
        object.__setattr__(
            self,
            "tool_body_name",
            _normalize_body_name(self.tool_body_name, "tool_body_name"),
        )
        object.__setattr__(
            self,
            "result_entities",
            tuple(sorted(entities, key=lambda item: logical_ref_sort_key(
                LogicalEntityRef(item.logical_id)
            ))),
        )
        object.__setattr__(
            self,
            "topology_mappings",
            tuple(
                sorted(
                    mappings,
                    key=lambda item: (
                        item.source,
                        logical_ref_sort_key(
                            LogicalEntityRef(item.source_logical_id)
                        ),
                        logical_ref_sort_key(
                            LogicalEntityRef(item.target_logical_id)
                        ),
                        item.relation,
                    ),
                )
            ),
        )

    @property
    def proven(self) -> bool:
        """Return whether the context carries complete persisted proof."""

        return bool(self.result_entities)


@dataclass(frozen=True, slots=True)
class PlanarBooleanContext:
    """Stable target/tool intent and optional proven planar lineage."""

    feature_id: str
    target_face_id: str
    tool_face_ids: tuple[str, ...]
    result_entities: tuple[BooleanLineageEntity, ...] = ()
    topology_mappings: tuple[BooleanLineageMapping, ...] = ()

    def __post_init__(self) -> None:
        if type(self.feature_id) is not str:
            raise TypeError("feature_id must be a string")
        feature_id = self.feature_id.strip()
        if re.fullmatch(r"PB[1-9][0-9]*", feature_id) is None:
            raise ValueError("feature_id must use PB1, PB2, PB3, ...")
        target = LogicalEntityRef(self.target_face_id)
        if target.kind != "face":
            raise ValueError("target_face_id must be a face logical ID")
        if isinstance(self.tool_face_ids, (str, bytes, bytearray)):
            raise TypeError("tool_face_ids must be a tuple of face logical IDs")
        tool_face_ids = tuple(self.tool_face_ids)
        if not tool_face_ids:
            raise ValueError("tool_face_ids must contain at least one Profile")
        if len(tool_face_ids) != len(set(tool_face_ids)):
            raise ValueError("tool_face_ids must not contain duplicates")
        if any(LogicalEntityRef(item).kind != "face" for item in tool_face_ids):
            raise ValueError("tool_face_ids must contain only face logical IDs")
        entities = tuple(self.result_entities)
        mappings = tuple(self.topology_mappings)
        if any(type(item) is not BooleanLineageEntity for item in entities):
            raise TypeError(
                "result_entities must contain BooleanLineageEntity values"
            )
        if any(type(item) is not BooleanLineageMapping for item in mappings):
            raise TypeError(
                "topology_mappings must contain BooleanLineageMapping values"
            )
        logical_ids = tuple(item.logical_id for item in entities)
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("planar Boolean lineage contains duplicate logical IDs")
        entity_ids = set(logical_ids)
        if bool(entities) != bool(mappings):
            raise ValueError(
                "planar Boolean entities and mappings must be present together"
            )
        if entities:
            body_ids = tuple(
                item.logical_id for item in entities if item.kind == "body"
            )
            if body_ids != ("body:domain",):
                raise ValueError(
                    "proven planar Boolean lineage must contain body:domain"
                )
            if not any(item.kind == "face" for item in entities):
                raise ValueError(
                    "proven planar Boolean lineage must contain material Faces"
                )
            if any(item.target_logical_id not in entity_ids for item in mappings):
                raise ValueError(
                    "planar Boolean lineage mapping targets an unknown entity"
                )
            if {item.target_logical_id for item in mappings} != entity_ids:
                raise ValueError(
                    "proven planar Boolean lineage must map every result entity"
                )
            if {item.source for item in mappings} != {"target", "tool"}:
                raise ValueError(
                    "proven planar Boolean lineage requires target and tool evidence"
                )
            if any(
                link not in entity_ids
                for item in entities
                for link in item.topology_links
            ):
                raise ValueError(
                    "planar Boolean topology link targets an unknown entity"
                )
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(
            self,
            "tool_face_ids",
            tuple(
                sorted(
                    tool_face_ids,
                    key=lambda item: logical_ref_sort_key(LogicalEntityRef(item)),
                )
            ),
        )
        object.__setattr__(
            self,
            "result_entities",
            tuple(
                sorted(
                    entities,
                    key=lambda item: logical_ref_sort_key(
                        LogicalEntityRef(item.logical_id)
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "topology_mappings",
            tuple(
                sorted(
                    mappings,
                    key=lambda item: (
                        item.source,
                        logical_ref_sort_key(
                            LogicalEntityRef(item.source_logical_id)
                        ),
                        logical_ref_sort_key(
                            LogicalEntityRef(item.target_logical_id)
                        ),
                        item.relation,
                    ),
                )
            ),
        )

    @property
    def proven(self) -> bool:
        """Return whether exact OCC lineage is persisted."""

        return bool(self.result_entities)


@dataclass(frozen=True, slots=True)
class MovedGeometry:
    """A geometry feature translated in global coordinates."""

    base: object
    dx: float
    dy: float
    dz: float = 0.0

    @property
    def name(self) -> str:
        return self.base.name

    def __post_init__(self) -> None:
        if not isinstance(
            self.base,
            (
                *BASE_GEOMETRY_TYPES,
                MovedGeometry,
                RotatedGeometry,
                ExtrudedGeometry,
                BooleanGeometry,
            ),
        ):
            raise TypeError("移动操作需要已有几何")
        values = tuple(float(value) for value in (self.dx, self.dy, self.dz))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("移动距离必须是有限数值")
        if geometry_dimension(self.base) == 2 and values[2] != 0.0:
            raise ValueError("二维几何只能在 XY 平面内移动")
        for field_name, value in zip(("dx", "dy", "dz"), values):
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class RotatedGeometry:
    """A geometry feature rotated about one global axis through the origin."""

    base: object
    axis: Literal["x", "y", "z"]
    angle_degrees: float

    @property
    def name(self) -> str:
        return self.base.name

    def __post_init__(self) -> None:
        if not isinstance(
            self.base,
            (
                *BASE_GEOMETRY_TYPES,
                MovedGeometry,
                RotatedGeometry,
                ExtrudedGeometry,
                BooleanGeometry,
            ),
        ):
            raise TypeError("旋转操作需要已有几何")
        normalized_axis = str(self.axis).lower()
        if normalized_axis not in {"x", "y", "z"}:
            raise ValueError("旋转轴只能是 X、Y 或 Z")
        angle = float(self.angle_degrees)
        if not math.isfinite(angle):
            raise ValueError("旋转角度必须是有限数值")
        if geometry_dimension(self.base) == 2 and normalized_axis != "z":
            raise ValueError("二维几何只能绕 Z 轴旋转")
        object.__setattr__(self, "axis", normalized_axis)
        object.__setattr__(self, "angle_degrees", angle)


@dataclass(frozen=True, slots=True)
class ExtrudedGeometry:
    """A planar geometry extruded along the positive Z axis."""

    base: object
    height: float
    source_face_ids: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.base.name

    def __post_init__(self) -> None:
        if not isinstance(
            self.base,
            (
                *BASE_GEOMETRY_TYPES,
                MovedGeometry,
                RotatedGeometry,
                BooleanGeometry,
            ),
        ):
            raise TypeError("拉伸操作需要已有二维几何")
        if geometry_dimension(self.base) != 2:
            raise ValueError("只有二维几何可以拉伸")
        height = float(self.height)
        if height <= 0.0 or not math.isfinite(height):
            raise ValueError("拉伸高度必须大于零")
        if isinstance(self.source_face_ids, (str, bytes, bytearray)):
            raise TypeError("source_face_ids 必须是 face logical ID iterable")
        try:
            requested_ids = tuple(self.source_face_ids)
        except TypeError as error:
            raise TypeError(
                "source_face_ids 必须是 face logical ID iterable"
            ) from error
        references = tuple(LogicalEntityRef(value) for value in requested_ids)
        if any(reference.kind != "face" for reference in references):
            raise ValueError("source_face_ids 只能包含 face logical IDs")
        if len(references) != len(set(reference.logical_id for reference in references)):
            raise ValueError("source_face_ids 不能包含重复 logical IDs")
        normalized_ids: tuple[str, ...] = ()
        if references:
            from .extrusion_selection import resolve_extrusion_source_faces

            normalized_ids = resolve_extrusion_source_faces(
                self.base,
                references,
            ).face_ids
        else:
            normalized_ids = tuple(
                reference.logical_id
                for reference in sorted(references, key=logical_ref_sort_key)
            )
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "source_face_ids", normalized_ids)


@dataclass(frozen=True, slots=True)
class BooleanGeometry:
    """A boolean feature combining one object and one tool geometry."""

    name: str
    operation: Literal["fuse", "cut", "fragment"]
    object_geometry: object
    tool_geometry: object
    body_context: BooleanBodyContext | None = None
    planar_context: PlanarBooleanContext | None = None

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        if not normalized_name:
            raise ValueError("布尔结果名称不能为空")
        if self.operation not in {"fuse", "cut", "fragment"}:
            raise ValueError("布尔操作只能是合并、切除或分割")
        supported = (
            *BASE_GEOMETRY_TYPES,
            MovedGeometry,
            RotatedGeometry,
            ExtrudedGeometry,
            BooleanGeometry,
        )
        if not isinstance(self.object_geometry, supported) or not isinstance(
            self.tool_geometry, supported
        ):
            raise TypeError("布尔操作需要两个已有几何")
        object_dimension = geometry_dimension(self.object_geometry)
        tool_dimension = geometry_dimension(self.tool_geometry)
        if object_dimension == 1:
            raise ValueError("布尔操作不支持一维线框几何")
        if object_dimension != tool_dimension:
            raise ValueError("布尔操作的主体和工具体维度必须一致")
        if self.body_context is not None and self.planar_context is not None:
            raise ValueError("body_context and planar_context are mutually exclusive")
        if self.body_context is not None:
            if type(self.body_context) is not BooleanBodyContext:
                raise TypeError(
                    "body_context must be BooleanBodyContext or None"
                )
            if self.operation not in {"fuse", "cut"}:
                raise ValueError("strict Body Boolean only supports fuse or cut")
            if object_dimension != 3:
                raise ValueError("strict Body Boolean requires 3D operands")
        if self.planar_context is not None:
            if type(self.planar_context) is not PlanarBooleanContext:
                raise TypeError(
                    "planar_context must be PlanarBooleanContext or None"
                )
            if self.operation not in {"fuse", "cut"}:
                raise ValueError("strict planar Boolean only supports fuse or cut")
            if object_dimension != 2:
                raise ValueError("strict planar Boolean requires 2D operands")
            from .planar_boolean_selection import resolve_planar_boolean_faces
            from .recipe_topology import describe_recipe_topology

            selection = resolve_planar_boolean_faces(
                self.object_geometry,
                self.planar_context.target_face_id,
                self.tool_geometry,
                self.planar_context.tool_face_ids,
            )
            object_topology = describe_recipe_topology(self.object_geometry)
            tool_topology = describe_recipe_topology(self.tool_geometry)
            if not object_topology.exact or not tool_topology.exact:
                raise ValueError(
                    "strict planar Boolean operands must have exact topology"
                )
            source_ids = {
                "target": set(object_topology.signature.logical_ids),
                "tool": set(tool_topology.signature.logical_ids),
            }
            if any(
                mapping.source_logical_id not in source_ids[mapping.source]
                for mapping in self.planar_context.topology_mappings
            ):
                raise ValueError(
                    "planar Boolean lineage references an unknown source entity"
                )
            object.__setattr__(
                self,
                "planar_context",
                replace(
                    self.planar_context,
                    target_face_id=selection.target_face_id,
                    tool_face_ids=selection.tool_face_ids,
                ),
            )
        object.__setattr__(self, "name", normalized_name)


@dataclass(frozen=True, slots=True)
class SolidBody:
    """One stable independently managed solid inside a native part."""

    id: str
    name: str
    recipe: object

    def __post_init__(self) -> None:
        body_id = _normalize_body_id(self.id)
        body_name = _normalize_body_name(self.name)
        if not _is_single_solid_recipe(self.recipe):
            raise ValueError(
                "SolidBody.recipe must be an exact single-solid 3D recipe"
            )
        object.__setattr__(self, "id", body_id)
        object.__setattr__(self, "name", body_name)


@dataclass(frozen=True, slots=True)
class MultiBodyGeometry:
    """Top-level ownership container for independent solid Bodies."""

    name: str
    bodies: tuple[SolidBody, ...]
    retired_body_ids: tuple[str, ...] = ()
    retired_boolean_feature_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _normalize_body_name(self.name, "MultiBodyGeometry.name")
        if isinstance(self.bodies, (str, bytes, bytearray)):
            raise TypeError("bodies must be an iterable of SolidBody values")
        try:
            bodies = tuple(self.bodies)
        except TypeError as error:
            raise TypeError(
                "bodies must be an iterable of SolidBody values"
            ) from error
        if not bodies:
            raise ValueError("MultiBodyGeometry requires at least one Body")
        if any(type(body) is not SolidBody for body in bodies):
            raise TypeError("bodies must contain only SolidBody values")
        ids = tuple(body.id for body in bodies)
        names = tuple(body.name for body in bodies)
        if len(ids) != len(set(ids)):
            raise ValueError("Body IDs must be unique")
        if len(names) != len(set(names)):
            raise ValueError("Body names must be unique")
        retired_body_ids = tuple(
            sorted(
                {
                    _normalize_body_id(value, "retired body id")
                    for value in self.retired_body_ids
                },
                key=lambda value: int(value[1:]),
            )
        )
        if set(ids) & set(retired_body_ids):
            raise ValueError("active and retired Body IDs must be disjoint")
        retired_feature_ids: list[str] = []
        for value in self.retired_boolean_feature_ids:
            if (
                type(value) is not str
                or value != value.strip()
                or re.fullmatch(r"BF[1-9][0-9]*", value) is None
            ):
                raise ValueError(
                    "retired Boolean feature IDs must use BF1, BF2, BF3, ..."
                )
            retired_feature_ids.append(value)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "bodies",
            tuple(sorted(bodies, key=lambda body: int(body.id[1:]))),
        )
        object.__setattr__(
            self,
            "retired_body_ids",
            retired_body_ids,
        )
        object.__setattr__(
            self,
            "retired_boolean_feature_ids",
            tuple(
                sorted(
                    set(retired_feature_ids),
                    key=lambda value: int(value[2:]),
                )
            ),
        )

    def body(self, body_id: str) -> SolidBody:
        """Return one Body by stable ID."""

        normalized = _normalize_body_id(body_id)
        for body in self.bodies:
            if body.id == normalized:
                return body
        raise KeyError(normalized)


def _is_single_solid_recipe(recipe: object) -> bool:
    if isinstance(recipe, (BoxGeometry, CylinderGeometry)):
        return True
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return _is_single_solid_recipe(recipe.base)
    if isinstance(recipe, ExtrudedGeometry):
        from .extrusion_selection import resolve_extrusion_source_faces

        return len(
            resolve_extrusion_source_faces(
                recipe.base,
                recipe.source_face_ids,
            ).face_ids
        ) == 1
    if isinstance(recipe, BooleanGeometry):
        if recipe.body_context is not None:
            return recipe.body_context.proven
        return (
            recipe.operation in {"fuse", "cut"}
            and _is_single_solid_recipe(recipe.object_geometry)
            and _is_single_solid_recipe(recipe.tool_geometry)
        )
    return False


NativeGeometry = (
    PrimitiveGeometry
    | SketchGeometry
    | WireGeometry
    | MovedGeometry
    | RotatedGeometry
    | ExtrudedGeometry
    | BooleanGeometry
    | MultiBodyGeometry
)
NATIVE_GEOMETRY_TYPES = (
    *PRIMITIVE_GEOMETRY_TYPES,
    SketchGeometry,
    WireGeometry,
    MovedGeometry,
    RotatedGeometry,
    ExtrudedGeometry,
    BooleanGeometry,
    MultiBodyGeometry,
)


def geometry_dimension(recipe: NativeGeometry) -> Literal[1, 2, 3]:
    """Return the topological dimension of a native geometry recipe."""
    if isinstance(recipe, BooleanGeometry):
        return geometry_dimension(recipe.object_geometry)
    if isinstance(recipe, MultiBodyGeometry):
        return 3
    if isinstance(recipe, ExtrudedGeometry):
        return 3
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return geometry_dimension(recipe.base)
    if isinstance(recipe, WireGeometry):
        return 1
    return 3 if isinstance(recipe, (BoxGeometry, CylinderGeometry)) else 2


__all__ = [
    "BASE_GEOMETRY_TYPES",
    "BooleanBodyContext",
    "BooleanGeometry",
    "BooleanLineageEntity",
    "BooleanLineageMapping",
    "BoxGeometry",
    "CylinderGeometry",
    "DiskGeometry",
    "ExtrudedGeometry",
    "MovedGeometry",
    "MultiBodyGeometry",
    "NATIVE_GEOMETRY_TYPES",
    "NativeGeometry",
    "PRIMITIVE_GEOMETRY_TYPES",
    "PlateWithHoleGeometry",
    "PlanarBooleanContext",
    "PrimitiveGeometry",
    "RectangleGeometry",
    "RotatedGeometry",
    "SKETCH_CONTOUR_TYPES",
    "STRICT_SKETCH_CURVE_TYPES",
    "SketchArc",
    "SketchCircle",
    "SketchContour",
    "SketchCurve",
    "SketchGeometry",
    "SketchLine",
    "SketchPlane",
    "SketchPoint",
    "SketchRectangle",
    "SolidBody",
    "WireGeometry",
    "WireMember",
    "WirePoint",
    "geometry_dimension",
]
