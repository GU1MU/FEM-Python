"""Headless geometry recipes for native model authoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
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


class FaceSketchBooleanOperation(str, Enum):
    """The two material operations supported by a face-supported sketch."""

    FUSE = "fuse"
    CUT = "cut"

    @property
    def chinese_name(self) -> str:
        return {
            type(self).FUSE: "合并材料",
            type(self).CUT: "切除材料",
        }[self]


class FaceSketchBooleanDirection(str, Enum):
    """Extrusion direction relative to the oriented supporting face."""

    OUTWARD = "outward"
    INWARD = "inward"

    @property
    def chinese_name(self) -> str:
        return {
            type(self).OUTWARD: "沿外法向",
            type(self).INWARD: "沿内法向",
        }[self]

    def vector(
        self,
        outward_normal: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Return the signed unit vector for this direction."""

        normal = _normalize_sketch_vector(
            outward_normal,
            "outward_normal",
            length=3,
        )
        magnitude = math.sqrt(sum(component * component for component in normal))
        if magnitude <= _SKETCH_GEOMETRY_TOLERANCE:
            raise ValueError("工作面外法向必须为非零向量")
        sign = 1.0 if self is type(self).OUTWARD else -1.0
        return tuple(sign * component / magnitude for component in normal)


@dataclass(frozen=True, slots=True)
class FaceSketchWorkplaneStrategy:
    """Persisted deterministic U-axis construction strategy."""

    seed_axis: Literal["x", "y", "z"]
    sign: Literal[-1, 1] = 1
    origin_rule: Literal["area_center"] = "area_center"

    def __post_init__(self) -> None:
        normalized_axis = str(self.seed_axis).lower()
        if normalized_axis not in {"x", "y", "z"}:
            raise ValueError("工作面 U 轴种子必须是全局 X、Y 或 Z 轴")
        if isinstance(self.sign, bool) or self.sign not in {-1, 1}:
            raise ValueError("工作面 U 轴符号必须是 1 或 -1")
        if self.origin_rule != "area_center":
            raise ValueError("工作面原点规则必须为面积中心")
        object.__setattr__(self, "seed_axis", normalized_axis)


class SketchExternalReferenceType(str, Enum):
    """A point derivation supported by associated sketch snapping."""

    TOPOLOGY_VERTEX = "topology_vertex"
    LINE_MIDPOINT = "line_midpoint"
    CIRCLE_CENTER = "circle_center"
    ARC_CENTER = "arc_center"
    FACE_CENTER = "face_center"


@dataclass(frozen=True, slots=True)
class SketchExternalReference:
    """One stable derived point reference into the supporting topology."""

    id: str
    source: LogicalEntityRef
    derived_type: SketchExternalReferenceType

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_sketch_id(self.id, "外部参考 ID"))
        if type(self.source) is not LogicalEntityRef:
            raise TypeError("外部参考来源必须是 LogicalEntityRef")
        if type(self.derived_type) is not SketchExternalReferenceType:
            raise TypeError("外部参考派生类型无效")
        expected_kind = {
            SketchExternalReferenceType.TOPOLOGY_VERTEX: "point",
            SketchExternalReferenceType.LINE_MIDPOINT: "edge",
            SketchExternalReferenceType.CIRCLE_CENTER: "edge",
            SketchExternalReferenceType.ARC_CENTER: "edge",
            SketchExternalReferenceType.FACE_CENTER: "face",
        }[self.derived_type]
        if self.source.kind != expected_kind:
            raise ValueError("外部参考来源类型与派生类型不匹配")


@dataclass(frozen=True, slots=True)
class SketchExternalCoincidence:
    """Association between one strict-sketch point and an external reference."""

    point_id: str
    reference_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point_id",
            _normalize_sketch_id(self.point_id, "外部重合草图点 ID"),
        )
        object.__setattr__(
            self,
            "reference_id",
            _normalize_sketch_id(self.reference_id, "外部重合参考 ID"),
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


SketchConstraintSource = Literal["manual", "inferred"]


def _normalize_constraint_common(
    value: object,
    source: object,
    enabled: object,
) -> tuple[str, SketchConstraintSource, bool]:
    constraint_id = _normalize_sketch_id(value, "constraint id")
    if source not in {"manual", "inferred"}:
        raise ValueError("constraint source must be 'manual' or 'inferred'")
    if type(enabled) is not bool:
        raise TypeError("constraint enabled must be a bool")
    return constraint_id, source, enabled


@dataclass(frozen=True, slots=True)
class SketchCoincidentConstraint:
    """Require two stable sketch points to occupy the same location."""

    id: str
    first_point_id: str
    second_point_id: str
    source: SketchConstraintSource = "manual"
    enabled: bool = True

    def __post_init__(self) -> None:
        constraint_id, source, enabled = _normalize_constraint_common(
            self.id, self.source, self.enabled
        )
        first = _normalize_sketch_id(self.first_point_id, "first_point_id")
        second = _normalize_sketch_id(self.second_point_id, "second_point_id")
        if first == second:
            raise ValueError("coincident constraint requires two different points")
        object.__setattr__(self, "id", constraint_id)
        object.__setattr__(self, "first_point_id", first)
        object.__setattr__(self, "second_point_id", second)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "enabled", enabled)


@dataclass(frozen=True, slots=True)
class SketchPointOnCurveConstraint:
    """Require a stable sketch point to lie on one stable curve."""

    id: str
    point_id: str
    curve_id: str
    source: SketchConstraintSource = "manual"
    enabled: bool = True

    def __post_init__(self) -> None:
        constraint_id, source, enabled = _normalize_constraint_common(
            self.id, self.source, self.enabled
        )
        object.__setattr__(self, "id", constraint_id)
        object.__setattr__(self, "point_id", _normalize_sketch_id(self.point_id, "point_id"))
        object.__setattr__(self, "curve_id", _normalize_sketch_id(self.curve_id, "curve_id"))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "enabled", enabled)


@dataclass(frozen=True, slots=True)
class SketchHorizontalConstraint:
    id: str
    line_id: str
    source: SketchConstraintSource = "manual"
    enabled: bool = True

    def __post_init__(self) -> None:
        constraint_id, source, enabled = _normalize_constraint_common(
            self.id, self.source, self.enabled
        )
        object.__setattr__(self, "id", constraint_id)
        object.__setattr__(self, "line_id", _normalize_sketch_id(self.line_id, "line_id"))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "enabled", enabled)


@dataclass(frozen=True, slots=True)
class SketchVerticalConstraint:
    id: str
    line_id: str
    source: SketchConstraintSource = "manual"
    enabled: bool = True

    def __post_init__(self) -> None:
        constraint_id, source, enabled = _normalize_constraint_common(
            self.id, self.source, self.enabled
        )
        object.__setattr__(self, "id", constraint_id)
        object.__setattr__(self, "line_id", _normalize_sketch_id(self.line_id, "line_id"))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "enabled", enabled)


@dataclass(frozen=True, slots=True)
class SketchFixedConstraint:
    id: str
    point_id: str
    u: float
    v: float
    source: SketchConstraintSource = "manual"
    enabled: bool = True

    def __post_init__(self) -> None:
        constraint_id, source, enabled = _normalize_constraint_common(
            self.id, self.source, self.enabled
        )
        object.__setattr__(self, "id", constraint_id)
        object.__setattr__(self, "point_id", _normalize_sketch_id(self.point_id, "point_id"))
        object.__setattr__(self, "u", _normalize_sketch_scalar(self.u, "fixed u"))
        object.__setattr__(self, "v", _normalize_sketch_scalar(self.v, "fixed v"))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "enabled", enabled)


@dataclass(frozen=True, slots=True)
class SketchDistanceDimension:
    id: str
    first_point_id: str
    second_point_id: str
    value: float
    driving: bool = True
    source: SketchConstraintSource = "manual"
    enabled: bool = True

    def __post_init__(self) -> None:
        constraint_id, source, enabled = _normalize_constraint_common(
            self.id, self.source, self.enabled
        )
        first = _normalize_sketch_id(self.first_point_id, "first_point_id")
        second = _normalize_sketch_id(self.second_point_id, "second_point_id")
        if first == second:
            raise ValueError("distance dimension requires two different points")
        value = _normalize_sketch_scalar(self.value, "dimension value")
        if value <= 0.0:
            raise ValueError("dimension value must be positive")
        if type(self.driving) is not bool:
            raise TypeError("dimension driving must be a bool")
        object.__setattr__(self, "id", constraint_id)
        object.__setattr__(self, "first_point_id", first)
        object.__setattr__(self, "second_point_id", second)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "enabled", enabled)


@dataclass(frozen=True, slots=True)
class SketchRadiusDimension:
    id: str
    curve_id: str
    value: float
    driving: bool = True
    source: SketchConstraintSource = "manual"
    enabled: bool = True

    def __post_init__(self) -> None:
        constraint_id, source, enabled = _normalize_constraint_common(
            self.id, self.source, self.enabled
        )
        value = _normalize_sketch_scalar(self.value, "dimension value")
        if value <= 0.0:
            raise ValueError("dimension value must be positive")
        if type(self.driving) is not bool:
            raise TypeError("dimension driving must be a bool")
        object.__setattr__(self, "id", constraint_id)
        object.__setattr__(self, "curve_id", _normalize_sketch_id(self.curve_id, "curve_id"))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "enabled", enabled)


SketchGeometricConstraint = (
    SketchCoincidentConstraint
    | SketchPointOnCurveConstraint
    | SketchHorizontalConstraint
    | SketchVerticalConstraint
    | SketchFixedConstraint
)
SketchDimensionalConstraint = SketchDistanceDimension | SketchRadiusDimension
SketchConstraint = SketchGeometricConstraint | SketchDimensionalConstraint
SKETCH_CONSTRAINT_TYPES = (
    SketchCoincidentConstraint,
    SketchPointOnCurveConstraint,
    SketchHorizontalConstraint,
    SketchVerticalConstraint,
    SketchFixedConstraint,
    SketchDistanceDimension,
    SketchRadiusDimension,
)


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
    constraints: tuple[SketchConstraint, ...]

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
        constraints: object = (),
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
        strict_constraints: object = constraints
        legacy_contours: object = contours
        if isinstance(contours, SketchPlane):
            if plane is None:
                strict_points = ()
            else:
                strict_points = plane
            strict_curves = points
            if curves:
                strict_constraints = curves
            strict_plane = contours
            legacy_contours = ()

        is_strict = isinstance(strict_plane, SketchPlane) or strict_plane is not None
        if is_strict:
            if not isinstance(strict_plane, SketchPlane):
                raise TypeError("strict sketch plane must be a SketchPlane")
            normalized_points = tuple(strict_points)
            normalized_curves = tuple(strict_curves)
            normalized_constraints = tuple(strict_constraints)
            self._validate_strict(
                strict_plane,
                normalized_points,
                normalized_curves,
                normalized_constraints,
            )
            object.__setattr__(self, "name", normalized_name)
            object.__setattr__(self, "contours", ())
            object.__setattr__(self, "plane", strict_plane)
            object.__setattr__(self, "points", normalized_points)
            object.__setattr__(self, "curves", normalized_curves)
            object.__setattr__(self, "constraints", normalized_constraints)
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
        object.__setattr__(self, "constraints", ())

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
                self.constraints,
            ) == (
                other.name,
                other.contours,
                other.plane,
                other.points,
                other.curves,
                other.constraints,
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
        constraints: tuple[object, ...],
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
        validate_sketch_constraints(constraints, point_map, curves)

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


def sketch_constraint_entity_ids(constraint: SketchConstraint) -> tuple[str, ...]:
    """Return stable geometry references used by one constraint."""

    if isinstance(constraint, (SketchCoincidentConstraint, SketchDistanceDimension)):
        return constraint.first_point_id, constraint.second_point_id
    if isinstance(constraint, (SketchPointOnCurveConstraint,)):
        return constraint.point_id, constraint.curve_id
    if isinstance(constraint, (SketchHorizontalConstraint, SketchVerticalConstraint)):
        return (constraint.line_id,)
    if isinstance(constraint, SketchFixedConstraint):
        return (constraint.point_id,)
    if isinstance(constraint, SketchRadiusDimension):
        return (constraint.curve_id,)
    raise TypeError("unsupported sketch constraint")


def constraints_without_entities(
    constraints: tuple[SketchConstraint, ...],
    entity_ids: set[str] | frozenset[str],
) -> tuple[SketchConstraint, ...]:
    """Cascade-delete constraints that reference removed geometry entities."""

    return tuple(
        constraint
        for constraint in constraints
        if not entity_ids.intersection(sketch_constraint_entity_ids(constraint))
    )


def copy_sketch_constraints(
    constraints: tuple[SketchConstraint, ...],
    entity_id_map: dict[str, str],
    constraint_id_map: dict[str, str],
) -> tuple[SketchConstraint, ...]:
    """Copy constraints using complete, explicit stable-ID mappings.

    Restore operations retain the original tuple unchanged.  Copy operations
    must provide a new ID for every constraint and every referenced entity;
    incomplete mappings are rejected instead of producing dangling copies.
    """

    copied: list[SketchConstraint] = []
    for constraint in constraints:
        try:
            new_id = constraint_id_map[constraint.id]
            mapped = {
                entity_id: entity_id_map[entity_id]
                for entity_id in sketch_constraint_entity_ids(constraint)
            }
        except KeyError as error:
            raise ValueError(f"incomplete sketch constraint copy mapping: {error.args[0]}") from error
        if isinstance(constraint, (SketchCoincidentConstraint, SketchDistanceDimension)):
            copied.append(replace(
                constraint,
                id=new_id,
                first_point_id=mapped[constraint.first_point_id],
                second_point_id=mapped[constraint.second_point_id],
            ))
        elif isinstance(constraint, SketchPointOnCurveConstraint):
            copied.append(replace(
                constraint,
                id=new_id,
                point_id=mapped[constraint.point_id],
                curve_id=mapped[constraint.curve_id],
            ))
        elif isinstance(constraint, (SketchHorizontalConstraint, SketchVerticalConstraint)):
            copied.append(replace(
                constraint, id=new_id, line_id=mapped[constraint.line_id]
            ))
        elif isinstance(constraint, SketchFixedConstraint):
            copied.append(replace(
                constraint, id=new_id, point_id=mapped[constraint.point_id]
            ))
        else:
            copied.append(replace(
                constraint, id=new_id, curve_id=mapped[constraint.curve_id]
            ))
    return tuple(copied)


def constraints_after_curve_split(
    constraints: tuple[SketchConstraint, ...],
    original_curve_id: str,
) -> tuple[SketchConstraint, ...]:
    """Phase-2 split rule: retain unrelated constraints and drop ambiguous ones.

    Phase 5 can replace this conservative hook with type-specific migration to
    deterministic derived curve IDs.
    """

    return constraints_without_entities(constraints, {original_curve_id})


def validate_sketch_constraints(
    constraints: tuple[object, ...],
    point_map: dict[str, SketchPoint],
    curves: tuple[object, ...],
) -> None:
    if any(type(item) not in SKETCH_CONSTRAINT_TYPES for item in constraints):
        raise TypeError("strict sketch constraints contain an unsupported value")
    seen_ids: set[str] = set()
    curve_map = {curve.id: curve for curve in curves}
    for constraint in constraints:
        folded = constraint.id.casefold()
        if folded in seen_ids:
            raise ValueError(f"duplicate sketch constraint id: {constraint.id!r}")
        seen_ids.add(folded)

        if isinstance(constraint, (SketchCoincidentConstraint, SketchDistanceDimension)):
            point_ids = (constraint.first_point_id, constraint.second_point_id)
        elif isinstance(constraint, (SketchPointOnCurveConstraint, SketchFixedConstraint)):
            point_ids = (constraint.point_id,)
        else:
            point_ids = ()
        missing_point = next((item for item in point_ids if item not in point_map), None)
        if missing_point is not None:
            raise ValueError(
                f"sketch constraint {constraint.id!r} references unknown point "
                f"{missing_point!r}"
            )

        if isinstance(constraint, SketchPointOnCurveConstraint):
            curve_id = constraint.curve_id
            expected = STRICT_SKETCH_CURVE_TYPES
        elif isinstance(constraint, (SketchHorizontalConstraint, SketchVerticalConstraint)):
            curve_id = constraint.line_id
            expected = (SketchLine,)
        elif isinstance(constraint, SketchRadiusDimension):
            curve_id = constraint.curve_id
            expected = (SketchCircle, SketchArc)
        else:
            continue
        curve = curve_map.get(curve_id)
        if curve is None:
            raise ValueError(
                f"sketch constraint {constraint.id!r} references unknown curve "
                f"{curve_id!r}"
            )
        if not isinstance(curve, expected):
            raise ValueError(
                f"sketch constraint {constraint.id!r} references the wrong curve type"
            )


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
            and _part_local_logical_id(self.source_logical_id)
            != _part_local_logical_id(self.target_logical_id)
        ):
            raise ValueError("preserved Boolean lineage must keep its logical ID")
        dimensions = {"point": 0, "edge": 1, "face": 2, "body": 3}
        descent = dimensions[source.kind] - dimensions[target.kind]
        if descent not in {0, 1, 2}:
            raise ValueError(
                "Boolean lineage may preserve dimension or derive a generated "
                "intersection edge/point from operand support"
            )
        if descent > 0 and "/intersection/" not in target.logical_id:
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
class PartBooleanContext:
    """Stable cross-Part Boolean intent and complete result lineage."""

    feature_id: str
    target_part_id: str
    tool_part_id: str
    result_part_id: str
    result_entities: tuple[BooleanLineageEntity, ...] = ()
    topology_mappings: tuple[BooleanLineageMapping, ...] = ()

    def __post_init__(self) -> None:
        from .part_namespace import (
            normalize_part_boolean_feature_id,
            normalize_part_id,
        )
        from .part_namespace import part_id_from_logical_id

        feature_id = normalize_part_boolean_feature_id(self.feature_id)
        target_id = normalize_part_id(self.target_part_id, "target_part_id")
        tool_id = normalize_part_id(self.tool_part_id, "tool_part_id")
        result_id = normalize_part_id(self.result_part_id, "result_part_id")
        if len({target_id, tool_id, result_id}) != 3:
            raise ValueError(
                "target, tool, and result Part identities must differ"
            )
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
            raise ValueError("Part Boolean lineage contains duplicate logical IDs")
        if bool(entities) != bool(mappings):
            raise ValueError(
                "Part Boolean entities and mappings must be present together"
            )
        entity_ids = set(logical_ids)
        if entities:
            if tuple(
                item.logical_id for item in entities if item.kind == "body"
            ) != (f"body:{result_id}/domain",):
                raise ValueError(
                    "proven Part Boolean lineage must contain exactly the "
                    "result Part body"
                )
            if not any(item.kind == "face" for item in entities):
                raise ValueError(
                    "proven Part Boolean lineage must contain boundary faces"
                )
            if {item.target_logical_id for item in mappings} != entity_ids:
                raise ValueError(
                    "proven Part Boolean lineage must map every result entity"
                )
            if {item.source for item in mappings} != {"target", "tool"}:
                raise ValueError(
                    "proven Part Boolean lineage requires target and tool proof"
                )
            if any(
                part_id_from_logical_id(item.logical_id) != result_id
                for item in entities
            ):
                raise ValueError(
                    "Part Boolean result entities must use result Part namespace"
                )
            source_ids = {"target": target_id, "tool": tool_id}
            if any(
                part_id_from_logical_id(item.source_logical_id)
                != source_ids[item.source]
                for item in mappings
            ):
                raise ValueError(
                    "Part Boolean source mappings use the wrong Part namespace"
                )
            if any(
                part_id_from_logical_id(item.target_logical_id) != result_id
                for item in mappings
            ):
                raise ValueError(
                    "Part Boolean mappings must target the result Part namespace"
                )
            if any(
                link not in entity_ids
                for item in entities
                for link in item.topology_links
            ):
                raise ValueError(
                    "Part Boolean topology link targets an unknown entity"
                )
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "target_part_id", target_id)
        object.__setattr__(self, "tool_part_id", tool_id)
        object.__setattr__(self, "result_part_id", result_id)
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
                RevolvedGeometry,
                PathSweptGeometry,
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
                RevolvedGeometry,
                PathSweptGeometry,
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
    """A planar geometry extruded along its positive sketch-plane normal."""

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
class RevolvedGeometry:
    """A planar Profile revolved about one global axis through the origin."""

    base: object
    axis: Literal["x", "y", "z"]
    angle_degrees: float
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
            raise TypeError("扫掠操作需要已有二维几何")
        if geometry_dimension(self.base) != 2:
            raise ValueError("只有二维几何可以扫掠")
        normalized_axis = str(self.axis).lower()
        if normalized_axis not in {"x", "y", "z"}:
            raise ValueError("扫掠轴只能是 X、Y 或 Z")
        angle = float(self.angle_degrees)
        if (
            not math.isfinite(angle)
            or angle <= 0.0
            or angle > 360.0
        ):
            raise ValueError("扫掠角度必须大于 0° 且不超过 360°")
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
        if len(references) != len(
            set(reference.logical_id for reference in references)
        ):
            raise ValueError("source_face_ids 不能包含重复 logical IDs")
        normalized_ids: tuple[str, ...] = ()
        if references:
            from .extrusion_selection import resolve_extrusion_source_faces

            normalized_ids = resolve_extrusion_source_faces(
                self.base,
                references,
            ).face_ids
        object.__setattr__(self, "axis", normalized_axis)
        object.__setattr__(self, "angle_degrees", angle)
        object.__setattr__(self, "source_face_ids", normalized_ids)


@dataclass(frozen=True, slots=True)
class PathSweptGeometry:
    """A planar Profile swept along one explicitly ordered open wire path.

    This recipe is the backend-neutral authoring contract.  Exact OCC/Gmsh
    compilation and path-shape validation belong to the sweep execution phase.
    """

    base: object
    path: WireGeometry
    source_face_ids: tuple[str, ...] = ()
    frame_strategy: Literal["fixed", "transport"] = "transport"

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
            raise TypeError("路径扫掠需要已有二维几何")
        if geometry_dimension(self.base) != 2:
            raise ValueError("只有二维几何可以沿路径扫掠")
        if type(self.path) is not WireGeometry:
            raise TypeError("路径扫掠 path 必须是 WireGeometry")
        degrees = {point.name: 0 for point in self.path.points}
        for member in self.path.members:
            degrees[member.start] += 1
            degrees[member.end] += 1
        if any(degree > 2 for degree in degrees.values()):
            raise ValueError("路径扫掠首版不支持分支路径")
        if sorted(degrees.values()).count(1) != 2:
            raise ValueError("路径扫掠需要一条有两个端点的开放路径")
        if any(
            current.end != following.start
            for current, following in zip(
                self.path.members,
                self.path.members[1:],
            )
        ):
            raise ValueError("路径扫掠 path 必须按连续遍历顺序给出")
        ordered_points = (
            self.path.members[0].start,
            *(member.end for member in self.path.members),
        )
        if len(ordered_points) != len(set(ordered_points)):
            raise ValueError("路径扫掠首版只支持开放路径")
        if set(ordered_points) != {point.name for point in self.path.points}:
            raise ValueError("路径扫掠 path 不得包含未使用的点")
        coordinates = {
            point.name: (point.x, point.y, point.z)
            for point in self.path.points
        }
        traversal = tuple(coordinates[name] for name in ordered_points)
        if len(traversal) != len(set(traversal)):
            raise ValueError("路径扫掠路径不得重访同一空间点")
        if _polyline_self_intersects(traversal):
            raise ValueError("路径扫掠首版不支持自相交路径")
        if self.frame_strategy not in {"fixed", "transport"}:
            raise ValueError("路径扫掠 frame_strategy 必须是 fixed 或 transport")
        if isinstance(self.source_face_ids, (str, bytes, bytearray)):
            raise TypeError("source_face_ids 必须是 face logical ID iterable")
        references = tuple(LogicalEntityRef(value) for value in self.source_face_ids)
        if any(reference.kind != "face" for reference in references):
            raise ValueError("source_face_ids 只能包含 face logical IDs")
        if len(references) != len(set(item.logical_id for item in references)):
            raise ValueError("source_face_ids 不能包含重复 logical IDs")
        normalized_ids: tuple[str, ...] = ()
        if references:
            from .extrusion_selection import resolve_extrusion_source_faces

            normalized_ids = resolve_extrusion_source_faces(
                self.base,
                references,
            ).face_ids
        object.__setattr__(self, "source_face_ids", normalized_ids)


def _polyline_self_intersects(
    points: tuple[tuple[float, float, float], ...],
) -> bool:
    """Return whether non-neighbouring straight path segments touch or cross."""

    segments = tuple(zip(points, points[1:]))
    scale = max(1.0, *(abs(value) for point in points for value in point))
    tolerance = 1.0e-10 * scale
    for first_index, (first_start, first_end) in enumerate(segments):
        for second_index in range(first_index + 2, len(segments)):
            second_start, second_end = segments[second_index]
            if _segment_distance_squared(
                first_start,
                first_end,
                second_start,
                second_end,
            ) <= tolerance * tolerance:
                return True
    return False


def _segment_distance_squared(
    first_start: tuple[float, float, float],
    first_end: tuple[float, float, float],
    second_start: tuple[float, float, float],
    second_end: tuple[float, float, float],
) -> float:
    """Squared closest distance between two finite spatial segments."""

    first = tuple(end - start for start, end in zip(first_start, first_end, strict=True))
    second = tuple(end - start for start, end in zip(second_start, second_end, strict=True))
    offset = tuple(start - end for start, end in zip(first_start, second_start, strict=True))
    aa = sum(value * value for value in first)
    bb = sum(left * right for left, right in zip(first, second, strict=True))
    cc = sum(value * value for value in second)
    dd = sum(left * right for left, right in zip(first, offset, strict=True))
    ee = sum(left * right for left, right in zip(second, offset, strict=True))
    denominator = aa * cc - bb * bb
    first_parameter = 0.0
    second_parameter = 0.0
    if denominator > 1.0e-30:
        first_parameter = min(1.0, max(0.0, (bb * ee - cc * dd) / denominator))
    second_parameter = (bb * first_parameter + ee) / cc
    if second_parameter < 0.0:
        second_parameter = 0.0
        first_parameter = min(1.0, max(0.0, -dd / aa))
    elif second_parameter > 1.0:
        second_parameter = 1.0
        first_parameter = min(1.0, max(0.0, (bb - dd) / aa))
    delta = tuple(
        offset_value + first_parameter * first_value - second_parameter * second_value
        for offset_value, first_value, second_value in zip(
            offset, first, second, strict=True
        )
    )
    return sum(value * value for value in delta)


@dataclass(frozen=True, slots=True)
class BooleanGeometry:
    """A boolean feature combining one object and one tool geometry."""

    name: str
    operation: Literal["fuse", "cut", "fragment"]
    object_geometry: object
    tool_geometry: object
    body_context: BooleanBodyContext | None = None
    planar_context: PlanarBooleanContext | None = None
    part_context: PartBooleanContext | None = None

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
            RevolvedGeometry,
            PathSweptGeometry,
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
        if sum(
            context is not None
            for context in (
                self.body_context,
                self.planar_context,
                self.part_context,
            )
        ) > 1:
            raise ValueError(
                "body_context, planar_context, and part_context are "
                "mutually exclusive"
            )
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
        if self.part_context is not None:
            if type(self.part_context) is not PartBooleanContext:
                raise TypeError(
                    "part_context must be PartBooleanContext or None"
                )
            if self.operation not in {"fuse", "cut"}:
                raise ValueError("strict Part Boolean only supports fuse or cut")
            if object_dimension != 3:
                raise ValueError("strict Part Boolean requires 3D operands")
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


@dataclass(frozen=True, slots=True)
class FaceSeedConnectionProof:
    """Positive-area support evidence for one face-seeded solid fuse."""

    support_face_id: str
    tool_start_face_id: str
    overlap_area: float

    def __post_init__(self) -> None:
        support = LogicalEntityRef(self.support_face_id)
        tool_start = LogicalEntityRef(self.tool_start_face_id)
        if support.kind != "face" or tool_start.kind != "face":
            raise ValueError("面种子连接证明必须引用两个面")
        area = _normalize_sketch_scalar(self.overlap_area, "面种子重叠面积")
        if area <= 0.0:
            raise ValueError("面种子连接证明需要正面积重叠")
        object.__setattr__(self, "overlap_area", area)


@dataclass(frozen=True, slots=True)
class FaceSketchBooleanStepProof:
    """Persisted exact lineage for one stable material-profile Boolean step."""

    profile_id: str
    result_entities: tuple[BooleanLineageEntity, ...]
    topology_mappings: tuple[BooleanLineageMapping, ...]
    connection_proof: FaceSeedConnectionProof | None = None

    def __post_init__(self) -> None:
        profile_id = _normalize_sketch_id(self.profile_id, "轮廓 ID")
        entities = tuple(self.result_entities)
        mappings = tuple(self.topology_mappings)
        if any(type(item) is not BooleanLineageEntity for item in entities):
            raise TypeError("分步布尔结果必须包含 BooleanLineageEntity")
        if any(type(item) is not BooleanLineageMapping for item in mappings):
            raise TypeError("分步布尔谱系必须包含 BooleanLineageMapping")
        if self.connection_proof is not None and type(
            self.connection_proof
        ) is not FaceSeedConnectionProof:
            raise TypeError("分步布尔连接证明无效")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "result_entities", entities)
        object.__setattr__(self, "topology_mappings", mappings)


@dataclass(frozen=True, slots=True)
class FaceSketchBooleanGeometry:
    """One three-dimensional Boolean feature authored on a planar solid face."""

    base: object
    feature_id: str
    name: str
    support_face_id: str
    workplane_strategy: FaceSketchWorkplaneStrategy
    sketch: SketchGeometry
    operation: FaceSketchBooleanOperation
    direction: FaceSketchBooleanDirection
    distance: float
    participating_profile_ids: tuple[str, ...]
    external_references: tuple[SketchExternalReference, ...] = ()
    external_coincidences: tuple[SketchExternalCoincidence, ...] = ()
    step_proofs: tuple[FaceSketchBooleanStepProof, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.base, NATIVE_GEOMETRY_TYPES):
            raise TypeError("面草图布尔需要已有原生几何")
        if geometry_dimension(self.base) != 3:
            raise ValueError("面草图布尔的基础几何必须为三维")
        if not isinstance(self.base, MultiBodyGeometry) and not is_single_solid_recipe(
            self.base
        ):
            raise ValueError("面草图布尔的基础几何必须为单实体或显式 MultiBody")
        feature_id = _normalize_sketch_id(self.feature_id, "特征 ID")
        name = _normalize_body_name(self.name, "特征名称")
        face_reference = LogicalEntityRef(self.support_face_id)
        if face_reference.kind != "face":
            raise ValueError("工作面逻辑 ID 必须引用面")
        if type(self.workplane_strategy) is not FaceSketchWorkplaneStrategy:
            raise TypeError("工作面坐标策略无效")
        if type(self.sketch) is not SketchGeometry or not self.sketch.is_strict:
            raise TypeError("面草图布尔必须保存严格平面草图")
        if type(self.operation) is not FaceSketchBooleanOperation:
            raise TypeError("拉伸布尔操作无效")
        if type(self.direction) is not FaceSketchBooleanDirection:
            raise TypeError("拉伸布尔方向无效")
        distance = _normalize_sketch_scalar(self.distance, "拉伸距离")
        if distance <= 0.0:
            raise ValueError("拉伸距离必须为有限正值")
        profile_ids = tuple(
            _normalize_sketch_id(value, "参与轮廓 ID")
            for value in self.participating_profile_ids
        )
        if not profile_ids:
            raise ValueError("至少需要一个参与轮廓")
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("参与轮廓 ID 不能重复")
        references = tuple(self.external_references)
        coincidences = tuple(self.external_coincidences)
        proofs = tuple(self.step_proofs)
        if any(type(item) is not SketchExternalReference for item in references):
            raise TypeError("外部参考集合包含无效值")
        if any(type(item) is not SketchExternalCoincidence for item in coincidences):
            raise TypeError("外部重合集合包含无效值")
        if any(type(item) is not FaceSketchBooleanStepProof for item in proofs):
            raise TypeError("分步布尔证明集合包含无效值")
        reference_ids = tuple(item.id for item in references)
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("外部参考 ID 不能重复")
        sketch_point_ids = {point.id for point in self.sketch.points}
        linked_point_ids: set[str] = set()
        for coincidence in coincidences:
            if coincidence.point_id not in sketch_point_ids:
                raise ValueError("外部重合引用了不存在的草图点")
            if coincidence.reference_id not in set(reference_ids):
                raise ValueError("外部重合引用了不存在的外部参考")
            if coincidence.point_id in linked_point_ids:
                raise ValueError("每个草图点最多绑定一个外部参考")
            linked_point_ids.add(coincidence.point_id)
        proof_ids = tuple(item.profile_id for item in proofs)
        if len(proof_ids) != len(set(proof_ids)):
            raise ValueError("分步布尔证明的轮廓 ID 不能重复")
        if not set(proof_ids).issubset(profile_ids):
            raise ValueError("分步布尔证明只能引用参与轮廓")
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "participating_profile_ids", tuple(sorted(profile_ids)))
        object.__setattr__(self, "external_references", references)
        object.__setattr__(self, "external_coincidences", coincidences)
        object.__setattr__(self, "step_proofs", proofs)

    @property
    def dimension(self) -> Literal[3]:
        """Declare the feature's invariant topological dimension."""

        return 3

    @property
    def result_body_count(self) -> Literal[1]:
        """Declare the single-solid constraint for the modified target Body."""

        return 1

    @property
    def result_constraint(self) -> Literal["single_solid"]:
        """Declare that the modified target must remain one solid."""

        return "single_solid"


def is_single_solid_recipe(recipe: object) -> bool:
    if isinstance(recipe, FaceSketchBooleanGeometry):
        return is_single_solid_recipe(recipe.base)
    if isinstance(recipe, (BoxGeometry, CylinderGeometry)):
        return True
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return is_single_solid_recipe(recipe.base)
    if isinstance(recipe, ExtrudedGeometry):
        from .extrusion_selection import resolve_extrusion_source_faces

        return len(
            resolve_extrusion_source_faces(
                recipe.base,
                recipe.source_face_ids,
            ).face_ids
        ) == 1
    if isinstance(recipe, RevolvedGeometry):
        from .extrusion_selection import resolve_extrusion_source_faces

        return len(
            resolve_extrusion_source_faces(
                recipe.base,
                recipe.source_face_ids,
            ).face_ids
        ) == 1
    if isinstance(recipe, PathSweptGeometry):
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
        if recipe.part_context is not None:
            return recipe.part_context.proven
        return (
            recipe.operation in {"fuse", "cut"}
            and is_single_solid_recipe(recipe.object_geometry)
            and is_single_solid_recipe(recipe.tool_geometry)
        )
    return False


_is_single_solid_recipe = is_single_solid_recipe


def _part_local_logical_id(logical_id: str) -> str:
    """Compare lineage semantics while allowing a Part namespace change."""

    reference = LogicalEntityRef(logical_id)
    semantic_name = logical_id.split(":", 1)[1]
    owner, separator, local_name = semantic_name.partition("/")
    if (
        separator == "/"
        and re.fullmatch(r"P[1-9][0-9]*", owner) is not None
        and local_name
    ):
        return f"{reference.kind}:{local_name}"
    return logical_id


NativeGeometry = (
    PrimitiveGeometry
    | SketchGeometry
    | WireGeometry
    | MovedGeometry
    | RotatedGeometry
    | ExtrudedGeometry
    | RevolvedGeometry
    | PathSweptGeometry
    | BooleanGeometry
    | MultiBodyGeometry
    | FaceSketchBooleanGeometry
)
NATIVE_GEOMETRY_TYPES = (
    *PRIMITIVE_GEOMETRY_TYPES,
    SketchGeometry,
    WireGeometry,
    MovedGeometry,
    RotatedGeometry,
    ExtrudedGeometry,
    RevolvedGeometry,
    PathSweptGeometry,
    BooleanGeometry,
    MultiBodyGeometry,
    FaceSketchBooleanGeometry,
)


def geometry_dimension(recipe: NativeGeometry) -> Literal[1, 2, 3]:
    """Return the topological dimension of a native geometry recipe."""
    if isinstance(recipe, BooleanGeometry):
        return geometry_dimension(recipe.object_geometry)
    if isinstance(recipe, (MultiBodyGeometry, FaceSketchBooleanGeometry)):
        return 3
    if isinstance(recipe, (ExtrudedGeometry, RevolvedGeometry, PathSweptGeometry)):
        return 3
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return geometry_dimension(recipe.base)
    if isinstance(recipe, WireGeometry):
        return 1
    return 3 if isinstance(recipe, (BoxGeometry, CylinderGeometry)) else 2


def planar_geometry_normal(recipe: NativeGeometry) -> tuple[float, float, float]:
    """Return the positive unit normal of one planar native recipe."""

    if geometry_dimension(recipe) != 2:
        raise ValueError("geometry must be planar")
    if isinstance(recipe, SketchGeometry) and recipe.is_strict:
        assert recipe.plane is not None
        return recipe.plane.normal
    if isinstance(recipe, MovedGeometry):
        return planar_geometry_normal(recipe.base)
    if isinstance(recipe, RotatedGeometry):
        x, y, z = planar_geometry_normal(recipe.base)
        angle = math.radians(recipe.angle_degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        if recipe.axis == "x":
            return x, y * cosine - z * sine, y * sine + z * cosine
        if recipe.axis == "y":
            return x * cosine + z * sine, y, -x * sine + z * cosine
        return x * cosine - y * sine, x * sine + y * cosine, z
    if isinstance(recipe, BooleanGeometry):
        object_normal = planar_geometry_normal(recipe.object_geometry)
        tool_normal = planar_geometry_normal(recipe.tool_geometry)
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(object_normal, tool_normal, strict=True)
        ):
            raise ValueError("planar Boolean operands must share one positive normal")
        return object_normal
    return 0.0, 0.0, 1.0


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
    "FaceSketchBooleanDirection",
    "FaceSketchBooleanGeometry",
    "FaceSketchBooleanOperation",
    "FaceSketchBooleanStepProof",
    "FaceSketchWorkplaneStrategy",
    "FaceSeedConnectionProof",
    "MovedGeometry",
    "MultiBodyGeometry",
    "NATIVE_GEOMETRY_TYPES",
    "NativeGeometry",
    "PRIMITIVE_GEOMETRY_TYPES",
    "PlateWithHoleGeometry",
    "PlanarBooleanContext",
    "PartBooleanContext",
    "PathSweptGeometry",
    "PrimitiveGeometry",
    "RectangleGeometry",
    "RevolvedGeometry",
    "RotatedGeometry",
    "SKETCH_CONTOUR_TYPES",
    "STRICT_SKETCH_CURVE_TYPES",
    "SketchArc",
    "SketchCircle",
    "SketchContour",
    "SketchCurve",
    "SketchGeometry",
    "SketchExternalCoincidence",
    "SketchExternalReference",
    "SketchExternalReferenceType",
    "SketchLine",
    "SketchPlane",
    "SketchPoint",
    "SketchRectangle",
    "SolidBody",
    "WireGeometry",
    "WireMember",
    "WirePoint",
    "geometry_dimension",
    "is_single_solid_recipe",
    "planar_geometry_normal",
]
