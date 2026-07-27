"""Headless geometry recipes for native model authoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal


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


@dataclass(frozen=True, slots=True)
class SketchCircle:
    """One circular contour in a planar sketch."""

    operation: Literal["material", "cut"]
    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        if self.operation not in {"material", "cut"}:
            raise ValueError("草图轮廓只能用于添加材料或切除材料")
        values = tuple(float(value) for value in (self.x, self.y, self.radius))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("草图轮廓参数必须是有限数值")
        if values[2] <= 0.0:
            raise ValueError("圆半径必须大于零")
        for field_name, value in zip(("x", "y", "radius"), values):
            object.__setattr__(self, field_name, value)


SketchContour = SketchRectangle | SketchCircle
SKETCH_CONTOUR_TYPES = (SketchRectangle, SketchCircle)


@dataclass(frozen=True, slots=True)
class SketchGeometry:
    """A planar sketch composed from material and cut contours."""

    name: str
    contours: tuple[SketchContour, ...]

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        contours = tuple(self.contours)
        if not normalized_name:
            raise ValueError("草图名称不能为空")
        if not contours or not all(
            isinstance(item, SKETCH_CONTOUR_TYPES) for item in contours
        ):
            raise ValueError("草图至少需要一个有效轮廓")
        if not any(item.operation == "material" for item in contours):
            raise ValueError("草图至少需要一个添加材料轮廓")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "contours", contours)


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
        object.__setattr__(self, "height", height)


@dataclass(frozen=True, slots=True)
class BooleanGeometry:
    """A boolean feature combining one object and one tool geometry."""

    name: str
    operation: Literal["fuse", "cut", "fragment"]
    object_geometry: object
    tool_geometry: object

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
        object.__setattr__(self, "name", normalized_name)


NativeGeometry = (
    PrimitiveGeometry
    | SketchGeometry
    | WireGeometry
    | MovedGeometry
    | RotatedGeometry
    | ExtrudedGeometry
    | BooleanGeometry
)
NATIVE_GEOMETRY_TYPES = (
    *PRIMITIVE_GEOMETRY_TYPES,
    SketchGeometry,
    WireGeometry,
    MovedGeometry,
    RotatedGeometry,
    ExtrudedGeometry,
    BooleanGeometry,
)


def geometry_dimension(recipe: NativeGeometry) -> Literal[1, 2, 3]:
    """Return the topological dimension of a native geometry recipe."""
    if isinstance(recipe, BooleanGeometry):
        return geometry_dimension(recipe.object_geometry)
    if isinstance(recipe, ExtrudedGeometry):
        return 3
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return geometry_dimension(recipe.base)
    if isinstance(recipe, WireGeometry):
        return 1
    return 3 if isinstance(recipe, (BoxGeometry, CylinderGeometry)) else 2


__all__ = [
    "BASE_GEOMETRY_TYPES",
    "BooleanGeometry",
    "BoxGeometry",
    "CylinderGeometry",
    "DiskGeometry",
    "ExtrudedGeometry",
    "MovedGeometry",
    "NATIVE_GEOMETRY_TYPES",
    "NativeGeometry",
    "PRIMITIVE_GEOMETRY_TYPES",
    "PlateWithHoleGeometry",
    "PrimitiveGeometry",
    "RectangleGeometry",
    "RotatedGeometry",
    "SKETCH_CONTOUR_TYPES",
    "SketchCircle",
    "SketchContour",
    "SketchGeometry",
    "SketchRectangle",
    "WireGeometry",
    "WireMember",
    "WirePoint",
    "geometry_dimension",
]
