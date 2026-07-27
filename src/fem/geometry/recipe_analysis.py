"""Pure, backend-neutral analysis of native geometry recipes."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchContour,
    SketchGeometry,
    SketchRectangle,
    WireGeometry,
)


_AXIS_ALIGNMENT_ABS_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class RectangleFrame:
    """One proven axis-aligned rectangle in global XY coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = tuple(
            float(value) for value in (self.x, self.y, self.width, self.height)
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rectangle frame values must be finite")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError("rectangle frame dimensions must be positive")
        for name, value in zip(
            ("x", "y", "width", "height"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    def strictly_contains_rectangle(self, other: RectangleFrame) -> bool:
        """Return whether ``other`` lies strictly inside this frame."""

        if type(other) is not RectangleFrame:
            raise TypeError("other must be a RectangleFrame")
        return (
            self.x < other.x
            and other.x + other.width < self.x + self.width
            and self.y < other.y
            and other.y + other.height < self.y + self.height
        )

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.width
        yield self.height

    def strictly_contains_circle(self, other: CircleFrame) -> bool:
        """Return whether ``other`` lies strictly inside this frame."""

        if type(other) is not CircleFrame:
            raise TypeError("other must be a CircleFrame")
        return (
            self.x < other.center_x - other.radius
            and other.center_x + other.radius < self.x + self.width
            and self.y < other.center_y - other.radius
            and other.center_y + other.radius < self.y + self.height
        )


@dataclass(frozen=True, slots=True)
class CircleFrame:
    """One proven circle in global XY coordinates."""

    center_x: float
    center_y: float
    radius: float

    def __post_init__(self) -> None:
        values = tuple(
            float(value) for value in (self.center_x, self.center_y, self.radius)
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("circle frame values must be finite")
        if values[2] <= 0.0:
            raise ValueError("circle frame radius must be positive")
        for name, value in zip(
            ("center_x", "center_y", "radius"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    @property
    def x(self) -> float:
        """Return the global X coordinate of the center."""

        return self.center_x

    @property
    def y(self) -> float:
        """Return the global Y coordinate of the center."""

        return self.center_y

    def __iter__(self):
        yield self.center_x
        yield self.center_y
        yield self.radius


def expand_sketch_recipe(recipe: SketchGeometry) -> NativeGeometry:
    """Return a detached evaluator recipe for one planar sketch.

    Contour indices and transforms follow the authoring order.  Material
    contours are fused in that order before cut contours are applied in their
    own authoring order, matching the native CAD construction contract.
    """

    if not isinstance(recipe, SketchGeometry):
        raise TypeError("recipe must be a SketchGeometry")

    def contour_geometry(
        contour: SketchContour,
        index: int,
    ) -> NativeGeometry:
        name = f"{recipe.name}-Contour-{index}"
        if isinstance(contour, SketchRectangle):
            result: NativeGeometry = RectangleGeometry(
                name,
                contour.width,
                contour.height,
            )
        elif isinstance(contour, SketchCircle):
            result = DiskGeometry(name, contour.radius)
        else:  # pragma: no cover - SketchGeometry validates contour types
            raise TypeError(f"Unsupported sketch contour: {type(contour).__name__}")
        if contour.x != 0.0 or contour.y != 0.0:
            result = MovedGeometry(result, contour.x, contour.y)
        return result

    indexed_contours = tuple(enumerate(recipe.contours, start=1))
    material = tuple(
        contour_geometry(contour, index)
        for index, contour in indexed_contours
        if contour.operation == "material"
    )
    cuts = tuple(
        contour_geometry(contour, index)
        for index, contour in indexed_contours
        if contour.operation == "cut"
    )
    result = material[0]
    for tool in material[1:]:
        result = BooleanGeometry(recipe.name, "fuse", result, tool)
    for tool in cuts:
        result = BooleanGeometry(recipe.name, "cut", result, tool)
    return result


def axis_aligned_rectangle(recipe: NativeGeometry) -> RectangleFrame | None:
    """Return the proven global XY rectangle frame, if one exists."""

    _require_native_geometry(recipe)
    return _axis_aligned_rectangle(recipe)


def _axis_aligned_rectangle(recipe: NativeGeometry) -> RectangleFrame | None:
    if isinstance(recipe, SketchGeometry):
        return _axis_aligned_rectangle(expand_sketch_recipe(recipe))
    if isinstance(recipe, RectangleGeometry):
        return RectangleFrame(0.0, 0.0, recipe.width, recipe.height)
    if isinstance(recipe, MovedGeometry):
        frame = _axis_aligned_rectangle(recipe.base)
        if frame is None or recipe.dz != 0.0:
            return None
        return RectangleFrame(
            frame.x + recipe.dx,
            frame.y + recipe.dy,
            frame.width,
            frame.height,
        )
    if isinstance(recipe, RotatedGeometry) and math.isclose(
        recipe.angle_degrees % 360.0,
        0.0,
        abs_tol=_AXIS_ALIGNMENT_ABS_TOLERANCE,
    ):
        return _axis_aligned_rectangle(recipe.base)
    return None


def transformed_circle(recipe: NativeGeometry) -> CircleFrame | None:
    """Return the proven global XY circle frame, if one exists."""

    _require_native_geometry(recipe)
    return _transformed_circle(recipe)


def _transformed_circle(recipe: NativeGeometry) -> CircleFrame | None:
    if isinstance(recipe, SketchGeometry):
        return _transformed_circle(expand_sketch_recipe(recipe))
    if isinstance(recipe, DiskGeometry):
        return CircleFrame(0.0, 0.0, recipe.radius)
    if isinstance(recipe, MovedGeometry):
        frame = _transformed_circle(recipe.base)
        if frame is None or recipe.dz != 0.0:
            return None
        return CircleFrame(
            frame.center_x + recipe.dx,
            frame.center_y + recipe.dy,
            frame.radius,
        )
    if isinstance(recipe, RotatedGeometry):
        frame = _transformed_circle(recipe.base)
        if frame is None:
            return None
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)
        return CircleFrame(
            frame.center_x * cosine - frame.center_y * sine,
            frame.center_x * sine + frame.center_y * cosine,
            frame.radius,
        )
    return None


def recipe_characteristic_size(recipe: NativeGeometry) -> float:
    """Return the deterministic characteristic size used by native meshing."""

    _require_native_geometry(recipe)
    if isinstance(recipe, BooleanGeometry):
        return min(
            recipe_characteristic_size(recipe.object_geometry),
            recipe_characteristic_size(recipe.tool_geometry),
        )
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return recipe_characteristic_size(recipe.base)
    if isinstance(recipe, ExtrudedGeometry):
        return min(recipe_characteristic_size(recipe.base), recipe.height)
    if isinstance(recipe, SketchGeometry):
        return recipe_characteristic_size(expand_sketch_recipe(recipe))
    if isinstance(recipe, WireGeometry):
        points = {point.name: point for point in recipe.points}
        return max(
            math.hypot(
                points[member.start].x - points[member.end].x,
                points[member.start].y - points[member.end].y,
                points[member.start].z - points[member.end].z,
            )
            for member in recipe.members
        )
    if isinstance(recipe, (RectangleGeometry, PlateWithHoleGeometry)):
        return min(recipe.width, recipe.height)
    if isinstance(recipe, DiskGeometry):
        return 2.0 * recipe.radius
    if isinstance(recipe, BoxGeometry):
        return min(recipe.width, recipe.depth, recipe.height)
    if isinstance(recipe, CylinderGeometry):
        return min(2.0 * recipe.radius, recipe.height)
    raise AssertionError("native geometry dispatch is incomplete")


def supports_structured_hexahedron(recipe: NativeGeometry) -> bool:
    """Return whether the existing structured Hex mesher supports ``recipe``."""

    _require_native_geometry(recipe)
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return supports_structured_hexahedron(recipe.base)
    if isinstance(recipe, BoxGeometry):
        return True
    if not isinstance(recipe, ExtrudedGeometry) or not isinstance(
        recipe.base,
        SketchGeometry,
    ):
        return False
    contours = recipe.base.contours
    return (
        len(contours) == 1
        and isinstance(contours[0], SketchRectangle)
        and contours[0].operation == "material"
    )


def _require_native_geometry(recipe: object) -> None:
    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError(
            f"Unsupported native geometry recipe: {type(recipe).__name__}"
        )


__all__ = [
    "CircleFrame",
    "RectangleFrame",
    "axis_aligned_rectangle",
    "expand_sketch_recipe",
    "recipe_characteristic_size",
    "supports_structured_hexahedron",
    "transformed_circle",
]
