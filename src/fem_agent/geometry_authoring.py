"""Typed A2 geometry drafts, bounded previews, and geometry proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

from fem.geometry import (
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    geometry_dimension,
)

from .authoring import (
    AgentProposal,
    AuthoringContext,
    AuthoringContractError,
    ModelOperation,
    OperationKind,
    ProposalKind,
    UnitContextSummary,
)
from .naming import NameAllocator


_PREVIEW_SEGMENTS = 24
_MAX_PREVIEW_POINTS = 64


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite real number")
    return result


@dataclass(frozen=True, slots=True)
class StaticGeometryPreview:
    """Detached line preview with a fixed point budget."""

    dimension: int
    points: tuple[tuple[float, float, float], ...]
    lines: tuple[tuple[int, ...], ...]
    bounds: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if self.dimension not in {2, 3}:
            raise ValueError("preview dimension must be 2 or 3")
        points = tuple(self.points)
        lines = tuple(self.lines)
        if not points or len(points) > _MAX_PREVIEW_POINTS:
            raise ValueError("preview point count is outside the A2 bound")
        if any(
            len(point) != 3
            or any(not math.isfinite(float(component)) for component in point)
            for point in points
        ):
            raise ValueError("preview points must be finite XYZ triples")
        if any(
            len(line) < 2 or any(index < 0 or index >= len(points) for index in line)
            for line in lines
        ):
            raise ValueError("preview lines contain invalid point indices")
        if len(self.bounds) != 6 or any(
            not math.isfinite(float(value)) for value in self.bounds
        ):
            raise ValueError("preview bounds must contain six finite values")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "bounded_wireframe",
            "dimension": self.dimension,
            "point_count": len(self.points),
            "line_count": len(self.lines),
            "points": [list(point) for point in self.points],
            "lines": [list(line) for line in self.lines],
            "bounds": list(self.bounds),
        }


@dataclass(frozen=True, slots=True)
class GeometryDraft:
    recipe: object
    recipe_payload: Mapping[str, object]
    preview: StaticGeometryPreview
    key_dimensions: Mapping[str, float]
    transforms: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, NATIVE_GEOMETRY_TYPES):
            raise TypeError("recipe must be native geometry")
        payload = geometry_recipe_to_payload(self.recipe)
        if dict(self.recipe_payload) != payload:
            raise ValueError("recipe_payload does not match recipe")
        dimensions = {
            str(key): _finite(value, str(key))
            for key, value in self.key_dimensions.items()
        }
        object.__setattr__(self, "recipe_payload", payload)
        object.__setattr__(self, "key_dimensions", dimensions)
        object.__setattr__(
            self,
            "transforms",
            tuple(dict(item) for item in self.transforms),
        )


def rectangle_geometry(
    name: str,
    *,
    width: Real,
    height: Real,
) -> GeometryDraft:
    recipe = RectangleGeometry(
        name,
        _finite(width, "width"),
        _finite(height, "height"),
    )
    return _draft(recipe, {"width": recipe.width, "height": recipe.height})


def disk_geometry(name: str, *, radius: Real) -> GeometryDraft:
    recipe = DiskGeometry(name, _finite(radius, "radius"))
    return _draft(recipe, {"radius": recipe.radius})


def plate_with_hole_geometry(
    name: str,
    *,
    width: Real,
    height: Real,
    hole_radius: Real,
    hole_center: Sequence[Real] | None = None,
    center_offset: Sequence[Real] | None = None,
) -> GeometryDraft:
    """Build a plate using exactly one explicit hole-position convention."""

    if (hole_center is None) == (center_offset is None):
        raise ValueError("provide exactly one of hole_center or center_offset")
    normalized_width = _finite(width, "width")
    normalized_height = _finite(height, "height")
    position = hole_center if hole_center is not None else center_offset
    if isinstance(position, (str, bytes, bytearray)):
        raise TypeError("hole position must be a two-component sequence")
    values = tuple(position or ())
    if len(values) != 2:
        raise ValueError("hole position must contain exactly two components")
    first = _finite(values[0], "hole position x")
    second = _finite(values[1], "hole position y")
    if hole_center is None:
        hole_x = normalized_width / 2.0 + first
        hole_y = normalized_height / 2.0 + second
        position_kind = "center_offset"
    else:
        hole_x, hole_y = first, second
        position_kind = "hole_center"
    recipe = PlateWithHoleGeometry(
        name,
        normalized_width,
        normalized_height,
        hole_x,
        hole_y,
        _finite(hole_radius, "hole_radius"),
    )
    draft = _draft(
        recipe,
        {
            "width": recipe.width,
            "height": recipe.height,
            "hole_x": recipe.hole_x,
            "hole_y": recipe.hole_y,
            "hole_radius": recipe.hole_radius,
        },
    )
    return GeometryDraft(
        recipe=draft.recipe,
        recipe_payload=draft.recipe_payload,
        preview=draft.preview,
        key_dimensions=draft.key_dimensions,
        transforms=(
            {
                "kind": position_kind,
                "x": first,
                "y": second,
            },
        ),
    )


def box_geometry(
    name: str,
    *,
    width: Real,
    depth: Real,
    height: Real,
) -> GeometryDraft:
    recipe = BoxGeometry(
        name,
        _finite(width, "width"),
        _finite(depth, "depth"),
        _finite(height, "height"),
    )
    return _draft(
        recipe,
        {
            "width": recipe.width,
            "depth": recipe.depth,
            "height": recipe.height,
        },
    )


def cylinder_geometry(
    name: str,
    *,
    radius: Real,
    height: Real,
) -> GeometryDraft:
    recipe = CylinderGeometry(
        name,
        _finite(radius, "radius"),
        _finite(height, "height"),
    )
    return _draft(
        recipe,
        {"radius": recipe.radius, "height": recipe.height},
    )


def translate_geometry(
    draft: GeometryDraft,
    *,
    dx: Real,
    dy: Real,
    dz: Real = 0.0,
) -> GeometryDraft:
    if type(draft) is not GeometryDraft:
        raise TypeError("draft must be GeometryDraft")
    recipe = MovedGeometry(
        draft.recipe,
        _finite(dx, "dx"),
        _finite(dy, "dy"),
        _finite(dz, "dz"),
    )
    return _draft(
        recipe,
        draft.key_dimensions,
        (
            *draft.transforms,
            {"kind": "translation", "dx": recipe.dx, "dy": recipe.dy, "dz": recipe.dz},
        ),
    )


def rotate_geometry(
    draft: GeometryDraft,
    *,
    axis: str,
    angle_degrees: Real,
) -> GeometryDraft:
    if type(draft) is not GeometryDraft:
        raise TypeError("draft must be GeometryDraft")
    recipe = RotatedGeometry(
        draft.recipe,
        axis,  # type: ignore[arg-type]
        _finite(angle_degrees, "angle_degrees"),
    )
    return _draft(
        recipe,
        draft.key_dimensions,
        (
            *draft.transforms,
            {
                "kind": "rotation",
                "axis": recipe.axis,
                "angle_degrees": recipe.angle_degrees,
            },
        ),
    )


def create_geometry_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    draft: GeometryDraft,
    part_function: str,
    unit_context: UnitContextSummary,
    project_function: str | None = None,
) -> AgentProposal:
    """Create one revision-bound proposal for blank or native sessions."""

    if type(context) is not AuthoringContext:
        raise TypeError("context must be AuthoringContext")
    if type(draft) is not GeometryDraft:
        raise TypeError("draft must be GeometryDraft")
    if type(unit_context) is not UnitContextSummary:
        raise TypeError("unit_context must be UnitContextSummary")
    binding = context.binding
    if not binding.supported or binding.source_kind not in {"blank", "native"}:
        raise AuthoringContractError(
            "geometry proposal requires blank or native context"
        )
    if context.unit_context is not None and context.unit_context != unit_context:
        raise AuthoringContractError(
            "A2 geometry proposal cannot change existing project units"
        )
    units = unit_context.to_dict()
    part_name = NameAllocator(
        {"parts": (part.name for part in context.parts)}
    ).allocate("parts", "部件", part_function)
    if binding.source_kind == "blank":
        if project_function is None:
            raise AuthoringContractError(
                "blank-session geometry proposal requires project_function"
            )
        project_name = NameAllocator().allocate(
            "models",
            "模型",
            project_function,
        )
        operation = ModelOperation(
            OperationKind.CREATE_NATIVE_PROJECT,
            {
                "project_name": project_name,
                "part_name": part_name,
                "recipe": dict(draft.recipe_payload),
                "unit_context": units,
            },
        )
        operation_label = "创建 native 项目并加入首部件"
        target_model = project_name
    else:
        if context.model_name is None:
            raise AuthoringContractError("native context requires model_name")
        operation = ModelOperation(
            OperationKind.ADD_NATIVE_PART,
            {
                "part_name": part_name,
                "recipe": dict(draft.recipe_payload),
                "unit_context": units,
            },
        )
        operation_label = "向 native 项目增加部件"
        target_model = context.model_name
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=binding.document_id,
        target_session_id=binding.session_id,
        base_session_revision=binding.session_revision,
        draft_revision=draft_revision,
        operations=(operation,),
        preconditions={
            "source_kind": binding.source_kind,
            "unit_context": units,
        },
        expected_changes={
            "part_count_delta": 1,
            "project_created": binding.source_kind == "blank",
            "projection_refresh_count": 1,
        },
        invalidation_impact={
            "mesh": False,
            "definitions": False,
            "results": False,
        },
        display_summary={
            "title": operation_label,
            "target_model": target_model,
            "operation": operation.kind.value,
            "part_name": part_name,
            "recipe_type": type(draft.recipe).__name__,
            "dimension": geometry_dimension(draft.recipe),
            "key_dimensions": dict(draft.key_dimensions),
            "length_unit": unit_context.length,
            "transforms": [dict(item) for item in draft.transforms],
            "expected_new_objects": [part_name],
            "invalidated_objects": [],
            "base_session_revision": binding.session_revision,
            "preview": draft.preview.to_dict(),
        },
    )


def geometry_recipe_to_payload(recipe: object) -> dict[str, object]:
    if type(recipe) is RectangleGeometry:
        return {
            "kind": "rectangle",
            "name": recipe.name,
            "width": recipe.width,
            "height": recipe.height,
        }
    if type(recipe) is DiskGeometry:
        return {"kind": "disk", "name": recipe.name, "radius": recipe.radius}
    if type(recipe) is PlateWithHoleGeometry:
        return {
            "kind": "plate_with_hole",
            "name": recipe.name,
            "width": recipe.width,
            "height": recipe.height,
            "hole_x": recipe.hole_x,
            "hole_y": recipe.hole_y,
            "hole_radius": recipe.hole_radius,
        }
    if type(recipe) is BoxGeometry:
        return {
            "kind": "box",
            "name": recipe.name,
            "width": recipe.width,
            "depth": recipe.depth,
            "height": recipe.height,
        }
    if type(recipe) is CylinderGeometry:
        return {
            "kind": "cylinder",
            "name": recipe.name,
            "radius": recipe.radius,
            "height": recipe.height,
        }
    if type(recipe) is MovedGeometry:
        return {
            "kind": "translated",
            "base": geometry_recipe_to_payload(recipe.base),
            "dx": recipe.dx,
            "dy": recipe.dy,
            "dz": recipe.dz,
        }
    if type(recipe) is RotatedGeometry:
        return {
            "kind": "rotated",
            "base": geometry_recipe_to_payload(recipe.base),
            "axis": recipe.axis,
            "angle_degrees": recipe.angle_degrees,
        }
    raise TypeError("recipe is outside the A2 geometry subset")


def geometry_recipe_from_payload(value: object) -> object:
    if not isinstance(value, Mapping):
        raise TypeError("geometry recipe payload must be an object")
    kind = value.get("kind")
    fields: dict[str, set[str]] = {
        "rectangle": {"kind", "name", "width", "height"},
        "disk": {"kind", "name", "radius"},
        "plate_with_hole": {
            "kind",
            "name",
            "width",
            "height",
            "hole_x",
            "hole_y",
            "hole_radius",
        },
        "box": {"kind", "name", "width", "depth", "height"},
        "cylinder": {"kind", "name", "radius", "height"},
        "translated": {"kind", "base", "dx", "dy", "dz"},
        "rotated": {"kind", "base", "axis", "angle_degrees"},
    }
    if kind not in fields or set(value) != fields[kind]:  # type: ignore[index]
        raise ValueError("geometry recipe fields do not match the A2 schema")
    if kind == "rectangle":
        return RectangleGeometry(value["name"], value["width"], value["height"])
    if kind == "disk":
        return DiskGeometry(value["name"], value["radius"])
    if kind == "plate_with_hole":
        return PlateWithHoleGeometry(
            value["name"],
            value["width"],
            value["height"],
            value["hole_x"],
            value["hole_y"],
            value["hole_radius"],
        )
    if kind == "box":
        return BoxGeometry(
            value["name"],
            value["width"],
            value["depth"],
            value["height"],
        )
    if kind == "cylinder":
        return CylinderGeometry(
            value["name"],
            value["radius"],
            value["height"],
        )
    if kind == "translated":
        return MovedGeometry(
            geometry_recipe_from_payload(value["base"]),
            value["dx"],
            value["dy"],
            value["dz"],
        )
    return RotatedGeometry(
        geometry_recipe_from_payload(value["base"]),
        value["axis"],
        value["angle_degrees"],
    )


def _draft(
    recipe: object,
    key_dimensions: Mapping[str, float],
    transforms: tuple[Mapping[str, object], ...] = (),
) -> GeometryDraft:
    return GeometryDraft(
        recipe=recipe,
        recipe_payload=geometry_recipe_to_payload(recipe),
        preview=_preview(recipe),
        key_dimensions=key_dimensions,
        transforms=transforms,
    )


def _preview(recipe: object) -> StaticGeometryPreview:
    if type(recipe) is MovedGeometry:
        base = _preview(recipe.base)
        points = tuple(
            (x + recipe.dx, y + recipe.dy, z + recipe.dz) for x, y, z in base.points
        )
        return _make_preview(base.dimension, points, base.lines)
    if type(recipe) is RotatedGeometry:
        base = _preview(recipe.base)
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)

        def rotated(point: tuple[float, float, float]) -> tuple[float, float, float]:
            x, y, z = point
            if recipe.axis == "x":
                return x, y * cosine - z * sine, y * sine + z * cosine
            if recipe.axis == "y":
                return x * cosine + z * sine, y, -x * sine + z * cosine
            return x * cosine - y * sine, x * sine + y * cosine, z

        return _make_preview(
            base.dimension, tuple(map(rotated, base.points)), base.lines
        )
    if type(recipe) is RectangleGeometry:
        points = (
            (0.0, 0.0, 0.0),
            (recipe.width, 0.0, 0.0),
            (recipe.width, recipe.height, 0.0),
            (0.0, recipe.height, 0.0),
        )
        return _make_preview(2, points, ((0, 1, 2, 3, 0),))
    if type(recipe) is DiskGeometry:
        points = _circle_points(0.0, 0.0, recipe.radius, 0.0)
        return _make_preview(2, points, (tuple(range(len(points))) + (0,),))
    if type(recipe) is PlateWithHoleGeometry:
        outer = (
            (0.0, 0.0, 0.0),
            (recipe.width, 0.0, 0.0),
            (recipe.width, recipe.height, 0.0),
            (0.0, recipe.height, 0.0),
        )
        hole = _circle_points(
            recipe.hole_x,
            recipe.hole_y,
            recipe.hole_radius,
            0.0,
        )
        return _make_preview(
            2,
            outer + hole,
            (
                (0, 1, 2, 3, 0),
                tuple(range(4, 4 + len(hole))) + (4,),
            ),
        )
    if type(recipe) is BoxGeometry:
        points = tuple(
            (x, y, z)
            for z in (0.0, recipe.height)
            for y in (0.0, recipe.depth)
            for x in (0.0, recipe.width)
        )
        lines = (
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        return _make_preview(3, points, lines)
    if type(recipe) is CylinderGeometry:
        bottom = _circle_points(0.0, 0.0, recipe.radius, 0.0)
        top = _circle_points(0.0, 0.0, recipe.radius, recipe.height)
        count = len(bottom)
        lines = (
            tuple(range(count)) + (0,),
            tuple(range(count, count * 2)) + (count,),
            (0, count),
            (count // 4, count + count // 4),
            (count // 2, count + count // 2),
            (3 * count // 4, count + 3 * count // 4),
        )
        return _make_preview(3, bottom + top, lines)
    raise TypeError("recipe is outside the A2 preview subset")


def _circle_points(
    center_x: float,
    center_y: float,
    radius: float,
    z: float,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            center_x + radius * math.cos(2.0 * math.pi * index / _PREVIEW_SEGMENTS),
            center_y + radius * math.sin(2.0 * math.pi * index / _PREVIEW_SEGMENTS),
            z,
        )
        for index in range(_PREVIEW_SEGMENTS)
    )


def _make_preview(
    dimension: int,
    points: tuple[tuple[float, float, float], ...],
    lines: tuple[tuple[int, ...], ...],
) -> StaticGeometryPreview:
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    zs = tuple(point[2] for point in points)
    return StaticGeometryPreview(
        dimension=dimension,
        points=points,
        lines=lines,
        bounds=(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)),
    )


__all__ = [
    "GeometryDraft",
    "StaticGeometryPreview",
    "box_geometry",
    "create_geometry_proposal",
    "cylinder_geometry",
    "disk_geometry",
    "geometry_recipe_from_payload",
    "geometry_recipe_to_payload",
    "plate_with_hole_geometry",
    "rectangle_geometry",
    "rotate_geometry",
    "translate_geometry",
]
