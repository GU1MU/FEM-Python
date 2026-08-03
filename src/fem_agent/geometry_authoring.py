"""Typed native geometry drafts, bounded previews, and edit proposals."""

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
    SketchArc,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    SketchRectangle,
    WireGeometry,
    WireMember,
    WirePoint,
    geometry_dimension,
    legacy_sketch_to_strict,
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
_MAX_PREVIEW_POINTS = 128


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
    point_names: tuple[str, ...] = ()
    member_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dimension not in {1, 2, 3}:
            raise ValueError("preview dimension must be 1, 2, or 3")
        points = tuple(self.points)
        lines = tuple(self.lines)
        point_names = tuple(self.point_names)
        member_names = tuple(self.member_names)
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
        if point_names and len(point_names) != len(points):
            raise ValueError("preview point names must match preview points")
        if member_names and len(member_names) != len(lines):
            raise ValueError("preview member names must match preview lines")
        if any(not name.strip() for name in (*point_names, *member_names)):
            raise ValueError("preview entity names must not be blank")
        object.__setattr__(self, "point_names", point_names)
        object.__setattr__(self, "member_names", member_names)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "kind": "bounded_wireframe",
            "dimension": self.dimension,
            "point_count": len(self.points),
            "line_count": len(self.lines),
            "points": [list(point) for point in self.points],
            "lines": [list(line) for line in self.lines],
            "bounds": list(self.bounds),
        }
        if self.point_names:
            payload["point_names"] = list(self.point_names)
        if self.member_names:
            payload["member_names"] = list(self.member_names)
        return payload


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
        if (
            type(self.recipe) is SketchGeometry
            and self.recipe.is_strict
            and (
                len(self.recipe.points) > 128
                or len(self.recipe.curves) > 128
            )
        ):
            raise ValueError("planar sketch exceeds the 128-entity bound")
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


def planar_sketch_geometry(
    name: str,
    *,
    contours: Sequence[SketchRectangle | SketchCircle],
) -> GeometryDraft:
    """Build a general strict XY sketch from bounded closed contours."""

    values = tuple(contours)
    if not values:
        raise ValueError("contours must contain at least one closed contour")
    if any(
        type(item) not in {SketchRectangle, SketchCircle}
        or (isinstance(item, SketchCircle) and not item.is_legacy)
        for item in values
    ):
        raise TypeError(
            "contours must contain legacy SketchRectangle or SketchCircle values"
        )
    recipe = legacy_sketch_to_strict(SketchGeometry(name, values))
    return _draft(
        recipe,
        {
            "point_count": float(len(recipe.points)),
            "curve_count": float(len(recipe.curves)),
        },
    )


def planar_polygon_geometry(
    name: str,
    *,
    vertices: Sequence[Sequence[Real]],
) -> GeometryDraft:
    """Build a strict XY sketch containing one polygonal profile."""

    values = tuple(vertices)
    if not 3 <= len(values) <= 64:
        raise ValueError("polygon vertices must contain between 3 and 64 points")
    coordinates: list[tuple[float, float]] = []
    for index, vertex in enumerate(values):
        if isinstance(vertex, (str, bytes, bytearray)):
            raise TypeError("polygon vertices must contain coordinate pairs")
        components = tuple(vertex)
        if len(components) != 2:
            raise ValueError("polygon vertices must contain coordinate pairs")
        coordinates.append(
            (
                _finite(components[0], f"vertices[{index}].x"),
                _finite(components[1], f"vertices[{index}].y"),
            )
        )
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("polygon vertices must be distinct")
    point_ids = tuple(f"P{index + 1}" for index in range(len(coordinates)))
    line_ids = tuple(f"L{index + 1}" for index in range(len(coordinates)))
    points = tuple(
        SketchPoint(point_id, x, y)
        for point_id, (x, y) in zip(point_ids, coordinates, strict=True)
    )
    curves = tuple(
        SketchLine(
            line_ids[index],
            point_ids[index],
            point_ids[(index + 1) % len(point_ids)],
        )
        for index in range(len(point_ids))
    )
    recipe = SketchGeometry(
        name,
        SketchPlane.xy(),
        points,
        curves,
    )
    return _draft(
        recipe,
        {
            "point_count": float(len(points)),
            "curve_count": float(len(curves)),
        },
    )


def plate_with_hole_geometry(
    name: str,
    *,
    width: Real,
    height: Real,
    hole_radius: Real,
    hole_center: Sequence[Real] | None = None,
    center_offset: Sequence[Real] | None = None,
) -> GeometryDraft:
    """Build the legacy single-hole compatibility recipe.

    Active Agent tools use general strict planar sketches. This constructor is
    retained only to decode and test projects authored before that migration.
    """

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


def wire_geometry(
    name: str,
    *,
    points: Sequence[WirePoint],
    members: Sequence[WireMember],
) -> GeometryDraft:
    """Build one bounded spatial wire from stable named entities."""

    point_values = tuple(points)
    member_values = tuple(members)
    if not 2 <= len(point_values) <= _MAX_PREVIEW_POINTS:
        raise ValueError("wire points must contain between 2 and 128 items")
    if not 1 <= len(member_values) <= _MAX_PREVIEW_POINTS:
        raise ValueError("wire members must contain between 1 and 128 items")
    recipe = WireGeometry(name, point_values, member_values)
    return _draft(
        recipe,
        {
            "point_count": float(len(recipe.points)),
            "member_count": float(len(recipe.members)),
        },
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
    summary: str | None = None,
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
    proposal_summary = (
        operation_label
        if summary is None
        else str(summary).strip()
    )
    if not proposal_summary:
        raise AuthoringContractError("geometry proposal summary is blank")
    if len(proposal_summary.encode("utf-8")) > 2048:
        raise AuthoringContractError("geometry proposal summary is too long")
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
            "summary": proposal_summary,
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


def create_geometry_edit_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    part_id: str,
    draft: GeometryDraft,
    summary: str,
) -> AgentProposal:
    """Create a revision-bound in-place Part geometry edit proposal."""

    if type(context) is not AuthoringContext:
        raise TypeError("context must be AuthoringContext")
    if type(draft) is not GeometryDraft:
        raise TypeError("draft must be GeometryDraft")
    target = next(
        (
            part
            for part in context.parts
            if part.part_id == str(part_id) and not part.suppressed
        ),
        None,
    )
    if (
        not context.binding.supported
        or context.binding.source_kind != "native"
        or target is None
    ):
        raise AuthoringContractError(
            "geometry edit requires one editable native Part"
        )
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=context.binding.document_id,
        target_session_id=context.binding.session_id,
        base_session_revision=context.binding.session_revision,
        draft_revision=draft_revision,
        operations=(
            ModelOperation(
                OperationKind.REPLACE_PART_GEOMETRY,
                {
                    "part_id": target.part_id,
                    "recipe": dict(draft.recipe_payload),
                },
            ),
        ),
        preconditions={
            "source_kind": "native",
            "part_id": target.part_id,
        },
        expected_changes={
            "part_count_delta": 0,
            "edited_part_id": target.part_id,
            "projection_refresh_count": 1,
        },
        invalidation_impact={
            "mesh": True,
            "definitions": True,
            "results": True,
        },
        display_summary={
            "title": f"修改部件 {target.name}",
            "target_model": context.model_name,
            "operation": OperationKind.REPLACE_PART_GEOMETRY.value,
            "part_id": target.part_id,
            "part_name": target.name,
            "recipe_type": "planar_sketch",
            "dimension": geometry_dimension(draft.recipe),
            "key_dimensions": dict(draft.key_dimensions),
            "summary": str(summary).strip(),
            "expected_new_objects": [],
            "invalidated_objects": ["mesh", "definitions", "results"],
            "base_session_revision": context.binding.session_revision,
            "preview": draft.preview.to_dict(),
        },
    )


def add_planar_circle(
    recipe: object,
    *,
    center_x: Real,
    center_y: Real,
    radius: Real,
) -> GeometryDraft:
    sketch = _as_strict_planar_sketch(recipe)
    point_id = _next_sketch_ids(sketch, "P", 1)[0]
    curve_id = _next_sketch_ids(sketch, "C", 1)[0]
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        (
            *sketch.points,
            SketchPoint(
                point_id,
                _finite(center_x, "center_x"),
                _finite(center_y, "center_y"),
            ),
        ),
        (
            *sketch.curves,
            SketchCircle(curve_id, point_id, _finite(radius, "radius")),
        ),
    )
    return _draft(
        updated,
        {
            "point_count": float(len(updated.points)),
            "curve_count": float(len(updated.curves)),
        },
    )


def add_planar_rectangle(
    recipe: object,
    *,
    x: Real,
    y: Real,
    width: Real,
    height: Real,
) -> GeometryDraft:
    sketch = _as_strict_planar_sketch(recipe)
    x_value = _finite(x, "x")
    y_value = _finite(y, "y")
    width_value = _finite(width, "width")
    height_value = _finite(height, "height")
    if width_value <= 0.0 or height_value <= 0.0:
        raise ValueError("rectangle width and height must be positive")
    point_ids = _next_sketch_ids(sketch, "P", 4)
    line_ids = _next_sketch_ids(sketch, "L", 4)
    points = (
        SketchPoint(point_ids[0], x_value, y_value),
        SketchPoint(point_ids[1], x_value + width_value, y_value),
        SketchPoint(
            point_ids[2],
            x_value + width_value,
            y_value + height_value,
        ),
        SketchPoint(point_ids[3], x_value, y_value + height_value),
    )
    lines = tuple(
        SketchLine(
            line_ids[index],
            point_ids[index],
            point_ids[(index + 1) % 4],
        )
        for index in range(4)
    )
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        (*sketch.points, *points),
        (*sketch.curves, *lines),
    )
    return _draft(
        updated,
        {
            "point_count": float(len(updated.points)),
            "curve_count": float(len(updated.curves)),
        },
    )


def add_planar_polygon(
    recipe: object,
    *,
    vertices: Sequence[Sequence[Real]],
) -> GeometryDraft:
    sketch = _as_strict_planar_sketch(recipe)
    values = tuple(vertices)
    if not 3 <= len(values) <= 64:
        raise ValueError("polygon vertices must contain between 3 and 64 points")
    coordinates: list[tuple[float, float]] = []
    for index, vertex in enumerate(values):
        if isinstance(vertex, (str, bytes, bytearray)):
            raise TypeError("polygon vertices must contain coordinate pairs")
        components = tuple(vertex)
        if len(components) != 2:
            raise ValueError("polygon vertices must contain coordinate pairs")
        coordinates.append(
            (
                _finite(components[0], f"vertices[{index}].x"),
                _finite(components[1], f"vertices[{index}].y"),
            )
        )
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("polygon vertices must be distinct")
    point_ids = _next_sketch_ids(sketch, "P", len(coordinates))
    line_ids = _next_sketch_ids(sketch, "L", len(coordinates))
    points = tuple(
        SketchPoint(point_id, x, y)
        for point_id, (x, y) in zip(point_ids, coordinates, strict=True)
    )
    lines = tuple(
        SketchLine(
            line_ids[index],
            point_ids[index],
            point_ids[(index + 1) % len(point_ids)],
        )
        for index in range(len(point_ids))
    )
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        (*sketch.points, *points),
        (*sketch.curves, *lines),
    )
    return _draft(
        updated,
        {
            "point_count": float(len(updated.points)),
            "curve_count": float(len(updated.curves)),
        },
    )


def update_planar_point(
    recipe: object,
    *,
    point_id: str,
    x: Real | None = None,
    y: Real | None = None,
) -> GeometryDraft:
    sketch = _as_strict_planar_sketch(recipe)
    if x is None and y is None:
        raise ValueError("update_planar_point requires x or y")
    if str(point_id) not in {point.id for point in sketch.points}:
        raise ValueError("point_id does not identify one editable point")
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        tuple(
            SketchPoint(
                point.id,
                point.u if point.id != str(point_id) or x is None else _finite(x, "x"),
                point.v if point.id != str(point_id) or y is None else _finite(y, "y"),
            )
            for point in sketch.points
        ),
        sketch.curves,
    )
    return _draft(
        updated,
        {
            "point_count": float(len(updated.points)),
            "curve_count": float(len(updated.curves)),
        },
    )


def update_planar_circle(
    recipe: object,
    *,
    circle_id: str,
    center_x: Real | None = None,
    center_y: Real | None = None,
    radius: Real | None = None,
) -> GeometryDraft:
    sketch = _as_strict_planar_sketch(recipe)
    target = next(
        (
            curve
            for curve in sketch.curves
            if isinstance(curve, SketchCircle) and curve.id == str(circle_id)
        ),
        None,
    )
    if target is None or target.center_point_id is None:
        raise ValueError("circle_id does not identify one editable circle")
    points = []
    for point in sketch.points:
        if point.id != target.center_point_id:
            points.append(point)
            continue
        points.append(
            SketchPoint(
                point.id,
                point.u if center_x is None else _finite(center_x, "center_x"),
                point.v if center_y is None else _finite(center_y, "center_y"),
            )
        )
    curves = tuple(
        (
            SketchCircle(
                curve.id,
                curve.center_point_id,
                curve.radius if radius is None else _finite(radius, "radius"),
            )
            if curve is target
            else curve
        )
        for curve in sketch.curves
    )
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        tuple(points),
        curves,
    )
    return _draft(
        updated,
        {
            "point_count": float(len(updated.points)),
            "curve_count": float(len(updated.curves)),
        },
    )


def planar_geometry_catalog(recipe: object) -> dict[str, object]:
    """Return a bounded editable projection without exposing CAD internals."""

    sketch = _as_strict_planar_sketch(recipe)
    point_by_id = {point.id: point for point in sketch.points}
    curves: list[dict[str, object]] = []
    for curve in sketch.curves:
        if isinstance(curve, SketchLine):
            curves.append(
                {
                    "kind": "line",
                    "id": curve.id,
                    "start_point_id": curve.start_point_id,
                    "end_point_id": curve.end_point_id,
                }
            )
        elif isinstance(curve, SketchArc):
            curves.append(
                {
                    "kind": "arc",
                    "id": curve.id,
                    "start_point_id": curve.start_point_id,
                    "center_point_id": curve.center_point_id,
                    "end_point_id": curve.end_point_id,
                    "orientation": curve.orientation,
                }
            )
        elif curve.center_point_id is not None:
            center = point_by_id[curve.center_point_id]
            curves.append(
                {
                    "kind": "circle",
                    "id": curve.id,
                    "center_point_id": curve.center_point_id,
                    "center_x": center.u,
                    "center_y": center.v,
                    "radius": curve.radius,
                }
            )
    return {
        "kind": "planar_sketch",
        "point_count": len(sketch.points),
        "curve_count": len(sketch.curves),
        "points": [
            {"id": point.id, "x": point.u, "y": point.v}
            for point in sketch.points
        ],
        "curves": curves,
    }


def geometry_recipe_to_payload(recipe: object) -> dict[str, object]:
    if type(recipe) is WireGeometry:
        return {
            "kind": "wire",
            "name": recipe.name,
            "points": [
                {
                    "name": point.name,
                    "x": point.x,
                    "y": point.y,
                    "z": point.z,
                }
                for point in recipe.points
            ],
            "members": [
                {
                    "name": member.name,
                    "start": member.start,
                    "end": member.end,
                }
                for member in recipe.members
            ],
        }
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
    if type(recipe) is SketchGeometry and recipe.is_strict:
        return {
            "kind": "planar_sketch",
            "name": recipe.name,
            "plane": {
                "origin": list(recipe.plane.origin),
                "x_direction": list(recipe.plane.x_direction),
                "y_direction": list(recipe.plane.y_direction),
            },
            "points": [
                {"id": point.id, "u": point.u, "v": point.v}
                for point in recipe.points
            ],
            "curves": [
                _sketch_curve_to_payload(curve)
                for curve in recipe.curves
            ],
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
        "planar_sketch": {"kind", "name", "plane", "points", "curves"},
        "wire": {"kind", "name", "points", "members"},
    }
    if kind not in fields or set(value) != fields[kind]:  # type: ignore[index]
        raise ValueError("geometry recipe fields do not match the A2 schema")
    if kind == "wire":
        points = value["points"]
        members = value["members"]
        if (
            not isinstance(points, list)
            or not isinstance(members, list)
            or not 2 <= len(points) <= _MAX_PREVIEW_POINTS
            or not 1 <= len(members) <= _MAX_PREVIEW_POINTS
        ):
            raise ValueError("wire entities exceed the bounded schema")
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "x", "y", "z"}
            for item in points
        ):
            raise ValueError("wire point fields do not match")
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "start", "end"}
            for item in members
        ):
            raise ValueError("wire member fields do not match")
        return WireGeometry(
            value["name"],
            tuple(
                WirePoint(item["name"], item["x"], item["y"], item["z"])
                for item in points
            ),
            tuple(
                WireMember(item["name"], item["start"], item["end"])
                for item in members
            ),
        )
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
    if kind == "planar_sketch":
        plane = value["plane"]
        if not isinstance(plane, Mapping) or set(plane) != {
            "origin",
            "x_direction",
            "y_direction",
        }:
            raise ValueError("planar sketch plane fields do not match")
        points = value["points"]
        curves = value["curves"]
        if (
            not isinstance(points, list)
            or not isinstance(curves, list)
            or not 1 <= len(points) <= 128
            or not 1 <= len(curves) <= 128
        ):
            raise ValueError("planar sketch entities exceed the bounded schema")
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"id", "u", "v"}
            for item in points
        ):
            raise ValueError("planar sketch point fields do not match")
        return SketchGeometry(
            value["name"],
            SketchPlane(
                tuple(plane["origin"]),
                tuple(plane["x_direction"]),
                tuple(plane["y_direction"]),
            ),
            tuple(
                SketchPoint(item["id"], item["u"], item["v"])
                for item in points
            ),
            tuple(_sketch_curve_from_payload(item) for item in curves),
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


def geometry_draft(recipe: object) -> GeometryDraft:
    """Wrap one accepted native recipe for a subsequent generic transform."""

    return _draft(recipe, {})


def _preview(recipe: object) -> StaticGeometryPreview:
    if type(recipe) is MovedGeometry:
        base = _preview(recipe.base)
        points = tuple(
            (x + recipe.dx, y + recipe.dy, z + recipe.dz) for x, y, z in base.points
        )
        return _make_preview(
            base.dimension,
            points,
            base.lines,
            point_names=base.point_names,
            member_names=base.member_names,
        )
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
            base.dimension,
            tuple(map(rotated, base.points)),
            base.lines,
            point_names=base.point_names,
            member_names=base.member_names,
        )
    if type(recipe) is WireGeometry:
        point_indexes = {
            point.name: index for index, point in enumerate(recipe.points)
        }
        return _make_preview(
            1,
            tuple((point.x, point.y, point.z) for point in recipe.points),
            tuple(
                (point_indexes[member.start], point_indexes[member.end])
                for member in recipe.members
            ),
            point_names=tuple(point.name for point in recipe.points),
            member_names=tuple(member.name for member in recipe.members),
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
    if type(recipe) is SketchGeometry:
        sketch = recipe if recipe.is_strict else legacy_sketch_to_strict(recipe)
        assert sketch.plane is not None
        points = tuple(
            sketch.plane.to_global(point.u, point.v)
            for point in sketch.points
        )
        point_indexes = {
            point.id: index for index, point in enumerate(sketch.points)
        }
        preview_points = list(points)
        lines: list[tuple[int, ...]] = []
        point_by_id = {point.id: point for point in sketch.points}
        curved_count = sum(
            isinstance(curve, (SketchCircle, SketchArc))
            for curve in sketch.curves
        )
        curved_samples = (
            0
            if curved_count == 0
            else min(
                _PREVIEW_SEGMENTS,
                max(
                    0,
                    (_MAX_PREVIEW_POINTS - len(preview_points))
                    // curved_count,
                ),
            )
        )
        for curve in sketch.curves:
            if isinstance(curve, SketchLine):
                lines.append(
                    (
                        point_indexes[curve.start_point_id],
                        point_indexes[curve.end_point_id],
                    )
                )
            elif isinstance(curve, SketchCircle):
                if curved_samples < 4:
                    continue
                center = point_by_id[curve.center_point_id]
                circle = tuple(
                    sketch.plane.to_global(
                        center.u
                        + curve.radius
                        * math.cos(2.0 * math.pi * index / curved_samples),
                        center.v
                        + curve.radius
                        * math.sin(2.0 * math.pi * index / curved_samples),
                    )
                    for index in range(curved_samples)
                )
                start = len(preview_points)
                preview_points.extend(circle)
                lines.append(
                    tuple(range(start, start + len(circle))) + (start,)
                )
            elif isinstance(curve, SketchArc):
                if curved_samples < 2:
                    continue
                start_point = point_by_id[curve.start_point_id]
                center = point_by_id[curve.center_point_id]
                end_point = point_by_id[curve.end_point_id]
                start_angle = math.atan2(
                    start_point.v - center.v,
                    start_point.u - center.u,
                )
                end_angle = math.atan2(
                    end_point.v - center.v,
                    end_point.u - center.u,
                )
                sweep = end_angle - start_angle
                if curve.orientation == "ccw" and sweep <= 0.0:
                    sweep += 2.0 * math.pi
                elif curve.orientation == "cw" and sweep >= 0.0:
                    sweep -= 2.0 * math.pi
                radius = math.hypot(
                    start_point.u - center.u,
                    start_point.v - center.v,
                )
                arc = tuple(
                    sketch.plane.to_global(
                        center.u
                        + radius
                        * math.cos(
                            start_angle
                            + sweep * index / (curved_samples - 1)
                        ),
                        center.v
                        + radius
                        * math.sin(
                            start_angle
                            + sweep * index / (curved_samples - 1)
                        ),
                    )
                    for index in range(curved_samples)
                )
                start = len(preview_points)
                preview_points.extend(arc)
                lines.append(tuple(range(start, start + len(arc))))
        return _make_preview(
            2,
            tuple(preview_points),
            tuple(lines),
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
    *,
    point_names: tuple[str, ...] = (),
    member_names: tuple[str, ...] = (),
) -> StaticGeometryPreview:
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    zs = tuple(point[2] for point in points)
    return StaticGeometryPreview(
        dimension=dimension,
        points=points,
        lines=lines,
        bounds=(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)),
        point_names=point_names,
        member_names=member_names,
    )


def _as_strict_planar_sketch(recipe: object) -> SketchGeometry:
    if type(recipe) is SketchGeometry:
        return recipe if recipe.is_strict else legacy_sketch_to_strict(recipe)
    if type(recipe) is RectangleGeometry:
        return planar_sketch_geometry(
            recipe.name,
            contours=(
                SketchRectangle(
                    "material",
                    0.0,
                    0.0,
                    recipe.width,
                    recipe.height,
                ),
            ),
        ).recipe
    if type(recipe) is DiskGeometry:
        return planar_sketch_geometry(
            recipe.name,
            contours=(SketchCircle("material", 0.0, 0.0, recipe.radius),),
        ).recipe
    if type(recipe) is PlateWithHoleGeometry:
        return planar_sketch_geometry(
            recipe.name,
            contours=(
                SketchRectangle(
                    "material",
                    0.0,
                    0.0,
                    recipe.width,
                    recipe.height,
                ),
                SketchCircle(
                    "cut",
                    recipe.hole_x,
                    recipe.hole_y,
                    recipe.hole_radius,
                ),
            ),
        ).recipe
    raise TypeError(
        "incremental planar edits require a planar primitive or sketch"
    )


def _next_sketch_ids(
    sketch: SketchGeometry,
    prefix: str,
    count: int,
) -> tuple[str, ...]:
    used = {
        item.id
        for item in (*sketch.points, *sketch.curves)
        if getattr(item, "id", None) is not None
    }
    index = 1
    allocated: list[str] = []
    while len(allocated) < count:
        candidate = f"{prefix}{index}"
        if candidate not in used:
            allocated.append(candidate)
            used.add(candidate)
        index += 1
    return tuple(allocated)


def _sketch_curve_to_payload(
    curve: SketchLine | SketchArc | SketchCircle,
) -> dict[str, object]:
    if isinstance(curve, SketchLine):
        return {
            "kind": "line",
            "id": curve.id,
            "start_point_id": curve.start_point_id,
            "end_point_id": curve.end_point_id,
        }
    if isinstance(curve, SketchArc):
        return {
            "kind": "arc",
            "id": curve.id,
            "start_point_id": curve.start_point_id,
            "center_point_id": curve.center_point_id,
            "end_point_id": curve.end_point_id,
            "orientation": curve.orientation,
        }
    if not curve.is_curve:
        raise TypeError("planar sketch payload requires strict circles")
    return {
        "kind": "circle",
        "id": curve.id,
        "center_point_id": curve.center_point_id,
        "radius": curve.radius,
    }


def _sketch_curve_from_payload(
    value: object,
) -> SketchLine | SketchArc | SketchCircle:
    if not isinstance(value, Mapping):
        raise TypeError("sketch curve payload must be an object")
    kind = value.get("kind")
    if kind == "line" and set(value) == {
        "kind",
        "id",
        "start_point_id",
        "end_point_id",
    }:
        return SketchLine(
            value["id"],
            value["start_point_id"],
            value["end_point_id"],
        )
    if kind == "arc" and set(value) == {
        "kind",
        "id",
        "start_point_id",
        "center_point_id",
        "end_point_id",
        "orientation",
    }:
        return SketchArc(
            value["id"],
            value["start_point_id"],
            value["center_point_id"],
            value["end_point_id"],
            value["orientation"],
        )
    if kind == "circle" and set(value) == {
        "kind",
        "id",
        "center_point_id",
        "radius",
    }:
        return SketchCircle(
            value["id"],
            value["center_point_id"],
            value["radius"],
        )
    raise ValueError("sketch curve fields do not match the bounded schema")


__all__ = [
    "GeometryDraft",
    "StaticGeometryPreview",
    "box_geometry",
    "add_planar_circle",
    "add_planar_polygon",
    "add_planar_rectangle",
    "create_geometry_edit_proposal",
    "create_geometry_proposal",
    "cylinder_geometry",
    "disk_geometry",
    "geometry_recipe_from_payload",
    "geometry_recipe_to_payload",
    "geometry_draft",
    "planar_geometry_catalog",
    "planar_polygon_geometry",
    "planar_sketch_geometry",
    "rectangle_geometry",
    "rotate_geometry",
    "translate_geometry",
    "update_planar_circle",
    "update_planar_point",
    "wire_geometry",
]
