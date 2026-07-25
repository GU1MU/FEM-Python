"""Compile native recipes into one-session CAD topology mappings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from fem.geometry.recipe_topology import (
    LogicalEntity,
    RecipeTopology,
    describe_recipe_topology,
)
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    NativeGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchContour,
    SketchGeometry,
    SketchRectangle,
    geometry_dimension,
)


class TopologyResolutionError(ValueError):
    """A logical authoring entity cannot be proven in the current CAD model."""


@dataclass(frozen=True, slots=True)
class CompiledRecipeTopology:
    """CAD domain and logical references valid in one GeometryModel session."""

    domain: tuple[Any, ...]
    boundary: tuple[Any, ...]
    catalog: RecipeTopology
    logical_entities: Mapping[str, tuple[Any, ...]]
    groups: Mapping[str, tuple[Any, ...]]
    hole_boundary: tuple[Any, ...] = ()

    def resolve(self, entity_kind: str, entity_id: int) -> tuple[Any, ...]:
        """Resolve one persisted one-based ID without consulting backend tags."""
        kind = entity_kind
        try:
            logical = self.catalog.logical_entity(kind, entity_id)
        except (KeyError, TypeError, ValueError) as error:
            if not self.catalog.exact and self.catalog.diagnostics:
                raise TopologyResolutionError(
                    f"几何{_kind_label(kind)}编号 {entity_id!r} "
                    f"不可用于建模：{self.catalog.diagnostics[0].message}"
                ) from error
            raise TopologyResolutionError(
                f"几何{_kind_label(kind)}编号 {entity_id!r} 已失效，请重新选择"
            ) from error
        if not logical.selectable:
            diagnostic = next(
                (
                    item.message
                    for item in self.catalog.diagnostics
                    if item.code == logical.diagnostic_code
                ),
                "当前几何操作无法证明该实体的拓扑身份",
            )
            raise TopologyResolutionError(
                f"几何{_kind_label(kind)}编号 {entity_id} 不可用于建模：{diagnostic}"
            )
        entities = tuple(self.logical_entities.get(logical.logical_id, ()))
        if not entities:
            raise TopologyResolutionError(
                f"几何{_kind_label(kind)}编号 {entity_id} 无法解析，请重新选择"
            )
        return entities


@dataclass(slots=True)
class _CompiledDraft:
    domain: tuple[Any, ...]
    logical_entities: dict[str, tuple[Any, ...]]
    groups: dict[str, tuple[Any, ...]]
    hole_boundary: tuple[Any, ...] = ()


def compile_recipe(
    cad: Any,
    recipe: NativeGeometry,
) -> CompiledRecipeTopology:
    """Build ``recipe`` and prove all selectable catalog entities."""
    catalog = describe_recipe_topology(recipe)
    if catalog.exact:
        draft = _compile_exact(cad, recipe)
    else:
        draft = _CompiledDraft(
            tuple(_build_domain_only(cad, recipe)),
            {},
            {},
        )
    return _finalize(cad, recipe, catalog, draft)


def _compile_exact(cad: Any, recipe: NativeGeometry) -> _CompiledDraft:
    if isinstance(recipe, SketchGeometry):
        return _compile_exact(cad, _compile_sketch(recipe))
    if isinstance(recipe, RectangleGeometry):
        return _compile_rectangle(cad, recipe)
    if isinstance(recipe, DiskGeometry):
        return _compile_disk(cad, recipe)
    if isinstance(recipe, PlateWithHoleGeometry):
        return _compile_plate_with_hole(cad, recipe)
    if isinstance(recipe, BoxGeometry):
        return _compile_box(cad, recipe)
    if isinstance(recipe, CylinderGeometry):
        return _compile_cylinder(cad, recipe)
    if isinstance(recipe, MovedGeometry):
        draft = _compile_exact(cad, recipe.base)
        draft.domain = tuple(
            cad.translate(draft.domain, recipe.dx, recipe.dy, recipe.dz)
        )
        return draft
    if isinstance(recipe, RotatedGeometry):
        draft = _compile_exact(cad, recipe.base)
        axis = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }[recipe.axis]
        draft.domain = tuple(
            cad.rotate(
                draft.domain,
                0.0,
                0.0,
                0.0,
                *axis,
                math.radians(recipe.angle_degrees),
            )
        )
        return draft
    if isinstance(recipe, ExtrudedGeometry):
        return _compile_extrusion(cad, recipe)
    if isinstance(recipe, BooleanGeometry):
        return _compile_boolean(cad, recipe)
    raise TypeError(f"不支持的几何配方: {type(recipe).__name__}")


def _compile_rectangle(cad: Any, recipe: RectangleGeometry) -> _CompiledDraft:
    domain = (cad.rectangle(0.0, 0.0, recipe.width, recipe.height),)
    boundary = tuple(cad.boundary(domain))
    edges = _rectangle_edges(cad, boundary, 0.0, 0.0, recipe.width, recipe.height)
    points = _boundary_of(cad, edges)
    logical = {
        "point:bottom-left": _select_one(cad, points, x=0.0, y=0.0),
        "point:bottom-right": _select_one(
            cad,
            points,
            x=recipe.width,
            y=0.0,
        ),
        "point:top-right": _select_one(
            cad,
            points,
            x=recipe.width,
            y=recipe.height,
        ),
        "point:top-left": _select_one(
            cad,
            points,
            x=0.0,
            y=recipe.height,
        ),
        "edge:bottom": edges[0],
        "edge:right": edges[1],
        "edge:top": edges[2],
        "edge:left": edges[3],
        "face:domain": domain,
        "body:domain": domain,
    }
    groups = {
        "BOTTOM": edges[0],
        "RIGHT": edges[1],
        "TOP": edges[2],
        "LEFT": edges[3],
    }
    return _CompiledDraft(domain, logical, groups)


def _compile_disk(cad: Any, recipe: DiskGeometry) -> _CompiledDraft:
    domain = (cad.disk(0.0, 0.0, recipe.radius),)
    outer = tuple(cad.boundary(domain))
    if not outer or any(entity.dimension != 1 for entity in outer):
        raise TopologyResolutionError("圆盘外边界识别失败")
    return _CompiledDraft(
        domain,
        {
            "edge:outer": outer,
            "face:domain": domain,
            "body:domain": domain,
        },
        {"OUTER": outer},
    )


def _compile_plate_with_hole(
    cad: Any,
    recipe: PlateWithHoleGeometry,
) -> _CompiledDraft:
    plate = cad.rectangle(0.0, 0.0, recipe.width, recipe.height)
    hole = cad.disk(recipe.hole_x, recipe.hole_y, recipe.hole_radius)
    domain = tuple(cad.cut((plate,), (hole,)).of_dimension(2))
    if len(domain) != 1:
        raise TopologyResolutionError("带孔板布尔切除没有生成唯一平面域")
    boundary = tuple(cad.boundary(domain))
    outer = _rectangle_edges(
        cad,
        boundary,
        0.0,
        0.0,
        recipe.width,
        recipe.height,
    )
    outer_set = set(_flatten(outer))
    hole_boundary = tuple(entity for entity in boundary if entity not in outer_set)
    if not hole_boundary:
        raise TopologyResolutionError("带孔板圆孔边界识别失败")
    points = _boundary_of(cad, outer)
    logical = {
        "point:bottom-left": _select_one(cad, points, x=0.0, y=0.0),
        "point:bottom-right": _select_one(
            cad,
            points,
            x=recipe.width,
            y=0.0,
        ),
        "point:top-right": _select_one(
            cad,
            points,
            x=recipe.width,
            y=recipe.height,
        ),
        "point:top-left": _select_one(
            cad,
            points,
            x=0.0,
            y=recipe.height,
        ),
        "edge:hole-loop": hole_boundary,
        "edge:outer-loop": _flatten(outer),
        "face:domain": domain,
        "body:domain": domain,
    }
    groups = {
        "BOTTOM": outer[0],
        "RIGHT": outer[1],
        "TOP": outer[2],
        "LEFT": outer[3],
    }
    return _CompiledDraft(domain, logical, groups, hole_boundary)


def _compile_box(cad: Any, recipe: BoxGeometry) -> _CompiledDraft:
    domain = (
        cad.box(
            0.0,
            0.0,
            0.0,
            recipe.width,
            recipe.depth,
            recipe.height,
        ),
    )
    faces = tuple(cad.boundary(domain))
    bottom = _select_one(cad, faces, z=0.0)
    top = _select_one(cad, faces, z=recipe.height)
    front = _select_one(cad, faces, y=0.0)
    right = _select_one(cad, faces, x=recipe.width)
    back = _select_one(cad, faces, y=recipe.depth)
    left = _select_one(cad, faces, x=0.0)
    curves = _boundary_of(cad, (bottom, top, front, right, back, left))
    points = _boundary_of(cad, tuple((curve,) for curve in curves))

    edge_selectors = (
        {"y": 0.0, "z": 0.0},
        {"x": recipe.width, "z": 0.0},
        {"y": recipe.depth, "z": 0.0},
        {"x": 0.0, "z": 0.0},
        {"y": 0.0, "z": recipe.height},
        {"x": recipe.width, "z": recipe.height},
        {"y": recipe.depth, "z": recipe.height},
        {"x": 0.0, "z": recipe.height},
        {"x": 0.0, "y": 0.0},
        {"x": recipe.width, "y": 0.0},
        {"x": recipe.width, "y": recipe.depth},
        {"x": 0.0, "y": recipe.depth},
    )
    edge_names = (
        "bottom-front",
        "bottom-right",
        "bottom-back",
        "bottom-left",
        "top-front",
        "top-right",
        "top-back",
        "top-left",
        "vertical-front-left",
        "vertical-front-right",
        "vertical-back-right",
        "vertical-back-left",
    )
    point_coordinates = (
        (0.0, 0.0, 0.0),
        (recipe.width, 0.0, 0.0),
        (recipe.width, recipe.depth, 0.0),
        (0.0, recipe.depth, 0.0),
        (0.0, 0.0, recipe.height),
        (recipe.width, 0.0, recipe.height),
        (recipe.width, recipe.depth, recipe.height),
        (0.0, recipe.depth, recipe.height),
    )
    point_names = (
        "bottom-front-left",
        "bottom-front-right",
        "bottom-back-right",
        "bottom-back-left",
        "top-front-left",
        "top-front-right",
        "top-back-right",
        "top-back-left",
    )
    logical: dict[str, tuple[Any, ...]] = {
        f"edge:{name}": _select_one(cad, curves, **selector)
        for name, selector in zip(edge_names, edge_selectors, strict=True)
    }
    logical.update(
        {
            f"point:{name}": _select_one(
                cad,
                points,
                x=coordinates[0],
                y=coordinates[1],
                z=coordinates[2],
            )
            for name, coordinates in zip(
                point_names,
                point_coordinates,
                strict=True,
            )
        }
    )
    logical.update(
        {
            "face:bottom": bottom,
            "face:top": top,
            "face:front": front,
            "face:right": right,
            "face:back": back,
            "face:left": left,
            "body:domain": domain,
        }
    )
    groups = {
        "BOTTOM": bottom,
        "TOP": top,
        "FRONT": front,
        "RIGHT": right,
        "BACK": back,
        "LEFT": left,
    }
    return _CompiledDraft(domain, logical, groups)


def _compile_cylinder(cad: Any, recipe: CylinderGeometry) -> _CompiledDraft:
    domain = (
        cad.cylinder(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            recipe.height,
            recipe.radius,
        ),
    )
    faces = tuple(cad.boundary(domain))
    bottom = _select_one(cad, faces, z=0.0)
    top = _select_one(cad, faces, z=recipe.height)
    cap_set = {*bottom, *top}
    outer = tuple(entity for entity in faces if entity not in cap_set)
    if len(outer) != 1:
        raise TopologyResolutionError("圆柱外侧面识别失败")
    bottom_rim = tuple(cad.boundary(bottom))
    top_rim = tuple(cad.boundary(top))
    if not bottom_rim or not top_rim:
        raise TopologyResolutionError("圆柱端部圆周识别失败")
    return _CompiledDraft(
        domain,
        {
            "edge:bottom-rim": bottom_rim,
            "edge:top-rim": top_rim,
            "face:bottom": bottom,
            "face:top": top,
            "face:outer": outer,
            "body:domain": domain,
        },
        {
            "BOTTOM": bottom,
            "TOP": top,
            "OUTER": outer,
        },
    )


def _compile_boolean(cad: Any, recipe: BooleanGeometry) -> _CompiledDraft:
    objects = tuple(_build_domain_only(cad, recipe.object_geometry))
    tools = tuple(_build_domain_only(cad, recipe.tool_geometry))
    operation = {
        "fuse": cad.fuse,
        "cut": cad.cut,
        "fragment": cad.fragment,
    }[recipe.operation]
    domain = tuple(operation(objects, tools).of_dimension(geometry_dimension(recipe)))
    if not domain:
        raise TopologyResolutionError("布尔操作没有生成有效几何")
    catalog = describe_recipe_topology(recipe)
    if not catalog.exact:
        return _CompiledDraft(domain, {}, {})
    outer = _axis_aligned_rectangle(recipe.object_geometry)
    if outer is None:
        raise TopologyResolutionError("无法证明布尔结果的外轮廓")
    x, y, width, height = outer
    boundary = tuple(cad.boundary(domain))
    outer_edges = _rectangle_edges(cad, boundary, x, y, width, height)
    outer_set = set(_flatten(outer_edges))
    hole_edges = tuple(entity for entity in boundary if entity not in outer_set)
    if not hole_edges:
        raise TopologyResolutionError("无法证明布尔切除的内轮廓")
    outer_points = _boundary_of(cad, outer_edges)
    logical: dict[str, tuple[Any, ...]] = {
        "point:bottom-left": _select_one(cad, outer_points, x=x, y=y),
        "point:bottom-right": _select_one(
            cad,
            outer_points,
            x=x + width,
            y=y,
        ),
        "point:top-right": _select_one(
            cad,
            outer_points,
            x=x + width,
            y=y + height,
        ),
        "point:top-left": _select_one(
            cad,
            outer_points,
            x=x,
            y=y + height,
        ),
        "edge:hole-loop": hole_edges,
        "edge:outer-loop": _flatten(outer_edges),
        "face:domain": domain,
        "body:domain": domain,
    }
    inner = _axis_aligned_rectangle(recipe.tool_geometry)
    if inner is not None:
        inner_x, inner_y, inner_width, inner_height = inner
        inner_points = _boundary_of(cad, (hole_edges,))
        logical.update(
            {
                "point:hole-bottom-left": _select_one(
                    cad,
                    inner_points,
                    x=inner_x,
                    y=inner_y,
                ),
                "point:hole-bottom-right": _select_one(
                    cad,
                    inner_points,
                    x=inner_x + inner_width,
                    y=inner_y,
                ),
                "point:hole-top-right": _select_one(
                    cad,
                    inner_points,
                    x=inner_x + inner_width,
                    y=inner_y + inner_height,
                ),
                "point:hole-top-left": _select_one(
                    cad,
                    inner_points,
                    x=inner_x,
                    y=inner_y + inner_height,
                ),
            }
        )
    return _CompiledDraft(
        domain,
        logical,
        {
            "BOTTOM": outer_edges[0],
            "RIGHT": outer_edges[1],
            "TOP": outer_edges[2],
            "LEFT": outer_edges[3],
        },
        hole_edges,
    )


def _compile_extrusion(cad: Any, recipe: ExtrudedGeometry) -> _CompiledDraft:
    base = _compile_exact(cad, recipe.base)
    base_catalog = describe_recipe_topology(recipe.base)
    feature = cad.extrude(base.domain, 0.0, 0.0, recipe.height)
    domain = tuple(feature.primary)
    bottom = tuple(feature.inputs)
    top = tuple(feature.ends)
    sides = tuple(feature.sides)
    if len(domain) != 1 or len(bottom) != 1 or len(top) != 1 or not sides:
        raise TopologyResolutionError("拉伸特征没有生成可证明的端面和侧面")

    side_boundaries = {
        side: set(cad.boundary((side,), combined=False)) for side in sides
    }
    top_curves = set(cad.boundary(top, combined=False))
    bottom_curves = set(cad.boundary(bottom, combined=False))
    logical: dict[str, tuple[Any, ...]] = {
        "face:bottom": bottom,
        "face:top": top,
        "body:domain": domain,
    }
    for base_edge in base_catalog.entities_of("edge", selectable_only=True):
        source_curves = tuple(base.logical_entities[base_edge.logical_id])
        resolved_sides: list[Any] = []
        resolved_top: list[Any] = []
        for source_curve in source_curves:
            matches = tuple(
                side
                for side, boundaries in side_boundaries.items()
                if source_curve in boundaries
            )
            if len(matches) != 1:
                raise TopologyResolutionError(
                    f"拉伸无法唯一追踪源边 {base_edge.logical_id}"
                )
            side = matches[0]
            top_matches = tuple(side_boundaries[side] & top_curves)
            if len(top_matches) != 1:
                raise TopologyResolutionError(
                    f"拉伸无法唯一追踪顶边 {base_edge.logical_id}"
                )
            resolved_sides.append(side)
            resolved_top.append(top_matches[0])
        name = _logical_name(base_edge)
        logical[f"edge:bottom/{name}"] = source_curves
        logical[f"edge:top/{name}"] = _unique(resolved_top)
        logical[f"face:side/{name}"] = _unique(resolved_sides)

    all_side_curves = _unique(
        entity
        for boundaries in side_boundaries.values()
        for entity in boundaries
        if entity.dimension == 1
    )
    vertical_curves = tuple(
        entity
        for entity in all_side_curves
        if entity not in bottom_curves and entity not in top_curves
    )
    top_points = set(_boundary_of(cad, tuple((curve,) for curve in top_curves)))
    vertical_endpoints = {
        curve: set(cad.boundary((curve,), combined=False)) for curve in vertical_curves
    }
    for base_point in base_catalog.entities_of("point", selectable_only=True):
        source_points = tuple(base.logical_entities[base_point.logical_id])
        if len(source_points) != 1:
            raise TopologyResolutionError(
                f"拉伸源点 {base_point.logical_id} 不是唯一实体"
            )
        source_point = source_points[0]
        vertical = tuple(
            curve
            for curve, endpoints in vertical_endpoints.items()
            if source_point in endpoints
        )
        if len(vertical) != 1:
            raise TopologyResolutionError(
                f"拉伸无法唯一追踪竖边 {base_point.logical_id}"
            )
        top_point = tuple(vertical_endpoints[vertical[0]] & top_points)
        if len(top_point) != 1:
            raise TopologyResolutionError(
                f"拉伸无法唯一追踪顶点 {base_point.logical_id}"
            )
        name = _logical_name(base_point)
        logical[f"point:bottom/{name}"] = (source_point,)
        logical[f"point:top/{name}"] = top_point
        logical[f"edge:vertical/{name}"] = vertical

    hole_sources = set(base.hole_boundary)
    hole_sides = _unique(
        side
        for side, boundaries in side_boundaries.items()
        if boundaries & hole_sources
    )
    outer_sides = _unique(side for side in sides if side not in set(hole_sides))
    groups = {
        "BOTTOM": bottom,
        "TOP": top,
        "OUTER": outer_sides,
    }
    if hole_sides:
        groups["HOLE"] = hole_sides
    return _CompiledDraft(domain, logical, groups, hole_sides)


def _finalize(
    cad: Any,
    recipe: NativeGeometry,
    catalog: RecipeTopology,
    draft: _CompiledDraft,
) -> CompiledRecipeTopology:
    if not draft.domain:
        raise TopologyResolutionError("几何配方没有生成有效计算域")
    boundary = tuple(cad.boundary(draft.domain))
    selectable = catalog.selectable_entities()
    logical: dict[str, tuple[Any, ...]] = {}
    for item in selectable:
        entities = tuple(draft.logical_entities.get(item.logical_id, ()))
        _validate_logical_entities(cad, item, entities)
        logical[item.logical_id] = entities
    return CompiledRecipeTopology(
        tuple(draft.domain),
        boundary,
        catalog,
        logical,
        {
            name: _unique(entities)
            for name, entities in draft.groups.items()
            if entities
        },
        _unique(draft.hole_boundary),
    )


def _validate_logical_entities(
    cad: Any,
    logical: LogicalEntity,
    entities: tuple[Any, ...],
) -> None:
    if not entities:
        raise TopologyResolutionError(
            f"逻辑实体 {logical.logical_id} 没有对应 CAD 实体"
        )
    if any(entity.dimension != logical.dimension for entity in entities):
        raise TopologyResolutionError(
            f"逻辑实体 {logical.logical_id} 的 CAD 维度不匹配"
        )
    for entity in entities:
        cad.bounding_box(entity)


def _build_domain_only(cad: Any, recipe: NativeGeometry) -> tuple[Any, ...]:
    if isinstance(recipe, SketchGeometry):
        return _build_domain_only(cad, _compile_sketch(recipe))
    if isinstance(recipe, BooleanGeometry):
        objects = _build_domain_only(cad, recipe.object_geometry)
        tools = _build_domain_only(cad, recipe.tool_geometry)
        operation = {
            "fuse": cad.fuse,
            "cut": cad.cut,
            "fragment": cad.fragment,
        }[recipe.operation]
        result = tuple(
            operation(objects, tools).of_dimension(geometry_dimension(recipe))
        )
        if not result:
            raise TopologyResolutionError("布尔操作没有生成有效几何")
        return result
    if isinstance(recipe, PlateWithHoleGeometry):
        plate = cad.rectangle(0.0, 0.0, recipe.width, recipe.height)
        hole = cad.disk(recipe.hole_x, recipe.hole_y, recipe.hole_radius)
        return tuple(cad.cut((plate,), (hole,)).of_dimension(2))
    if isinstance(recipe, RectangleGeometry):
        return (cad.rectangle(0.0, 0.0, recipe.width, recipe.height),)
    if isinstance(recipe, DiskGeometry):
        return (cad.disk(0.0, 0.0, recipe.radius),)
    if isinstance(recipe, BoxGeometry):
        return (
            cad.box(
                0.0,
                0.0,
                0.0,
                recipe.width,
                recipe.depth,
                recipe.height,
            ),
        )
    if isinstance(recipe, CylinderGeometry):
        return (
            cad.cylinder(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                recipe.height,
                recipe.radius,
            ),
        )
    if isinstance(recipe, MovedGeometry):
        return tuple(
            cad.translate(
                _build_domain_only(cad, recipe.base),
                recipe.dx,
                recipe.dy,
                recipe.dz,
            )
        )
    if isinstance(recipe, RotatedGeometry):
        axis = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }[recipe.axis]
        return tuple(
            cad.rotate(
                _build_domain_only(cad, recipe.base),
                0.0,
                0.0,
                0.0,
                *axis,
                math.radians(recipe.angle_degrees),
            )
        )
    if isinstance(recipe, ExtrudedGeometry):
        feature = cad.extrude(
            _build_domain_only(cad, recipe.base),
            0.0,
            0.0,
            recipe.height,
        )
        return tuple(feature.primary)
    raise TypeError(f"不支持的几何配方: {type(recipe).__name__}")


def _compile_sketch(recipe: SketchGeometry) -> NativeGeometry:
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
        else:  # pragma: no cover - recipe validation owns contour types
            raise TypeError(f"不支持的草图轮廓: {type(contour).__name__}")
        if contour.x != 0.0 or contour.y != 0.0:
            result = MovedGeometry(result, contour.x, contour.y)
        return result

    material = [
        contour_geometry(contour, index)
        for index, contour in enumerate(recipe.contours, start=1)
        if contour.operation == "material"
    ]
    cuts = [
        contour_geometry(contour, index)
        for index, contour in enumerate(recipe.contours, start=1)
        if contour.operation == "cut"
    ]
    result = material[0]
    for tool in material[1:]:
        result = BooleanGeometry(recipe.name, "fuse", result, tool)
    for tool in cuts:
        result = BooleanGeometry(recipe.name, "cut", result, tool)
    return result


def _rectangle_edges(
    cad: Any,
    candidates: tuple[Any, ...],
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[tuple[Any, ...], ...]:
    return (
        _select_one(cad, candidates, y=y),
        _select_one(cad, candidates, x=x + width),
        _select_one(cad, candidates, y=y + height),
        _select_one(cad, candidates, x=x),
    )


def _select_one(cad: Any, entities: tuple[Any, ...], **coordinates: float):
    selected = tuple(cad.select(entities, **coordinates))
    if len(selected) != 1:
        description = ", ".join(
            f"{axis}={value:g}" for axis, value in coordinates.items()
        )
        raise TopologyResolutionError(
            f"几何实体选择 {description} 得到 {len(selected)} 个候选，要求唯一"
        )
    return selected


def _boundary_of(
    cad: Any,
    entity_groups: tuple[tuple[Any, ...], ...],
) -> tuple[Any, ...]:
    entities = _flatten(entity_groups)
    if not entities:
        return ()
    return _unique(cad.boundary(entities, combined=False))


def _flatten(groups: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
    return _unique(entity for group in groups for entity in group)


def _unique(entities) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(entities))


def _logical_name(entity: LogicalEntity) -> str:
    return entity.logical_id.split(":", 1)[1]


def _axis_aligned_rectangle(
    recipe: NativeGeometry,
) -> tuple[float, float, float, float] | None:
    if isinstance(recipe, RectangleGeometry):
        return 0.0, 0.0, recipe.width, recipe.height
    if isinstance(recipe, MovedGeometry):
        frame = _axis_aligned_rectangle(recipe.base)
        if frame is None or recipe.dz != 0.0:
            return None
        x, y, width, height = frame
        return x + recipe.dx, y + recipe.dy, width, height
    if isinstance(recipe, RotatedGeometry) and math.isclose(
        recipe.angle_degrees % 360.0,
        0.0,
        abs_tol=1.0e-12,
    ):
        return _axis_aligned_rectangle(recipe.base)
    return None


def _kind_label(kind: str) -> str:
    return {
        "point": "点",
        "edge": "边",
        "face": "面",
        "body": "体",
    }.get(kind, "实体")


__all__ = [
    "CompiledRecipeTopology",
    "TopologyResolutionError",
    "compile_recipe",
]
