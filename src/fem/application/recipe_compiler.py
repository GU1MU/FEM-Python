"""Compile native recipes into one-session CAD topology mappings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from fem.geometry.recipe_analysis import (
    analyze_sketch_profiles,
    axis_aligned_rectangle,
    expand_sketch_recipe,
)
from fem.geometry.extrusion_selection import (
    ExtrusionSourceResolutionError,
    extrusion_face_boundary_ids,
    resolve_extrusion_source_faces,
)
from fem.geometry.planar_boolean_lineage import (
    PlanarBooleanLineageResolutionError,
    capture_planar_operand_evidence,
    resolve_planar_boolean_lineage,
    validate_planar_boolean_input_map,
)
from fem.geometry.planar_boolean_selection import (
    PlanarBooleanSelectionError,
    resolve_planar_boolean_faces,
)
from fem.geometry.recipe_topology import (
    LogicalEntity,
    RecipeTopology,
    describe_recipe_topology,
)
from fem.geometry.references import LogicalEntityRef
from fem.geometry.solid_boolean_lineage import (
    BooleanLineageResolutionError,
    capture_boolean_operand_evidence,
    resolve_solid_boolean_lineage,
    validate_solid_boolean_input_map,
)
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    NativeGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchArc,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    WireGeometry,
    WirePoint,
    geometry_dimension,
)

from .native_regions import RecipeRegionSelector


class TopologyResolutionError(ValueError):
    """A logical authoring entity cannot be proven in the current CAD model."""


@dataclass(frozen=True, slots=True)
class CompiledRecipeTopology:
    """CAD domain and logical references valid in one GeometryModel session."""

    domain: tuple[Any, ...]
    boundary: tuple[Any, ...]
    catalog: RecipeTopology
    logical_entities: Mapping[str, tuple[Any, ...]]
    region_bindings: Mapping[RecipeRegionSelector, tuple[Any, ...]]

    def resolve(self, reference: LogicalEntityRef) -> tuple[Any, ...]:
        """Resolve one stable logical reference without consulting backend tags."""
        if type(reference) is not LogicalEntityRef:
            raise TypeError("reference must be a LogicalEntityRef")
        try:
            logical = self.catalog.entity(reference.logical_id)
        except KeyError as error:
            if not self.catalog.exact and self.catalog.diagnostics:
                raise TopologyResolutionError(
                    f"逻辑实体 {reference.logical_id!r} "
                    f"不可用于建模：{self.catalog.diagnostics[0].message}"
                ) from error
            raise TopologyResolutionError(
                f"逻辑实体 {reference.logical_id!r} 已失效，请重新选择"
            ) from error
        if logical.kind != reference.kind:
            raise TopologyResolutionError(
                f"逻辑实体 {reference.logical_id!r} 的类型不匹配"
            )
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
                f"逻辑实体 {reference.logical_id!r} 不可用于建模：{diagnostic}"
            )
        entities = tuple(self.logical_entities.get(logical.logical_id, ()))
        if not entities:
            raise TopologyResolutionError(
                f"逻辑实体 {reference.logical_id!r} 无法解析，请重新选择"
            )
        return entities


@dataclass(slots=True)
class _CompiledDraft:
    domain: tuple[Any, ...]
    logical_entities: dict[str, tuple[Any, ...]]
    region_bindings: dict[RecipeRegionSelector, tuple[Any, ...]]
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
    transformed_wire = _transformed_wire_recipe(recipe)
    if transformed_wire is not None:
        return _compile_wire(cad, transformed_wire)
    if isinstance(recipe, SketchGeometry):
        if recipe.is_strict:
            return _compile_strict_sketch(cad, recipe)
        return _compile_exact(cad, expand_sketch_recipe(recipe))
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
    if isinstance(recipe, MultiBodyGeometry):
        return _compile_multi_body(cad, recipe)
    if isinstance(recipe, BooleanGeometry):
        return _compile_boolean(cad, recipe)
    raise TypeError(f"不支持的几何配方: {type(recipe).__name__}")


def _transformed_wire_recipe(recipe: object) -> WireGeometry | None:
    if isinstance(recipe, WireGeometry):
        return recipe
    if isinstance(recipe, MovedGeometry):
        wire = _transformed_wire_recipe(recipe.base)
        if wire is None:
            return None
        return WireGeometry(
            wire.name,
            tuple(
                WirePoint(
                    point.name,
                    point.x + recipe.dx,
                    point.y + recipe.dy,
                    point.z + recipe.dz,
                )
                for point in wire.points
            ),
            wire.members,
        )
    if isinstance(recipe, RotatedGeometry):
        wire = _transformed_wire_recipe(recipe.base)
        if wire is None:
            return None
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)

        def rotate(point: WirePoint) -> tuple[float, float, float]:
            x, y, z = point.x, point.y, point.z
            if recipe.axis == "x":
                return x, y * cosine - z * sine, y * sine + z * cosine
            if recipe.axis == "y":
                return x * cosine + z * sine, y, -x * sine + z * cosine
            return x * cosine - y * sine, x * sine + y * cosine, z

        return WireGeometry(
            wire.name,
            tuple(
                WirePoint(point.name, *rotate(point))
                for point in wire.points
            ),
            wire.members,
        )
    return None


def _compile_wire(cad: Any, recipe: WireGeometry) -> _CompiledDraft:
    """Compile the declared wire graph without inferring any extra topology."""

    point_refs: dict[str, Any] = {}
    for point in recipe.points:
        try:
            point_refs[point.name] = cad.point(point.x, point.y, point.z)
        except (TypeError, ValueError) as error:
            raise TopologyResolutionError(
                f"无法创建 point:{point.name}：{error}"
            ) from error

    member_refs: dict[str, Any] = {}
    for member in recipe.members:
        try:
            member_refs[member.name] = cad.line(
                point_refs[member.start],
                point_refs[member.end],
            )
        except (TypeError, ValueError) as error:
            raise TopologyResolutionError(
                f"无法创建 edge:{member.name}，端点 "
                f"{member.start!r} 和 {member.end!r}：{error}"
            ) from error

    return _CompiledDraft(
        tuple(member_refs.values()),
        {
            **{
                f"point:{point.name}": (point_refs[point.name],)
                for point in recipe.points
            },
            **{
                f"edge:{member.name}": (member_refs[member.name],)
                for member in recipe.members
            },
            "body:domain": tuple(member_refs.values()),
        },
        {},
    )


def _compile_strict_sketch(
    cad: Any,
    recipe: SketchGeometry,
) -> _CompiledDraft:
    """Compile one validated curve-first sketch without primitive expansion."""

    if not recipe.is_strict or recipe.plane is None:
        raise TypeError("strict sketch compilation requires a curve-first sketch")
    analysis = analyze_sketch_profiles(recipe)
    if analysis.blocking_diagnostics or not analysis.profiles:
        message = (
            analysis.blocking_diagnostics[0].message
            if analysis.blocking_diagnostics
            else "严格草图没有可构建的 Profile"
        )
        raise TopologyResolutionError(message)

    point_refs: dict[str, Any] = {}
    for point in recipe.points:
        try:
            point_refs[point.id] = cad.point(
                *recipe.plane.to_global(point.u, point.v)
            )
        except (TypeError, ValueError) as error:
            raise TopologyResolutionError(
                f"无法创建 point:{point.id}：{error}"
            ) from error

    # Each semantic curve can compile to multiple OCC curves.  The boolean
    # records whether the OCC curve must be reversed to traverse it in the
    # semantic curve's forward direction.
    curve_pieces: dict[str, tuple[tuple[Any, bool], ...]] = {}
    for curve in recipe.curves:
        try:
            curve_pieces[curve.id] = _compile_strict_sketch_curve(
                cad,
                recipe,
                curve,
                point_refs,
            )
        except (TypeError, ValueError) as error:
            raise TopologyResolutionError(
                f"无法创建 edge:{curve.id}：{error}"
            ) from error

    loop_refs: dict[str, Any] = {}
    for profile in analysis.profiles:
        oriented = []
        for signed_curve_id in profile.curve_ids:
            reversed_curve = signed_curve_id.startswith("-")
            curve_id = signed_curve_id.lstrip("-")
            pieces = curve_pieces[curve_id]
            if reversed_curve:
                pieces = tuple(
                    (entity, not forward_reversed)
                    for entity, forward_reversed in reversed(pieces)
                )
            oriented.extend(
                cad.orient(entity, reversed=forward_reversed)
                for entity, forward_reversed in pieces
            )
        try:
            loop_refs[profile.id] = cad.curve_loop(tuple(oriented))
        except (TypeError, ValueError) as error:
            raise TopologyResolutionError(
                f"无法创建 Profile {profile.id!r} 的闭合曲线环：{error}"
            ) from error

    material_profiles = tuple(
        profile for profile in analysis.profiles if profile.is_material
    )
    hole_profiles = tuple(
        profile for profile in analysis.profiles if profile.is_hole
    )
    surfaces: dict[str, Any] = {}
    for profile in material_profiles:
        holes = tuple(
            loop_refs[hole.id]
            for hole in hole_profiles
            if hole.parent_profile_id == profile.id
        )
        try:
            surfaces[profile.id] = cad.plane_surface(
                loop_refs[profile.id],
                holes=holes,
            )
        except (TypeError, ValueError) as error:
            raise TopologyResolutionError(
                f"无法创建 face:profile/{profile.id.split('/', 1)[-1]}：{error}"
            ) from error

    domain = tuple(surfaces[profile.id] for profile in material_profiles)
    logical: dict[str, tuple[Any, ...]] = {
        **{
            f"point:{point.id}": (point_refs[point.id],)
            for point in recipe.points
        },
        **{
            f"edge:{curve.id}": tuple(
                entity for entity, _reversed in curve_pieces[curve.id]
            )
            for curve in recipe.curves
        },
        **{
            f"face:profile/{profile.id.split('/', 1)[-1]}": (
                surfaces[profile.id],
            )
            for profile in material_profiles
        },
        "body:domain": domain,
    }
    catalog = describe_recipe_topology(recipe)
    _populate_linked_logical_aliases(catalog, logical)

    region_bindings: dict[RecipeRegionSelector, tuple[Any, ...]] = {}
    if "edge:outer-loop" in logical:
        region_bindings[RecipeRegionSelector.OUTER] = logical["edge:outer-loop"]
    if "edge:hole-loop" in logical:
        region_bindings[RecipeRegionSelector.HOLE] = logical["edge:hole-loop"]
    hole_boundary = _unique(
        entity
        for profile in hole_profiles
        for entity in _strict_profile_curve_entities(
            profile.curve_ids,
            curve_pieces,
        )
    )
    return _CompiledDraft(
        domain,
        logical,
        region_bindings,
        hole_boundary,
    )


def _compile_strict_sketch_curve(
    cad: Any,
    recipe: SketchGeometry,
    curve: SketchLine | SketchArc | SketchCircle,
    point_refs: Mapping[str, Any],
) -> tuple[tuple[Any, bool], ...]:
    if recipe.plane is None:
        raise TypeError("strict sketch plane is required")
    if isinstance(curve, SketchLine):
        return (
            (
                cad.line(
                    point_refs[curve.start_point_id],
                    point_refs[curve.end_point_id],
                ),
                False,
            ),
        )
    if isinstance(curve, SketchCircle):
        if curve.center_point_id is None:
            raise TypeError("strict circle center point is required")
        center = recipe.point(curve.center_point_id)
        perimeter = tuple(
            cad.point(
                *recipe.plane.to_global(
                    center.u + curve.radius * math.cos(angle),
                    center.v + curve.radius * math.sin(angle),
                )
            )
            for angle in (
                0.0,
                0.5 * math.pi,
                math.pi,
                1.5 * math.pi,
            )
        )
        center_ref = point_refs[curve.center_point_id]
        return tuple(
            (
                cad.circular_arc(
                    perimeter[index],
                    center_ref,
                    perimeter[(index + 1) % len(perimeter)],
                ),
                False,
            )
            for index in range(len(perimeter))
        )

    start = recipe.point(curve.start_point_id)
    center = recipe.point(curve.center_point_id)
    end = recipe.point(curve.end_point_id)
    start_angle = math.atan2(start.v - center.v, start.u - center.u)
    end_angle = math.atan2(end.v - center.v, end.u - center.u)
    if curve.orientation == "ccw":
        sweep = (end_angle - start_angle) % (2.0 * math.pi)
    else:
        sweep = -((start_angle - end_angle) % (2.0 * math.pi))
    segment_count = max(
        1,
        int(math.ceil(abs(sweep) / (0.5 * math.pi))),
    )
    radius = math.hypot(start.u - center.u, start.v - center.v)
    traversal_points = [point_refs[curve.start_point_id]]
    for index in range(1, segment_count):
        angle = start_angle + sweep * index / segment_count
        traversal_points.append(
            cad.point(
                *recipe.plane.to_global(
                    center.u + radius * math.cos(angle),
                    center.v + radius * math.sin(angle),
                )
            )
        )
    traversal_points.append(point_refs[curve.end_point_id])
    center_ref = point_refs[curve.center_point_id]
    pieces: list[tuple[Any, bool]] = []
    for start_ref, end_ref in zip(
        traversal_points,
        traversal_points[1:],
    ):
        if curve.orientation == "ccw":
            pieces.append(
                (cad.circular_arc(start_ref, center_ref, end_ref), False)
            )
        else:
            pieces.append(
                (cad.circular_arc(end_ref, center_ref, start_ref), True)
            )
    return tuple(pieces)


def _strict_profile_curve_entities(
    signed_curve_ids: tuple[str, ...],
    curve_pieces: Mapping[str, tuple[tuple[Any, bool], ...]],
) -> tuple[Any, ...]:
    return _unique(
        entity
        for signed_curve_id in signed_curve_ids
        for entity, _reversed in curve_pieces[signed_curve_id.lstrip("-")]
    )


def _populate_linked_logical_aliases(
    catalog: RecipeTopology,
    logical: dict[str, tuple[Any, ...]],
) -> None:
    """Resolve compatibility aliases whose links already have CAD mappings."""

    pending = {
        entity.logical_id: entity
        for entity in catalog.selectable_entities()
        if entity.logical_id not in logical and entity.topology_links
    }
    progressed = True
    while pending and progressed:
        progressed = False
        for logical_id, entity in tuple(pending.items()):
            linked_groups = tuple(
                logical.get(link, ())
                for link in entity.topology_links
            )
            if not linked_groups or any(not group for group in linked_groups):
                continue
            entities = _unique(
                item for group in linked_groups for item in group
            )
            if not entities or any(
                item.dimension != entity.dimension for item in entities
            ):
                continue
            logical[logical_id] = entities
            del pending[logical_id]
            progressed = True


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
    region_bindings = {
        RecipeRegionSelector.BOTTOM: edges[0],
        RecipeRegionSelector.RIGHT: edges[1],
        RecipeRegionSelector.TOP: edges[2],
        RecipeRegionSelector.LEFT: edges[3],
    }
    return _CompiledDraft(domain, logical, region_bindings)


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
        {RecipeRegionSelector.OUTER: outer},
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
    region_bindings = {
        RecipeRegionSelector.BOTTOM: outer[0],
        RecipeRegionSelector.RIGHT: outer[1],
        RecipeRegionSelector.TOP: outer[2],
        RecipeRegionSelector.LEFT: outer[3],
        RecipeRegionSelector.HOLE: hole_boundary,
    }
    return _CompiledDraft(domain, logical, region_bindings, hole_boundary)


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
    region_bindings = {
        RecipeRegionSelector.BOTTOM: bottom,
        RecipeRegionSelector.TOP: top,
        RecipeRegionSelector.FRONT: front,
        RecipeRegionSelector.RIGHT: right,
        RecipeRegionSelector.BACK: back,
        RecipeRegionSelector.LEFT: left,
    }
    return _CompiledDraft(domain, logical, region_bindings)


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
            RecipeRegionSelector.BOTTOM: bottom,
            RecipeRegionSelector.TOP: top,
            RecipeRegionSelector.OUTER: outer,
        },
    )


def _compile_boolean(cad: Any, recipe: BooleanGeometry) -> _CompiledDraft:
    if recipe.body_context is not None:
        return _compile_strict_body_boolean(cad, recipe)
    if recipe.planar_context is not None:
        return _compile_strict_planar_boolean(cad, recipe)
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
    outer = axis_aligned_rectangle(recipe.object_geometry)
    if outer is None:
        raise TopologyResolutionError("无法证明布尔结果的外轮廓")
    x, y, width, height = outer.x, outer.y, outer.width, outer.height
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
    inner = axis_aligned_rectangle(recipe.tool_geometry)
    if inner is not None:
        inner_x = inner.x
        inner_y = inner.y
        inner_width = inner.width
        inner_height = inner.height
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
            RecipeRegionSelector.BOTTOM: outer_edges[0],
            RecipeRegionSelector.RIGHT: outer_edges[1],
            RecipeRegionSelector.TOP: outer_edges[2],
            RecipeRegionSelector.LEFT: outer_edges[3],
            RecipeRegionSelector.HOLE: hole_edges,
        },
        hole_edges,
    )


def _compile_strict_planar_boolean(
    cad: Any,
    recipe: BooleanGeometry,
) -> _CompiledDraft:
    context = recipe.planar_context
    if context is None or not context.proven:
        raise TopologyResolutionError(
            "planar-boolean.lineage.unproven: strict planar Boolean "
            "lacks persisted proof"
        )
    object_draft = _compile_exact(cad, recipe.object_geometry)
    tool_draft = _compile_exact(cad, recipe.tool_geometry)
    object_compiled = _finalize(
        cad,
        recipe.object_geometry,
        describe_recipe_topology(recipe.object_geometry),
        object_draft,
    )
    tool_compiled = _finalize(
        cad,
        recipe.tool_geometry,
        describe_recipe_topology(recipe.tool_geometry),
        tool_draft,
    )
    try:
        selection = resolve_planar_boolean_faces(
            recipe.object_geometry,
            context.target_face_id,
            recipe.tool_geometry,
            context.tool_face_ids,
        )
        target_surfaces = tuple(
            object_compiled.logical_entities[selection.target_face_id]
        )
        tool_surfaces = tuple(
            tool_compiled.logical_entities[face_id][0]
            for face_id in selection.tool_face_ids
        )
        target_evidence = capture_planar_operand_evidence(
            cad,
            object_compiled,
            (selection.target_face_id,),
        )
        tool_evidence = capture_planar_operand_evidence(
            cad,
            tool_compiled,
            selection.tool_face_ids,
        )
        unaffected = _planar_unaffected_logical_entities(
            object_compiled,
            selection,
        )
        operation = cad.fuse if recipe.operation == "fuse" else cad.cut
        result = operation(target_surfaces, tool_surfaces)
        validate_planar_boolean_input_map(
            result,
            tool_count=len(tool_surfaces),
            operation=recipe.operation,
        )
        proof = resolve_planar_boolean_lineage(
            cad,
            target_evidence,
            tool_evidence,
            result,
            context,
            operation=recipe.operation,
            unaffected_logical_entities=unaffected,
        )
    except (
        KeyError,
        PlanarBooleanLineageResolutionError,
        PlanarBooleanSelectionError,
    ) as error:
        raise TopologyResolutionError(str(error)) from error
    if (
        frozenset(proof.result_entities)
        != frozenset(context.result_entities)
        or frozenset(proof.topology_mappings)
        != frozenset(context.topology_mappings)
    ):
        raise TopologyResolutionError(
            "planar-boolean.lineage.catalog-mismatch: persisted proof "
            "does not match the current OCC result"
        )
    return _CompiledDraft(
        tuple(proof.logical_entities["body:domain"]),
        dict(proof.logical_entities),
        {},
        _unique(
            entity
            for logical_id in proof.generated_intersections
            if logical_id.startswith("edge:")
            for entity in proof.logical_entities[logical_id]
        ),
    )


def _planar_unaffected_logical_entities(
    compiled: CompiledRecipeTopology,
    selection: Any,
) -> dict[str, tuple[Any, ...]]:
    target_ids = {
        selection.target_face_id,
        *selection.target.boundary_edge_ids,
        *selection.target.boundary_point_ids,
    }
    target_entities = {
        entity
        for logical_id in target_ids
        for entity in compiled.logical_entities.get(logical_id, ())
    }
    unaffected: dict[str, tuple[Any, ...]] = {}
    for item in compiled.catalog.selectable_entities():
        if item.kind == "body":
            continue
        if _canonical_catalog_logical_id(compiled.catalog, item.logical_id) != (
            item.logical_id
        ):
            continue
        entities = tuple(compiled.logical_entities.get(item.logical_id, ()))
        if not entities or any(entity in target_entities for entity in entities):
            continue
        unaffected[item.logical_id] = entities
    return unaffected


def _canonical_catalog_logical_id(catalog: RecipeTopology, logical_id: str) -> str:
    current = logical_id
    visited: set[str] = set()
    kind = LogicalEntityRef(logical_id).kind
    while current not in visited:
        visited.add(current)
        item = catalog.entity(current)
        links = tuple(
            link
            for link in item.topology_links
            if LogicalEntityRef(link).kind == kind
        )
        if len(links) != 1:
            return current
        current = links[0]
    return logical_id


def _compile_extrusion(cad: Any, recipe: ExtrudedGeometry) -> _CompiledDraft:
    base = _compile_exact(cad, recipe.base)
    base_catalog = describe_recipe_topology(recipe.base)
    try:
        selection = resolve_extrusion_source_faces(
            recipe.base,
            recipe.source_face_ids,
        )
    except ExtrusionSourceResolutionError as error:
        raise TopologyResolutionError(
            f"{error.code}: {error}"
        ) from error
    single_source = len(selection.face_ids) == 1
    domain: list[Any] = []
    bottom_faces: list[Any] = []
    top_faces: list[Any] = []
    outer_sides: list[Any] = []
    hole_sides: list[Any] = []
    logical: dict[str, tuple[Any, ...]] = {}

    for source_face_id in selection.face_ids:
        source_surfaces = tuple(base.logical_entities.get(source_face_id, ()))
        if len(source_surfaces) != 1 or source_surfaces[0].dimension != 2:
            raise TopologyResolutionError(
                "extrude.compile.surface-not-unique: "
                f"{source_face_id} 没有唯一 OCC plane surface"
            )
        feature = cad.extrude(
            source_surfaces,
            0.0,
            0.0,
            recipe.height,
        )
        source_domain = tuple(feature.primary)
        bottom = tuple(feature.inputs)
        top = tuple(feature.ends)
        sides = tuple(feature.sides)
        if len(source_domain) != 1 or source_domain[0].dimension != 3:
            raise TopologyResolutionError(
                "extrude.compile.empty-result: "
                f"{source_face_id} 没有生成唯一 volume"
            )
        if len(bottom) != 1:
            raise TopologyResolutionError(
                "extrude.compile.bottom-not-unique: "
                f"{source_face_id} 没有生成唯一 bottom face"
            )
        if len(top) != 1:
            raise TopologyResolutionError(
                "extrude.compile.top-not-unique: "
                f"{source_face_id} 没有生成唯一 top face"
            )
        if not sides:
            raise TopologyResolutionError(
                "extrude.compile.side-not-unique: "
                f"{source_face_id} 没有生成 side faces"
            )

        source_face_name = source_face_id.split(":", 1)[1]
        bottom_id = (
            "face:bottom"
            if single_source
            else f"face:bottom/{source_face_name}"
        )
        top_id = (
            "face:top"
            if single_source
            else f"face:top/{source_face_name}"
        )
        logical[bottom_id] = bottom
        logical[top_id] = top
        domain.extend(source_domain)
        bottom_faces.extend(bottom)
        top_faces.extend(top)

        side_boundaries = {
            side: set(cad.boundary((side,), combined=False))
            for side in sides
        }
        top_curves = set(cad.boundary(top, combined=False))
        bottom_curves = set(cad.boundary(bottom, combined=False))
        edge_ids, point_ids = extrusion_face_boundary_ids(
            recipe.base,
            source_face_id,
        )
        for edge_id in edge_ids:
            base_edge = base_catalog.entity(edge_id)
            source_curves = tuple(base.logical_entities[edge_id])
            resolved_sides: list[Any] = []
            resolved_top: list[Any] = []
            for source_curve in source_curves:
                if source_curve not in bottom_curves:
                    raise TopologyResolutionError(
                        "extrude.compile.side-not-unique: "
                        f"源边 {edge_id} 不属于 {source_face_id}"
                    )
                matches = tuple(
                    side
                    for side, boundaries in side_boundaries.items()
                    if source_curve in boundaries
                )
                if len(matches) != 1:
                    raise TopologyResolutionError(
                        "extrude.compile.side-not-unique: "
                        f"拉伸无法唯一追踪源边 {edge_id}"
                    )
                side = matches[0]
                top_matches = tuple(side_boundaries[side] & top_curves)
                if len(top_matches) != 1:
                    raise TopologyResolutionError(
                        "extrude.compile.top-not-unique: "
                        f"拉伸无法唯一追踪顶边 {edge_id}"
                    )
                resolved_sides.append(side)
                resolved_top.append(top_matches[0])
            edge_name = _logical_name(base_edge)
            namespace = (
                edge_name
                if single_source
                else f"{source_face_name}/{edge_name}"
            )
            logical[f"edge:bottom/{namespace}"] = source_curves
            logical[f"edge:top/{namespace}"] = _unique(resolved_top)
            logical[f"face:side/{namespace}"] = _unique(resolved_sides)

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
        top_points = set(
            _boundary_of(cad, tuple((curve,) for curve in top_curves))
        )
        vertical_endpoints = {
            curve: set(cad.boundary((curve,), combined=False))
            for curve in vertical_curves
        }
        for point_id in point_ids:
            base_point = base_catalog.entity(point_id)
            source_points = tuple(base.logical_entities[point_id])
            if len(source_points) != 1:
                raise TopologyResolutionError(
                    f"拉伸源点 {point_id} 不是唯一实体"
                )
            source_point = source_points[0]
            vertical = tuple(
                curve
                for curve, endpoints in vertical_endpoints.items()
                if source_point in endpoints
            )
            if len(vertical) != 1:
                raise TopologyResolutionError(
                    f"拉伸无法唯一追踪竖边 {point_id}"
                )
            top_point = tuple(vertical_endpoints[vertical[0]] & top_points)
            if len(top_point) != 1:
                raise TopologyResolutionError(
                    f"拉伸无法唯一追踪顶点 {point_id}"
                )
            point_name = _logical_name(base_point)
            namespace = (
                point_name
                if single_source
                else f"{source_face_name}/{point_name}"
            )
            logical[f"point:bottom/{namespace}"] = (source_point,)
            logical[f"point:top/{namespace}"] = top_point
            logical[f"edge:vertical/{namespace}"] = vertical

        source_hole_curves = set(base.hole_boundary) & bottom_curves
        source_hole_sides = _unique(
            side
            for side, boundaries in side_boundaries.items()
            if boundaries & source_hole_curves
        )
        hole_sides.extend(source_hole_sides)
        hole_side_set = set(source_hole_sides)
        outer_sides.extend(
            side for side in sides if side not in hole_side_set
        )

    compiled_domain = _unique(domain)
    logical["body:domain"] = compiled_domain
    bottom = _unique(bottom_faces)
    top = _unique(top_faces)
    hole = _unique(hole_sides)
    outer = _unique(outer_sides)
    region_bindings = {
        RecipeRegionSelector.BOTTOM: bottom,
        RecipeRegionSelector.TOP: top,
        RecipeRegionSelector.OUTER: outer,
    }
    if hole:
        region_bindings[RecipeRegionSelector.HOLE] = hole
    return _CompiledDraft(
        compiled_domain,
        logical,
        region_bindings,
        hole,
    )


def _compile_strict_body_boolean(
    cad: Any,
    recipe: BooleanGeometry,
) -> _CompiledDraft:
    context = recipe.body_context
    if context is None or not context.proven:
        raise TopologyResolutionError(
            "boolean.lineage.unproven: strict Body Boolean lacks persisted proof"
        )
    target = _compile_exact(cad, recipe.object_geometry)
    tool = _compile_exact(cad, recipe.tool_geometry)
    target_compiled = _finalize(
        cad,
        recipe.object_geometry,
        describe_recipe_topology(recipe.object_geometry),
        target,
    )
    tool_compiled = _finalize(
        cad,
        recipe.tool_geometry,
        describe_recipe_topology(recipe.tool_geometry),
        tool,
    )
    target_evidence = capture_boolean_operand_evidence(cad, target_compiled)
    tool_evidence = capture_boolean_operand_evidence(cad, tool_compiled)
    operation = cad.fuse if recipe.operation == "fuse" else cad.cut
    result = operation(target.domain, tool.domain)
    try:
        validate_solid_boolean_input_map(result)
    except BooleanLineageResolutionError as error:
        raise TopologyResolutionError(str(error)) from error
    volumes = result.of_dimension(3)
    boundary = (
        ()
        if len(volumes) != 1
        else tuple(cad.boundary(volumes, combined=False))
    )
    try:
        proof = resolve_solid_boolean_lineage(
            cad,
            target_evidence,
            tool_evidence,
            result,
            boundary,
            context,
            operation=recipe.operation,
        )
    except BooleanLineageResolutionError as error:
        raise TopologyResolutionError(str(error)) from error
    if (
        frozenset(proof.result_entities)
        != frozenset(context.result_entities)
        or frozenset(proof.topology_mappings)
        != frozenset(context.topology_mappings)
    ):
        raise TopologyResolutionError(
            "boolean.lineage.catalog-mismatch: persisted proof does not "
            "match the current OCC result"
        )
    return _CompiledDraft(
        (proof.result_volume,),
        dict(proof.logical_entities),
        {},
    )


def _compile_multi_body(
    cad: Any,
    recipe: MultiBodyGeometry,
) -> _CompiledDraft:
    domain: list[Any] = []
    logical: dict[str, tuple[Any, ...]] = {}
    region_bindings: dict[RecipeRegionSelector, list[Any]] = {}
    hole_boundary: list[Any] = []
    for body in recipe.bodies:
        local = _compile_exact(cad, body.recipe)
        local_domain = _unique(local.domain)
        if len(local_domain) != 1 or local_domain[0].dimension != 3:
            raise TopologyResolutionError(
                f"multi-body.single-volume: Body {body.id} "
                "must compile to exactly one volume"
            )
        domain.extend(local_domain)
        for logical_id, entities in local.logical_entities.items():
            reference = LogicalEntityRef(logical_id)
            if reference.kind == "body":
                target_id = f"body:{body.id}"
            else:
                _kind, local_name = logical_id.split(":", 1)
                target_id = f"{reference.kind}:{body.id}/{local_name}"
            logical[target_id] = tuple(entities)
        for selector, entities in local.region_bindings.items():
            region_bindings.setdefault(selector, []).extend(entities)
        hole_boundary.extend(local.hole_boundary)
    compiled_domain = _unique(domain)
    logical["body:domain"] = compiled_domain
    return _CompiledDraft(
        compiled_domain,
        logical,
        {
            selector: _unique(entities)
            for selector, entities in region_bindings.items()
        },
        _unique(hole_boundary),
    )


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
            selector: _unique(entities)
            for selector, entities in draft.region_bindings.items()
            if entities
        },
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
    if isinstance(recipe, MultiBodyGeometry):
        return _compile_multi_body(cad, recipe).domain
    if isinstance(recipe, SketchGeometry):
        if recipe.is_strict:
            return _compile_strict_sketch(cad, recipe).domain
        return _build_domain_only(cad, expand_sketch_recipe(recipe))
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
        return _compile_extrusion(cad, recipe).domain
    raise TypeError(f"不支持的几何配方: {type(recipe).__name__}")


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


__all__ = [
    "CompiledRecipeTopology",
    "TopologyResolutionError",
    "compile_recipe",
]
