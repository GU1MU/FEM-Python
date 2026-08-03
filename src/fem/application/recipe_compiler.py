"""Compile native recipes into one-session CAD topology mappings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from typing import Any

from fem.geometry.recipe_analysis import (
    analyze_sketch_profiles,
    axis_aligned_rectangle,
    expand_sketch_recipe,
)
from fem.geometry.sketch_support import resolve_face_workplane
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
from fem.geometry.part_boolean import (
    localize_part_boolean_context,
    namespace_part_boolean_context,
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
    FaceSeedConnectionProof,
    FaceSketchBooleanGeometry,
    FaceSketchBooleanOperation,
    FaceSketchBooleanStepProof,
    MovedGeometry,
    MultiBodyGeometry,
    NativeGeometry,
    PlateWithHoleGeometry,
    PathSweptGeometry,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SketchArc,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    WireGeometry,
    WirePoint,
    geometry_dimension,
    planar_geometry_normal,
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


@dataclass(frozen=True, slots=True)
class _RigidEntityFingerprint:
    center: tuple[float, float, float]
    measure: float
    geometry_type: str
    boundary: tuple[Any, ...]


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
        return _translate_compiled_draft(
            cad,
            draft,
            recipe.dx,
            recipe.dy,
            recipe.dz,
        )
    if isinstance(recipe, RotatedGeometry):
        draft = _compile_exact(cad, recipe.base)
        return _rotate_compiled_draft(
            cad,
            draft,
            recipe.axis,
            math.radians(recipe.angle_degrees),
        )
    if isinstance(recipe, ExtrudedGeometry):
        return _compile_extrusion(cad, recipe)
    if isinstance(recipe, RevolvedGeometry):
        return _compile_revolution(cad, recipe)
    if isinstance(recipe, PathSweptGeometry):
        return _compile_path_sweep(cad, recipe)
    if isinstance(recipe, FaceSketchBooleanGeometry):
        return _compile_face_sketch_boolean(cad, recipe)
    if isinstance(recipe, MultiBodyGeometry):
        return _compile_multi_body(cad, recipe)
    if isinstance(recipe, BooleanGeometry):
        return _compile_boolean(cad, recipe)
    raise TypeError(f"不支持的几何配方: {type(recipe).__name__}")


def _translate_compiled_draft(
    cad: Any,
    draft: _CompiledDraft,
    dx: float,
    dy: float,
    dz: float,
) -> _CompiledDraft:
    fingerprints = _capture_rigid_entity_fingerprints(cad, draft)
    original_domain = draft.domain
    transformed_domain = tuple(cad.translate(original_domain, dx, dy, dz))
    return _rebind_rigid_transform(
        cad,
        draft,
        original_domain,
        transformed_domain,
        fingerprints,
        lambda point: (
            point[0] + dx,
            point[1] + dy,
            point[2] + dz,
        ),
    )


def _rotate_compiled_draft(
    cad: Any,
    draft: _CompiledDraft,
    axis_name: str,
    angle: float,
) -> _CompiledDraft:
    axis = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }[axis_name]
    fingerprints = _capture_rigid_entity_fingerprints(cad, draft)
    original_domain = draft.domain
    transformed_domain = tuple(
        cad.rotate(
            original_domain,
            0.0,
            0.0,
            0.0,
            *axis,
            angle,
        )
    )
    cosine, sine = math.cos(angle), math.sin(angle)

    def transform(point: tuple[float, float, float]):
        x, y, z = point
        if axis_name == "x":
            return x, y * cosine - z * sine, y * sine + z * cosine
        if axis_name == "y":
            return x * cosine + z * sine, y, -x * sine + z * cosine
        return x * cosine - y * sine, x * sine + y * cosine, z

    return _rebind_rigid_transform(
        cad,
        draft,
        original_domain,
        transformed_domain,
        fingerprints,
        transform,
    )


def _capture_rigid_entity_fingerprints(
    cad: Any,
    draft: _CompiledDraft,
) -> dict[Any, _RigidEntityFingerprint]:
    closure = _domain_boundary_closure(cad, draft.domain)
    closure_set = set(closure)
    referenced = _unique(
        entity
        for group in (
            *draft.logical_entities.values(),
            *draft.region_bindings.values(),
            draft.hole_boundary,
        )
        for entity in group
    )
    outside = tuple(
        entity for entity in referenced if entity not in closure_set
    )
    if outside:
        entity = outside[0]
        raise TopologyResolutionError(
            "rigid-transform.rebind.outside-domain: "
            f"逻辑实体 ({entity.dimension}, {entity.tag}) 不属于计算域边界"
        )
    required = list(referenced)
    frontier = tuple(
        entity for entity in required if entity.dimension > 0
    )
    while frontier:
        boundary = _unique(
            child
            for parent in frontier
            for child in cad.boundary(
                (parent,),
                combined=False,
            )
        )
        frontier = tuple(
            entity for entity in boundary if entity not in required
        )
        required.extend(frontier)
    return {
        entity: _RigidEntityFingerprint(
            tuple(float(value) for value in cad.center_of_mass(entity)),
            _entity_measure(cad, entity),
            str(cad.geometry_type(entity)),
            (
                ()
                if entity.dimension == 0
                else tuple(
                    cad.boundary(
                        (entity,),
                        combined=False,
                    )
                )
            ),
        )
        for entity in required
    }


def _rebind_rigid_transform(
    cad: Any,
    draft: _CompiledDraft,
    original_domain: tuple[Any, ...],
    transformed_domain: tuple[Any, ...],
    fingerprints: dict[Any, _RigidEntityFingerprint],
    transform_point,
) -> _CompiledDraft:
    if len(original_domain) != len(transformed_domain):
        raise TopologyResolutionError(
            "rigid-transform.rebind.domain-count: "
            "刚体变换改变了计算域数量"
        )
    transformed_closure = _domain_boundary_closure(
        cad,
        transformed_domain,
    )
    transformed_fingerprints = {
        entity: _RigidEntityFingerprint(
            tuple(float(value) for value in cad.center_of_mass(entity)),
            _entity_measure(cad, entity),
            str(cad.geometry_type(entity)),
            (
                ()
                if entity.dimension == 0
                else tuple(
                    cad.boundary(
                        (entity,),
                        combined=False,
                    )
                )
            ),
        )
        for entity in transformed_closure
    }
    rebound = dict(
        zip(
            original_domain,
            transformed_domain,
            strict=True,
        )
    )
    used = set(transformed_domain)
    tolerance = max(
        1.0e-8,
        float(cad.effective_bounding_box_tolerance(1.0e-9)),
    )
    maximum_dimension = max(
        (entity.dimension for entity in original_domain),
        default=0,
    )
    for dimension in range(maximum_dimension):
        originals = tuple(
            entity
            for entity in fingerprints
            if entity.dimension == dimension
        )
        candidates = tuple(
            entity
            for entity in transformed_closure
            if entity.dimension == dimension and entity not in used
        )
        for original in originals:
            fingerprint = fingerprints[original]
            expected_center = transform_point(fingerprint.center)
            expected_boundary = {
                rebound[entity] for entity in fingerprint.boundary
            }
            matches = tuple(
                candidate
                for candidate in candidates
                if candidate not in used
                and _rigid_fingerprint_matches(
                    transformed_fingerprints[candidate],
                    expected_center,
                    fingerprint.measure,
                    fingerprint.geometry_type,
                    expected_boundary,
                    tolerance,
                )
            )
            if len(matches) != 1:
                raise TopologyResolutionError(
                    "rigid-transform.rebind.ambiguous: "
                    f"实体 ({original.dimension}, {original.tag}) "
                    f"匹配到 {len(matches)} 个变换后候选"
                )
            rebound[original] = matches[0]
            used.add(matches[0])

    def remap(group: tuple[Any, ...]) -> tuple[Any, ...]:
        try:
            return tuple(rebound[entity] for entity in group)
        except KeyError as error:
            entity = error.args[0]
            raise TopologyResolutionError(
                "rigid-transform.rebind.missing: "
                f"无法重新绑定实体 ({entity.dimension}, {entity.tag})"
            ) from error

    return _CompiledDraft(
        transformed_domain,
        {
            logical_id: remap(entities)
            for logical_id, entities in draft.logical_entities.items()
        },
        {
            selector: remap(entities)
            for selector, entities in draft.region_bindings.items()
        },
        remap(draft.hole_boundary),
    )


def _rigid_fingerprint_matches(
    fingerprint: _RigidEntityFingerprint,
    expected_center: tuple[float, float, float],
    expected_measure: float,
    expected_type: str,
    expected_boundary: set[Any],
    tolerance: float,
) -> bool:
    return (
        fingerprint.geometry_type == expected_type
        and all(
            math.isclose(
                actual,
                expected,
                rel_tol=1.0e-9,
                abs_tol=tolerance,
            )
            for actual, expected in zip(
                fingerprint.center,
                expected_center,
                strict=True,
            )
        )
        and math.isclose(
            fingerprint.measure,
            expected_measure,
            rel_tol=1.0e-9,
            abs_tol=tolerance,
        )
        and set(fingerprint.boundary) == expected_boundary
    )


def _domain_boundary_closure(
    cad: Any,
    domain: tuple[Any, ...],
) -> tuple[Any, ...]:
    closure = list(_unique(domain))
    frontier = tuple(closure)
    while frontier and frontier[0].dimension > 0:
        boundary = _unique(
            cad.boundary(
                frontier,
                combined=False,
            )
        )
        frontier = tuple(
            entity for entity in boundary if entity not in closure
        )
        closure.extend(frontier)
    return tuple(closure)


def _entity_measure(cad: Any, entity: Any) -> float:
    if entity.dimension == 0:
        return 0.0
    if entity.dimension == 1:
        return float(cad.length(entity))
    if entity.dimension == 2:
        return float(cad.area(entity))
    return float(cad.volume(entity))


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
    if recipe.body_context is not None or recipe.part_context is not None:
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
    normal = planar_geometry_normal(recipe.base)

    for source_face_id in selection.face_ids:
        source_surfaces = tuple(base.logical_entities.get(source_face_id, ()))
        if len(source_surfaces) != 1 or source_surfaces[0].dimension != 2:
            raise TopologyResolutionError(
                "extrude.compile.surface-not-unique: "
                f"{source_face_id} 没有唯一 OCC plane surface"
            )
        feature = cad.extrude(
            source_surfaces,
            normal[0] * recipe.height,
            normal[1] * recipe.height,
            normal[2] * recipe.height,
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


def _compile_revolution(cad: Any, recipe: RevolvedGeometry) -> _CompiledDraft:
    base = _compile_exact(cad, recipe.base)
    try:
        selection = resolve_extrusion_source_faces(
            recipe.base,
            recipe.source_face_ids,
        )
    except ExtrusionSourceResolutionError as error:
        raise TopologyResolutionError(
            f"{error.code}: {error}"
        ) from error
    axis = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }[recipe.axis]
    if len(selection.face_ids) != 1:
        raise TopologyResolutionError(
            "revolve.compile.source-not-unique: strict revolution lineage "
            "requires one selected material Profile"
        )
    domain: list[Any] = []
    logical: dict[str, tuple[Any, ...]] = {}
    for source_face_id in selection.face_ids:
        source_surfaces = tuple(base.logical_entities.get(source_face_id, ()))
        if len(source_surfaces) != 1 or source_surfaces[0].dimension != 2:
            raise TopologyResolutionError(
                "revolve.compile.surface-not-unique: "
                f"{source_face_id} 没有唯一 OCC plane surface"
            )
        surface_normal = cad.geometry_direction(source_surfaces[0])
        if surface_normal is not None:
            axis_dot_normal = sum(
                left * right for left, right in zip(axis, surface_normal, strict=True)
            )
            surface_center = tuple(cad.center_of_mass(source_surfaces[0]))
            plane_offset = sum(
                left * right
                for left, right in zip(surface_center, surface_normal, strict=True)
            )
            tolerance = max(
                1.0e-9,
                float(cad.effective_bounding_box_tolerance(1.0e-9)),
            )
            if (
                math.isclose(axis_dot_normal, 0.0, rel_tol=0.0, abs_tol=tolerance)
                and math.isclose(plane_offset, 0.0, rel_tol=0.0, abs_tol=tolerance)
            ):
                radial = (
                    surface_normal[1] * axis[2] - surface_normal[2] * axis[1],
                    surface_normal[2] * axis[0] - surface_normal[0] * axis[2],
                    surface_normal[0] * axis[1] - surface_normal[1] * axis[0],
                )
                point_coordinates = tuple(
                    tuple(cad.center_of_mass(entity))
                    for entity in _domain_boundary_closure(cad, source_surfaces)
                    if entity.dimension == 0
                )
                signed = tuple(
                    sum(
                        left * right
                        for left, right in zip(point, radial, strict=True)
                    )
                    for point in point_coordinates
                )
                if signed and min(signed) < -tolerance and max(signed) > tolerance:
                    raise TopologyResolutionError(
                        "revolve.compile.profile-crosses-axis: "
                        f"{source_face_id} 穿过旋转轴，会生成重叠或退化体"
                    )
        feature = cad.revolve(
            source_surfaces,
            0.0,
            0.0,
            0.0,
            *axis,
            math.radians(recipe.angle_degrees),
        )
        volumes = tuple(feature.primary)
        if len(volumes) != 1 or volumes[0].dimension != 3:
            raise TopologyResolutionError(
                "revolve.compile.empty-result: "
                f"{source_face_id} 没有生成唯一 volume"
            )
        if cad.volume(volumes[0]) <= 0.0:
            raise TopologyResolutionError(
                "revolve.compile.zero-volume: "
                "当前二维面、扫掠轴和角度会生成零体积或退化实体；"
                "请改用与草图平面不垂直的轴"
            )
        sides = tuple(feature.sides)
        if not sides:
            raise TopologyResolutionError(
                "revolve.compile.side-missing: 旋转扫掠没有可证明的侧面"
            )
        logical["face:sides"] = sides
        if recipe.angle_degrees < 360.0:
            ends = tuple(feature.ends)
            if len(ends) != 1:
                raise TopologyResolutionError(
                    "revolve.compile.end-not-unique: 旋转终止面 lineage 不唯一"
                )
            logical["face:start"] = source_surfaces
            logical["face:end"] = ends
        domain.extend(volumes)
    compiled_domain = _unique(domain)
    logical["body:domain"] = compiled_domain
    return _CompiledDraft(
        compiled_domain,
        logical,
        {},
    )


def _compile_path_sweep(cad: Any, recipe: PathSweptGeometry) -> _CompiledDraft:
    """Compile one explicit ordered polyline path through the Gmsh pipe seam."""

    base = _compile_exact(cad, recipe.base)
    try:
        selection = resolve_extrusion_source_faces(
            recipe.base,
            recipe.source_face_ids,
        )
    except ExtrusionSourceResolutionError as error:
        raise TopologyResolutionError(f"{error.code}: {error}") from error
    if len(selection.face_ids) != 1:
        raise TopologyResolutionError(
            "path-sweep.compile.source-not-unique: "
            "路径扫掠必须显式选择一个 material Profile"
        )
    source_face_id = selection.face_ids[0]
    source_surfaces = tuple(base.logical_entities.get(source_face_id, ()))
    if len(source_surfaces) != 1 or source_surfaces[0].dimension != 2:
        raise TopologyResolutionError(
            "path-sweep.compile.surface-not-unique: "
            f"{source_face_id} 没有唯一 OCC plane surface"
        )

    ordered_names = (
        recipe.path.members[0].start,
        *(member.end for member in recipe.path.members),
    )
    points = {point.name: point for point in recipe.path.points}
    start = points[ordered_names[0]]
    following = points[ordered_names[1]]
    origin = _planar_recipe_origin(recipe.base)
    normal = planar_geometry_normal(recipe.base)
    tolerance = 1.0e-9 * max(
        1.0,
        *(abs(value) for value in (*origin, start.x, start.y, start.z)),
    )
    if math.dist(origin, (start.x, start.y, start.z)) > tolerance:
        raise TopologyResolutionError(
            "path-sweep.compile.start-position-mismatch: "
            "路径起点必须与 Profile 平面原点一致，以显式确定起始姿态"
        )
    tangent = (
        following.x - start.x,
        following.y - start.y,
        following.z - start.z,
    )
    tangent_length = math.sqrt(sum(value * value for value in tangent))
    alignment = abs(
        sum(left * right for left, right in zip(tangent, normal, strict=True))
    ) / tangent_length
    if not math.isclose(alignment, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise TopologyResolutionError(
            "path-sweep.compile.start-orientation-mismatch: "
            "Profile 正法向必须与路径首段平行，以显式确定起始姿态"
        )

    path_points = {
        name: cad.point(points[name].x, points[name].y, points[name].z)
        for name in ordered_names
    }
    path_curves = tuple(
        cad.line(path_points[member.start], path_points[member.end])
        for member in recipe.path.members
    )
    path_wire = cad.wire(
        tuple(cad.orient(curve) for curve in path_curves),
        closed=False,
    )
    feature = cad.sweep(
        source_surfaces,
        path_wire,
        frame="fixed" if recipe.frame_strategy == "fixed" else "discrete",
    )
    volumes = tuple(feature.primary)
    if len(volumes) != 1 or volumes[0].dimension != 3:
        raise TopologyResolutionError(
            "path-sweep.compile.multi-solid: "
            "路径扫掠未生成唯一 volume"
        )
    if cad.volume(volumes[0]) <= 0.0:
        raise TopologyResolutionError(
            "path-sweep.compile.zero-volume: 路径扫掠生成了非正体积实体"
        )
    source_center = tuple(cad.center_of_mass(source_surfaces[0]))
    end_faces = tuple(feature.ends)
    start_faces = tuple(
        entity
        for entity in end_faces
        if all(
            math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=tolerance)
            for actual, expected in zip(
                cad.center_of_mass(entity), source_center, strict=True
            )
        )
    )
    terminals = tuple(entity for entity in end_faces if entity not in start_faces)
    if len(start_faces) != 1 or len(terminals) != 1:
        raise TopologyResolutionError(
            "path-sweep.compile.end-not-unique: 路径扫掠端面 lineage 不唯一"
        )
    sides = tuple(feature.sides)
    if not sides:
        raise TopologyResolutionError(
            "path-sweep.compile.side-missing: 路径扫掠没有可证明的侧面"
        )
    side_boundaries = {
        side: set(cad.boundary((side,), combined=False))
        for side in sides
    }
    start_curves = tuple(cad.boundary(start_faces, combined=False))
    terminal_curves = set(cad.boundary(terminals, combined=False))
    edge_ids, _point_ids = extrusion_face_boundary_ids(
        recipe.base,
        source_face_id,
    )
    logical: dict[str, tuple[Any, ...]] = {
        "face:start": start_faces,
        "face:end": terminals,
        "body:domain": volumes,
    }
    base_catalog = describe_recipe_topology(recipe.base)
    hole_edges = {
        edge_id
        for edge_id in edge_ids
        if "hole" in base_catalog.entity(edge_id).semantic_role
    }
    outer_sides: list[Any] = []
    hole_sides: list[Any] = []
    claimed: set[Any] = set()
    edge_seed_indexes: dict[str, int] = {}
    for edge_id in edge_ids:
        source_curves = set(base.logical_entities.get(edge_id, ()))
        copied_curves: set[Any] = set()
        for source_curve in source_curves:
            source_curve_center = tuple(cad.center_of_mass(source_curve))
            source_curve_measure = _entity_measure(cad, source_curve)
            source_curve_type = cad.geometry_type(source_curve)
            matches = tuple(
                candidate
                for candidate in start_curves
                if cad.geometry_type(candidate) == source_curve_type
                and math.isclose(
                    _entity_measure(cad, candidate),
                    source_curve_measure,
                    rel_tol=1.0e-9,
                    abs_tol=tolerance,
                )
                and all(
                    math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=tolerance)
                    for actual, expected in zip(
                        cad.center_of_mass(candidate),
                        source_curve_center,
                        strict=True,
                    )
                )
            )
            if len(matches) != 1:
                raise TopologyResolutionError(
                    "path-sweep.compile.start-edge-lineage: "
                    f"无法唯一追踪源边 {edge_id} 的起始副本"
                )
            copied_curves.add(matches[0])
        seed_indexes = tuple(
            index
            for index, side in enumerate(sides)
            if side_boundaries[side] & copied_curves
        )
        if len(seed_indexes) != 1:
            raise TopologyResolutionError(
                "path-sweep.compile.side-seed-not-unique: "
                f"源边 {edge_id} 没有唯一起始侧面"
            )
        edge_seed_indexes[edge_id] = seed_indexes[0]
    ordered_seeds = tuple(edge_seed_indexes[edge_id] for edge_id in edge_ids)
    if tuple(sorted(ordered_seeds)) != ordered_seeds:
        raise TopologyResolutionError(
            "path-sweep.compile.side-order-ambiguous: "
            "Gmsh 侧面顺序无法与 Profile 边界顺序对齐"
        )
    for edge_index, edge_id in enumerate(edge_ids):
        start_index = edge_seed_indexes[edge_id]
        stop_index = (
            len(sides)
            if edge_index + 1 == len(edge_ids)
            else edge_seed_indexes[edge_ids[edge_index + 1]]
        )
        matched = sides[start_index:stop_index]
        if not matched:
            raise TopologyResolutionError(
                "path-sweep.compile.side-lineage-missing: "
                f"无法追踪源边 {edge_id} 的侧面"
            )
        if any(
            not (side_boundaries[first] & side_boundaries[second])
            for first, second in zip(matched, matched[1:])
        ) or not (side_boundaries[matched[-1]] & terminal_curves):
            raise TopologyResolutionError(
                "path-sweep.compile.side-lineage-disconnected: "
                f"源边 {edge_id} 的侧面链未连续达到终端面"
            )
        logical[f"face:side/{edge_id.split(':', 1)[1]}"] = matched
        claimed.update(matched)
        if edge_id in hole_edges:
            hole_sides.extend(matched)
        else:
            outer_sides.extend(matched)
    if claimed != set(sides):
        raise TopologyResolutionError(
            "path-sweep.compile.side-lineage-incomplete: 路径扫掠侧面 lineage 不完整"
        )
    region_bindings = {
        RecipeRegionSelector.BOTTOM: start_faces,
        RecipeRegionSelector.TOP: terminals,
        RecipeRegionSelector.OUTER: _unique(outer_sides),
    }
    if hole_sides:
        region_bindings[RecipeRegionSelector.HOLE] = _unique(hole_sides)
    return _CompiledDraft(
        volumes,
        logical,
        region_bindings,
        _unique(hole_sides),
    )


def _planar_recipe_origin(recipe: NativeGeometry) -> tuple[float, float, float]:
    """Return the authoring origin that anchors an explicit sweep start pose."""

    if isinstance(recipe, SketchGeometry) and recipe.is_strict:
        assert recipe.plane is not None
        return recipe.plane.origin
    if isinstance(recipe, MovedGeometry):
        x, y, z = _planar_recipe_origin(recipe.base)
        return x + recipe.dx, y + recipe.dy, z + recipe.dz
    if isinstance(recipe, RotatedGeometry):
        x, y, z = _planar_recipe_origin(recipe.base)
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)
        if recipe.axis == "x":
            return x, y * cosine - z * sine, y * sine + z * cosine
        if recipe.axis == "y":
            return x * cosine + z * sine, y, -x * sine + z * cosine
        return x * cosine - y * sine, x * sine + y * cosine, z
    if isinstance(recipe, BooleanGeometry):
        return _planar_recipe_origin(recipe.object_geometry)
    return 0.0, 0.0, 0.0


@dataclass(frozen=True, slots=True)
class _FaceSketchBooleanCompileResult:
    draft: _CompiledDraft
    target_body_id: str
    target_logical_entities: Mapping[str, tuple[Any, ...]]
    step_proofs: tuple[FaceSketchBooleanStepProof, ...]


class _BooleanProofCatalog:
    def __init__(self, entities: tuple[Any, ...]) -> None:
        self._entities = {item.logical_id: item for item in entities}

    def entity(self, logical_id: str) -> Any:
        try:
            return self._entities[logical_id]
        except KeyError as error:
            raise KeyError(logical_id) from error


def _compile_face_sketch_boolean(
    cad: Any,
    recipe: FaceSketchBooleanGeometry,
) -> _CompiledDraft:
    prepared = _prepare_face_sketch_boolean_draft(cad, recipe)
    if prepared.step_proofs != recipe.step_proofs:
        raise TopologyResolutionError(
            "face-sketch-boolean.lineage.catalog-mismatch: "
            "保存的逐轮廓证明与当前 OCC 结果不一致"
        )
    return prepared.draft


def _prepare_face_sketch_boolean_draft(
    cad: Any,
    recipe: FaceSketchBooleanGeometry,
) -> _FaceSketchBooleanCompileResult:
    """Build and prove a detached face-sketch Boolean chain."""

    analysis = analyze_sketch_profiles(recipe.sketch)
    if analysis.blocking_diagnostics:
        raise TopologyResolutionError(analysis.blocking_diagnostics[0].message)
    material = {item.id: item for item in analysis.profiles if item.is_material}
    missing = tuple(
        profile_id
        for profile_id in recipe.participating_profile_ids
        if profile_id not in material
    )
    if missing:
        raise TopologyResolutionError(
            "face-sketch-boolean.profile.invalid: 参与轮廓不存在或不是材料轮廓 "
            f"{missing!r}"
        )

    base = compile_recipe(cad, recipe.base)
    workplane = resolve_face_workplane(
        cad,
        base.logical_entities,
        recipe.support_face_id,
        recipe.workplane_strategy,
    )
    _require_matching_sketch_plane(recipe.sketch.plane, workplane.plane)
    target_compiled = _target_body_operand(cad, base, workplane.volume)

    tool_rows: list[tuple[str, CompiledRecipeTopology, Any]] = []
    for profile_id in recipe.participating_profile_ids:
        tool_sketch = _profile_tool_sketch(
            recipe.sketch,
            analysis.profiles,
            profile_id,
            inward=recipe.direction.value == "inward",
        )
        tool_recipe = ExtrudedGeometry(
            tool_sketch,
            recipe.distance,
            (f"face:{profile_id}",),
        )
        tool = compile_recipe(cad, tool_recipe)
        start_faces = tuple(tool.logical_entities.get("face:bottom", ()))
        if len(start_faces) != 1:
            raise TopologyResolutionError(
                "face-sketch-boolean.tool.start-face: 工具体起始面无法唯一解析"
            )
        _positive_face_overlap(
            cad,
            workplane.surface,
            start_faces[0],
        )
        tool_rows.append((profile_id, tool, start_faces[0]))

    proofs: list[FaceSketchBooleanStepProof] = []
    current = target_compiled
    final_proof = None
    for step_index, (profile_id, tool, tool_start) in enumerate(tool_rows, start=1):
        target_evidence = capture_boolean_operand_evidence(cad, current)
        tool_evidence = capture_boolean_operand_evidence(cad, tool)
        seed_proof = (
            _resolve_current_face_seed(cad, current, tool_start)
            if recipe.operation is FaceSketchBooleanOperation.FUSE
            and recipe.direction.value == "outward"
            else None
        )
        operation = (
            cad.fuse
            if recipe.operation is FaceSketchBooleanOperation.FUSE
            else cad.cut
        )
        boolean_result = operation(current.domain, tool.domain)
        try:
            validate_solid_boolean_input_map(
                boolean_result,
                face_seed_connection=(
                    seed_proof
                ),
            )
            volumes = boolean_result.of_dimension(3)
            boundary = (
                ()
                if len(volumes) != 1
                else tuple(cad.boundary(volumes, combined=False))
            )
            context = _face_step_context(step_index)
            final_proof = resolve_solid_boolean_lineage(
                cad,
                target_evidence,
                tool_evidence,
                boolean_result,
                boundary,
                context,
                operation=recipe.operation.value,
                face_seed_connection=(
                    seed_proof
                    if recipe.operation is FaceSketchBooleanOperation.FUSE
                    and recipe.direction.value == "outward"
                    else None
                ),
                feature_id=f"{recipe.feature_id}/{profile_id}",
            )
        except BooleanLineageResolutionError as error:
            raise TopologyResolutionError(
                f"face-sketch-boolean.step.{profile_id}: {error}"
            ) from error
        step = FaceSketchBooleanStepProof(
            profile_id,
            final_proof.result_entities,
            final_proof.topology_mappings,
            final_proof.connection_proof,
        )
        proofs.append(step)
        current = CompiledRecipeTopology(
            (final_proof.result_volume,),
            boundary,
            _BooleanProofCatalog(final_proof.result_entities),
            final_proof.logical_entities,
            {},
        )

    if final_proof is None:
        raise TopologyResolutionError(
            "face-sketch-boolean.profile.required: 至少需要一个参与轮廓"
        )
    _require_unaffected_body_isolation(
        cad,
        base,
        workplane.volume,
        final_proof.result_volume,
    )
    draft, target_logical = _assemble_face_sketch_result(
        base,
        workplane.volume,
        workplane.target_body_id,
        final_proof,
    )
    return _FaceSketchBooleanCompileResult(
        draft,
        workplane.target_body_id,
        target_logical,
        tuple(proofs),
    )


def _face_step_context(step_index: int) -> Any:
    from fem.geometry.recipes import BooleanBodyContext

    return BooleanBodyContext(
        f"BF{step_index}",
        "B1",
        "B2",
        f"面草图工具-{step_index}",
    )


def _target_body_operand(
    cad: Any,
    base: CompiledRecipeTopology,
    volume: Any,
) -> CompiledRecipeTopology:
    faces = tuple(cad.boundary((volume,), combined=False))
    edges = tuple(cad.boundary(faces, combined=False))
    points = tuple(cad.boundary(edges, combined=False))
    closure = {volume, *faces, *edges, *points}
    logical = {
        logical_id: tuple(entity for entity in entities if entity in closure)
        for logical_id, entities in base.logical_entities.items()
    }
    logical = {
        logical_id: entities
        for logical_id, entities in logical.items()
        if entities and not logical_id.startswith("body:")
    }
    logical["body:domain"] = (volume,)
    return CompiledRecipeTopology(
        (volume,),
        faces,
        base.catalog,
        logical,
        {},
    )


def _require_matching_sketch_plane(
    sketch_plane: SketchPlane | None,
    support_plane: SketchPlane,
) -> None:
    if sketch_plane is None:
        raise TopologyResolutionError(
            "face-sketch-boolean.sketch.strict: 面草图必须是严格平面草图"
        )
    left = (*sketch_plane.origin, *sketch_plane.x_direction, *sketch_plane.y_direction)
    right = (*support_plane.origin, *support_plane.x_direction, *support_plane.y_direction)
    scale = max(*(abs(value) for value in (*left, *right)), 1.0)
    if any(abs(a - b) > 1.0e-8 * scale for a, b in zip(left, right, strict=True)):
        raise TopologyResolutionError(
            "face-sketch-boolean.sketch.workplane-mismatch: 草图坐标系与工作面不一致"
        )


def _profile_tool_sketch(
    sketch: SketchGeometry,
    profiles: tuple[Any, ...],
    profile_id: str,
    *,
    inward: bool,
) -> SketchGeometry:
    profile = next(item for item in profiles if item.id == profile_id)
    children = tuple(
        item
        for item in profiles
        if item.is_hole and item.parent_profile_id == profile_id
    )
    curve_ids = {
        signed_id.lstrip("-")
        for item in (profile, *children)
        for signed_id in item.curve_ids
    }
    curves = tuple(curve for curve in sketch.curves if curve.id in curve_ids)
    point_ids = {
        point_id
        for curve in curves
        for point_id in _strict_curve_point_ids(curve)
    }
    points = tuple(point for point in sketch.points if point.id in point_ids)
    plane = sketch.plane
    if plane is None:
        raise TopologyResolutionError("面草图必须是严格平面草图")
    if inward:
        plane = SketchPlane(
            plane.origin,
            plane.x_direction,
            tuple(-value for value in plane.y_direction),
        )
        points = tuple(replace(point, v=-point.v) for point in points)
        curves = tuple(
            replace(
                curve,
                orientation="cw" if curve.orientation == "ccw" else "ccw",
            )
            if isinstance(curve, SketchArc)
            else curve
            for curve in curves
        )
    return SketchGeometry(
        f"{sketch.name}-{profile_id}",
        plane,
        points,
        curves,
    )


def _strict_curve_point_ids(curve: Any) -> tuple[str, ...]:
    if isinstance(curve, SketchLine):
        return curve.start_point_id, curve.end_point_id
    if isinstance(curve, SketchArc):
        return curve.start_point_id, curve.center_point_id, curve.end_point_id
    if isinstance(curve, SketchCircle):
        return (curve.center_point_id,)
    raise TypeError(f"不支持的草图曲线: {type(curve).__name__}")


def _positive_face_overlap(cad: Any, support: Any, tool_start: Any) -> float:
    try:
        support_copy = cad.copy((support,))
        tool_copy = cad.copy((tool_start,))
        result = cad.intersect(support_copy, tool_copy)
        area = sum(float(cad.area(item)) for item in result.of_dimension(2))
        cad.remove(result.outputs, recursive=True)
    except Exception as error:
        raise TopologyResolutionError(
            "face-sketch-boolean.profile.overlap-unresolved: "
            "无法校验轮廓与工作面的材料重叠"
        ) from error
    tolerance = (
        float(cad.effective_bounding_box_tolerance(1.0e-9)) ** 2
        if hasattr(cad, "effective_bounding_box_tolerance")
        else 1.0e-18
    )
    if not math.isfinite(area) or area <= tolerance:
        raise TopologyResolutionError(
            "face-sketch-boolean.profile.no-overlap: 参与轮廓与工作面没有正面积材料重叠"
        )
    return area


def _resolve_current_face_seed(
    cad: Any,
    target: CompiledRecipeTopology,
    tool_start: Any,
) -> FaceSeedConnectionProof:
    for logical_id, entities in sorted(target.logical_entities.items()):
        if not logical_id.startswith("face:"):
            continue
        for entity in entities:
            if entity.dimension != 2:
                continue
            if str(cad.geometry_type(entity)).casefold() != "plane":
                continue
            try:
                area = _positive_face_overlap(cad, entity, tool_start)
            except TopologyResolutionError:
                continue
            return FaceSeedConnectionProof(logical_id, "face:bottom", area)
    raise TopologyResolutionError(
        "face-sketch-boolean.fuse.face-seed-unresolved: "
        "当前结果与工具体起始面没有可证明的正面积面种子连接"
    )


def _require_unaffected_body_isolation(
    cad: Any,
    base: CompiledRecipeTopology,
    target_volume: Any,
    result_volume: Any,
) -> None:
    tolerance = (
        float(cad.effective_bounding_box_tolerance(1.0e-9))
        if hasattr(cad, "effective_bounding_box_tolerance")
        else 1.0e-9
    )
    for volume in base.domain:
        if volume == target_volume:
            continue
        if float(cad.distance(result_volume, volume)) <= tolerance:
            raise TopologyResolutionError(
                "face-sketch-boolean.multibody.isolation: "
                "结果与非目标 Body 相交或接触"
            )


def _assemble_face_sketch_result(
    base: CompiledRecipeTopology,
    original_target: Any,
    target_body_id: str,
    proof: Any,
) -> tuple[_CompiledDraft, Mapping[str, tuple[Any, ...]]]:
    target_logical = dict(proof.logical_entities)
    if target_body_id == "body:domain":
        return (
            _CompiledDraft(
                (proof.result_volume,),
                target_logical,
                {},
            ),
            target_logical,
        )
    target_logical[target_body_id] = target_logical.pop("body:domain")
    owner = target_body_id.split(":", 1)[1]
    target_prefix = f"{owner}/"
    unaffected = {
        logical_id: entities
        for logical_id, entities in base.logical_entities.items()
        if logical_id != "body:domain"
        and logical_id != target_body_id
        and not logical_id.split(":", 1)[1].startswith(target_prefix)
    }
    logical = {**unaffected, **target_logical}
    domain = tuple(
        proof.result_volume if volume == original_target else volume
        for volume in base.domain
    )
    logical["body:domain"] = domain
    return _CompiledDraft(domain, logical, {}), target_logical


def _compile_strict_body_boolean(
    cad: Any,
    recipe: BooleanGeometry,
) -> _CompiledDraft:
    part_context = recipe.part_context
    context = (
        recipe.body_context
        if part_context is None
        else localize_part_boolean_context(part_context)
    )
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
    persisted_entities = context.result_entities
    persisted_mappings = context.topology_mappings
    if part_context is not None:
        persisted = namespace_part_boolean_context(
            feature_id=part_context.feature_id,
            target_part_id=part_context.target_part_id,
            tool_part_id=part_context.tool_part_id,
            result_part_id=part_context.result_part_id,
            result_entities=proof.result_entities,
            topology_mappings=proof.topology_mappings,
        )
        replay_matches = (
            frozenset(persisted.result_entities)
            == frozenset(part_context.result_entities)
            and frozenset(persisted.topology_mappings)
            == frozenset(part_context.topology_mappings)
        )
    else:
        replay_matches = (
            frozenset(proof.result_entities)
            == frozenset(persisted_entities)
            and frozenset(proof.topology_mappings)
            == frozenset(persisted_mappings)
        )
    if not replay_matches:
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
    if isinstance(recipe, FaceSketchBooleanGeometry):
        if len(recipe.step_proofs) != len(recipe.participating_profile_ids):
            raise TopologyResolutionError(
                "face-sketch-boolean.lineage.unproven: 面草图布尔缺少完整逐轮廓证明"
            )
        return _compile_face_sketch_boolean(cad, recipe).domain
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
    if isinstance(recipe, RevolvedGeometry):
        return _compile_revolution(cad, recipe).domain
    if isinstance(recipe, PathSweptGeometry):
        return _compile_path_sweep(cad, recipe).domain
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
