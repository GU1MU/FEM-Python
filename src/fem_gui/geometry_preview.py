"""Deterministic display tessellation for native geometry recipes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    MovedGeometry,
    MultiBodyGeometry,
    NativeGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchArc,
    SketchCircle,
    SketchGeometry,
    StrictBodyBooleanPreview,
    SketchLine,
    WireGeometry,
    analyze_sketch_profiles,
    axis_aligned_rectangle,
    expand_sketch_recipe,
    geometry_dimension,
    resolve_extrusion_source_faces,
    transformed_circle,
)
from fem.geometry.recipe_topology import (
    EntityKind,
    RecipeTopology,
    describe_recipe_topology,
)


@dataclass(frozen=True, slots=True)
class GeometryPreview:
    """Backend-neutral display cells with explicit selectable logical IDs."""

    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]
    face_logical_ids: tuple[str | None, ...] = ()
    edge_logical_ids: tuple[str | None, ...] = ()
    point_logical_ids: tuple[str | None, ...] = ()
    body_logical_id: str | None = None
    topological_dimension: Literal[1, 2, 3] = 2
    face_body_logical_ids: tuple[str | None, ...] = ()
    edge_body_logical_ids: tuple[str | None, ...] = ()
    point_body_logical_ids: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.topological_dimension, bool)
            or not isinstance(self.topological_dimension, int)
            or self.topological_dimension not in {1, 2, 3}
        ):
            raise ValueError(
                "geometry preview topological_dimension must be 1, 2, or 3"
            )
        for logical_name, cells in (
            ("face_logical_ids", self.faces),
            ("edge_logical_ids", self.edges),
            ("point_logical_ids", self.points),
        ):
            logical_ids = tuple(getattr(self, logical_name))
            if not logical_ids:
                logical_ids = (None,) * len(cells)
                object.__setattr__(self, logical_name, logical_ids)
            if len(logical_ids) != len(cells):
                raise ValueError(f"{logical_name} 必须与对应显示实体数量一致")
            for logical_id in logical_ids:
                if logical_id is not None:
                    LogicalEntityRef(logical_id)
        for logical_name, cells in (
            ("face_body_logical_ids", self.faces),
            ("edge_body_logical_ids", self.edges),
            ("point_body_logical_ids", self.points),
        ):
            logical_ids = tuple(getattr(self, logical_name))
            if not logical_ids:
                logical_ids = (self.body_logical_id,) * len(cells)
                object.__setattr__(self, logical_name, logical_ids)
            if len(logical_ids) != len(cells):
                raise ValueError(f"{logical_name} 必须与对应显示实体数量一致")
            for logical_id in logical_ids:
                if logical_id is not None:
                    reference = LogicalEntityRef(logical_id)
                    if reference.kind != "body":
                        raise ValueError(
                            f"{logical_name} 只能包含 body logical IDs"
                        )
        if self.body_logical_id is not None:
            reference = LogicalEntityRef(self.body_logical_id)
            if reference.kind != "body":
                raise ValueError("body_logical_id 必须引用 body")

    @property
    def dimension(self) -> int:
        """Return the display topology dimension used by legacy callers."""

        return self.topological_dimension


def _validate_preview_logical_ids(
    topology: RecipeTopology,
    recipe_type: str,
    kind: EntityKind,
    logical_ids: tuple[str | None, ...],
) -> None:
    for logical_id in logical_ids:
        if logical_id is None:
            continue
        try:
            entity = topology.entity(logical_id)
        except KeyError as exc:
            raise RuntimeError(
                f"{recipe_type} 预览引用了 catalog 中不存在的 "
                f"{kind} logical_id: {logical_id}"
            ) from exc
        if entity.kind != kind or not entity.selectable:
            raise RuntimeError(
                f"{recipe_type} 预览引用了不可选的 {kind} "
                f"logical_id: {logical_id}"
            )


def _make_preview(
    recipe: NativeGeometry,
    points: tuple[tuple[float, float, float], ...],
    faces: tuple[tuple[int, ...], ...],
    edges: tuple[tuple[int, ...], ...],
    face_logical_ids: tuple[str | None, ...],
    edge_logical_ids: tuple[str | None, ...],
    point_logical_ids: tuple[str | None, ...],
) -> GeometryPreview:
    """Build a display preview containing stable logical identity only."""
    topology = describe_recipe_topology(recipe)
    recipe_type = type(recipe).__name__
    for kind, logical_ids in (
        ("point", point_logical_ids),
        ("edge", edge_logical_ids),
        ("face", face_logical_ids),
    ):
        _validate_preview_logical_ids(
            topology,
            recipe_type,
            kind,
            logical_ids,
        )
    selectable_bodies = topology.entities_of("body", selectable_only=True)
    body_logical_id = (
        selectable_bodies[0].logical_id
        if len(selectable_bodies) == 1
        else None
    )
    return GeometryPreview(
        points=points,
        faces=faces,
        edges=edges,
        face_logical_ids=face_logical_ids,
        edge_logical_ids=edge_logical_ids,
        point_logical_ids=point_logical_ids,
        body_logical_id=body_logical_id,
        topological_dimension=geometry_dimension(recipe),
    )


def build_geometry_preview(
    recipe: NativeGeometry,
    *,
    segments: int = 48,
) -> GeometryPreview:
    """Build a deterministic display mesh and verify its logical-ID contract."""
    preview = _build_geometry_preview(recipe, max(12, int(segments)))
    _validate_preview_topology(recipe, preview)
    return preview


def build_strict_sketch_draft_preview(
    recipe: SketchGeometry,
    *,
    segments: int = 48,
) -> GeometryPreview:
    """Preview analyzable profiles even while other draft curves are invalid."""

    if type(recipe) is not SketchGeometry or not recipe.is_strict:
        raise TypeError("draft preview requires a strict SketchGeometry")
    return _strict_sketch_preview(
        recipe,
        max(12, int(segments)),
        allow_partial=True,
    )


def build_strict_body_boolean_preview(
    recipe: MultiBodyGeometry,
    boolean_preview: StrictBodyBooleanPreview,
    *,
    segments: int = 48,
) -> GeometryPreview:
    """Merge one true OCC Boolean tessellation with unaffected Body previews."""

    return build_strict_body_boolean_previews(
        recipe,
        (boolean_preview,),
        segments=segments,
    )


def build_strict_body_boolean_previews(
    recipe: MultiBodyGeometry,
    boolean_previews: tuple[StrictBodyBooleanPreview, ...],
    *,
    segments: int = 48,
) -> GeometryPreview:
    """Merge all replayed strict Bodies with deterministic unaffected Bodies."""

    if type(recipe) is not MultiBodyGeometry:
        raise TypeError("strict Boolean preview requires MultiBodyGeometry")
    if any(
        type(preview) is not StrictBodyBooleanPreview
        for preview in boolean_previews
    ):
        raise TypeError(
            "boolean_previews must contain StrictBodyBooleanPreview values"
        )
    preview_by_body = {
        preview.target_body_id: preview
        for preview in boolean_previews
    }
    if len(preview_by_body) != len(boolean_previews):
        raise ValueError("strict Boolean previews contain duplicate Body IDs")
    for body_id in preview_by_body:
        recipe.body(body_id)
    local_by_body: dict[str, GeometryPreview] = {}
    for body in recipe.bodies:
        boolean_preview = preview_by_body.get(body.id)
        if boolean_preview is not None:
            local_by_body[body.id] = GeometryPreview(
                boolean_preview.points,
                boolean_preview.faces,
                boolean_preview.edges,
                boolean_preview.face_logical_ids,
                boolean_preview.edge_logical_ids,
                boolean_preview.point_logical_ids,
                "body:domain",
                3,
            )
        else:
            local_by_body[body.id] = _build_geometry_preview(
                body.recipe,
                max(12, int(segments)),
            )
    preview = _merge_multi_body_previews(recipe, local_by_body)
    _validate_preview_topology(
        recipe,
        preview,
        allow_occ_fallback=False,
    )
    return preview


def _build_geometry_preview(
    recipe: NativeGeometry,
    segments: int,
) -> GeometryPreview:
    if isinstance(recipe, SketchGeometry):
        if recipe.is_strict:
            return _strict_sketch_preview(recipe, segments)
        preview = _build_geometry_preview(expand_sketch_recipe(recipe), segments)
        return _make_preview(
            recipe,
            preview.points,
            preview.faces,
            preview.edges,
            preview.face_logical_ids,
            preview.edge_logical_ids,
            preview.point_logical_ids,
        )
    if isinstance(recipe, BooleanGeometry):
        return _boolean_preview(recipe, segments)
    if isinstance(recipe, MultiBodyGeometry):
        return _multi_body_preview(recipe, segments)
    if isinstance(recipe, MovedGeometry):
        preview = _build_geometry_preview(recipe.base, segments)
        return _make_preview(
            recipe,
            tuple(
                (x + recipe.dx, y + recipe.dy, z + recipe.dz)
                for x, y, z in preview.points
            ),
            preview.faces,
            preview.edges,
            preview.face_logical_ids,
            preview.edge_logical_ids,
            preview.point_logical_ids,
        )
    if isinstance(recipe, RotatedGeometry):
        preview = _build_geometry_preview(recipe.base, segments)
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)

        def rotate(point):
            x, y, z = point
            if recipe.axis == "x":
                return x, y * cosine - z * sine, y * sine + z * cosine
            if recipe.axis == "y":
                return x * cosine + z * sine, y, -x * sine + z * cosine
            return x * cosine - y * sine, x * sine + y * cosine, z

        return _make_preview(
            recipe,
            tuple(rotate(point) for point in preview.points),
            preview.faces,
            preview.edges,
            preview.face_logical_ids,
            preview.edge_logical_ids,
            preview.point_logical_ids,
        )
    if isinstance(recipe, ExtrudedGeometry):
        return _extruded_preview(recipe, segments)
    if isinstance(recipe, WireGeometry):
        return _wire_preview(recipe)
    if isinstance(recipe, RectangleGeometry):
        return _rectangle_preview(recipe)
    if isinstance(recipe, BoxGeometry):
        return _box_preview(recipe)
    if isinstance(recipe, DiskGeometry):
        return _disk_preview(recipe, segments)
    if isinstance(recipe, PlateWithHoleGeometry):
        return _plate_with_hole_preview(recipe, segments)
    return _cylinder_preview(recipe, segments)


def _multi_body_preview(
    recipe: MultiBodyGeometry,
    segments: int,
) -> GeometryPreview:
    return _merge_multi_body_previews(
        recipe,
        {
            body.id: _build_geometry_preview(body.recipe, segments)
            for body in recipe.bodies
        },
    )


def _merge_multi_body_previews(
    recipe: MultiBodyGeometry,
    local_by_body: dict[str, GeometryPreview],
) -> GeometryPreview:
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    edges: list[tuple[int, ...]] = []
    face_ids: list[str | None] = []
    edge_ids: list[str | None] = []
    point_ids: list[str | None] = []
    face_body_ids: list[str | None] = []
    edge_body_ids: list[str | None] = []
    point_body_ids: list[str | None] = []

    def namespace(body_id: str, logical_id: str | None) -> str | None:
        if logical_id is None:
            return None
        reference = LogicalEntityRef(logical_id)
        if reference.kind == "body":
            return f"body:{body_id}"
        _kind, local_name = logical_id.split(":", 1)
        return f"{reference.kind}:{body_id}/{local_name}"

    for body in recipe.bodies:
        local = local_by_body[body.id]
        offset = len(points)
        points.extend(local.points)
        faces.extend(
            tuple(index + offset for index in face)
            for face in local.faces
        )
        edges.extend(
            tuple(index + offset for index in edge)
            for edge in local.edges
        )
        face_ids.extend(
            namespace(body.id, logical_id)
            for logical_id in local.face_logical_ids
        )
        edge_ids.extend(
            namespace(body.id, logical_id)
            for logical_id in local.edge_logical_ids
        )
        point_ids.extend(
            namespace(body.id, logical_id)
            for logical_id in local.point_logical_ids
        )
        body_logical_id = f"body:{body.id}"
        face_body_ids.extend((body_logical_id,) * len(local.faces))
        edge_body_ids.extend((body_logical_id,) * len(local.edges))
        point_body_ids.extend((body_logical_id,) * len(local.points))
    return GeometryPreview(
        tuple(points),
        tuple(faces),
        tuple(edges),
        tuple(face_ids),
        tuple(edge_ids),
        tuple(point_ids),
        None,
        3,
        tuple(face_body_ids),
        tuple(edge_body_ids),
        tuple(point_body_ids),
    )


def _wire_preview(recipe: WireGeometry) -> GeometryPreview:
    """Build an exact, face-free preview for one named spatial wire."""

    point_indices = {
        point.name: index for index, point in enumerate(recipe.points)
    }
    points = tuple((point.x, point.y, point.z) for point in recipe.points)
    edges = tuple(
        (point_indices[member.start], point_indices[member.end])
        for member in recipe.members
    )
    return _make_preview(
        recipe,
        points,
        (),
        edges,
        (),
        tuple(f"edge:{member.name}" for member in recipe.members),
        tuple(f"point:{point.name}" for point in recipe.points),
    )


def _strict_sketch_preview(
    recipe: SketchGeometry,
    segments: int,
    *,
    allow_partial: bool = False,
) -> GeometryPreview:
    """Tessellate a strict sketch while retaining every stable logical ID."""

    if not recipe.is_strict or recipe.plane is None:
        raise TypeError("strict sketch preview requires a curve-first sketch")
    analysis = analyze_sketch_profiles(recipe)
    if (
        (analysis.blocking_diagnostics and not allow_partial)
        or not analysis.profiles
    ):
        message = (
            analysis.blocking_diagnostics[0].message
            if analysis.blocking_diagnostics
            else "严格草图没有可显示的 Profile"
        )
        raise ValueError(message)

    points: list[tuple[float, float, float]] = [
        recipe.plane.to_global(point.u, point.v)
        for point in recipe.points
    ]
    local_points: list[tuple[float, float]] = [
        (point.u, point.v) for point in recipe.points
    ]
    point_logical_ids: list[str | None] = [
        f"point:{point.id}" for point in recipe.points
    ]
    point_indices = {
        point.id: index for index, point in enumerate(recipe.points)
    }
    point_cells_by_id: dict[str, tuple[int, ...]] = {
        f"point:{point.id}": (point_indices[point.id],)
        for point in recipe.points
    }

    def add_tessellation_point(u: float, v: float) -> int:
        index = len(points)
        local_points.append((float(u), float(v)))
        points.append(recipe.plane.to_global(u, v))
        point_logical_ids.append(None)
        return index

    curve_paths: dict[str, tuple[int, ...]] = {}
    for curve in recipe.curves:
        if isinstance(curve, SketchLine):
            path = (
                point_indices[curve.start_point_id],
                point_indices[curve.end_point_id],
            )
        elif isinstance(curve, SketchCircle):
            if curve.center_point_id is None:
                raise TypeError("strict circle center point is required")
            center = recipe.point(curve.center_point_id)
            perimeter = tuple(
                add_tessellation_point(
                    center.u + curve.radius * math.cos(angle),
                    center.v + curve.radius * math.sin(angle),
                )
                for angle in _angles(segments)
            )
            path = perimeter + (perimeter[0],)
        elif isinstance(curve, SketchArc):
            start = recipe.point(curve.start_point_id)
            center = recipe.point(curve.center_point_id)
            end = recipe.point(curve.end_point_id)
            start_angle = math.atan2(start.v - center.v, start.u - center.u)
            end_angle = math.atan2(end.v - center.v, end.u - center.u)
            if curve.orientation == "ccw":
                sweep = (end_angle - start_angle) % (2.0 * math.pi)
            else:
                sweep = -((start_angle - end_angle) % (2.0 * math.pi))
            count = max(
                2,
                int(math.ceil(segments * abs(sweep) / (2.0 * math.pi))),
            )
            radius = math.hypot(start.u - center.u, start.v - center.v)
            path_values = [point_indices[curve.start_point_id]]
            for index in range(1, count):
                angle = start_angle + sweep * index / count
                path_values.append(
                    add_tessellation_point(
                        center.u + radius * math.cos(angle),
                        center.v + radius * math.sin(angle),
                    )
                )
            path_values.append(point_indices[curve.end_point_id])
            path = tuple(path_values)
        else:  # pragma: no cover - strict SketchGeometry validates curve types
            raise TypeError(f"Unsupported strict sketch curve: {type(curve).__name__}")
        curve_paths[curve.id] = path

    edges: list[tuple[int, ...]] = []
    edge_logical_ids: list[str | None] = []
    edge_cells_by_id: dict[str, tuple[int, ...]] = {}
    for curve in recipe.curves:
        cell_index = len(edges)
        edges.append(curve_paths[curve.id])
        logical_id = f"edge:{curve.id}"
        edge_logical_ids.append(logical_id)
        edge_cells_by_id[logical_id] = (cell_index,)

    profile_rings = {
        profile.id: _strict_profile_ring(profile.curve_ids, curve_paths)
        for profile in analysis.profiles
    }
    material_profiles = tuple(
        profile for profile in analysis.profiles if profile.is_material
    )
    hole_profiles = tuple(
        profile for profile in analysis.profiles if profile.is_hole
    )
    faces: list[tuple[int, ...]] = []
    face_logical_ids: list[str | None] = []
    face_cells_by_id: dict[str, tuple[int, ...]] = {}
    for profile in material_profiles:
        holes = tuple(
            profile_rings[hole.id]
            for hole in hole_profiles
            if hole.parent_profile_id == profile.id
        )
        triangles = _triangulate_strict_profile(
            profile_rings[profile.id],
            holes,
            tuple(local_points),
        )
        logical_id = f"face:profile/{profile.id.split('/', 1)[-1]}"
        start = len(faces)
        faces.extend(triangles)
        face_logical_ids.extend((logical_id,) * len(triangles))
        face_cells_by_id[logical_id] = tuple(
            range(start, start + len(triangles))
        )

    if not analysis.blocking_diagnostics:
        topology = describe_recipe_topology(recipe)
        _append_preview_alias_cells(
            topology,
            points,
            point_logical_ids,
            point_cells_by_id,
            edges,
            edge_logical_ids,
            edge_cells_by_id,
            faces,
            face_logical_ids,
            face_cells_by_id,
        )
    elif allow_partial:
        return GeometryPreview(
            tuple(points),
            tuple(faces),
            tuple(edges),
            tuple(face_logical_ids),
            tuple(edge_logical_ids),
            tuple(point_logical_ids),
            topological_dimension=2,
        )
    return _make_preview(
        recipe,
        tuple(points),
        tuple(faces),
        tuple(edges),
        tuple(face_logical_ids),
        tuple(edge_logical_ids),
        tuple(point_logical_ids),
    )


def _strict_profile_ring(
    signed_curve_ids: tuple[str, ...],
    curve_paths: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    ring: list[int] = []
    for signed_curve_id in signed_curve_ids:
        path = curve_paths[signed_curve_id.lstrip("-")]
        if signed_curve_id.startswith("-"):
            path = tuple(reversed(path))
        if ring and ring[-1] == path[0]:
            ring.extend(path[1:])
        else:
            ring.extend(path)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    if len(ring) < 3:
        raise ValueError("严格草图 Profile 至少需要三个显示顶点")
    return tuple(ring)


def _triangulate_strict_profile(
    outer: tuple[int, ...],
    holes: tuple[tuple[int, ...], ...],
    local_points: tuple[tuple[float, float], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Triangulate sampled planar rings and discard cells outside the region."""

    from scipy.spatial import Delaunay

    candidate_indices = tuple(
        dict.fromkeys(
            index
            for ring in (outer, *holes)
            for index in ring
        )
    )
    coordinates = tuple(local_points[index] for index in candidate_indices)
    if len(candidate_indices) < 3:
        raise ValueError("严格草图 Profile 无法形成显示面")
    if len(candidate_indices) == 3 and not holes:
        return (tuple(candidate_indices),)
    triangulation = Delaunay(coordinates)
    boundary_segments = tuple(
        (ring[index], ring[(index + 1) % len(ring)])
        for ring in (outer, *holes)
        for index in range(len(ring))
    )

    def in_region(point: tuple[float, float]) -> bool:
        outer_state = _point_in_preview_ring(point, outer, local_points)
        if outer_state == 0:
            return False
        return not any(
            _point_in_preview_ring(point, hole, local_points) == 1
            for hole in holes
        )

    accepted: list[tuple[int, int, int]] = []
    for simplex in triangulation.simplices:
        triangle = tuple(candidate_indices[int(index)] for index in simplex)
        values = tuple(local_points[index] for index in triangle)
        signed_area = _preview_triangle_area(*values)
        if math.isclose(signed_area, 0.0, abs_tol=1.0e-14):
            continue
        if signed_area < 0.0:
            triangle = (triangle[0], triangle[2], triangle[1])
            values = (values[0], values[2], values[1])
        probes = (
            (
                sum(value[0] for value in values) / 3.0,
                sum(value[1] for value in values) / 3.0,
            ),
            *tuple(
                (
                    0.5 * (values[index][0] + values[(index + 1) % 3][0]),
                    0.5 * (values[index][1] + values[(index + 1) % 3][1]),
                )
                for index in range(3)
            ),
        )
        if not all(in_region(point) for point in probes):
            continue
        if any(
            _preview_segment_crosses_boundary(
                triangle[index],
                triangle[(index + 1) % 3],
                boundary_segments,
                local_points,
            )
            for index in range(3)
        ):
            continue
        accepted.append(triangle)
    if not accepted:
        raise ValueError("严格草图 Profile 无法生成有效显示三角形")
    return tuple(sorted(set(accepted)))


def _point_in_preview_ring(
    point: tuple[float, float],
    ring: tuple[int, ...],
    local_points: tuple[tuple[float, float], ...],
) -> int:
    """Return 0 outside, 1 inside, or 2 on the sampled ring boundary."""

    x, y = point
    inside = False
    for index in range(len(ring)):
        start = local_points[ring[index]]
        end = local_points[ring[(index + 1) % len(ring)]]
        if _preview_point_on_segment(point, start, end):
            return 2
        if (start[1] > y) != (end[1] > y):
            crossing_x = (
                start[0]
                + (y - start[1])
                * (end[0] - start[0])
                / (end[1] - start[1])
            )
            if crossing_x > x:
                inside = not inside
    return 1 if inside else 0


def _preview_point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    area = _preview_triangle_area(start, end, point)
    if not math.isclose(area, 0.0, abs_tol=1.0e-10):
        return False
    return (
        min(start[0], end[0]) - 1.0e-10
        <= point[0]
        <= max(start[0], end[0]) + 1.0e-10
        and min(start[1], end[1]) - 1.0e-10
        <= point[1]
        <= max(start[1], end[1]) + 1.0e-10
    )


def _preview_triangle_area(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return 0.5 * (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _preview_segment_crosses_boundary(
    start_index: int,
    end_index: int,
    boundary_segments: tuple[tuple[int, int], ...],
    local_points: tuple[tuple[float, float], ...],
) -> bool:
    start = local_points[start_index]
    end = local_points[end_index]
    for boundary_start_index, boundary_end_index in boundary_segments:
        if {start_index, end_index} & {
            boundary_start_index,
            boundary_end_index,
        }:
            continue
        boundary_start = local_points[boundary_start_index]
        boundary_end = local_points[boundary_end_index]
        first = _preview_triangle_area(start, end, boundary_start)
        second = _preview_triangle_area(start, end, boundary_end)
        third = _preview_triangle_area(
            boundary_start,
            boundary_end,
            start,
        )
        fourth = _preview_triangle_area(
            boundary_start,
            boundary_end,
            end,
        )
        if first * second < -1.0e-14 and third * fourth < -1.0e-14:
            return True
    return False


def _append_preview_alias_cells(
    topology: RecipeTopology,
    points: list[tuple[float, float, float]],
    point_logical_ids: list[str | None],
    point_cells_by_id: dict[str, tuple[int, ...]],
    edges: list[tuple[int, ...]],
    edge_logical_ids: list[str | None],
    edge_cells_by_id: dict[str, tuple[int, ...]],
    faces: list[tuple[int, ...]],
    face_logical_ids: list[str | None],
    face_cells_by_id: dict[str, tuple[int, ...]],
) -> None:
    """Duplicate display cells for topology aliases required by validation."""

    values = {
        "point": (points, point_logical_ids, point_cells_by_id),
        "edge": (edges, edge_logical_ids, edge_cells_by_id),
        "face": (faces, face_logical_ids, face_cells_by_id),
    }
    for kind in ("point", "edge", "face"):
        cells, logical_ids, cells_by_id = values[kind]
        pending = {
            entity.logical_id: entity
            for entity in topology.entities_of(kind, selectable_only=True)
            if entity.logical_id not in cells_by_id
        }
        progressed = True
        while pending and progressed:
            progressed = False
            for logical_id, entity in tuple(pending.items()):
                linked = tuple(
                    cells_by_id.get(link, ())
                    for link in entity.topology_links
                    if link.startswith(f"{kind}:")
                )
                if not linked or any(not group for group in linked):
                    continue
                source_indices = tuple(
                    index for group in linked for index in group
                )
                appended: list[int] = []
                for source_index in source_indices:
                    if kind == "point":
                        source_point = cells[source_index]
                        appended_index = len(points)
                        points.append(source_point)
                        point_logical_ids.append(logical_id)
                    else:
                        appended_index = len(cells)
                        cells.append(cells[source_index])
                        logical_ids.append(logical_id)
                    appended.append(appended_index)
                cells_by_id[logical_id] = tuple(appended)
                del pending[logical_id]
                progressed = True


def _boolean_preview(
    recipe: BooleanGeometry,
    segments: int,
) -> GeometryPreview:
    if recipe.body_context is not None and recipe.body_context.proven:
        return _strict_body_boolean_preview(recipe, segments)
    if recipe.operation == "cut":
        outer = axis_aligned_rectangle(recipe.object_geometry)
        inner_rectangle = axis_aligned_rectangle(recipe.tool_geometry)
        if outer is not None and inner_rectangle is not None:
            x, y = outer.x, outer.y
            width, height = outer.width, outer.height
            inner_x, inner_y = inner_rectangle.x, inner_rectangle.y
            inner_width, inner_height = (
                inner_rectangle.width,
                inner_rectangle.height,
            )
            if (
                x < inner_x < inner_x + inner_width < x + width
                and y < inner_y < inner_y + inner_height < y + height
            ):
                points = (
                    (x, y, 0.0),
                    (x + width, y, 0.0),
                    (x + width, y + height, 0.0),
                    (x, y + height, 0.0),
                    (inner_x, inner_y, 0.0),
                    (inner_x + inner_width, inner_y, 0.0),
                    (
                        inner_x + inner_width,
                        inner_y + inner_height,
                        0.0,
                    ),
                    (inner_x, inner_y + inner_height, 0.0),
                )
                faces = tuple(
                    (
                        index,
                        (index + 1) % 4,
                        4 + (index + 1) % 4,
                        4 + index,
                    )
                    for index in range(4)
                )
                return _make_preview(
                    recipe,
                    points,
                    faces,
                    (
                        (4, 5, 6, 7, 4),
                        (0, 1, 2, 3, 0),
                    ),
                    ("face:domain",) * len(faces),
                    ("edge:hole-loop", "edge:outer-loop"),
                    (
                        "point:bottom-left",
                        "point:bottom-right",
                        "point:top-right",
                        "point:top-left",
                        "point:hole-bottom-left",
                        "point:hole-bottom-right",
                        "point:hole-top-right",
                        "point:hole-top-left",
                    ),
                )
        circle = transformed_circle(recipe.tool_geometry)
        if outer is not None and circle is not None:
            x, y = outer.x, outer.y
            width, height = outer.width, outer.height
            center_x, center_y, radius = (
                circle.center_x,
                circle.center_y,
                circle.radius,
            )
            local_x, local_y = center_x - x, center_y - y
            if radius < local_x < width - radius and radius < local_y < height - radius:
                ring = _plate_with_hole_preview(
                    PlateWithHoleGeometry(
                        "cut-preview",
                        width,
                        height,
                        local_x,
                        local_y,
                        radius,
                    ),
                    segments,
                )
                return _make_preview(
                    recipe,
                    tuple(
                        (point_x + x, point_y + y, point_z)
                        for point_x, point_y, point_z in ring.points
                    ),
                    ring.faces,
                    ring.edges,
                    ring.face_logical_ids,
                    ring.edge_logical_ids,
                    ring.point_logical_ids,
                )

    object_preview = _build_geometry_preview(
        recipe.object_geometry,
        segments,
    )
    tool_preview = _build_geometry_preview(recipe.tool_geometry, segments)
    offset = len(object_preview.points)
    points = object_preview.points + tool_preview.points
    tool_edges = tuple(
        tuple(index + offset for index in edge) for edge in tool_preview.edges
    )
    edges = object_preview.edges + tool_edges
    if recipe.operation == "cut":
        faces = object_preview.faces
    else:
        faces = object_preview.faces + tuple(
            tuple(index + offset for index in face) for face in tool_preview.faces
        )
    # General boolean cells are illustrative. CAD lineage is deliberately
    # unavailable until a future persistent-naming implementation proves it.
    return _make_preview(
        recipe,
        points,
        faces,
        edges,
        (None,) * len(faces),
        (None,) * len(edges),
        (None,) * len(points),
    )


def _derived_logical_id(
    topology: RecipeTopology,
    source_logical_id: str | None,
    kind: EntityKind,
    semantic_prefix: str,
) -> str | None:
    if source_logical_id is None:
        return None
    candidates = []
    for mapping in topology.transition.mappings:
        if (
            mapping.source_logical_id == source_logical_id
            and mapping.relation == "derived"
        ):
            entity = topology.entity(mapping.target_logical_id)
            if entity.kind == kind and entity.semantic_role.startswith(semantic_prefix):
                candidates.append(entity.logical_id)
    if len(candidates) != 1:
        raise RuntimeError(
            f"ExtrudedGeometry catalog 无法为 {source_logical_id} 唯一解析 "
            f"{kind}({semantic_prefix})"
        )
    return candidates[0]


def _extruded_preview(
    recipe: ExtrudedGeometry,
    segments: int,
) -> GeometryPreview:
    base = _build_geometry_preview(recipe.base, segments)
    topology = describe_recipe_topology(recipe)
    selection = resolve_extrusion_source_faces(
        recipe.base,
        recipe.source_face_ids,
    )
    selected_faces = set(selection.face_ids)
    selected_edges = set(selection.boundary_edge_ids)
    selected_points = set(selection.boundary_point_ids)
    base_face_indices = tuple(
        index
        for index, logical_id in enumerate(base.face_logical_ids)
        if logical_id in selected_faces
    )
    base_edge_indices = tuple(
        index
        for index, logical_id in enumerate(base.edge_logical_ids)
        if logical_id in selected_edges
    )
    required_point_indices = {
        point_index
        for face_index in base_face_indices
        for point_index in base.faces[face_index]
    }
    required_point_indices.update(
        point_index
        for edge_index in base_edge_indices
        for point_index in base.edges[edge_index]
    )
    required_point_indices.update(
        index
        for index, logical_id in enumerate(base.point_logical_ids)
        if logical_id in selected_points
    )
    ordered_point_indices = tuple(sorted(required_point_indices))
    point_index_map = {
        old_index: new_index
        for new_index, old_index in enumerate(ordered_point_indices)
    }
    bottom_points = tuple(base.points[index] for index in ordered_point_indices)
    point_count = len(bottom_points)
    points = bottom_points + tuple(
        (x, y, z + recipe.height)
        for x, y, z in bottom_points
    )
    bottom_faces = tuple(
        tuple(point_index_map[index] for index in base.faces[face_index])
        for face_index in base_face_indices
    )
    bottom_face_source_ids = tuple(
        base.face_logical_ids[face_index]
        for face_index in base_face_indices
    )
    faces = list(bottom_faces)
    face_logical_ids = [
        _derived_logical_id(topology, logical_id, "face", "copy.bottom.")
        for logical_id in bottom_face_source_ids
    ]
    faces.extend(
        tuple(reversed(tuple(index + point_count for index in face)))
        for face in bottom_faces
    )
    face_logical_ids.extend(
        _derived_logical_id(topology, logical_id, "face", "copy.top.")
        for logical_id in bottom_face_source_ids
    )

    bottom_edges = tuple(
        tuple(point_index_map[index] for index in base.edges[edge_index])
        for edge_index in base_edge_indices
    )
    bottom_edge_source_ids = tuple(
        base.edge_logical_ids[edge_index]
        for edge_index in base_edge_indices
    )
    for edge, logical_id in zip(
        bottom_edges,
        bottom_edge_source_ids,
        strict=True,
    ):
        for start, end in zip(edge, edge[1:]):
            faces.append((start, end, end + point_count, start + point_count))
            face_logical_ids.append(
                _derived_logical_id(topology, logical_id, "face", "sweep.")
            )

    edges = list(bottom_edges)
    edges.extend(
        tuple(index + point_count for index in edge)
        for edge in bottom_edges
    )
    edge_logical_ids = [
        _derived_logical_id(topology, logical_id, "edge", "copy.bottom.")
        for logical_id in bottom_edge_source_ids
    ]
    edge_logical_ids.extend(
        _derived_logical_id(topology, logical_id, "edge", "copy.top.")
        for logical_id in bottom_edge_source_ids
    )
    bottom_point_source_ids = tuple(
        (
            logical_id
            if logical_id in selected_points
            else None
        )
        for logical_id in (
            base.point_logical_ids[index]
            for index in ordered_point_indices
        )
    )
    point_logical_ids = tuple(
        _derived_logical_id(topology, logical_id, "point", "copy.bottom.")
        for logical_id in bottom_point_source_ids
    ) + tuple(
        _derived_logical_id(topology, logical_id, "point", "copy.top.")
        for logical_id in bottom_point_source_ids
    )
    feature_points = tuple(
        index
        for index, logical_id in enumerate(bottom_point_source_ids)
        if logical_id is not None
    )
    edges.extend((index, index + point_count) for index in feature_points)
    edge_logical_ids.extend(
        _derived_logical_id(
            topology,
            bottom_point_source_ids[index],
            "edge",
            "sweep.",
        )
        for index in feature_points
    )
    return _make_preview(
        recipe,
        points,
        tuple(faces),
        tuple(edges),
        tuple(face_logical_ids),
        tuple(edge_logical_ids),
        point_logical_ids,
    )


def _strict_body_boolean_preview(
    recipe: BooleanGeometry,
    segments: int,
) -> GeometryPreview:
    """Build an explicitly unselectable fallback until OCC data is available."""

    target = _build_geometry_preview(recipe.object_geometry, segments)
    return _make_preview(
        recipe,
        target.points,
        target.faces,
        target.edges,
        (None,) * len(target.faces),
        (None,) * len(target.edges),
        (None,) * len(target.points),
    )


def _rectangle_preview(recipe: RectangleGeometry) -> GeometryPreview:
    points = (
        (0.0, 0.0, 0.0),
        (recipe.width, 0.0, 0.0),
        (recipe.width, recipe.height, 0.0),
        (0.0, recipe.height, 0.0),
    )
    return _make_preview(
        recipe,
        points,
        ((0, 1, 2, 3),),
        ((0, 1), (1, 2), (2, 3), (3, 0)),
        ("face:domain",),
        ("edge:bottom", "edge:right", "edge:top", "edge:left"),
        (
            "point:bottom-left",
            "point:bottom-right",
            "point:top-right",
            "point:top-left",
        ),
    )


def _box_preview(recipe: BoxGeometry) -> GeometryPreview:
    width, depth, height = recipe.width, recipe.depth, recipe.height
    points = (
        (0.0, 0.0, 0.0),
        (width, 0.0, 0.0),
        (width, depth, 0.0),
        (0.0, depth, 0.0),
        (0.0, 0.0, height),
        (width, 0.0, height),
        (width, depth, height),
        (0.0, depth, height),
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    return _make_preview(
        recipe,
        points,
        faces,
        edges,
        (
            "face:bottom",
            "face:top",
            "face:front",
            "face:right",
            "face:back",
            "face:left",
        ),
        (
            "edge:bottom-front",
            "edge:bottom-right",
            "edge:bottom-back",
            "edge:bottom-left",
            "edge:top-front",
            "edge:top-right",
            "edge:top-back",
            "edge:top-left",
            "edge:vertical-front-left",
            "edge:vertical-front-right",
            "edge:vertical-back-right",
            "edge:vertical-back-left",
        ),
        (
            "point:bottom-front-left",
            "point:bottom-front-right",
            "point:bottom-back-right",
            "point:bottom-back-left",
            "point:top-front-left",
            "point:top-front-right",
            "point:top-back-right",
            "point:top-back-left",
        ),
    )


def _disk_preview(
    recipe: DiskGeometry,
    segments: int,
) -> GeometryPreview:
    angles = _angles(segments)
    points = ((0.0, 0.0, 0.0),) + tuple(
        (
            recipe.radius * math.cos(angle),
            recipe.radius * math.sin(angle),
            0.0,
        )
        for angle in angles
    )
    faces = tuple(
        (0, 1 + index, 1 + (index + 1) % segments) for index in range(segments)
    )
    edge = tuple(range(1, segments + 1)) + (1,)
    return _make_preview(
        recipe,
        points,
        faces,
        (edge,),
        ("face:domain",) * len(faces),
        ("edge:outer",),
        (None,) * len(points),
    )


def _plate_with_hole_preview(
    recipe: PlateWithHoleGeometry,
    segments: int,
) -> GeometryPreview:
    hole_rays = _rectangle_hole_rays(recipe, segments)
    hole_angles = tuple(angle for angle, _logical_id in hole_rays)
    count = len(hole_angles)
    inner = tuple(
        (
            recipe.hole_x + recipe.hole_radius * math.cos(angle),
            recipe.hole_y + recipe.hole_radius * math.sin(angle),
            0.0,
        )
        for angle in hole_angles
    )
    outer = []
    for angle in hole_angles:
        delta_x, delta_y = math.cos(angle), math.sin(angle)
        distances = []
        if delta_x > 0.0:
            distances.append((recipe.width - recipe.hole_x) / delta_x)
        elif delta_x < 0.0:
            distances.append(-recipe.hole_x / delta_x)
        if delta_y > 0.0:
            distances.append((recipe.height - recipe.hole_y) / delta_y)
        elif delta_y < 0.0:
            distances.append(-recipe.hole_y / delta_y)
        distance = min(value for value in distances if value > 0.0)
        outer.append(
            (
                recipe.hole_x + distance * delta_x,
                recipe.hole_y + distance * delta_y,
                0.0,
            )
        )
    points = inner + tuple(outer)
    faces = tuple(
        (
            index,
            count + index,
            count + (index + 1) % count,
            (index + 1) % count,
        )
        for index in range(count)
    )
    return _make_preview(
        recipe,
        points,
        faces,
        (
            tuple(range(count)) + (0,),
            tuple(range(count, 2 * count)) + (count,),
        ),
        ("face:domain",) * len(faces),
        ("edge:hole-loop", "edge:outer-loop"),
        (None,) * count
        + tuple(logical_id for _angle, logical_id in hole_rays),
    )


def _cylinder_preview(
    recipe: CylinderGeometry,
    segments: int,
) -> GeometryPreview:
    angles = _angles(segments)
    bottom = tuple(
        (
            recipe.radius * math.cos(angle),
            recipe.radius * math.sin(angle),
            0.0,
        )
        for angle in angles
    )
    top = tuple((x, y, recipe.height) for x, y, _z in bottom)
    points = ((0.0, 0.0, 0.0), (0.0, 0.0, recipe.height)) + bottom + top
    bottom_start, top_start = 2, 2 + segments
    faces = []
    face_logical_ids = []
    for index in range(segments):
        next_index = (index + 1) % segments
        faces.extend(
            (
                (0, bottom_start + next_index, bottom_start + index),
                (1, top_start + index, top_start + next_index),
                (
                    bottom_start + index,
                    bottom_start + next_index,
                    top_start + next_index,
                    top_start + index,
                ),
            )
        )
        face_logical_ids.extend(("face:bottom", "face:top", "face:outer"))
    edges = (
        tuple(range(bottom_start, bottom_start + segments)) + (bottom_start,),
        tuple(range(top_start, top_start + segments)) + (top_start,),
    )
    return _make_preview(
        recipe,
        points,
        tuple(faces),
        edges,
        tuple(face_logical_ids),
        ("edge:bottom-rim", "edge:top-rim"),
        (None,) * len(points),
    )


def _validate_preview_topology(
    recipe: NativeGeometry,
    preview: GeometryPreview,
    *,
    allow_occ_fallback: bool = True,
) -> None:
    topology = describe_recipe_topology(recipe)
    if preview.topological_dimension != topology.dimension:
        raise RuntimeError(
            f"{type(recipe).__name__} 预览 dimension 与 recipe topology 不一致: "
            f"{preview.topological_dimension} != {topology.dimension}"
        )
    for kind, cells, logical_ids in (
        ("point", preview.points, preview.point_logical_ids),
        ("edge", preview.edges, preview.edge_logical_ids),
        ("face", preview.faces, preview.face_logical_ids),
    ):
        if len(logical_ids) != len(cells):
            raise RuntimeError(
                f"{type(recipe).__name__} 预览的 {kind} cell 缺少 logical_id 绑定"
            )
        _validate_preview_logical_ids(
            topology,
            type(recipe).__name__,
            kind,
            logical_ids,
        )
        actual = {
            logical_id
            for logical_id in logical_ids
            if logical_id is not None
        }
        expected = {
            entity.logical_id
            for entity in topology.entities_of(kind, selectable_only=True)
        }
        incomplete_allowed = (
            allow_occ_fallback
            and _contains_proven_strict_boolean(recipe)
        )
        if actual != expected and not (
            incomplete_allowed and actual.issubset(expected)
        ):
            raise RuntimeError(
                f"{type(recipe).__name__} 预览的 {kind} logical ID"
                f"与 recipe topology 不一致: {sorted(actual)} != "
                f"{sorted(expected)}"
            )
    actual_body = {
        logical_id
        for logical_ids in (
            preview.face_body_logical_ids,
            preview.edge_body_logical_ids,
            preview.point_body_logical_ids,
        )
        for logical_id in logical_ids
        if logical_id is not None
    }
    if preview.body_logical_id is not None:
        actual_body.add(preview.body_logical_id)
    expected_body = {
        entity.logical_id
        for entity in topology.entities_of("body", selectable_only=True)
    }
    if actual_body != expected_body:
        raise RuntimeError(
            f"{type(recipe).__name__} 预览的 body logical ID"
            f"与 recipe topology 不一致: {sorted(actual_body)} != "
            f"{sorted(expected_body)}"
        )


def _contains_proven_strict_boolean(recipe: object) -> bool:
    if isinstance(recipe, BooleanGeometry):
        return (
            recipe.body_context is not None
            and recipe.body_context.proven
        ) or _contains_proven_strict_boolean(
            recipe.object_geometry
        ) or _contains_proven_strict_boolean(recipe.tool_geometry)
    if isinstance(recipe, (MovedGeometry, RotatedGeometry, ExtrudedGeometry)):
        return _contains_proven_strict_boolean(recipe.base)
    if isinstance(recipe, MultiBodyGeometry):
        return any(
            _contains_proven_strict_boolean(body.recipe)
            for body in recipe.bodies
        )
    return False


def _rectangle_hole_rays(
    recipe: PlateWithHoleGeometry,
    segments: int,
) -> tuple[tuple[float, str | None], ...]:
    segment_count = max(12, int(segments))
    rays: list[tuple[float, str | None]] = [
        (2.0 * math.pi * index / segment_count, None)
        for index in range(segment_count)
    ]
    corners = (
        ("point:bottom-left", 0.0, 0.0),
        ("point:bottom-right", recipe.width, 0.0),
        ("point:top-right", recipe.width, recipe.height),
        ("point:top-left", 0.0, recipe.height),
    )
    angular_tolerance = 64.0 * math.ulp(2.0 * math.pi)
    for logical_id, x, y in corners:
        angle = math.atan2(y - recipe.hole_y, x - recipe.hole_x) % (
            2.0 * math.pi
        )
        match = next(
            (
                index
                for index, (candidate, _candidate_id) in enumerate(rays)
                if abs((angle - candidate + math.pi) % (2.0 * math.pi) - math.pi)
                <= angular_tolerance
            ),
            None,
        )
        if match is None:
            rays.append((angle, logical_id))
        else:
            rays[match] = (rays[match][0], logical_id)
    return tuple(sorted(rays, key=lambda item: item[0]))


def _angles(count: int) -> tuple[float, ...]:
    return tuple(2.0 * math.pi * index / count for index in range(count))


__all__ = [
    "GeometryPreview",
    "build_geometry_preview",
    "build_strict_body_boolean_preview",
    "build_strict_body_boolean_previews",
    "build_strict_sketch_draft_preview",
]
