"""Deterministic display tessellation for native geometry recipes."""

from __future__ import annotations

from dataclasses import dataclass
import math

from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    MovedGeometry,
    NativeGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchGeometry,
    axis_aligned_rectangle,
    expand_sketch_recipe,
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

    def __post_init__(self) -> None:
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
        if self.body_logical_id is not None:
            reference = LogicalEntityRef(self.body_logical_id)
            if reference.kind != "body":
                raise ValueError("body_logical_id 必须引用 body")


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


def _build_geometry_preview(
    recipe: NativeGeometry,
    segments: int,
) -> GeometryPreview:
    if isinstance(recipe, SketchGeometry):
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
    if isinstance(recipe, RectangleGeometry):
        return _rectangle_preview(recipe)
    if isinstance(recipe, BoxGeometry):
        return _box_preview(recipe)
    if isinstance(recipe, DiskGeometry):
        return _disk_preview(recipe, segments)
    if isinstance(recipe, PlateWithHoleGeometry):
        return _plate_with_hole_preview(recipe, segments)
    return _cylinder_preview(recipe, segments)


def _boolean_preview(
    recipe: BooleanGeometry,
    segments: int,
) -> GeometryPreview:
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
    point_count = len(base.points)
    points = base.points + tuple((x, y, z + recipe.height) for x, y, z in base.points)
    faces = list(base.faces)
    face_logical_ids = [
        _derived_logical_id(topology, logical_id, "face", "copy.bottom.")
        for logical_id in base.face_logical_ids
    ]
    faces.extend(
        tuple(reversed(tuple(index + point_count for index in face)))
        for face in base.faces
    )
    face_logical_ids.extend(
        _derived_logical_id(topology, logical_id, "face", "copy.top.")
        for logical_id in base.face_logical_ids
    )

    for edge, logical_id in zip(
        base.edges,
        base.edge_logical_ids,
        strict=True,
    ):
        for start, end in zip(edge, edge[1:]):
            faces.append((start, end, end + point_count, start + point_count))
            face_logical_ids.append(
                _derived_logical_id(topology, logical_id, "face", "sweep.")
            )

    edges = list(base.edges)
    edges.extend(tuple(index + point_count for index in edge) for edge in base.edges)
    edge_logical_ids = [
        _derived_logical_id(topology, logical_id, "edge", "copy.bottom.")
        for logical_id in base.edge_logical_ids
    ]
    edge_logical_ids.extend(
        _derived_logical_id(topology, logical_id, "edge", "copy.top.")
        for logical_id in base.edge_logical_ids
    )
    point_logical_ids = tuple(
        _derived_logical_id(topology, logical_id, "point", "copy.bottom.")
        for logical_id in base.point_logical_ids
    ) + tuple(
        _derived_logical_id(topology, logical_id, "point", "copy.top.")
        for logical_id in base.point_logical_ids
    )
    feature_points = tuple(
        index
        for index, logical_id in enumerate(base.point_logical_ids)
        if logical_id is not None
    )
    edges.extend((index, index + point_count) for index in feature_points)
    edge_logical_ids.extend(
        _derived_logical_id(
            topology,
            base.point_logical_ids[index],
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
) -> None:
    topology = describe_recipe_topology(recipe)
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
        if actual != expected:
            raise RuntimeError(
                f"{type(recipe).__name__} 预览的 {kind} logical ID"
                f"与 recipe topology 不一致: {sorted(actual)} != "
                f"{sorted(expected)}"
            )
    actual_body = (
        set()
        if preview.body_logical_id is None
        else {preview.body_logical_id}
    )
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
]
