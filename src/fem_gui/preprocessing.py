"""Small, persistent preprocessing inputs for the native Gmsh workflow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from fem import geometry
from fem.core.model import (
    Edge,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    NodeSet,
    Surface,
)
# Compatibility re-exports; remove once GUI callers use the headless modules.
from fem.geometry.recipes import (  # noqa: F401 - compatibility re-exports
    BASE_GEOMETRY_TYPES,
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    PRIMITIVE_GEOMETRY_TYPES,
    PlateWithHoleGeometry,
    PrimitiveGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SKETCH_CONTOUR_TYPES,
    SketchCircle,
    SketchContour,
    SketchGeometry,
    SketchRectangle,
    geometry_dimension,
)
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem.selection import edges as mesh_edges
from fem.selection import faces as mesh_faces


@dataclass(frozen=True, slots=True)
class GeometryPreview:
    """Backend-neutral surface and feature edges for viewport preview."""

    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]
    face_ids: tuple[int, ...] = ()
    edge_ids: tuple[int, ...] = ()


def geometry_characteristic_size(recipe: NativeGeometry) -> float:
    if isinstance(recipe, BooleanGeometry):
        return min(
            geometry_characteristic_size(recipe.object_geometry),
            geometry_characteristic_size(recipe.tool_geometry),
        )
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return geometry_characteristic_size(recipe.base)
    if isinstance(recipe, ExtrudedGeometry):
        return min(geometry_characteristic_size(recipe.base), recipe.height)
    if isinstance(recipe, SketchGeometry):
        return geometry_characteristic_size(_compile_sketch(recipe))
    if isinstance(recipe, (RectangleGeometry, PlateWithHoleGeometry)):
        return min(recipe.width, recipe.height)
    if isinstance(recipe, DiskGeometry):
        return 2.0 * recipe.radius
    if isinstance(recipe, BoxGeometry):
        return min(recipe.width, recipe.depth, recipe.height)
    return min(2.0 * recipe.radius, recipe.height)


def supports_hexahedron(recipe: NativeGeometry) -> bool:
    """Return whether the existing structured box workflow can mesh this recipe."""
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


def geometry_feature_rows(recipe: NativeGeometry) -> tuple[str, ...]:
    """Return a flat, user-facing feature history for the manager dialog."""
    if isinstance(recipe, SketchGeometry):
        material_count = sum(item.operation == "material" for item in recipe.contours)
        cut_count = len(recipe.contours) - material_count
        return (
            f"草图  轮廓={len(recipe.contours)}，材料={material_count}，切除={cut_count}",
        )
    if isinstance(recipe, MovedGeometry):
        return geometry_feature_rows(recipe.base) + (
            f"移动  X={recipe.dx:g}，Y={recipe.dy:g}，Z={recipe.dz:g}",
        )
    if isinstance(recipe, RotatedGeometry):
        return geometry_feature_rows(recipe.base) + (
            f"旋转  {recipe.axis.upper()} 轴，{recipe.angle_degrees:g}°",
        )
    if isinstance(recipe, ExtrudedGeometry):
        return geometry_feature_rows(recipe.base) + (f"拉伸  高度={recipe.height:g}",)
    if isinstance(recipe, BooleanGeometry):
        names = {"fuse": "合并", "cut": "切除", "fragment": "分割"}
        return geometry_feature_rows(recipe.object_geometry) + (
            f"{names[recipe.operation]}  工具体={recipe.tool_geometry.name}",
        )
    if isinstance(recipe, RectangleGeometry):
        description = f"矩形  {recipe.width:g} × {recipe.height:g}"
    elif isinstance(recipe, DiskGeometry):
        description = f"圆盘  半径={recipe.radius:g}"
    elif isinstance(recipe, PlateWithHoleGeometry):
        description = (
            f"带孔板  {recipe.width:g} × {recipe.height:g}，孔半径={recipe.hole_radius:g}"
        )
    elif isinstance(recipe, BoxGeometry):
        description = f"长方体  {recipe.width:g} × {recipe.depth:g} × {recipe.height:g}"
    else:
        description = f"圆柱  半径={recipe.radius:g}，高度={recipe.height:g}"
    return (f"基础体  {description}",)


def build_geometry_preview(recipe: NativeGeometry, *, segments: int = 48) -> GeometryPreview:
    """Build a small deterministic display mesh without generating FE elements."""
    count = max(12, int(segments))
    if isinstance(recipe, SketchGeometry):
        return build_geometry_preview(_compile_sketch(recipe), segments=count)
    if isinstance(recipe, BooleanGeometry):
        if recipe.operation == "cut":
            rectangular = _axis_aligned_rectangle(recipe.object_geometry)
            tool_rectangle = _axis_aligned_rectangle(recipe.tool_geometry)
            if rectangular is not None and tool_rectangle is not None:
                x, y, width, height = rectangular
                inner_x, inner_y, inner_width, inner_height = tool_rectangle
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
                        (inner_x + inner_width, inner_y + inner_height, 0.0),
                        (inner_x, inner_y + inner_height, 0.0),
                    )
                    faces = tuple(
                        (index, (index + 1) % 4, 4 + (index + 1) % 4, 4 + index)
                        for index in range(4)
                    )
                    edges = ((0, 1, 2, 3, 0), (4, 5, 6, 7, 4))
                    return GeometryPreview(
                        points,
                        faces,
                        edges,
                        (1, 1, 1, 1),
                        (1, 2),
                    )
            circular = _axis_aligned_circle(recipe.tool_geometry)
            if rectangular is not None and circular is not None:
                x, y, width, height = rectangular
                center_x, center_y, radius = circular
                local_x = center_x - x
                local_y = center_y - y
                if (
                    radius < local_x < width - radius
                    and radius < local_y < height - radius
                ):
                    ring = build_geometry_preview(
                        PlateWithHoleGeometry(
                            "cut-preview",
                            width,
                            height,
                            local_x,
                            local_y,
                            radius,
                        ),
                        segments=count,
                    )
                    return GeometryPreview(
                        tuple((px + x, py + y, pz) for px, py, pz in ring.points),
                        ring.faces,
                        ring.edges,
                        ring.face_ids,
                        ring.edge_ids,
                    )
        object_preview = build_geometry_preview(recipe.object_geometry, segments=count)
        tool_preview = build_geometry_preview(recipe.tool_geometry, segments=count)
        point_offset = len(object_preview.points)
        points = object_preview.points + tool_preview.points
        tool_edges = tuple(
            tuple(index + point_offset for index in edge)
            for edge in tool_preview.edges
        )
        edges = object_preview.edges + tool_edges
        edge_offset = max(object_preview.edge_ids or (0,))
        edge_ids = (
            object_preview.edge_ids
            + tuple(edge_offset + value for value in (tool_preview.edge_ids or ()))
        )
        if recipe.operation == "cut":
            return GeometryPreview(
                points,
                object_preview.faces,
                edges,
                object_preview.face_ids,
                edge_ids,
            )
        tool_faces = tuple(
            tuple(index + point_offset for index in face)
            for face in tool_preview.faces
        )
        face_offset = max(object_preview.face_ids or (0,))
        return GeometryPreview(
            points,
            object_preview.faces + tool_faces,
            edges,
            object_preview.face_ids
            + tuple(face_offset + value for value in (tool_preview.face_ids or ())),
            edge_ids,
        )
    if isinstance(recipe, MovedGeometry):
        preview = build_geometry_preview(recipe.base, segments=count)
        return GeometryPreview(
            tuple((x + recipe.dx, y + recipe.dy, z + recipe.dz) for x, y, z in preview.points),
            preview.faces,
            preview.edges,
            preview.face_ids,
            preview.edge_ids,
        )
    if isinstance(recipe, RotatedGeometry):
        preview = build_geometry_preview(recipe.base, segments=count)
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)

        def rotate_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
            x, y, z = point
            if recipe.axis == "x":
                return x, y * cosine - z * sine, y * sine + z * cosine
            if recipe.axis == "y":
                return x * cosine + z * sine, y, -x * sine + z * cosine
            return x * cosine - y * sine, x * sine + y * cosine, z

        return GeometryPreview(
            tuple(rotate_point(point) for point in preview.points),
            preview.faces,
            preview.edges,
            preview.face_ids,
            preview.edge_ids,
        )
    if isinstance(recipe, ExtrudedGeometry):
        preview = build_geometry_preview(recipe.base, segments=count)
        point_count = len(preview.points)
        points = preview.points + tuple(
            (x, y, z + recipe.height) for x, y, z in preview.points
        )
        faces = list(preview.faces)
        face_ids = [1] * len(preview.faces)
        faces.extend(tuple(reversed(tuple(index + point_count for index in face))) for face in preview.faces)
        face_ids.extend([2] * len(preview.faces))
        base_edge_ids = preview.edge_ids or tuple(range(1, len(preview.edges) + 1))
        for edge, edge_id in zip(preview.edges, base_edge_ids):
            for start, end in zip(edge, edge[1:]):
                faces.append((start, end, end + point_count, start + point_count))
                face_ids.append(2 + edge_id)
        edges = list(preview.edges)
        edges.extend(tuple(index + point_count for index in edge) for edge in preview.edges)
        feature_points = sorted({index for edge in preview.edges for index in edge})
        edges.extend((index, index + point_count) for index in feature_points)
        return GeometryPreview(
            points,
            tuple(faces),
            tuple(edges),
            tuple(face_ids),
            tuple(range(1, len(edges) + 1)),
        )
    if isinstance(recipe, RectangleGeometry):
        points = (
            (0.0, 0.0, 0.0),
            (recipe.width, 0.0, 0.0),
            (recipe.width, recipe.height, 0.0),
            (0.0, recipe.height, 0.0),
        )
        return GeometryPreview(
            points,
            ((0, 1, 2, 3),),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            (1,),
            (1, 2, 3, 4),
        )
    if isinstance(recipe, BoxGeometry):
        w, d, h = recipe.width, recipe.depth, recipe.height
        points = (
            (0.0, 0.0, 0.0), (w, 0.0, 0.0), (w, d, 0.0), (0.0, d, 0.0),
            (0.0, 0.0, h), (w, 0.0, h), (w, d, h), (0.0, d, h),
        )
        faces = (
            (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        )
        edges = (
            (0, 1, 2, 3, 0), (4, 5, 6, 7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        return GeometryPreview(points, faces, edges, tuple(range(1, 7)), tuple(range(1, 7)))

    angles = tuple(2.0 * math.pi * index / count for index in range(count))
    if isinstance(recipe, DiskGeometry):
        points = ((0.0, 0.0, 0.0),) + tuple(
            (recipe.radius * math.cos(angle), recipe.radius * math.sin(angle), 0.0)
            for angle in angles
        )
        faces = tuple((0, 1 + index, 1 + (index + 1) % count) for index in range(count))
        edge = tuple(range(1, count + 1)) + (1,)
        return GeometryPreview(points, faces, (edge,), (1,) * len(faces), (1,))
    if isinstance(recipe, PlateWithHoleGeometry):
        hole_angles = _rectangle_hole_angles(recipe, count)
        hole_count = len(hole_angles)
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
            dx, dy = math.cos(angle), math.sin(angle)
            distances = []
            if dx > 0.0:
                distances.append((recipe.width - recipe.hole_x) / dx)
            elif dx < 0.0:
                distances.append(-recipe.hole_x / dx)
            if dy > 0.0:
                distances.append((recipe.height - recipe.hole_y) / dy)
            elif dy < 0.0:
                distances.append(-recipe.hole_y / dy)
            distance = min(value for value in distances if value > 0.0)
            outer.append((recipe.hole_x + distance * dx, recipe.hole_y + distance * dy, 0.0))
        points = inner + tuple(outer)
        faces = tuple(
            (
                index,
                hole_count + index,
                hole_count + (index + 1) % hole_count,
                (index + 1) % hole_count,
            )
            for index in range(hole_count)
        )
        return GeometryPreview(
            points,
            faces,
            (
                tuple(range(hole_count)) + (0,),
                tuple(range(hole_count, 2 * hole_count)) + (hole_count,),
            ),
            (1,) * len(faces),
            (1, 2),
        )

    bottom_center, top_center = (0.0, 0.0, 0.0), (0.0, 0.0, recipe.height)
    bottom = tuple(
        (recipe.radius * math.cos(angle), recipe.radius * math.sin(angle), 0.0)
        for angle in angles
    )
    top = tuple((x, y, recipe.height) for x, y, _z in bottom)
    points = (bottom_center, top_center) + bottom + top
    bottom_start, top_start = 2, 2 + count
    faces = []
    face_ids = []
    for index in range(count):
        next_index = (index + 1) % count
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
        face_ids.extend((1, 2, 3))
    edges = [
        tuple(range(bottom_start, bottom_start + count)) + (bottom_start,),
        tuple(range(top_start, top_start + count)) + (top_start,),
    ]
    quarter = max(1, count // 4)
    edges.extend(
        (bottom_start + index, top_start + index)
        for index in range(0, count, quarter)
    )
    return GeometryPreview(
        points,
        tuple(faces),
        tuple(edges),
        tuple(face_ids),
        tuple(range(1, len(edges) + 1)),
    )


def _axis_aligned_rectangle(recipe: NativeGeometry) -> tuple[float, float, float, float] | None:
    if isinstance(recipe, RectangleGeometry):
        return 0.0, 0.0, recipe.width, recipe.height
    if isinstance(recipe, MovedGeometry):
        base = _axis_aligned_rectangle(recipe.base)
        if base is None or recipe.dz != 0.0:
            return None
        x, y, width, height = base
        return x + recipe.dx, y + recipe.dy, width, height
    return None


def _rectangle_hole_angles(
    recipe: PlateWithHoleGeometry,
    segments: int,
) -> tuple[float, ...]:
    """Sample a rectangular plate radially while preserving all four corners."""
    regular = [
        2.0 * math.pi * index / max(12, int(segments))
        for index in range(max(12, int(segments)))
    ]
    corners = (
        (0.0, 0.0),
        (recipe.width, 0.0),
        (recipe.width, recipe.height),
        (0.0, recipe.height),
    )
    angles = regular + [
        math.atan2(y - recipe.hole_y, x - recipe.hole_x) % (2.0 * math.pi)
        for x, y in corners
    ]
    return tuple(sorted({round(angle, 12) for angle in angles}))


def _axis_aligned_circle(recipe: NativeGeometry) -> tuple[float, float, float] | None:
    if isinstance(recipe, DiskGeometry):
        return 0.0, 0.0, recipe.radius
    if isinstance(recipe, MovedGeometry):
        base = _axis_aligned_circle(recipe.base)
        if base is None or recipe.dz != 0.0:
            return None
        x, y, radius = base
        return x + recipe.dx, y + recipe.dy, radius
    return None


def generate_fem_model(
    recipe: NativeGeometry,
    settings: MeshSettings,
    *,
    named_regions: Iterable[Any] = (),
):
    """Build one labeled native geometry and return the canonical FEM model."""
    try:
        import gmsh
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "几何与网格功能需要 Gmsh；请安装项目的 cad 可选依赖"
        ) from error

    owns_session = not bool(gmsh.isInitialized())
    try:
        if owns_session:
            # Gmsh's default SIGINT handler can only be installed by Python's
            # main thread. GUI generation intentionally runs in one worker.
            gmsh.initialize(interruptible=False)
        dimension = geometry_dimension(recipe)
        with geometry.model(recipe.name, dimension=dimension) as cad:
            domain = _build_cad_domain(cad, recipe)
            boundary = cad.boundary(domain)
            groups: dict[str, tuple] = {}
            hole_boundary: tuple = ()
            if isinstance(recipe, (RectangleGeometry, PlateWithHoleGeometry)):
                groups = {
                    "LEFT": cad.select(boundary, x=0.0),
                    "RIGHT": cad.select(boundary, x=recipe.width),
                    "BOTTOM": cad.select(boundary, y=0.0),
                    "TOP": cad.select(boundary, y=recipe.height),
                }
                if any(len(entities) != 1 for entities in groups.values()):
                    raise RuntimeError("矩形边界识别失败")
                outer = {entity for entities in groups.values() for entity in entities}
                hole_boundary = tuple(entity for entity in boundary if entity not in outer)
            elif isinstance(recipe, DiskGeometry):
                groups = {"OUTER": boundary}
            elif isinstance(recipe, BoxGeometry):
                groups = {
                    "LEFT": cad.select(boundary, x=0.0),
                    "RIGHT": cad.select(boundary, x=recipe.width),
                    "FRONT": cad.select(boundary, y=0.0),
                    "BACK": cad.select(boundary, y=recipe.depth),
                    "BOTTOM": cad.select(boundary, z=0.0),
                    "TOP": cad.select(boundary, z=recipe.height),
                }
                if any(len(entities) != 1 for entities in groups.values()):
                    raise RuntimeError("长方体边界识别失败")
            elif isinstance(recipe, CylinderGeometry):
                bottom = cad.select(boundary, z=0.0)
                top = cad.select(boundary, z=recipe.height)
                caps = {*bottom, *top}
                groups = {
                    "BOTTOM": bottom,
                    "TOP": top,
                    "OUTER": tuple(entity for entity in boundary if entity not in caps),
                }
            elif isinstance(recipe, ExtrudedGeometry):
                bottom = cad.select(boundary, z=0.0)
                top = cad.select(boundary, z=recipe.height)
                caps = {*bottom, *top}
                groups = {
                    "BOTTOM": bottom,
                    "TOP": top,
                    "OUTER": tuple(entity for entity in boundary if entity not in caps),
                }
            else:
                # CAE users create named regions explicitly; do not expose a
                # synthetic all-boundary set as a default modelling concept.
                groups = {}
            if isinstance(recipe, PlateWithHoleGeometry) and not hole_boundary:
                raise RuntimeError("圆孔边界识别失败")
            mesher = gmsh_meshing.Mesher(cad)
            if settings.cell_shape == "hexahedron":
                if not supports_hexahedron(recipe):
                    raise ValueError("六面体结构化网格当前仅支持长方体或矩形草图拉伸体")
                curves = tuple(
                    sorted(
                        {
                            entity
                            for entity in cad.boundary(boundary)
                            if entity.dimension == 1
                        },
                        key=lambda item: item.tag,
                    )
                )
                node_count = max(
                    2,
                    int(math.ceil(geometry_characteristic_size(recipe) / settings.size)) + 1,
                )
                for curve in curves:
                    mesher.transfinite_curve(curve, num_nodes=node_count)
                for surface in boundary:
                    mesher.transfinite_surface(surface)
                    mesher.recombine(surface)
                mesher.transfinite_volume(domain[0])

            entity_groups: dict[str, tuple] = {
                "DOMAIN": tuple(domain),
                **groups,
            }
            for region in named_regions:
                region_name = str(getattr(region, "name", "")).strip()
                if not region_name or region_name == "DOMAIN":
                    continue
                region_kind = str(getattr(region, "entity_kind", ""))
                region_ids = tuple(
                    int(value) for value in getattr(region, "entity_ids", ())
                )
                region_entities = _named_region_entities(
                    cad,
                    recipe,
                    domain,
                    boundary,
                    groups,
                    hole_boundary,
                    region_kind,
                    region_ids,
                )
                if region_entities:
                    entity_groups[region_name] = region_entities
            if hole_boundary:
                entity_groups["HOLE"] = hole_boundary
            mesh_size = settings.size
            refinements = []
            if hole_boundary and settings.local_size is not None:
                distance = mesher.distance_field(curves=hole_boundary, sampling=100)
                refinements.append(mesher.threshold_field(
                    distance,
                    size_min=settings.local_size,
                    size_max=settings.size,
                    dist_min=recipe.hole_radius * 0.25,
                    dist_max=recipe.hole_radius * 2.0,
                ))
            for control in settings.local_controls:
                sources = _local_control_sources(
                    cad,
                    recipe,
                    domain,
                    boundary,
                    groups,
                    hole_boundary,
                    control,
                )
                distance = mesher.distance_field(**sources, sampling=100)
                refinements.append(mesher.threshold_field(
                    distance,
                    size_min=control.size,
                    size_max=settings.size,
                    dist_min=0.0,
                    dist_max=settings.size * 2.0,
                ))
            if refinements:
                background = (
                    refinements[0]
                    if len(refinements) == 1
                    else mesher.min_field(refinements)
                )
                mesher.background_field(background)
                mesh_size = None
            native_mesh = mesher.generate(
                gmsh_meshing.MeshSpec(
                    size=mesh_size,
                    order=settings.order,
                    recombine=settings.cell_shape
                    in {"quadrilateral", "hexahedron"},
                )
            )
            mesh = gmsh_io.read(native_mesh)
            return _build_native_fem_model(
                mesh,
                native_mesh,
                recipe.name,
                dimension,
                entity_groups,
            )
    finally:
        if owns_session and bool(gmsh.isInitialized()):
            gmsh.finalize()


def _build_cad_domain(cad, recipe: NativeGeometry):
    if isinstance(recipe, SketchGeometry):
        return _build_cad_domain(cad, _compile_sketch(recipe))
    if isinstance(recipe, BooleanGeometry):
        objects = _build_cad_domain(cad, recipe.object_geometry)
        tools = _build_cad_domain(cad, recipe.tool_geometry)
        operation = {
            "fuse": cad.fuse,
            "cut": cad.cut,
            "fragment": cad.fragment,
        }[recipe.operation]
        result = operation(objects, tools).of_dimension(geometry_dimension(recipe))
        if not result:
            raise RuntimeError("布尔操作没有生成有效几何")
        return result
    if isinstance(recipe, (RectangleGeometry, PlateWithHoleGeometry)):
        plate = cad.rectangle(0.0, 0.0, recipe.width, recipe.height)
        if isinstance(recipe, PlateWithHoleGeometry):
            hole = cad.disk(recipe.hole_x, recipe.hole_y, recipe.hole_radius)
            domain = cad.cut([plate], [hole]).of_dimension(2)
            if len(domain) != 1:
                raise RuntimeError("带孔板布尔切除失败")
            return domain
        return (plate,)
    if isinstance(recipe, DiskGeometry):
        return (cad.disk(0.0, 0.0, recipe.radius),)
    if isinstance(recipe, BoxGeometry):
        return (cad.box(0.0, 0.0, 0.0, recipe.width, recipe.depth, recipe.height),)
    if isinstance(recipe, CylinderGeometry):
        return (cad.cylinder(0.0, 0.0, 0.0, 0.0, 0.0, recipe.height, recipe.radius),)
    if isinstance(recipe, MovedGeometry):
        domain = _build_cad_domain(cad, recipe.base)
        return cad.translate(domain, recipe.dx, recipe.dy, recipe.dz)
    if isinstance(recipe, RotatedGeometry):
        domain = _build_cad_domain(cad, recipe.base)
        axis = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }[recipe.axis]
        return cad.rotate(domain, 0.0, 0.0, 0.0, *axis, math.radians(recipe.angle_degrees))
    source = _build_cad_domain(cad, recipe.base)
    return cad.extrude(source, 0.0, 0.0, recipe.height).of_dimension(3)


def _build_native_fem_model(
    mesh: Any,
    native_mesh: Any,
    name: str,
    dimension: int,
    entity_groups: dict[str, tuple],
) -> FEMModel:
    """Convert CAD entity groups to FEM sets without public topology numbering."""
    model = FEMModel(mesh=mesh, name=name)
    backend = native_mesh._borrow_model()
    boundary_edges = mesh_edges.boundary(mesh) if dimension == 2 else ()
    boundary_faces = mesh_faces.boundary(mesh) if dimension == 3 else ()

    for group_name, entities in entity_groups.items():
        if not entities:
            continue
        entity_dimension = entities[0].dimension
        if any(entity.dimension != entity_dimension for entity in entities):
            raise ValueError(
                f"几何区域 {group_name} 包含不同维度的实体"
            )
        if entity_dimension == dimension:
            element_ids = _gmsh_entity_element_ids(backend, entities)
            model.element_sets[group_name] = ElementSet(group_name, element_ids)
            continue

        node_ids = _gmsh_entity_node_ids(backend, entities)
        model.node_sets[group_name] = NodeSet(group_name, node_ids)
        node_id_set = set(node_ids)
        if dimension == 2 and entity_dimension == 1:
            model.edges[group_name] = Edge(
                group_name,
                [
                    ElementEdge(element_id, local_edge, edge_node_ids)
                    for element_id, local_edge, edge_node_ids in boundary_edges
                    if set(edge_node_ids).issubset(node_id_set)
                ],
            )
        elif dimension == 3 and entity_dimension == 2:
            model.surfaces[group_name] = Surface(
                group_name,
                [
                    ElementFace(element_id, local_face, face_node_ids)
                    for element_id, local_face, face_node_ids in boundary_faces
                    if set(face_node_ids).issubset(node_id_set)
                ],
            )
    return model


def _gmsh_entity_node_ids(backend: Any, entities: tuple) -> tuple[int, ...]:
    """Return generated node ids attached to the supplied CAD entities."""
    node_ids: set[int] = set()
    for entity in entities:
        raw_ids, _coordinates, _parameters = backend.mesh.getNodes(
            entity.dimension,
            entity.tag,
            True,
            False,
        )
        node_ids.update(int(node_id) for node_id in raw_ids)
    return tuple(sorted(node_ids))


def _gmsh_entity_element_ids(backend: Any, entities: tuple) -> tuple[int, ...]:
    """Return generated top-dimensional element ids for CAD entities."""
    element_ids: set[int] = set()
    for entity in entities:
        _types, tag_blocks, _connectivity = backend.mesh.getElements(
            entity.dimension,
            entity.tag,
        )
        for tags in tag_blocks:
            element_ids.update(int(element_id) for element_id in tags)
    return tuple(sorted(element_ids))


def _compile_sketch(recipe: SketchGeometry) -> NativeGeometry:
    """Compile a generic sketch to the existing, tested native feature chain."""

    def contour_geometry(contour: SketchContour, index: int) -> NativeGeometry:
        contour_name = f"{recipe.name}-Contour-{index}"
        if isinstance(contour, SketchRectangle):
            result: NativeGeometry = RectangleGeometry(
                contour_name,
                contour.width,
                contour.height,
            )
        else:
            result = DiskGeometry(contour_name, contour.radius)
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


def _local_control_sources(
    cad,
    recipe: NativeGeometry,
    domain: tuple,
    boundary: tuple,
    groups: dict[str, tuple],
    hole_boundary: tuple,
    control: LocalMeshControl,
) -> dict[str, tuple]:
    """Map stable preview entity ids to current OCC topology."""
    dimension = geometry_dimension(recipe)
    if control.entity_kind == "point":
        points = tuple(
            entity
            for entity in cad.boundary(domain, recursive=True)
            if entity.dimension == 0
        )
        if not points:
            curves = tuple(entity for entity in cad.boundary(domain, recursive=True) if entity.dimension == 1)
            points = tuple(entity for entity in cad.boundary(curves) if entity.dimension == 0)
        unique = tuple(sorted(set(points), key=lambda item: item.tag))
        if control.entity_id > len(unique):
            raise ValueError("所选几何点已失效，请重新选择")
        return {"points": (unique[control.entity_id - 1],)}
    if control.entity_kind == "edge":
        if dimension == 2:
            edge_groups = _two_dimensional_edge_groups(
                cad,
                recipe,
                boundary,
                hole_boundary,
            )
            curves = (
                edge_groups[control.entity_id - 1]
                if control.entity_id <= len(edge_groups)
                else ()
            )
        else:
            curves = tuple(
                entity
                for entity in cad.boundary(boundary)
                if entity.dimension == 1
            )
            curves = tuple(sorted(set(curves), key=lambda item: item.tag))
            if control.entity_id <= len(curves):
                curves = (curves[control.entity_id - 1],)
        if not curves:
            raise ValueError("所选几何边已失效，请重新选择")
        return {"curves": curves}
    if dimension == 2:
        return {"surfaces": domain}
    if isinstance(recipe, BoxGeometry):
        names = ("BOTTOM", "TOP", "FRONT", "RIGHT", "BACK", "LEFT")
        surfaces = groups.get(names[min(control.entity_id - 1, len(names) - 1)], ())
    elif isinstance(recipe, CylinderGeometry):
        names = ("BOTTOM", "TOP", "OUTER")
        surfaces = groups.get(names[min(control.entity_id - 1, 2)], ())
    elif isinstance(recipe, ExtrudedGeometry):
        face_groups = _extruded_face_groups(
            cad,
            recipe,
            boundary,
            groups,
        )
        surfaces = (
            face_groups[control.entity_id - 1]
            if control.entity_id <= len(face_groups)
            else ()
        )
    else:
        ordered = tuple(sorted(boundary, key=lambda item: item.tag))
        surfaces = (
            (ordered[control.entity_id - 1],)
            if control.entity_id <= len(ordered)
            else ()
        )
    if not surfaces:
        raise ValueError("所选几何面已失效，请重新选择")
    return {"surfaces": surfaces}


def _extruded_face_groups(
    cad,
    recipe: ExtrudedGeometry,
    boundary: tuple,
    groups: dict[str, tuple],
) -> tuple[tuple, ...]:
    """Match extruded preview face ids to caps and individual side groups."""
    bottom = tuple(groups.get("BOTTOM", ()))
    top = tuple(groups.get("TOP", ()))
    outer = tuple(groups.get("OUTER", ()))
    logical = (
        _compile_sketch(recipe.base)
        if isinstance(recipe.base, SketchGeometry)
        else recipe.base
    )
    rectangle = _axis_aligned_rectangle(logical)
    if rectangle is not None:
        x, y, width, height = rectangle
        sides = (
            tuple(cad.select(boundary, y=y)),
            tuple(cad.select(boundary, x=x + width)),
            tuple(cad.select(boundary, y=y + height)),
            tuple(cad.select(boundary, x=x)),
        )
        if all(sides):
            return bottom, top, *sides
    if isinstance(logical, BooleanGeometry) and logical.operation == "cut":
        rectangle = _axis_aligned_rectangle(logical.object_geometry)
        if rectangle is not None:
            x, y, width, height = rectangle
            outer_sides = tuple(sorted({
                *cad.select(boundary, x=x),
                *cad.select(boundary, x=x + width),
                *cad.select(boundary, y=y),
                *cad.select(boundary, y=y + height),
            }, key=lambda item: item.tag))
            inner_sides = tuple(
                entity for entity in outer if entity not in set(outer_sides)
            )
            if inner_sides and outer_sides:
                if _axis_aligned_circle(logical.tool_geometry) is not None:
                    return bottom, top, inner_sides, outer_sides
                return bottom, top, outer_sides, inner_sides
    return bottom, top, outer


def _named_region_entities(
    cad,
    recipe: NativeGeometry,
    domain: tuple,
    boundary: tuple,
    groups: dict[str, tuple],
    hole_boundary: tuple,
    entity_kind: str,
    entity_ids: tuple[int, ...],
) -> tuple:
    """Resolve a user-created geometry collection to real CAD entities."""
    if entity_kind == "body":
        return tuple(domain)
    entities: list = []
    for entity_id in entity_ids:
        # Region selection intentionally reuses the same stable preview-to-CAD
        # mapping as local mesh controls.
        control = LocalMeshControl(
            entity_kind, entity_id, 1.0,
        )
        sources = _local_control_sources(
            cad,
            recipe,
            domain,
            boundary,
            groups,
            hole_boundary,
            control,
        )
        for values in sources.values():
            entities.extend(values)
    return tuple(sorted(set(entities), key=lambda item: (item.dimension, item.tag)))


def _two_dimensional_edge_groups(
    cad,
    recipe: NativeGeometry,
    boundary: tuple,
    hole_boundary: tuple,
) -> tuple[tuple, ...]:
    """Map preview loop ids to final OCC curves for 2-D local sizing."""
    if isinstance(recipe, PlateWithHoleGeometry):
        outer = tuple(
            entity for entity in boundary if entity not in set(hole_boundary)
        )
        return hole_boundary, outer
    logical = _compile_sketch(recipe) if isinstance(recipe, SketchGeometry) else recipe
    rectangle = _axis_aligned_rectangle(logical)
    if rectangle is not None:
        x, y, width, height = rectangle
        # Preview ids follow the perimeter clockwise from the bottom edge.
        groups = (
            tuple(cad.select(boundary, y=y)),
            tuple(cad.select(boundary, x=x + width)),
            tuple(cad.select(boundary, y=y + height)),
            tuple(cad.select(boundary, x=x)),
        )
        if all(groups):
            return groups
    if isinstance(logical, BooleanGeometry) and logical.operation == "cut":
        rectangle = _axis_aligned_rectangle(logical.object_geometry)
        if rectangle is not None:
            x, y, width, height = rectangle
            outer = {
                *cad.select(boundary, x=x),
                *cad.select(boundary, x=x + width),
                *cad.select(boundary, y=y),
                *cad.select(boundary, y=y + height),
            }
            inner = tuple(entity for entity in boundary if entity not in outer)
            if inner and outer:
                return inner, tuple(sorted(outer, key=lambda item: item.tag))
    return (tuple(boundary),)
