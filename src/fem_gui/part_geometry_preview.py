"""Application-Part adapter for deterministic geometry previews."""

from __future__ import annotations

from typing import Iterable, Mapping

from fem.application import NativePart
from fem.geometry.part_namespace import part_id_sort_key

from .geometry_preview import (
    GeometryPreview,
    build_geometry_preview,
    namespace_part_geometry_preview,
)


def build_multi_part_geometry_preview(
    parts: Iterable[NativePart],
    *,
    include_suppressed: bool = False,
    segments: int = 48,
) -> GeometryPreview:
    """Merge stable Part-owned previews in canonical Part order."""

    return merge_multi_part_geometry_previews(
        parts,
        include_suppressed=include_suppressed,
        segments=segments,
    )


def merge_multi_part_geometry_previews(
    parts: Iterable[NativePart],
    *,
    exact_previews: Mapping[str, GeometryPreview] | None = None,
    include_suppressed: bool = False,
    segments: int = 48,
) -> GeometryPreview:
    """Merge Parts while replacing selected rows with exact namespaced previews."""

    owned = tuple(
        sorted(
            (part for part in parts if include_suppressed or not part.suppressed),
            key=lambda part: part_id_sort_key(part.id),
        )
    )
    if any(
        type(part) is not NativePart or part.geometry_recipe is None for part in owned
    ):
        raise TypeError("multi-Part preview requires canonical NativeParts")
    overrides = {} if exact_previews is None else dict(exact_previews)
    unknown = set(overrides).difference(part.id for part in owned)
    if unknown:
        raise ValueError(
            "exact Part preview has no visible owner: " + ", ".join(sorted(unknown))
        )
    if any(type(preview) is not GeometryPreview for preview in overrides.values()):
        raise TypeError("exact Part previews must be GeometryPreview values")
    if not owned:
        return GeometryPreview((), (), (), topological_dimension=2)

    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    edges: list[tuple[int, ...]] = []
    face_ids: list[str | None] = []
    edge_ids: list[str | None] = []
    point_ids: list[str | None] = []
    face_bodies: list[str | None] = []
    edge_bodies: list[str | None] = []
    point_bodies: list[str | None] = []
    face_parts: list[str | None] = []
    edge_parts: list[str | None] = []
    point_parts: list[str | None] = []
    dimensions: set[int] = set()

    for part in owned:
        local = overrides.get(part.id)
        if local is None:
            local = namespace_part_geometry_preview(
                part.id,
                build_geometry_preview(
                    part.geometry_recipe,
                    segments=segments,
                ),
            )
        elif any(
            owner not in {None, part.id}
            for owners in (
                local.face_part_ids,
                local.edge_part_ids,
                local.point_part_ids,
            )
            for owner in owners
        ):
            raise ValueError(
                f"exact Part preview contains a foreign owner for {part.id}"
            )
        offset = len(points)
        points.extend(local.points)
        faces.extend(tuple(offset + index for index in cell) for cell in local.faces)
        edges.extend(tuple(offset + index for index in cell) for cell in local.edges)

        face_ids.extend(local.face_logical_ids)
        edge_ids.extend(local.edge_logical_ids)
        point_ids.extend(local.point_logical_ids)
        face_bodies.extend(local.face_body_logical_ids)
        edge_bodies.extend(local.edge_body_logical_ids)
        point_bodies.extend(local.point_body_logical_ids)
        face_parts.extend((part.id,) * len(local.faces))
        edge_parts.extend((part.id,) * len(local.edges))
        point_parts.extend((part.id,) * len(local.points))
        dimensions.add(local.dimension)

    return GeometryPreview(
        points=tuple(points),
        faces=tuple(faces),
        edges=tuple(edges),
        face_logical_ids=tuple(face_ids),
        edge_logical_ids=tuple(edge_ids),
        point_logical_ids=tuple(point_ids),
        body_logical_id=None,
        topological_dimension=max(dimensions),
        face_body_logical_ids=tuple(face_bodies),
        edge_body_logical_ids=tuple(edge_bodies),
        point_body_logical_ids=tuple(point_bodies),
        face_part_ids=tuple(face_parts),
        edge_part_ids=tuple(edge_parts),
        point_part_ids=tuple(point_parts),
    )


__all__ = [
    "build_multi_part_geometry_preview",
    "merge_multi_part_geometry_previews",
]
