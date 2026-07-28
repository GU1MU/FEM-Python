"""Canonical source-face selection for positive-Z extrusion features."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .references import LogicalEntityRef, logical_ref_sort_key


class ExtrusionSourceResolutionError(ValueError):
    """One extrusion source selection cannot be proven from logical topology."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        logical_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.logical_id = logical_id


@dataclass(frozen=True, slots=True)
class ExtrusionSourceSelection:
    """Canonical selected faces and their exact boundary closure."""

    face_ids: tuple[str, ...]
    boundary_edge_ids: tuple[str, ...]
    boundary_point_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for kind, values in (
            ("face", self.face_ids),
            ("edge", self.boundary_edge_ids),
            ("point", self.boundary_point_ids),
        ):
            if type(values) is not tuple:
                raise TypeError(f"{kind} IDs must be a tuple")
            if len(values) != len(set(values)):
                raise ValueError(f"{kind} IDs must be unique")
            for value in values:
                reference = LogicalEntityRef(value)
                if reference.kind != kind:
                    raise ValueError(f"{value!r} is not a {kind} logical ID")


def resolve_extrusion_source_faces(
    base: object,
    requested_ids: Iterable[str | LogicalEntityRef] = (),
) -> ExtrusionSourceSelection:
    """Resolve requested face IDs to canonical exact material faces.

    Empty input keeps the legacy headless meaning: select every canonical
    material face in the exact base topology.
    """

    # Imported lazily so recipe_topology can use this resolver without a module
    # import cycle.
    from .recipe_topology import describe_recipe_topology

    topology = describe_recipe_topology(base)  # type: ignore[arg-type]
    if not topology.exact:
        diagnostic = (
            topology.diagnostics[0].message
            if topology.diagnostics
            else "当前二维拓扑无法安全拉伸"
        )
        raise ExtrusionSourceResolutionError(
            "extrude.source-face.topology-unproven",
            diagnostic,
        )

    try:
        raw_requested = tuple(requested_ids)
    except TypeError as error:
        raise TypeError("requested extrusion face IDs must be iterable") from error
    references: list[LogicalEntityRef] = []
    for item in raw_requested:
        reference = item if type(item) is LogicalEntityRef else LogicalEntityRef(item)
        if reference.kind != "face":
            raise ExtrusionSourceResolutionError(
                "extrude.source-face.wrong-kind",
                f"拉伸源 {reference.logical_id!r} 必须引用二维面",
                logical_id=reference.logical_id,
            )
        references.append(reference)

    selectable_faces = topology.entities_of("face", selectable_only=True)
    canonical_by_input = {
        face.logical_id: _canonical_face_id(topology, face.logical_id)
        for face in selectable_faces
    }
    canonical_faces = tuple(
        sorted(
            set(canonical_by_input.values()),
            key=lambda logical_id: logical_ref_sort_key(
                LogicalEntityRef(logical_id)
            ),
        )
    )
    if not canonical_faces:
        raise ExtrusionSourceResolutionError(
            "extrude.source-face.required",
            "当前二维几何没有可拉伸的 material Profile",
        )

    if references:
        resolved: list[str] = []
        for reference in references:
            try:
                entity = topology.entity(reference.logical_id)
            except KeyError as error:
                raise ExtrusionSourceResolutionError(
                    "extrude.source-face.unknown",
                    f"所选 Profile {reference.logical_id!r} 已失效，请重新选择",
                    logical_id=reference.logical_id,
                ) from error
            if entity.kind != "face":
                raise ExtrusionSourceResolutionError(
                    "extrude.source-face.wrong-kind",
                    f"拉伸源 {reference.logical_id!r} 必须引用二维面",
                    logical_id=reference.logical_id,
                )
            if not entity.selectable:
                raise ExtrusionSourceResolutionError(
                    "extrude.source-face.unselectable",
                    f"所选 Profile {reference.logical_id!r} 当前不可选择",
                    logical_id=reference.logical_id,
                )
            resolved.append(_canonical_face_id(topology, entity.logical_id))
        face_ids = tuple(
            sorted(
                set(resolved),
                key=lambda logical_id: logical_ref_sort_key(
                    LogicalEntityRef(logical_id)
                ),
            )
        )
    else:
        face_ids = canonical_faces

    edge_ids_by_face = {
        face_id: _boundary_edge_ids(
            base,
            topology,
            face_id,
            canonical_faces,
        )
        for face_id in face_ids
    }
    for index, face_id in enumerate(face_ids):
        boundary = set(edge_ids_by_face[face_id])
        for other_id in face_ids[index + 1 :]:
            shared = boundary & set(edge_ids_by_face[other_id])
            if shared:
                raise ExtrusionSourceResolutionError(
                    "extrude.source-face.shared-boundary",
                    "所选 Profiles 共享边界，当前阶段无法证明稳定拉伸拓扑",
                    logical_id=sorted(shared)[0],
                )

    boundary_edge_ids = _sorted_logical_ids(
        edge_id
        for face_id in face_ids
        for edge_id in edge_ids_by_face[face_id]
    )
    boundary_point_ids = _sorted_logical_ids(
        point_id
        for face_id in face_ids
        for point_id in _boundary_point_ids(
            topology,
            edge_ids_by_face[face_id],
            face_has_explicit_edges=bool(
                _linked_ids(topology.entity(face_id), "edge")
            ),
        )
    )
    return ExtrusionSourceSelection(
        face_ids,
        boundary_edge_ids,
        boundary_point_ids,
    )


def extrusion_face_boundary_ids(
    base: object,
    face_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the exact edge/point closure for one canonical source face."""

    from .recipe_topology import describe_recipe_topology

    selection = resolve_extrusion_source_faces(base, (face_id,))
    topology = describe_recipe_topology(base)  # type: ignore[arg-type]
    edge_ids = set(selection.boundary_edge_ids)
    point_ids = set(selection.boundary_point_ids)
    return (
        tuple(
            entity.logical_id
            for entity in topology.entities_of("edge", selectable_only=True)
            if entity.logical_id in edge_ids
        ),
        tuple(
            entity.logical_id
            for entity in topology.entities_of("point", selectable_only=True)
            if entity.logical_id in point_ids
        ),
    )


def _canonical_face_id(topology, logical_id: str) -> str:
    current = logical_id
    visited: set[str] = set()
    while True:
        if current in visited:
            raise ExtrusionSourceResolutionError(
                "extrude.source-face.alias-ambiguous",
                f"Profile alias {logical_id!r} 形成循环，无法唯一解析",
                logical_id=logical_id,
            )
        visited.add(current)
        entity = topology.entity(current)
        linked_faces = _linked_ids(entity, "face")
        if not linked_faces:
            return current
        if len(linked_faces) != 1:
            raise ExtrusionSourceResolutionError(
                "extrude.source-face.alias-ambiguous",
                f"Profile alias {logical_id!r} 无法唯一解析",
                logical_id=logical_id,
            )
        current = linked_faces[0]


def _boundary_edge_ids(
    base,
    topology,
    face_id: str,
    canonical_faces: tuple[str, ...],
) -> tuple[str, ...]:
    linked_edges = _linked_ids(topology.entity(face_id), "edge")
    from .recipes import MovedGeometry, RotatedGeometry, SketchGeometry

    strict_base = base
    while isinstance(strict_base, (MovedGeometry, RotatedGeometry)):
        strict_base = strict_base.base
    if (
        type(strict_base) is SketchGeometry
        and strict_base.is_strict
        and topology.entity(face_id).semantic_role == "sketch.profile"
    ):
        from .recipe_analysis import analyze_sketch_profiles

        profile_id = face_id.split(":", 1)[1]
        analysis = analyze_sketch_profiles(strict_base)
        boundary_profiles = tuple(
            profile
            for profile in analysis.profiles
            if profile.id == profile_id
            or profile.parent_profile_id == profile_id
        )
        if boundary_profiles:
            return _sorted_logical_ids(
                f"edge:{curve_id.lstrip('-')}"
                for profile in boundary_profiles
                for curve_id in profile.curve_ids
            )
    if linked_edges:
        return _sorted_logical_ids(linked_edges)
    if len(canonical_faces) == 1:
        return tuple(
            entity.logical_id
            for entity in topology.entities_of("edge", selectable_only=True)
        )
    raise ExtrusionSourceResolutionError(
        "extrude.source-face.topology-unproven",
        f"Profile {face_id!r} 没有可证明的边界拓扑",
        logical_id=face_id,
    )


def _boundary_point_ids(
    topology,
    edge_ids: tuple[str, ...],
    *,
    face_has_explicit_edges: bool,
) -> tuple[str, ...]:
    linked_points = _sorted_logical_ids(
        point_id
        for edge_id in edge_ids
        for point_id in _linked_ids(topology.entity(edge_id), "point")
    )
    if linked_points or face_has_explicit_edges:
        return linked_points
    return tuple(
        entity.logical_id
        for entity in topology.entities_of("point", selectable_only=True)
    )


def _linked_ids(entity, kind: str) -> tuple[str, ...]:
    return tuple(
        logical_id
        for logical_id in entity.topology_links
        if LogicalEntityRef(logical_id).kind == kind
    )


def _sorted_logical_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda logical_id: logical_ref_sort_key(
                LogicalEntityRef(logical_id)
            ),
        )
    )


__all__ = [
    "ExtrusionSourceResolutionError",
    "ExtrusionSourceSelection",
    "extrusion_face_boundary_ids",
    "resolve_extrusion_source_faces",
]
