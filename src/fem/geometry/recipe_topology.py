"""Backend-neutral logical topology for native geometry recipes.

The catalog in this module deliberately describes *semantic* entities rather
than OpenCASCADE implementation details.  Logical identifiers therefore do not
contain, derive from, or depend on Gmsh tags.  Periodic seams and other
backend-created artifacts are omitted when they have no stable modelling
meaning.

Only topology that can be proved from a recipe is selectable.  Operations
whose result depends on geometric intersection return an unselectable result
placeholder and a diagnostic instead of guessing an entity ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .recipe_analysis import (
    analyze_sketch_profiles,
    axis_aligned_rectangle,
    expand_sketch_recipe,
    transformed_circle,
)
from .extrusion_selection import (
    ExtrusionSourceResolutionError,
    extrusion_face_boundary_ids,
    resolve_extrusion_source_faces,
)
from .references import (
    EntityKind,
    LogicalEntityRef,
    logical_ref_sort_key,
)
from .recipes import (
    BooleanBodyContext,
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    PlateWithHoleGeometry,
    PathSweptGeometry,
    PlanarBooleanContext,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SolidBody,
    SketchArc,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    WireGeometry,
    geometry_dimension,
)


TransitionRelation = Literal["preserved", "derived"]

_ENTITY_KINDS = ("point", "edge", "face", "body")
TOPOLOGY_REFERENCE_CONTRACT = 2


@dataclass(frozen=True, slots=True)
class LogicalEntity:
    """One stable, backend-neutral entity exposed for authoring.

    ``logical_id`` is stable across parameter-only edits and rigid
    transformations that preserve topology.  ``semantic_role`` describes the
    entity in recipe coordinates; it is not a backend tag or display index.
    """

    kind: EntityKind
    dimension: Literal[0, 1, 2, 3]
    logical_id: str
    semantic_role: str
    selectable: bool = True
    diagnostic_code: str | None = None
    topology_links: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _ENTITY_KINDS:
            raise ValueError(f"Unsupported logical entity kind: {self.kind!r}")
        expected_dimension = {"point": 0, "edge": 1, "face": 2}.get(self.kind)
        if expected_dimension is not None and self.dimension != expected_dimension:
            raise ValueError(
                f"Logical {self.kind} entities must have CAD dimension "
                f"{expected_dimension}"
            )
        if self.kind == "body" and self.dimension not in {1, 2, 3}:
            raise ValueError(
                "Logical body entities must have CAD dimension 1, 2, or 3"
            )
        if not self.logical_id.startswith(f"{self.kind}:"):
            raise ValueError(
                f"Logical id {self.logical_id!r} must start with {self.kind!r}"
            )
        if not self.semantic_role.strip():
            raise ValueError("Logical entity semantic role cannot be empty")
        if not self.selectable and not self.diagnostic_code:
            raise ValueError(
                "An unselectable logical entity must reference a diagnostic code"
            )


@dataclass(frozen=True, slots=True)
class TopologyDiagnostic:
    """A stable explanation for topology that cannot safely be selected."""

    code: str
    message: str
    affected_logical_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("Topology diagnostic code cannot be empty")
        if not self.message.strip():
            raise ValueError("Topology diagnostic message cannot be empty")


@dataclass(frozen=True, slots=True)
class TopologySignature:
    """Hashable structural signature independent of recipe dimensions and tags."""

    dimension: Literal[1, 2, 3]
    entity_keys: tuple[tuple[EntityKind, str, str, bool], ...]
    exact: bool

    @property
    def logical_ids(self) -> tuple[str, ...]:
        return tuple(key[1] for key in self.entity_keys)

    def count(self, kind: EntityKind, *, selectable_only: bool = False) -> int:
        """Return the number of catalog entries of one kind."""
        if kind not in _ENTITY_KINDS:
            raise ValueError(f"Unsupported logical entity kind: {kind!r}")
        return sum(
            entity_kind == kind and (selectable or not selectable_only)
            for entity_kind, _logical_id, _role, selectable in self.entity_keys
        )


@dataclass(frozen=True, slots=True)
class TopologyFingerprintEntity:
    """One canonical entity record in a topology compatibility fingerprint."""

    kind: EntityKind
    logical_id: str
    semantic_role: str
    selectable: bool
    topology_links: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reference = LogicalEntityRef(self.logical_id)
        if self.kind != reference.kind:
            raise ValueError(
                f"fingerprint entity kind {self.kind!r} does not match "
                f"logical ID {self.logical_id!r}"
            )
        if type(self.semantic_role) is not str or not self.semantic_role.strip():
            raise ValueError("fingerprint semantic role must be a non-empty string")
        if type(self.selectable) is not bool:
            raise TypeError("fingerprint selectable must be a boolean")
        if type(self.topology_links) is not tuple or any(
            type(link) is not str or not link.strip()
            for link in self.topology_links
        ):
            raise TypeError("fingerprint topology links must be non-empty strings")


@dataclass(frozen=True, slots=True)
class TopologyFingerprint:
    """Canonical structured evidence for one logical topology contract."""

    dimension: Literal[1, 2, 3]
    exact: bool
    entities: tuple[TopologyFingerprintEntity, ...]
    contract: int = TOPOLOGY_REFERENCE_CONTRACT

    def __post_init__(self) -> None:
        if (
            isinstance(self.contract, bool)
            or not isinstance(self.contract, int)
            or self.contract != TOPOLOGY_REFERENCE_CONTRACT
        ):
            raise ValueError(
                "unsupported logical topology reference contract: "
                f"{self.contract!r}"
            )
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension not in {1, 2, 3}
        ):
            raise ValueError("topology fingerprint dimension must be 1, 2, or 3")
        if type(self.exact) is not bool:
            raise TypeError("topology fingerprint exact must be a boolean")
        records = tuple(self.entities)
        if any(type(record) is not TopologyFingerprintEntity for record in records):
            raise TypeError(
                "topology fingerprint entities must be TopologyFingerprintEntity values"
            )
        logical_ids = tuple(record.logical_id for record in records)
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("topology fingerprint contains duplicate logical IDs")
        ordered = tuple(
            sorted(
                records,
                key=lambda record: logical_ref_sort_key(
                    LogicalEntityRef(record.logical_id)
                ),
            )
        )
        object.__setattr__(self, "entities", ordered)


@dataclass(frozen=True, slots=True)
class TopologyMapping:
    """Lineage from one input logical entity to one output logical entity."""

    source: str
    source_logical_id: str
    target_logical_id: str
    relation: TransitionRelation

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Topology mapping source cannot be empty")
        if self.relation not in {"preserved", "derived"}:
            raise ValueError(
                f"Unsupported topology mapping relation: {self.relation!r}"
            )
        if self.relation == "preserved" and (
            self.source_logical_id != self.target_logical_id
        ):
            raise ValueError("A preserved topology mapping must keep its logical id")


@dataclass(frozen=True, slots=True)
class TopologyTransition:
    """Structural transition from recipe inputs to the resulting topology."""

    operation: str
    source_signatures: tuple[tuple[str, TopologySignature], ...]
    target_signature: TopologySignature
    mappings: tuple[TopologyMapping, ...]
    proven: bool

    @property
    def preserved_logical_ids(self) -> tuple[str, ...]:
        """Return output IDs whose identity is proven to survive unchanged."""
        return tuple(
            dict.fromkeys(
                mapping.target_logical_id
                for mapping in self.mappings
                if mapping.relation == "preserved"
            )
        )

    @property
    def derived_logical_ids(self) -> tuple[str, ...]:
        """Return output IDs with proven lineage and a new identity."""
        return tuple(
            dict.fromkeys(
                mapping.target_logical_id
                for mapping in self.mappings
                if mapping.relation == "derived"
            )
        )


@dataclass(frozen=True, slots=True)
class RecipeTopology:
    """Logical entities, signature, transition, and diagnostics for one recipe."""

    recipe_type: str
    dimension: Literal[1, 2, 3]
    entities: tuple[LogicalEntity, ...]
    signature: TopologySignature
    transition: TopologyTransition
    diagnostics: tuple[TopologyDiagnostic, ...] = ()

    @property
    def exact(self) -> bool:
        return self.signature.exact

    def entities_of(
        self,
        kind: EntityKind,
        *,
        selectable_only: bool = False,
    ) -> tuple[LogicalEntity, ...]:
        """Return catalog entities of ``kind`` in deterministic logical order."""
        if kind not in _ENTITY_KINDS:
            raise ValueError(f"Unsupported logical entity kind: {kind!r}")
        return tuple(
            entity
            for entity in self.entities
            if entity.kind == kind and (entity.selectable or not selectable_only)
        )

    def selectable_entities(
        self,
        kind: EntityKind | None = None,
    ) -> tuple[LogicalEntity, ...]:
        """Return every entity that is safe to expose to authoring tools."""
        if kind is not None and kind not in _ENTITY_KINDS:
            raise ValueError(f"Unsupported logical entity kind: {kind!r}")
        return tuple(
            entity
            for entity in self.entities
            if entity.selectable and (kind is None or entity.kind == kind)
        )

    def entity(self, logical_id: str) -> LogicalEntity:
        """Look up one entity by its stable logical identifier."""
        for entity in self.entities:
            if entity.logical_id == logical_id:
                return entity
        raise KeyError(logical_id)

def describe_recipe_topology(recipe: NativeGeometry) -> RecipeTopology:
    """Return the conservative logical topology catalog for ``recipe``.

    The function is pure Python and performs no CAD construction.  Unsupported
    recipe objects raise ``TypeError``; supported but geometrically ambiguous
    recipes return a non-exact catalog with diagnostics.
    """

    if isinstance(recipe, RectangleGeometry):
        return _rectangle_topology(recipe)
    if isinstance(recipe, DiskGeometry):
        return _disk_topology(recipe)
    if isinstance(recipe, PlateWithHoleGeometry):
        return _plate_with_hole_topology(recipe)
    if isinstance(recipe, BoxGeometry):
        return _box_topology(recipe)
    if isinstance(recipe, CylinderGeometry):
        return _cylinder_topology(recipe)
    if isinstance(recipe, WireGeometry):
        return _wire_topology(recipe)
    if isinstance(recipe, SketchGeometry):
        return _sketch_topology(recipe)
    if isinstance(recipe, MovedGeometry):
        return _rigid_transform_topology(recipe, "move")
    if isinstance(recipe, RotatedGeometry):
        return _rigid_transform_topology(recipe, "rotate")
    if isinstance(recipe, ExtrudedGeometry):
        return _extruded_topology(recipe)
    if isinstance(recipe, RevolvedGeometry):
        return _revolved_topology(recipe)
    if isinstance(recipe, PathSweptGeometry):
        return _path_swept_topology(recipe)
    if isinstance(recipe, MultiBodyGeometry):
        return _multi_body_topology(recipe)
    if isinstance(recipe, BooleanGeometry):
        return _boolean_topology(recipe)
    raise TypeError(f"Unsupported native geometry recipe: {type(recipe).__name__}")


def topology_fingerprint_from_topology(
    topology: RecipeTopology,
) -> TopologyFingerprint:
    """Canonicalize a complete recipe topology as structured contract evidence."""

    if type(topology) is not RecipeTopology:
        raise TypeError("topology must be a RecipeTopology")
    return TopologyFingerprint(
        dimension=topology.dimension,
        exact=topology.exact,
        entities=tuple(
            TopologyFingerprintEntity(
                entity.kind,
                entity.logical_id,
                entity.semantic_role,
                entity.selectable,
                entity.topology_links,
            )
            for entity in topology.entities
        ),
    )


def topology_fingerprint_for_recipe(recipe: NativeGeometry) -> TopologyFingerprint:
    """Recompute canonical topology evidence from one geometry recipe."""

    return topology_fingerprint_from_topology(describe_recipe_topology(recipe))


def can_preserve_logical_references(before: object, after: object) -> bool:
    """Return whether logical references remain valid across a recipe edit.

    Unsupported recipes and topology that cannot be proved exactly are
    deliberately incompatible.  Parameter-only edits may preserve references
    when both catalogs expose the same exact structural signature.
    """

    if not isinstance(before, NATIVE_GEOMETRY_TYPES) or not isinstance(
        after,
        NATIVE_GEOMETRY_TYPES,
    ):
        return False
    before_fingerprint = topology_fingerprint_for_recipe(before)
    after_fingerprint = topology_fingerprint_for_recipe(after)
    return (
        before_fingerprint.exact
        and after_fingerprint.exact
        and before_fingerprint == after_fingerprint
    )


def surviving_logical_reference_ids(
    before: object,
    after: object,
) -> frozenset[str]:
    """Return output IDs whose identity and feature lineage survive an edit."""

    if not isinstance(before, NATIVE_GEOMETRY_TYPES) or not isinstance(
        after,
        NATIVE_GEOMETRY_TYPES,
    ):
        return frozenset()
    before_topology = describe_recipe_topology(before)
    after_topology = describe_recipe_topology(after)
    if (
        not before_topology.exact
        or not after_topology.exact
        or before_topology.dimension != after_topology.dimension
    ):
        return frozenset()
    if (
        topology_fingerprint_from_topology(before_topology)
        == topology_fingerprint_from_topology(after_topology)
    ):
        return frozenset(before_topology.signature.logical_ids)

    before_entities = {
        entity.logical_id: entity for entity in before_topology.entities
    }
    after_entities = {
        entity.logical_id: entity for entity in after_topology.entities
    }

    def provenance(topology: RecipeTopology, logical_id: str):
        return frozenset(
            (
                mapping.source,
                mapping.source_logical_id,
                mapping.relation,
            )
            for mapping in topology.transition.mappings
            if mapping.target_logical_id == logical_id
        )

    survivors: set[str] = set()
    for logical_id in before_entities.keys() & after_entities.keys():
        before_entity = before_entities[logical_id]
        after_entity = after_entities[logical_id]
        if (
            before_entity.kind != after_entity.kind
            or before_entity.dimension != after_entity.dimension
            or before_entity.semantic_role != after_entity.semantic_role
            or before_entity.selectable != after_entity.selectable
            or before_entity.topology_links != after_entity.topology_links
        ):
            continue
        before_provenance = provenance(before_topology, logical_id)
        after_provenance = provenance(after_topology, logical_id)
        if before_provenance and before_provenance == after_provenance:
            survivors.add(logical_id)
    return frozenset(survivors)


def logical_reference_transition_map(
    before: object,
    after: object,
) -> dict[str, tuple[str, ...]]:
    """Return proven same-kind logical-reference rewrites for one edit.

    Stable survivors map to themselves.  A committed strict Body Boolean maps
    target/tool references through its persisted proof; undo applies the same
    proof in reverse.  Cross-dimensional provenance is intentionally excluded
    because a face scope cannot safely become an edge scope.
    """

    if not isinstance(before, NATIVE_GEOMETRY_TYPES) or not isinstance(
        after,
        NATIVE_GEOMETRY_TYPES,
    ):
        return {}
    rewrites: dict[str, set[str]] = {
        logical_id: {logical_id}
        for logical_id in surviving_logical_reference_ids(before, after)
    }
    planar_forward = (
        after
        if isinstance(after, BooleanGeometry)
        and after.planar_context is not None
        and after.object_geometry == before
        else None
    )
    planar_reverse = (
        before
        if isinstance(before, BooleanGeometry)
        and before.planar_context is not None
        and before.object_geometry == after
        else None
    )
    if planar_forward is not None:
        _extend_planar_boolean_reference_map(
            rewrites,
            planar_forward.planar_context,
            reverse=False,
        )
    elif planar_reverse is not None:
        _extend_planar_boolean_reference_map(
            rewrites,
            planar_reverse.planar_context,
            reverse=True,
        )
    if not isinstance(before, MultiBodyGeometry) or not isinstance(
        after,
        MultiBodyGeometry,
    ):
        return {
            source: tuple(sorted(targets, key=_logical_id_sort_key))
            for source, targets in rewrites.items()
        }

    before_by_id = {body.id: body for body in before.bodies}
    after_by_id = {body.id: body for body in after.bodies}
    for target_id in before_by_id.keys() & after_by_id.keys():
        before_body = before_by_id[target_id]
        after_body = after_by_id[target_id]
        forward = _top_strict_body_boolean(after_body.recipe)
        reverse = _top_strict_body_boolean(before_body.recipe)
        if (
            forward is not None
            and forward.body_context is not None
            and forward.body_context.tool_body_id in before_by_id
            and forward.body_context.tool_body_id not in after_by_id
        ):
            _extend_strict_boolean_reference_map(
                rewrites,
                forward.body_context,
                reverse=False,
            )
        elif (
            reverse is not None
            and reverse.body_context is not None
            and reverse.body_context.tool_body_id not in before_by_id
            and reverse.body_context.tool_body_id in after_by_id
        ):
            _extend_strict_boolean_reference_map(
                rewrites,
                reverse.body_context,
                reverse=True,
            )
    return {
        source: tuple(sorted(targets, key=_logical_id_sort_key))
        for source, targets in rewrites.items()
        if targets
    }


def _extend_planar_boolean_reference_map(
    rewrites: dict[str, set[str]],
    context: PlanarBooleanContext,
    *,
    reverse: bool,
) -> None:
    candidates = tuple(
        (
            mapping.source_logical_id,
            mapping.target_logical_id,
        )
        for mapping in context.topology_mappings
        if mapping.source == "target"
        and LogicalEntityRef(mapping.source_logical_id).kind
        == LogicalEntityRef(mapping.target_logical_id).kind
    )
    if not reverse:
        targets_by_source: dict[str, set[str]] = {}
        for source, target in candidates:
            targets_by_source.setdefault(source, set()).add(target)
        for source, targets in targets_by_source.items():
            kind = LogicalEntityRef(source).kind
            if kind != "face" and len(targets) != 1:
                # Face scopes expand across a proven split.  Lower-dimensional
                # boundary scopes require one unique continuation.
                continue
            rewrites.setdefault(source, set()).update(targets)
        return
    sources_by_target: dict[str, set[str]] = {}
    for source, target in candidates:
        sources_by_target.setdefault(target, set()).add(source)
    for target, sources in sources_by_target.items():
        if len(sources) != 1:
            continue
        source = next(iter(sources))
        existing = rewrites.get(target, set())
        if existing and existing != {source}:
            continue
        rewrites.setdefault(target, set()).add(source)


def _top_strict_body_boolean(recipe: object) -> BooleanGeometry | None:
    if isinstance(recipe, BooleanGeometry) and recipe.body_context is not None:
        return recipe
    return None


def _extend_strict_boolean_reference_map(
    rewrites: dict[str, set[str]],
    context: BooleanBodyContext,
    *,
    reverse: bool,
) -> None:
    candidates: list[tuple[str, str]] = []
    for mapping in context.topology_mappings:
        if (
            not reverse
            and mapping.source == "tool"
            and LogicalEntityRef(mapping.source_logical_id).kind == "body"
        ):
            # A consumed Body scope cannot become the surviving target Body.
            # The Session-level undo tombstone restores it exactly on undo.
            continue
        source_body_id = (
            context.target_body_id
            if mapping.source == "target"
            else context.tool_body_id
        )
        source = _namespace_local_logical_id(
            source_body_id,
            mapping.source_logical_id,
        )
        target = _namespace_local_logical_id(
            context.target_body_id,
            mapping.target_logical_id,
        )
        if LogicalEntityRef(source).kind != LogicalEntityRef(target).kind:
            continue
        candidates.append((source, target))
    if not reverse:
        for source, target in candidates:
            rewrites.setdefault(source, set()).add(target)
        return
    sources_by_target: dict[str, set[str]] = {}
    for source, target in candidates:
        sources_by_target.setdefault(target, set()).add(source)
    for target, sources in sources_by_target.items():
        if len(sources) != 1:
            continue
        source = next(iter(sources))
        existing = rewrites.get(target, set())
        if existing and existing != {source}:
            continue
        rewrites.setdefault(target, set()).add(source)


def _namespace_local_logical_id(body_id: str, logical_id: str) -> str:
    reference = LogicalEntityRef(logical_id)
    if reference.kind == "body":
        return f"body:{body_id}"
    _kind, local_name = logical_id.split(":", 1)
    return f"{reference.kind}:{body_id}/{local_name}"


def _logical_id_sort_key(logical_id: str):
    return logical_ref_sort_key(LogicalEntityRef(logical_id))


def _logical_entity(
    kind: EntityKind,
    name: str,
    role: str,
    *,
    dimension: Literal[0, 1, 2, 3] | None = None,
    selectable: bool = True,
    diagnostic_code: str | None = None,
    topology_links: tuple[str, ...] = (),
) -> LogicalEntity:
    if dimension is None:
        if kind == "body":
            raise ValueError("Logical body entities require an explicit CAD dimension")
        dimension = {"point": 0, "edge": 1, "face": 2}[kind]
    return LogicalEntity(
        kind,
        dimension,
        f"{kind}:{name}",
        role,
        selectable,
        diagnostic_code,
        topology_links,
    )


def _make_topology(
    recipe: NativeGeometry,
    entities: tuple[LogicalEntity, ...],
    *,
    exact: bool,
    operation: str,
    diagnostics: tuple[TopologyDiagnostic, ...] = (),
    source_signatures: tuple[tuple[str, TopologySignature], ...] = (),
    mappings: tuple[TopologyMapping, ...] = (),
    transition_proven: bool | None = None,
) -> RecipeTopology:
    # Group kinds for predictable queries while retaining each recipe's
    # declared semantic order.  This preserves the existing one-based preview
    # contract (for example rectangle edges bottom/right/top/left).
    ordered = tuple(
        entity for kind in _ENTITY_KINDS for entity in entities if entity.kind == kind
    )
    logical_ids = tuple(entity.logical_id for entity in ordered)
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError("Recipe topology contains duplicate logical ids")

    diagnostic_codes = {diagnostic.code for diagnostic in diagnostics}
    for entity in ordered:
        if entity.kind == "body" and entity.dimension != geometry_dimension(recipe):
            raise ValueError(
                "Logical body CAD dimension must match the recipe dimension"
            )
        if (
            entity.diagnostic_code is not None
            and entity.diagnostic_code not in diagnostic_codes
        ):
            raise ValueError(
                f"Logical entity {entity.logical_id!r} references an unknown diagnostic"
            )

    source_names = {name for name, _signature in source_signatures}
    if len(source_names) != len(source_signatures):
        raise ValueError("Recipe topology transition contains duplicate source names")
    for mapping in mappings:
        if mapping.source not in source_names:
            raise ValueError(
                f"Topology mapping references unknown source {mapping.source!r}"
            )
        if mapping.target_logical_id not in logical_ids:
            raise ValueError(
                f"Topology mapping references unknown target {mapping.target_logical_id!r}"
            )
        source_signature = dict(source_signatures)[mapping.source]
        if mapping.source_logical_id not in source_signature.logical_ids:
            raise ValueError(
                f"Topology mapping references unknown input "
                f"{mapping.source_logical_id!r}"
            )

    dimension = geometry_dimension(recipe)
    signature = TopologySignature(
        dimension,
        tuple(
            (
                entity.kind,
                entity.logical_id,
                entity.semantic_role,
                entity.selectable,
            )
            for entity in ordered
        ),
        exact,
    )
    transition = TopologyTransition(
        operation,
        source_signatures,
        signature,
        mappings,
        exact if transition_proven is None else transition_proven,
    )
    return RecipeTopology(
        type(recipe).__name__,
        dimension,
        ordered,
        signature,
        transition,
        diagnostics,
    )


def _rectangle_entities() -> tuple[LogicalEntity, ...]:
    return (
        _logical_entity("point", "bottom-left", "corner.bottom-left"),
        _logical_entity("point", "bottom-right", "corner.bottom-right"),
        _logical_entity("point", "top-right", "corner.top-right"),
        _logical_entity("point", "top-left", "corner.top-left"),
        _logical_entity("edge", "bottom", "boundary.bottom"),
        _logical_entity("edge", "right", "boundary.right"),
        _logical_entity("edge", "top", "boundary.top"),
        _logical_entity("edge", "left", "boundary.left"),
        _logical_entity("face", "domain", "domain"),
        _logical_entity("body", "domain", "domain", dimension=2),
    )


def _rectangle_topology(recipe: RectangleGeometry) -> RecipeTopology:
    return _make_topology(
        recipe,
        _rectangle_entities(),
        exact=True,
        operation="primitive.rectangle",
    )


def _disk_entities() -> tuple[LogicalEntity, ...]:
    return (
        _logical_entity("edge", "outer", "boundary.outer"),
        _logical_entity("face", "domain", "domain"),
        _logical_entity("body", "domain", "domain", dimension=2),
    )


def _disk_topology(recipe: DiskGeometry) -> RecipeTopology:
    return _make_topology(
        recipe,
        _disk_entities(),
        exact=True,
        operation="primitive.disk",
    )


def _plate_with_hole_entities() -> tuple[LogicalEntity, ...]:
    rectangle = _rectangle_entities()
    return (
        *tuple(entity for entity in rectangle if entity.kind == "point"),
        _logical_entity("edge", "hole-loop", "boundary.hole-loop"),
        _logical_entity("edge", "outer-loop", "boundary.outer-loop"),
        *tuple(entity for entity in rectangle if entity.kind in {"face", "body"}),
    )


def _plate_with_hole_topology(recipe: PlateWithHoleGeometry) -> RecipeTopology:
    return _make_topology(
        recipe,
        _plate_with_hole_entities(),
        exact=True,
        operation="primitive.plate-with-hole",
    )


def _box_topology(recipe: BoxGeometry) -> RecipeTopology:
    entities = (
        _logical_entity("point", "bottom-front-left", "corner.bottom-front-left"),
        _logical_entity("point", "bottom-front-right", "corner.bottom-front-right"),
        _logical_entity("point", "bottom-back-right", "corner.bottom-back-right"),
        _logical_entity("point", "bottom-back-left", "corner.bottom-back-left"),
        _logical_entity("point", "top-front-left", "corner.top-front-left"),
        _logical_entity("point", "top-front-right", "corner.top-front-right"),
        _logical_entity("point", "top-back-right", "corner.top-back-right"),
        _logical_entity("point", "top-back-left", "corner.top-back-left"),
        _logical_entity("edge", "bottom-front", "boundary.bottom-front"),
        _logical_entity("edge", "bottom-right", "boundary.bottom-right"),
        _logical_entity("edge", "bottom-back", "boundary.bottom-back"),
        _logical_entity("edge", "bottom-left", "boundary.bottom-left"),
        _logical_entity("edge", "top-front", "boundary.top-front"),
        _logical_entity("edge", "top-right", "boundary.top-right"),
        _logical_entity("edge", "top-back", "boundary.top-back"),
        _logical_entity("edge", "top-left", "boundary.top-left"),
        _logical_entity("edge", "vertical-front-left", "boundary.vertical-front-left"),
        _logical_entity(
            "edge", "vertical-front-right", "boundary.vertical-front-right"
        ),
        _logical_entity("edge", "vertical-back-right", "boundary.vertical-back-right"),
        _logical_entity("edge", "vertical-back-left", "boundary.vertical-back-left"),
        _logical_entity("face", "bottom", "boundary.bottom"),
        _logical_entity("face", "top", "boundary.top"),
        _logical_entity("face", "front", "boundary.front"),
        _logical_entity("face", "right", "boundary.right"),
        _logical_entity("face", "back", "boundary.back"),
        _logical_entity("face", "left", "boundary.left"),
        _logical_entity("body", "domain", "domain", dimension=3),
    )
    return _make_topology(
        recipe,
        entities,
        exact=True,
        operation="primitive.box",
    )


def _cylinder_entities() -> tuple[LogicalEntity, ...]:
    return (
        _logical_entity("edge", "bottom-rim", "boundary.bottom-rim"),
        _logical_entity("edge", "top-rim", "boundary.top-rim"),
        _logical_entity("face", "bottom", "boundary.bottom"),
        _logical_entity("face", "top", "boundary.top"),
        _logical_entity("face", "outer", "boundary.outer"),
        _logical_entity("body", "domain", "domain", dimension=3),
    )


def _cylinder_topology(recipe: CylinderGeometry) -> RecipeTopology:
    return _make_topology(
        recipe,
        _cylinder_entities(),
        exact=True,
        operation="primitive.cylinder",
    )


def _wire_topology(recipe: WireGeometry) -> RecipeTopology:
    entities = (
        tuple(
            _logical_entity("point", point.name, "wire.point")
            for point in recipe.points
        )
        + tuple(
            _logical_entity(
                "edge",
                member.name,
                "wire.member",
                topology_links=tuple(
                    sorted(
                        (
                            f"point:{member.start}",
                            f"point:{member.end}",
                        )
                    )
                ),
            )
            for member in recipe.members
        )
        + (_logical_entity("body", "domain", "domain", dimension=1),)
    )
    return _make_topology(
        recipe,
        entities,
        exact=True,
        operation="primitive.wire",
    )


def _sketch_topology(recipe: SketchGeometry) -> RecipeTopology:
    """Resolve strict curve graphs or the frozen legacy contour contract."""
    if recipe.is_strict:
        return _strict_sketch_topology(recipe)
    expanded = expand_sketch_recipe(recipe)
    if len(recipe.contours) == 1:
        if axis_aligned_rectangle(expanded) is not None:
            entities = _rectangle_entities()
        elif transformed_circle(expanded) is not None:
            entities = _disk_entities()
        else:  # pragma: no cover - expansion owns the validated contour catalog
            raise TypeError(
                f"Unsupported expanded sketch recipe: {type(expanded).__name__}"
            )
        return _make_topology(
            recipe,
            entities,
            exact=True,
            operation="sketch.single-contour",
        )

    if isinstance(expanded, BooleanGeometry) and expanded.operation == "cut":
        outer = axis_aligned_rectangle(expanded.object_geometry)
        circle = transformed_circle(expanded.tool_geometry)
        if (
            outer is not None
            and circle is not None
            and outer.strictly_contains_circle(circle)
        ):
            return _make_topology(
                recipe,
                _grouped_hole_entities(include_hole_points=False),
                exact=True,
                operation="sketch.cut-contained-circle",
            )
        rectangle = axis_aligned_rectangle(expanded.tool_geometry)
        if (
            outer is not None
            and rectangle is not None
            and outer.strictly_contains_rectangle(rectangle)
        ):
            return _make_topology(
                recipe,
                _grouped_hole_entities(include_hole_points=True),
                exact=True,
                operation="sketch.cut-contained-rectangle",
            )

    return _unknown_topology(
        recipe,
        code="sketch.topology-unproven",
        message=(
            "Multiple material or cut contours require geometric intersection "
            "analysis; their result topology is not selectable before CAD build."
        ),
        operation="sketch.composite",
    )


def _strict_sketch_topology(recipe: SketchGeometry) -> RecipeTopology:
    analysis = analyze_sketch_profiles(recipe)
    if analysis.blocking_diagnostics or not analysis.profiles:
        first = next(
            iter(analysis.blocking_diagnostics),
        ) if analysis.blocking_diagnostics else None
        return _unknown_topology(
            recipe,
            code="sketch.profile-invalid" if first is None else first.code,
            message=(
                "严格草图 Profile 无法证明为可提交的平面拓扑"
                if first is None
                else first.message
            ),
            operation="sketch.curve-graph",
        )
    entities: list[LogicalEntity] = []
    for point in recipe.points:
        entities.append(
            _logical_entity(
                "point",
                point.id,
                "sketch.point",
                topology_links=tuple(
                    f"edge:{curve.id}"
                    for curve in recipe.curves
                    if point.id in _strict_curve_point_ids(curve)
                ),
            )
        )
    for curve in recipe.curves:
        links = tuple(
            sorted(
                f"point:{point_id}"
                for point_id in _strict_curve_point_ids(curve)
                if point_id != getattr(curve, "center_point_id", None)
            )
        )
        entities.append(
            _logical_entity(
                "edge",
                curve.id,
                "sketch.curve",
                topology_links=links,
            )
        )
    material_profiles = tuple(profile for profile in analysis.profiles if profile.is_material)
    hole_profiles = tuple(profile for profile in analysis.profiles if profile.is_hole)
    existing_curve_ids = {curve.id.casefold() for curve in recipe.curves}
    if len(material_profiles) == 1 and "outer-loop" not in existing_curve_ids:
        profile = material_profiles[0]
        entities.append(
            _logical_entity(
                "edge",
                "outer-loop",
                "boundary.outer-loop",
                topology_links=tuple(
                    f"edge:{curve_id.lstrip('-')}" for curve_id in profile.curve_ids
                ),
            )
        )
    if len(hole_profiles) == 1 and "hole-loop" not in existing_curve_ids:
        profile = hole_profiles[0]
        entities.append(
            _logical_entity(
                "edge",
                "hole-loop",
                "boundary.hole-loop",
                topology_links=tuple(
                    f"edge:{curve_id.lstrip('-')}" for curve_id in profile.curve_ids
                ),
            )
        )
    for profile in material_profiles:
        members = tuple(
            f"edge:{curve_id.lstrip('-')}"
            for curve_id in profile.curve_ids
        )
        entities.append(
            _logical_entity(
                "face",
                f"profile/{profile.id.split('/', 1)[-1]}",
                "sketch.profile",
                topology_links=members,
            )
        )
    # Keep deterministic aliases for the v1/v2 primitive contour contract.
    # They let an explicitly migrated rectangle retain named-region and mesh
    # intent while the strict graph exposes its stable point/curve/profile IDs.
    entity_ids = {entity.logical_id for entity in entities}

    def add_compatibility_alias(
        kind: EntityKind,
        name: str,
        role: str,
        links: tuple[str, ...] = (),
    ) -> None:
        logical_id = f"{kind}:{name}"
        if logical_id in entity_ids:
            return
        entities.append(
            _logical_entity(
                kind,
                name,
                role,
                topology_links=links,
            )
        )
        entity_ids.add(logical_id)

    if len(material_profiles) == 1:
        material_profile = material_profiles[0]
        profile_curve_ids = tuple(
            curve_id.lstrip("-") for curve_id in material_profile.curve_ids
        )
        profile_curves = tuple(
            recipe.curve(curve_id) for curve_id in profile_curve_ids
        )
        if len(profile_curves) == 1 and isinstance(profile_curves[0], SketchCircle):
            add_compatibility_alias(
                "edge",
                "outer",
                "boundary.outer",
                (f"edge:{profile_curves[0].id}",),
            )
        if len(profile_curves) == 4 and all(
            isinstance(curve, SketchLine) for curve in profile_curves
        ):
            profile_points = tuple(
                point
                for point in recipe.points
                if point.id
                in {
                    point_id
                    for curve in profile_curves
                    for point_id in _strict_curve_point_ids(curve)
                }
            )
            if len({round(point.u, 12) for point in profile_points}) == 2 and len(
                {round(point.v, 12) for point in profile_points}
            ) == 2:
                min_u = min(point.u for point in profile_points)
                max_u = max(point.u for point in profile_points)
                min_v = min(point.v for point in profile_points)
                max_v = max(point.v for point in profile_points)
                side_aliases: dict[str, str] = {}
                for curve in profile_curves:
                    start = recipe.point(curve.start_point_id)
                    end = recipe.point(curve.end_point_id)
                    if math.isclose(start.v, min_v) and math.isclose(end.v, min_v):
                        side_aliases["bottom"] = curve.id
                    elif math.isclose(start.u, max_u) and math.isclose(end.u, max_u):
                        side_aliases["right"] = curve.id
                    elif math.isclose(start.v, max_v) and math.isclose(end.v, max_v):
                        side_aliases["top"] = curve.id
                    elif math.isclose(start.u, min_u) and math.isclose(end.u, min_u):
                        side_aliases["left"] = curve.id
                if len(side_aliases) == 4:
                    for side, curve_id in side_aliases.items():
                        add_compatibility_alias(
                            "edge",
                            side,
                            f"boundary.{side}",
                            (f"edge:{curve_id}",),
                        )
                    point_aliases = {
                        "bottom-left": (min_u, min_v),
                        "bottom-right": (max_u, min_v),
                        "top-right": (max_u, max_v),
                        "top-left": (min_u, max_v),
                    }
                    for name, coordinate in point_aliases.items():
                        point = next(
                            (
                                value
                                for value in profile_points
                                if math.isclose(value.u, coordinate[0])
                                and math.isclose(value.v, coordinate[1])
                            ),
                            None,
                        )
                        if point is not None:
                            add_compatibility_alias(
                                "point",
                                name,
                                f"corner.{name}",
                                (f"point:{point.id}",),
                            )
        material_face = next(
            (
                entity
                for entity in entities
                if entity.kind == "face"
                and entity.logical_id == f"face:{material_profile.id}"
            ),
            None,
        )
        if material_face is not None:
            add_compatibility_alias(
                "face",
                "domain",
                "domain",
                (material_face.logical_id,),
            )
    face_ids = tuple(
        entity.logical_id for entity in entities if entity.kind == "face"
    )
    entities.append(
        _logical_entity(
            "body",
            "domain",
            "sketch.domain",
            dimension=2,
            topology_links=face_ids,
        )
    )
    return _make_topology(
        recipe,
        tuple(entities),
        exact=True,
        operation="sketch.curve-graph",
    )


def _strict_curve_point_ids(curve: object) -> tuple[str, ...]:
    if isinstance(curve, SketchLine):
        return curve.start_point_id, curve.end_point_id
    if isinstance(curve, SketchArc):
        return curve.start_point_id, curve.center_point_id, curve.end_point_id
    if isinstance(curve, SketchCircle):
        return (curve.center_point_id,)
    raise TypeError(f"Unsupported strict sketch curve: {type(curve).__name__}")


def _rigid_transform_topology(
    recipe: MovedGeometry | RotatedGeometry,
    operation: Literal["move", "rotate"],
) -> RecipeTopology:
    base = describe_recipe_topology(recipe.base)
    mappings = tuple(
        TopologyMapping(
            "base",
            entity.logical_id,
            entity.logical_id,
            "preserved",
        )
        for entity in base.entities
    )
    return _make_topology(
        recipe,
        base.entities,
        exact=base.exact,
        operation=operation,
        diagnostics=base.diagnostics,
        source_signatures=(("base", base.signature),),
        mappings=mappings,
        transition_proven=base.exact,
    )


def _logical_name(logical_id: str) -> str:
    return logical_id.split(":", 1)[1]


def _extruded_topology(recipe: ExtrudedGeometry) -> RecipeTopology:
    base = describe_recipe_topology(recipe.base)
    base_bodies = base.entities_of("body", selectable_only=True)
    try:
        selection = resolve_extrusion_source_faces(
            recipe.base,
            recipe.source_face_ids,
        )
    except ExtrusionSourceResolutionError as error:
        return _unknown_topology(
            recipe,
            code=error.code,
            message=str(error),
            operation="extrude",
            source_signatures=(("base", base.signature),),
        )
    if len(base_bodies) != 1:
        return _unknown_topology(
            recipe,
            code="extrude.source-face.topology-unproven",
            message="拉伸源必须具有唯一的二维 logical body",
            operation="extrude",
            source_signatures=(("base", base.signature),),
        )

    entities: list[LogicalEntity] = []
    mappings: list[TopologyMapping] = []
    single_source = len(selection.face_ids) == 1
    hole_edges_by_face = _extrusion_hole_edges_by_face(
        recipe.base,
        selection.face_ids,
    )

    for source_face_id in selection.face_ids:
        source_face = base.entity(source_face_id)
        source_face_name = _logical_name(source_face.logical_id)
        edge_ids, point_ids = extrusion_face_boundary_ids(
            recipe.base,
            source_face_id,
        )
        base_edges = tuple(base.entity(logical_id) for logical_id in edge_ids)
        base_points = tuple(base.entity(logical_id) for logical_id in point_ids)

        for level in ("bottom", "top"):
            for point in base_points:
                name = _logical_name(point.logical_id)
                target_name = (
                    f"{level}/{name}"
                    if single_source
                    else f"{level}/{source_face_name}/{name}"
                )
                target = _logical_entity(
                    "point",
                    target_name,
                    f"copy.{level}.{point.semantic_role}",
                )
                entities.append(target)
                mappings.append(
                    TopologyMapping(
                        "base",
                        point.logical_id,
                        target.logical_id,
                        "derived",
                    )
                )

        for level in ("bottom", "top"):
            for edge in base_edges:
                name = _logical_name(edge.logical_id)
                target_name = (
                    f"{level}/{name}"
                    if single_source
                    else f"{level}/{source_face_name}/{name}"
                )
                target = _logical_entity(
                    "edge",
                    target_name,
                    f"copy.{level}.{edge.semantic_role}",
                )
                entities.append(target)
                mappings.append(
                    TopologyMapping(
                        "base",
                        edge.logical_id,
                        target.logical_id,
                        "derived",
                    )
                )

        for point in base_points:
            name = _logical_name(point.logical_id)
            target_name = (
                f"vertical/{name}"
                if single_source
                else f"vertical/{source_face_name}/{name}"
            )
            vertical = _logical_entity(
                "edge",
                target_name,
                f"sweep.{point.semantic_role}",
            )
            entities.append(vertical)
            mappings.append(
                TopologyMapping(
                    "base",
                    point.logical_id,
                    vertical.logical_id,
                    "derived",
                )
            )

        for level in ("bottom", "top"):
            target_name = (
                level
                if single_source
                else f"{level}/{source_face_name}"
            )
            target = _logical_entity(
                "face",
                target_name,
                f"copy.{level}.{source_face.semantic_role}",
            )
            entities.append(target)
            mappings.append(
                TopologyMapping(
                    "base",
                    source_face.logical_id,
                    target.logical_id,
                    "derived",
                )
            )

        for edge in base_edges:
            name = _logical_name(edge.logical_id)
            target_name = (
                f"side/{name}"
                if single_source
                else f"side/{source_face_name}/{name}"
            )
            side = _logical_entity(
                "face",
                target_name,
                (
                    "sweep.boundary.hole"
                    if edge.logical_id in hole_edges_by_face[source_face_id]
                    or "hole" in edge.semantic_role
                    else "sweep.boundary.outer"
                ),
            )
            entities.append(side)
            mappings.append(
                TopologyMapping(
                    "base",
                    edge.logical_id,
                    side.logical_id,
                    "derived",
                )
            )

    body = _logical_entity("body", "domain", "sweep.domain", dimension=3)
    entities.append(body)
    mappings.append(
        TopologyMapping(
            "base",
            base_bodies[0].logical_id,
            body.logical_id,
            "derived",
        )
    )
    return _make_topology(
        recipe,
        tuple(entities),
        exact=True,
        operation="extrude",
        source_signatures=(("base", base.signature),),
        mappings=tuple(mappings),
    )


def _extrusion_hole_edges_by_face(
    base: object,
    face_ids: tuple[str, ...],
) -> dict[str, frozenset[str]]:
    strict_base = base
    while isinstance(strict_base, (MovedGeometry, RotatedGeometry)):
        strict_base = strict_base.base
    if type(strict_base) is not SketchGeometry or not strict_base.is_strict:
        return {face_id: frozenset() for face_id in face_ids}
    analysis = analyze_sketch_profiles(strict_base)
    return {
        face_id: frozenset(
            f"edge:{curve_id.lstrip('-')}"
            for profile in analysis.profiles
            if profile.is_hole
            and profile.parent_profile_id == face_id.split(":", 1)[1]
            for curve_id in profile.curve_ids
        )
        for face_id in face_ids
    }


def _revolved_topology(recipe: RevolvedGeometry) -> RecipeTopology:
    """Expose deterministic cap and grouped-side lineage for one revolution."""

    base = describe_recipe_topology(recipe.base)
    try:
        selection = resolve_extrusion_source_faces(
            recipe.base,
            recipe.source_face_ids,
        )
    except ExtrusionSourceResolutionError as error:
        return _unknown_topology(
            recipe,
            code=error.code,
            message=str(error),
            operation="revolve",
            source_signatures=(("base", base.signature),),
        )
    base_bodies = base.entities_of("body", selectable_only=True)
    if (
        not base.exact
        or len(base_bodies) != 1
        or len(selection.face_ids) != 1
    ):
        return _unknown_topology(
            recipe,
            code="revolve.source-face.topology-unproven",
            message="扫掠源必须具有唯一且可验证的二维 logical body",
            operation="revolve",
            source_signatures=(("base", base.signature),),
        )
    source_face = base.entity(selection.face_ids[0])
    entities: list[LogicalEntity] = []
    mappings: list[TopologyMapping] = []
    if recipe.angle_degrees < 360.0:
        for level in ("start", "end"):
            face = _logical_entity(
                "face",
                level,
                f"revolve.{level}.{source_face.semantic_role}",
            )
            entities.append(face)
            mappings.append(
                TopologyMapping(
                    "base",
                    source_face.logical_id,
                    face.logical_id,
                    "derived",
                )
            )
    sides = _logical_entity(
        "face",
        "sides",
        "revolve.boundary.sides",
    )
    entities.append(sides)
    mappings.append(
        TopologyMapping(
            "base",
            source_face.logical_id,
            sides.logical_id,
            "derived",
        )
    )
    body = _logical_entity(
        "body",
        "domain",
        "revolve.domain",
        dimension=3,
    )
    entities.append(body)
    mappings.append(
        TopologyMapping(
            "base",
            base_bodies[0].logical_id,
            body.logical_id,
            "derived",
        )
    )
    return _make_topology(
        recipe,
        tuple(entities),
        exact=True,
        operation="revolve",
        source_signatures=(("base", base.signature),),
        mappings=tuple(mappings),
    )


def _path_swept_topology(recipe: PathSweptGeometry) -> RecipeTopology:
    """Describe the deterministic lineage of one validated open-path sweep."""

    base = describe_recipe_topology(recipe.base)
    path = describe_recipe_topology(recipe.path)
    try:
        selection = resolve_extrusion_source_faces(
            recipe.base,
            recipe.source_face_ids,
        )
    except ExtrusionSourceResolutionError as error:
        return _unknown_topology(
            recipe,
            code=error.code,
            message=str(error),
            operation="path_sweep",
            source_signatures=(("base", base.signature), ("path", path.signature)),
        )
    if (
        not base.exact
        or not path.exact
        or len(selection.face_ids) != 1
        or len(base.entities_of("body", selectable_only=True)) != 1
        or len(path.entities_of("body", selectable_only=True)) != 1
    ):
        return _unknown_topology(
            recipe,
            code="path-sweep.source.topology-unproven",
            message="路径扫掠需要一个可验证的 material Profile 和开放路径",
            operation="path_sweep",
            source_signatures=(("base", base.signature), ("path", path.signature)),
        )
    source_face_id = selection.face_ids[0]
    source_face = base.entity(source_face_id)
    edge_ids, _point_ids = extrusion_face_boundary_ids(
        recipe.base,
        source_face_id,
    )
    base_edges = tuple(base.entity(logical_id) for logical_id in edge_ids)
    entities: list[LogicalEntity] = [
        _logical_entity(
            "face",
            "start",
            f"copy.start.{source_face.semantic_role}",
        ),
        _logical_entity(
            "face",
            "end",
            f"copy.end.{source_face.semantic_role}",
        ),
    ]
    mappings: list[TopologyMapping] = [
        TopologyMapping("base", source_face_id, "face:start", "derived"),
        TopologyMapping("base", source_face_id, "face:end", "derived"),
    ]
    hole_edges = _extrusion_hole_edges_by_face(
        recipe.base,
        selection.face_ids,
    )[source_face_id]
    for edge in base_edges:
        name = _logical_name(edge.logical_id)
        side = _logical_entity(
            "face",
            f"side/{name}",
            (
                "sweep.boundary.hole"
                if edge.logical_id in hole_edges or "hole" in edge.semantic_role
                else "sweep.boundary.outer"
            ),
        )
        entities.append(side)
        mappings.append(
            TopologyMapping("base", edge.logical_id, side.logical_id, "derived")
        )
    body = _logical_entity("body", "domain", "path-sweep.domain", dimension=3)
    entities.append(body)
    base_body = base.entities_of("body", selectable_only=True)[0]
    path_body = path.entities_of("body", selectable_only=True)[0]
    mappings.extend(
        (
            TopologyMapping("base", base_body.logical_id, body.logical_id, "derived"),
            TopologyMapping("path", path_body.logical_id, body.logical_id, "derived"),
        )
    )
    return _make_topology(
        recipe,
        tuple(entities),
        exact=True,
        operation="path_sweep",
        source_signatures=(("base", base.signature), ("path", path.signature)),
        mappings=tuple(mappings),
    )
def _boolean_topology(recipe: BooleanGeometry) -> RecipeTopology:
    if recipe.body_context is not None:
        return _strict_body_boolean_topology(recipe, recipe.body_context)
    if recipe.part_context is not None:
        from .part_boolean import localize_part_boolean_context

        return _strict_body_boolean_topology(
            recipe,
            localize_part_boolean_context(recipe.part_context),
        )
    if recipe.planar_context is not None:
        return _strict_planar_boolean_topology(recipe, recipe.planar_context)
    object_topology = describe_recipe_topology(recipe.object_geometry)
    tool_topology = describe_recipe_topology(recipe.tool_geometry)
    sources = (
        ("object", object_topology.signature),
        ("tool", tool_topology.signature),
    )

    if recipe.operation == "cut":
        object_frame = axis_aligned_rectangle(recipe.object_geometry)
        circle_frame = transformed_circle(recipe.tool_geometry)
        if (
            object_frame is not None
            and circle_frame is not None
            and object_frame.strictly_contains_circle(circle_frame)
        ):
            entities = _grouped_hole_entities(include_hole_points=False)
            mappings = _outer_group_mappings(object_topology) + (
                TopologyMapping(
                    "tool",
                    "edge:outer",
                    "edge:hole-loop",
                    "derived",
                ),
            )
            return _make_topology(
                recipe,
                entities,
                exact=True,
                operation="boolean.cut-contained-circle",
                source_signatures=sources,
                mappings=mappings,
            )

        tool_frame = axis_aligned_rectangle(recipe.tool_geometry)
        if (
            object_frame is not None
            and tool_frame is not None
            and object_frame.strictly_contains_rectangle(tool_frame)
        ):
            entities = _grouped_hole_entities(include_hole_points=True)
            tool_mappings = _rectangular_hole_mappings()
            return _make_topology(
                recipe,
                entities,
                exact=True,
                operation="boolean.cut-contained-rectangle",
                source_signatures=sources,
                mappings=_outer_group_mappings(object_topology) + tool_mappings,
            )

    return _unknown_topology(
        recipe,
        code="boolean.topology-unproven",
        message=(
            "Boolean result entities depend on geometric intersections; the "
            "logical catalog refuses to infer selectable topology."
        ),
        operation=f"boolean.{recipe.operation}",
        source_signatures=sources,
    )


def _strict_planar_boolean_topology(
    recipe: BooleanGeometry,
    context: PlanarBooleanContext,
) -> RecipeTopology:
    target = describe_recipe_topology(recipe.object_geometry)
    tool = describe_recipe_topology(recipe.tool_geometry)
    sources = (("target", target.signature), ("tool", tool.signature))
    if not context.proven:
        return _unknown_topology(
            recipe,
            code="planar-boolean.lineage.unproven",
            message=(
                "Strict planar Boolean has not completed OCC lineage proof."
            ),
            operation=f"planar.boolean.{recipe.operation}",
            source_signatures=sources,
        )
    entities = tuple(
        LogicalEntity(
            item.kind,
            {"point": 0, "edge": 1, "face": 2, "body": 2}[item.kind],
            item.logical_id,
            item.semantic_role,
            True,
            None,
            item.topology_links,
        )
        for item in context.result_entities
    )
    mappings = tuple(
        TopologyMapping(
            item.source,
            item.source_logical_id,
            item.target_logical_id,
            item.relation,
        )
        for item in context.topology_mappings
    )
    return _make_topology(
        recipe,
        entities,
        exact=True,
        operation=f"planar.boolean.{recipe.operation}",
        source_signatures=sources,
        mappings=mappings,
    )


def _strict_body_boolean_topology(
    recipe: BooleanGeometry,
    context: BooleanBodyContext,
) -> RecipeTopology:
    target = describe_recipe_topology(recipe.object_geometry)
    tool = describe_recipe_topology(recipe.tool_geometry)
    sources = (("target", target.signature), ("tool", tool.signature))
    if not context.proven:
        return _unknown_topology(
            recipe,
            code="boolean.lineage.unproven",
            message="Strict Body Boolean has not completed OCC lineage proof.",
            operation=f"body.boolean.{recipe.operation}",
            source_signatures=sources,
        )
    entities = tuple(
        LogicalEntity(
            item.kind,
            {"point": 0, "edge": 1, "face": 2, "body": 3}[item.kind],
            item.logical_id,
            item.semantic_role,
            True,
            None,
            item.topology_links,
        )
        for item in context.result_entities
    )
    mappings = tuple(
        TopologyMapping(
            item.source,
            item.source_logical_id,
            item.target_logical_id,
            item.relation,
        )
        for item in context.topology_mappings
    )
    return _make_topology(
        recipe,
        entities,
        exact=True,
        operation=f"body.boolean.{recipe.operation}",
        source_signatures=sources,
        mappings=mappings,
    )


def _multi_body_topology(recipe: MultiBodyGeometry) -> RecipeTopology:
    diagnostics = (
        TopologyDiagnostic(
            "multi_body.aggregate",
            "body:domain is an internal aggregate and is not selectable.",
            ("body:domain",),
        ),
    )
    entities: list[LogicalEntity] = []
    sources: list[tuple[str, TopologySignature]] = []
    mappings: list[TopologyMapping] = []
    for body in recipe.bodies:
        local = describe_recipe_topology(body.recipe)
        if not local.exact:
            return _unknown_topology(
                recipe,
                code="multi_body.body-topology-unproven",
                message=f"Body {body.id} does not have exact local topology.",
                operation="multi-body",
                source_signatures=tuple(sources),
            )
        source_name = f"body:{body.id}"
        sources.append((source_name, local.signature))
        for item in local.entities:
            target_id = _body_namespaced_id(body, item.logical_id)
            links = tuple(
                _body_namespaced_id(body, link)
                for link in item.topology_links
            )
            entities.append(
                LogicalEntity(
                    item.kind,
                    item.dimension,
                    target_id,
                    item.semantic_role,
                    item.selectable,
                    item.diagnostic_code,
                    links,
                )
            )
            mappings.append(
                TopologyMapping(
                    source_name,
                    item.logical_id,
                    target_id,
                    "derived",
                )
            )
        local_bodies = local.entities_of("body", selectable_only=True)
        if len(local_bodies) != 1:
            raise ValueError(
                f"Body {body.id} must expose exactly one selectable local body"
            )
    entities.append(
        LogicalEntity(
            "body",
            3,
            "body:domain",
            "aggregate.domain",
            False,
            "multi_body.aggregate",
        )
    )
    for body in recipe.bodies:
        local = describe_recipe_topology(body.recipe)
        mappings.append(
            TopologyMapping(
                f"body:{body.id}",
                local.entities_of("body", selectable_only=True)[0].logical_id,
                "body:domain",
                "derived",
            )
        )
    return _make_topology(
        recipe,
        tuple(entities),
        exact=True,
        operation="multi-body",
        diagnostics=diagnostics,
        source_signatures=tuple(sources),
        mappings=tuple(mappings),
    )


def _body_namespaced_id(body: SolidBody, logical_id: str) -> str:
    reference = LogicalEntityRef(logical_id)
    if reference.kind == "body":
        return f"body:{body.id}"
    _kind, local_name = logical_id.split(":", 1)
    return f"{reference.kind}:{body.id}/{local_name}"


def canonicalize_multi_body_logical_id(
    recipe: MultiBodyGeometry,
    logical_id: str,
) -> str:
    """Resolve a canonical ID or an unambiguous legacy single-Body alias."""

    reference = LogicalEntityRef(logical_id)
    topology = describe_recipe_topology(recipe)
    try:
        topology.entity(reference.logical_id)
    except KeyError:
        pass
    else:
        if logical_id == "body:domain" and len(recipe.bodies) == 1:
            return f"body:{recipe.bodies[0].id}"
        return logical_id
    if len(recipe.bodies) != 1:
        raise KeyError(logical_id)
    body = recipe.bodies[0]
    if reference.kind == "body" and logical_id == "body:domain":
        return f"body:{body.id}"
    _kind, local_name = logical_id.split(":", 1)
    candidate = f"{reference.kind}:{body.id}/{local_name}"
    topology.entity(candidate)
    return candidate


def _outer_group_mappings(
    topology: RecipeTopology,
) -> tuple[TopologyMapping, ...]:
    preserved = tuple(
        TopologyMapping(
            "object",
            entity.logical_id,
            entity.logical_id,
            "preserved",
        )
        for entity in topology.entities
        if entity.kind in {"point", "face", "body"}
    )
    grouped_edges = tuple(
        TopologyMapping(
            "object",
            entity.logical_id,
            "edge:outer-loop",
            "derived",
        )
        for entity in topology.entities_of("edge")
    )
    return preserved + grouped_edges


def _grouped_hole_entities(
    *,
    include_hole_points: bool,
) -> tuple[LogicalEntity, ...]:
    rectangle = _rectangle_entities()
    names = ("bottom-left", "bottom-right", "top-right", "top-left")
    hole_points = (
        tuple(
            _logical_entity(
                "point",
                f"hole-{name}",
                f"corner.hole-{name}",
            )
            for name in names
        )
        if include_hole_points
        else ()
    )
    return (
        *tuple(entity for entity in rectangle if entity.kind == "point"),
        *hole_points,
        _logical_entity("edge", "hole-loop", "boundary.hole-loop"),
        _logical_entity("edge", "outer-loop", "boundary.outer-loop"),
        *tuple(entity for entity in rectangle if entity.kind in {"face", "body"}),
    )


def _rectangular_hole_mappings() -> tuple[TopologyMapping, ...]:
    names = ("bottom-left", "bottom-right", "top-right", "top-left")
    edge_names = ("bottom", "right", "top", "left")
    point_mappings = tuple(
        TopologyMapping(
            "tool",
            f"point:{name}",
            f"point:hole-{name}",
            "derived",
        )
        for name in names
    )
    edge_mappings = tuple(
        TopologyMapping(
            "tool",
            f"edge:{name}",
            "edge:hole-loop",
            "derived",
        )
        for name in edge_names
    )
    return point_mappings + edge_mappings


def _unknown_topology(
    recipe: NativeGeometry,
    *,
    code: str,
    message: str,
    operation: str,
    source_signatures: tuple[tuple[str, TopologySignature], ...] = (),
) -> RecipeTopology:
    dimension = geometry_dimension(recipe)
    if dimension == 1:
        entities = (
            _logical_entity(
                "body",
                "result",
                "result.unproven",
                dimension=1,
                selectable=False,
                diagnostic_code=code,
            ),
        )
    elif dimension == 2:
        entities = (
            _logical_entity(
                "face",
                "result",
                "result.unproven",
                selectable=False,
                diagnostic_code=code,
            ),
            _logical_entity(
                "body",
                "result",
                "result.unproven",
                dimension=2,
                selectable=False,
                diagnostic_code=code,
            ),
        )
    else:
        entities = (
            _logical_entity(
                "body",
                "result",
                "result.unproven",
                dimension=3,
                selectable=False,
                diagnostic_code=code,
            ),
        )
    logical_ids = tuple(entity.logical_id for entity in entities)
    diagnostic = TopologyDiagnostic(code, message, logical_ids)
    return _make_topology(
        recipe,
        entities,
        exact=False,
        operation=operation,
        diagnostics=(diagnostic,),
        source_signatures=source_signatures,
        transition_proven=False,
    )


__all__ = [
    "EntityKind",
    "LogicalEntity",
    "RecipeTopology",
    "TOPOLOGY_REFERENCE_CONTRACT",
    "TopologyDiagnostic",
    "TopologyFingerprint",
    "TopologyFingerprintEntity",
    "TopologyMapping",
    "TopologySignature",
    "TopologyTransition",
    "TransitionRelation",
    "can_preserve_logical_references",
    "canonicalize_multi_body_logical_id",
    "describe_recipe_topology",
    "logical_reference_transition_map",
    "surviving_logical_reference_ids",
    "topology_fingerprint_for_recipe",
    "topology_fingerprint_from_topology",
]
