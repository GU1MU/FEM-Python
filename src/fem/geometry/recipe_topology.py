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

from .references import (
    EntityKind,
    LogicalEntityRef,
    logical_ref_sort_key,
)
from .recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
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

    def __post_init__(self) -> None:
        if self.kind not in _ENTITY_KINDS:
            raise ValueError(f"Unsupported logical entity kind: {self.kind!r}")
        expected_dimension = {"point": 0, "edge": 1, "face": 2}.get(self.kind)
        if expected_dimension is not None and self.dimension != expected_dimension:
            raise ValueError(
                f"Logical {self.kind} entities must have CAD dimension "
                f"{expected_dimension}"
            )
        if self.kind == "body" and self.dimension not in {2, 3}:
            raise ValueError("Logical body entities must have CAD dimension 2 or 3")
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

    dimension: Literal[2, 3]
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


@dataclass(frozen=True, slots=True)
class TopologyFingerprint:
    """Canonical structured evidence for one logical topology contract."""

    dimension: Literal[2, 3]
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
            or self.dimension not in {2, 3}
        ):
            raise ValueError("topology fingerprint dimension must be 2 or 3")
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
    dimension: Literal[2, 3]
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
    if isinstance(recipe, SketchGeometry):
        return _sketch_topology(recipe)
    if isinstance(recipe, MovedGeometry):
        return _rigid_transform_topology(recipe, "move")
    if isinstance(recipe, RotatedGeometry):
        return _rigid_transform_topology(recipe, "rotate")
    if isinstance(recipe, ExtrudedGeometry):
        return _extruded_topology(recipe)
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


def _logical_entity(
    kind: EntityKind,
    name: str,
    role: str,
    *,
    dimension: Literal[0, 1, 2, 3] | None = None,
    selectable: bool = True,
    diagnostic_code: str | None = None,
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


def _sketch_topology(recipe: SketchGeometry) -> RecipeTopology:
    """Resolve the proven single-domain sketch whitelist exactly."""
    if len(recipe.contours) == 1 and recipe.contours[0].operation == "material":
        contour = recipe.contours[0]
        if isinstance(contour, SketchRectangle):
            entities = _rectangle_entities()
        elif isinstance(contour, SketchCircle):
            entities = _disk_entities()
        else:  # pragma: no cover - SketchGeometry validates contour types
            raise TypeError(f"Unsupported sketch contour: {type(contour).__name__}")
        return _make_topology(
            recipe,
            entities,
            exact=True,
            operation="sketch.single-contour",
        )

    material = tuple(
        contour for contour in recipe.contours if contour.operation == "material"
    )
    cuts = tuple(contour for contour in recipe.contours if contour.operation == "cut")
    if (
        len(material) == 1
        and isinstance(material[0], SketchRectangle)
        and len(cuts) == 1
    ):
        outer = (
            material[0].x,
            material[0].y,
            material[0].width,
            material[0].height,
        )
        cut = cuts[0]
        if isinstance(cut, SketchCircle) and _circle_strictly_inside_rectangle(
            (cut.x, cut.y, cut.radius),
            outer,
        ):
            return _make_topology(
                recipe,
                _grouped_hole_entities(include_hole_points=False),
                exact=True,
                operation="sketch.cut-contained-circle",
            )
        if isinstance(cut, SketchRectangle) and (
            _rectangle_strictly_inside_rectangle(
                (cut.x, cut.y, cut.width, cut.height),
                outer,
            )
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
    base_faces = base.entities_of("face", selectable_only=True)
    base_bodies = base.entities_of("body", selectable_only=True)
    if not base.exact or len(base_faces) != 1 or len(base_bodies) != 1:
        return _unknown_topology(
            recipe,
            code="extrude.base-topology-unproven",
            message=(
                "Extrusion needs one exact selectable source face; the source "
                "recipe topology cannot be propagated safely."
            ),
            operation="extrude",
            source_signatures=(("base", base.signature),),
        )

    entities: list[LogicalEntity] = []
    mappings: list[TopologyMapping] = []
    base_points = base.entities_of("point", selectable_only=True)
    base_edges = base.entities_of("edge", selectable_only=True)

    # Match the compatibility preview order: bottom copies, top copies, then
    # vertical/swept entities.  The ordering is semantic and tag-independent.
    for level in ("bottom", "top"):
        for point in base_points:
            name = _logical_name(point.logical_id)
            target = _logical_entity(
                "point",
                f"{level}/{name}",
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
            target = _logical_entity(
                "edge",
                f"{level}/{name}",
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
        vertical = _logical_entity(
            "edge",
            f"vertical/{name}",
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

    source_face = base_faces[0]
    for level in ("bottom", "top"):
        target = _logical_entity(
            "face",
            level,
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
        side = _logical_entity(
            "face",
            f"side/{name}",
            f"sweep.{edge.semantic_role}",
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


def _boolean_topology(recipe: BooleanGeometry) -> RecipeTopology:
    object_topology = describe_recipe_topology(recipe.object_geometry)
    tool_topology = describe_recipe_topology(recipe.tool_geometry)
    sources = (
        ("object", object_topology.signature),
        ("tool", tool_topology.signature),
    )

    if recipe.operation == "cut":
        object_frame = _axis_aligned_rectangle(recipe.object_geometry)
        circle_frame = _translated_circle(recipe.tool_geometry)
        if (
            object_frame is not None
            and circle_frame is not None
            and _circle_strictly_inside_rectangle(circle_frame, object_frame)
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

        tool_frame = _axis_aligned_rectangle(recipe.tool_geometry)
        if (
            object_frame is not None
            and tool_frame is not None
            and _rectangle_strictly_inside_rectangle(tool_frame, object_frame)
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


def _translated_circle(
    recipe: NativeGeometry,
) -> tuple[float, float, float] | None:
    if isinstance(recipe, DiskGeometry):
        return 0.0, 0.0, recipe.radius
    if isinstance(recipe, MovedGeometry):
        circle = _translated_circle(recipe.base)
        if circle is None or recipe.dz != 0.0:
            return None
        x, y, radius = circle
        return x + recipe.dx, y + recipe.dy, radius
    if isinstance(recipe, RotatedGeometry):
        circle = _translated_circle(recipe.base)
        if circle is None:
            return None
        x, y, radius = circle
        angle = math.radians(recipe.angle_degrees)
        return (
            x * math.cos(angle) - y * math.sin(angle),
            x * math.sin(angle) + y * math.cos(angle),
            radius,
        )
    return None


def _circle_strictly_inside_rectangle(
    circle: tuple[float, float, float],
    rectangle: tuple[float, float, float, float],
) -> bool:
    center_x, center_y, radius = circle
    x, y, width, height = rectangle
    return (
        x < center_x - radius
        and center_x + radius < x + width
        and y < center_y - radius
        and center_y + radius < y + height
    )


def _rectangle_strictly_inside_rectangle(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    inner_x, inner_y, inner_width, inner_height = inner
    outer_x, outer_y, outer_width, outer_height = outer
    return (
        outer_x < inner_x
        and inner_x + inner_width < outer_x + outer_width
        and outer_y < inner_y
        and inner_y + inner_height < outer_y + outer_height
    )


def _unknown_topology(
    recipe: NativeGeometry,
    *,
    code: str,
    message: str,
    operation: str,
    source_signatures: tuple[tuple[str, TopologySignature], ...] = (),
) -> RecipeTopology:
    dimension = geometry_dimension(recipe)
    if dimension == 2:
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
    "describe_recipe_topology",
    "topology_fingerprint_for_recipe",
    "topology_fingerprint_from_topology",
]
