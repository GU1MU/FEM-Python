"""A4 stable scopes and reversible material/section authoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from fem.application import (
    BeamOrientation,
    MeshEntityRef,
    ModelDefinitions,
    NamedRegion,
    RegionAssignment,
    ScopedDefinitionBatch,
    SectionDefinition,
    describe_native_regions,
    validate_logical_reference,
)
from fem.application.native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
    mesh_references_for_logical_entities,
)
from fem.core.model import (
    AnalysisStep,
    BodyForce,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)
from fem.geometry import (
    LogicalEntityRef,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    analyze_sketch_profiles,
    namespace_part_logical_id,
)
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.geometry.recipes import PlateWithHoleGeometry
from fem.selection import edges as mesh_edges

from .authoring import (
    AgentProposal,
    AuthoringContext,
    ModelOperation,
    ModelPatch,
    OperationKind,
    ProposalKind,
)
from .naming import NameAllocator


class ScopeSelectionError(ValueError):
    """A semantic scope cannot be proved against the accepted mesh."""


@dataclass(frozen=True, slots=True)
class _PlateScopeGeometry:
    left_x: float
    right_x: float
    height: float
    hole_curves: tuple[tuple[str, float, float, float], ...]


class _Snapshot(Protocol):
    session_id: str
    session_revision: int
    source_kind: str | None
    active_part_id: str | None
    parts: Sequence[object]
    named_regions: Mapping[str, NamedRegion]
    materials: Sequence[MaterialDefinition]
    sections: Sequence[SectionDefinition]
    assignments: Sequence[RegionAssignment]
    steps: Sequence[object]
    artifact: object | None
    runs: Sequence[object]
    model_current: bool


@dataclass(frozen=True, slots=True)
class ScopeSelectionEvidence:
    """Bounded proof for one semantic mesh scope."""

    alias: str
    semantic: str
    source: str
    entity_kind: str
    expected_count: int
    matched_count: int
    bounds: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if self.entity_kind not in {"edge", "element"}:
            raise ValueError("A4 evidence supports only edge or element scopes")
        if (
            isinstance(self.expected_count, bool)
            or not isinstance(self.expected_count, int)
            or self.expected_count < 1
            or self.matched_count != self.expected_count
        ):
            raise ScopeSelectionError(
                f"{self.semantic} scope count does not match its exact evidence"
            )
        if (
            len(self.bounds) != 4
            or any(not math.isfinite(float(value)) for value in self.bounds)
        ):
            raise ValueError("scope bounds must contain four finite values")

    def to_dict(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "semantic": self.semantic,
            "source": self.source,
            "entity_kind": self.entity_kind,
            "expected_count": self.expected_count,
            "matched_count": self.matched_count,
            "bounds": list(self.bounds),
        }


@dataclass(frozen=True, slots=True)
class PlateScopeSet:
    """The four proven scopes required by the first milestone."""

    regions: tuple[NamedRegion, ...]
    evidence: tuple[ScopeSelectionEvidence, ...]

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        evidence = tuple(self.evidence)
        if len(regions) != 4 or len(evidence) != 4:
            raise ScopeSelectionError(
                "eccentric plate requires exactly four semantic scopes"
            )
        if {item.name for item in regions} != {
            item.alias for item in evidence
        }:
            raise ScopeSelectionError(
                "scope definitions and evidence aliases do not match"
            )
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "evidence", evidence)


def build_eccentric_plate_scopes(
    snapshot: _Snapshot,
    *,
    tolerance: float = 1.0e-8,
) -> PlateScopeSet:
    """Prove and return fixed/load/hole/domain scopes for one accepted plate."""

    if (
        snapshot.source_kind != "native"
        or not snapshot.model_current
        or snapshot.artifact is None
    ):
        raise ScopeSelectionError(
            "A4 scopes require a current generated native model"
        )
    if (
        not math.isfinite(float(tolerance))
        or isinstance(tolerance, bool)
        or tolerance <= 0.0
    ):
        raise ValueError("scope tolerance must be a positive finite number")
    part = _active_part(snapshot)
    recipe = getattr(part, "geometry_recipe", None)
    scope_geometry = _plate_scope_geometry(recipe)
    topology = describe_recipe_topology(recipe)
    if not topology.exact:
        raise ScopeSelectionError(
            "scope selection requires exact recipe topology lineage"
        )
    descriptors = describe_native_regions(recipe)
    required_builtins = {"LEFT", "RIGHT", "DOMAIN"}
    if {
        item.name for item in descriptors if item.name in required_builtins
    } != required_builtins:
        raise ScopeSelectionError(
            "plate built-in regions are incomplete or ambiguous"
        )

    model = getattr(snapshot.artifact, "model", None)
    mesh = getattr(model, "mesh", None)
    if mesh is None:
        raise ScopeSelectionError("current artifact has no mesh")
    part_id = str(getattr(part, "id"))
    allocator = NameAllocator(
        {"regions": tuple(snapshot.named_regions)}
    )
    names = {
        "fixed": allocator.allocate("regions", "边", "固定端"),
        "load": allocator.allocate("regions", "边", "加载端"),
        "hole": allocator.allocate("regions", "边", "孔边"),
        "domain": allocator.allocate("regions", "域", "板体"),
    }

    outer_ref = _part_logical_ref(
        part_id,
        recipe,
        "edge:outer-loop",
        allowed_kind="edge",
    )
    hole_refs = tuple(
        _part_logical_ref(
            part_id,
            recipe,
            f"edge:{curve_id}",
            allowed_kind="edge",
        )
        for curve_id, _center_x, _center_y, _radius
        in scope_geometry.hole_curves
    )
    domain_ref = _part_logical_ref(
        part_id,
        recipe,
        "face:domain",
        allowed_kind="face",
    )
    outer = mesh_references_for_logical_entities(
        model,
        (outer_ref,),
        mesh_kind="edge",
    )
    hole = mesh_references_for_logical_entities(
        model,
        hole_refs,
        mesh_kind="edge",
    )
    domain = mesh_references_for_logical_entities(
        model,
        (domain_ref,),
        mesh_kind="element",
    )

    node_by_id = {int(node.id): node for node in mesh.nodes}
    selection_tolerance = tolerance * max(
        1.0,
        abs(scope_geometry.right_x - scope_geometry.left_x),
        scope_geometry.height,
    )
    expected_left = tuple(
        reference
        for reference in outer
        if _reference_on_x(
            reference,
            node_by_id,
            scope_geometry.left_x,
            selection_tolerance,
        )
    )
    expected_right = tuple(
        reference
        for reference in outer
        if _reference_on_x(
            reference,
            node_by_id,
            scope_geometry.right_x,
            selection_tolerance,
        )
    )
    actual_left = _coordinate_edge_references(
        model,
        part_id,
        x=scope_geometry.left_x,
        tolerance=selection_tolerance,
    )
    actual_right = _coordinate_edge_references(
        model,
        part_id,
        x=scope_geometry.right_x,
        tolerance=selection_tolerance,
    )
    _require_exact_reference_set(
        "fixed end",
        expected_left,
        actual_left,
    )
    _require_exact_reference_set(
        "load end",
        expected_right,
        actual_right,
    )

    radial = tuple(
        reference
        for reference in _boundary_edge_references(model, part_id)
        if any(
            _reference_on_circle(
                reference,
                node_by_id,
                center_x,
                center_y,
                radius,
                selection_tolerance,
            )
            for _curve_id, center_x, center_y, radius
            in scope_geometry.hole_curves
        )
    )
    _require_exact_reference_set("hole edge", hole, radial)
    actual_domain = _owned_element_references(model, part_id)
    _require_exact_reference_set("plate domain", domain, actual_domain)

    rows = (
        (
            names["fixed"],
            "fixed_end",
            "builtin.LEFT + coordinate rule",
            expected_left,
        ),
        (
            names["load"],
            "load_end",
            "builtin.RIGHT + coordinate rule",
            expected_right,
        ),
        (
            names["hole"],
            "hole_edge",
            "logical sketch circles + radial rule",
            hole,
        ),
        (
            names["domain"],
            "plate_domain",
            "LogicalEntityRef(face:domain) + ownership rule",
            domain,
        ),
    )
    regions: list[NamedRegion] = []
    evidence: list[ScopeSelectionEvidence] = []
    for alias, semantic, source, references in rows:
        if not references:
            raise ScopeSelectionError(
                f"{semantic} scope resolved to an empty selection"
            )
        regions.append(NamedRegion(alias, tuple(references)))
        evidence.append(
            ScopeSelectionEvidence(
                alias=alias,
                semantic=semantic,
                source=source,
                entity_kind=references[0].kind,
                expected_count=len(references),
                matched_count=len(references),
                bounds=_reference_bounds(references, node_by_id, mesh),
            )
        )
    return PlateScopeSet(tuple(regions), tuple(evidence))


def create_scope_definition_change(
    *,
    patch_id: str,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    snapshot: _Snapshot,
    draft_revision: int,
    material_function: str,
    material_properties: Mapping[str, object],
    section_function: str,
    plane_type: str,
    thickness: float,
    tolerance: float = 1.0e-8,
) -> ModelPatch | AgentProposal:
    """Build an automatic patch or a confirmation proposal when results exist."""

    _require_context_matches_snapshot(context, snapshot)
    scopes = build_eccentric_plate_scopes(
        snapshot,
        tolerance=tolerance,
    )
    allocator = NameAllocator(
        {
            "materials": (
                str(material.name) for material in snapshot.materials
            ),
            "sections": (
                str(section.name) for section in snapshot.sections
            ),
        }
    )
    material_name = allocator.allocate(
        "materials",
        "材料",
        material_function,
    )
    section_name = allocator.allocate(
        "sections",
        "截面",
        section_function,
    )
    if plane_type not in {"stress", "strain"}:
        raise ValueError("plane_type must be stress or strain")
    if (
        isinstance(thickness, bool)
        or not isinstance(thickness, (int, float))
        or not math.isfinite(float(thickness))
        or float(thickness) <= 0.0
    ):
        raise ValueError("section thickness must be a positive finite number")
    properties = deepcopy(dict(material_properties))
    material = MaterialDefinition(material_name, properties)
    section = SectionDefinition(
        section_name,
        material_name,
        "solid",
        {
            "plane_type": plane_type,
            "thickness": float(thickness),
        },
    )
    domain_name = next(
        item.name
        for item in scopes.regions
        if item.name.startswith("域-板体")
    )
    definitions = ModelDefinitions(
        tuple(snapshot.materials) + (material,),
        tuple(snapshot.sections) + (section,),
        tuple(snapshot.assignments)
        + (RegionAssignment(section_name, domain_name),),
        tuple(snapshot.steps),
    )
    regions = tuple(snapshot.named_regions.values()) + scopes.regions
    operations = _state_operations(regions, definitions)
    evidence = [item.to_dict() for item in scopes.evidence]
    summary = {
        "title": "Agent 已创建作用域和材料定义",
        "summary": (
            f"创建 4 个语义作用域、材料 {material_name}、"
            f"截面 {section_name} 及板体分配"
        ),
        "objects": [
            *(item.name for item in scopes.regions),
            material_name,
            section_name,
        ],
        "scope_evidence": evidence,
        "undo_label": "撤销本次 Agent 修改",
    }
    common = {
        "agent_session_id": agent_session_id,
        "turn_id": turn_id,
        "source_tool_call_ids": tuple(source_tool_call_ids),
        "target_document_id": context.binding.document_id,
        "target_session_id": context.binding.session_id,
        "base_session_revision": context.binding.session_revision,
        "draft_revision": draft_revision,
        "operations": operations,
        "preconditions": {
            "source_kind": "native",
            "model_current": True,
            "exact_plate_topology": True,
            "no_destructive_name_overwrite": True,
        },
        "expected_changes": {
            "named_regions_added": len(scopes.regions),
            "materials_added": 1,
            "sections_added": 1,
            "assignments_added": 1,
        },
        "invalidation_impact": {
            "model": True,
            "validation": True,
            "results": _has_accepted_result(snapshot),
        },
        "display_summary": summary,
    }
    if _has_accepted_result(snapshot):
        return AgentProposal.create(
            proposal_id=proposal_id,
            proposal_kind=ProposalKind.DESTRUCTIVE_EDIT,
            **{
                **common,
                "display_summary": {
                    **summary,
                    "title": "定义修改将使已有结果失效",
                    "impact": "已有验证、作业和结果将失效",
                    "confirm_label": "确认修改",
                },
            },
        )
    return ModelPatch.create(patch_id=patch_id, **common)


def scoped_definition_batch_from_operations(
    operations: Sequence[ModelOperation],
    snapshot: _Snapshot,
    *,
    base_session_revision: int,
) -> ScopedDefinitionBatch:
    """Decode the closed A4 operation pair into one application command."""

    values = tuple(operations)
    if tuple(item.kind for item in values) != (
        OperationKind.UPSERT_NAMED_REGIONS,
        OperationKind.UPSERT_MODEL_DEFINITIONS,
    ):
        raise ValueError(
            "A4 changes require exact region and definition post-state operations"
        )
    regions = _decode_regions(values[0].parameters["regions"])
    definitions = _decode_definitions(
        values[1].parameters["definitions"],
        fallback_steps=tuple(snapshot.steps),
    )
    return ScopedDefinitionBatch(
        base_session_revision,
        regions,
        definitions.materials,
        definitions.sections,
        definitions.assignments,
        definitions.steps,
    )


def inverse_operations_for_snapshot(
    snapshot: _Snapshot,
) -> tuple[ModelOperation, ModelOperation]:
    """Return the exact A4 pre-state as closed inverse operations."""

    return _state_operations(
        tuple(snapshot.named_regions.values()),
        ModelDefinitions(
            tuple(snapshot.materials),
            tuple(snapshot.sections),
            tuple(snapshot.assignments),
            tuple(snapshot.steps),
        ),
    )


def require_non_destructive_a4_batch(
    snapshot: _Snapshot,
    batch: ScopedDefinitionBatch,
) -> None:
    """Reject an automatic batch that changes or removes accepted objects."""

    current_regions = dict(snapshot.named_regions)
    next_regions = {region.name: region for region in batch.regions}
    current_materials = {
        material.name: material for material in snapshot.materials
    }
    next_materials = {
        material.name: material for material in batch.materials
    }
    current_sections = {
        section.name: section for section in snapshot.sections
    }
    next_sections = {
        section.name: section for section in batch.sections
    }
    for label, current, after in (
        ("named region", current_regions, next_regions),
        ("material", current_materials, next_materials),
        ("section", current_sections, next_sections),
    ):
        for name, value in current.items():
            if name not in after or after[name] != value:
                raise ValueError(
                    f"automatic A4 patch cannot overwrite or remove {label} {name!r}"
                )
    if any(
        assignment not in batch.assignments
        for assignment in snapshot.assignments
    ):
        raise ValueError(
            "automatic A4 patch cannot overwrite or remove assignments"
        )
    if tuple(batch.steps) != tuple(snapshot.steps):
        raise ValueError("automatic A4 patch cannot edit analysis steps")


def _active_part(snapshot: _Snapshot) -> object:
    matches = tuple(
        part
        for part in snapshot.parts
        if str(getattr(part, "id", "")) == snapshot.active_part_id
    )
    if len(matches) != 1:
        raise ScopeSelectionError(
            "active native Part is missing or ambiguous"
        )
    return matches[0]


def _plate_scope_geometry(recipe: object) -> _PlateScopeGeometry:
    if type(recipe) is PlateWithHoleGeometry:
        return _PlateScopeGeometry(
            0.0,
            float(recipe.width),
            float(recipe.height),
            (
                (
                    "hole-loop",
                    float(recipe.hole_x),
                    float(recipe.hole_y),
                    float(recipe.hole_radius),
                ),
            ),
        )
    if type(recipe) is not SketchGeometry or not recipe.is_strict:
        raise ScopeSelectionError(
            "plate scopes require one rectangular planar profile with circular holes"
        )
    assert recipe.plane is not None
    if recipe.plane != recipe.plane.xy():
        raise ScopeSelectionError(
            "automatic plate scopes currently require the global XY sketch plane"
        )
    analysis = analyze_sketch_profiles(recipe)
    if analysis.blocking_diagnostics:
        raise ScopeSelectionError(
            "plate scopes require a valid closed planar sketch"
        )
    material_profiles = tuple(
        profile
        for profile in analysis.profiles
        if profile.is_material and profile.nesting_depth == 0
    )
    hole_profiles = tuple(
        profile for profile in analysis.profiles if profile.is_hole
    )
    if len(material_profiles) != 1 or not hole_profiles:
        raise ScopeSelectionError(
            "plate scopes require one outer profile and at least one hole"
        )
    material_curves = tuple(
        recipe.curve(curve_id.lstrip("-"))
        for curve_id in material_profiles[0].curve_ids
    )
    if len(material_curves) != 4 or any(
        not isinstance(curve, SketchLine) for curve in material_curves
    ):
        raise ScopeSelectionError(
            "automatic plate scopes require a rectangular outer profile"
        )
    material_points = tuple(
        recipe.point(point_id)
        for point_id in {
            point_id
            for curve in material_curves
            for point_id in (curve.start_point_id, curve.end_point_id)
        }
    )
    x_values = {round(point.u, 12) for point in material_points}
    y_values = {round(point.v, 12) for point in material_points}
    if len(material_points) != 4 or len(x_values) != 2 or len(y_values) != 2:
        raise ScopeSelectionError(
            "automatic plate scopes require an axis-aligned rectangle"
        )
    hole_curves: list[tuple[str, float, float, float]] = []
    for profile in hole_profiles:
        curve_ids = tuple(
            curve_id.lstrip("-") for curve_id in profile.curve_ids
        )
        if len(curve_ids) != 1:
            raise ScopeSelectionError(
                "automatic plate scopes require circular hole profiles"
            )
        circle = recipe.curve(curve_ids[0])
        if not isinstance(circle, SketchCircle) or circle.center_point_id is None:
            raise ScopeSelectionError(
                "automatic plate scopes require circular hole profiles"
            )
        center = recipe.point(circle.center_point_id)
        hole_curves.append(
            (
                circle.id,
                float(center.u),
                float(center.v),
                float(circle.radius),
            )
        )
    min_x = min(float(point.u) for point in material_points)
    max_x = max(float(point.u) for point in material_points)
    min_y = min(float(point.v) for point in material_points)
    max_y = max(float(point.v) for point in material_points)
    return _PlateScopeGeometry(
        min_x,
        max_x,
        max_y - min_y,
        tuple(hole_curves),
    )


def _part_logical_ref(
    part_id: str,
    recipe: object,
    logical_id: str,
    *,
    allowed_kind: str,
) -> LogicalEntityRef:
    reference = LogicalEntityRef(logical_id)
    validate_logical_reference(
        recipe,
        reference,
        allowed_kinds=(allowed_kind,),
        require_exact=True,
    )
    return LogicalEntityRef(
        namespace_part_logical_id(part_id, logical_id)
    )


def _reference_on_x(
    reference: MeshEntityRef,
    node_by_id: Mapping[int, object],
    x: float,
    tolerance: float,
) -> bool:
    return bool(reference.node_ids) and all(
        abs(float(getattr(node_by_id[node_id], "x")) - x) <= tolerance
        for node_id in reference.node_ids
    )


def _reference_on_circle(
    reference: MeshEntityRef,
    node_by_id: Mapping[int, object],
    center_x: float,
    center_y: float,
    radius: float,
    tolerance: float,
) -> bool:
    return bool(reference.node_ids) and all(
        abs(
            math.hypot(
                float(getattr(node_by_id[node_id], "x")) - center_x,
                float(getattr(node_by_id[node_id], "y")) - center_y,
            )
            - radius
        )
        <= tolerance
        for node_id in reference.node_ids
    )


def _boundary_edge_references(
    model: object,
    part_id: str,
) -> tuple[MeshEntityRef, ...]:
    mesh = getattr(model, "mesh")
    owned_elements = _part_element_ids(model, part_id)
    return tuple(
        sorted(
            (
                MeshEntityRef.edge(
                    int(element_id),
                    int(local_index),
                    tuple(int(node_id) for node_id in node_ids),
                    part_id=part_id,
                )
                for element_id, local_index, node_ids
                in mesh_edges.boundary(mesh)
                if int(element_id) in owned_elements
            ),
            key=lambda item: (item.identity, item.node_ids),
        )
    )


def _coordinate_edge_references(
    model: object,
    part_id: str,
    *,
    x: float,
    tolerance: float,
) -> tuple[MeshEntityRef, ...]:
    mesh = getattr(model, "mesh")
    owned_elements = _part_element_ids(model, part_id)
    return tuple(
        sorted(
            (
                MeshEntityRef.edge(
                    int(element_id),
                    int(local_index),
                    tuple(int(node_id) for node_id in node_ids),
                    part_id=part_id,
                )
                for element_id, local_index, node_ids
                in mesh_edges.by_x(
                    mesh,
                    x,
                    tol=tolerance,
                    boundary_only=True,
                )
                if int(element_id) in owned_elements
            ),
            key=lambda item: (item.identity, item.node_ids),
        )
    )


def _owned_element_references(
    model: object,
    part_id: str,
) -> tuple[MeshEntityRef, ...]:
    ids = _part_element_ids(model, part_id)
    return tuple(
        MeshEntityRef.element(element_id, part_id=part_id)
        for element_id in sorted(ids)
    )


def _part_element_ids(
    model: object,
    part_id: str,
) -> frozenset[int]:
    metadata = getattr(model, "metadata", None)
    ownership = (
        metadata.get(NATIVE_PART_OWNERSHIP_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(ownership, Mapping):
        raise ScopeSelectionError(
            "current model lacks stable native Part ownership"
        )
    row = ownership.get(part_id)
    if not isinstance(row, Mapping):
        raise ScopeSelectionError(
            "active Part is absent from mesh ownership"
        )
    ids = frozenset(int(value) for value in row.get("element_ids", ()))
    if not ids:
        raise ScopeSelectionError("active Part owns no mesh elements")
    return ids


def _require_exact_reference_set(
    semantic: str,
    expected: Sequence[MeshEntityRef],
    actual: Sequence[MeshEntityRef],
) -> None:
    if not expected:
        raise ScopeSelectionError(
            f"{semantic} expected selection is empty"
        )
    if tuple(expected) != tuple(actual):
        raise ScopeSelectionError(
            f"{semantic} selection count or identities are abnormal"
        )


def _reference_bounds(
    references: Sequence[MeshEntityRef],
    node_by_id: Mapping[int, object],
    mesh: object,
) -> tuple[float, float, float, float]:
    node_ids = {
        int(node_id)
        for reference in references
        for node_id in reference.node_ids
    }
    if not node_ids and references[0].kind == "element":
        element_by_id = {
            int(element.id): element for element in getattr(mesh, "elements")
        }
        node_ids = {
            int(node_id)
            for reference in references
            for node_id in element_by_id[int(reference.element_id)].node_ids
        }
    if not node_ids:
        raise ScopeSelectionError("scope has no bounded mesh coordinates")
    xs = [float(getattr(node_by_id[node_id], "x")) for node_id in node_ids]
    ys = [float(getattr(node_by_id[node_id], "y")) for node_id in node_ids]
    return min(xs), min(ys), max(xs), max(ys)


def _state_operations(
    regions: Sequence[NamedRegion],
    definitions: ModelDefinitions,
) -> tuple[ModelOperation, ModelOperation]:
    return (
        ModelOperation(
            OperationKind.UPSERT_NAMED_REGIONS,
            {"regions": _encode_regions(regions)},
        ),
        ModelOperation(
            OperationKind.UPSERT_MODEL_DEFINITIONS,
            {"definitions": _encode_definitions(definitions)},
        ),
    )


definition_state_operations = _state_operations


def _encode_regions(
    regions: Sequence[NamedRegion],
) -> list[dict[str, object]]:
    return [
        {
            "name": region.name,
            "references": [
                _encode_reference(reference)
                for reference in region.references
            ],
        }
        for region in regions
    ]


def _encode_reference(
    reference: MeshEntityRef | LogicalEntityRef,
) -> dict[str, object]:
    if type(reference) is LogicalEntityRef:
        return {
            "reference_type": "logical",
            "logical_id": reference.logical_id,
        }
    if type(reference) is not MeshEntityRef:
        raise TypeError("unsupported scope reference")
    return {
        "reference_type": "mesh",
        "kind": reference.kind,
        "node_id": reference.node_id,
        "element_id": reference.element_id,
        "local_index": reference.local_index,
        "node_ids": list(reference.node_ids),
        "part_id": reference.part_id,
    }


def _decode_regions(value: object) -> tuple[NamedRegion, ...]:
    rows = _require_list(value, "regions")
    regions: list[NamedRegion] = []
    for row in rows:
        data = _require_mapping(row, "region")
        _require_keys(data, {"name", "references"}, "region")
        references = tuple(
            _decode_reference(item)
            for item in _require_list(
                data["references"],
                "region references",
            )
        )
        regions.append(NamedRegion(str(data["name"]), references))
    return tuple(regions)


def _decode_reference(
    value: object,
) -> MeshEntityRef | LogicalEntityRef:
    data = _require_mapping(value, "scope reference")
    reference_type = data.get("reference_type")
    if reference_type == "logical":
        _require_keys(
            data,
            {"reference_type", "logical_id"},
            "logical reference",
        )
        return LogicalEntityRef(str(data["logical_id"]))
    if reference_type != "mesh":
        raise ValueError("scope reference type is unknown")
    _require_keys(
        data,
        {
            "reference_type",
            "kind",
            "node_id",
            "element_id",
            "local_index",
            "node_ids",
            "part_id",
        },
        "mesh reference",
    )
    return MeshEntityRef(
        kind=str(data["kind"]),
        node_id=data["node_id"],
        element_id=data["element_id"],
        local_index=data["local_index"],
        node_ids=tuple(_require_list(data["node_ids"], "mesh node_ids")),
        part_id=(
            None
            if data["part_id"] is None
            else str(data["part_id"])
        ),
    )


def _encode_definitions(
    definitions: ModelDefinitions,
) -> dict[str, object]:
    return {
        "materials": [
            {
                "name": material.name,
                "properties": deepcopy(dict(material.properties)),
            }
            for material in definitions.materials
        ],
        "sections": [
            {
                "name": section.name,
                "material": section.material,
                "section_type": section.section_type,
                "properties": deepcopy(dict(section.properties)),
            }
            for section in definitions.sections
        ],
        "assignments": [
            _encode_assignment(assignment)
            for assignment in definitions.assignments
        ],
        "steps": [_encode_step(step) for step in definitions.steps],
    }


def _encode_assignment(assignment: RegionAssignment) -> dict[str, object]:
    encoded: dict[str, object] = {
        "section_name": assignment.section_name,
        "region_name": assignment.region_name,
    }
    if assignment.beam_orientation is not None:
        encoded["beam_orientation"] = list(
            assignment.beam_orientation.local_y_reference
        )
    return encoded


def _decode_definitions(
    value: object,
    *,
    fallback_steps: Sequence[object] = (),
) -> ModelDefinitions:
    data = _require_mapping(value, "definitions")
    fields = frozenset(data)
    legacy = {"materials", "sections", "assignments"}
    current = {*legacy, "steps"}
    if fields not in {frozenset(legacy), frozenset(current)}:
        raise ValueError(
            "definitions fields do not match the strict A4/A5 schema"
        )
    materials = []
    for row in _require_list(data["materials"], "materials"):
        item = _require_mapping(row, "material")
        _require_keys(item, {"name", "properties"}, "material")
        materials.append(
            MaterialDefinition(
                str(item["name"]),
                deepcopy(dict(_require_mapping(
                    item["properties"],
                    "material properties",
                ))),
            )
        )
    sections = []
    for row in _require_list(data["sections"], "sections"):
        item = _require_mapping(row, "section")
        _require_keys(
            item,
            {
                "name",
                "material",
                "section_type",
                "properties",
            },
            "section",
        )
        sections.append(
            SectionDefinition(
                str(item["name"]),
                str(item["material"]),
                str(item["section_type"]),
                deepcopy(dict(_require_mapping(
                    item["properties"],
                    "section properties",
                ))),
            )
        )
    assignments = []
    for row in _require_list(data["assignments"], "assignments"):
        item = _require_mapping(row, "assignment")
        fields = set(item)
        if fields not in (
            {"section_name", "region_name"},
            {"section_name", "region_name", "beam_orientation"},
        ):
            raise ValueError(
                "assignment fields do not match the strict A4/A5 schema"
            )
        has_orientation = "beam_orientation" in item
        orientation = item.get("beam_orientation")
        assignments.append(
            RegionAssignment(
                str(item["section_name"]),
                str(item["region_name"]),
                (
                    None
                    if not has_orientation
                    else BeamOrientation(
                        _strict_number_array(
                            orientation,
                            "beam orientation",
                        )
                    )
                ),
            )
        )
    return ModelDefinitions(
        tuple(materials),
        tuple(sections),
        tuple(assignments),
        (
            tuple(deepcopy(tuple(fallback_steps)))
            if "steps" not in data
            else tuple(
                _decode_step(row)
                for row in _require_list(data["steps"], "steps")
            )
        ),
    )


def _encode_step(step: object) -> dict[str, object]:
    if type(step) is not AnalysisStep:
        raise TypeError("analysis step must be exactly AnalysisStep")
    return {
        "name": step.name,
        "procedure": step.procedure,
        "metadata": deepcopy(dict(step.metadata)),
        "boundaries": [
            {
                "name": item.name,
                "target": item.target,
                "target_kind": item.target_kind,
                "first_component": item.first_component,
                "last_component": item.last_component,
                "value": item.value,
            }
            for item in step.boundaries
        ],
        "cloads": [
            {
                "name": item.name,
                "target": item.target,
                "component": item.component,
                "value": item.value,
            }
            for item in step.cloads
        ],
        "edge_loads": [
            {
                "name": item.name,
                "edge": item.edge,
                "vector": list(item.vector),
                "magnitude": item.magnitude,
                "load_type": item.load_type,
            }
            for item in step.edge_loads
        ],
        "surface_loads": [
            {
                "name": item.name,
                "surface": item.surface,
                "vector": list(item.vector),
                "magnitude": item.magnitude,
                "load_type": item.load_type,
            }
            for item in step.surface_loads
        ],
        "line_loads": [
            {
                "name": item.name,
                "target": item.target,
                "vector": list(item.vector),
                "coordinate_system": item.coordinate_system,
            }
            for item in step.line_loads
        ],
        "body_loads": [
            {
                "name": item.name,
                "target": item.target,
                "vector": list(item.vector),
            }
            for item in step.body_loads
        ],
        "gravity_loads": [
            {
                "name": item.name,
                "target": item.target,
                "acceleration": list(item.acceleration),
            }
            for item in step.gravity_loads
        ],
        "outputs": [
            {
                "name": item.name,
                "kind": item.kind,
                "target": item.target,
                "variables": list(item.variables),
                "metadata": deepcopy(dict(item.metadata)),
            }
            for item in step.outputs
        ],
    }


def _decode_step(value: object) -> AnalysisStep:
    data = _require_mapping(value, "analysis step")
    _require_keys(
        data,
        {
            "name",
            "procedure",
            "metadata",
            "boundaries",
            "cloads",
            "edge_loads",
            "surface_loads",
            "line_loads",
            "body_loads",
            "gravity_loads",
            "outputs",
        },
        "analysis step",
    )
    return AnalysisStep(
        name=_strict_string(data["name"], "analysis step name"),
        procedure=_strict_string(data["procedure"], "analysis procedure"),
        metadata=deepcopy(
            dict(_require_mapping(data["metadata"], "step metadata"))
        ),
        boundaries=tuple(
            _decode_boundary(item)
            for item in _require_list(data["boundaries"], "boundaries")
        ),
        cloads=tuple(
            _decode_cload(item)
            for item in _require_list(data["cloads"], "cloads")
        ),
        edge_loads=tuple(
            _decode_edge_load(item)
            for item in _require_list(data["edge_loads"], "edge_loads")
        ),
        surface_loads=tuple(
            _decode_surface_load(item)
            for item in _require_list(
                data["surface_loads"],
                "surface_loads",
            )
        ),
        line_loads=tuple(
            _decode_line_load(item)
            for item in _require_list(data["line_loads"], "line_loads")
        ),
        body_loads=tuple(
            _decode_body_load(item)
            for item in _require_list(data["body_loads"], "body_loads")
        ),
        gravity_loads=tuple(
            _decode_gravity_load(item)
            for item in _require_list(
                data["gravity_loads"],
                "gravity_loads",
            )
        ),
        outputs=tuple(
            _decode_output(item)
            for item in _require_list(data["outputs"], "outputs")
        ),
    )


def _decode_boundary(value: object) -> DisplacementConstraint:
    data = _require_mapping(value, "boundary")
    _require_keys(
        data,
        {
            "name",
            "target",
            "target_kind",
            "first_component",
            "last_component",
            "value",
        },
        "boundary",
    )
    return DisplacementConstraint(
        _strict_target(data["target"], "boundary target"),
        _strict_integer(data["first_component"], "first_component"),
        _strict_integer(data["last_component"], "last_component"),
        _strict_number(data["value"], "boundary value"),
        _strict_string(data["target_kind"], "boundary target_kind"),
        _optional_name(data["name"]),
    )


def _decode_cload(value: object) -> NodalLoad:
    data = _require_mapping(value, "nodal load")
    _require_keys(
        data,
        {"name", "target", "component", "value"},
        "nodal load",
    )
    return NodalLoad(
        _strict_target(data["target"], "nodal load target"),
        _strict_integer(data["component"], "nodal load component"),
        _strict_number(data["value"], "nodal load value"),
        _optional_name(data["name"]),
    )


def _decode_edge_load(value: object) -> EdgeLoad:
    data = _require_mapping(value, "edge load")
    _require_keys(
        data,
        {"name", "edge", "vector", "magnitude", "load_type"},
        "edge load",
    )
    return EdgeLoad(
        _strict_string(data["edge"], "edge load target"),
        _strict_number_array(data["vector"], "edge load vector"),
        (
            None
            if data["magnitude"] is None
            else _strict_number(data["magnitude"], "edge load magnitude")
        ),
        _strict_string(data["load_type"], "edge load type"),
        _optional_name(data["name"]),
    )


def _decode_surface_load(value: object) -> SurfaceLoad:
    data = _require_mapping(value, "surface load")
    _require_keys(
        data,
        {"name", "surface", "vector", "magnitude", "load_type"},
        "surface load",
    )
    return SurfaceLoad(
        _strict_string(data["surface"], "surface load target"),
        _strict_number_array(data["vector"], "surface load vector"),
        (
            None
            if data["magnitude"] is None
            else _strict_number(
                data["magnitude"],
                "surface load magnitude",
            )
        ),
        _strict_string(data["load_type"], "surface load type"),
        _optional_name(data["name"]),
    )


def _decode_line_load(value: object) -> LineLoad:
    data = _require_mapping(value, "line load")
    _require_keys(
        data,
        {"name", "target", "vector", "coordinate_system"},
        "line load",
    )
    return LineLoad(
        _strict_target(data["target"], "line load target"),
        _strict_number_array(data["vector"], "line load vector"),
        _strict_string(data["coordinate_system"], "coordinate system"),
        _optional_name(data["name"]),
    )


def _decode_body_load(value: object) -> BodyForce:
    data = _require_mapping(value, "body load")
    _require_keys(data, {"name", "target", "vector"}, "body load")
    return BodyForce(
        _strict_target(data["target"], "body load target"),
        _strict_number_array(data["vector"], "body load vector"),
        _optional_name(data["name"]),
    )


def _decode_gravity_load(value: object) -> GravityLoad:
    data = _require_mapping(value, "gravity load")
    _require_keys(
        data,
        {"name", "target", "acceleration"},
        "gravity load",
    )
    return GravityLoad(
        _strict_number_array(data["acceleration"], "gravity acceleration"),
        (
            None
            if data["target"] is None
            else _strict_target(data["target"], "gravity target")
        ),
        _optional_name(data["name"]),
    )


def _decode_output(value: object) -> OutputRequest:
    data = _require_mapping(value, "output request")
    _require_keys(
        data,
        {"name", "kind", "target", "variables", "metadata"},
        "output request",
    )
    return OutputRequest(
        _strict_string(data["kind"], "output kind"),
        _strict_string(data["target"], "output target"),
        tuple(
            _strict_string(item, "output variable")
            for item in _require_list(data["variables"], "variables")
        ),
        deepcopy(dict(_require_mapping(data["metadata"], "output metadata"))),
        None,
        _optional_name(data["name"]),
    )


def _optional_name(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError("analysis object name must be null or nonblank string")
    return value


def _strict_string(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{label} must be a nonblank exact string")
    return value


def _strict_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def _strict_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or type(value) not in {int, float}
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{label} must be a finite JSON number")
    return float(value)


def _strict_target(value: object, label: str) -> str | int:
    if type(value) is int:
        return value
    return _strict_string(value, label)


def _strict_number_array(value: object, label: str) -> tuple[float, ...]:
    return tuple(
        _strict_number(item, f"{label}[{index}]")
        for index, item in enumerate(_require_list(value, label))
    )


def _require_context_matches_snapshot(
    context: AuthoringContext,
    snapshot: _Snapshot,
) -> None:
    # Multi-document workspaces bind stable integer document identities such
    # as "2"; only session identity and revision prove freshness, so the
    # document_id format is deliberately not compared here.
    if (
        context.binding.session_id != snapshot.session_id
        or context.binding.session_revision != snapshot.session_revision
        or context.binding.source_kind != "native"
    ):
        raise ValueError("authoring context is stale")


def _has_accepted_result(snapshot: _Snapshot) -> bool:
    return any(bool(getattr(run, "has_result", False)) for run in snapshot.runs)


def _require_mapping(
    value: object,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


def _require_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields do not match the strict A4/A5 schema"
        )


__all__ = [
    "PlateScopeSet",
    "ScopeSelectionError",
    "ScopeSelectionEvidence",
    "build_eccentric_plate_scopes",
    "create_scope_definition_change",
    "definition_state_operations",
    "inverse_operations_for_snapshot",
    "require_non_destructive_a4_batch",
    "scoped_definition_batch_from_operations",
]
