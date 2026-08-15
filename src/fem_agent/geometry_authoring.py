"""Typed native geometry drafts, bounded previews, and edit proposals."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import json
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

from fem.geometry import (
    BooleanBodyContext,
    BooleanGeometry,
    BooleanLineageEntity,
    BooleanLineageMapping,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    FaceSketchBooleanGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    PlateWithHoleGeometry,
    PartBooleanContext,
    PathSweptGeometry,
    PlanarBooleanContext,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SketchArc,
    SketchAngleDimension,
    SketchCircle,
    SketchCoincidentConstraint,
    SketchConcentricConstraint,
    SketchConstraint,
    SketchDistanceDimension,
    SketchEqualLengthConstraint,
    SketchEqualRadiusConstraint,
    SketchFixedConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchParallelConstraint,
    SketchPlane,
    SketchPoint,
    SketchPointOnCurveConstraint,
    SketchPerpendicularConstraint,
    SketchRadiusDimension,
    SketchRectangle,
    SketchTangentConstraint,
    SketchVerticalConstraint,
    SolidBody,
    WireGeometry,
    WireMember,
    WirePoint,
    geometry_dimension,
    analyze_sketch_profiles,
    legacy_sketch_to_strict,
    planar_geometry_normal,
    sketch_constraint_entity_ids,
    solve_sketch_constraints,
)
from fem.application.feature_history import derive_feature_history
from fem.application.planar_construction import CompiledPlanarConstruction
from fem.geometry.construction_ir import PlanarConstructionIR
from fem.geometry.recipe_topology import (
    describe_recipe_topology,
    topology_fingerprint_for_recipe,
)

from .authoring import (
    AgentProposal,
    AuthoringContext,
    AuthoringContractError,
    ModelOperation,
    OperationKind,
    ProposalKind,
    UnitContextSummary,
)
from .naming import NameAllocator


_PREVIEW_SEGMENTS = 24
_MAX_PREVIEW_POINTS = 128
_RECIPE_SCHEMA_VERSION = 1
_MAX_RECIPE_BYTES = 65536
_MAX_RECIPE_NODES = 128
_MAX_BOOLEAN_RECIPE_PAYLOAD_NODES = 512
_MAX_RECIPE_DEPTH = 16
_SKETCH_AUTHORING_TOLERANCE = 1.0e-9
GEOMETRY_FEATURE_CATALOG_TOOL_NAME = "read_geometry_feature_catalog"
PROFILE_TRANSFORM_CONTEXT_SCHEMA_VERSION = 1
PROFILE_TRANSFORM_MAX_PROFILES = 32
PROFILE_TRANSFORM_MAX_BYTES = 24_576


@dataclass(frozen=True, slots=True)
class GeometryContractProof:
    """Bounded provider-safe proof derived without exposing CAD state."""

    exact: bool
    topology_fingerprint: Mapping[str, object]
    expected_body_count: int
    diagnostics: tuple[Mapping[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "local_recipe_topology_proof",
            "exact": self.exact,
            "expected_body_count": self.expected_body_count,
            "body_count_proven": self.exact,
            "topology_fingerprint": dict(self.topology_fingerprint),
            "diagnostics": [dict(item) for item in self.diagnostics],
        }


class PlanarEditValidationError(AuthoringContractError):
    """An incremental planar edit failed exact Profile validation."""

    def __init__(self, diagnostics: Sequence[Mapping[str, object]]) -> None:
        records = tuple(dict(item) for item in diagnostics[:16])
        if not records:
            records = (
                {
                    "code": "sketch.topology-unproven",
                    "message": "Planar Profile topology could not be proven",
                    "affected_logical_ids": [],
                    "severity": "error",
                },
            )
        self.diagnostics = records
        codes = tuple(
            dict.fromkeys(
                str(item.get("code", "sketch.topology-unproven"))
                for item in records
            )
        )
        affected = tuple(
            dict.fromkeys(
                str(entity_id)
                for item in records
                for entity_id in item.get("affected_logical_ids", [])
            )
        )
        detail = ", ".join(codes)
        if affected:
            detail += f"; affected={','.join(affected[:16])}"
        super().__init__(f"planar-edit.topology-invalid: {detail}")

    def to_provider_dict(self) -> dict[str, object]:
        return {
            "kind": "planar_edit_validation",
            "status": "rejected",
            "exact": False,
            "diagnostics": [dict(item) for item in self.diagnostics],
        }


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite real number")
    return result


@dataclass(frozen=True, slots=True)
class StaticGeometryPreview:
    """Detached line preview with a fixed point budget."""

    dimension: int
    points: tuple[tuple[float, float, float], ...]
    lines: tuple[tuple[int, ...], ...]
    bounds: tuple[float, float, float, float, float, float]
    point_names: tuple[str, ...] = ()
    member_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dimension not in {1, 2, 3}:
            raise ValueError("preview dimension must be 1, 2, or 3")
        points = tuple(self.points)
        lines = tuple(self.lines)
        point_names = tuple(self.point_names)
        member_names = tuple(self.member_names)
        if not points or len(points) > _MAX_PREVIEW_POINTS:
            raise ValueError("preview point count is outside the A2 bound")
        if any(
            len(point) != 3
            or any(not math.isfinite(float(component)) for component in point)
            for point in points
        ):
            raise ValueError("preview points must be finite XYZ triples")
        if any(
            len(line) < 2 or any(index < 0 or index >= len(points) for index in line)
            for line in lines
        ):
            raise ValueError("preview lines contain invalid point indices")
        if len(self.bounds) != 6 or any(
            not math.isfinite(float(value)) for value in self.bounds
        ):
            raise ValueError("preview bounds must contain six finite values")
        if point_names and len(point_names) != len(points):
            raise ValueError("preview point names must match preview points")
        if member_names and len(member_names) != len(lines):
            raise ValueError("preview member names must match preview lines")
        if any(not name.strip() for name in (*point_names, *member_names)):
            raise ValueError("preview entity names must not be blank")
        object.__setattr__(self, "point_names", point_names)
        object.__setattr__(self, "member_names", member_names)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "kind": "bounded_wireframe",
            "dimension": self.dimension,
            "point_count": len(self.points),
            "line_count": len(self.lines),
            "points": [list(point) for point in self.points],
            "lines": [list(line) for line in self.lines],
            "bounds": list(self.bounds),
        }
        if self.point_names:
            payload["point_names"] = list(self.point_names)
        if self.member_names:
            payload["member_names"] = list(self.member_names)
        return payload


@dataclass(frozen=True, slots=True)
class GeometryDraft:
    recipe: object
    recipe_payload: Mapping[str, object]
    preview: StaticGeometryPreview
    key_dimensions: Mapping[str, float]
    transforms: tuple[Mapping[str, object], ...] = ()
    proof: GeometryContractProof | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, NATIVE_GEOMETRY_TYPES):
            raise TypeError("recipe must be native geometry")
        if (
            type(self.recipe) is SketchGeometry
            and self.recipe.is_strict
            and (
                len(self.recipe.points) > 128
                or len(self.recipe.curves) > 128
                or len(self.recipe.constraints) > 128
            )
        ):
            raise ValueError("planar sketch exceeds the 128-entity bound")
        payload = geometry_recipe_to_payload(self.recipe)
        if dict(self.recipe_payload) != payload:
            raise ValueError("recipe_payload does not match recipe")
        dimensions = {
            str(key): _finite(value, str(key))
            for key, value in self.key_dimensions.items()
        }
        object.__setattr__(self, "recipe_payload", payload)
        object.__setattr__(self, "key_dimensions", dimensions)
        object.__setattr__(
            self,
            "transforms",
            tuple(dict(item) for item in self.transforms),
        )
        proof = geometry_contract_proof(self.recipe)
        if self.proof is not None and self.proof != proof:
            raise ValueError("proof does not match recipe")
        object.__setattr__(self, "proof", proof)


def rectangle_geometry(
    name: str,
    *,
    width: Real,
    height: Real,
) -> GeometryDraft:
    recipe = RectangleGeometry(
        name,
        _finite(width, "width"),
        _finite(height, "height"),
    )
    return _draft(recipe, {"width": recipe.width, "height": recipe.height})


def disk_geometry(name: str, *, radius: Real) -> GeometryDraft:
    recipe = DiskGeometry(name, _finite(radius, "radius"))
    return _draft(recipe, {"radius": recipe.radius})


def planar_sketch_geometry(
    name: str,
    *,
    contours: Sequence[SketchRectangle | SketchCircle],
) -> GeometryDraft:
    """Build a general strict XY sketch from bounded closed contours."""

    values = tuple(contours)
    if not values:
        raise ValueError("contours must contain at least one closed contour")
    if any(
        type(item) not in {SketchRectangle, SketchCircle}
        or (isinstance(item, SketchCircle) and not item.is_legacy)
        for item in values
    ):
        raise TypeError(
            "contours must contain legacy SketchRectangle or SketchCircle values"
        )
    recipe = legacy_sketch_to_strict(SketchGeometry(name, values))
    return _draft(
        recipe,
        {
            "point_count": float(len(recipe.points)),
            "curve_count": float(len(recipe.curves)),
        },
    )


def planar_polygon_geometry(
    name: str,
    *,
    vertices: Sequence[Sequence[Real]],
) -> GeometryDraft:
    """Build a strict XY sketch containing one polygonal profile."""

    values = tuple(vertices)
    if not 3 <= len(values) <= 64:
        raise ValueError("polygon vertices must contain between 3 and 64 points")
    coordinates: list[tuple[float, float]] = []
    for index, vertex in enumerate(values):
        if isinstance(vertex, (str, bytes, bytearray)):
            raise TypeError("polygon vertices must contain coordinate pairs")
        components = tuple(vertex)
        if len(components) != 2:
            raise ValueError("polygon vertices must contain coordinate pairs")
        coordinates.append(
            (
                _finite(components[0], f"vertices[{index}].x"),
                _finite(components[1], f"vertices[{index}].y"),
            )
        )
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("polygon vertices must be distinct")
    point_ids = tuple(f"P{index + 1}" for index in range(len(coordinates)))
    line_ids = tuple(f"L{index + 1}" for index in range(len(coordinates)))
    points = tuple(
        SketchPoint(point_id, x, y)
        for point_id, (x, y) in zip(point_ids, coordinates, strict=True)
    )
    curves = tuple(
        SketchLine(
            line_ids[index],
            point_ids[index],
            point_ids[(index + 1) % len(point_ids)],
        )
        for index in range(len(point_ids))
    )
    recipe = SketchGeometry(
        name,
        SketchPlane.xy(),
        points,
        curves,
    )
    return _draft(
        recipe,
        {
            "point_count": float(len(points)),
            "curve_count": float(len(curves)),
        },
    )


def planar_path_slot_vertices(
    points: Sequence[Sequence[Real]],
    width: Real,
) -> tuple[tuple[float, float], ...]:
    """Return one closed constant-width slot around an open XY polyline.

    The returned vertices use square end caps and bounded miter joins.  Exact
    sketch topology validation remains authoritative for rejecting a path whose
    offset self-intersects.
    """

    values = tuple(points)
    if not 2 <= len(values) <= 32:
        raise ValueError("slot path must contain between 2 and 32 points")
    coordinates: list[tuple[float, float]] = []
    for index, point in enumerate(values):
        if isinstance(point, (str, bytes, bytearray)):
            raise TypeError("slot path points must contain coordinate pairs")
        components = tuple(point)
        if len(components) != 2:
            raise ValueError("slot path points must contain coordinate pairs")
        coordinates.append(
            (
                _finite(components[0], f"points[{index}].x"),
                _finite(components[1], f"points[{index}].y"),
            )
        )
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("slot path points must be distinct")

    width_value = _finite(width, "width")
    if width_value <= 0.0:
        raise ValueError("slot width must be positive")
    half_width = width_value / 2.0

    tangents: list[tuple[float, float]] = []
    normals: list[tuple[float, float]] = []
    for start, end in zip(coordinates[:-1], coordinates[1:], strict=True):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= _SKETCH_AUTHORING_TOLERANCE:
            raise ValueError("slot path contains a zero-length segment")
        tangent = (dx / length, dy / length)
        tangents.append(tangent)
        normals.append((-tangent[1], tangent[0]))

    start_center = (
        coordinates[0][0] - tangents[0][0] * half_width,
        coordinates[0][1] - tangents[0][1] * half_width,
    )
    end_center = (
        coordinates[-1][0] + tangents[-1][0] * half_width,
        coordinates[-1][1] + tangents[-1][1] * half_width,
    )
    left = [
        (
            start_center[0] + normals[0][0] * half_width,
            start_center[1] + normals[0][1] * half_width,
        )
    ]
    right = [
        (
            start_center[0] - normals[0][0] * half_width,
            start_center[1] - normals[0][1] * half_width,
        )
    ]

    for index, center in enumerate(coordinates[1:-1], start=1):
        previous_normal = normals[index - 1]
        next_normal = normals[index]
        miter_x = previous_normal[0] + next_normal[0]
        miter_y = previous_normal[1] + next_normal[1]
        miter_length = math.hypot(miter_x, miter_y)
        if miter_length <= _SKETCH_AUTHORING_TOLERANCE:
            raise ValueError("slot path cannot reverse direction at one point")
        miter = (miter_x / miter_length, miter_y / miter_length)
        denominator = miter[0] * next_normal[0] + miter[1] * next_normal[1]
        if abs(denominator) <= _SKETCH_AUTHORING_TOLERANCE:
            raise ValueError("slot path contains an unsupported sharp turn")
        scale = half_width / denominator
        if abs(scale) > half_width * 8.0:
            raise ValueError("slot path turn is too sharp for the requested width")
        offset = (miter[0] * scale, miter[1] * scale)
        left.append((center[0] + offset[0], center[1] + offset[1]))
        right.append((center[0] - offset[0], center[1] - offset[1]))

    left.append(
        (
            end_center[0] + normals[-1][0] * half_width,
            end_center[1] + normals[-1][1] * half_width,
        )
    )
    right.append(
        (
            end_center[0] - normals[-1][0] * half_width,
            end_center[1] - normals[-1][1] * half_width,
        )
    )
    vertices = tuple((*left, *reversed(right)))
    if len(set(vertices)) != len(vertices):
        raise ValueError("slot path produced duplicate boundary vertices")
    return vertices


def plate_with_hole_geometry(
    name: str,
    *,
    width: Real,
    height: Real,
    hole_radius: Real,
    hole_center: Sequence[Real] | None = None,
    center_offset: Sequence[Real] | None = None,
) -> GeometryDraft:
    """Build the legacy single-hole compatibility recipe.

    Active Agent tools use general strict planar sketches. This constructor is
    retained only to decode and test projects authored before that migration.
    """

    if (hole_center is None) == (center_offset is None):
        raise ValueError("provide exactly one of hole_center or center_offset")
    normalized_width = _finite(width, "width")
    normalized_height = _finite(height, "height")
    position = hole_center if hole_center is not None else center_offset
    if isinstance(position, (str, bytes, bytearray)):
        raise TypeError("hole position must be a two-component sequence")
    values = tuple(position or ())
    if len(values) != 2:
        raise ValueError("hole position must contain exactly two components")
    first = _finite(values[0], "hole position x")
    second = _finite(values[1], "hole position y")
    if hole_center is None:
        hole_x = normalized_width / 2.0 + first
        hole_y = normalized_height / 2.0 + second
        position_kind = "center_offset"
    else:
        hole_x, hole_y = first, second
        position_kind = "hole_center"
    recipe = PlateWithHoleGeometry(
        name,
        normalized_width,
        normalized_height,
        hole_x,
        hole_y,
        _finite(hole_radius, "hole_radius"),
    )
    draft = _draft(
        recipe,
        {
            "width": recipe.width,
            "height": recipe.height,
            "hole_x": recipe.hole_x,
            "hole_y": recipe.hole_y,
            "hole_radius": recipe.hole_radius,
        },
    )
    return GeometryDraft(
        recipe=draft.recipe,
        recipe_payload=draft.recipe_payload,
        preview=draft.preview,
        key_dimensions=draft.key_dimensions,
        transforms=(
            {
                "kind": position_kind,
                "x": first,
                "y": second,
            },
        ),
    )


def box_geometry(
    name: str,
    *,
    width: Real,
    depth: Real,
    height: Real,
) -> GeometryDraft:
    recipe = BoxGeometry(
        name,
        _finite(width, "width"),
        _finite(depth, "depth"),
        _finite(height, "height"),
    )
    return _draft(
        recipe,
        {
            "width": recipe.width,
            "depth": recipe.depth,
            "height": recipe.height,
        },
    )


def cylinder_geometry(
    name: str,
    *,
    radius: Real,
    height: Real,
) -> GeometryDraft:
    recipe = CylinderGeometry(
        name,
        _finite(radius, "radius"),
        _finite(height, "height"),
    )
    return _draft(
        recipe,
        {"radius": recipe.radius, "height": recipe.height},
    )


def wire_geometry(
    name: str,
    *,
    points: Sequence[WirePoint],
    members: Sequence[WireMember],
) -> GeometryDraft:
    """Build one bounded spatial wire from stable named entities."""

    point_values = tuple(points)
    member_values = tuple(members)
    if not 2 <= len(point_values) <= _MAX_PREVIEW_POINTS:
        raise ValueError("wire points must contain between 2 and 128 items")
    if not 1 <= len(member_values) <= _MAX_PREVIEW_POINTS:
        raise ValueError("wire members must contain between 1 and 128 items")
    recipe = WireGeometry(name, point_values, member_values)
    return _draft(
        recipe,
        {
            "point_count": float(len(recipe.points)),
            "member_count": float(len(recipe.members)),
        },
    )


def translate_geometry(
    draft: GeometryDraft,
    *,
    dx: Real,
    dy: Real,
    dz: Real = 0.0,
) -> GeometryDraft:
    if type(draft) is not GeometryDraft:
        raise TypeError("draft must be GeometryDraft")
    recipe = MovedGeometry(
        draft.recipe,
        _finite(dx, "dx"),
        _finite(dy, "dy"),
        _finite(dz, "dz"),
    )
    return _draft(
        recipe,
        draft.key_dimensions,
        (
            *draft.transforms,
            {"kind": "translation", "dx": recipe.dx, "dy": recipe.dy, "dz": recipe.dz},
        ),
    )


def rotate_geometry(
    draft: GeometryDraft,
    *,
    axis: str,
    angle_degrees: Real,
) -> GeometryDraft:
    if type(draft) is not GeometryDraft:
        raise TypeError("draft must be GeometryDraft")
    recipe = RotatedGeometry(
        draft.recipe,
        axis,  # type: ignore[arg-type]
        _finite(angle_degrees, "angle_degrees"),
    )
    return _draft(
        recipe,
        draft.key_dimensions,
        (
            *draft.transforms,
            {
                "kind": "rotation",
                "axis": recipe.axis,
                "angle_degrees": recipe.angle_degrees,
            },
        ),
    )


def create_geometry_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    draft: GeometryDraft,
    part_function: str,
    unit_context: UnitContextSummary,
    project_function: str | None = None,
    summary: str | None = None,
    local_evidence: Mapping[str, object] | None = None,
    include_static_preview: bool = True,
) -> AgentProposal:
    """Create one revision-bound proposal for blank or native sessions."""

    if type(context) is not AuthoringContext:
        raise TypeError("context must be AuthoringContext")
    if type(draft) is not GeometryDraft:
        raise TypeError("draft must be GeometryDraft")
    if type(unit_context) is not UnitContextSummary:
        raise TypeError("unit_context must be UnitContextSummary")
    binding = context.binding
    if not binding.supported or binding.source_kind not in {"blank", "native"}:
        raise AuthoringContractError(
            "geometry proposal requires blank or native context"
        )
    if context.unit_context is not None and context.unit_context != unit_context:
        raise AuthoringContractError(
            "A2 geometry proposal cannot change existing project units"
        )
    units = unit_context.to_dict()
    part_name = NameAllocator(
        {"parts": (part.name for part in context.parts)}
    ).allocate("parts", "部件", part_function)
    if binding.source_kind == "blank":
        if project_function is None:
            raise AuthoringContractError(
                "blank-session geometry proposal requires project_function"
            )
        project_name = NameAllocator().allocate(
            "models",
            "模型",
            project_function,
        )
        operation = ModelOperation(
            OperationKind.CREATE_NATIVE_PROJECT,
            {
                "project_name": project_name,
                "part_name": part_name,
                "recipe": dict(draft.recipe_payload),
                "unit_context": units,
            },
        )
        operation_label = "加入部件"
        target_model = project_name
    else:
        if context.model_name is None:
            raise AuthoringContractError("native context requires model_name")
        operation = ModelOperation(
            OperationKind.ADD_NATIVE_PART,
            {
                "part_name": part_name,
                "recipe": dict(draft.recipe_payload),
                "unit_context": units,
            },
        )
        operation_label = "加入部件"
        target_model = context.model_name
    proposal_summary = (
        operation_label
        if summary is None
        else str(summary).strip()
    )
    if not proposal_summary:
        raise AuthoringContractError("geometry proposal summary is blank")
    if len(proposal_summary.encode("utf-8")) > 2048:
        raise AuthoringContractError("geometry proposal summary is too long")
    preconditions: dict[str, object] = {
        "source_kind": binding.source_kind,
        "unit_context": units,
    }
    if local_evidence is not None:
        if not isinstance(local_evidence, Mapping):
            raise TypeError("local_evidence must be an object or None")
        preconditions["local_evidence"] = dict(local_evidence)
    display_summary = {
        "title": operation_label,
        "summary": proposal_summary,
        "target_model": target_model,
        "operation": operation.kind.value,
        "part_name": part_name,
        "recipe_type": type(draft.recipe).__name__,
        "source": _proposal_source(draft.recipe),
        "feature_operation": _proposal_operation(draft.recipe),
        "dimension": geometry_dimension(draft.recipe),
        "key_dimensions": dict(draft.key_dimensions),
        "expected_entity_count": draft.proof.expected_body_count,
        "length_unit": unit_context.length,
        "transforms": [dict(item) for item in draft.transforms],
        "expected_new_objects": [part_name],
        "invalidated_objects": [],
        "invalidation_impact": {
            "mesh": False,
            "definitions": False,
            "results": False,
        },
        "base_session_revision": binding.session_revision,
        "proof": draft.proof.to_dict(),
    }
    if include_static_preview:
        display_summary["preview"] = draft.preview.to_dict()
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=binding.document_id,
        target_session_id=binding.session_id,
        base_session_revision=binding.session_revision,
        draft_revision=draft_revision,
        operations=(operation,),
        preconditions=preconditions,
        expected_changes={
            "part_count_delta": 1,
            "project_created": binding.source_kind == "blank",
            "projection_refresh_count": 1,
        },
        invalidation_impact={
            "mesh": False,
            "definitions": False,
            "results": False,
        },
        display_summary=display_summary,
    )


def planar_construction_proposal_evidence(
    construction: PlanarConstructionIR,
    compiled: CompiledPlanarConstruction,
    *,
    output_kind: str = "planar",
    output_recipe: object | None = None,
) -> dict[str, object]:
    """Return bounded local evidence hashed into an IR-derived proposal."""

    if type(construction) is not PlanarConstructionIR:
        raise TypeError("construction must be PlanarConstructionIR")
    if type(compiled) is not CompiledPlanarConstruction:
        raise TypeError("compiled must be CompiledPlanarConstruction")
    if compiled.construction_digest != construction.digest():
        raise ValueError("compiled construction digest does not match the IR")
    if output_kind not in {"planar", "extrusion", "revolution", "path_sweep"}:
        raise ValueError("unsupported planar construction output kind")
    if output_recipe is None:
        output_recipe = compiled.recipe
    output_recipe_digest = hashlib.sha256(
        json.dumps(
            geometry_recipe_to_payload(output_recipe),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    output_proof_digest = hashlib.sha256(
        json.dumps(
            geometry_contract_proof(output_recipe).to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    proof = compiled.proof
    proof_summary = {
        "equivalent": proof.equivalent,
        "tolerance_version": proof.tolerance_version,
        "area": proof.area,
        "bounding_box": list(proof.bounding_box),
        "component_count": proof.component_count,
        "profile_count": proof.profile_count,
        "material_profile_count": proof.material_profile_count,
        "hole_count": proof.hole_count,
        "curve_type_counts": dict(proof.curve_type_counts),
        "recipe_digest": proof.recipe_digest,
    }
    proof_digest = hashlib.sha256(
        json.dumps(
            proof_summary,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    construction_summary = construction.provider_safe_summary().to_dict()
    construction_summary["node_kind_counts"] = dict(
        sorted(Counter(node.kind for node in construction.nodes).items())
    )
    return {
        "kind": "planar_construction_ir_v1",
        "output_kind": output_kind,
        "plane": construction.plane,
        "construction_digest": compiled.construction_digest,
        "recipe_proof_digest": proof_digest,
        "output_recipe_digest": output_recipe_digest,
        "output_proof_digest": output_proof_digest,
        "construction_summary": construction_summary,
        "proof_summary": proof_summary,
    }


def create_geometry_edit_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    part_id: str,
    draft: GeometryDraft,
    summary: str,
    edit_mode: str = "in_place",
) -> AgentProposal:
    """Create a revision-bound Part geometry edit proposal with explicit mode."""

    if type(context) is not AuthoringContext:
        raise TypeError("context must be AuthoringContext")
    if type(draft) is not GeometryDraft:
        raise TypeError("draft must be GeometryDraft")
    if edit_mode not in {"in_place", "branch"}:
        raise ValueError("edit_mode must be in_place or branch")
    target = next(
        (
            part
            for part in context.parts
            if part.part_id == str(part_id) and not part.suppressed
        ),
        None,
    )
    if (
        not context.binding.supported
        or context.binding.source_kind != "native"
        or target is None
    ):
        raise AuthoringContractError(
            "geometry edit requires one editable native Part"
        )
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=context.binding.document_id,
        target_session_id=context.binding.session_id,
        base_session_revision=context.binding.session_revision,
        draft_revision=draft_revision,
        operations=(
            ModelOperation(
                OperationKind.REPLACE_PART_GEOMETRY,
                {
                    "part_id": target.part_id,
                    "recipe": dict(draft.recipe_payload),
                },
            ),
        ),
        preconditions={
            "source_kind": "native",
            "part_id": target.part_id,
            "geometry_edit_mode": edit_mode,
        },
        expected_changes={
            "part_count_delta": 0,
            "edited_part_id": target.part_id,
            "projection_refresh_count": 1,
            "geometry_edit_mode": edit_mode,
            "creates_iteration_model": edit_mode == "branch",
        },
        invalidation_impact=(
            {"mesh": False, "definitions": False, "results": False}
            if edit_mode == "branch"
            else {"mesh": True, "definitions": True, "results": True}
        ),
        display_summary={
            "title": f"修改部件 {target.name}",
            "geometry_edit_mode": edit_mode,
            "creates_iteration_model": edit_mode == "branch",
            "migration_summary": (
                "创建迭代模型；迁移可保留的网格设置与模型定义；"
                "不迁移实际网格、验证、运行或结果"
                if edit_mode == "branch"
                else "在当前模型中替换部件几何"
            ),
            "target_model": context.model_name,
            "operation": OperationKind.REPLACE_PART_GEOMETRY.value,
            "part_id": target.part_id,
            "part_name": target.name,
            "recipe_type": type(draft.recipe).__name__,
            "source": _proposal_source(draft.recipe),
            "feature_operation": _proposal_operation(draft.recipe),
            "dimension": geometry_dimension(draft.recipe),
            "key_dimensions": dict(draft.key_dimensions),
            "expected_entity_count": draft.proof.expected_body_count,
            "summary": str(summary).strip(),
            "expected_new_objects": [],
            "invalidated_objects": (
                []
                if edit_mode == "branch"
                else ["mesh", "definitions", "results"]
            ),
            "source_state": (
                {
                    "mesh": "retained",
                    "definitions": "retained",
                    "runs": "retained",
                    "results": "retained",
                }
                if edit_mode == "branch"
                else {"mesh": "invalidated", "results": "invalidated"}
            ),
            "target_state": (
                {
                    "mesh": "not_migrated",
                    "validations": "reset",
                    "runs": "not_migrated",
                    "results": "not_migrated",
                }
                if edit_mode == "branch"
                else {"document": "current"}
            ),
            "invalidation_impact": (
                {"mesh": False, "definitions": False, "results": False}
                if edit_mode == "branch"
                else {"mesh": True, "definitions": True, "results": True}
            ),
            "base_session_revision": context.binding.session_revision,
            "preview": draft.preview.to_dict(),
            "proof": draft.proof.to_dict(),
        },
    )


def create_profile_extrusion_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    part_id: str,
    base_recipe: object,
    source_face_ids: Sequence[str],
    height: Real,
    summary: str,
    edit_mode: str = "in_place",
) -> AgentProposal:
    """Create one atomic proposal for explicitly selected material Profiles."""

    if type(context) is not AuthoringContext:
        raise TypeError("context must be AuthoringContext")
    if edit_mode not in {"in_place", "branch"}:
        raise ValueError("edit_mode must be in_place or branch")
    target = next(
        (
            part
            for part in context.parts
            if part.part_id == str(part_id) and not part.suppressed
        ),
        None,
    )
    if (
        not context.binding.supported
        or context.binding.source_kind != "native"
        or target is None
    ):
        raise AuthoringContractError(
            "Profile extrusion requires one editable native Part"
        )
    if type(base_recipe) is not SketchGeometry or not base_recipe.is_strict:
        raise AuthoringContractError(
            "Profile extrusion requires one strict planar sketch recipe"
        )
    requested = tuple(source_face_ids)
    if not requested:
        raise AuthoringContractError(
            "source_face_ids must explicitly select at least one material Profile"
        )
    normalized_height = _finite(height, "height")
    recipes = tuple(
        ExtrudedGeometry(base_recipe, normalized_height, (source_face_id,))
        for source_face_id in requested
    )
    canonical_ids = tuple(recipe.source_face_ids[0] for recipe in recipes)
    if len(canonical_ids) != len(set(canonical_ids)):
        raise AuthoringContractError("source_face_ids select duplicate Profiles")
    drafts = tuple(geometry_draft(recipe) for recipe in recipes)
    if any(
        not draft.proof.exact or draft.proof.expected_body_count != 1
        for draft in drafts
    ):
        raise AuthoringContractError(
            "each selected Profile must prove exactly one solid Body"
        )
    proposal_summary = str(summary).strip()
    if not proposal_summary:
        raise AuthoringContractError("Profile extrusion proposal summary is blank")
    operation = ModelOperation(
        OperationKind.EXTRUDE_PART_PROFILES,
        {
            "part_id": target.part_id,
            "base_recipe": geometry_recipe_to_payload(base_recipe),
            "source_face_ids": list(canonical_ids),
            "height": normalized_height,
        },
    )
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=context.binding.document_id,
        target_session_id=context.binding.session_id,
        base_session_revision=context.binding.session_revision,
        draft_revision=draft_revision,
        operations=(operation,),
        preconditions={
            "source_kind": "native",
            "part_id": target.part_id,
            "source_face_ids": list(canonical_ids),
            "geometry_edit_mode": edit_mode,
        },
        expected_changes={
            "part_count_delta": len(canonical_ids) - 1,
            "edited_part_id": target.part_id,
            "result_part_count": len(canonical_ids),
            "projection_refresh_count": 1,
            "geometry_edit_mode": edit_mode,
            "creates_iteration_model": edit_mode == "branch",
        },
        invalidation_impact=(
            {"mesh": False, "definitions": False, "results": False}
            if edit_mode == "branch"
            else {"mesh": True, "definitions": True, "results": True}
        ),
        display_summary={
            "title": f"拉伸部件 {target.name} 的选定 Profiles",
            "summary": proposal_summary,
            "target_model": context.model_name,
            "operation": OperationKind.EXTRUDE_PART_PROFILES.value,
            "geometry_edit_mode": edit_mode,
            "creates_iteration_model": edit_mode == "branch",
            "feature_operation": "extrude",
            "part_id": target.part_id,
            "part_name": target.name,
            "source": list(canonical_ids),
            "key_dimensions": {"height": normalized_height},
            "direction": "positive_sketch_normal",
            "expected_entity_count": len(canonical_ids),
            "expected_part_count": len(canonical_ids),
            "expected_new_objects": [
                f"{len(canonical_ids)} independent solid Part(s)"
            ],
            "invalidated_objects": (
                [] if edit_mode == "branch" else ["mesh", "definitions", "results"]
            ),
            "invalidation_impact": (
                {"mesh": False, "definitions": False, "results": False}
                if edit_mode == "branch"
                else {"mesh": True, "definitions": True, "results": True}
            ),
            "migration_summary": (
                "创建迭代模型；迁移可保留的网格设置与模型定义；"
                "不迁移实际网格、验证、运行或结果"
                if edit_mode == "branch"
                else "在当前模型中转换 Profile 几何"
            ),
            "base_session_revision": context.binding.session_revision,
            "proofs": [draft.proof.to_dict() for draft in drafts],
        },
    )


def create_profile_revolution_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    part_id: str,
    base_recipe: object,
    source_face_id: str,
    axis: str,
    angle_degrees: Real,
    summary: str,
    edit_mode: str = "in_place",
) -> AgentProposal:
    """Create a revision-bound proposal for one canonical Profile revolution."""

    recipe = RevolvedGeometry(
        base_recipe,
        axis,
        _finite(angle_degrees, "angle_degrees"),
        (source_face_id,),
    )
    return _create_single_profile_derived_proposal(
        proposal_id=proposal_id,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=source_tool_call_ids,
        context=context,
        draft_revision=draft_revision,
        part_id=part_id,
        base_recipe=base_recipe,
        recipe=recipe,
        operation_kind=OperationKind.REVOLVE_PART_PROFILE,
        parameters={
            "axis": recipe.axis,
            "angle_degrees": recipe.angle_degrees,
        },
        summary=summary,
        edit_mode=edit_mode,
    )


def create_profile_path_sweep_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    part_id: str,
    base_recipe: object,
    source_face_id: str,
    path: WireGeometry,
    frame_strategy: str,
    summary: str,
    edit_mode: str = "in_place",
) -> AgentProposal:
    """Create a revision-bound proposal for one explicit open-path sweep."""

    recipe = PathSweptGeometry(
        base_recipe,
        path,
        (source_face_id,),
        frame_strategy,
    )
    return _create_single_profile_derived_proposal(
        proposal_id=proposal_id,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=source_tool_call_ids,
        context=context,
        draft_revision=draft_revision,
        part_id=part_id,
        base_recipe=base_recipe,
        recipe=recipe,
        operation_kind=OperationKind.SWEEP_PART_PROFILE,
        parameters={
            "ordered_wire": _geometry_recipe_to_payload(path),
            "frame_strategy": recipe.frame_strategy,
        },
        summary=summary,
        edit_mode=edit_mode,
    )


def _create_single_profile_derived_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    part_id: str,
    base_recipe: object,
    recipe: RevolvedGeometry | PathSweptGeometry,
    operation_kind: OperationKind,
    parameters: Mapping[str, object],
    summary: str,
    edit_mode: str,
) -> AgentProposal:
    if type(context) is not AuthoringContext:
        raise TypeError("context must be AuthoringContext")
    if edit_mode not in {"in_place", "branch"}:
        raise ValueError("edit_mode must be in_place or branch")
    target = next(
        (
            part
            for part in context.parts
            if part.part_id == str(part_id) and not part.suppressed
        ),
        None,
    )
    if (
        not context.binding.supported
        or context.binding.source_kind != "native"
        or target is None
    ):
        raise AuthoringContractError(
            "derived Profile feature requires one editable native Part"
        )
    if type(base_recipe) is not SketchGeometry or not base_recipe.is_strict:
        raise AuthoringContractError(
            "derived Profile feature requires one strict planar sketch recipe"
        )
    if len(recipe.source_face_ids) != 1:
        raise AuthoringContractError(
            "derived Profile feature requires one explicit canonical source Profile"
        )
    draft = geometry_draft(recipe)
    if not draft.proof.exact or draft.proof.expected_body_count != 1:
        raise AuthoringContractError(
            "derived Profile feature must prove exactly one solid Body"
        )
    normalized_summary = str(summary).strip()
    if not normalized_summary:
        raise AuthoringContractError("derived Profile proposal summary is blank")
    source_face_id = recipe.source_face_ids[0]
    operation_parameters = {
        "part_id": target.part_id,
        "base_recipe": geometry_recipe_to_payload(base_recipe),
        "source_face_id": source_face_id,
        **dict(parameters),
    }
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=context.binding.document_id,
        target_session_id=context.binding.session_id,
        base_session_revision=context.binding.session_revision,
        draft_revision=draft_revision,
        operations=(ModelOperation(operation_kind, operation_parameters),),
        preconditions={
            "source_kind": "native",
            "part_id": target.part_id,
            "source_face_id": source_face_id,
            "geometry_edit_mode": edit_mode,
        },
        expected_changes={
            "part_count_delta": 0,
            "edited_part_id": target.part_id,
            "result_part_count": 1,
            "projection_refresh_count": 1,
            "geometry_edit_mode": edit_mode,
            "creates_iteration_model": edit_mode == "branch",
        },
        invalidation_impact=(
            {"mesh": False, "definitions": False, "results": False}
            if edit_mode == "branch"
            else {"mesh": True, "definitions": True, "results": True}
        ),
        display_summary={
            "title": f"从部件 {target.name} 创建三维派生特征",
            "summary": normalized_summary,
            "target_model": context.model_name,
            "operation": operation_kind.value,
            "geometry_edit_mode": edit_mode,
            "creates_iteration_model": edit_mode == "branch",
            "feature_operation": _proposal_operation(recipe),
            "part_id": target.part_id,
            "part_name": target.name,
            "source": [source_face_id],
            "key_dimensions": dict(draft.key_dimensions),
            "frame_strategy": (
                recipe.frame_strategy
                if isinstance(recipe, PathSweptGeometry)
                else None
            ),
            "expected_entity_count": 1,
            "expected_part_count": 1,
            "expected_new_objects": ["1 solid Part"],
            "invalidated_objects": (
                [] if edit_mode == "branch" else ["mesh", "definitions", "results"]
            ),
            "migration_summary": (
                "创建迭代模型；迁移可保留的网格设置与模型定义；"
                "不迁移实际网格、验证、运行或结果"
                if edit_mode == "branch"
                else "在当前模型中转换 Profile 几何"
            ),
            "invalidation_impact": (
                {"mesh": False, "definitions": False, "results": False}
                if edit_mode == "branch"
                else {"mesh": True, "definitions": True, "results": True}
            ),
            "base_session_revision": context.binding.session_revision,
            "preview": draft.preview.to_dict(),
            "proof": draft.proof.to_dict(),
        },
    )


def add_planar_circle(
    recipe: object,
    *,
    center_x: Real,
    center_y: Real,
    radius: Real,
) -> GeometryDraft:
    return _planar_edit_draft(
        _add_planar_circle_sketch(
            recipe, center_x=center_x, center_y=center_y, radius=radius
        )
    )


def _add_planar_circle_sketch(
    recipe: object,
    *,
    center_x: Real,
    center_y: Real,
    radius: Real,
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    point_id = _next_sketch_ids(sketch, "P", 1)[0]
    curve_id = _next_sketch_ids(sketch, "C", 1)[0]
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        (
            *sketch.points,
            SketchPoint(
                point_id,
                _finite(center_x, "center_x"),
                _finite(center_y, "center_y"),
            ),
        ),
        (
            *sketch.curves,
            SketchCircle(curve_id, point_id, _finite(radius, "radius")),
        ),
        sketch.constraints,
    )
    solved = _solve_planar_candidate(updated, fixed_point_ids={str(point_id)})
    return solved


def add_planar_line(
    recipe: object,
    *,
    start: Mapping[str, object],
    end: Mapping[str, object],
) -> GeometryDraft:
    return _planar_edit_draft(_add_planar_line_sketch(recipe, start, end))


def _add_planar_line_sketch(
    recipe: object,
    start: Mapping[str, object],
    end: Mapping[str, object],
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    sketch, start_id, _ = _resolve_planar_point_ref(sketch, start, "start")
    sketch, end_id, _ = _resolve_planar_point_ref(sketch, end, "end")
    line_id = _next_sketch_ids(sketch, "L", 1)[0]
    return SketchGeometry(
        sketch.name,
        sketch.plane,
        sketch.points,
        (*sketch.curves, SketchLine(line_id, start_id, end_id)),
        sketch.constraints,
    )


def add_planar_arc(
    recipe: object,
    *,
    start: Mapping[str, object],
    center: Mapping[str, object],
    end: Mapping[str, object],
    orientation: str,
) -> GeometryDraft:
    return _planar_edit_draft(
        _add_planar_arc_sketch(recipe, start, center, end, orientation)
    )


def _add_planar_arc_sketch(
    recipe: object,
    start: Mapping[str, object],
    center: Mapping[str, object],
    end: Mapping[str, object],
    orientation: str,
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    sketch, start_id, _ = _resolve_planar_point_ref(sketch, start, "start")
    sketch, center_id, _ = _resolve_planar_point_ref(sketch, center, "center")
    sketch, end_id, _ = _resolve_planar_point_ref(sketch, end, "end")
    arc_id = _next_sketch_ids(sketch, "A", 1)[0]
    return SketchGeometry(
        sketch.name,
        sketch.plane,
        sketch.points,
        (
            *sketch.curves,
            SketchArc(arc_id, start_id, center_id, end_id, orientation),
        ),
        sketch.constraints,
    )


def update_planar_line(
    recipe: object,
    *,
    line_id: str,
    start: Mapping[str, object] | None = None,
    end: Mapping[str, object] | None = None,
) -> GeometryDraft:
    return _planar_edit_draft(
        _update_planar_line_sketch(recipe, line_id=line_id, start=start, end=end)
    )


def _update_planar_line_sketch(
    recipe: object,
    *,
    line_id: str,
    start: Mapping[str, object] | None = None,
    end: Mapping[str, object] | None = None,
) -> SketchGeometry:
    if start is None and end is None:
        raise ValueError("update_planar_line requires start or end")
    sketch = _as_strict_planar_sketch(recipe)
    target = next(
        (item for item in sketch.curves if type(item) is SketchLine and item.id == line_id),
        None,
    )
    if target is None:
        raise ValueError("line_id does not identify one editable line")
    fixed: set[str] = set()
    start_id = target.start_point_id
    end_id = target.end_point_id
    if start is not None:
        sketch, start_id, authored = _resolve_planar_point_ref(sketch, start, "start")
        if authored:
            fixed.add(start_id)
    if end is not None:
        sketch, end_id, authored = _resolve_planar_point_ref(sketch, end, "end")
        if authored:
            fixed.add(end_id)
    candidate = SketchGeometry(
        sketch.name,
        sketch.plane,
        sketch.points,
        tuple(
            SketchLine(item.id, start_id, end_id) if item is target else item
            for item in sketch.curves
        ),
        sketch.constraints,
    )
    return _solve_planar_candidate(candidate, fixed_point_ids=fixed)


def update_planar_arc(
    recipe: object,
    *,
    arc_id: str,
    start: Mapping[str, object] | None = None,
    center: Mapping[str, object] | None = None,
    end: Mapping[str, object] | None = None,
    orientation: str | None = None,
) -> GeometryDraft:
    return _planar_edit_draft(
        _update_planar_arc_sketch(
            recipe,
            arc_id=arc_id,
            start=start,
            center=center,
            end=end,
            orientation=orientation,
        )
    )


def _update_planar_arc_sketch(
    recipe: object,
    *,
    arc_id: str,
    start: Mapping[str, object] | None = None,
    center: Mapping[str, object] | None = None,
    end: Mapping[str, object] | None = None,
    orientation: str | None = None,
) -> SketchGeometry:
    if start is None and center is None and end is None and orientation is None:
        raise ValueError("update_planar_arc requires one changed field")
    sketch = _as_strict_planar_sketch(recipe)
    target = next(
        (item for item in sketch.curves if type(item) is SketchArc and item.id == arc_id),
        None,
    )
    if target is None:
        raise ValueError("arc_id does not identify one editable arc")
    refs = {
        "start": target.start_point_id,
        "center": target.center_point_id,
        "end": target.end_point_id,
    }
    fixed: set[str] = set()
    for label, value in (("start", start), ("center", center), ("end", end)):
        if value is None:
            continue
        sketch, refs[label], authored = _resolve_planar_point_ref(sketch, value, label)
        if authored:
            fixed.add(refs[label])
    candidate = SketchGeometry(
        sketch.name,
        sketch.plane,
        sketch.points,
        tuple(
            SketchArc(
                item.id,
                refs["start"],
                refs["center"],
                refs["end"],
                item.orientation if orientation is None else orientation,
            )
            if item is target
            else item
            for item in sketch.curves
        ),
        sketch.constraints,
    )
    return _solve_planar_candidate(candidate, fixed_point_ids=fixed)


def delete_planar_curves(
    recipe: object,
    *,
    curve_ids: Sequence[str],
) -> GeometryDraft:
    return _planar_edit_draft(
        _delete_planar_curves_sketch(recipe, curve_ids=curve_ids)
    )


def _delete_planar_curves_sketch(
    recipe: object,
    *,
    curve_ids: Sequence[str],
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    ids = _exact_id_list(curve_ids, "curve_ids", maximum=32)
    available = {
        item.id for item in sketch.curves if type(item) in {SketchLine, SketchArc}
    }
    if not set(ids) <= available:
        raise ValueError("curve_ids must identify exact line or arc IDs")
    if any(
        set(ids).intersection(sketch_constraint_entity_ids(item))
        for item in sketch.constraints
    ):
        raise AuthoringContractError(
            "cannot delete curves referenced by remaining sketch constraints"
        )
    return SketchGeometry(
        sketch.name,
        sketch.plane,
        sketch.points,
        tuple(item for item in sketch.curves if item.id not in set(ids)),
        sketch.constraints,
    )


def add_planar_constraint(
    recipe: object,
    *,
    constraint: Mapping[str, object],
) -> GeometryDraft:
    return _planar_edit_draft(
        _add_planar_constraint_sketch(recipe, constraint=constraint)
    )


def _add_planar_constraint_sketch(
    recipe: object,
    *,
    constraint: Mapping[str, object],
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    constraint_id = _next_sketch_ids(sketch, "K", 1)[0]
    created = _constraint_from_agent_spec(sketch, constraint_id, constraint)
    candidate = SketchGeometry(
        sketch.name,
        sketch.plane,
        sketch.points,
        sketch.curves,
        (*sketch.constraints, created),
    )
    solved = _solve_planar_candidate(candidate, new_constraint_ids=(constraint_id,))
    return solved


def replace_planar_constraint(
    recipe: object,
    *,
    constraint_id: str,
    constraint: Mapping[str, object],
) -> GeometryDraft:
    return _planar_edit_draft(
        _replace_planar_constraint_sketch(
            recipe,
            constraint_id=constraint_id,
            constraint=constraint,
        )
    )


def _replace_planar_constraint_sketch(
    recipe: object,
    *,
    constraint_id: str,
    constraint: Mapping[str, object],
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    if constraint_id not in {item.id for item in sketch.constraints}:
        raise ValueError("constraint_id does not identify one constraint")
    replacement = _constraint_from_agent_spec(sketch, constraint_id, constraint)
    candidate = SketchGeometry(
        sketch.name,
        sketch.plane,
        sketch.points,
        sketch.curves,
        tuple(
            replacement if item.id == constraint_id else item
            for item in sketch.constraints
        ),
    )
    solved = _solve_planar_candidate(candidate, new_constraint_ids=(constraint_id,))
    return solved


def delete_planar_constraints(
    recipe: object,
    *,
    constraint_ids: Sequence[str],
) -> GeometryDraft:
    return _planar_edit_draft(
        _delete_planar_constraints_sketch(recipe, constraint_ids=constraint_ids)
    )


def _delete_planar_constraints_sketch(
    recipe: object,
    *,
    constraint_ids: Sequence[str],
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    ids = _exact_id_list(constraint_ids, "constraint_ids", maximum=32)
    if not set(ids) <= {item.id for item in sketch.constraints}:
        raise ValueError("constraint_ids must identify existing constraints")
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        sketch.points,
        sketch.curves,
        tuple(item for item in sketch.constraints if item.id not in set(ids)),
    )
    solve_sketch_constraints(updated)
    return updated


def add_planar_rectangle(
    recipe: object,
    *,
    x: Real,
    y: Real,
    width: Real,
    height: Real,
) -> GeometryDraft:
    return _planar_edit_draft(
        _add_planar_rectangle_sketch(
            recipe, x=x, y=y, width=width, height=height
        )
    )


def _add_planar_rectangle_sketch(
    recipe: object,
    *,
    x: Real,
    y: Real,
    width: Real,
    height: Real,
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    x_value = _finite(x, "x")
    y_value = _finite(y, "y")
    width_value = _finite(width, "width")
    height_value = _finite(height, "height")
    if width_value <= 0.0 or height_value <= 0.0:
        raise ValueError("rectangle width and height must be positive")
    point_ids = _next_sketch_ids(sketch, "P", 4)
    line_ids = _next_sketch_ids(sketch, "L", 4)
    points = (
        SketchPoint(point_ids[0], x_value, y_value),
        SketchPoint(point_ids[1], x_value + width_value, y_value),
        SketchPoint(
            point_ids[2],
            x_value + width_value,
            y_value + height_value,
        ),
        SketchPoint(point_ids[3], x_value, y_value + height_value),
    )
    lines = tuple(
        SketchLine(
            line_ids[index],
            point_ids[index],
            point_ids[(index + 1) % 4],
        )
        for index in range(4)
    )
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        (*sketch.points, *points),
        (*sketch.curves, *lines),
        sketch.constraints,
    )
    return updated


def add_planar_polygon(
    recipe: object,
    *,
    vertices: Sequence[Sequence[Real]],
) -> GeometryDraft:
    return _planar_edit_draft(
        _add_planar_polygon_sketch(recipe, vertices=vertices)
    )


def _add_planar_polygon_sketch(
    recipe: object,
    *,
    vertices: Sequence[Sequence[Real]],
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    values = tuple(vertices)
    if not 3 <= len(values) <= 64:
        raise ValueError("polygon vertices must contain between 3 and 64 points")
    coordinates: list[tuple[float, float]] = []
    for index, vertex in enumerate(values):
        if isinstance(vertex, (str, bytes, bytearray)):
            raise TypeError("polygon vertices must contain coordinate pairs")
        components = tuple(vertex)
        if len(components) != 2:
            raise ValueError("polygon vertices must contain coordinate pairs")
        coordinates.append(
            (
                _finite(components[0], f"vertices[{index}].x"),
                _finite(components[1], f"vertices[{index}].y"),
            )
        )
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("polygon vertices must be distinct")
    point_ids = _next_sketch_ids(sketch, "P", len(coordinates))
    line_ids = _next_sketch_ids(sketch, "L", len(coordinates))
    points = tuple(
        SketchPoint(point_id, x, y)
        for point_id, (x, y) in zip(point_ids, coordinates, strict=True)
    )
    lines = tuple(
        SketchLine(
            line_ids[index],
            point_ids[index],
            point_ids[(index + 1) % len(point_ids)],
        )
        for index in range(len(point_ids))
    )
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        (*sketch.points, *points),
        (*sketch.curves, *lines),
        sketch.constraints,
    )
    return updated


def update_planar_point(
    recipe: object,
    *,
    point_id: str,
    x: Real | None = None,
    y: Real | None = None,
) -> GeometryDraft:
    return _planar_edit_draft(
        _update_planar_point_sketch(recipe, point_id=point_id, x=x, y=y)
    )


def _update_planar_point_sketch(
    recipe: object,
    *,
    point_id: str,
    x: Real | None = None,
    y: Real | None = None,
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    if x is None and y is None:
        raise ValueError("update_planar_point requires x or y")
    if str(point_id) not in {point.id for point in sketch.points}:
        raise ValueError("point_id does not identify one editable point")
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        tuple(
            SketchPoint(
                point.id,
                point.u if point.id != str(point_id) or x is None else _finite(x, "x"),
                point.v if point.id != str(point_id) or y is None else _finite(y, "y"),
            )
            for point in sketch.points
        ),
        sketch.curves,
        sketch.constraints,
    )
    solved = _solve_planar_candidate(updated, fixed_point_ids={str(point_id)})
    return solved


def update_planar_circle(
    recipe: object,
    *,
    circle_id: str,
    center_x: Real | None = None,
    center_y: Real | None = None,
    radius: Real | None = None,
) -> GeometryDraft:
    return _planar_edit_draft(
        _update_planar_circle_sketch(
            recipe,
            circle_id=circle_id,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
        )
    )


def _update_planar_circle_sketch(
    recipe: object,
    *,
    circle_id: str,
    center_x: Real | None = None,
    center_y: Real | None = None,
    radius: Real | None = None,
) -> SketchGeometry:
    sketch = _as_strict_planar_sketch(recipe)
    target = next(
        (
            curve
            for curve in sketch.curves
            if isinstance(curve, SketchCircle) and curve.id == str(circle_id)
        ),
        None,
    )
    if target is None or target.center_point_id is None:
        raise ValueError("circle_id does not identify one editable circle")
    points = []
    for point in sketch.points:
        if point.id != target.center_point_id:
            points.append(point)
            continue
        points.append(
            SketchPoint(
                point.id,
                point.u if center_x is None else _finite(center_x, "center_x"),
                point.v if center_y is None else _finite(center_y, "center_y"),
            )
        )
    requested_radius = None if radius is None else _finite(radius, "radius")
    curves = tuple(
        (
            SketchCircle(
                curve.id,
                curve.center_point_id,
                curve.radius if requested_radius is None else requested_radius,
            )
            if curve is target
            else curve
        )
        for curve in sketch.curves
    )
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        tuple(points),
        curves,
        sketch.constraints,
    )
    solved = _solve_planar_candidate(
        updated,
        fixed_point_ids=(
            {str(target.center_point_id)}
            if center_x is not None or center_y is not None
            else set()
        ),
    )
    if requested_radius is not None:
        solved_target = next(
            curve
            for curve in solved.curves
            if isinstance(curve, SketchCircle) and curve.id == target.id
        )
        if not math.isclose(
            solved_target.radius,
            requested_radius,
            rel_tol=_SKETCH_AUTHORING_TOLERANCE,
            abs_tol=_SKETCH_AUTHORING_TOLERANCE,
        ):
            raise AuthoringContractError(
                "circle radius edit conflicts with a driving or equality "
                "constraint; replace or remove that constraint first"
            )
    return solved


def delete_planar_circles(
    recipe: object,
    *,
    circle_ids: Sequence[str],
) -> GeometryDraft:
    """Atomically remove exact unconstrained circles from a detached sketch."""

    return _planar_edit_draft(
        _delete_planar_circles_sketch(recipe, circle_ids=circle_ids)
    )


def _delete_planar_circles_sketch(
    recipe: object,
    *,
    circle_ids: Sequence[str],
) -> SketchGeometry:

    sketch = _as_strict_planar_sketch(recipe)
    target_ids = _exact_circle_ids(circle_ids, "circle_ids")
    circle_by_id = {
        curve.id: curve
        for curve in sketch.curves
        if isinstance(curve, SketchCircle) and curve.center_point_id is not None
    }
    if any(circle_id not in circle_by_id for circle_id in target_ids):
        raise ValueError("circle_ids must identify existing editable circles")
    removed_centers = {
        str(circle_by_id[circle_id].center_point_id) for circle_id in target_ids
    }
    referenced_entities = set(target_ids) | removed_centers
    if any(
        referenced_entities.intersection(sketch_constraint_entity_ids(constraint))
        for constraint in sketch.constraints
    ):
        raise AuthoringContractError(
            "cannot delete circles or centers referenced by sketch constraints"
        )
    remaining_curves = tuple(
        curve for curve in sketch.curves if curve.id not in set(target_ids)
    )
    remaining_point_references = {
        entity_id
        for curve in remaining_curves
        for entity_id in _curve_point_ids(curve)
    }
    removable_centers = removed_centers - remaining_point_references
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        tuple(
            point for point in sketch.points if point.id not in removable_centers
        ),
        remaining_curves,
        sketch.constraints,
    )
    return updated


def replace_planar_circle_pattern(
    recipe: object,
    *,
    target_circle_ids: Sequence[str],
    count: int,
    start_center_x: Real,
    start_center_y: Real,
    spacing_x: Real,
    spacing_y: Real,
    radius: Real,
) -> GeometryDraft:
    """Replace exact circles with one deterministic detached linear pattern."""

    return _planar_edit_draft(
        _replace_planar_circle_pattern_sketch(
            recipe,
            target_circle_ids=target_circle_ids,
            count=count,
            start_center_x=start_center_x,
            start_center_y=start_center_y,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            radius=radius,
        )
    )


def _replace_planar_circle_pattern_sketch(
    recipe: object,
    *,
    target_circle_ids: Sequence[str],
    count: int,
    start_center_x: Real,
    start_center_y: Real,
    spacing_x: Real,
    spacing_y: Real,
    radius: Real,
) -> SketchGeometry:

    sketch = _as_strict_planar_sketch(recipe)
    target_ids = _exact_circle_ids(target_circle_ids, "target_circle_ids")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 32:
        raise ValueError("count must be an integer between 1 and 32")
    start_x = _finite(start_center_x, "start_center_x")
    start_y = _finite(start_center_y, "start_center_y")
    delta_x = _finite(spacing_x, "spacing_x")
    delta_y = _finite(spacing_y, "spacing_y")
    radius_value = _finite(radius, "radius")
    if radius_value <= 0.0:
        raise ValueError("radius must be positive")
    if count > 1 and delta_x == 0.0 and delta_y == 0.0:
        raise ValueError("spacing vector must be non-zero when count is greater than one")

    circle_by_id = {
        curve.id: curve
        for curve in sketch.curves
        if isinstance(curve, SketchCircle) and curve.center_point_id is not None
    }
    if any(circle_id not in circle_by_id for circle_id in target_ids):
        raise ValueError("target_circle_ids must identify existing editable circles")
    removed_centers = {
        str(circle_by_id[circle_id].center_point_id) for circle_id in target_ids
    }
    referenced_entities = set(target_ids) | removed_centers
    if any(
        referenced_entities.intersection(sketch_constraint_entity_ids(constraint))
        for constraint in sketch.constraints
    ):
        raise AuthoringContractError(
            "cannot replace circles or centers referenced by sketch constraints"
        )

    new_point_ids = _next_sketch_ids(sketch, "P", count)
    new_curve_ids = _next_sketch_ids(sketch, "C", count)
    remaining_curves = tuple(
        curve for curve in sketch.curves if curve.id not in set(target_ids)
    )
    remaining_point_references = {
        entity_id
        for curve in remaining_curves
        for entity_id in _curve_point_ids(curve)
    }
    removable_centers = removed_centers - remaining_point_references
    new_points = tuple(
        SketchPoint(
            point_id,
            start_x + index * delta_x,
            start_y + index * delta_y,
        )
        for index, point_id in enumerate(new_point_ids)
    )
    new_circles = tuple(
        SketchCircle(curve_id, point_id, radius_value)
        for curve_id, point_id in zip(new_curve_ids, new_point_ids, strict=True)
    )
    updated = SketchGeometry(
        sketch.name,
        sketch.plane,
        (
            *(point for point in sketch.points if point.id not in removable_centers),
            *new_points,
        ),
        (*remaining_curves, *new_circles),
        sketch.constraints,
    )
    return updated


def apply_planar_edit_batch(
    recipe: object,
    *,
    edits: Sequence[Mapping[str, object]],
) -> GeometryDraft:
    """Apply bounded non-nested planar edits to one detached draft."""

    values = tuple(edits)
    if not 1 <= len(values) <= 16:
        raise ValueError("batch edits must contain between 1 and 16 operations")
    current = _as_strict_planar_sketch(recipe)
    handlers = {
        "add_circle": _add_planar_circle_sketch,
        "add_rectangle": _add_planar_rectangle_sketch,
        "add_polygon": _batch_add_polygon_sketch,
        "update_point": _update_planar_point_sketch,
        "update_circle": _update_planar_circle_sketch,
        "delete_circles": _delete_planar_circles_sketch,
        "replace_circle_pattern": _replace_planar_circle_pattern_sketch,
    }
    for index, raw_edit in enumerate(values):
        if not isinstance(raw_edit, Mapping):
            raise TypeError(f"edits[{index}] must be an object")
        edit = dict(raw_edit)
        operation = edit.pop("operation", None)
        if operation == "batch" or operation not in handlers:
            if operation == "add_line":
                current = _add_planar_line_sketch(
                    current, edit.pop("start"), edit.pop("end")
                )
            elif operation == "add_arc":
                current = _add_planar_arc_sketch(
                    current,
                    edit.pop("start"),
                    edit.pop("center"),
                    edit.pop("end"),
                    edit.pop("orientation"),
                )
            elif operation == "update_line":
                current = _update_planar_line_sketch(current, **edit)
                edit.clear()
            elif operation == "update_arc":
                current = _update_planar_arc_sketch(current, **edit)
                edit.clear()
            elif operation == "delete_curves":
                current = _delete_planar_curves_sketch(current, **edit)
                edit.clear()
            elif operation == "add_constraint":
                current = _add_planar_constraint_sketch(current, **edit)
                edit.clear()
            elif operation == "replace_constraint":
                current = _replace_planar_constraint_sketch(current, **edit)
                edit.clear()
            elif operation == "delete_constraints":
                current = _delete_planar_constraints_sketch(current, **edit)
                edit.clear()
            else:
                raise ValueError(f"edits[{index}] uses an unsupported operation")
            if edit:
                raise ValueError(f"edits[{index}] has unsupported fields")
        else:
            current = handlers[str(operation)](current, **edit)
    return _planar_edit_draft(current)


def _batch_add_polygon(recipe: object, *, vertices: object) -> GeometryDraft:
    return _planar_edit_draft(
        _batch_add_polygon_sketch(recipe, vertices=vertices)
    )


def _batch_add_polygon_sketch(
    recipe: object,
    *,
    vertices: object,
) -> SketchGeometry:
    if not isinstance(vertices, Sequence) or isinstance(
        vertices, (str, bytes, bytearray)
    ):
        raise TypeError("vertices must be an array")
    coordinates: list[tuple[object, object]] = []
    for vertex in vertices:
        if not isinstance(vertex, Mapping) or set(vertex) != {"x", "y"}:
            raise ValueError("vertices must contain closed x/y objects")
        coordinates.append((vertex["x"], vertex["y"]))
    return _add_planar_polygon_sketch(recipe, vertices=coordinates)


def _resolve_planar_point_ref(
    sketch: SketchGeometry,
    value: Mapping[str, object],
    label: str,
) -> tuple[SketchGeometry, str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a point reference object")
    data = dict(value)
    if set(data) == {"point_id"}:
        point_id = data["point_id"]
        if type(point_id) is not str or not point_id:
            raise ValueError(f"{label}.point_id must be a non-empty string")
        if point_id not in {item.id for item in sketch.points}:
            raise ValueError(f"{label}.point_id is unavailable")
        return sketch, point_id, False
    if set(data) != {"x", "y"}:
        raise ValueError(f"{label} point reference fields do not match")
    x = _finite(data["x"], f"{label}.x")
    y = _finite(data["y"], f"{label}.y")
    tolerance = _SKETCH_AUTHORING_TOLERANCE
    matches = tuple(
        item
        for item in sketch.points
        if math.hypot(item.u - x, item.v - y) <= tolerance
    )
    if len(matches) > 1:
        raise AuthoringContractError(
            f"{label} coordinate matches multiple points; use point_id"
        )
    if matches:
        return sketch, matches[0].id, True
    point_id = _next_sketch_ids(sketch, "P", 1)[0]
    return (
        SketchGeometry(
            sketch.name,
            sketch.plane,
            (*sketch.points, SketchPoint(point_id, x, y)),
            sketch.curves,
            sketch.constraints,
        ),
        point_id,
        True,
    )


def _constraint_from_agent_spec(
    sketch: SketchGeometry,
    constraint_id: str,
    value: Mapping[str, object],
) -> SketchConstraint:
    if not isinstance(value, Mapping):
        raise TypeError("constraint must be an object")
    data = dict(value)
    kind = data.get("kind")
    enabled = data.pop("enabled", True)
    data.pop("kind", None)
    if type(enabled) is not bool:
        raise TypeError("constraint enabled must be a bool")
    simple: dict[str, tuple[type, tuple[str, ...]]] = {
        "coincident": (SketchCoincidentConstraint, ("first_point_id", "second_point_id")),
        "point_on_curve": (SketchPointOnCurveConstraint, ("point_id", "curve_id")),
        "horizontal": (SketchHorizontalConstraint, ("line_id",)),
        "vertical": (SketchVerticalConstraint, ("line_id",)),
        "parallel": (SketchParallelConstraint, ("first_line_id", "second_line_id")),
        "perpendicular": (SketchPerpendicularConstraint, ("first_line_id", "second_line_id")),
        "equal_length": (SketchEqualLengthConstraint, ("first_line_id", "second_line_id")),
        "equal_radius": (SketchEqualRadiusConstraint, ("first_curve_id", "second_curve_id")),
        "concentric": (SketchConcentricConstraint, ("first_curve_id", "second_curve_id")),
    }
    if kind in simple:
        constraint_type, fields = simple[str(kind)]
        if set(data) != set(fields):
            raise ValueError("constraint fields do not match its kind")
        return constraint_type(
            constraint_id,
            *(data[field] for field in fields),
            source="manual",
            enabled=enabled,
        )
    if kind == "tangent":
        allowed = {"first_curve_id", "second_curve_id", "branch_hint"}
        if not {"first_curve_id", "second_curve_id"} <= set(data) <= allowed:
            raise ValueError("constraint fields do not match tangent")
        return SketchTangentConstraint(
            constraint_id,
            data["first_curve_id"],
            data["second_curve_id"],
            data.get("branch_hint", 0),
            source="manual",
            enabled=enabled,
        )
    if kind == "fixed":
        if set(data) != {"point_id"}:
            raise ValueError("fixed constraint requires point_id only")
        point = next(
            (item for item in sketch.points if item.id == data["point_id"]),
            None,
        )
        if point is None:
            raise ValueError("fixed constraint point_id is unavailable")
        return SketchFixedConstraint(
            constraint_id,
            point.id,
            point.u,
            point.v,
            source="manual",
            enabled=enabled,
        )
    driving = data.pop("driving", True)
    if type(driving) is not bool:
        raise TypeError("dimension driving must be a bool")
    if kind == "distance":
        fields = {"first_point_id", "second_point_id", "value"}
        if set(data) != fields:
            raise ValueError("constraint fields do not match distance")
        return SketchDistanceDimension(
            constraint_id,
            data["first_point_id"],
            data["second_point_id"],
            data["value"],
            driving,
            source="manual",
            enabled=enabled,
        )
    if kind == "radius":
        if set(data) != {"curve_id", "value"}:
            raise ValueError("constraint fields do not match radius")
        return SketchRadiusDimension(
            constraint_id,
            data["curve_id"],
            data["value"],
            driving,
            source="manual",
            enabled=enabled,
        )
    if kind == "angle":
        fields = {"first_line_id", "second_line_id", "angle_degrees"}
        if set(data) != fields:
            raise ValueError("constraint fields do not match angle")
        return SketchAngleDimension(
            constraint_id,
            data["first_line_id"],
            data["second_line_id"],
            math.radians(_finite(data["angle_degrees"], "angle_degrees")),
            driving,
            source="manual",
            enabled=enabled,
        )
    raise ValueError("constraint kind is unsupported")


def _solve_planar_candidate(
    sketch: SketchGeometry,
    *,
    fixed_point_ids: set[str] | tuple[str, ...] = (),
    new_constraint_ids: tuple[str, ...] = (),
) -> SketchGeometry:
    result = solve_sketch_constraints(
        sketch,
        fixed_point_ids=fixed_point_ids,
        previous_solution=sketch,
        new_constraint_ids=new_constraint_ids,
    )
    if not result.succeeded:
        conflicts = ", ".join(result.conflicting_constraint_ids) or "none"
        raise AuthoringContractError(
            f"sketch constraint solve {result.status}; conflict IDs: {conflicts}"
        )
    return SketchGeometry(
        sketch.name,
        sketch.plane,
        result.points,
        result.curves,
        sketch.constraints,
    )


def _exact_id_list(
    values: Sequence[str],
    label: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be an array")
    result = tuple(values)
    if not 1 <= len(result) <= maximum:
        raise ValueError(f"{label} must contain between 1 and {maximum} IDs")
    if any(type(item) is not str or not item for item in result):
        raise TypeError(f"{label} must contain non-empty exact string IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique IDs")
    return result


def _exact_circle_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be an array")
    normalized = tuple(values)
    if not 1 <= len(normalized) <= 32:
        raise ValueError(f"{label} must contain between 1 and 32 IDs")
    if any(type(value) is not str or not value for value in normalized):
        raise TypeError(f"{label} must contain non-empty exact string IDs")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must contain unique IDs")
    return normalized


def _curve_point_ids(curve: SketchLine | SketchArc | SketchCircle) -> tuple[str, ...]:
    if isinstance(curve, SketchLine):
        return curve.start_point_id, curve.end_point_id
    if isinstance(curve, SketchArc):
        return curve.start_point_id, curve.center_point_id, curve.end_point_id
    if curve.center_point_id is None:
        return ()
    return (curve.center_point_id,)


def _planar_edit_draft(sketch: SketchGeometry) -> GeometryDraft:
    draft = _draft(
        sketch,
        {
            "point_count": float(len(sketch.points)),
            "curve_count": float(len(sketch.curves)),
        },
    )
    assert draft.proof is not None
    if not draft.proof.exact:
        analysis = analyze_sketch_profiles(sketch)
        diagnostics = [
            {
                "code": item.code,
                "message": item.message,
                "affected_logical_ids": list(item.affected_ids),
                "severity": item.severity,
            }
            for item in analysis.blocking_diagnostics[:16]
        ]
        if not diagnostics:
            diagnostics = [
                {
                    "code": str(
                        item.get("diagnostic_id", "sketch.topology-unproven")
                    ),
                    "message": str(
                        item.get(
                            "message",
                            "Planar Profile topology could not be proven",
                        )
                    ),
                    "affected_logical_ids": list(
                        item.get("affected_logical_ids", [])
                    ),
                    "severity": "error",
                }
                for item in draft.proof.diagnostics[:16]
            ]
        raise PlanarEditValidationError(diagnostics)
    return draft


def planar_geometry_catalog(recipe: object) -> dict[str, object]:
    """Return a bounded editable projection without exposing CAD internals."""

    sketch = _as_strict_planar_sketch(recipe)
    point_by_id = {point.id: point for point in sketch.points}
    curves: list[dict[str, object]] = []
    for curve in sketch.curves:
        if isinstance(curve, SketchLine):
            curves.append(
                {
                    "kind": "line",
                    "id": curve.id,
                    "start_point_id": curve.start_point_id,
                    "end_point_id": curve.end_point_id,
                }
            )
        elif isinstance(curve, SketchArc):
            curves.append(
                {
                    "kind": "arc",
                    "id": curve.id,
                    "start_point_id": curve.start_point_id,
                    "center_point_id": curve.center_point_id,
                    "end_point_id": curve.end_point_id,
                    "orientation": curve.orientation,
                }
            )
        elif curve.center_point_id is not None:
            center = point_by_id[curve.center_point_id]
            curves.append(
                {
                    "kind": "circle",
                    "id": curve.id,
                    "center_point_id": curve.center_point_id,
                    "center_x": center.u,
                    "center_y": center.v,
                    "radius": curve.radius,
                }
            )
    solved = solve_sketch_constraints(sketch)
    return {
        "kind": "planar_sketch",
        "point_count": len(sketch.points),
        "curve_count": len(sketch.curves),
        "constraint_summary": _sketch_constraint_summary(sketch),
        "constraints": [
            _sketch_constraint_catalog_item(item)
            for item in sketch.constraints
        ],
        "solve": {
            "status": solved.status,
            "remaining_dof": solved.remaining_dof,
            "max_residual": solved.max_residual,
            "redundant_constraint_ids": list(solved.redundant_constraint_ids),
            "conflicting_constraint_ids": list(solved.conflicting_constraint_ids),
        },
        "points": [
            {"id": point.id, "x": point.u, "y": point.v}
            for point in sketch.points
        ],
        "curves": curves,
    }


def profile_transform_context(
    recipe: object,
    *,
    part_id: str | None = None,
    session_revision: int | None = None,
) -> dict[str, object]:
    """Return a bounded, canonical Profile-transform read projection.

    The projection is deliberately derived from the native recipe and the
    exact topology/profile analyser.  It contains no OCC/Gmsh tags and never
    exposes the complete sketch payload.  Dedicated GUI tools use this helper
    before dispatching any transform proposal.
    """

    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError("recipe must be native geometry")
    if session_revision is not None and (
        type(session_revision) is not int or session_revision < 0
    ):
        raise ValueError("session_revision must be a non-negative integer")

    dimension = geometry_dimension(recipe)
    native_recipe_kind = type(recipe).__name__
    recipe_kind = {
        "SketchGeometry": "planar_sketch",
        "PlateWithHoleGeometry": "planar_sketch",
        "RectangleGeometry": "planar_profile",
        "DiskGeometry": "planar_profile",
        "WireGeometry": "wire",
        "BoxGeometry": "solid_primitive",
        "CylinderGeometry": "solid_primitive",
        "MovedGeometry": "transformed_geometry",
        "RotatedGeometry": "transformed_geometry",
    }.get(native_recipe_kind, "native_geometry")
    # Legacy primitive recipes are still persisted in existing projects.  The
    # Profile-transform contract is defined over the strict sketch graph, so
    # canonicalize those bounded planar primitives before deriving topology or
    # Profile IDs.  Keep the original/native kind in the read payload so the
    # caller can distinguish a legacy source from an already strict sketch.
    analysis_recipe = recipe
    if type(recipe) in {
        RectangleGeometry,
        DiskGeometry,
        PlateWithHoleGeometry,
    }:
        analysis_recipe = _as_strict_planar_sketch(recipe)

    topology_exact = False
    topology_diagnostics: list[dict[str, object]] = []
    material_profiles: list[dict[str, object]] = []
    material_profile_total = 0
    hole_count = 0
    strict_planar = (
        type(analysis_recipe) is SketchGeometry and analysis_recipe.is_strict
    )

    try:
        topology = describe_recipe_topology(analysis_recipe)
    except (TypeError, ValueError):
        topology = None
    if topology is not None:
        topology_exact = bool(topology.exact)
        topology_diagnostics.extend(
            {
                "code": item.code,
                "message": str(item.message).strip()[:512],
                "affected_logical_ids": list(item.affected_logical_ids)[:16],
            }
            for item in topology.diagnostics[:16]
        )

    analysis = None
    if strict_planar:
        try:
            analysis = analyze_sketch_profiles(analysis_recipe)
        except (TypeError, ValueError):
            analysis = None
        if analysis is not None:
            hole_count = sum(
                1 for profile in analysis.profiles if profile.role == "hole"
            )
            if not any(
                diagnostic.blocking for diagnostic in analysis.diagnostics
            ):
                material = tuple(
                    profile
                    for profile in analysis.profiles
                    if profile.role == "outer"
                )
                material_profile_total = len(material)
                for profile in material[:PROFILE_TRANSFORM_MAX_PROFILES]:
                    direct_holes = sum(
                        1
                        for candidate in analysis.profiles
                        if candidate.role == "hole"
                        and candidate.parent_profile_id == profile.id
                    )
                    material_profiles.append(
                        {
                            "face_id": f"face:{profile.id}",
                            "canonical_face_id": f"face:{profile.id}",
                            "profile_id": profile.id,
                            "semantic_role": "sketch.profile",
                            "semantic_summary": (
                                f"material Profile {profile.id}; "
                                f"holes={direct_holes}"
                            ),
                            "curve_count": len(profile.curve_ids),
                            "hole_count": direct_holes,
                            "nesting_depth": profile.nesting_depth,
                            "area": abs(float(profile.signed_area)),
                            "bounding_box": list(profile.bounding_box),
                        }
                    )

    blocking_reason: str | None = None
    blocking_code: str | None = None
    if dimension != 2:
        blocking_code = "profile-transform.source-not-planar"
        blocking_reason = "Profile transform requires a two-dimensional Part"
    elif not strict_planar:
        blocking_code = "profile-transform.source-not-strict"
        blocking_reason = "Profile transform requires a strict planar sketch"
    elif analysis is None or any(
        diagnostic.blocking for diagnostic in analysis.diagnostics
    ):
        blocking_code = "profile-transform.topology-unproven"
        if analysis is not None and analysis.blocking_diagnostics:
            blocking_reason = str(
                analysis.blocking_diagnostics[0].message
            ).strip()[:512]
        else:
            blocking_reason = "Planar Profile topology could not be proven"
    elif not topology_exact:
        blocking_code = "profile-transform.topology-unproven"
        blocking_reason = "Planar Profile topology is not exact"
    elif not material_profiles:
        blocking_code = "profile-transform.no-material-profile"
        blocking_reason = "The Part has no material Profile"

    available = blocking_reason is None
    operation_defaults: dict[str, object] = {
        "axis": "z",
        "angle_degrees": 360.0,
        "frame_strategy": "transport",
        "allowed_frame_strategies": ["fixed", "transport"],
    }
    operations = {
        "extrusion": {
            "available": available,
            "blocking_reason": blocking_reason,
            "blocking_code": blocking_code,
            "requires_unique_or_explicit_profile": True,
        },
        "revolution": {
            "available": available,
            "blocking_reason": blocking_reason,
            "blocking_code": blocking_code,
            "defaults": {
                "axis": operation_defaults["axis"],
                "angle_degrees": operation_defaults["angle_degrees"],
            },
        },
        "path_sweep": {
            "available": available,
            "blocking_reason": blocking_reason,
            "blocking_code": blocking_code,
            "defaults": {
                "frame_strategy": operation_defaults["frame_strategy"],
                "allowed_frame_strategies": operation_defaults[
                    "allowed_frame_strategies"
                ],
            },
        },
    }
    diagnostics = list(topology_diagnostics)
    if blocking_reason is not None and not diagnostics:
        diagnostics.append(
            {
                "code": blocking_code,
                "message": blocking_reason,
                "affected_logical_ids": [],
            }
        )
    profile_count = material_profile_total
    data: dict[str, object] = {
        "kind": "profile_transform_context",
        "schema_version": PROFILE_TRANSFORM_CONTEXT_SCHEMA_VERSION,
        "part_id": part_id,
        "dimension": dimension,
        "recipe_kind": recipe_kind,
        "native_recipe_kind": native_recipe_kind,
        "session_revision": session_revision,
        "revision": session_revision,
        "topology_exact": topology_exact,
        "material_profile_count": profile_count,
        "profiles": material_profiles,
        "material_profiles": material_profiles,
        "hole_count": hole_count,
        "operations": operations,
        "extrusion": operations["extrusion"],
        "revolution": operations["revolution"],
        "path_sweep": operations["path_sweep"],
        "diagnostics": diagnostics[:16],
        "truncated": False,
    }
    # Keep read results comfortably below the common provider payload budget.
    # Profiles are the only potentially unbounded collection and are trimmed
    # deterministically while preserving count and a truncation marker.
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    omitted = max(0, material_profile_total - len(material_profiles))
    while len(encoded) > PROFILE_TRANSFORM_MAX_BYTES and data["profiles"]:
        profiles = list(data["profiles"])
        profiles.pop()
        data["profiles"] = profiles
        data["material_profiles"] = profiles
        omitted += 1
        encoded = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    if omitted:
        data["truncated"] = True
        data["omitted_profile_count"] = omitted
    return data


def feature_topology_catalog(
    recipe: object,
    *,
    part_id: str | None = None,
) -> dict[str, object]:
    """Return a bounded read-only feature and logical-topology projection."""

    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError("recipe must be native geometry")
    topology = describe_recipe_topology(recipe)
    features = derive_feature_history(recipe)
    if (
        len(features) > _MAX_RECIPE_NODES
        or len(topology.entities) > _MAX_RECIPE_NODES
    ):
        raise ValueError("feature/topology catalog exceeds the bounded contract")
    catalog = {
        "kind": "native_feature_topology_catalog",
        "schema_version": _RECIPE_SCHEMA_VERSION,
        "part_id": part_id,
        "recipe_type": type(recipe).__name__,
        "dimension": topology.dimension,
        "exact": topology.exact,
        "features": [
            {
                "feature_id": record.name,
                "kind": record.kind,
                "summary": record.payload.get("summary"),
            }
            for record in features
        ],
        "entities": [
            {
                "kind": entity.kind,
                "logical_id": entity.logical_id,
                "semantic_role": entity.semantic_role,
                "selectable": entity.selectable,
                "topology_links": list(entity.topology_links),
            }
            for entity in topology.entities
        ],
        "diagnostics": [
            {
                "diagnostic_id": item.code,
                "message": item.message,
                "affected_logical_ids": list(item.affected_logical_ids),
            }
            for item in topology.diagnostics[:32]
        ],
    }
    if isinstance(recipe, MultiBodyGeometry):
        catalog["canonical_part_ownership"] = [
            {"body_id": body.id, "part_id": f"P{body.id[1:]}"}
            for body in recipe.bodies
        ]
    if any(record.kind.startswith("face_sketch_boolean_") for record in features):
        catalog["face_sketch_boolean_capability"] = {
            "read": True,
            "create": False,
            "edit": False,
            "message": "Agent 可只读识别面草图拉伸布尔特征，暂不支持创建或编辑。",
        }
    return catalog


def geometry_feature_catalog_tool_schema() -> dict[str, object]:
    """Return the strict no-argument schema for the local catalog read."""

    return {
        "name": GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
        "description": (
            "Read the bounded native Part feature and logical-topology catalog. "
            "This local read does not require a current mesh and exposes no CAD tags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }


def geometry_contract_proof(recipe: object) -> GeometryContractProof:
    """Perform one detached, bounded recipe/topology preflight."""

    catalog = feature_topology_catalog(recipe)
    fingerprint = topology_fingerprint_for_recipe(recipe)
    fingerprint_payload = {
        "contract": fingerprint.contract,
        "dimension": fingerprint.dimension,
        "exact": fingerprint.exact,
        "entities": [
            {
                "kind": entity.kind,
                "logical_id": entity.logical_id,
                "semantic_role": entity.semantic_role,
                "selectable": entity.selectable,
                "topology_links": list(entity.topology_links),
            }
            for entity in fingerprint.entities
        ],
    }
    bodies = sum(
        item["kind"] == "body" and item["selectable"]
        for item in catalog["entities"]
    )
    return GeometryContractProof(
        exact=bool(catalog["exact"]),
        topology_fingerprint=fingerprint_payload,
        expected_body_count=bodies,
        diagnostics=tuple(catalog["diagnostics"]),
    )


def geometry_recipe_to_payload(recipe: object) -> dict[str, object]:
    payload = _geometry_recipe_to_payload(recipe)
    payload = {"schema_version": _RECIPE_SCHEMA_VERSION, **payload}
    _validate_recipe_payload_budget(payload)
    return payload


def _geometry_recipe_to_payload(recipe: object) -> dict[str, object]:
    if type(recipe) is WireGeometry:
        return {
            "kind": "wire",
            "name": recipe.name,
            "points": [
                {
                    "name": point.name,
                    "x": point.x,
                    "y": point.y,
                    "z": point.z,
                }
                for point in recipe.points
            ],
            "members": [
                {
                    "name": member.name,
                    "start": member.start,
                    "end": member.end,
                }
                for member in recipe.members
            ],
        }
    if type(recipe) is RectangleGeometry:
        return {
            "kind": "rectangle",
            "name": recipe.name,
            "width": recipe.width,
            "height": recipe.height,
        }
    if type(recipe) is DiskGeometry:
        return {"kind": "disk", "name": recipe.name, "radius": recipe.radius}
    if type(recipe) is PlateWithHoleGeometry:
        return {
            "kind": "plate_with_hole",
            "name": recipe.name,
            "width": recipe.width,
            "height": recipe.height,
            "hole_x": recipe.hole_x,
            "hole_y": recipe.hole_y,
            "hole_radius": recipe.hole_radius,
        }
    if type(recipe) is BoxGeometry:
        return {
            "kind": "box",
            "name": recipe.name,
            "width": recipe.width,
            "depth": recipe.depth,
            "height": recipe.height,
        }
    if type(recipe) is CylinderGeometry:
        return {
            "kind": "cylinder",
            "name": recipe.name,
            "radius": recipe.radius,
            "height": recipe.height,
        }
    if type(recipe) is MovedGeometry:
        return {
            "kind": "translated",
            "base": _geometry_recipe_to_payload(recipe.base),
            "dx": recipe.dx,
            "dy": recipe.dy,
            "dz": recipe.dz,
        }
    if type(recipe) is RotatedGeometry:
        return {
            "kind": "rotated",
            "base": _geometry_recipe_to_payload(recipe.base),
            "axis": recipe.axis,
            "angle_degrees": recipe.angle_degrees,
        }
    if type(recipe) is ExtrudedGeometry:
        return {
            "kind": "extruded",
            "base": _geometry_recipe_to_payload(recipe.base),
            "height": recipe.height,
            "source_face_ids": list(recipe.source_face_ids),
        }
    if type(recipe) is RevolvedGeometry:
        return {
            "kind": "revolved",
            "base": _geometry_recipe_to_payload(recipe.base),
            "axis": recipe.axis,
            "angle_degrees": recipe.angle_degrees,
            "source_face_ids": list(recipe.source_face_ids),
        }
    if type(recipe) is PathSweptGeometry:
        return {
            "kind": "path_swept",
            "base": _geometry_recipe_to_payload(recipe.base),
            "path": _geometry_recipe_to_payload(recipe.path),
            "source_face_ids": list(recipe.source_face_ids),
            "frame_strategy": recipe.frame_strategy,
        }
    if type(recipe) is BooleanGeometry:
        return {
            "kind": "boolean",
            "name": recipe.name,
            "operation": recipe.operation,
            "object": _geometry_recipe_to_payload(recipe.object_geometry),
            "tool": _geometry_recipe_to_payload(recipe.tool_geometry),
            "body_context": _boolean_context_to_payload(recipe.body_context),
            "planar_context": _boolean_context_to_payload(recipe.planar_context),
            "part_context": _boolean_context_to_payload(recipe.part_context),
        }
    if type(recipe) is FaceSketchBooleanGeometry:
        return {
            "kind": "face_sketch_boolean",
            "feature_id": recipe.feature_id,
            "name": recipe.name,
            "base": _geometry_recipe_to_payload(recipe.base),
            "support_face_id": recipe.support_face_id,
            "workplane_strategy": {
                "seed_axis": recipe.workplane_strategy.seed_axis,
                "sign": recipe.workplane_strategy.sign,
                "origin_rule": recipe.workplane_strategy.origin_rule,
            },
            "sketch": _geometry_recipe_to_payload(recipe.sketch),
            "operation": recipe.operation.value,
            "direction": recipe.direction.value,
            "distance": recipe.distance,
            "participating_profile_ids": list(recipe.participating_profile_ids),
            "external_reference_count": len(recipe.external_references),
            "step_proof_count": len(recipe.step_proofs),
            "capability": {
                "read": True,
                "create": False,
                "edit": False,
                "message": "Agent 可只读识别面草图拉伸布尔特征，暂不支持创建或编辑。",
            },
        }
    if type(recipe) is MultiBodyGeometry:
        return {
            "kind": "multi_body",
            "name": recipe.name,
            "bodies": [
                {
                    "id": body.id,
                    "name": body.name,
                    "recipe": _geometry_recipe_to_payload(body.recipe),
                }
                for body in recipe.bodies
            ],
            "retired_body_ids": list(recipe.retired_body_ids),
            "retired_boolean_feature_ids": list(
                recipe.retired_boolean_feature_ids
            ),
        }
    if type(recipe) is SketchGeometry and recipe.is_strict:
        if len(recipe.constraints) > 128:
            raise ValueError("planar sketch constraints exceed the bounded schema")
        return {
            "kind": "planar_sketch",
            "name": recipe.name,
            "plane": {
                "origin": list(recipe.plane.origin),
                "x_direction": list(recipe.plane.x_direction),
                "y_direction": list(recipe.plane.y_direction),
            },
            "points": [
                {"id": point.id, "u": point.u, "v": point.v}
                for point in recipe.points
            ],
            "curves": [
                _sketch_curve_to_payload(curve)
                for curve in recipe.curves
            ],
            "constraints": [
                _sketch_constraint_to_payload(constraint)
                for constraint in recipe.constraints
            ],
        }
    raise TypeError("recipe is outside the native Agent geometry subset")


def geometry_recipe_from_payload(value: object) -> object:
    if not isinstance(value, Mapping):
        raise TypeError("geometry recipe payload must be an object")
    payload = dict(value)
    version = payload.pop("schema_version", None)
    if version is not None and version != _RECIPE_SCHEMA_VERSION:
        raise ValueError("unknown geometry recipe schema_version")
    if version is None and payload.get("kind") in {
        "extruded",
        "revolved",
        "path_swept",
        "boolean",
        "multi_body",
    }:
        raise ValueError("3D feature payload requires schema_version")
    _validate_recipe_payload_budget(value)
    return _geometry_recipe_from_payload(payload)


def _geometry_recipe_from_payload(value: object) -> object:
    if not isinstance(value, Mapping):
        raise TypeError("geometry recipe payload must be an object")
    kind = value.get("kind")
    if kind == "face_sketch_boolean":
        raise ValueError(
            "Agent 暂不支持创建或编辑面草图拉伸布尔特征；仅支持只读识别。"
        )
    fields: dict[str, set[str]] = {
        "rectangle": {"kind", "name", "width", "height"},
        "disk": {"kind", "name", "radius"},
        "plate_with_hole": {
            "kind",
            "name",
            "width",
            "height",
            "hole_x",
            "hole_y",
            "hole_radius",
        },
        "box": {"kind", "name", "width", "depth", "height"},
        "cylinder": {"kind", "name", "radius", "height"},
        "translated": {"kind", "base", "dx", "dy", "dz"},
        "rotated": {"kind", "base", "axis", "angle_degrees"},
        "planar_sketch": {
            "kind", "name", "plane", "points", "curves", "constraints",
        },
        "wire": {"kind", "name", "points", "members"},
        "extruded": {"kind", "base", "height", "source_face_ids"},
        "revolved": {
            "kind", "base", "axis", "angle_degrees", "source_face_ids"
        },
        "path_swept": {
            "kind", "base", "path", "source_face_ids", "frame_strategy"
        },
        "boolean": {
            "kind", "name", "operation", "object", "tool",
            "body_context", "planar_context", "part_context",
        },
        "multi_body": {
            "kind", "name", "bodies", "retired_body_ids",
            "retired_boolean_feature_ids",
        },
    }
    if kind not in fields or set(value) != fields[kind]:  # type: ignore[index]
        raise ValueError("geometry recipe fields do not match the A2 schema")
    if kind == "wire":
        points = value["points"]
        members = value["members"]
        if (
            not isinstance(points, list)
            or not isinstance(members, list)
            or not 2 <= len(points) <= _MAX_PREVIEW_POINTS
            or not 1 <= len(members) <= _MAX_PREVIEW_POINTS
        ):
            raise ValueError("wire entities exceed the bounded schema")
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "x", "y", "z"}
            for item in points
        ):
            raise ValueError("wire point fields do not match")
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "start", "end"}
            for item in members
        ):
            raise ValueError("wire member fields do not match")
        return WireGeometry(
            value["name"],
            tuple(
                WirePoint(item["name"], item["x"], item["y"], item["z"])
                for item in points
            ),
            tuple(
                WireMember(item["name"], item["start"], item["end"])
                for item in members
            ),
        )
    if kind == "rectangle":
        return RectangleGeometry(value["name"], value["width"], value["height"])
    if kind == "disk":
        return DiskGeometry(value["name"], value["radius"])
    if kind == "plate_with_hole":
        return PlateWithHoleGeometry(
            value["name"],
            value["width"],
            value["height"],
            value["hole_x"],
            value["hole_y"],
            value["hole_radius"],
        )
    if kind == "box":
        return BoxGeometry(
            value["name"],
            value["width"],
            value["depth"],
            value["height"],
        )
    if kind == "cylinder":
        return CylinderGeometry(
            value["name"],
            value["radius"],
            value["height"],
        )
    if kind == "extruded":
        return ExtrudedGeometry(
            _geometry_recipe_from_payload(value["base"]),
            value["height"],
            _logical_id_list(value["source_face_ids"], "source_face_ids"),
        )
    if kind == "revolved":
        return RevolvedGeometry(
            _geometry_recipe_from_payload(value["base"]),
            value["axis"],
            value["angle_degrees"],
            _logical_id_list(value["source_face_ids"], "source_face_ids"),
        )
    if kind == "path_swept":
        path = _geometry_recipe_from_payload(value["path"])
        if type(path) is not WireGeometry:
            raise ValueError("path_swept.path must decode to WireGeometry")
        return PathSweptGeometry(
            _geometry_recipe_from_payload(value["base"]),
            path,
            _logical_id_list(value["source_face_ids"], "source_face_ids"),
            value["frame_strategy"],
        )
    if kind == "boolean":
        return BooleanGeometry(
            value["name"],
            value["operation"],
            _geometry_recipe_from_payload(value["object"]),
            _geometry_recipe_from_payload(value["tool"]),
            _boolean_context_from_payload(value["body_context"], "body"),
            _boolean_context_from_payload(value["planar_context"], "planar"),
            _boolean_context_from_payload(value["part_context"], "part"),
        )
    if kind == "multi_body":
        bodies = value["bodies"]
        if not isinstance(bodies, list) or not 1 <= len(bodies) <= _MAX_RECIPE_NODES:
            raise ValueError("multi_body bodies exceed the bounded schema")
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"id", "name", "recipe"}
            for item in bodies
        ):
            raise ValueError("multi_body Body fields do not match")
        return MultiBodyGeometry(
            value["name"],
            tuple(
                SolidBody(
                    item["id"],
                    item["name"],
                    _geometry_recipe_from_payload(item["recipe"]),
                )
                for item in bodies
            ),
            _string_list(value["retired_body_ids"], "retired_body_ids"),
            _string_list(
                value["retired_boolean_feature_ids"],
                "retired_boolean_feature_ids",
            ),
        )
    if kind == "translated":
        return MovedGeometry(
            _geometry_recipe_from_payload(value["base"]),
            value["dx"],
            value["dy"],
            value["dz"],
        )
    if kind == "planar_sketch":
        plane = value["plane"]
        if not isinstance(plane, Mapping) or set(plane) != {
            "origin",
            "x_direction",
            "y_direction",
        }:
            raise ValueError("planar sketch plane fields do not match")
        points = value["points"]
        curves = value["curves"]
        constraints = value["constraints"]
        if (
            not isinstance(points, list)
            or not isinstance(curves, list)
            or not isinstance(constraints, list)
            or not 1 <= len(points) <= 128
            or not 1 <= len(curves) <= 128
            or len(constraints) > 128
        ):
            raise ValueError("planar sketch entities exceed the bounded schema")
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"id", "u", "v"}
            for item in points
        ):
            raise ValueError("planar sketch point fields do not match")
        return SketchGeometry(
            value["name"],
            SketchPlane(
                tuple(plane["origin"]),
                tuple(plane["x_direction"]),
                tuple(plane["y_direction"]),
            ),
            tuple(
                SketchPoint(item["id"], item["u"], item["v"])
                for item in points
            ),
            tuple(_sketch_curve_from_payload(item) for item in curves),
            tuple(
                _sketch_constraint_from_payload(item)
                for item in constraints
            ),
        )
    return RotatedGeometry(
        _geometry_recipe_from_payload(value["base"]),
        value["axis"],
        value["angle_degrees"],
    )


def _validate_recipe_payload_budget(value: object) -> None:
    try:
        size = len(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("geometry recipe payload must be finite JSON") from error
    if size > _MAX_RECIPE_BYTES:
        raise ValueError("geometry recipe payload exceeds the byte budget")
    nodes = 0
    root_kind = value.get("kind") if isinstance(value, Mapping) else None
    node_budget = (
        _MAX_BOOLEAN_RECIPE_PAYLOAD_NODES
        if root_kind in {"boolean", "multi_body"}
        else _MAX_RECIPE_NODES
    )

    def visit(item: object, depth: int, *, planar_entity: bool = False) -> None:
        nonlocal nodes
        if depth > _MAX_RECIPE_DEPTH:
            raise ValueError("geometry recipe payload exceeds the depth budget")
        if isinstance(item, Mapping):
            if not planar_entity:
                nodes += 1
                if nodes > node_budget:
                    raise ValueError("geometry recipe payload exceeds the node budget")
            sketch_fields = (
                {"points", "curves", "constraints"}
                if item.get("kind") == "planar_sketch"
                else set()
            )
            for key, child in item.items():
                visit(
                    child,
                    depth + 1,
                    planar_entity=key in sketch_fields,
                )
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1, planar_entity=planar_entity)

    visit(value, 0)


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > _MAX_RECIPE_NODES:
        raise ValueError(f"{field_name} must be a bounded array")
    if any(type(item) is not str for item in value):
        raise ValueError(f"{field_name} must contain strings")
    return tuple(value)


def _logical_id_list(value: object, field_name: str) -> tuple[str, ...]:
    return _string_list(value, field_name)


def _lineage_to_payload(
    entities: Sequence[BooleanLineageEntity],
    mappings: Sequence[BooleanLineageMapping],
) -> dict[str, object]:
    return {
        "result_entities": [
            {
                "kind": item.kind,
                "logical_id": item.logical_id,
                "semantic_role": item.semantic_role,
                "topology_links": list(item.topology_links),
            }
            for item in entities
        ],
        "topology_mappings": [
            {
                "source": item.source,
                "source_logical_id": item.source_logical_id,
                "target_logical_id": item.target_logical_id,
                "relation": item.relation,
            }
            for item in mappings
        ],
    }


def _boolean_context_to_payload(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    lineage = _lineage_to_payload(value.result_entities, value.topology_mappings)
    if type(value) is BooleanBodyContext:
        return {
            "feature_id": value.feature_id,
            "target_body_id": value.target_body_id,
            "tool_body_id": value.tool_body_id,
            "tool_body_name": value.tool_body_name,
            **lineage,
        }
    if type(value) is PlanarBooleanContext:
        return {
            "feature_id": value.feature_id,
            "target_face_id": value.target_face_id,
            "tool_face_ids": list(value.tool_face_ids),
            **lineage,
        }
    if type(value) is PartBooleanContext:
        return {
            "feature_id": value.feature_id,
            "target_part_id": value.target_part_id,
            "tool_part_id": value.tool_part_id,
            "result_part_id": value.result_part_id,
            **lineage,
        }
    raise TypeError("unknown Boolean context type")


def _boolean_context_from_payload(value: object, kind: str) -> object:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind}_context must be an object or null")
    common = {"feature_id", "result_entities", "topology_mappings"}
    specific = {
        "body": {"target_body_id", "tool_body_id", "tool_body_name"},
        "planar": {"target_face_id", "tool_face_ids"},
        "part": {"target_part_id", "tool_part_id", "result_part_id"},
    }[kind]
    if set(value) != common | specific:
        raise ValueError(f"{kind}_context fields do not match")
    raw_entities = value["result_entities"]
    raw_mappings = value["topology_mappings"]
    if not isinstance(raw_entities, list) or not isinstance(raw_mappings, list):
        raise ValueError("Boolean lineage must use arrays")
    if len(raw_entities) > 512 or len(raw_mappings) > 1024:
        raise ValueError("Boolean lineage exceeds the bounded schema")
    if any(
        not isinstance(item, Mapping)
        or set(item) != {"kind", "logical_id", "semantic_role", "topology_links"}
        for item in raw_entities
    ):
        raise ValueError("Boolean lineage entity fields do not match")
    if any(
        not isinstance(item, Mapping)
        or set(item) != {
            "source", "source_logical_id", "target_logical_id", "relation"
        }
        for item in raw_mappings
    ):
        raise ValueError("Boolean lineage mapping fields do not match")
    entities = tuple(
        BooleanLineageEntity(
            item["kind"],
            item["logical_id"],
            item["semantic_role"],
            _string_list(item["topology_links"], "topology_links"),
        )
        for item in raw_entities
    )
    mappings = tuple(
        BooleanLineageMapping(
            item["source"],
            item["source_logical_id"],
            item["target_logical_id"],
            item["relation"],
        )
        for item in raw_mappings
    )
    if kind == "body":
        return BooleanBodyContext(
            value["feature_id"], value["target_body_id"],
            value["tool_body_id"], value["tool_body_name"], entities, mappings,
        )
    if kind == "planar":
        return PlanarBooleanContext(
            value["feature_id"], value["target_face_id"],
            _string_list(value["tool_face_ids"], "tool_face_ids"),
            entities, mappings,
        )
    return PartBooleanContext(
        value["feature_id"], value["target_part_id"], value["tool_part_id"],
        value["result_part_id"], entities, mappings,
    )


def _draft(
    recipe: object,
    key_dimensions: Mapping[str, float],
    transforms: tuple[Mapping[str, object], ...] = (),
) -> GeometryDraft:
    return GeometryDraft(
        recipe=recipe,
        recipe_payload=geometry_recipe_to_payload(recipe),
        preview=_preview(recipe),
        key_dimensions=key_dimensions,
        transforms=transforms,
    )


def geometry_draft(recipe: object) -> GeometryDraft:
    """Wrap one accepted native recipe for a subsequent generic transform."""

    dimensions: dict[str, float] = {}
    if isinstance(recipe, ExtrudedGeometry):
        dimensions["height"] = recipe.height
    elif isinstance(recipe, RevolvedGeometry):
        dimensions["angle_degrees"] = recipe.angle_degrees
    elif isinstance(recipe, PathSweptGeometry):
        dimensions["path_segment_count"] = float(len(recipe.path.members))
    return _draft(recipe, dimensions)


def _proposal_source(recipe: object) -> object:
    if isinstance(recipe, (ExtrudedGeometry, RevolvedGeometry, PathSweptGeometry)):
        return list(recipe.source_face_ids)
    if isinstance(recipe, BooleanGeometry):
        context = recipe.body_context or recipe.part_context or recipe.planar_context
        if isinstance(context, BooleanBodyContext):
            return {
                "target_body_id": context.target_body_id,
                "tool_body_id": context.tool_body_id,
            }
        if isinstance(context, PartBooleanContext):
            return {
                "target_part_id": context.target_part_id,
                "tool_part_id": context.tool_part_id,
            }
        if isinstance(context, PlanarBooleanContext):
            return {
                "target_face_id": context.target_face_id,
                "tool_face_ids": list(context.tool_face_ids),
            }
        return {
            "target": recipe.object_geometry.name,
            "tool": recipe.tool_geometry.name,
        }
    return recipe.name


def _proposal_operation(recipe: object) -> str:
    if isinstance(recipe, BooleanGeometry):
        return recipe.operation
    return {
        ExtrudedGeometry: "extrude",
        RevolvedGeometry: "revolve",
        PathSweptGeometry: "path_sweep",
        MultiBodyGeometry: "multi_body",
    }.get(type(recipe), "create")


def _preview(
    recipe: object,
    *,
    point_budget: int = _MAX_PREVIEW_POINTS,
) -> StaticGeometryPreview:
    if (
        isinstance(point_budget, bool)
        or not isinstance(point_budget, int)
        or not 1 <= point_budget <= _MAX_PREVIEW_POINTS
    ):
        raise ValueError("preview point budget is outside the A2 bound")
    if type(recipe) is MovedGeometry:
        base = _preview(recipe.base, point_budget=point_budget)
        points = tuple(
            (x + recipe.dx, y + recipe.dy, z + recipe.dz) for x, y, z in base.points
        )
        return _make_preview(
            base.dimension,
            points,
            base.lines,
            point_names=base.point_names,
            member_names=base.member_names,
        )
    if type(recipe) is RotatedGeometry:
        base = _preview(recipe.base, point_budget=point_budget)
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)

        def rotated(point: tuple[float, float, float]) -> tuple[float, float, float]:
            x, y, z = point
            if recipe.axis == "x":
                return x, y * cosine - z * sine, y * sine + z * cosine
            if recipe.axis == "y":
                return x * cosine + z * sine, y, -x * sine + z * cosine
            return x * cosine - y * sine, x * sine + y * cosine, z

        return _make_preview(
            base.dimension,
            tuple(map(rotated, base.points)),
            base.lines,
            point_names=base.point_names,
            member_names=base.member_names,
        )
    if type(recipe) is ExtrudedGeometry:
        base = _preview(
            recipe.base,
            point_budget=max(1, point_budget // 2),
        )
        count = len(base.points)
        if count * 2 > point_budget:
            raise ValueError("extruded preview exceeds the point budget")
        nx, ny, nz = planar_geometry_normal(recipe.base)
        points = base.points + tuple(
            (
                x + nx * recipe.height,
                y + ny * recipe.height,
                z + nz * recipe.height,
            )
            for x, y, z in base.points
        )
        lines = base.lines + tuple(
            tuple(index + count for index in line) for line in base.lines
        ) + tuple((index, index + count) for index in range(count))
        return _make_preview(3, points, lines)
    if type(recipe) is RevolvedGeometry:
        base = _preview(recipe.base)
        samples = min(4, max(2, _MAX_PREVIEW_POINTS // len(base.points)))
        previews = []
        for index in range(samples):
            angle = math.radians(
                recipe.angle_degrees * index / (samples - 1)
            )
            cosine, sine = math.cos(angle), math.sin(angle)

            def rotate(point: tuple[float, float, float]) -> tuple[float, float, float]:
                x, y, z = point
                if recipe.axis == "x":
                    return x, y * cosine - z * sine, y * sine + z * cosine
                if recipe.axis == "y":
                    return x * cosine + z * sine, y, -x * sine + z * cosine
                return x * cosine - y * sine, x * sine + y * cosine, z

            previews.append(
                _make_preview(
                    3,
                    tuple(rotate(point) for point in base.points),
                    base.lines,
                )
            )
        return _combine_previews(tuple(previews), 3)
    if type(recipe) is PathSweptGeometry:
        return _combine_previews((_preview(recipe.base), _preview(recipe.path)), 3)
    if type(recipe) is BooleanGeometry:
        return _combine_previews(
            (_preview(recipe.object_geometry), _preview(recipe.tool_geometry)),
            3,
        )
    if type(recipe) is MultiBodyGeometry:
        return _combine_previews(
            tuple(_preview(body.recipe) for body in recipe.bodies),
            3,
        )
    if type(recipe) is WireGeometry:
        point_indexes = {
            point.name: index for index, point in enumerate(recipe.points)
        }
        return _make_preview(
            1,
            tuple((point.x, point.y, point.z) for point in recipe.points),
            tuple(
                (point_indexes[member.start], point_indexes[member.end])
                for member in recipe.members
            ),
            point_names=tuple(point.name for point in recipe.points),
            member_names=tuple(member.name for member in recipe.members),
        )
    if type(recipe) is RectangleGeometry:
        points = (
            (0.0, 0.0, 0.0),
            (recipe.width, 0.0, 0.0),
            (recipe.width, recipe.height, 0.0),
            (0.0, recipe.height, 0.0),
        )
        return _make_preview(2, points, ((0, 1, 2, 3, 0),))
    if type(recipe) is DiskGeometry:
        points = _circle_points(0.0, 0.0, recipe.radius, 0.0)
        return _make_preview(2, points, (tuple(range(len(points))) + (0,),))
    if type(recipe) is PlateWithHoleGeometry:
        outer = (
            (0.0, 0.0, 0.0),
            (recipe.width, 0.0, 0.0),
            (recipe.width, recipe.height, 0.0),
            (0.0, recipe.height, 0.0),
        )
        hole = _circle_points(
            recipe.hole_x,
            recipe.hole_y,
            recipe.hole_radius,
            0.0,
        )
        return _make_preview(
            2,
            outer + hole,
            (
                (0, 1, 2, 3, 0),
                tuple(range(4, 4 + len(hole))) + (4,),
            ),
        )
    if type(recipe) is SketchGeometry:
        sketch = recipe if recipe.is_strict else legacy_sketch_to_strict(recipe)
        assert sketch.plane is not None
        points = tuple(
            sketch.plane.to_global(point.u, point.v)
            for point in sketch.points
        )
        if len(points) > point_budget:
            raise ValueError("sketch preview exceeds the point budget")
        point_indexes = {
            point.id: index for index, point in enumerate(sketch.points)
        }
        preview_points = list(points)
        lines: list[tuple[int, ...]] = []
        point_by_id = {point.id: point for point in sketch.points}
        curved_count = sum(
            isinstance(curve, (SketchCircle, SketchArc))
            for curve in sketch.curves
        )
        curved_samples = (
            0
            if curved_count == 0
            else min(
                _PREVIEW_SEGMENTS,
                max(
                    0,
                    (point_budget - len(preview_points))
                    // curved_count,
                ),
            )
        )
        for curve in sketch.curves:
            if isinstance(curve, SketchLine):
                lines.append(
                    (
                        point_indexes[curve.start_point_id],
                        point_indexes[curve.end_point_id],
                    )
                )
            elif isinstance(curve, SketchCircle):
                if curved_samples < 4:
                    continue
                center = point_by_id[curve.center_point_id]
                circle = tuple(
                    sketch.plane.to_global(
                        center.u
                        + curve.radius
                        * math.cos(2.0 * math.pi * index / curved_samples),
                        center.v
                        + curve.radius
                        * math.sin(2.0 * math.pi * index / curved_samples),
                    )
                    for index in range(curved_samples)
                )
                start = len(preview_points)
                preview_points.extend(circle)
                lines.append(
                    tuple(range(start, start + len(circle))) + (start,)
                )
            elif isinstance(curve, SketchArc):
                if curved_samples < 2:
                    continue
                start_point = point_by_id[curve.start_point_id]
                center = point_by_id[curve.center_point_id]
                end_point = point_by_id[curve.end_point_id]
                start_angle = math.atan2(
                    start_point.v - center.v,
                    start_point.u - center.u,
                )
                end_angle = math.atan2(
                    end_point.v - center.v,
                    end_point.u - center.u,
                )
                sweep = end_angle - start_angle
                if curve.orientation == "ccw" and sweep <= 0.0:
                    sweep += 2.0 * math.pi
                elif curve.orientation == "cw" and sweep >= 0.0:
                    sweep -= 2.0 * math.pi
                radius = math.hypot(
                    start_point.u - center.u,
                    start_point.v - center.v,
                )
                arc = tuple(
                    sketch.plane.to_global(
                        center.u
                        + radius
                        * math.cos(
                            start_angle
                            + sweep * index / (curved_samples - 1)
                        ),
                        center.v
                        + radius
                        * math.sin(
                            start_angle
                            + sweep * index / (curved_samples - 1)
                        ),
                    )
                    for index in range(curved_samples)
                )
                start = len(preview_points)
                preview_points.extend(arc)
                lines.append(tuple(range(start, start + len(arc))))
        return _make_preview(
            2,
            tuple(preview_points),
            tuple(lines),
        )
    if type(recipe) is BoxGeometry:
        points = tuple(
            (x, y, z)
            for z in (0.0, recipe.height)
            for y in (0.0, recipe.depth)
            for x in (0.0, recipe.width)
        )
        lines = (
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        return _make_preview(3, points, lines)
    if type(recipe) is CylinderGeometry:
        bottom = _circle_points(0.0, 0.0, recipe.radius, 0.0)
        top = _circle_points(0.0, 0.0, recipe.radius, recipe.height)
        count = len(bottom)
        lines = (
            tuple(range(count)) + (0,),
            tuple(range(count, count * 2)) + (count,),
            (0, count),
            (count // 4, count + count // 4),
            (count // 2, count + count // 2),
            (3 * count // 4, count + 3 * count // 4),
        )
        return _make_preview(3, bottom + top, lines)
    raise TypeError("recipe is outside the A2 preview subset")


def _circle_points(
    center_x: float,
    center_y: float,
    radius: float,
    z: float,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            center_x + radius * math.cos(2.0 * math.pi * index / _PREVIEW_SEGMENTS),
            center_y + radius * math.sin(2.0 * math.pi * index / _PREVIEW_SEGMENTS),
            z,
        )
        for index in range(_PREVIEW_SEGMENTS)
    )


def _combine_previews(
    previews: tuple[StaticGeometryPreview, ...],
    dimension: int,
) -> StaticGeometryPreview:
    points: list[tuple[float, float, float]] = []
    lines: list[tuple[int, ...]] = []
    for preview in previews:
        if len(points) + len(preview.points) > _MAX_PREVIEW_POINTS:
            raise ValueError("combined preview exceeds the point budget")
        offset = len(points)
        points.extend(preview.points)
        lines.extend(
            tuple(index + offset for index in line)
            for line in preview.lines
        )
    return _make_preview(dimension, tuple(points), tuple(lines))


def _make_preview(
    dimension: int,
    points: tuple[tuple[float, float, float], ...],
    lines: tuple[tuple[int, ...], ...],
    *,
    point_names: tuple[str, ...] = (),
    member_names: tuple[str, ...] = (),
) -> StaticGeometryPreview:
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    zs = tuple(point[2] for point in points)
    return StaticGeometryPreview(
        dimension=dimension,
        points=points,
        lines=lines,
        bounds=(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)),
        point_names=point_names,
        member_names=member_names,
    )


def _as_strict_planar_sketch(recipe: object) -> SketchGeometry:
    if type(recipe) is SketchGeometry:
        return recipe if recipe.is_strict else legacy_sketch_to_strict(recipe)
    if type(recipe) is RectangleGeometry:
        return planar_sketch_geometry(
            recipe.name,
            contours=(
                SketchRectangle(
                    "material",
                    0.0,
                    0.0,
                    recipe.width,
                    recipe.height,
                ),
            ),
        ).recipe
    if type(recipe) is DiskGeometry:
        return planar_sketch_geometry(
            recipe.name,
            contours=(SketchCircle("material", 0.0, 0.0, recipe.radius),),
        ).recipe
    if type(recipe) is PlateWithHoleGeometry:
        return planar_sketch_geometry(
            recipe.name,
            contours=(
                SketchRectangle(
                    "material",
                    0.0,
                    0.0,
                    recipe.width,
                    recipe.height,
                ),
                SketchCircle(
                    "cut",
                    recipe.hole_x,
                    recipe.hole_y,
                    recipe.hole_radius,
                ),
            ),
        ).recipe
    raise TypeError(
        "incremental planar edits require a planar primitive or sketch"
    )


def _next_sketch_ids(
    sketch: SketchGeometry,
    prefix: str,
    count: int,
) -> tuple[str, ...]:
    used = {
        item.id.casefold()
        for item in (*sketch.points, *sketch.curves, *sketch.constraints)
        if getattr(item, "id", None) is not None
    }
    index = 1
    allocated: list[str] = []
    while len(allocated) < count:
        candidate = f"{prefix}{index}"
        if candidate.casefold() not in used:
            allocated.append(candidate)
            used.add(candidate.casefold())
        index += 1
    return tuple(allocated)


def _sketch_constraint_summary(sketch: SketchGeometry) -> dict[str, object]:
    constraints = sketch.constraints
    return {
        "count": len(constraints),
        "enabled_count": sum(item.enabled for item in constraints),
        "driving_dimension_count": sum(
            bool(getattr(item, "driving", False)) for item in constraints
        ),
        "types": sorted({type(item).__name__ for item in constraints}),
        "capability": {"read": True, "create": True, "edit": True},
    }


def _sketch_constraint_catalog_item(
    constraint: SketchConstraint,
) -> dict[str, object]:
    item = _sketch_constraint_to_payload(constraint)
    if item["kind"] == "angle":
        item["angle_degrees"] = math.degrees(float(item.pop("value")))
    return item


def _sketch_curve_to_payload(
    curve: SketchLine | SketchArc | SketchCircle,
) -> dict[str, object]:
    if isinstance(curve, SketchLine):
        return {
            "kind": "line",
            "id": curve.id,
            "start_point_id": curve.start_point_id,
            "end_point_id": curve.end_point_id,
        }
    if isinstance(curve, SketchArc):
        return {
            "kind": "arc",
            "id": curve.id,
            "start_point_id": curve.start_point_id,
            "center_point_id": curve.center_point_id,
            "end_point_id": curve.end_point_id,
            "orientation": curve.orientation,
        }
    if not curve.is_curve:
        raise TypeError("planar sketch payload requires strict circles")
    return {
        "kind": "circle",
        "id": curve.id,
        "center_point_id": curve.center_point_id,
        "radius": curve.radius,
    }


def _sketch_constraint_to_payload(
    constraint: SketchConstraint,
) -> dict[str, object]:
    common: dict[str, object] = {
        "id": constraint.id,
        "source": constraint.source,
        "enabled": constraint.enabled,
    }
    cases: tuple[tuple[type, str, tuple[str, ...]], ...] = (
        (SketchCoincidentConstraint, "coincident", ("first_point_id", "second_point_id")),
        (SketchPointOnCurveConstraint, "point_on_curve", ("point_id", "curve_id")),
        (SketchHorizontalConstraint, "horizontal", ("line_id",)),
        (SketchVerticalConstraint, "vertical", ("line_id",)),
        (SketchParallelConstraint, "parallel", ("first_line_id", "second_line_id")),
        (SketchPerpendicularConstraint, "perpendicular", ("first_line_id", "second_line_id")),
        (SketchTangentConstraint, "tangent", ("first_curve_id", "second_curve_id", "branch_hint")),
        (SketchEqualLengthConstraint, "equal_length", ("first_line_id", "second_line_id")),
        (SketchEqualRadiusConstraint, "equal_radius", ("first_curve_id", "second_curve_id")),
        (SketchConcentricConstraint, "concentric", ("first_curve_id", "second_curve_id")),
        (SketchFixedConstraint, "fixed", ("point_id", "u", "v")),
        (SketchDistanceDimension, "distance", ("first_point_id", "second_point_id", "value", "driving")),
        (SketchRadiusDimension, "radius", ("curve_id", "value", "driving")),
        (SketchAngleDimension, "angle", ("first_line_id", "second_line_id", "value", "driving")),
    )
    for expected, kind, fields in cases:
        if type(constraint) is expected:
            return {
                "kind": kind,
                **common,
                **{field: getattr(constraint, field) for field in fields},
            }
    raise TypeError("unsupported sketch constraint type")


def _sketch_constraint_from_payload(value: object) -> SketchConstraint:
    if not isinstance(value, Mapping):
        raise TypeError("sketch constraint payload must be an object")
    data = dict(value)
    kind = data.get("kind")
    common = {"kind", "id", "source", "enabled"}
    fields: dict[str, tuple[type, tuple[str, ...]]] = {
        "coincident": (SketchCoincidentConstraint, ("first_point_id", "second_point_id")),
        "point_on_curve": (SketchPointOnCurveConstraint, ("point_id", "curve_id")),
        "horizontal": (SketchHorizontalConstraint, ("line_id",)),
        "vertical": (SketchVerticalConstraint, ("line_id",)),
        "parallel": (SketchParallelConstraint, ("first_line_id", "second_line_id")),
        "perpendicular": (SketchPerpendicularConstraint, ("first_line_id", "second_line_id")),
        "tangent": (SketchTangentConstraint, ("first_curve_id", "second_curve_id", "branch_hint")),
        "equal_length": (SketchEqualLengthConstraint, ("first_line_id", "second_line_id")),
        "equal_radius": (SketchEqualRadiusConstraint, ("first_curve_id", "second_curve_id")),
        "concentric": (SketchConcentricConstraint, ("first_curve_id", "second_curve_id")),
        "fixed": (SketchFixedConstraint, ("point_id", "u", "v")),
        "distance": (SketchDistanceDimension, ("first_point_id", "second_point_id", "value", "driving")),
        "radius": (SketchRadiusDimension, ("curve_id", "value", "driving")),
        "angle": (SketchAngleDimension, ("first_line_id", "second_line_id", "value", "driving")),
    }
    if kind not in fields:
        raise ValueError("sketch constraint kind is unsupported")
    constraint_type, specific = fields[str(kind)]
    if set(data) != common | set(specific):
        raise ValueError("sketch constraint fields do not match its kind")
    return constraint_type(
        data["id"],
        *(data[field] for field in specific),
        source=data["source"],
        enabled=data["enabled"],
    )


def _sketch_curve_from_payload(
    value: object,
) -> SketchLine | SketchArc | SketchCircle:
    if not isinstance(value, Mapping):
        raise TypeError("sketch curve payload must be an object")
    kind = value.get("kind")
    if kind == "line" and set(value) == {
        "kind",
        "id",
        "start_point_id",
        "end_point_id",
    }:
        return SketchLine(
            value["id"],
            value["start_point_id"],
            value["end_point_id"],
        )
    if kind == "arc" and set(value) == {
        "kind",
        "id",
        "start_point_id",
        "center_point_id",
        "end_point_id",
        "orientation",
    }:
        return SketchArc(
            value["id"],
            value["start_point_id"],
            value["center_point_id"],
            value["end_point_id"],
            value["orientation"],
        )
    if kind == "circle" and set(value) == {
        "kind",
        "id",
        "center_point_id",
        "radius",
    }:
        return SketchCircle(
            value["id"],
            value["center_point_id"],
            value["radius"],
        )
    raise ValueError("sketch curve fields do not match the bounded schema")


__all__ = [
    "GEOMETRY_FEATURE_CATALOG_TOOL_NAME",
    "GeometryContractProof",
    "GeometryDraft",
    "PlanarEditValidationError",
    "StaticGeometryPreview",
    "box_geometry",
    "apply_planar_edit_batch",
    "add_planar_arc",
    "add_planar_circle",
    "add_planar_constraint",
    "add_planar_line",
    "add_planar_polygon",
    "add_planar_rectangle",
    "create_geometry_edit_proposal",
    "create_geometry_proposal",
    "create_profile_extrusion_proposal",
    "create_profile_path_sweep_proposal",
    "create_profile_revolution_proposal",
    "cylinder_geometry",
    "delete_planar_circles",
    "delete_planar_constraints",
    "delete_planar_curves",
    "disk_geometry",
    "geometry_recipe_from_payload",
    "geometry_recipe_to_payload",
    "geometry_draft",
    "geometry_contract_proof",
    "geometry_feature_catalog_tool_schema",
    "profile_transform_context",
    "feature_topology_catalog",
    "planar_geometry_catalog",
    "planar_construction_proposal_evidence",
    "planar_path_slot_vertices",
    "planar_polygon_geometry",
    "planar_sketch_geometry",
    "rectangle_geometry",
    "replace_planar_circle_pattern",
    "replace_planar_constraint",
    "rotate_geometry",
    "translate_geometry",
    "update_planar_circle",
    "update_planar_arc",
    "update_planar_line",
    "update_planar_point",
    "wire_geometry",
]
