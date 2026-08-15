"""A1 GUI boundary for detached FEM Agent authoring.

The adapter reads a detached session snapshot into bounded DTOs.  The bridge
owns only those DTOs and an ``AuthoringPort``; it never stores or mutates a
``ModelSession``.
"""

from __future__ import annotations

import json
import math
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Protocol

from fem import geometry as geometry_runtime
from fem.application import (
    ModelSession,
    PlanarConstructionCompileError,
    UnitContext,
    compile_planar_construction,
    prepare_part_boolean,
    prepare_solid_body_boolean,
)
from fem.application.changes import SessionDelta
from fem.application.recipe_compiler import compile_recipe
from fem.application.native_scope_materialization import (
    mesh_references_for_logical_entities,
)
from fem.application.results import (
    FieldMaterializationKey,
    FieldPosition,
    FieldState,
    ResultProvider,
    ResultQuery as NativeResultQuery,
    ResultQueryRecord,
    ResultQueryValidationError,
    ResultSourceKey,
    ResultVariable,
)

from fem_agent.authoring import (
    AgentProposal,
    AuthoringAuthorizationError,
    AuthoringContext,
    AuthoringContractError,
    AuthoringPort,
    CapabilitySummary,
    DefinitionSummary,
    LocalModelBinding,
    MeshSummary,
    ModelPatch,
    OperationKind,
    PartSummary,
    ProposalKind,
    ProposalPortRecord,
    ProposalState,
    RequirementLedger,
    RequirementReview,
    UnitContextSummary,
)
from fem_agent.analysis_authoring import require_non_destructive_a5_batch
from fem_agent.boolean_authoring import (
    BODY_BOOLEAN_TOOL_HANDLING,
    PART_BOOLEAN_TOOL_HANDLING,
    create_body_boolean_proposal,
    create_part_boolean_proposal,
)
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
    provider_safe_authoring_payload,
)
from fem_agent.definition_authoring import (
    inverse_operations_for_snapshot,
    require_non_destructive_a4_batch,
    scoped_definition_batch_from_operations,
)
from fem_agent.definition_action_authoring import (
    create_definition_change,
    require_strict_definition_batch,
)
from fem_agent.diagnostics import (
    PROFILE_TRANSFORM_DIAGNOSTIC_CODES,
    profile_transform_diagnostic,
)
from fem_agent.deletion_authoring import (
    apply_delete_operation,
    create_delete_proposal,
    deletable_object_catalog,
)
from fem_agent.editing_authoring import (
    apply_edit_operation,
    create_edit_patch,
    editable_object_catalog,
)
from fem_agent.geometry_authoring import (
    apply_planar_edit_batch,
    add_planar_arc,
    add_planar_circle,
    add_planar_constraint,
    add_planar_line,
    add_planar_polygon,
    add_planar_rectangle,
    _as_strict_planar_sketch,
    box_geometry,
    create_geometry_edit_proposal,
    create_geometry_proposal,
    create_profile_extrusion_proposal,
    create_profile_path_sweep_proposal,
    create_profile_revolution_proposal,
    cylinder_geometry,
    delete_planar_constraints,
    delete_planar_curves,
    delete_planar_circles,
    geometry_draft,
    feature_topology_catalog,
    geometry_recipe_from_payload,
    planar_construction_proposal_evidence,
    planar_geometry_catalog,
    planar_path_slot_vertices,
    planar_polygon_geometry,
    planar_sketch_geometry,
    PlanarEditValidationError,
    profile_transform_context,
    rotate_geometry,
    replace_planar_constraint,
    replace_planar_circle_pattern,
    translate_geometry,
    update_planar_circle,
    update_planar_arc,
    update_planar_line,
    update_planar_point,
    wire_geometry,
)
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_agent.incremental_authoring import (
    require_incremental_definition_batch,
)
from fem_agent.naming import NameAllocator
from fem_agent.solve_authoring import (
    SolveValidationStamp,
    create_solve_proposal,
    solve_operation_identity,
    validation_stamp_for_snapshot,
)
from fem_agent.result_authoring import (
    ANALYSIS_RUN_CATALOG_MAX_LIMIT,
    AcceptedResultSource,
    AnalysisRunCatalog,
    AnalysisRunCatalogEntry,
    AgentResultAggregation,
    AgentResultCatalog,
    AgentResultCatalogResponse,
    AgentResultComparisonQuery,
    AgentResultComparisonResponse,
    AgentResultField,
    AgentResultLocation,
    AgentResultQuery,
    AgentResultQueryResponse,
    AgentResultScalar,
    AgentResultVariable,
    AgentResultQueryBridge,
    compare_result_scalars,
)
from fem_agent.workspace_catalog import (
    WorkspaceCatalogBridge,
    WorkspaceDocumentIdentity,
)
from fem.geometry import (
    BooleanGeometry,
    BooleanLineageResolutionError,
    ExtrudedGeometry,
    LogicalEntityRef,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    PlanarConstructionIR,
    PlanarIRValidationError,
    PathSweptGeometry,
    RevolvedGeometry,
    SketchGeometry,
    SketchCircle,
    SketchRectangle,
    SolidBody,
    WireMember,
    WireGeometry,
    WirePoint,
    describe_recipe_topology,
    geometry_dimension,
    namespace_part_logical_id,
    resolve_extrusion_source_faces,
    resolve_target_radius,
)
from .workspace import FEMWorkspace, WorkspaceDocument
from fem.mesh.settings import LocalMeshControl, MeshSizeFalloff


class _SessionSnapshot(Protocol):
    session_id: str
    session_revision: int
    model_revision: int
    source_kind: str | None
    can_save: bool
    model_name: str | None
    active_part_id: str | None
    parts: object
    named_regions: object
    materials: object
    sections: object
    assignments: object
    steps: object
    artifact: object | None
    validations: object
    runs: object
    selected_run_id: str | None
    displayed_result_run_id: str | None
    displayed_result: object | None
    mesh_current: bool
    unit_context: object | None


@dataclass(frozen=True, slots=True)
class AgentProposalPreview:
    """GUI-only detached tessellation bound to one proven proposal Recipe."""

    dimension: int
    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]
    recipe_digest: str
    proof_digest: str


@dataclass(frozen=True, slots=True)
class _DetachedRecipeMesh:
    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]


def _normalize_profile_preflight_error(error: BaseException) -> AuthoringContractError:
    message = str(error).strip()
    prefix = message.split(":", 1)[0].strip()
    if (
        isinstance(error, AuthoringContractError)
        and prefix in PROFILE_TRANSFORM_DIAGNOSTIC_CODES
    ):
        return error
    return AuthoringContractError(
        "profile-transform.preflight-failed: native geometry preflight failed"
    )


def _run_profile_transform_preflight(callback: Callable[..., None], *args: object) -> None:
    """Normalize backend failures at the detached preflight boundary."""

    try:
        callback(*args)
    except Exception as error:
        raise _normalize_profile_preflight_error(error) from error


def _preflight_profile_extrusions(
    recipes: tuple[ExtrudedGeometry, ...],
) -> None:
    """Compile selected Profiles in detached OCC models before proposal display."""

    def compile_all() -> None:
        for index, recipe in enumerate(recipes, start=1):
            with geometry_runtime.model(
                f"agent-extrusion-preflight-{index}",
                dimension=3,
            ) as cad:
                compile_recipe(cad, recipe)

    try:
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="agent-extrusion-preflight",
        ) as executor:
            executor.submit(compile_all).result()
    except Exception as error:
        raise _normalize_profile_preflight_error(error) from error


def _preflight_derived_geometry(
    recipe: RevolvedGeometry | PathSweptGeometry,
) -> None:
    """Compile OCC evidence off the GUI owner thread and finalize there."""

    def compile_one() -> None:
        with geometry_runtime.model(
            f"agent-{type(recipe).__name__}-preflight",
            dimension=3,
        ) as cad:
            compiled = compile_recipe(cad, recipe)
            if len(compiled.domain) != 1 or cad.volume(compiled.domain[0]) <= 0.0:
                raise AuthoringContractError(
                    "derived Profile preflight did not prove one positive volume"
                )

    try:
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="agent-derived-preflight",
        ) as executor:
            executor.submit(compile_one).result()
    except Exception as error:
        raise _normalize_profile_preflight_error(error) from error


def _preflight_composite_geometry(
    recipe: object,
    *,
    include_preview: bool = False,
) -> _DetachedRecipeMesh | None:
    """Prove exact, positive-volume Bodies before a blank proposal is shown."""

    try:
        topology = describe_recipe_topology(recipe)
        if not topology.exact:
            raise AuthoringContractError(
                "profile-transform.topology-unproven: composite recipe topology "
                "could not be proven exactly"
            )
        expected_bodies = len(topology.entities_of("body", selectable_only=True))
        if expected_bodies < 1:
            raise AuthoringContractError(
                "profile-transform.unexpected-body-count: composite recipe must "
                "describe at least one selectable Body"
            )

        def compile_one() -> _DetachedRecipeMesh | None:
            with geometry_runtime.model(
                f"agent-composite-{type(recipe).__name__}-preflight",
                dimension=3,
            ) as cad:
                compiled = compile_recipe(cad, recipe)
                if len(compiled.domain) != expected_bodies:
                    raise AuthoringContractError(
                        "profile-transform.unexpected-body-count: composite "
                        "preflight domain count does not match the exact topology"
                    )
                if any(cad.volume(item) <= 0.0 for item in compiled.domain):
                    raise AuthoringContractError(
                        "profile-transform.topology-unproven: composite preflight "
                        "did not prove positive volume"
                    )
                if not include_preview:
                    return None
                faces = tuple(compiled.boundary)
                edges = tuple(cad.boundary(faces, combined=False))
                points = tuple(cad.boundary(edges, combined=False))
                tessellation = cad.tessellate_surfaces(faces, edges, points)
                return _DetachedRecipeMesh(
                    tessellation.points,
                    tessellation.faces,
                    tessellation.edges,
                )

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="agent-composite-preflight",
        ) as executor:
            return executor.submit(compile_one).result()
    except Exception as error:
        raise _normalize_profile_preflight_error(error) from error


def _canonical_profile_source_recipe(recipe: object) -> object:
    """Canonicalize legacy planar primitives for proposal replay checks."""

    try:
        return _as_strict_planar_sketch(recipe)
    except (TypeError, ValueError):
        return recipe


def _preflight_part_boolean(
    target,
    tool,
    operation: str,
    *,
    result_part_id: str,
    feature_id: str,
    result_name: str,
):
    """Prepare a detached exact Part proof on the Phase-3 worker seam."""

    def compile_one():
        with geometry_runtime.model(
            f"agent-part-boolean-{operation}-preflight",
            dimension=3,
        ) as cad:
            return prepare_part_boolean(
                cad,
                target,
                tool,
                operation,
                result_part_id=result_part_id,
                feature_id=feature_id,
                result_name=result_name,
            )

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="agent-part-boolean-preflight",
    ) as executor:
        return executor.submit(compile_one).result()


def _preflight_body_boolean(
    geometry: MultiBodyGeometry,
    target_body_id: str,
    tool_body_id: str,
    operation: str,
    *,
    result_name: str,
):
    """Prepare a detached exact same-Part Body proof off the GUI thread."""

    def compile_one():
        with geometry_runtime.model(
            f"agent-body-boolean-{operation}-preflight",
            dimension=3,
        ) as cad:
            return prepare_solid_body_boolean(
                cad,
                geometry,
                target_body_id,
                tool_body_id,
                operation,
                result_name=result_name,
            )

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="agent-body-boolean-preflight",
    ) as executor:
        return executor.submit(compile_one).result()


def _bounded_count(value: object) -> int:
    try:
        return max(0, int(len(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _display_number(value: object) -> str:
    normalized = float(value)
    if normalized == 0.0:
        normalized = 0.0
    return format(normalized, ".12g")


def _planar_profile_design_summary(
    kind: str,
    values: Mapping[str, object],
    index: int,
) -> str:
    if kind == "rectangle":
        return (
            f"矩形{index}(x={_display_number(values['x'])}, "
            f"y={_display_number(values['y'])}, "
            f"宽={_display_number(values['width'])}, "
            f"高={_display_number(values['height'])})"
        )
    if kind == "circle":
        return (
            f"圆{index}(圆心="
            f"({_display_number(values['center_x'])}, "
            f"{_display_number(values['center_y'])}), "
            f"半径={_display_number(values['radius'])})"
        )
    vertices = tuple(values["vertices"])  # type: ignore[arg-type]
    shown = vertices[:8]
    coordinates = ", ".join(
        f"({_display_number(vertex[0])}, {_display_number(vertex[1])})"
        for vertex in shown
    )
    if len(vertices) > len(shown):
        coordinates += f", …共{len(vertices)}点"
    return f"多边形{index}(顶点={coordinates})"


def _bounded_geometry_design_summary(
    prefix: str,
    details: list[str],
) -> str:
    kept: list[str] = []
    for index, detail in enumerate(details):
        candidate = f"{prefix}：" + "；".join((*kept, detail))
        if len(candidate) > 720:
            return (
                f"{prefix}：" + "；".join(kept)
                + f"；另有 {len(details) - index} 个轮廓"
            )
        kept.append(detail)
    return f"{prefix}：" + "；".join(kept)


class _GeometryIntentMismatchError(AuthoringContractError):
    """A syntactically valid recipe conflicts with its declared Part function."""

    def __init__(self, submitted_kind: str, expected_kinds: tuple[str, ...]) -> None:
        self.submitted_kind = submitted_kind
        self.expected_kinds = expected_kinds
        super().__init__(
            "geometry.intent-dimension-mismatch: a 1D wire cannot represent "
            "a plate, sheet, solid, hole, slot, or cutout. Submit the final "
            "planar or extruded geometry instead of a centerline placeholder."
        )


_WIRE_INCOMPATIBLE_PART_MARKERS = (
    "板",
    "薄片",
    "片材",
    "孔",
    "槽",
    "切口",
    "切除",
    "实体",
    "plate",
    "sheet",
    "panel",
    "slab",
    "solid",
    "hole",
    "slot",
    "cutout",
    "cut-out",
    "notch",
)


def _require_geometry_intent_compatibility(
    part_function: str,
    geometry_kind: str,
) -> None:
    if geometry_kind != "wire":
        return
    normalized = part_function.casefold()
    if any(marker in normalized for marker in _WIRE_INCOMPATIBLE_PART_MARKERS):
        raise _GeometryIntentMismatchError(
            geometry_kind,
            (
                "planar_profiles",
                "extruded_profiles",
                "extruded_path_slot_plate",
            ),
        )


def _geometry_unit_summary(
    requirements: Mapping[str, object],
    defaulted_keys: tuple[str, ...],
) -> str:
    label = (
        f"{requirements['length_unit']}-"
        f"{requirements['force_unit']}-"
        f"{requirements['stress_unit']}"
    )
    if len(defaulted_keys) == 3:
        return f"{label}（默认）"
    if defaulted_keys:
        names = {
            "length_unit": "长度",
            "force_unit": "力",
            "stress_unit": "应力",
        }
        defaulted = "、".join(names[key] for key in defaulted_keys)
        return f"{label}（{defaulted}使用默认值）"
    return label


def _recipe_dimension(recipe: object | None) -> int | None:
    if recipe is None:
        return None
    dimension = getattr(recipe, "dimension", None)
    if dimension in {1, 2, 3}:
        return int(dimension)
    if isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        return int(geometry_dimension(recipe))
    name = type(recipe).__name__.casefold()
    if any(
        token in name for token in ("rectangle", "disk", "plate", "sketch", "planar")
    ):
        return 2
    if any(token in name for token in ("box", "cylinder", "extruded")):
        return 3
    if "wire" in name or "line" in name:
        return 1
    return None


def _model_counts(artifact: object | None) -> tuple[int | None, int | None]:
    model = None if artifact is None else getattr(artifact, "model", None)
    if model is None:
        return None, None
    mesh = getattr(model, "mesh", None)
    count_source = model if mesh is None else mesh
    nodes = getattr(count_source, "nodes", None)
    elements = getattr(count_source, "elements", None)
    return _bounded_count(nodes), _bounded_count(elements)


def _provider_recipe_kind(recipe: object | None) -> str | None:
    if recipe is None:
        return None
    return {
        "SketchGeometry": "planar_sketch",
        "PlateWithHoleGeometry": "planar_sketch",
        "RectangleGeometry": "planar_profile",
        "DiskGeometry": "planar_profile",
        "WireGeometry": "wire",
        "BoxGeometry": "solid_primitive",
        "CylinderGeometry": "solid_primitive",
        "MovedGeometry": "transformed_geometry",
        "RotatedGeometry": "transformed_geometry",
    }.get(type(recipe).__name__, "native_geometry")


def _profile_vertices(value: object) -> tuple[tuple[object, object], ...]:
    if (
        not isinstance(value, list)
        or not 3 <= len(value) <= 64
        or any(
            not isinstance(item, Mapping) or set(item) != {"x", "y"}
            for item in value
        )
    ):
        raise ValueError("polygon vertices must contain 3 to 64 x/y objects")
    return tuple((item["x"], item["y"]) for item in value)


def _composite_profile_order_key(value: object) -> tuple[str, str]:
    """Return a geometry-only key so contour order cannot alter identity."""

    if not isinstance(value, Mapping):
        return ("", repr(value))
    kind = str(value.get("kind", ""))
    if kind == "rectangle":
        fields = ("x", "y", "width", "height")
    elif kind == "circle":
        fields = ("center_x", "center_y", "radius")
    elif kind == "polygon":
        vertices = value.get("vertices")
        if isinstance(vertices, list):
            return (
                kind,
                repr(tuple(
                    (item.get("x"), item.get("y"))
                    for item in vertices
                    if isinstance(item, Mapping)
                )),
            )
        return (kind, repr(vertices))
    else:
        return (kind, repr(sorted(value.items(), key=lambda item: str(item[0]))))
    return (kind, repr(tuple(value.get(field) for field in fields)))


def _pop_profile_annotations(item: dict[str, object]) -> None:
    """Validate optional provider hints while leaving topology authoritative."""

    role = item.pop("role", None)
    operation = item.pop("operation", None)
    if role is not None and role not in {"material", "hole"}:
        raise ValueError("composite profile role must be material or hole")
    if operation is not None and operation not in {"material", "cut"}:
        raise ValueError("composite profile operation must be material or cut")
    if role is not None and operation is not None:
        expected_role = "material" if operation == "material" else "hole"
        if role != expected_role:
            raise ValueError("composite profile role and operation disagree")


def _composite_profile_contours(
    value: object,
    *,
    recipe_name: str,
) -> tuple[object, dict[str, object], tuple[str, ...]]:
    """Build one strict XY sketch and prove one material Profile locally."""

    if not isinstance(value, list) or not value:
        raise ValueError("composite profiles must be a non-empty array")
    draft = None
    summaries: list[str] = []
    for index, raw in enumerate(
        sorted(value, key=_composite_profile_order_key),
        start=1,
    ):
        if not isinstance(raw, Mapping):
            raise TypeError("each composite profile must be an object")
        item = dict(raw)
        kind = str(item.pop("kind", ""))
        _pop_profile_annotations(item)
        # ``role``/``operation`` are annotations only.  The strict sketch
        # analyser below determines material versus hole from containment, so
        # contour order cannot change the selected Profile.
        try:
            if kind == "rectangle":
                if set(item) != {"x", "y", "width", "height"}:
                    raise ValueError("rectangle profile fields do not match")
                shape = SketchRectangle(
                    "material",
                    item["x"],
                    item["y"],
                    item["width"],
                    item["height"],
                )
                summaries.append(
                    _planar_profile_design_summary(kind, item, index)
                )
                if draft is None:
                    draft = planar_sketch_geometry(
                        recipe_name,
                        contours=(shape,),
                    )
                else:
                    draft = add_planar_rectangle(
                        draft.recipe,
                        x=item["x"],
                        y=item["y"],
                        width=item["width"],
                        height=item["height"],
                    )
            elif kind == "circle":
                if set(item) != {"center_x", "center_y", "radius"}:
                    raise ValueError("circle profile fields do not match")
                shape = SketchCircle(
                    "material",
                    item["center_x"],
                    item["center_y"],
                    item["radius"],
                )
                summaries.append(
                    _planar_profile_design_summary(kind, item, index)
                )
                if draft is None:
                    draft = planar_sketch_geometry(
                        recipe_name,
                        contours=(shape,),
                    )
                else:
                    draft = add_planar_circle(
                        draft.recipe,
                        center_x=item["center_x"],
                        center_y=item["center_y"],
                        radius=item["radius"],
                    )
            elif kind == "polygon":
                if set(item) != {"vertices"}:
                    raise ValueError("polygon profile fields do not match")
                vertices = _profile_vertices(item["vertices"])
                summaries.append(
                    _planar_profile_design_summary(
                        kind,
                        {"vertices": vertices},
                        index,
                    )
                )
                if draft is None:
                    draft = planar_polygon_geometry(
                        recipe_name,
                        vertices=vertices,
                    )
                else:
                    draft = add_planar_polygon(
                        draft.recipe,
                        vertices=vertices,
                    )
            else:
                raise ValueError("unsupported composite profile kind")
        except (TypeError, ValueError) as error:
            raise ValueError(f"composite profile {index} is invalid: {error}") from error
    if draft is None:  # pragma: no cover - guarded by the non-empty input check
        raise ValueError("composite profiles did not produce a sketch")
    sketch = draft.recipe
    context = profile_transform_context(sketch)
    if not context.get("topology_exact"):
        raise AuthoringContractError(
            "profile-transform.topology-unproven: composite sketch topology "
            "could not be proven"
        )
    if int(context.get("material_profile_count", 0)) != 1:
        raise AuthoringContractError(
            "profile-transform.unexpected-body-count: composite geometry "
            "requires exactly one material Profile"
        )
    selection = resolve_extrusion_source_faces(sketch)
    if len(selection.face_ids) != 1:
        raise AuthoringContractError(
            "profile-transform.unexpected-body-count: composite geometry "
            "requires exactly one canonical source face"
        )
    context["design_summaries"] = list(summaries)
    return sketch, context, selection.face_ids


def _composite_path(value: object, *, name: str) -> WireGeometry:
    if not isinstance(value, Mapping) or set(value) != {"points", "members"}:
        raise ValueError("composite path fields do not match")
    raw_points = value["points"]
    raw_members = value["members"]
    if not isinstance(raw_points, list) or not isinstance(raw_members, list):
        raise TypeError("composite path points and members must be arrays")
    if any(
        not isinstance(item, Mapping) or set(item) != {"name", "x", "y", "z"}
        for item in raw_points
    ):
        raise ValueError("composite path point fields do not match")
    if any(
        not isinstance(item, Mapping) or set(item) != {"name", "start", "end"}
        for item in raw_members
    ):
        raise ValueError("composite path member fields do not match")
    return WireGeometry(
        name,
        tuple(
            WirePoint(item["name"], item["x"], item["y"], item["z"])
            for item in raw_points
        ),
        tuple(
            WireMember(item["name"], item["start"], item["end"])
            for item in raw_members
        ),
    )


def authoring_context_from_snapshot(
    snapshot: _SessionSnapshot,
    *,
    document_id: str | int | None = None,
) -> AuthoringContext:
    """Copy a session projection into a bounded, provider-safe DTO."""

    session_id = str(snapshot.session_id)
    source_kind = "blank" if snapshot.source_kind is None else str(snapshot.source_kind)
    supported = source_kind in {"blank", "native"}
    binding = LocalModelBinding(
        document_id=(
            f"document:{session_id}"
            if document_id is None
            else str(document_id)
        ),
        session_id=session_id,
        session_revision=int(snapshot.session_revision),
        source_kind=source_kind,
        supported=supported,
    )

    parts: list[PartSummary] = []
    if source_kind == "native":
        for part in tuple(snapshot.parts)[:128]:  # type: ignore[arg-type]
            recipe = getattr(part, "geometry_recipe", None)
            parts.append(
                PartSummary(
                    part_id=str(getattr(part, "id")),
                    name=str(getattr(part, "name")),
                    recipe_kind=_provider_recipe_kind(recipe),
                    dimension=_recipe_dimension(recipe),
                    suppressed=bool(getattr(part, "suppressed", False)),
                )
            )

    node_count, element_count = _model_counts(snapshot.artifact)
    mesh_present = bool(
        snapshot.artifact is not None
        and element_count is not None
        and element_count > 0
    )
    mesh_current = bool(
        mesh_present and getattr(snapshot, "mesh_current", False)
    )
    validation_status = "not_run"
    validations = getattr(snapshot, "validations", {})
    if _bounded_count(validations):
        records = tuple(validations.values())  # type: ignore[union-attr]
        validation_status = (
            "passed"
            if records and any(bool(getattr(item, "passed", False)) for item in records)
            else "blocked"
        )
    runs = tuple(getattr(snapshot, "runs", ()))
    result_count = sum(
        1
        for item in runs
        if getattr(item, "result_id", None) is not None
        and str(getattr(item, "status", "")).casefold().endswith("succeeded")
    )
    job_status = "idle"
    if any(
        str(getattr(item, "status", "")).casefold().endswith("running") for item in runs
    ):
        job_status = "running"
    elif runs:
        job_status = (
            str(getattr(runs[-1], "status", "completed")).split(".")[-1].casefold()
        )

    blocked_reason = None if supported else "V1 只绑定空白或 native 文档"
    deletable_objects_available = bool(deletable_object_catalog(snapshot))
    editable_objects_available = bool(editable_object_catalog(snapshot))
    editable_geometry_available = bool(
        supported
        and source_kind == "native"
        and any(
            getattr(part, "geometry_recipe", None) is not None
            and not bool(getattr(part, "suppressed", False))
            for part in tuple(snapshot.parts)[:128]  # type: ignore[arg-type]
        )
    )
    capabilities = (
        CapabilitySummary("read_authoring_context", supported, blocked_reason),
        CapabilitySummary(
            "read_geometry_feature_catalog",
            editable_geometry_available,
            (
                None
                if editable_geometry_available
                else "当前 native 项目没有可读取的部件几何"
            ),
        ),
        CapabilitySummary("review_requirements", supported, blocked_reason),
        CapabilitySummary("build_agent_draft", supported, blocked_reason),
        CapabilitySummary("present_static_proposal", supported, blocked_reason),
        CapabilitySummary("draft_native_geometry", supported, blocked_reason),
        CapabilitySummary("commit_native_geometry", supported, blocked_reason),
        CapabilitySummary(
            "edit_native_geometry",
            editable_geometry_available,
            (
                None
                if editable_geometry_available
                else "当前 native 项目没有可编辑的部件几何"
            ),
        ),
        CapabilitySummary(
            "draft_mesh_intent",
            supported and source_kind == "native",
            (
                None
                if supported and source_kind == "native"
                else "网格意图需要 native 项目"
            ),
        ),
        CapabilitySummary(
            "request_mesh_proposal",
            supported and source_kind == "native",
            (
                None
                if supported and source_kind == "native"
                else "网格提案需要 native 项目"
            ),
        ),
        CapabilitySummary(
            "run_model_preflight",
            (
                supported
                and source_kind == "native"
                and bool(getattr(snapshot, "mesh_current", False))
            ),
            (
                None
                if (
                    supported
                    and source_kind == "native"
                    and bool(getattr(snapshot, "mesh_current", False))
                )
                else "模型预检需要当前 native 网格"
            ),
        ),
        CapabilitySummary(
            "request_solve_proposal",
            (
                supported
                and source_kind == "native"
                and bool(getattr(snapshot, "mesh_current", False))
                and validation_status == "passed"
            ),
            (
                None
                if (
                    supported
                    and source_kind == "native"
                    and bool(getattr(snapshot, "mesh_current", False))
                    and validation_status == "passed"
                )
                else "求解提案需要当前 native 网格和通过的预检"
            ),
        ),
        CapabilitySummary(
            "request_project_save",
            bool(getattr(snapshot, "can_save", False)),
            (
                None
                if bool(getattr(snapshot, "can_save", False))
                else "项目保存需要当前已打开的 native 项目"
            ),
        ),
        CapabilitySummary(
            "delete_model_objects",
            deletable_objects_available,
            (
                None
                if deletable_objects_available
                else "当前 native 项目没有可删除对象"
            ),
        ),
        CapabilitySummary(
            "edit_model_objects",
            editable_objects_available,
            (
                None
                if editable_objects_available
                else "当前 native 项目没有可编辑的定义对象"
            ),
        ),
        CapabilitySummary(
            "query_accepted_result",
            (
                supported
                and source_kind == "native"
                and result_count > 0
            ),
            (
                None
                if (
                    supported
                    and source_kind == "native"
                    and result_count > 0
                )
                else "结果查询需要当前已接受的 native 结果"
            ),
        ),
        CapabilitySummary(
            "compare_accepted_results",
            supported and source_kind == "native" and result_count >= 2,
            (
                None
                if supported and source_kind == "native" and result_count >= 2
                else "结果比较需要当前会话中至少两个已接受的 native 结果"
            ),
        ),
    )
    return AuthoringContext(
        binding=binding,
        model_name=(
            str(snapshot.model_name) if snapshot.model_name is not None else None
        ),
        active_part_id=(
            str(snapshot.active_part_id)
            if snapshot.active_part_id is not None
            else None
        ),
        parts=tuple(parts),
        mesh=MeshSummary(
            present=mesh_present,
            current=mesh_current,
            node_count=node_count,
            element_count=element_count,
        ),
        definitions=DefinitionSummary(
            named_region_count=_bounded_count(snapshot.named_regions),
            material_count=_bounded_count(snapshot.materials),
            section_count=_bounded_count(snapshot.sections),
            assignment_count=_bounded_count(snapshot.assignments),
            analysis_step_count=_bounded_count(snapshot.steps),
        ),
        validation_status=validation_status,
        job_status=job_status,
        result_available=snapshot.displayed_result is not None,
        run_count=len(runs),
        result_count=result_count,
        selected_run_id=getattr(snapshot, "selected_run_id", None),
        displayed_result_run_id=getattr(
            snapshot,
            "displayed_result_run_id",
            None,
        ),
        capabilities=capabilities,
        unit_context=(
            None
            if getattr(snapshot, "unit_context", None) is None
            else UnitContextSummary(
                length=str(snapshot.unit_context.length),
                force=str(snapshot.unit_context.force),
                stress=str(snapshot.unit_context.stress),
                density=(
                    None
                    if snapshot.unit_context.density is None
                    else str(snapshot.unit_context.density)
                ),
                acceleration=(
                    None
                    if snapshot.unit_context.acceleration is None
                    else str(snapshot.unit_context.acceleration)
                ),
                convention=(
                    None
                    if snapshot.unit_context.convention is None
                    else str(snapshot.unit_context.convention)
                ),
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class BridgeReceipt:
    proposal_id: str
    state: ProposalState
    message: str = ""
    replayed: bool = False


class AppliedPatchState(str, Enum):
    APPLIED = "applied"
    UNDONE = "undone"
    STALE = "stale"


class AgentPreflightState(str, Enum):
    RUNNING = "running"
    PASSED = "passed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class AgentPreflightRecord:
    request_id: str
    step_name: str
    base_session_revision: int
    state: AgentPreflightState
    message: str = ""
    validation_stamp: SolveValidationStamp | None = None


@dataclass(frozen=True, slots=True)
class AppliedPatchRecord:
    """One local automatic patch and its exact one-shot inverse."""

    patch: ModelPatch
    inverse_patch: ModelPatch
    session_revision: int
    delta: SessionDelta
    state: AppliedPatchState = AppliedPatchState.APPLIED
    message: str = ""
    replayed: bool = False

    @property
    def undo_available(self) -> bool:
        return self.state is AppliedPatchState.APPLIED

    @property
    def display_summary(self) -> object:
        return self.patch.display_summary


@dataclass(frozen=True, slots=True)
class _GuiControlAuthorization:
    proposal_id: str
    action: str
    nonce: int


@dataclass(frozen=True, slots=True)
class AgentMeshTaskRequest:
    proposal_id: str
    proposal_hash: str
    task: object


@dataclass(frozen=True, slots=True)
class AgentSolveTaskRequest:
    """Metadata-only request passed to the existing GUI job entry."""

    proposal_id: str
    proposal_hash: str
    step_name: str
    job_name: str
    base_session_revision: int
    artifact_id: str
    model_revision: int
    validation_stamp: SolveValidationStamp


@dataclass(frozen=True, slots=True)
class AgentGeometryMutation:
    """One validated synchronous geometry mutation for in-place or child commit."""

    operation_kind: OperationKind
    affected_part_ids: tuple[str, ...]
    apply: Callable[[ModelSession, int], None]

    def __post_init__(self) -> None:
        if type(self.operation_kind) is not OperationKind:
            raise TypeError("operation_kind must be OperationKind")
        if not self.affected_part_ids or any(
            type(item) is not str or not item.strip()
            for item in self.affected_part_ids
        ):
            raise ValueError("affected_part_ids must contain nonblank strings")
        if len(set(self.affected_part_ids)) != len(self.affected_part_ids):
            raise ValueError("affected_part_ids must be unique")
        if not callable(self.apply):
            raise TypeError("apply must be callable")


@dataclass(frozen=True, slots=True)
class AgentPreflightTaskRequest:
    """Exact metadata used to run an automatic local preflight."""

    request_id: str
    step_name: str
    base_session_revision: int
    session_id: str
    artifact_id: str
    model_revision: int


@dataclass(frozen=True, slots=True)
class _ResolvedResultTarget:
    document: WorkspaceDocument | None
    document_id: str
    session: ModelSession


class _WorkspaceResultResolutionError(ValueError):
    pass


class SessionResultQueryPort:
    """A7 read-only adapter over the Session's exact accepted result provider."""

    def __init__(
        self,
        session: ModelSession,
        workspace: FEMWorkspace | None = None,
    ) -> None:
        if type(session) is not ModelSession:
            raise TypeError("session must be ModelSession")
        if workspace is not None and type(workspace) is not FEMWorkspace:
            raise TypeError("workspace must be FEMWorkspace or None")
        self._session = session
        self._workspace = workspace

    @property
    def session(self) -> ModelSession:
        """Return the currently bound model Session."""

        return self._session

    def bind_session(
        self,
        session: ModelSession,
        *,
        idle: Callable[[], bool] | None = None,
    ) -> None:
        """Rebind the read-only adapter while the shared Agent is idle."""

        if type(session) is not ModelSession:
            raise TypeError("session must be ModelSession")
        if idle is not None and not idle():
            raise RuntimeError("Agent runtime must be idle before rebinding")
        self._session = session

    rebind_session = bind_session

    def _resolve_target(
        self,
        target: WorkspaceDocumentIdentity | None,
    ) -> _ResolvedResultTarget:
        workspace = self._workspace
        if workspace is None:
            if target is not None and target.session_id != self._session.session_id:
                raise _WorkspaceResultResolutionError(
                    "target session is unavailable"
                )
            return _ResolvedResultTarget(
                None,
                (
                    f"document:{self._session.session_id}"
                    if target is None
                    else target.document_id
                ),
                self._session,
            )
        if target is None:
            matches = tuple(
                document
                for document in workspace.documents()
                if document.session is self._session
            )
        else:
            matches = tuple(
                document
                for document in workspace.documents()
                if (
                    str(document.document_id) == target.document_id
                    and document.session.session_id == target.session_id
                )
            )
        if len(matches) != 1:
            raise _WorkspaceResultResolutionError(
                "workspace target is missing, foreign, or ambiguous"
            )
        document = matches[0]
        return _ResolvedResultTarget(
            document,
            str(document.document_id),
            document.session,
        )

    def _resolve_source(self, session_id: str) -> _ResolvedResultTarget:
        workspace = self._workspace
        if workspace is None:
            if session_id != self._session.session_id:
                raise _WorkspaceResultResolutionError(
                    "result source session is unavailable"
                )
            return _ResolvedResultTarget(
                None,
                f"document:{session_id}",
                self._session,
            )
        matches = tuple(
            document
            for document in workspace.documents()
            if document.session.session_id == session_id
        )
        if len(matches) != 1:
            raise _WorkspaceResultResolutionError(
                "result source session is missing or ambiguous"
            )
        document = matches[0]
        return _ResolvedResultTarget(
            document,
            str(document.document_id),
            document.session,
        )

    def _target_is_current(self, target: _ResolvedResultTarget) -> bool:
        if target.document is None:
            return target.session is self._session
        workspace = self._workspace
        if workspace is None:
            return False
        return any(
            document is target.document and document.session is target.session
            for document in workspace.documents()
        )

    def analysis_runs(
        self,
        *,
        cursor: int = 0,
        limit: int = ANALYSIS_RUN_CATALOG_MAX_LIMIT,
        target: WorkspaceDocumentIdentity | None = None,
    ) -> AnalysisRunCatalog:
        resolved = self._resolve_target(target)
        snapshot = resolved.session.snapshot()
        runs = tuple(snapshot.runs)
        if type(cursor) is not int or cursor < 0 or cursor > len(runs):
            raise _WorkspaceResultResolutionError("run catalog cursor is invalid")
        if type(limit) is not int or not 1 <= limit <= ANALYSIS_RUN_CATALOG_MAX_LIMIT:
            raise _WorkspaceResultResolutionError("run catalog limit is invalid")
        page = runs[cursor : cursor + limit]
        entries = tuple(
            AnalysisRunCatalogEntry(
                run_id=str(run.run_id),
                name=str(run.name),
                step_name=str(run.step_name),
                status=str(getattr(run.status, "value", run.status)),
                artifact_id=str(run.artifact_id),
                model_revision=int(run.model_revision),
                source_run_id=(
                    None if run.source_run_id is None else str(run.source_run_id)
                ),
                result_id=(None if run.result_id is None else str(run.result_id)),
                materialization_generation=snapshot.result_generations.get(
                    run.run_id
                ),
            )
            for run in page
        )
        if not self._target_is_current(resolved):
            raise _WorkspaceResultResolutionError(
                "workspace target changed while reading runs"
            )
        next_cursor = cursor + len(entries)
        return AnalysisRunCatalog(
            document_id=resolved.document_id,
            session_id=str(snapshot.session_id),
            session_revision=int(snapshot.session_revision),
            selected_run_id=snapshot.selected_run_id,
            displayed_result_run_id=snapshot.displayed_result_run_id,
            cursor=cursor,
            limit=limit,
            runs=entries,
            next_cursor=(next_cursor if next_cursor < len(runs) else None),
            total_count=len(runs),
        )

    def catalog(
        self,
        run_id: str | None = None,
        *,
        target: WorkspaceDocumentIdentity | None = None,
    ) -> AgentResultCatalogResponse:
        try:
            resolved = self._resolve_target(target)
        except _WorkspaceResultResolutionError:
            return _result_catalog_failure(
                "result.catalog.target_unavailable",
                "The exact workspace result target is missing, foreign, or ambiguous.",
                clarification_required=True,
            )
        session = resolved.session
        projection = session.projection_snapshot()
        uses_displayed_result = run_id is None
        target_run_id = (
            projection.displayed_result_run_id
            if run_id is None
            else str(run_id).strip()
        )
        if not target_run_id:
            return _result_catalog_failure(
                "result.catalog.no_accepted_result",
                "No accepted native result is available for the requested run.",
                retryable=True,
            )
        provider = session.result_provider_for(target_run_id)
        identity = session.result_identity_for(target_run_id)
        if provider is None or identity is None:
            return _result_catalog_failure(
                "result.catalog.no_accepted_result",
                "No current accepted native result is available.",
                retryable=True,
            )
        source, generation = identity
        if (
            provider.source != source
            or provider.snapshot.generation != generation
        ):
            return _result_catalog_failure(
                "result.catalog.current_identity_invalid",
                "The current accepted result provider identity is inconsistent.",
                retryable=True,
            )
        units = projection.unit_context
        if projection.source_kind not in {"native", "result"} or units is None:
            return _result_catalog_failure(
                "result.catalog.units_unavailable",
                "A project or result-archive unit context is required for result values.",
                clarification_required=True,
            )
        fields = tuple(
            AgentResultField(
                variable=AgentResultVariable(
                    item.descriptor.field_id.variable.value
                ),
                position=item.descriptor.field_id.position.value,
                components=item.descriptor.columns,
                unit=_result_unit(
                    units,
                    AgentResultVariable(
                        item.descriptor.field_id.variable.value
                    ),
                ),
            )
            for item in provider.catalog().fields
            if (
                item.state is FieldState.READY
                and item.descriptor.field_id.variable
                in {
                    ResultVariable.U,
                    ResultVariable.UR,
                    ResultVariable.RF,
                    ResultVariable.RM,
                    ResultVariable.SF,
                    ResultVariable.SM,
                    ResultVariable.LE,
                    ResultVariable.S,
                }
            )
        )
        if not fields:
            return _result_catalog_failure(
                "result.catalog.no_supported_ready_fields",
                "The accepted result has no supported READY result fields.",
                clarification_required=True,
            )
        nodal_regions, element_regions = _published_result_regions(
            projection, provider, source, target_run_id
        )
        catalog = AgentResultCatalog(
            source=_accepted_source(source),
            materialization_generation=generation,
            fields=fields,
            nodal_regions=nodal_regions,
            element_regions=element_regions,
        )
        if (
            uses_displayed_result
            and session.projection_snapshot().displayed_result_run_id
            != target_run_id
        ) or session.result_identity_for(target_run_id) != (
            source,
            generation,
        ) or not self._target_is_current(resolved):
            return _result_catalog_failure(
                "result.catalog.stale",
                "The accepted result changed before the catalog was returned.",
                retryable=True,
            )
        return AgentResultCatalogResponse.success(catalog)

    def query(self, request: AgentResultQuery) -> AgentResultQueryResponse:
        if type(request) is not AgentResultQuery:
            raise TypeError("request must be AgentResultQuery")
        try:
            resolved = self._resolve_source(request.expected_source.session_id)
        except _WorkspaceResultResolutionError:
            return _result_query_failure(
                "result.query.source_unavailable",
                "The exact workspace result source session is missing or ambiguous.",
                clarification_required=True,
            )
        return self._query_resolved(request, resolved)

    def _query_resolved(
        self,
        request: AgentResultQuery,
        resolved: _ResolvedResultTarget,
    ) -> AgentResultQueryResponse:
        session = resolved.session
        target_run_id = request.expected_source.run_id
        provider = session.result_provider_for(target_run_id)
        identity = session.result_identity_for(target_run_id)
        if provider is None or identity is None:
            return _result_query_failure(
                "result.query.no_accepted_result",
                "No current accepted native result is available.",
                retryable=True,
            )
        source, generation = identity
        if (
            provider.source != source
            or provider.snapshot.generation != generation
        ):
            return _result_query_failure(
                "result.query.current_identity_invalid",
                "The current accepted result provider identity is inconsistent.",
                retryable=True,
            )
        if (
            _accepted_source(source) != request.expected_source
            or generation != request.expected_materialization_generation
        ):
            return _result_query_failure(
                "result.query.stale",
                "The requested result source or materialization generation is stale.",
                retryable=True,
            )

        projection = session.projection_snapshot()
        units = projection.unit_context
        if projection.source_kind not in {"native", "result"} or units is None:
            return _result_query_failure(
                "result.query.units_unavailable",
                "A project or result-archive unit context is required for result values.",
                clarification_required=True,
            )

        try:
            nodal_regions, element_regions = _published_result_regions(
                projection, provider, source, target_run_id
            )
            _require_published_result_region(
                request,
                nodal_regions=nodal_regions,
                element_regions=element_regions,
            )
            availability = _resolve_result_availability(provider, request)
            native_query = _native_result_query(
                provider,
                availability.key,
                request,
            )
            checked = provider.validate_query(native_query)
            if checked.state is FieldState.LAZY:
                return _result_query_failure(
                    "result.query.field_not_materialized",
                    "The requested field is not READY in this accepted generation.",
                    retryable=True,
                )
            if checked.state is not FieldState.READY:
                return _result_query_failure(
                    "result.query.field_unavailable",
                    "The requested field is unavailable for this accepted result.",
                    clarification_required=True,
                )
            result = provider.query(native_query)
            if (
                result.source != source
                or result.materialization_generation != generation
                or result.query != native_query
            ):
                return _result_query_failure(
                    "result.query.provider_identity_invalid",
                    "The native query result does not match the requested identity.",
                    retryable=True,
                )
            if not result.records:
                return _result_query_failure(
                    "result.query.empty_region",
                    "The requested field has no values in the selected region.",
                    clarification_required=True,
                )
            scalar = _aggregate_native_result(
                request,
                result.records,
                source,
                generation,
                _result_unit(units, request.variable),
            )
        except _AgentResultQueryRejected as error:
            return _result_query_failure(
                error.code,
                str(error),
                clarification_required=error.clarification_required,
            )
        except ResultQueryValidationError as error:
            return _result_query_failure(
                error.code,
                str(error),
                clarification_required=True,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return _result_query_failure(
                "result.query.rejected",
                f"The accepted-result query was rejected: {type(error).__name__}.",
                clarification_required=True,
            )

        if session.result_identity_for(target_run_id) != (
            source,
            generation,
        ) or not self._target_is_current(resolved):
            return _result_query_failure(
                "result.query.stale",
                "The accepted result changed before the query completed.",
                retryable=True,
            )
        return AgentResultQueryResponse.success(scalar)

    def compare(
        self,
        request: AgentResultComparisonQuery,
    ) -> AgentResultComparisonResponse:
        if type(request) is not AgentResultComparisonQuery:
            raise TypeError("request must be AgentResultComparisonQuery")
        references = (request.baseline, request.candidate)
        try:
            resolved = tuple(
                self._resolve_source(reference.expected_source.session_id)
                for reference in references
            )
        except _WorkspaceResultResolutionError:
            return _result_comparison_failure(
                "result.comparison.source_unavailable",
                "One or both workspace result source sessions are missing or ambiguous.",
                clarification_required=True,
            )

        initial_identities = tuple(
            target.session.result_identity_for(
                reference.expected_source.run_id
            )
            for target, reference in zip(resolved, references, strict=True)
        )
        if any(identity is None for identity in initial_identities):
            return _result_comparison_failure(
                "result.comparison.stale",
                "One or both accepted results are no longer available.",
                retryable=True,
            )
        for reference, identity in zip(
            references,
            initial_identities,
            strict=True,
        ):
            source, generation = identity
            if (
                _accepted_source(source) != reference.expected_source
                or generation
                != reference.expected_materialization_generation
            ):
                return _result_comparison_failure(
                    "result.comparison.stale",
                    "One or both result identities or generations are stale.",
                    retryable=True,
                )

        baseline_response = self._query_resolved(
            request.result_query(request.baseline), resolved[0]
        )
        if not baseline_response.ok:
            return _comparison_query_failure("baseline", baseline_response)
        candidate_response = self._query_resolved(
            request.result_query(request.candidate), resolved[1]
        )
        if not candidate_response.ok:
            return _comparison_query_failure("candidate", candidate_response)

        final_identities = tuple(
            target.session.result_identity_for(
                reference.expected_source.run_id
            )
            for target, reference in zip(resolved, references, strict=True)
        )
        if final_identities != initial_identities or not all(
            self._target_is_current(target) for target in resolved
        ):
            return _result_comparison_failure(
                "result.comparison.stale",
                "An accepted result changed while the comparison was running.",
                retryable=True,
            )
        baseline = baseline_response.scalar
        candidate = candidate_response.scalar
        if baseline is None or candidate is None:
            raise RuntimeError("successful result query omitted its scalar")
        return compare_result_scalars(request, baseline, candidate)


class _AgentResultQueryRejected(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        clarification_required: bool = True,
    ) -> None:
        self.code = code
        self.clarification_required = clarification_required
        super().__init__(message)


def _result_regions_are_current(
    projection: _SessionSnapshot,
    source: ResultSourceKey,
    run_id: str,
) -> bool:
    artifact = getattr(projection, "artifact", None)
    return bool(
        run_id == projection.displayed_result_run_id
        and artifact is not None
        and str(getattr(artifact, "artifact_id", "")) == source.artifact_id
        and int(getattr(projection, "model_revision", -1))
        == source.model_revision
    )


def _resolve_result_availability(
    provider: ResultProvider,
    request: AgentResultQuery,
):
    variable = ResultVariable(request.variable.value)
    try:
        position = FieldPosition(request.position)
    except ValueError as error:
        raise _AgentResultQueryRejected(
            "result.query.position_unsupported",
            f"Result position {request.position!r} is unsupported.",
        ) from error
    matches = tuple(
        item
        for item in provider.catalog().fields
        if (
            item.descriptor.field_id.variable is variable
            and item.descriptor.field_id.position is position
        )
    )
    if not matches:
        raise _AgentResultQueryRejected(
            "result.query.field_not_available",
            "The requested variable and position are absent from the result catalog.",
        )
    if len(matches) != 1:
        raise _AgentResultQueryRejected(
            "result.query.field_ambiguous",
            "The requested variable and position do not identify one catalog field.",
        )
    availability = matches[0]
    if request.component not in availability.descriptor.columns:
        raise _AgentResultQueryRejected(
            "result.query.component_not_available",
            f"Result component {request.component!r} is not available.",
        )
    return availability


def _require_published_result_region(
    request: AgentResultQuery,
    *,
    nodal_regions: tuple[str, ...],
    element_regions: tuple[str, ...],
) -> None:
    nodal_variable = request.variable in {
        AgentResultVariable.DISPLACEMENT,
        AgentResultVariable.ROTATION,
        AgentResultVariable.REACTION_FORCE,
        AgentResultVariable.REACTION_MOMENT,
    }
    allowed = nodal_regions if nodal_variable else element_regions
    wrong_entity_regions = element_regions if nodal_variable else nodal_regions
    if request.region not in allowed and request.region in wrong_entity_regions:
        raise _AgentResultQueryRejected(
            "result.query.region_entity_unsupported",
            "The requested region has the wrong entity kind for this field.",
        )
    if request.region not in allowed:
        raise _AgentResultQueryRejected(
            "result.query.region_not_published",
            "The requested region is absent from the bounded result catalog.",
        )


def _published_result_regions(
    projection: _SessionSnapshot,
    provider: ResultProvider,
    source: ResultSourceKey,
    run_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    nodal: tuple[str, ...] = ("all_nodes",)
    element: tuple[str, ...] = ("all_elements",)
    if projection.source_kind == "result":
        archive = provider.model_projection
        if archive is not None:
            nodal = (
                "all_nodes",
                *tuple(archive.named_region_node_ids)[:127],
            )
            element = (
                "all_elements",
                *tuple(archive.named_region_element_ids)[:127],
            )
    elif _result_regions_are_current(projection, source, run_id):
        regions = tuple(projection.named_regions.values())[:127]  # type: ignore[union-attr]
        nodal = (
            "all_nodes",
            *(region.name for region in regions if region.entity_kind in {"node", "edge", "face"}),
        )
        element = (
            "all_elements",
            *(region.name for region in regions if region.entity_kind == "element"),
        )
    return tuple(dict.fromkeys(nodal)), tuple(dict.fromkeys(element))


def _native_result_query(
    provider: ResultProvider,
    field_key: FieldMaterializationKey,
    request: AgentResultQuery,
) -> NativeResultQuery:
    variable = request.variable
    region = request.region
    if variable in {
        AgentResultVariable.DISPLACEMENT,
        AgentResultVariable.ROTATION,
        AgentResultVariable.REACTION_FORCE,
        AgentResultVariable.REACTION_MOMENT,
    }:
        if region == "all_elements":
            raise _AgentResultQueryRejected(
                "result.query.region_entity_unsupported",
                "Nodal result queries cannot target all_elements.",
            )
        node_ids = (
            ()
            if region == "all_nodes"
            else provider.named_region_node_ids(region)
        )
        return NativeResultQuery(
            field_key,
            request.component,
            node_ids=node_ids,
        )
    if region == "all_nodes":
        raise _AgentResultQueryRejected(
            "result.query.region_entity_unsupported",
            "Element result queries cannot target all_nodes.",
        )
    element_ids = (
        ()
        if region == "all_elements"
        else provider.named_region_element_ids(region)
    )
    return NativeResultQuery(
        field_key,
        request.component,
        element_ids=element_ids,
    )


def _aggregate_native_result(
    request: AgentResultQuery,
    records: tuple[ResultQueryRecord, ...],
    source: ResultSourceKey,
    generation: int,
    unit: str,
) -> AgentResultScalar:
    aggregation = request.aggregation
    if aggregation is AgentResultAggregation.SUM:
        value = math.fsum(float(record.value) for record in records)
        location = None
    else:
        selector = {
            AgentResultAggregation.MAXIMUM: lambda record: float(record.value),
            AgentResultAggregation.MINIMUM: lambda record: -float(record.value),
            AgentResultAggregation.ABSOLUTE_EXTREME: (
                lambda record: abs(float(record.value))
            ),
        }[aggregation]
        selected = max(records, key=selector)
        value = float(selected.value)
        location = _agent_result_location(selected.location)
    return AgentResultScalar(
        variable=request.variable,
        component=request.component,
        position=request.position,
        region=request.region,
        aggregation=aggregation,
        value=value,
        unit=unit,
        source=_accepted_source(source),
        materialization_generation=generation,
        location=location,
    )


def _agent_result_location(location: object) -> AgentResultLocation:
    association = getattr(location, "association")
    return AgentResultLocation(
        association=str(getattr(association, "value", association)),
        node_id=getattr(location, "node_id"),
        element_id=getattr(location, "element_id"),
        integration_point=getattr(location, "integration_point"),
        local_node=getattr(location, "local_node"),
    )


def _result_unit(
    units: UnitContext,
    variable: AgentResultVariable,
) -> str:
    return {
        AgentResultVariable.DISPLACEMENT: units.length,
        AgentResultVariable.ROTATION: "rad",
        AgentResultVariable.REACTION_FORCE: units.force,
        AgentResultVariable.REACTION_MOMENT: f"{units.force}*{units.length}",
        AgentResultVariable.SECTION_FORCE: units.force,
        AgentResultVariable.SECTION_MOMENT: f"{units.force}*{units.length}",
        AgentResultVariable.LOGARITHMIC_STRAIN: "1",
        AgentResultVariable.STRESS: units.stress,
    }[variable]


def _accepted_source(source: ResultSourceKey) -> AcceptedResultSource:
    return AcceptedResultSource(
        result_id=source.result_id,
        session_id=source.session_id,
        artifact_id=source.artifact_id,
        model_revision=source.model_revision,
        step_name=source.step_name,
        run_id=source.run_id,
    )


def _result_query_failure(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    clarification_required: bool = False,
) -> AgentResultQueryResponse:
    return AgentResultQueryResponse.failure(
        code,
        message,
        retryable=retryable,
        clarification_required=clarification_required,
    )


def _comparison_query_failure(
    side: str,
    response: AgentResultQueryResponse,
) -> AgentResultComparisonResponse:
    diagnostic = response.diagnostics[0]
    if diagnostic.retryable:
        return _result_comparison_failure(
            "result.comparison.stale",
            f"The {side} accepted result changed or is not ready; retry with a fresh catalog.",
            retryable=True,
        )
    return _result_comparison_failure(
        "result.comparison.not_comparable",
        f"The {side} accepted result cannot answer the common query.",
        clarification_required=diagnostic.clarification_required,
    )


def _result_comparison_failure(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    clarification_required: bool = False,
) -> AgentResultComparisonResponse:
    return AgentResultComparisonResponse.failure(
        code,
        message,
        retryable=retryable,
        clarification_required=clarification_required,
    )


def _result_catalog_failure(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    clarification_required: bool = False,
) -> AgentResultCatalogResponse:
    return AgentResultCatalogResponse.failure(
        code,
        message,
        retryable=retryable,
        clarification_required=clarification_required,
    )


def _ignore_projection_refresh() -> None:
    pass


class SessionGeometryAuthoringPort:
    """A2/A3 port for atomic geometry writes and detached mesh tasks."""

    def __init__(
        self,
        session: ModelSession,
        refresh_projection: Callable[[], None],
        start_mesh_task: Callable[[AgentMeshTaskRequest], bool] | None = None,
        apply_definition_delta: Callable[[SessionDelta], None] | None = None,
        start_solve_task: Callable[[AgentSolveTaskRequest], bool] | None = None,
        start_preflight_task: (
            Callable[[AgentPreflightTaskRequest], bool] | None
        ) = None,
        geometry_edit_mode: Callable[[], str] | None = None,
        commit_geometry_edit: (
            Callable[[AgentGeometryMutation, int], Mapping[str, object]] | None
        ) = None,
    ) -> None:
        if type(session) is not ModelSession:
            raise TypeError("session must be ModelSession")
        if not callable(refresh_projection):
            raise TypeError("refresh_projection must be callable")
        self._session = session
        self._refresh_callback = refresh_projection
        self._start_mesh_task = start_mesh_task
        self._start_solve_task = start_solve_task
        self._start_preflight_task = start_preflight_task
        if geometry_edit_mode is not None and not callable(geometry_edit_mode):
            raise TypeError("geometry_edit_mode must be callable or None")
        if commit_geometry_edit is not None and not callable(commit_geometry_edit):
            raise TypeError("commit_geometry_edit must be callable or None")
        self._geometry_edit_mode_callback = geometry_edit_mode
        self._commit_geometry_edit_callback = commit_geometry_edit
        self._latest_geometry_iteration_report: dict[str, object] | None = None
        self._geometry_iteration_report_binding: tuple[str, str] | None = None
        if apply_definition_delta is not None and not callable(
            apply_definition_delta
        ):
            raise TypeError("apply_definition_delta must be callable or None")
        self._apply_definition_delta = apply_definition_delta
        self._context: AuthoringContext | None = None
        self._records: dict[str, ProposalPortRecord] = {}
        self._mesh_tasks: dict[str, object] = {}
        self._patch_records: dict[str, AppliedPatchRecord] = {}
        self._preflight_records: dict[str, AgentPreflightRecord] = {}
        self._preflight_counter = 0
        self._record_listener: Callable[[ProposalPortRecord], None] | None = None

    @property
    def session(self) -> ModelSession:
        """Return the currently bound model Session."""

        return self._session

    def bind_session(
        self,
        session: ModelSession,
        *,
        idle: Callable[[], bool] | None = None,
    ) -> None:
        """Rebind all authoring writes to one idle document Session."""

        if type(session) is not ModelSession:
            raise TypeError("session must be ModelSession")
        if idle is not None and not idle():
            raise RuntimeError("Agent runtime must be idle before rebinding")
        if session.session_id != self._session.session_id:
            self._clear_geometry_iteration_report()
        self._session = session

    rebind_session = bind_session

    def set_record_listener(
        self,
        callback: Callable[[ProposalPortRecord], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("record listener must be callable")
        self._record_listener = callback

    def detach_gui_callbacks(self) -> None:
        """Release callbacks owned by a closing GUI window."""

        self._refresh_callback = _ignore_projection_refresh
        self._start_mesh_task = None
        self._start_solve_task = None
        self._start_preflight_task = None
        self._apply_definition_delta = None
        self._record_listener = None
        self._geometry_edit_mode_callback = None
        self._commit_geometry_edit_callback = None

    def geometry_edit_mode(self) -> str:
        callback = self._geometry_edit_mode_callback
        mode = "in_place" if callback is None else str(callback())
        if mode not in {"in_place", "branch"}:
            raise AuthoringContractError("invalid geometry edit mode")
        return mode

    def latest_geometry_iteration_report(self) -> Mapping[str, object] | None:
        report = self._latest_geometry_iteration_report
        context = self._context
        binding = None if context is None else (
            context.binding.document_id,
            context.binding.session_id,
        )
        if report is None or binding != self._geometry_iteration_report_binding:
            return None
        return deepcopy(report)

    def _clear_geometry_iteration_report(self) -> None:
        self._latest_geometry_iteration_report = None
        self._geometry_iteration_report_binding = None

    def set_context(self, context: AuthoringContext) -> None:
        if type(context) is not AuthoringContext:
            raise AuthoringContractError("context must be AuthoringContext")
        previous = self._context
        previous_binding = None if previous is None else (
            previous.binding.document_id,
            previous.binding.session_id,
        )
        next_binding = (
            context.binding.document_id,
            context.binding.session_id,
        )
        if previous_binding is not None and previous_binding != next_binding:
            self._clear_geometry_iteration_report()
        self._context = context

    def clear_context(self) -> None:
        """Detach the port from any document that is no longer editable."""

        self._context = None
        self._clear_geometry_iteration_report()

    def present(self, proposal: AgentProposal) -> ProposalPortRecord:
        if proposal.proposal_kind not in {
            ProposalKind.GEOMETRY,
            ProposalKind.MESH,
            ProposalKind.SOLVE,
            ProposalKind.DESTRUCTIVE_EDIT,
        }:
            raise AuthoringContractError(
                "session authoring port does not accept this proposal kind"
            )
        geometry_valid = (
            proposal.proposal_kind is ProposalKind.GEOMETRY
            and len(proposal.operations) == 1
            and proposal.operations[0].kind
            in {
                OperationKind.CREATE_NATIVE_PROJECT,
                OperationKind.ADD_NATIVE_PART,
                OperationKind.REPLACE_PART_GEOMETRY,
                OperationKind.EXTRUDE_PART_PROFILES,
                OperationKind.REVOLVE_PART_PROFILE,
                OperationKind.SWEEP_PART_PROFILE,
                OperationKind.APPLY_PART_BOOLEAN,
                OperationKind.APPLY_BODY_BOOLEAN,
            }
        )
        mesh_valid = (
            proposal.proposal_kind is ProposalKind.MESH
            and tuple(item.kind for item in proposal.operations)
            == (
                OperationKind.SET_PART_MESH_INTENT,
                OperationKind.REQUEST_MESH,
            )
        )
        destructive_valid = (
            proposal.proposal_kind is ProposalKind.DESTRUCTIVE_EDIT
            and (
                tuple(item.kind for item in proposal.operations)
                == (
                    OperationKind.UPSERT_NAMED_REGIONS,
                    OperationKind.UPSERT_MODEL_DEFINITIONS,
                )
                or (
                    len(proposal.operations) == 1
                    and proposal.operations[0].kind
                    in {
                        OperationKind.DELETE_MODEL_OBJECT,
                        OperationKind.EDIT_MODEL_OBJECT,
                    }
                )
            )
        )
        solve_valid = (
            proposal.proposal_kind is ProposalKind.SOLVE
            and len(proposal.operations) == 1
            and proposal.operations[0].kind is OperationKind.REQUEST_SOLVE
            and isinstance(proposal.preconditions, Mapping)
            and proposal.preconditions.get("authoring_phase") == "A6"
        )
        if solve_valid:
            solve_operation_identity(proposal.operations[0])
        if (
            not geometry_valid
            and not mesh_valid
            and not destructive_valid
            and not solve_valid
        ):
            raise AuthoringContractError(
                "proposal operations do not match its authoring kind"
            )
        if proposal.proposal_id in self._records:
            raise AuthoringContractError("proposal_id is already registered")
        record = ProposalPortRecord(
            proposal,
            ProposalState.PENDING_CONFIRMATION,
        )
        self._records[proposal.proposal_id] = record
        return record

    def accept(self, proposal_id: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        proposal = record.proposal
        current_id = self._session.session_id
        current_revision = self._session.session_revision
        bound_context = self._context
        current_document_id = (
            bound_context.binding.document_id
            if bound_context is not None
            else f"document:{current_id}"
        )
        if (
            proposal.target_session_id != current_id
            or proposal.target_document_id != current_document_id
            or proposal.base_session_revision != current_revision
        ):
            raise AuthoringContractError("authoring proposal target is stale")
        if proposal.proposal_kind is ProposalKind.MESH:
            return self._accept_mesh(record)
        if proposal.proposal_kind is ProposalKind.SOLVE:
            return self._accept_solve(record)
        if proposal.proposal_kind is ProposalKind.DESTRUCTIVE_EDIT:
            if len(proposal.operations) == 1:
                operation = proposal.operations[0]
                if operation.kind is OperationKind.DELETE_MODEL_OBJECT:
                    delta = apply_delete_operation(
                        self._session,
                        operation,
                        base_session_revision=proposal.base_session_revision,
                    )
                    if operation.parameters["object_type"] not in {
                        "part",
                        "generated_mesh",
                    }:
                        self._project_definition_delta(delta)
                elif operation.kind is OperationKind.EDIT_MODEL_OBJECT:
                    delta = apply_edit_operation(
                        self._session,
                        operation,
                        base_session_revision=proposal.base_session_revision,
                    )
                    self._project_definition_delta(delta)
                else:
                    raise AuthoringContractError(
                        "unsupported destructive edit operation"
                    )
            else:
                batch = scoped_definition_batch_from_operations(
                    proposal.operations,
                    self._session.snapshot(),
                    base_session_revision=proposal.base_session_revision,
                )
                authoring_mode = proposal.preconditions.get("authoring_mode")
                if authoring_mode == "strict_incremental":
                    require_strict_definition_batch(
                        self._session.snapshot(),
                        batch,
                        proposal.preconditions.get("direct_action"),
                    )
                elif authoring_mode == "direct_incremental":
                    require_incremental_definition_batch(
                        self._session.snapshot(),
                        batch,
                        proposal.preconditions.get("direct_action"),
                    )
                delta = self._session.apply_scoped_definition_batch(batch)
                self._project_definition_delta(delta)
            succeeded = replace(record, state=ProposalState.SUCCEEDED)
            self._records[proposal_id] = succeeded
            return succeeded
        operation = proposal.operations[0]
        parameters = operation.parameters
        if operation.kind is OperationKind.APPLY_PART_BOOLEAN:
            if parameters["tool_handling"] != PART_BOOLEAN_TOOL_HANDLING:
                raise AuthoringContractError(
                    "Part Boolean tool handling does not match canonical policy"
                )
            recipe = geometry_recipe_from_payload(
                json.loads(str(parameters["recipe_json"]))
            )
            if not isinstance(recipe, BooleanGeometry):
                raise AuthoringContractError(
                    "Part Boolean proposal recipe is not BooleanGeometry"
                )

            def commit_part_boolean(
                session: ModelSession,
                expected_revision: int,
            ) -> None:
                session.apply_part_boolean(
                    str(parameters["target_part_id"]),
                    str(parameters["tool_part_id"]),
                    str(parameters["operation"]),
                    str(parameters["result_name"]),
                    result_recipe=recipe,
                    expected_session_revision=expected_revision,
                )

            return self._accept_geometry_mutation(
                record,
                AgentGeometryMutation(
                    operation.kind,
                    (
                        str(parameters["target_part_id"]),
                        str(parameters["tool_part_id"]),
                        str(parameters["result_part_id"]),
                    ),
                    commit_part_boolean,
                ),
            )
        if operation.kind is OperationKind.APPLY_BODY_BOOLEAN:
            if parameters["tool_handling"] != BODY_BOOLEAN_TOOL_HANDLING:
                raise AuthoringContractError(
                    "Body Boolean tool handling does not match canonical policy"
                )
            geometry = geometry_recipe_from_payload(
                json.loads(str(parameters["recipe_json"]))
            )
            if type(geometry) is not MultiBodyGeometry:
                raise AuthoringContractError(
                    "Body Boolean proposal recipe is not MultiBodyGeometry"
                )

            def commit_body_boolean(
                session: ModelSession,
                expected_revision: int,
            ) -> None:
                session.apply_body_boolean(
                    str(parameters["part_id"]),
                    str(parameters["target_body_id"]),
                    str(parameters["tool_body_id"]),
                    str(parameters["operation"]),
                    str(parameters["result_name"]),
                    result=geometry,
                    expected_session_revision=expected_revision,
                )

            return self._accept_geometry_mutation(
                record,
                AgentGeometryMutation(
                    operation.kind,
                    (str(parameters["part_id"]),),
                    commit_body_boolean,
                ),
            )
        if operation.kind is OperationKind.EXTRUDE_PART_PROFILES:
            base_recipe = geometry_recipe_from_payload(parameters["base_recipe"])
            raw_source_ids = parameters["source_face_ids"]
            if not isinstance(raw_source_ids, list) or not raw_source_ids:
                raise AuthoringContractError(
                    "Profile extrusion requires explicit source_face_ids"
                )
            source_face_ids = tuple(str(item) for item in raw_source_ids)
            selection = resolve_extrusion_source_faces(
                base_recipe,
                source_face_ids,
            )
            if selection.face_ids != source_face_ids:
                raise AuthoringContractError(
                    "Profile extrusion sources are not canonical"
                )
            part_id = str(parameters["part_id"])
            snapshot = self._session.snapshot()
            source_part = next(
                (
                    part
                    for part in snapshot.parts
                    if str(part.id) == part_id and not part.suppressed
                ),
                None,
            )
            if source_part is None:
                raise AuthoringContractError(
                    "Profile extrusion source Part no longer matches its proposal"
                )
            source_recipe = _canonical_profile_source_recipe(
                source_part.geometry_recipe
            )
            if source_recipe != base_recipe:
                raise AuthoringContractError(
                    "Profile extrusion source Part no longer matches its proposal"
                )
            recipes = tuple(
                ExtrudedGeometry(
                    base_recipe,
                    parameters["height"],
                    (face_id,),
                )
                for face_id in source_face_ids
            )
            def commit_profile_extrusion(
                session: ModelSession,
                expected_revision: int,
            ) -> None:
                session.replace_part_with_extruded_siblings(
                    part_id,
                    recipes,
                    expected_session_revision=expected_revision,
                )

            return self._accept_geometry_mutation(
                record,
                AgentGeometryMutation(
                    operation.kind,
                    (part_id,),
                    commit_profile_extrusion,
                ),
            )
        if operation.kind in {
            OperationKind.REVOLVE_PART_PROFILE,
            OperationKind.SWEEP_PART_PROFILE,
        }:
            base_recipe = geometry_recipe_from_payload(parameters["base_recipe"])
            source_face_id = str(parameters["source_face_id"])
            selection = resolve_extrusion_source_faces(
                base_recipe,
                (source_face_id,),
            )
            if selection.face_ids != (source_face_id,):
                raise AuthoringContractError(
                    "derived Profile source is not canonical"
                )
            part_id = str(parameters["part_id"])
            snapshot = self._session.snapshot()
            source_part = next(
                (
                    part
                    for part in snapshot.parts
                    if str(part.id) == part_id and not part.suppressed
                ),
                None,
            )
            if source_part is None:
                raise AuthoringContractError(
                    "derived Profile source Part no longer matches its proposal"
                )
            source_recipe = _canonical_profile_source_recipe(
                source_part.geometry_recipe
            )
            if source_recipe != base_recipe:
                raise AuthoringContractError(
                    "derived Profile source Part no longer matches its proposal"
                )
            if operation.kind is OperationKind.REVOLVE_PART_PROFILE:
                recipe = RevolvedGeometry(
                    base_recipe,
                    str(parameters["axis"]),
                    parameters["angle_degrees"],
                    (source_face_id,),
                )
            else:
                path = geometry_recipe_from_payload(parameters["ordered_wire"])
                if type(path) is not WireGeometry:
                    raise AuthoringContractError(
                        "path sweep proposal path is not an explicit WireGeometry"
                    )
                recipe = PathSweptGeometry(
                    base_recipe,
                    path,
                    (source_face_id,),
                    str(parameters["frame_strategy"]),
                )
            def commit_derived_feature(
                session: ModelSession,
                expected_revision: int,
            ) -> None:
                session.replace_part_geometry(
                    part_id,
                    recipe,
                    expected_session_revision=expected_revision,
                )

            return self._accept_geometry_mutation(
                record,
                AgentGeometryMutation(
                    operation.kind,
                    (part_id,),
                    commit_derived_feature,
                ),
            )
        recipe = geometry_recipe_from_payload(parameters["recipe"])
        snapshot = self._session.snapshot()
        if operation.kind is OperationKind.REPLACE_PART_GEOMETRY:
            part_id = str(parameters["part_id"])

            def commit_replacement(
                session: ModelSession,
                expected_revision: int,
            ) -> None:
                session.replace_part_geometry(
                    part_id,
                    recipe,
                    expected_session_revision=expected_revision,
                )

            return self._accept_geometry_mutation(
                record,
                AgentGeometryMutation(
                    operation.kind,
                    (part_id,),
                    commit_replacement,
                ),
            )
        part_name = NameAllocator(
            {"parts": (part.name for part in snapshot.parts)}
        ).require_next(
            "parts",
            "部件",
            str(parameters["part_name"]),
        )
        raw_units = parameters.get("unit_context")
        if not isinstance(raw_units, dict):
            raise AuthoringContractError(
                "geometry proposal requires confirmed unit_context"
            )
        units = UnitContext.from_dict(raw_units)
        if operation.kind is OperationKind.CREATE_NATIVE_PROJECT:
            if snapshot.source_kind is not None:
                raise AuthoringContractError(
                    "create_native_project requires a blank session"
                )
            self._session.create_native_project_with_first_part(
                NameAllocator().require_next(
                    "models",
                    "模型",
                    str(parameters["project_name"]),
                ),
                units,
                recipe,
                part_name=part_name,
                expected_session_revision=proposal.base_session_revision,
            )
        else:
            if snapshot.source_kind != "native":
                raise AuthoringContractError(
                    "add_native_part requires a native project"
                )
            self._session.add_native_part(
                recipe,
                name=part_name,
                mesh_settings=None,
                expected_session_revision=proposal.base_session_revision,
                unit_context=units,
            )
        succeeded = replace(record, state=ProposalState.SUCCEEDED)
        self._records[proposal_id] = succeeded
        return succeeded

    def _accept_geometry_mutation(
        self,
        record: ProposalPortRecord,
        mutation: AgentGeometryMutation,
    ) -> ProposalPortRecord:
        proposal = record.proposal
        planned_mode = str(
            proposal.preconditions.get("geometry_edit_mode", "in_place")
        )
        if planned_mode not in {"in_place", "branch"}:
            raise AuthoringContractError(
                "geometry proposal omitted a valid frozen edit mode"
            )
        if planned_mode != self.geometry_edit_mode():
            raise AuthoringContractError(
                "geometry edit policy changed after proposal creation"
            )
        commit = self._commit_geometry_edit_callback
        if commit is None:
            if planned_mode != "in_place":
                raise AuthoringContractError(
                    "branch geometry edit requires a workspace-aware commit seam"
                )
            mutation.apply(self._session, proposal.base_session_revision)
            context = self._context
            document_id = (
                f"document:{self._session.session_id}"
                if context is None
                else context.binding.document_id
            )
            report = {
                "mode": "in_place",
                "source": {
                    "document_id": document_id,
                    "session_id": self._session.session_id,
                },
                "target": {
                    "document_id": document_id,
                    "session_id": self._session.session_id,
                },
                "part_id": mutation.affected_part_ids[0],
                "affected_part_ids": list(mutation.affected_part_ids),
                "requires_remesh": True,
                "validations": "reset",
                "runs": "not_migrated",
                "results": "not_migrated",
            }
        else:
            report = dict(commit(mutation, proposal.base_session_revision))
        report.setdefault("mode", planned_mode)
        if report.get("mode") != planned_mode:
            raise AuthoringContractError(
                "geometry edit commit mode does not match its proposal"
            )
        provider_safe_authoring_payload(report)
        self._latest_geometry_iteration_report = deepcopy(report)
        context = self._context
        self._geometry_iteration_report_binding = (
            None
            if context is None
            else (
                context.binding.document_id,
                context.binding.session_id,
            )
        )
        succeeded = replace(record, state=ProposalState.SUCCEEDED)
        self._records[proposal.proposal_id] = succeeded
        return succeeded

    def can_accept(self, proposal_id: str) -> bool:
        """Return whether the current Session still satisfies the proposal."""

        try:
            record = self._pending(proposal_id)
            proposal = record.proposal
            if proposal.proposal_kind is not ProposalKind.SOLVE:
                return True
            self._require_current_solve_identity(proposal)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def request_preflight(self, step_name: str) -> AgentPreflightRecord:
        """Start one confirmation-free preflight through the GUI task owner."""

        if self._start_preflight_task is None:
            raise AuthoringContractError(
                "automatic preflight execution is not configured"
            )
        if any(
            item.state is AgentPreflightState.RUNNING
            for item in self._preflight_records.values()
        ):
            raise AuthoringContractError("an Agent preflight is already running")
        snapshot = self._session.snapshot()
        clean_step = str(step_name)
        artifact = snapshot.artifact
        if (
            snapshot.source_kind != "native"
            or not snapshot.model_current
            or not snapshot.mesh_current
            or artifact is None
            or clean_step not in snapshot.runnable_step_names()
        ):
            raise AuthoringContractError(
                "automatic preflight requires a current native model and step"
            )
        self._preflight_counter += 1
        request_id = f"preflight-{self._preflight_counter}"
        record = AgentPreflightRecord(
            request_id=request_id,
            step_name=clean_step,
            base_session_revision=snapshot.session_revision,
            state=AgentPreflightState.RUNNING,
            message="正在后台执行确定性模型预检",
        )
        self._preflight_records[request_id] = record
        request = AgentPreflightTaskRequest(
            request_id=request_id,
            step_name=clean_step,
            base_session_revision=snapshot.session_revision,
            session_id=snapshot.session_id,
            artifact_id=artifact.artifact_id,
            model_revision=snapshot.model_revision,
        )
        try:
            started = self._start_preflight_task(request)
        except Exception:
            self._preflight_records[request_id] = replace(
                record,
                state=AgentPreflightState.FAILED,
                message="GUI 预检任务启动失败",
            )
            raise
        if not started:
            failed = replace(
                record,
                state=AgentPreflightState.FAILED,
                message="GUI 后台任务控制器忙或拒绝启动",
            )
            self._preflight_records[request_id] = failed
            return failed
        return self._preflight_records[request_id]

    def complete_preflight(
        self,
        request_id: str,
        state: AgentPreflightState,
        message: str = "",
    ) -> AgentPreflightRecord:
        if state is AgentPreflightState.RUNNING:
            raise ValueError("preflight completion requires a terminal state")
        record = self._preflight_records[str(request_id)]
        if record.state is not AgentPreflightState.RUNNING:
            raise AuthoringAuthorizationError(
                f"preflight is already {record.state.value}"
            )
        stamp = None
        if state is AgentPreflightState.PASSED:
            stamp = validation_stamp_for_snapshot(
                self._session.snapshot(),
                record.step_name,
            )
        completed = replace(
            record,
            state=state,
            message=str(message).strip(),
            validation_stamp=stamp,
        )
        self._preflight_records[str(request_id)] = completed
        return completed

    def preflight_record(self, request_id: str) -> AgentPreflightRecord:
        try:
            return self._preflight_records[str(request_id)]
        except KeyError as error:
            raise AuthoringContractError(
                "automatic preflight request is not registered"
            ) from error

    def apply_patch(self, patch: ModelPatch) -> AppliedPatchRecord:
        """Apply one non-destructive A4/A5 patch and retain its exact inverse."""

        if type(patch) is not ModelPatch:
            raise TypeError("patch must be ModelPatch")
        current = self._patch_records.get(patch.patch_id)
        if current is not None:
            if current.patch.patch_hash != patch.patch_hash:
                raise AuthoringContractError(
                    "patch_id was reused with different content"
                )
            return replace(current, replayed=True)
        snapshot = self._session.snapshot()
        current_document_id = (
            self._context.binding.document_id
            if self._context is not None
            else f"document:{snapshot.session_id}"
        )
        if (
            patch.target_session_id != snapshot.session_id
            or patch.target_document_id
            != current_document_id
            or patch.base_session_revision != snapshot.session_revision
        ):
            raise AuthoringContractError("automatic patch target is stale")
        preconditions = patch.preconditions
        authoring_mode = (
            preconditions.get("authoring_mode")
            if isinstance(preconditions, Mapping)
            else None
        )
        has_results = any(
            bool(getattr(run, "has_result", False)) for run in snapshot.runs
        )
        if (
            has_results
            and patch.invalidation_impact.get("results") is True
        ):
            raise AuthoringAuthorizationError(
                "a result-invalidating edit requires GUI confirmation"
            )
        inverse_operations = inverse_operations_for_snapshot(snapshot)
        if authoring_mode == "direct_edit":
            if (
                len(patch.operations) != 1
                or patch.operations[0].kind
                is not OperationKind.EDIT_MODEL_OBJECT
            ):
                raise AuthoringContractError(
                    "direct edit patch requires one edit operation"
                )
            delta = apply_edit_operation(
                self._session,
                patch.operations[0],
                base_session_revision=patch.base_session_revision,
            )
        else:
            batch = scoped_definition_batch_from_operations(
                patch.operations,
                snapshot,
                base_session_revision=patch.base_session_revision,
            )
            if authoring_mode == "direct_incremental":
                require_incremental_definition_batch(
                    snapshot,
                    batch,
                    preconditions.get("direct_action"),
                )
            elif authoring_mode == "strict_incremental":
                require_strict_definition_batch(
                    snapshot,
                    batch,
                    preconditions.get("direct_action"),
                )
            elif (
                isinstance(preconditions, Mapping)
                and preconditions.get("authoring_phase") == "A5"
            ):
                require_non_destructive_a5_batch(snapshot, batch)
            else:
                require_non_destructive_a4_batch(snapshot, batch)
            delta = self._session.apply_scoped_definition_batch(batch)
        inverse = ModelPatch.create(
            patch_id=f"inverse-{patch.patch_hash[:24]}",
            agent_session_id=patch.agent_session_id,
            turn_id=patch.turn_id,
            source_tool_call_ids=patch.source_tool_call_ids,
            target_document_id=patch.target_document_id,
            target_session_id=patch.target_session_id,
            base_session_revision=delta.session_revision,
            draft_revision=patch.draft_revision,
            operations=inverse_operations,
            preconditions={
                "forward_patch_hash": patch.patch_hash,
                "expected_session_revision": delta.session_revision,
                "one_shot": True,
            },
            expected_changes={"restore_exact_pre_state": True},
            invalidation_impact={
                "model": True,
                "validation": True,
                "results": False,
                "historical_results_retained": True,
                "current_validation_reset": True,
                "current_result_display_reset": True,
            },
            display_summary={
                "title": "撤销本次 Agent 修改",
                "forward_patch_id": patch.patch_id,
            },
        )
        applied = AppliedPatchRecord(
            patch,
            inverse,
            delta.session_revision,
            delta,
        )
        self._patch_records[patch.patch_id] = applied
        self._project_definition_delta(delta)
        return applied

    def can_undo_patch(self, patch_id: str) -> bool:
        record = self._patch_records.get(str(patch_id))
        return (
            record is not None
            and record.state is AppliedPatchState.APPLIED
            and self._session.session_id
            == record.inverse_patch.target_session_id
            and self._session.session_revision
            == record.inverse_patch.base_session_revision
        )

    def undo_patch(self, patch_id: str) -> AppliedPatchRecord:
        """Apply an inverse exactly once while its post revision is current."""

        try:
            record = self._patch_records[str(patch_id)]
        except KeyError as error:
            raise AuthoringContractError(
                "automatic patch is not registered"
            ) from error
        if record.state is not AppliedPatchState.APPLIED:
            raise AuthoringAuthorizationError(
                f"automatic patch is already {record.state.value}"
            )
        if not self.can_undo_patch(patch_id):
            stale = replace(
                record,
                state=AppliedPatchState.STALE,
                message="session revision changed after the Agent patch",
            )
            self._patch_records[str(patch_id)] = stale
            raise AuthoringAuthorizationError(stale.message)
        snapshot = self._session.snapshot()
        inverse = record.inverse_patch
        batch = scoped_definition_batch_from_operations(
            inverse.operations,
            snapshot,
            base_session_revision=inverse.base_session_revision,
        )
        delta = self._session.apply_scoped_definition_batch(batch)
        undone = replace(
            record,
            session_revision=delta.session_revision,
            delta=delta,
            state=AppliedPatchState.UNDONE,
            message="Agent patch inverse applied",
        )
        self._patch_records[str(patch_id)] = undone
        self._project_definition_delta(delta)
        return undone

    def patch_record(self, patch_id: str) -> AppliedPatchRecord:
        try:
            return self._patch_records[str(patch_id)]
        except KeyError as error:
            raise AuthoringContractError(
                "automatic patch is not registered"
            ) from error

    def _project_definition_delta(self, delta: SessionDelta) -> None:
        if self._apply_definition_delta is not None:
            self._apply_definition_delta(delta)

    def _accept_mesh(self, record: ProposalPortRecord) -> ProposalPortRecord:
        if self._start_mesh_task is None:
            raise AuthoringContractError(
                "mesh proposal execution is not configured"
            )
        proposal = record.proposal
        intent_operation, request_operation = proposal.operations
        intent = MeshIntent.from_dict(
            intent_operation.parameters["mesh_intent"]  # type: ignore[arg-type]
        )
        part_id = str(intent_operation.parameters["part_id"])
        if (
            str(request_operation.parameters["part_id"]) != part_id
            or str(request_operation.parameters["mesh_intent_hash"])
            != intent.intent_hash
        ):
            raise AuthoringContractError(
                "mesh request does not match its retained MeshIntent"
            )
        snapshot = self._session.snapshot()
        part = next(
            (item for item in snapshot.parts if item.id == part_id),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError("mesh proposal Part is unavailable")
        settings = intent.to_mesh_settings(part.geometry_recipe)
        task = self._session.prepare_agent_mesh_generation(
            part_id,
            settings,
            intent.intent_hash,
            expected_session_revision=proposal.base_session_revision,
        )
        self._mesh_tasks[proposal.proposal_id] = task
        running = replace(record, state=ProposalState.RUNNING)
        self._records[proposal.proposal_id] = running
        try:
            started = self._start_mesh_task(
                AgentMeshTaskRequest(
                    proposal.proposal_id,
                    proposal.proposal_hash,
                    task,
                )
            )
        except Exception as error:
            self._session.terminate_agent_mesh_task(task.token, str(error))
            self._mesh_tasks.pop(proposal.proposal_id, None)
            raise
        if not started:
            self._session.terminate_agent_mesh_task(
                task.token,
                "GUI background task controller is busy",
            )
            self._mesh_tasks.pop(proposal.proposal_id, None)
            failed = replace(
                running,
                state=ProposalState.FAILED,
                message="GUI background task controller is busy",
            )
            self._records[proposal.proposal_id] = failed
            raise AuthoringContractError(failed.message)
        return self._records[proposal.proposal_id]

    def _accept_solve(
        self,
        record: ProposalPortRecord,
    ) -> ProposalPortRecord:
        if self._start_solve_task is None:
            raise AuthoringContractError(
                "solve proposal execution is not configured"
            )
        proposal = record.proposal
        (
            step_name,
            job_name,
            artifact_id,
            model_revision,
            stamp,
        ) = self._require_current_solve_identity(proposal)
        running = replace(
            record,
            state=ProposalState.RUNNING,
            message="GUI 已授权，正在提交后台作业",
        )
        self._records[proposal.proposal_id] = running
        if self._record_listener is not None:
            self._record_listener(running)
        request = AgentSolveTaskRequest(
            proposal_id=proposal.proposal_id,
            proposal_hash=proposal.proposal_hash,
            step_name=step_name,
            job_name=job_name,
            base_session_revision=proposal.base_session_revision,
            artifact_id=artifact_id,
            model_revision=model_revision,
            validation_stamp=stamp,
        )
        try:
            started = self._start_solve_task(request)
        except Exception:
            self._records[proposal.proposal_id] = replace(
                running,
                state=ProposalState.FAILED,
                message="GUI 后台作业启动失败",
            )
            raise
        if not started:
            failed = replace(
                running,
                state=ProposalState.FAILED,
                message="GUI 后台任务控制器忙或拒绝启动",
            )
            self._records[proposal.proposal_id] = failed
            raise AuthoringContractError(failed.message)
        return self._records[proposal.proposal_id]

    def _require_current_solve_identity(
        self,
        proposal: AgentProposal,
    ) -> tuple[str, str, str, int, SolveValidationStamp]:
        identity = solve_operation_identity(proposal.operations[0])
        step_name, job_name, artifact_id, model_revision, stamp = identity
        snapshot = self._session.snapshot()
        current_stamp = validation_stamp_for_snapshot(snapshot, step_name)
        current_document_id = (
            self._context.binding.document_id
            if self._context is not None
            else f"document:{snapshot.session_id}"
        )
        if (
            proposal.target_session_id != snapshot.session_id
            or proposal.target_document_id
            != current_document_id
            or proposal.base_session_revision != snapshot.session_revision
            or artifact_id
            != getattr(snapshot.artifact, "artifact_id", None)
            or model_revision != snapshot.model_revision
            or stamp != current_stamp
        ):
            raise AuthoringContractError(
                "solve proposal revision or validation stamp is stale"
            )
        if self._session.find_run(job_name) is not None:
            raise AuthoringContractError("solve proposal job name is already used")
        return identity

    def progress_solve(
        self,
        proposal_id: str,
        message: str,
    ) -> ProposalPortRecord:
        record = self._records[str(proposal_id)]
        if record.state is not ProposalState.RUNNING:
            raise AuthoringAuthorizationError(
                "solve progress requires a running proposal"
            )
        updated = replace(record, message=str(message).strip())
        self._records[str(proposal_id)] = updated
        if self._record_listener is not None:
            self._record_listener(updated)
        return updated

    def complete_solve(
        self,
        proposal_id: str,
        state: ProposalState,
        message: str = "",
    ) -> ProposalPortRecord:
        if state not in {
            ProposalState.SUCCEEDED,
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
        }:
            raise ValueError("solve completion state must be terminal")
        record = self._records[str(proposal_id)]
        if record.state is not ProposalState.RUNNING:
            raise AuthoringAuthorizationError(
                f"solve proposal is already {record.state.value}"
            )
        completed = replace(
            record,
            state=state,
            message=str(message).strip(),
        )
        self._records[str(proposal_id)] = completed
        if self._record_listener is not None:
            self._record_listener(completed)
        return completed

    def accept_mesh_result(
        self,
        proposal_id: str,
        model: object,
    ) -> SessionDelta:
        record = self._records.get(proposal_id)
        if record is None or record.state is not ProposalState.RUNNING:
            raise AuthoringAuthorizationError(
                "mesh result requires a running proposal"
            )
        task = self._mesh_tasks[proposal_id]
        delta = self._session.accept_agent_generated_model(task.token, model)
        if delta.accepted:
            self.complete_mesh(
                proposal_id,
                ProposalState.SUCCEEDED,
                "网格意图和生成模型已原子提交",
            )
        return delta

    def terminate_mesh(
        self,
        proposal_id: str,
        state: ProposalState,
        message: str,
    ) -> SessionDelta:
        if state not in {
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
        }:
            raise ValueError("terminated mesh proposal requires a failure state")
        task = self._mesh_tasks[proposal_id]
        delta = self._session.terminate_agent_mesh_task(task.token, message)
        self.complete_mesh(proposal_id, state, message)
        return delta

    def refresh_projection(self) -> None:
        self._refresh_callback()

    def reject(self, proposal_id: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        rejected = replace(record, state=ProposalState.REJECTED)
        self._records[proposal_id] = rejected
        return rejected

    def cancel(self, proposal_id: str, reason: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        cancelled = replace(
            record,
            state=ProposalState.CANCELLED,
            message=str(reason).strip(),
        )
        self._records[proposal_id] = cancelled
        return cancelled

    def stale(self, proposal_id: str, reason: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        stale = replace(
            record,
            state=ProposalState.STALE,
            message=str(reason).strip(),
        )
        self._records[proposal_id] = stale
        return stale

    def mark_failed(self, proposal_id: str, message: str) -> ProposalPortRecord:
        record = self._records[proposal_id]
        failed = replace(
            record,
            state=ProposalState.FAILED,
            message=str(message).strip(),
        )
        self._records[proposal_id] = failed
        return failed

    def complete_mesh(
        self,
        proposal_id: str,
        state: ProposalState,
        message: str = "",
    ) -> ProposalPortRecord:
        if state not in {
            ProposalState.SUCCEEDED,
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
        }:
            raise ValueError("mesh completion state must be terminal")
        try:
            record = self._records[proposal_id]
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc
        if record.state is not ProposalState.RUNNING:
            raise AuthoringAuthorizationError(
                f"mesh proposal is already {record.state.value}"
            )
        completed = replace(
            record,
            state=state,
            message=str(message).strip(),
        )
        self._records[proposal_id] = completed
        self._mesh_tasks.pop(proposal_id, None)
        if self._record_listener is not None:
            self._record_listener(completed)
        return completed

    def _pending(self, proposal_id: str) -> ProposalPortRecord:
        try:
            record = self._records[proposal_id]
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc
        if record.state is not ProposalState.PENDING_CONFIRMATION:
            raise AuthoringAuthorizationError(
                f"proposal is already {record.state.value}"
            )
        return record


class AgentAuthoringBridge:
    """A1 single write boundary with a Fake Port and no model writes."""

    def __init__(self, port: AuthoringPort) -> None:
        self._port = port
        self._context: AuthoringContext | None = None
        self._records: dict[str, ProposalPortRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._patch_idempotency: dict[str, str] = {}
        self._authorization_nonce = 0
        self._unused_authorizations: set[_GuiControlAuthorization] = set()
        self._accepting_proposal_id: str | None = None
        self._gui_thread_id = threading.get_ident()
        self._patch_listener: Callable[[AppliedPatchRecord], None] | None = None
        self._result_invalidation_confirmation: Callable[[], bool] | None = None
        self._lifecycle_listener: (
            Callable[[AgentProposal, ProposalState, str], None] | None
        ) = None
        self._preview_listener: (
            Callable[[str, AgentProposalPreview | None], None] | None
        ) = None
        self._proposal_previews: dict[str, AgentProposalPreview] = {}
        self._last_lifecycle_notice: dict[str, tuple[ProposalState, str]] = {}
        listener = getattr(port, "set_record_listener", None)
        if callable(listener):
            listener(self._receive_port_record)

    @property
    def context(self) -> AuthoringContext | None:
        return self._context

    @property
    def port(self) -> AuthoringPort:
        return self._port

    def set_patch_listener(
        self,
        callback: Callable[[AppliedPatchRecord], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("patch listener must be callable")
        self._patch_listener = callback

    def set_lifecycle_listener(
        self,
        callback: Callable[[AgentProposal, ProposalState, str], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("lifecycle listener must be callable")
        self._lifecycle_listener = callback

    def set_preview_listener(
        self,
        callback: Callable[[str, AgentProposalPreview | None], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("preview listener must be callable")
        self._preview_listener = callback

    def set_result_invalidation_confirmation(
        self,
        callback: Callable[[], bool],
    ) -> None:
        """Install the GUI's canonical unsaved-result confirmation gate."""

        if not callable(callback):
            raise TypeError("result invalidation confirmation must be callable")
        self._result_invalidation_confirmation = callback

    def detach_gui_callbacks(self) -> None:
        """Break GUI-owned callback cycles when the window closes."""

        self.clear_proposal_previews()
        self._patch_listener = None
        self._lifecycle_listener = None
        self._preview_listener = None
        self._result_invalidation_confirmation = None
        detach_port = getattr(self._port, "detach_gui_callbacks", None)
        if callable(detach_port):
            detach_port()

    def bind_snapshot(
        self,
        snapshot: _SessionSnapshot,
        *,
        document_id: str | int | None = None,
    ) -> tuple[str, ...]:
        return self.bind_context(
            authoring_context_from_snapshot(snapshot, document_id=document_id)
        )

    def bind_context(self, context: AuthoringContext) -> tuple[str, ...]:
        if type(context) is not AuthoringContext:
            raise AuthoringContractError("context must be AuthoringContext")
        prior_binding = None if self._context is None else self._context.binding
        self._context = context
        self._port.set_context(context)
        if prior_binding is None or prior_binding == context.binding:
            return ()

        stale_ids: list[str] = []
        for proposal_id, record in tuple(self._records.items()):
            if (
                record.state is ProposalState.PENDING_CONFIRMATION
                and proposal_id != self._accepting_proposal_id
            ):
                stale = self._port.stale(
                    proposal_id,
                    "绑定文档、session 或 revision 已改变",
                )
                self._records[proposal_id] = stale
                self._notify_lifecycle(stale)
                stale_ids.append(proposal_id)
        return tuple(stale_ids)

    def unbind_context(self, reason: str) -> tuple[str, ...]:
        """Invalidate Agent authoring state when no editable model is bound."""

        self._require_gui_thread()
        message = str(reason).strip()
        if not message:
            raise ValueError("unbind reason must be non-blank")
        stale_ids = self.stale_pending_proposals_from_gui(message)
        clear_context = getattr(self._port, "clear_context", None)
        if callable(clear_context):
            clear_context()
        self._context = None
        return stale_ids

    def stale_pending_proposals_from_gui(
        self,
        reason: str,
    ) -> tuple[str, ...]:
        """Make every old Agent-session proposal terminal on the GUI owner."""

        self._require_gui_thread()
        message = str(reason).strip()
        if not message:
            raise ValueError("stale reason must be non-blank")
        stale_ids: list[str] = []
        for proposal_id, record in tuple(self._records.items()):
            if record.state is not ProposalState.PENDING_CONFIRMATION:
                continue
            stale = self._port.stale(proposal_id, message)
            self._records[proposal_id] = stale
            self._notify_lifecycle(stale)
            stale_ids.append(proposal_id)
        return tuple(stale_ids)

    def cancel_pending_proposals_from_gui(
        self,
        reason: str,
    ) -> tuple[str, ...]:
        self._require_gui_thread()
        message = str(reason).strip()
        if not message:
            raise ValueError("cancel reason must be non-blank")
        cancelled_ids: list[str] = []
        for proposal_id, record in tuple(self._records.items()):
            if record.state is not ProposalState.PENDING_CONFIRMATION:
                continue
            cancelled = self._port.cancel(proposal_id, message)
            self._records[proposal_id] = cancelled
            self._notify_lifecycle(cancelled)
            cancelled_ids.append(proposal_id)
        return tuple(cancelled_ids)

    def register_proposal(
        self,
        proposal: AgentProposal,
        detached_preview: AgentProposalPreview | None = None,
    ) -> BridgeReceipt:
        if type(proposal) is not AgentProposal:
            raise AuthoringContractError("proposal must be AgentProposal")
        current = self._records.get(proposal.proposal_id)
        if current is not None:
            if current.proposal.proposal_hash != proposal.proposal_hash:
                raise AuthoringContractError(
                    "proposal_id was reused with different content"
                )
            return self._receipt(current, replayed=True)
        replay_id = self._idempotency.get(proposal.idempotency_key)
        if replay_id is not None:
            return self._receipt(self._records[replay_id], replayed=True)
        if detached_preview is not None:
            evidence = proposal.preconditions.get("local_evidence")
            if not isinstance(evidence, Mapping) or (
                evidence.get("output_recipe_digest") != detached_preview.recipe_digest
                or evidence.get("output_proof_digest") != detached_preview.proof_digest
            ):
                raise AuthoringContractError(
                    "detached preview does not match the proposal Recipe proof"
                )
        self._require_live_target(proposal)
        record = self._port.present(proposal)
        self._records[proposal.proposal_id] = record
        self._idempotency[proposal.idempotency_key] = proposal.proposal_id
        if detached_preview is not None:
            self._proposal_previews[proposal.proposal_id] = detached_preview
            if self._preview_listener is not None:
                self._preview_listener(proposal.proposal_id, detached_preview)
        return self._receipt(record)

    def clear_proposal_previews(self) -> None:
        for proposal_id in tuple(self._proposal_previews):
            self._clear_proposal_preview(proposal_id)

    def _clear_proposal_preview(self, proposal_id: str) -> None:
        if self._proposal_previews.pop(proposal_id, None) is not None:
            if self._preview_listener is not None:
                self._preview_listener(proposal_id, None)

    def apply_automatic_patch(
        self,
        patch: ModelPatch,
    ) -> AppliedPatchRecord:
        """Apply a revision-bound reversible edit without a confirmation click."""

        if type(patch) is not ModelPatch:
            raise TypeError("patch must be ModelPatch")
        replay_id = self._patch_idempotency.get(patch.idempotency_key)
        if replay_id is not None:
            record_getter = getattr(self._port, "patch_record")
            record = record_getter(replay_id)
            if record.patch.patch_hash != patch.patch_hash:
                raise AuthoringContractError(
                    "patch idempotency key was reused with different content"
                )
            return replace(record, replayed=True)
        self._require_live_patch_target(patch)
        port_apply = getattr(self._port, "apply_patch", None)
        if not callable(port_apply):
            raise AuthoringContractError(
                "authoring port does not support automatic patches"
            )
        record = port_apply(patch)
        self._patch_idempotency[patch.idempotency_key] = patch.patch_id
        if self._patch_listener is not None:
            self._patch_listener(record)
        return record

    def can_undo_patch(self, patch_id: str) -> bool:
        check = getattr(self._port, "can_undo_patch", None)
        return bool(callable(check) and check(patch_id))

    def undo_patch_from_gui_control(
        self,
        patch_id: str,
    ) -> AppliedPatchRecord:
        self._require_gui_thread()
        undo = getattr(self._port, "undo_patch", None)
        if not callable(undo):
            raise AuthoringContractError(
                "authoring port does not support patch inverse"
            )
        record = undo(patch_id)
        if self._patch_listener is not None:
            self._patch_listener(record)
        return record

    def accept_proposal(
        self,
        proposal_id: str,
        authorization: object | None = None,
    ) -> BridgeReceipt:
        token = self._consume_gui_authorization(
            proposal_id,
            "accept",
            authorization,
        )
        del token
        record = self._pending_record(proposal_id)
        self._require_live_target(record.proposal)
        self._accepting_proposal_id = proposal_id
        try:
            accepted = self._port.accept(proposal_id)
        except Exception as exc:
            failed = replace(
                record,
                state=ProposalState.FAILED,
                message=str(exc).strip() or type(exc).__name__,
            )
            marker = getattr(self._port, "mark_failed", None)
            if callable(marker):
                failed = marker(proposal_id, failed.message)
            self._records[proposal_id] = failed
            self._notify_lifecycle(failed)
            return self._receipt(failed)
        finally:
            self._accepting_proposal_id = None
        self._records[proposal_id] = accepted
        self._notify_lifecycle(accepted)
        if (
            accepted.state is ProposalState.SUCCEEDED
            and (
                accepted.proposal.proposal_kind is ProposalKind.GEOMETRY
                or (
                    accepted.proposal.proposal_kind
                    is ProposalKind.DESTRUCTIVE_EDIT
                    and len(accepted.proposal.operations) == 1
                    and accepted.proposal.operations[0].kind
                    in {
                        OperationKind.DELETE_MODEL_OBJECT,
                        OperationKind.EDIT_MODEL_OBJECT,
                    }
                )
            )
        ):
            refresh = getattr(self._port, "refresh_projection", None)
            if callable(refresh):
                refresh()
        return self._receipt(accepted)

    def reject_proposal(
        self,
        proposal_id: str,
        authorization: object | None = None,
    ) -> BridgeReceipt:
        token = self._consume_gui_authorization(
            proposal_id,
            "reject",
            authorization,
        )
        del token
        self._pending_record(proposal_id)
        rejected = self._port.reject(proposal_id)
        self._records[proposal_id] = rejected
        self._notify_lifecycle(rejected)
        return self._receipt(rejected)

    def accept_from_gui_control(self, proposal_id: str) -> BridgeReceipt:
        self._require_gui_thread()
        record = self._pending_record(proposal_id)
        if record.proposal.invalidation_impact.get("results") is True:
            confirmation = self._result_invalidation_confirmation
            if confirmation is not None and not confirmation():
                raise AuthoringAuthorizationError(
                    "result-invalidating proposal was cancelled by the user"
                )
        authorization = self._issue_gui_authorization(
            proposal_id,
            "accept",
        )
        return self.accept_proposal(proposal_id, authorization)

    def can_accept_from_gui_control(self, proposal_id: str) -> bool:
        """Return the exact local gate used to enable a GUI accept button."""

        try:
            record = self._pending_record(proposal_id)
            self._require_live_target(record.proposal)
            port_check = getattr(self._port, "can_accept", None)
            return not callable(port_check) or bool(port_check(proposal_id))
        except (
            AuthoringAuthorizationError,
            AuthoringContractError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return False

    def request_preflight(
        self,
        step_name: str,
    ) -> AgentPreflightRecord:
        """Start a local read-only preflight without a confirmation token."""

        self._require_gui_thread()
        request = getattr(self._port, "request_preflight", None)
        if not callable(request):
            raise AuthoringContractError(
                "authoring port does not support automatic preflight"
            )
        return request(step_name)

    def reject_from_gui_control(self, proposal_id: str) -> BridgeReceipt:
        self._require_gui_thread()
        authorization = self._issue_gui_authorization(
            proposal_id,
            "reject",
        )
        return self.reject_proposal(proposal_id, authorization)

    def confirm_requirement_review_from_gui(
        self,
        ledger: RequirementLedger,
        review: RequirementReview,
    ) -> RequirementReview:
        self._require_gui_thread()
        return ledger._confirm_review_from_gui(review)

    def reject_requirement_review_from_gui(
        self,
        ledger: RequirementLedger,
        review: RequirementReview,
    ) -> RequirementReview:
        self._require_gui_thread()
        return ledger._reject_review_from_gui(review)

    def state(self, proposal_id: str) -> ProposalState:
        try:
            return self._records[proposal_id].state
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc

    def ensure_display_identity_from_gui(
        self,
        proposal_id: str,
        proposal_hash: str,
        agent_session_id: str,
        turn_id: str,
    ) -> bool:
        """Stale a pending proposal if its rendered identity was substituted."""

        self._require_gui_thread()
        record = self._records.get(str(proposal_id))
        if record is None:
            return False
        proposal = record.proposal
        matches = (
            proposal.proposal_hash == str(proposal_hash)
            and proposal.agent_session_id == str(agent_session_id)
            and proposal.turn_id == str(turn_id)
        )
        if matches:
            return record.state is ProposalState.PENDING_CONFIRMATION
        if record.state is ProposalState.PENDING_CONFIRMATION:
            stale = self._port.stale(
                proposal.proposal_id,
                "proposal hash、Agent session 或 turn identity 不匹配",
            )
            self._records[proposal.proposal_id] = stale
            self._notify_lifecycle(stale)
        return False

    def _require_live_target(self, proposal: AgentProposal) -> None:
        context = self._context
        if context is None:
            raise AuthoringContractError("there is no local model binding")
        binding = context.binding
        if not binding.supported:
            raise AuthoringContractError(
                "the current document cannot be bound for V1 authoring"
            )
        if (
            proposal.target_document_id != binding.document_id
            or proposal.target_session_id != binding.session_id
            or proposal.base_session_revision != binding.session_revision
        ):
            raise AuthoringContractError("proposal target is stale")

    def _require_live_patch_target(self, patch: ModelPatch) -> None:
        context = self._context
        if context is None:
            raise AuthoringContractError("there is no local model binding")
        binding = context.binding
        if (
            not binding.supported
            or patch.target_document_id != binding.document_id
            or patch.target_session_id != binding.session_id
            or patch.base_session_revision != binding.session_revision
        ):
            raise AuthoringContractError("patch target is stale")

    def _pending_record(self, proposal_id: str) -> ProposalPortRecord:
        try:
            record = self._records[proposal_id]
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc
        if record.state is not ProposalState.PENDING_CONFIRMATION:
            raise AuthoringAuthorizationError(
                f"proposal is already {record.state.value}"
            )
        return record

    def _issue_gui_authorization(
        self,
        proposal_id: str,
        action: str,
    ) -> _GuiControlAuthorization:
        self._authorization_nonce += 1
        authorization = _GuiControlAuthorization(
            proposal_id,
            action,
            self._authorization_nonce,
        )
        self._unused_authorizations.add(authorization)
        return authorization

    def _consume_gui_authorization(
        self,
        proposal_id: str,
        action: str,
        authorization: object | None,
    ) -> _GuiControlAuthorization:
        if (
            type(authorization) is not _GuiControlAuthorization
            or authorization not in self._unused_authorizations
            or authorization.proposal_id != proposal_id
            or authorization.action != action
        ):
            raise AuthoringAuthorizationError(
                "a live GUI control authorization is required"
            )
        self._unused_authorizations.remove(authorization)
        return authorization

    def _require_gui_thread(self) -> None:
        if threading.get_ident() != self._gui_thread_id:
            raise AuthoringAuthorizationError(
                "GUI authorization must run on the bridge owner thread"
            )

    def _receive_port_record(self, record: ProposalPortRecord) -> None:
        current = self._records.get(record.proposal.proposal_id)
        if current is None or current.proposal.proposal_hash != record.proposal.proposal_hash:
            raise AuthoringContractError(
                "port lifecycle update does not match a registered proposal"
            )
        self._records[record.proposal.proposal_id] = record
        self._notify_lifecycle(record)

    def _notify_lifecycle(self, record: ProposalPortRecord) -> None:
        notice = (record.state, record.message)
        proposal_id = record.proposal.proposal_id
        if self._last_lifecycle_notice.get(proposal_id) == notice:
            return
        self._last_lifecycle_notice[proposal_id] = notice
        if record.state in {
            ProposalState.SUCCEEDED,
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
            ProposalState.REJECTED,
        }:
            self._clear_proposal_preview(proposal_id)
        if self._lifecycle_listener is not None:
            self._lifecycle_listener(
                record.proposal,
                record.state,
                record.message,
            )

    @staticmethod
    def _receipt(
        record: ProposalPortRecord,
        *,
        replayed: bool = False,
    ) -> BridgeReceipt:
        return BridgeReceipt(
            proposal_id=record.proposal.proposal_id,
            state=record.state,
            message=record.message,
            replayed=replayed,
        )


def create_session_authoring_workflow_controller(
    session: ModelSession,
    authoring_bridge: AgentAuthoringBridge,
    result_bridge: AgentResultQueryBridge,
    *,
    next_job_name: Callable[[], str] | None = None,
    workspace_catalog_bridge: WorkspaceCatalogBridge | None = None,
    create_model_document: Callable[[str | None], str] | None = None,
) -> AuthoringWorkflowController:
    """Wire A1-A7 handlers to one GUI-owner A8 controller."""

    if type(session) is not ModelSession:
        raise TypeError("session must be exactly ModelSession")
    if type(authoring_bridge) is not AgentAuthoringBridge:
        raise TypeError("authoring_bridge must be AgentAuthoringBridge")
    if type(result_bridge) is not AgentResultQueryBridge:
        raise TypeError("result_bridge must be AgentResultQueryBridge")
    if next_job_name is not None and not callable(next_job_name):
        raise TypeError("next_job_name must be callable or None")
    if create_model_document is not None and not callable(create_model_document):
        raise TypeError("create_model_document must be callable or None")
    if (
        workspace_catalog_bridge is not None
        and type(workspace_catalog_bridge) is not WorkspaceCatalogBridge
    ):
        raise TypeError(
            "workspace_catalog_bridge must be WorkspaceCatalogBridge or None"
        )

    def current_context() -> AuthoringContext:
        context = authoring_bridge.context
        if context is None:
            raise AuthoringContractError("there is no current authoring binding")
        return context

    def current_session() -> ModelSession:
        """Resolve the Session through the rebindable authoring port."""

        port = authoring_bridge.port
        session_reader = getattr(port, "session", None)
        if type(session_reader) is not ModelSession:
            raise AuthoringContractError("there is no current model Session")
        return session_reader

    def planned_geometry_edit_mode() -> str:
        reader = getattr(authoring_bridge.port, "geometry_edit_mode", None)
        return "in_place" if not callable(reader) else str(reader())

    def geometry_edit_impact(edit_mode: str, in_place: str) -> str:
        return (
            "确认后创建迭代模型，迁移可保留的网格设置与模型定义；"
            "实际网格、验证、运行和结果不迁移"
            if edit_mode == "branch"
            else in_place
        )

    def resolved_step_name(
        arguments: Mapping[str, object],
        *,
        operation: str,
    ) -> str:
        if set(arguments) - {"step_name"}:
            raise AuthoringContractError(f"{operation} has unknown fields")
        runnable = current_session().runnable_step_names()
        if "step_name" not in arguments:
            if len(runnable) != 1:
                raise AuthoringContractError(
                    f"{operation} requires step_name when multiple steps are runnable"
                )
            return runnable[0]
        value = arguments["step_name"]
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or len(value) > 160
        ):
            raise AuthoringContractError("step_name must be a nonblank string")
        step_name = value
        if step_name not in runnable:
            raise AuthoringContractError(
                f"{operation} requires a current runnable analysis step"
            )
        return step_name

    def envelope(
        controller: AuthoringWorkflowController,
        prefix: str,
    ) -> dict[str, object]:
        return controller.invocation_metadata(prefix)

    def proposal_outcome(
        proposal: AgentProposal,
        *,
        summary: str,
        impact: str,
        confirm_label: str,
        extra_data: Mapping[str, object] | None = None,
        detached_preview: AgentProposalPreview | None = None,
    ) -> AuthoringToolOutcome:
        data = {
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.proposal_hash,
            "state": ProposalState.PENDING_CONFIRMATION.value,
            "proposal_view": {
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.proposal_hash,
                "proposal_kind": proposal.proposal_kind.value,
                "title": str(proposal.display_summary.get("title", summary)),
                "summary": summary,
                "impact": impact,
                "confirm_label": confirm_label,
                "target_document_id": proposal.target_document_id,
                "target_session_id": proposal.target_session_id,
                "base_session_revision": proposal.base_session_revision,
            },
            "continuation_checkpoint": {
                "session_id": proposal.agent_session_id,
                "source_turn_id": proposal.turn_id,
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.proposal_hash,
                "model_revision": proposal.base_session_revision,
                "proposal_kind": proposal.proposal_kind.value,
            },
        }
        if extra_data is not None:
            data.update(dict(extra_data))
        provisional = AuthoringToolOutcome(summary, data)
        provider_safe_authoring_payload(provisional.data)
        receipt = authoring_bridge.register_proposal(
            proposal,
            detached_preview=detached_preview,
        )
        if receipt.state is ProposalState.PENDING_CONFIRMATION:
            return provisional
        final = AuthoringToolOutcome(
            summary,
            {**data, "state": receipt.state.value},
        )
        provider_safe_authoring_payload(final.data)
        return final

    def prepare_geometry(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) != {"part_function", "geometry"}:
            raise AuthoringContractError(
                "prepare_geometry_proposal requires part_function and geometry"
            )
        part_function = str(arguments["part_function"]).strip()
        raw_geometry = arguments["geometry"]
        if not part_function or not isinstance(raw_geometry, Mapping):
            raise AuthoringContractError(
                "part_function and geometry must be non-empty"
            )
        geometry = dict(raw_geometry)
        kind = str(geometry.get("kind", ""))
        _require_geometry_intent_compatibility(part_function, kind)
        recipe_name = (
            f"草图-{part_function}"
            if kind == "planar_profiles"
            else (
                f"线框-{part_function}"
                if kind == "wire"
                else f"实体-{part_function}"
            )
        )
        if kind == "planar_profiles":
            if set(geometry) != {"kind", "profiles"}:
                raise ValueError("planar geometry fields do not match")
            raw_profiles = geometry["profiles"]
            if not isinstance(raw_profiles, list) or not raw_profiles:
                raise ValueError("profiles must be a non-empty array")
            profiles = []
            for raw_profile in raw_profiles:
                if not isinstance(raw_profile, Mapping):
                    raise ValueError("each profile must be an object")
                profiles.append(dict(raw_profile))
            profile_summaries: list[str] = []
            first = profiles[0]
            first_kind = str(first.pop("kind", ""))
            _pop_profile_annotations(first)
            if first_kind == "rectangle":
                if set(first) != {"x", "y", "width", "height"}:
                    raise ValueError("rectangle profile fields do not match")
                draft = planar_sketch_geometry(
                    recipe_name,
                    contours=(
                        SketchRectangle(
                            "material",
                            first["x"],
                            first["y"],
                            first["width"],
                            first["height"],
                        ),
                    ),
                )
                profile_summaries.append(
                    _planar_profile_design_summary(
                        first_kind,
                        first,
                        1,
                    )
                )
            elif first_kind == "circle":
                if set(first) != {"center_x", "center_y", "radius"}:
                    raise ValueError("circle profile fields do not match")
                draft = planar_sketch_geometry(
                    recipe_name,
                    contours=(
                        SketchCircle(
                            "material",
                            first["center_x"],
                            first["center_y"],
                            first["radius"],
                        ),
                    ),
                )
                profile_summaries.append(
                    _planar_profile_design_summary(
                        first_kind,
                        first,
                        1,
                    )
                )
            elif first_kind == "polygon":
                if set(first) != {"vertices"}:
                    raise ValueError("polygon profile fields do not match")
                vertices = _profile_vertices(first["vertices"])
                draft = planar_polygon_geometry(
                    recipe_name,
                    vertices=vertices,
                )
                profile_summaries.append(
                    _planar_profile_design_summary(
                        first_kind,
                        {"vertices": vertices},
                        1,
                    )
                )
            else:
                raise ValueError("unsupported planar profile kind")
            for index, profile in enumerate(profiles[1:], start=2):
                profile_kind = str(profile.pop("kind", ""))
                _pop_profile_annotations(profile)
                if profile_kind == "rectangle":
                    if set(profile) != {"x", "y", "width", "height"}:
                        raise ValueError(
                            "rectangle profile fields do not match"
                        )
                    draft = add_planar_rectangle(draft.recipe, **profile)
                    profile_summaries.append(
                        _planar_profile_design_summary(
                            profile_kind,
                            profile,
                            index,
                        )
                    )
                elif profile_kind == "circle":
                    if set(profile) != {"center_x", "center_y", "radius"}:
                        raise ValueError("circle profile fields do not match")
                    draft = add_planar_circle(draft.recipe, **profile)
                    profile_summaries.append(
                        _planar_profile_design_summary(
                            profile_kind,
                            profile,
                            index,
                        )
                    )
                elif profile_kind == "polygon":
                    if set(profile) != {"vertices"}:
                        raise ValueError("polygon profile fields do not match")
                    vertices = _profile_vertices(profile["vertices"])
                    draft = add_planar_polygon(
                        draft.recipe,
                        vertices=vertices,
                    )
                    profile_summaries.append(
                        _planar_profile_design_summary(
                            profile_kind,
                            {"vertices": vertices},
                            index,
                        )
                    )
                else:
                    raise ValueError("unsupported planar profile kind")
            geometry_summary = _bounded_geometry_design_summary(
                "2D 平面轮廓",
                profile_summaries,
            )
        elif kind == "extruded_path_slot_plate":
            allowed = {
                "kind",
                "plate",
                "slot_path",
                "slot_width",
                "height",
                "provisional",
            }
            required = allowed - {"provisional"}
            if set(geometry) - allowed or not required <= set(geometry):
                raise ValueError(
                    "extruded_path_slot_plate fields do not match"
                )
            if "provisional" in geometry and type(geometry["provisional"]) is not bool:
                raise TypeError("path slot provisional must be a boolean")
            raw_plate = geometry["plate"]
            if not isinstance(raw_plate, Mapping) or set(raw_plate) != {
                "x",
                "y",
                "width",
                "height",
            }:
                raise ValueError("path slot plate fields do not match")
            raw_path = geometry["slot_path"]
            if (
                not isinstance(raw_path, list)
                or not 2 <= len(raw_path) <= 32
                or any(
                    not isinstance(item, Mapping) or set(item) != {"x", "y"}
                    for item in raw_path
                )
            ):
                raise ValueError(
                    "slot_path must contain 2 to 32 ordered x/y objects"
                )
            path_points = tuple(
                (item["x"], item["y"])
                for item in raw_path
            )
            slot_vertices = planar_path_slot_vertices(
                path_points,
                geometry["slot_width"],
            )
            profiles = [
                {
                    "kind": "rectangle",
                    "x": raw_plate["x"],
                    "y": raw_plate["y"],
                    "width": raw_plate["width"],
                    "height": raw_plate["height"],
                    "role": "material",
                },
                {
                    "kind": "polygon",
                    "vertices": [
                        {"x": x, "y": y} for x, y in slot_vertices
                    ],
                    "role": "hole",
                },
            ]
            sketch, profile_context, source_face_ids = _composite_profile_contours(
                profiles,
                recipe_name=recipe_name,
            )
            height = float(geometry["height"])
            if not math.isfinite(height) or height <= 0.0:
                raise ValueError("path slot extrusion height must be positive")
            recipe = ExtrudedGeometry(
                sketch,
                height,
                tuple(source_face_ids),
            )
            _run_profile_transform_preflight(
                _preflight_composite_geometry,
                recipe,
            )
            draft = geometry_draft(recipe)
            length_unit = str(
                controller.collected_requirements("geometry").get(
                    "length_unit",
                    "mm",
                )
            )
            shown_path = ", ".join(
                f"({_display_number(x)}, {_display_number(y)})"
                for x, y in path_points[:8]
            )
            if len(path_points) > 8:
                shown_path += f", …共{len(path_points)}点"
            provisional = bool(geometry.get("provisional", False))
            path_slot_details = [
                _planar_profile_design_summary(
                    "rectangle",
                    raw_plate,
                    1,
                ),
                f"槽中心路径={shown_path}",
                (
                    f"槽宽={_display_number(geometry['slot_width'])} "
                    f"{length_unit}"
                ),
                f"拉伸高={_display_number(height)} {length_unit}",
                "holes=1",
            ]
            if provisional:
                path_slot_details.append("尺寸标记为 provisional 提案值")
            geometry_summary = _bounded_geometry_design_summary(
                "3D 路径槽平板",
                path_slot_details,
            )
        elif kind in {"extruded_profiles", "path_swept_profile"}:
            allowed = {
                "kind",
                "profiles",
                "height",
                "path",
                "frame_strategy",
                "provisional",
            }
            if set(geometry) - allowed:
                raise ValueError("composite geometry fields do not match")
            if "profiles" not in geometry:
                raise ValueError("composite geometry requires profiles")
            if "provisional" in geometry and type(geometry["provisional"]) is not bool:
                raise TypeError("composite provisional must be a boolean")
            if kind == "extruded_profiles" and (
                "height" not in geometry
                or "path" in geometry
                or "frame_strategy" in geometry
            ):
                raise ValueError("extruded_profiles requires profiles and height")
            if kind == "path_swept_profile" and (
                "path" not in geometry
                or "frame_strategy" not in geometry
                or "height" in geometry
            ):
                raise ValueError(
                    "path_swept_profile requires profiles, path, and frame_strategy"
                )
            raw_profiles = geometry["profiles"]
            sketch, profile_context, source_face_ids = _composite_profile_contours(
                raw_profiles,
                recipe_name=recipe_name,
            )
            length_unit = str(
                controller.collected_requirements("geometry").get(
                    "length_unit",
                    "mm",
                )
            )
            if kind == "extruded_profiles":
                height = float(geometry["height"])
                if not math.isfinite(height) or height <= 0.0:
                    raise ValueError("composite extrusion height must be positive")
                recipe = ExtrudedGeometry(
                    sketch,
                    height,
                    tuple(source_face_ids),
                )
                _run_profile_transform_preflight(
                    _preflight_composite_geometry,
                    recipe,
                )
                draft = geometry_draft(recipe)
                transform_summary = (
                    f"拉伸高={_display_number(height)} {length_unit}，"
                    "方向=XY 正法向"
                )
            else:
                path = _composite_path(
                    geometry["path"],
                    name=f"扫掠路径-{part_function}",
                )
                frame_strategy = str(geometry["frame_strategy"])
                if frame_strategy not in {"fixed", "transport"}:
                    raise ValueError(
                        "path_swept_profile frame_strategy must be fixed or transport"
                    )
                recipe = PathSweptGeometry(
                    sketch,
                    path,
                    tuple(source_face_ids),
                    frame_strategy,
                )
                _run_profile_transform_preflight(
                    _preflight_composite_geometry,
                    recipe,
                )
                draft = geometry_draft(recipe)
                transform_summary = (
                    f"路径扫掠段数={len(path.members)}，"
                    f"frame={frame_strategy}"
                )
            profile_summary = _bounded_geometry_design_summary(
                "2D Profile",
                [
                    *(
                        str(item)
                        for item in profile_context.get("design_summaries", [])
                    ),
                    f"material={len(profile_context.get('profiles', []))}",
                    f"holes={profile_context.get('hole_count', 0)}",
                ],
            )
            provisional = bool(geometry.get("provisional", False))
            provisional_summary = "；尺寸标记为 provisional 提案值" if provisional else ""
            geometry_summary = (
                f"{profile_summary}；3D {kind}：{transform_summary}"
                f"{provisional_summary}"
            )
        elif kind == "wire":
            if set(geometry) != {"kind", "points", "members"}:
                raise ValueError("wire geometry fields do not match")
            raw_points = geometry["points"]
            raw_members = geometry["members"]
            if not isinstance(raw_points, list):
                raise ValueError("wire points must be an array")
            if not isinstance(raw_members, list):
                raise ValueError("wire members must be an array")
            points = []
            for raw_point in raw_points:
                if (
                    not isinstance(raw_point, Mapping)
                    or set(raw_point) != {"name", "x", "y", "z"}
                ):
                    raise ValueError("wire point fields do not match")
                points.append(
                    WirePoint(
                        raw_point["name"],
                        raw_point["x"],
                        raw_point["y"],
                        raw_point["z"],
                    )
                )
            members = []
            for raw_member in raw_members:
                if (
                    not isinstance(raw_member, Mapping)
                    or set(raw_member) != {"name", "start", "end"}
                ):
                    raise ValueError("wire member fields do not match")
                members.append(
                    WireMember(
                        raw_member["name"],
                        raw_member["start"],
                        raw_member["end"],
                    )
                )
            draft = wire_geometry(
                recipe_name,
                points=points,
                members=members,
            )
            geometry_summary = (
                f"1D 空间线几何(点={len(points)}，杆件={len(members)})"
            )
        elif kind == "box":
            if set(geometry) != {"kind", "width", "depth", "height"}:
                raise ValueError("box geometry fields do not match")
            draft = box_geometry(
                recipe_name,
                width=geometry["width"],
                depth=geometry["depth"],
                height=geometry["height"],
            )
            geometry_summary = (
                f"3D 长方体(宽={_display_number(geometry['width'])}, "
                f"深={_display_number(geometry['depth'])}, "
                f"高={_display_number(geometry['height'])})"
            )
        elif kind == "cylinder":
            if set(geometry) != {"kind", "radius", "height"}:
                raise ValueError("cylinder geometry fields do not match")
            draft = cylinder_geometry(
                recipe_name,
                radius=geometry["radius"],
                height=geometry["height"],
            )
            geometry_summary = (
                f"3D 圆柱(半径={_display_number(geometry['radius'])}, "
                f"高={_display_number(geometry['height'])})"
            )
        else:
            raise ValueError("unsupported geometry kind")
        requirements = controller.collected_requirements("geometry")
        defaulted_keys = controller.defaulted_requirement_keys("geometry")
        proposal_summary = (
            f"设计提案：{geometry_summary}；单位制 "
            f"{_geometry_unit_summary(requirements, defaulted_keys)}"
        )
        metadata = envelope(controller, "geometry")
        context = current_context()
        suffix = str(metadata.pop("identity_suffix"))
        proposal = create_geometry_proposal(
            proposal_id=f"proposal-{suffix}",
            context=context,
            draft=draft,
            part_function=part_function,
            project_function=(
                part_function
                if context.binding.source_kind == "blank"
                else None
            ),
            summary=proposal_summary,
            unit_context=UnitContextSummary(
                str(requirements["length_unit"]),
                str(requirements["force_unit"]),
                str(requirements["stress_unit"]),
                convention=(
                    f"{requirements['force_unit']}-"
                    f"{requirements['length_unit']}-"
                    f"{requirements['stress_unit']}"
                ),
            ),
            **metadata,
        )
        legacy_composite = kind in {
            "planar_profiles",
            "extruded_profiles",
            "extruded_path_slot_plate",
            "path_swept_profile",
        }
        return proposal_outcome(
            proposal,
            summary=proposal_summary,
            impact="确认后创建该几何并刷新 GUI",
            confirm_label="加入部件",
            extra_data={
                "authoring_path": (
                    "legacy_planar_profiles"
                    if kind == "planar_profiles"
                    else "legacy_geometry"
                ),
                **(
                    {
                        "compatibility_status": "deprecated",
                        "replacement_tool": "prepare_planar_construction_proposal",
                    }
                    if legacy_composite
                    else {}
                ),
            },
        )

    _PLANAR_DIAGNOSTIC_MESSAGES = {
        "planar-ir.schema-invalid": "二维构造字段或版本无效，请修正标记字段。",
        "planar-ir.budget-exceeded": "二维构造超过本地预算，请简化构造图。",
        "planar-ir.duplicate-node-id": "二维构造包含重复节点 ID，请重命名冲突节点。",
        "planar-ir.reference-missing": "二维构造引用了不存在的节点，请修正失败引用。",
        "planar-ir.cycle-detected": "二维构造节点形成循环，请断开诊断指出的依赖。",
        "planar-ir.unreachable-node": "二维构造包含未参与结果的节点，请删除或连接该节点。",
        "planar-ir.invalid-primitive": "二维基础区域参数无效，请修正诊断指出的节点。",
        "planar-ir.invalid-path-stroke": "定宽路径无效，请修正失败线段、宽度或连接方式。",
        "planar-ir.boolean-empty": "二维布尔运算结果为空，请调整 operand 或尺寸。",
        "planar-ir.degenerate-result": "二维构造产生退化边或区域，请调整尺寸关系。",
        "planar-ir.unsupported-boundary": "结果包含当前版本不支持的边界曲线。",
        "planar-ir.materialization-failed": "二维边界无法物化为严格草图，模型保持不变。",
        "planar-ir.profile-invalid": "物化草图未通过 Profile 拓扑证明。",
        "planar-ir.equivalence-failed": "物化草图与布尔结果不等价，已阻止提案。",
        "planar-ir.transform-invalid": "三维变换参数或源 Profile 无效，请修正 output。",
        "planar-ir.preflight-failed": "最终三维 Recipe 未通过本地精确预检。",
        "planar-ir.stale-context": "文档、Session 或 revision 已变化，请重新读取上下文。",
    }

    def _planar_construction_failure(
        diagnostic: object,
        request: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        code = str(getattr(diagnostic, "code"))
        allowed_fields = tuple(getattr(diagnostic, "allowed_fields"))
        retry = controller.assess_planar_construction_failure(
            request,
            code=code,
            node_id=getattr(diagnostic, "node_id"),
            retryable=bool(getattr(diagnostic, "retryable")),
            allowed_fields=allowed_fields,
        )
        payload = {
            "code": code,
            "message": str(getattr(diagnostic, "message"))[:240],
            "node_id": getattr(diagnostic, "node_id"),
            "retryable": retry["retryable"],
            "allowed_fields": list(allowed_fields),
            "model_unchanged": True,
        }
        user_message = _PLANAR_DIAGNOSTIC_MESSAGES.get(
            code,
            "二维构造未通过本地验证，模型保持不变。",
        )
        if retry["blocker"] is not None:
            user_message = f"{user_message} {retry['blocker']}"
        return AuthoringToolOutcome(
            user_message,
            {
                "diagnostic": payload,
                "retry": retry,
                "authoring_path": "planar_construction_ir_v1",
                "required_action": "revise_same_planar_construction_ir",
            },
            ok=False,
        )

    def _planar_construction_profile_selection(
        sketch: SketchGeometry,
        raw_selection: object,
    ) -> tuple[str, ...]:
        context = profile_transform_context(sketch)
        profiles = context.get("profiles")
        if not context.get("topology_exact") or not isinstance(profiles, list):
            raise AuthoringContractError(
                "profile-transform.topology-unproven: IR-derived Profile "
                "topology is not exact"
            )
        canonical = tuple(
            str(item["face_id"])
            for item in profiles
            if isinstance(item, Mapping) and isinstance(item.get("face_id"), str)
        )
        if raw_selection == "unique_material_profile":
            if len(canonical) != 1 or int(context["material_profile_count"]) != 1:
                candidates = ", ".join(canonical) or "none"
                raise AuthoringContractError(
                    "profile-transform.ambiguous-material-profiles: "
                    "unique_material_profile requires exactly one material "
                    f"Profile; candidates={candidates}"
                )
            return canonical
        if isinstance(raw_selection, Mapping):
            if set(raw_selection) != {"source_face_ids"}:
                raise ValueError("profile_selection fields do not match")
            raw_selection = raw_selection["source_face_ids"]
        if not isinstance(raw_selection, list) or not raw_selection:
            raise ValueError(
                "profile_selection must be unique_material_profile or an "
                "explicit source_face_ids array"
            )
        if any(not isinstance(item, str) for item in raw_selection):
            raise TypeError("source_face_ids must contain strings")
        requested = tuple(raw_selection)
        if len(requested) != len(set(requested)):
            raise AuthoringContractError(
                "profile-transform.invalid-source-id: source_face_ids must be unique"
            )
        unknown = tuple(item for item in requested if item not in canonical)
        if unknown:
            raise AuthoringContractError(
                "profile-transform.invalid-source-id: explicit IDs must be "
                f"canonical material Profile IDs; invalid={unknown[0]}"
            )
        return requested

    def _planar_construction_output_recipe(
        sketch: SketchGeometry,
        output: object,
        *,
        part_function: str,
    ) -> tuple[str, object, _DetachedRecipeMesh | None]:
        if output == "planar":
            return "planar", sketch, None
        if not isinstance(output, Mapping) or not isinstance(output.get("kind"), str):
            raise ValueError("output must be planar or a derived output object")
        kind = str(output["kind"])
        required = {
            "extrusion": {"kind", "profile_selection", "height"},
            "revolution": {
                "kind",
                "profile_selection",
                "axis",
                "angle_degrees",
            },
            "path_sweep": {
                "kind",
                "profile_selection",
                "path",
                "frame_strategy",
            },
        }.get(kind)
        if required is None or set(output) != required:
            raise ValueError("derived output fields do not match")
        selected = _planar_construction_profile_selection(
            sketch,
            output["profile_selection"],
        )
        if kind == "extrusion":
            extrusions = tuple(
                ExtrudedGeometry(sketch, output["height"], (face_id,))
                for face_id in selected
            )
            recipe = (
                extrusions[0]
                if len(extrusions) == 1
                else MultiBodyGeometry(
                    f"{part_function} Geometry",
                    tuple(
                        SolidBody(f"B{index}", f"Body-{index}", extrusion)
                        for index, extrusion in enumerate(extrusions, start=1)
                    ),
                )
            )
        elif kind == "revolution":
            if len(selected) != 1:
                raise AuthoringContractError(
                    "profile-transform.ambiguous-material-profiles: revolution "
                    "requires exactly one canonical material Profile"
                )
            recipe = RevolvedGeometry(
                sketch,
                str(output["axis"]),
                output["angle_degrees"],
                selected,
            )
        else:
            if len(selected) != 1:
                raise AuthoringContractError(
                    "profile-transform.ambiguous-material-profiles: path sweep "
                    "requires exactly one canonical material Profile"
                )
            path = _composite_path(
                output["path"],
                name=f"扫掠路径-{part_function}",
            )
            recipe = PathSweptGeometry(
                sketch,
                path,
                selected,
                str(output["frame_strategy"]),
            )
        preview = _preflight_composite_geometry(recipe, include_preview=True)
        if preview is None:
            raise AuthoringContractError(
                "planar-ir.preflight-failed: detached preview was not produced"
            )
        return kind, recipe, preview

    def prepare_planar_construction(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) != {"part_function", "construction", "output"}:
            raise AuthoringContractError(
                "prepare_planar_construction_proposal fields do not match"
            )
        part_function = arguments["part_function"]
        raw_construction = arguments["construction"]
        if (
            type(part_function) is not str
            or not part_function.strip()
            or part_function != part_function.strip()
            or len(part_function) > 96
        ):
            raise AuthoringContractError(
                "part_function must be a bounded nonblank string"
            )
        if not isinstance(raw_construction, Mapping):
            raise AuthoringContractError("construction must be an object")

        initial_context = current_context()
        try:
            construction = PlanarConstructionIR.from_dict(raw_construction)
            compiled = compile_planar_construction(construction)
        except (PlanarIRValidationError, PlanarConstructionCompileError) as error:
            return _planar_construction_failure(
                error.diagnostic,
                arguments,
                controller,
            )
        if type(compiled.recipe) is not SketchGeometry or not compiled.recipe.is_strict:
            raise AuthoringContractError(
                "planar construction compiler returned no strict sketch"
            )
        try:
            output_kind, output_recipe, output_mesh = _planar_construction_output_recipe(
                compiled.recipe,
                arguments["output"],
                part_function=part_function,
            )
        except (AuthoringContractError, TypeError, ValueError) as error:
            code = (
                "planar-ir.preflight-failed"
                if "preflight" in str(error).casefold()
                else "planar-ir.transform-invalid"
            )
            return _planar_construction_failure(
                geometry_runtime.PlanarIRDiagnostic(
                    code,
                    str(error),
                    None,
                    True,
                    ("output",),
                ),
                arguments,
                controller,
            )

        current = current_context()
        if (
            initial_context.binding.document_id,
            initial_context.binding.session_id,
            initial_context.binding.session_revision,
        ) != (
            current.binding.document_id,
            current.binding.session_id,
            current.binding.session_revision,
        ):
            return _planar_construction_failure(
                geometry_runtime.PlanarIRDiagnostic(
                    "planar-ir.stale-context",
                    "The document, session, or revision changed during compilation.",
                    None,
                    True,
                    ("construction",),
                ),
                arguments,
                controller,
            )

        evidence = planar_construction_proposal_evidence(
            construction,
            compiled,
            output_kind=output_kind,
            output_recipe=output_recipe,
        )
        source_preview = compiled.preview if output_mesh is None else output_mesh
        detached_preview = AgentProposalPreview(
            dimension=2 if output_kind == "planar" else 3,
            points=source_preview.points,
            faces=source_preview.faces,
            edges=source_preview.edges,
            recipe_digest=str(evidence["output_recipe_digest"]),
            proof_digest=str(evidence["output_proof_digest"]),
        )
        draft = geometry_draft(output_recipe)
        requirements = controller.collected_requirements("geometry")
        defaulted_keys = controller.defaulted_requirement_keys("geometry")
        construction_summary = evidence["construction_summary"]
        proof_summary = evidence["proof_summary"]
        output_label = {
            "planar": "2D 平面构造",
            "extrusion": "3D Profile 拉伸",
            "revolution": "3D Profile 旋转扫掠",
            "path_sweep": "3D Profile 路径扫掠",
        }[output_kind]
        proposal_summary = (
            f"设计提案：{output_label}"
            f"（节点={construction_summary['node_count']}，"
            f"材料区={proof_summary['material_profile_count']}，"
            f"孔洞={proof_summary['hole_count']}）；单位制 "
            f"{_geometry_unit_summary(requirements, defaulted_keys)}"
        )
        metadata = envelope(controller, "planar-construction")
        suffix = str(metadata.pop("identity_suffix"))
        proposal = create_geometry_proposal(
            proposal_id=f"proposal-{suffix}",
            context=current,
            draft=draft,
            part_function=part_function,
            project_function=(
                part_function
                if current.binding.source_kind == "blank"
                else None
            ),
            summary=proposal_summary,
            unit_context=UnitContextSummary(
                str(requirements["length_unit"]),
                str(requirements["force_unit"]),
                str(requirements["stress_unit"]),
                convention=(
                    f"{requirements['force_unit']}-"
                    f"{requirements['length_unit']}-"
                    f"{requirements['stress_unit']}"
                ),
            ),
            local_evidence=evidence,
            include_static_preview=False,
            **metadata,
        )
        outcome = proposal_outcome(
            proposal,
            summary=proposal_summary,
            impact=(
                "确认后创建该二维几何并刷新 GUI"
                if output_kind == "planar"
                else "确认后直接创建最终三维几何并刷新 GUI"
            ),
            confirm_label="加入部件",
            detached_preview=detached_preview,
            extra_data={
                "authoring_path": "planar_construction_ir_v1",
                "output_kind": output_kind,
                "construction_summary": construction_summary,
                "proof_summary": proof_summary,
                "construction_digest_short": str(
                    evidence["construction_digest"]
                )[:12],
                "recipe_proof_digest_short": str(
                    evidence["recipe_proof_digest"]
                )[:12],
                "output_recipe_digest_short": str(
                    evidence["output_recipe_digest"]
                )[:12],
                "output_proof_digest_short": str(
                    evidence["output_proof_digest"]
                )[:12],
            },
        )
        controller.record_planar_construction_proposal(
            str(evidence["construction_digest"]),
            proposal.proposal_id,
        )
        return outcome

    def read_geometry_edit_context(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        part_id = str(arguments["part_id"])
        snapshot = current_session().snapshot()
        part = next(
            (
                candidate
                for candidate in snapshot.parts
                if str(candidate.id) == part_id and not candidate.suppressed
            ),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError(
                "part_id does not identify one editable native Part"
            )
        try:
            catalog = planar_geometry_catalog(part.geometry_recipe)
        except TypeError:
            catalog = {
                "kind": _provider_recipe_kind(part.geometry_recipe),
                "supported_edits": ["translate", "rotate"],
            }
        else:
            catalog["supported_edits"] = [
                "add_line",
                "add_arc",
                "add_circle",
                "add_rectangle",
                "add_polygon",
                "update_point",
                "update_line",
                "update_arc",
                "update_circle",
                "delete_curves",
                "delete_circles",
                "replace_circle_pattern",
                "add_constraint",
                "replace_constraint",
                "delete_constraints",
                "batch",
                "translate",
                "rotate",
            ]
            if (
                type(part.geometry_recipe) is SketchGeometry
                and part.geometry_recipe.is_strict
            ):
                catalog["supported_edits"][:0] = [
                    "extrude_profiles",
                    "revolve_profile",
                    "path_sweep_profile",
                ]
            sketch = _as_strict_planar_sketch(part.geometry_recipe)
            profile_analysis = geometry_runtime.analyze_sketch_profiles(sketch)
            profiles = profile_analysis.profiles
            catalog["profile_summary"] = {
                "topology_exact": profile_analysis.valid,
                "profile_count": len(profiles),
                "material_profile_count": sum(
                    item.role == "outer" for item in profiles
                ),
                "hole_count": sum(item.role == "hole" for item in profiles),
                "profiles": [
                    {
                        "profile_id": item.id,
                        "role": item.role,
                        "curve_count": len(item.curve_ids),
                        "area": abs(float(item.signed_area)),
                        "bounding_box": list(item.bounding_box),
                        "nesting_depth": item.nesting_depth,
                    }
                    for item in profiles[:16]
                ],
                "truncated": len(profiles) > 16,
                "diagnostics": [
                    {
                        "code": item.code,
                        "message": item.message,
                        "affected_logical_ids": list(item.affected_ids),
                        "severity": item.severity,
                    }
                    for item in profile_analysis.diagnostics[:16]
                ],
            }
            catalog["freeform_profile_policy"] = {
                "two_dimensional_cut_representation": "closed_inner_profile",
                "part_boolean_required": False,
                "preferred_operation": "add_polygon",
                "alternate_operation": "one_batch_of_ordered_lines_and_arcs",
                "required_invariants": [
                    "closed",
                    "non_self_intersecting",
                    "contained_by_material_profile",
                ],
                "forbidden_intermediate_geometry": [
                    "open_centerline",
                    "placeholder_unrelated_to_final_contour",
                ],
                "postcondition_for_one_cutout": "hole_count increases by 1",
            }
        if part.dimension == 3:
            if type(part.geometry_recipe) is MultiBodyGeometry:
                catalog["supported_edits"][:0] = ["body_boolean"]
                body_candidates = [
                    {"body_id": body.id, "body_name": body.name}
                    for body in part.geometry_recipe.bodies
                ]
            else:
                catalog["supported_edits"][:0] = ["part_boolean"]
                body_candidates = []
            catalog["exact_boolean"] = {
                "supported_operations": ["fuse", "cut"],
                "disabled_operations": [
                    {
                        "operation": "intersect",
                        "code": "boolean.agent.operation-disabled",
                        "message": (
                            "intersect is disabled until stable result Body IDs, "
                            "lineage replay, and edit semantics are proven"
                        ),
                    },
                    {
                        "operation": "fragment",
                        "code": "boolean.agent.operation-disabled",
                        "message": (
                            "fragment is disabled until stable multi-result Body "
                            "IDs, lineage replay, and edit semantics are proven"
                        ),
                    },
                ],
                "part_tool_handling": PART_BOOLEAN_TOOL_HANDLING,
                "body_tool_handling": BODY_BOOLEAN_TOOL_HANDLING,
                "body_candidates": body_candidates,
                "part_tool_candidates": [
                    {"part_id": candidate.id, "part_name": candidate.name}
                    for candidate in snapshot.parts
                    if (
                        candidate.id != part.id
                        and not candidate.suppressed
                        and candidate.dimension == 3
                        and type(candidate.geometry_recipe) is not MultiBodyGeometry
                    )
                ],
            }
        return AuthoringToolOutcome(
            "Editable geometry context read locally.",
            {
                "part_id": part_id,
                "part_name": str(part.name),
                **catalog,
            },
        )

    def _profile_transform_part(part_id: object):
        normalized_part_id = str(part_id).strip()
        snapshot = current_session().snapshot()
        if snapshot.source_kind != "native":
            raise AuthoringContractError(
                "profile-transform.part-not-found: current document is not a "
                "native authoring Session"
            )
        part = next(
            (
                candidate
                for candidate in snapshot.parts
                if str(candidate.id) == normalized_part_id
                and not candidate.suppressed
            ),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError(
                "profile-transform.part-not-found: part_id does not identify "
                "one active native Part"
            )
        return snapshot, part

    def _strict_profile_recipe(part):
        try:
            return _as_strict_planar_sketch(part.geometry_recipe)
        except (TypeError, ValueError) as error:
            raise AuthoringContractError(
                "profile-transform.source-not-planar: Profile transforms "
                "require a canonical planar sketch source"
            ) from error

    def _profile_selection(
        arguments: Mapping[str, object],
        *,
        snapshot,
        part,
    ) -> tuple[str, ...]:
        if "profile_selection" not in arguments and "source_face_ids" in arguments:
            raw_selection = arguments["source_face_ids"]
        else:
            raw_selection = arguments.get("profile_selection")
        unique_selection = raw_selection == "unique_material_profile" or (
            isinstance(raw_selection, Mapping)
            and set(raw_selection) == {"mode"}
            and raw_selection.get("mode") == "unique_material_profile"
        ) or (
            isinstance(raw_selection, Mapping)
            and set(raw_selection) == {"kind"}
            and raw_selection.get("kind") == "unique_material_profile"
        )
        if not unique_selection and "context_revision" not in arguments:
            raise AuthoringContractError(
                "profile-transform.stale-context: explicit source_face_ids "
                "require context_revision from read_profile_transform_context"
            )
        if "context_revision" in arguments:
            revision = arguments["context_revision"]
            if type(revision) is not int or revision != snapshot.session_revision:
                raise AuthoringContractError(
                    "profile-transform.stale-context: transform context revision "
                    "does not match the current Session"
                )
        context = profile_transform_context(
            part.geometry_recipe,
            part_id=str(part.id),
            session_revision=snapshot.session_revision,
        )
        operation_context = context.get("extrusion")
        if not isinstance(operation_context, Mapping) or not operation_context.get(
            "available", False
        ):
            blocking_code = (
                str(operation_context.get("blocking_code"))
                if isinstance(operation_context, Mapping)
                and operation_context.get("blocking_code")
                else "profile-transform.topology-unproven"
            )
            blocking_reason = (
                str(operation_context.get("blocking_reason"))
                if isinstance(operation_context, Mapping)
                and operation_context.get("blocking_reason")
                else "Profile topology is not available"
            )
            raise AuthoringContractError(
                f"{blocking_code}: {blocking_reason}"
            )
        if not context["topology_exact"]:
            reason = str(
                context.get("extrusion", {}).get(
                    "blocking_reason",
                    "Profile topology is not exact",
                )
            )
            raise AuthoringContractError(
                f"profile-transform.topology-unproven: {reason}"
            )
        profiles = context.get("profiles")
        if not isinstance(profiles, list):
            raise AuthoringContractError(
                "profile-transform.no-material-profile: no canonical Profiles"
            )
        canonical = tuple(
            str(item["face_id"])
            for item in profiles
            if isinstance(item, Mapping) and isinstance(item.get("face_id"), str)
        )
        if unique_selection:
            if len(canonical) != 1 or int(context["material_profile_count"]) != 1:
                candidates = ", ".join(canonical[:32]) or "none"
                raise AuthoringContractError(
                    "profile-transform.ambiguous-material-profiles: "
                    "unique_material_profile requires exactly one material "
                    f"Profile; candidates={candidates}"
                )
            return canonical
        if isinstance(raw_selection, Mapping):
            if set(raw_selection) != {"source_face_ids"}:
                raise ValueError("profile_selection fields do not match")
            raw_selection = raw_selection["source_face_ids"]
        if not isinstance(raw_selection, list) or not raw_selection:
            raise ValueError(
                "profile_selection must be unique_material_profile or an "
                "explicit source_face_ids array"
            )
        if any(not isinstance(item, str) for item in raw_selection):
            raise TypeError("source_face_ids must contain strings")
        requested = tuple(raw_selection)
        if len(requested) != len(set(requested)):
            raise AuthoringContractError(
                "profile-transform.invalid-source-id: source_face_ids must be unique"
            )
        unknown_ids = tuple(item for item in requested if item not in canonical)
        if unknown_ids or int(context["material_profile_count"]) > len(canonical):
            if unknown_ids:
                raise AuthoringContractError(
                    "profile-transform.invalid-source-id: explicit IDs must be "
                    "canonical material Profile IDs from the current context; "
                    f"invalid={unknown_ids[0]}"
                )
            raise AuthoringContractError(
                "profile-transform.topology-unproven: Profile context was truncated; "
                "reread the transform context before selecting IDs"
            )
        return requested

    def _profile_transform_error(
        error: BaseException,
        *,
        operation: str = "Profile transform",
        required_fields: tuple[str, ...] | None = None,
        first_failed_member: str | None = None,
    ) -> AuthoringToolOutcome:
        """Normalize one transform failure without changing the Session."""

        raw_message = str(error).strip() or type(error).__name__
        code = "profile-transform.preflight-failed"
        detail = raw_message
        if raw_message.startswith("profile-transform."):
            prefix, separator, remainder = raw_message.partition(":")
            if prefix in {
                "profile-transform.part-not-found",
                "profile-transform.source-not-planar",
                "profile-transform.source-not-strict",
                "profile-transform.no-material-profile",
                "profile-transform.ambiguous-material-profiles",
                "profile-transform.invalid-profile",
                "profile-transform.invalid-source-id",
                "profile-transform.nonpositive-height",
                "profile-transform.invalid-path",
                "profile-transform.unsupported-frame",
                "profile-transform.topology-unproven",
                "profile-transform.unexpected-body-count",
                "profile-transform.stale-context",
                "profile-transform.preflight-failed",
            }:
                code = prefix
                detail = remainder.strip() if separator else prefix
        else:
            lowered = raw_message.casefold()
            if "preflight" in lowered or "occ" in lowered or "gmsh" in lowered:
                code = "profile-transform.preflight-failed"
            elif "frame_strategy" in lowered or "frame" in lowered:
                code = "profile-transform.unsupported-frame"
            elif any(
                marker in lowered
                for marker in (
                    "path",
                    "路径",
                    "branch",
                    "self-intersect",
                    "open",
                    "wire",
                    "member",
                )
            ):
                code = "profile-transform.invalid-path"
            elif "height" in lowered or "高度" in lowered:
                code = "profile-transform.nonpositive-height"
            elif "body" in lowered or "volume" in lowered:
                code = "profile-transform.unexpected-body-count"
            elif "topolog" in lowered or "strict" in lowered:
                code = "profile-transform.topology-unproven"
            elif any(
                marker in lowered
                for marker in (
                    "composite profile",
                    "polygon vertices",
                    "rectangle profile",
                    "circle profile",
                    "slot path",
                    "slot width",
                )
            ):
                code = "profile-transform.invalid-profile"
            elif "source" in lowered or "profile" in lowered:
                code = "profile-transform.invalid-source-id"

        candidates: tuple[str, ...] = ()
        marker = "candidates="
        if marker in detail:
            candidate_text = detail.split(marker, 1)[1].split(";", 1)[0]
            candidates = tuple(
                item.strip().strip("()[]{}'\"")[:192]
                for item in candidate_text.split(",")
                if item.strip().strip("()[]{}'\"")
                and item.strip().strip("()[]{}'\"") != "none"
            )[:32]
        if first_failed_member is None:
            member_marker = "member "
            if member_marker in detail.casefold():
                fragment = detail.casefold().split(member_marker, 1)[1]
                first_failed_member = fragment.split()[0].strip("'\"`()[]{}:,")
        diagnostic = profile_transform_diagnostic(
            code,
            operation=operation,
            detail=detail,
            required_fields=required_fields,
            candidates=candidates,
            first_failed_member=first_failed_member,
        )
        payload = {"diagnostic": diagnostic.to_dict()}
        return AuthoringToolOutcome(diagnostic.message, payload, ok=False)

    def _first_failed_path_member(value: object) -> str | None:
        """Return the first member that breaks the caller-provided order."""

        if not isinstance(value, Mapping):
            return None
        raw_members = value.get("members")
        if not isinstance(raw_members, list):
            return None
        members = [item for item in raw_members if isinstance(item, Mapping)]
        for previous, current in zip(members, members[1:]):
            if previous.get("end") != current.get("start"):
                name = current.get("name")
                return str(name) if name is not None else None
        return None

    def read_profile_transform_context(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        try:
            if set(arguments) != {"part_id"}:
                raise ValueError(
                    "read_profile_transform_context fields do not match"
                )
            snapshot, part = _profile_transform_part(arguments["part_id"])
            data = profile_transform_context(
                part.geometry_recipe,
                part_id=str(part.id),
                session_revision=snapshot.session_revision,
            )
            data["part_name"] = str(part.name)
            data["active"] = str(part.id) == str(snapshot.active_part_id)
            provider_safe_authoring_payload(data)
            return AuthoringToolOutcome(
                "Profile transform context read locally.",
                data,
            )
        except (AuthoringContractError, TypeError, ValueError) as error:
            return _profile_transform_error(
                error,
                operation="读取 Profile 变换上下文",
                required_fields=("part_id",),
            )

    def prepare_profile_extrusion(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        try:
            required = {"part_id", "profile_selection", "height"}
            if "source_face_ids" in arguments and "profile_selection" not in arguments:
                required = {"part_id", "source_face_ids", "height"}
            missing = required - set(arguments)
            extra = set(arguments) - required - {"context_revision"}
            if missing:
                missing = tuple(
                    field
                    for field in ("part_id", "profile_selection", "height")
                    if field not in arguments
                )
                missing_code = (
                    "profile-transform.nonpositive-height"
                    if "height" in missing
                    else (
                        "profile-transform.part-not-found"
                        if "part_id" in missing
                        else "profile-transform.ambiguous-material-profiles"
                    )
                )
                raise AuthoringContractError(
                    missing_code + ": "
                    + ("missing height" if "height" in missing else "missing required fields")
                )
            if extra:
                raise ValueError("prepare_profile_extrusion fields do not match")
            snapshot, part = _profile_transform_part(arguments["part_id"])
            selected = _profile_selection(arguments, snapshot=snapshot, part=part)
            height = arguments["height"]
            base_recipe = _strict_profile_recipe(part)
            return prepare_geometry_edit(
                {
                    "part_id": str(part.id),
                    "_canonical_base_recipe": base_recipe,
                    "edit": {
                        "operation": "extrude_profiles",
                        "source_face_ids": list(selected),
                        "height": height,
                    },
                },
                controller,
            )
        except (AuthoringContractError, TypeError, ValueError) as error:
            return _profile_transform_error(
                error,
                operation="Profile 拉伸",
            )

    def prepare_profile_revolution(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        try:
            required = {"part_id", "profile_selection", "axis", "angle_degrees"}
            if "source_face_ids" in arguments and "profile_selection" not in arguments:
                required = {"part_id", "source_face_ids", "axis", "angle_degrees"}
            missing = required - set(arguments)
            extra = set(arguments) - required - {"context_revision"}
            if missing or extra:
                raise AuthoringContractError(
                    "profile-transform.preflight-failed: missing required fields"
                )
            snapshot, part = _profile_transform_part(arguments["part_id"])
            selected = _profile_selection(arguments, snapshot=snapshot, part=part)
            if len(selected) != 1:
                raise AuthoringContractError(
                    "profile-transform.ambiguous-material-profiles: revolution "
                    "requires exactly one canonical material Profile"
                )
            base_recipe = _strict_profile_recipe(part)
            return prepare_geometry_edit(
                {
                    "part_id": str(part.id),
                    "_canonical_base_recipe": base_recipe,
                    "edit": {
                        "operation": "revolve_profile",
                        "source_face_id": selected[0],
                        "axis": arguments["axis"],
                        "angle_degrees": arguments["angle_degrees"],
                    },
                },
                controller,
            )
        except (AuthoringContractError, TypeError, ValueError) as error:
            return _profile_transform_error(error, operation="Profile 旋转扫掠")

    def prepare_profile_path_sweep(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        try:
            required = {
                "part_id",
                "profile_selection",
                "path",
                "frame_strategy",
            }
            if "source_face_ids" in arguments and "profile_selection" not in arguments:
                required = {"part_id", "source_face_ids", "path", "frame_strategy"}
            missing = required - set(arguments)
            extra = set(arguments) - required - {"context_revision"}
            if missing:
                if "frame_strategy" in missing:
                    raise AuthoringContractError(
                        "profile-transform.unsupported-frame: missing frame_strategy"
                    )
                raise AuthoringContractError(
                    "profile-transform.invalid-path: missing path"
                )
            if extra:
                raise ValueError("prepare_profile_path_sweep fields do not match")
            snapshot, part = _profile_transform_part(arguments["part_id"])
            selected = _profile_selection(arguments, snapshot=snapshot, part=part)
            if len(selected) != 1:
                raise AuthoringContractError(
                    "profile-transform.ambiguous-material-profiles: path sweep "
                    "requires exactly one canonical material Profile"
                )
            base_recipe = _strict_profile_recipe(part)
            return prepare_geometry_edit(
                {
                    "part_id": str(part.id),
                    "_canonical_base_recipe": base_recipe,
                    "edit": {
                        "operation": "path_sweep_profile",
                        "source_face_id": selected[0],
                        "path": arguments["path"],
                        "frame_strategy": arguments["frame_strategy"],
                    },
                },
                controller,
            )
        except (AuthoringContractError, TypeError, ValueError) as error:
            return _profile_transform_error(
                error,
                operation="Profile 路径扫掠",
                first_failed_member=_first_failed_path_member(arguments.get("path")),
            )

    def prepare_geometry_edit(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        part_id = str(arguments["part_id"])
        raw_edit = arguments["edit"]
        if not isinstance(raw_edit, Mapping):
            raise TypeError("edit must be an object")
        edit = dict(raw_edit)
        operation = str(edit.pop("operation"))
        snapshot = current_session().snapshot()
        part = next(
            (
                candidate
                for candidate in snapshot.parts
                if str(candidate.id) == part_id and not candidate.suppressed
            ),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError(
                "part_id does not identify one editable native Part"
            )
        base_recipe_override = arguments.get("_canonical_base_recipe")
        if base_recipe_override is not None and (
            type(base_recipe_override) is not SketchGeometry
            or not base_recipe_override.is_strict
        ):
            raise AuthoringContractError(
                "profile-transform.source-not-strict: canonical base recipe "
                "must be a strict planar sketch"
            )
        if operation == "part_boolean":
            if set(edit) != {
                "boolean_operation",
                "tool_part_id",
                "result_name",
                "tool_handling",
            }:
                raise ValueError("part_boolean fields do not match")
            if type(part.geometry_recipe) is MultiBodyGeometry or part.dimension != 3:
                raise AuthoringContractError(
                    "Part Boolean operands must each own one exact solid"
                )
            boolean_operation = str(edit["boolean_operation"])
            if boolean_operation not in {"fuse", "cut"}:
                raise AuthoringContractError(
                    "intersect/fragment are disabled until stable result Body "
                    "IDs and lineage replay are proven"
                )
            tool_part_id = str(edit["tool_part_id"])
            tool = next(
                (
                    candidate
                    for candidate in snapshot.parts
                    if (
                        candidate.id == tool_part_id
                        and candidate.id != part.id
                        and not candidate.suppressed
                        and candidate.dimension == 3
                        and type(candidate.geometry_recipe) is not MultiBodyGeometry
                    )
                ),
                None,
            )
            if tool is None:
                raise AuthoringContractError(
                    "tool_part_id must identify another active single-solid Part"
                )
            result_name = str(edit["result_name"]).strip()
            tool_handling = str(edit["tool_handling"])
            if tool_handling != PART_BOOLEAN_TOOL_HANDLING:
                raise AuthoringContractError(
                    f"tool_handling must be {PART_BOOLEAN_TOOL_HANDLING!r}"
                )
            try:
                prepared = _preflight_part_boolean(
                    part,
                    tool,
                    boolean_operation,
                    result_part_id=current_session().next_native_part_id,
                    feature_id=current_session().next_part_boolean_feature_id,
                    result_name=result_name,
                )
            except BooleanLineageResolutionError as error:
                return AuthoringToolOutcome(str(error), {}, ok=False)
            summary = (
                f"精确 {boolean_operation}：target Part {part.name} [{part.id}]，"
                f"tool Part {tool.name} [{tool.id}]；结果 {result_name} "
                f"[{prepared.context.result_part_id}]；两源 Part 抑制并可撤销恢复，"
                f"tool policy={tool_handling}"
            )
            metadata = envelope(controller, "geometry-part-boolean")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = planned_geometry_edit_mode()
            proposal = create_part_boolean_proposal(
                proposal_id=f"proposal-{suffix}",
                context=current_context(),
                target_part_id=part.id,
                tool_part_id=tool.id,
                operation=boolean_operation,
                result_name=result_name,
                tool_handling=tool_handling,
                prepared=prepared,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "确认后创建一个 proven 结果 Part，抑制 target/tool 源 Part，"
                    "并使旧网格、定义与结果失效",
                ),
                confirm_label="执行精确 Part 布尔",
                extra_data={
                    "result_part_id": prepared.context.result_part_id,
                    "feature_id": prepared.context.feature_id,
                    "lineage_proven": True,
                    "geometry_edit_mode": edit_mode,
                },
            )
        if operation == "body_boolean":
            if set(edit) != {
                "boolean_operation",
                "target_body_id",
                "tool_body_id",
                "result_name",
                "tool_handling",
            }:
                raise ValueError("body_boolean fields do not match")
            geometry = part.geometry_recipe
            if type(geometry) is not MultiBodyGeometry:
                raise AuthoringContractError(
                    "Body Boolean requires one canonical same-Part MultiBodyGeometry"
                )
            boolean_operation = str(edit["boolean_operation"])
            if boolean_operation not in {"fuse", "cut"}:
                raise AuthoringContractError(
                    "intersect/fragment are disabled until stable result Body "
                    "IDs and lineage replay are proven"
                )
            target_body_id = str(edit["target_body_id"])
            tool_body_id = str(edit["tool_body_id"])
            if target_body_id == tool_body_id:
                raise AuthoringContractError("target and tool Bodies must differ")
            geometry.body(target_body_id)
            geometry.body(tool_body_id)
            result_name = str(edit["result_name"]).strip()
            tool_handling = str(edit["tool_handling"])
            if tool_handling != BODY_BOOLEAN_TOOL_HANDLING:
                raise AuthoringContractError(
                    f"tool_handling must be {BODY_BOOLEAN_TOOL_HANDLING!r}"
                )
            try:
                prepared = _preflight_body_boolean(
                    geometry,
                    target_body_id,
                    tool_body_id,
                    boolean_operation,
                    result_name=result_name,
                )
            except BooleanLineageResolutionError as error:
                return AuthoringToolOutcome(str(error), {}, ok=False)
            summary = (
                f"精确 {boolean_operation}：Part {part.name} [{part.id}] 内 "
                f"target Body {target_body_id}，tool Body {tool_body_id}；"
                f"结果特征 {result_name}，保留 target ID、消费 tool，"
                f"policy={tool_handling}"
            )
            metadata = envelope(controller, "geometry-body-boolean")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = planned_geometry_edit_mode()
            proposal = create_body_boolean_proposal(
                proposal_id=f"proposal-{suffix}",
                context=current_context(),
                part_id=part.id,
                target_body_id=target_body_id,
                tool_body_id=tool_body_id,
                operation=boolean_operation,
                result_name=result_name,
                tool_handling=tool_handling,
                prepared=prepared,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "确认后在同一 Part 内保留 target Body ID、消费 tool Body，"
                    "并使旧网格、定义与结果失效",
                ),
                confirm_label="执行精确 Body 布尔",
                extra_data={
                    "target_body_id": target_body_id,
                    "consumed_tool_body_id": tool_body_id,
                    "lineage_proven": True,
                    "geometry_edit_mode": edit_mode,
                },
            )
        if operation == "extrude_profiles":
            if set(edit) != {"source_face_ids", "height"}:
                raise ValueError("extrude_profiles fields do not match")
            base_recipe = (
                base_recipe_override
                if base_recipe_override is not None
                else part.geometry_recipe
            )
            if type(base_recipe) is not SketchGeometry or not base_recipe.is_strict:
                raise AuthoringContractError(
                    "Agent Profile extrusion requires a strict planar sketch Part"
                )
            raw_source_ids = edit["source_face_ids"]
            if not isinstance(raw_source_ids, list) or not raw_source_ids:
                raise ValueError(
                    "source_face_ids must explicitly select material Profiles"
                )
            if any(not isinstance(item, str) for item in raw_source_ids):
                raise TypeError("source_face_ids must contain strings")
            selection = resolve_extrusion_source_faces(
                base_recipe,
                tuple(raw_source_ids),
            )
            if len(selection.face_ids) != len(raw_source_ids):
                raise ValueError("source_face_ids contain duplicate Profile aliases")
            height = float(edit["height"])
            recipes = tuple(
                ExtrudedGeometry(base_recipe, height, (face_id,))
                for face_id in selection.face_ids
            )
            _run_profile_transform_preflight(
                _preflight_profile_extrusions,
                recipes,
            )
            summary = (
                f"选择式拉伸部件 {part.name} 的 {len(selection.face_ids)} 个 "
                f"Profile：高度 {_display_number(height)}，沿草图正法向，"
                f"生成 {len(selection.face_ids)} 个独立 Part"
            )
            metadata = envelope(controller, "geometry-edit")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = planned_geometry_edit_mode()
            proposal = create_profile_extrusion_proposal(
                proposal_id=f"proposal-{suffix}",
                context=current_context(),
                part_id=part_id,
                base_recipe=base_recipe,
                source_face_ids=selection.face_ids,
                height=height,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "确认后将选定 Profiles 原子转换为独立实体 Part，"
                    "并使旧网格、定义与结果失效",
                ),
                confirm_label="拉伸选定 Profiles",
                extra_data={"geometry_edit_mode": edit_mode},
            )
        if operation == "revolve_profile":
            if set(edit) != {"source_face_id", "axis", "angle_degrees"}:
                raise ValueError("revolve_profile fields do not match")
            base_recipe = (
                base_recipe_override
                if base_recipe_override is not None
                else part.geometry_recipe
            )
            if type(base_recipe) is not SketchGeometry or not base_recipe.is_strict:
                raise AuthoringContractError(
                    "Agent Profile revolution requires a strict planar sketch Part"
                )
            source_face_id = str(edit["source_face_id"])
            selection = resolve_extrusion_source_faces(
                base_recipe,
                (source_face_id,),
            )
            if selection.face_ids != (source_face_id,):
                raise AuthoringContractError(
                    "revolve_profile requires one canonical material Profile"
                )
            recipe = RevolvedGeometry(
                base_recipe,
                str(edit["axis"]),
                edit["angle_degrees"],
                (source_face_id,),
            )
            _run_profile_transform_preflight(
                _preflight_derived_geometry,
                recipe,
            )
            summary = (
                f"绕 {recipe.axis.upper()} 轴旋转扫掠部件 {part.name} 的 "
                f"Profile {source_face_id}：角度 {recipe.angle_degrees:g}°，"
                "生成 1 个实体 Part"
            )
            metadata = envelope(controller, "geometry-revolve")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = planned_geometry_edit_mode()
            proposal = create_profile_revolution_proposal(
                proposal_id=f"proposal-{suffix}",
                context=current_context(),
                part_id=part_id,
                base_recipe=base_recipe,
                source_face_id=source_face_id,
                axis=recipe.axis,
                angle_degrees=recipe.angle_degrees,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "确认后将 Profile 原子替换为旋转实体，"
                    "并使旧网格、定义与结果失效",
                ),
                confirm_label="旋转扫掠 Profile",
                extra_data={"geometry_edit_mode": edit_mode},
            )
        if operation == "path_sweep_profile":
            if set(edit) != {"source_face_id", "path", "frame_strategy"}:
                raise ValueError("path_sweep_profile fields do not match")
            base_recipe = (
                base_recipe_override
                if base_recipe_override is not None
                else part.geometry_recipe
            )
            if type(base_recipe) is not SketchGeometry or not base_recipe.is_strict:
                raise AuthoringContractError(
                    "Agent path sweep requires a strict planar sketch Part"
                )
            source_face_id = str(edit["source_face_id"])
            selection = resolve_extrusion_source_faces(
                base_recipe,
                (source_face_id,),
            )
            if selection.face_ids != (source_face_id,):
                raise AuthoringContractError(
                    "path_sweep_profile requires one canonical material Profile"
                )
            raw_path = edit["path"]
            if not isinstance(raw_path, Mapping) or set(raw_path) != {"points", "members"}:
                raise ValueError("path fields do not match")
            raw_points = raw_path["points"]
            raw_members = raw_path["members"]
            if not isinstance(raw_points, list) or not isinstance(raw_members, list):
                raise TypeError("path points and members must be arrays")
            if any(
                not isinstance(item, Mapping)
                or set(item) != {"name", "x", "y", "z"}
                for item in raw_points
            ):
                raise ValueError("path point fields do not match")
            if any(
                not isinstance(item, Mapping)
                or set(item) != {"name", "start", "end"}
                for item in raw_members
            ):
                raise ValueError("path member fields do not match")
            path = WireGeometry(
                f"扫掠路径-{part.id}",
                tuple(
                    WirePoint(item["name"], item["x"], item["y"], item["z"])
                    for item in raw_points
                ),
                tuple(
                    WireMember(item["name"], item["start"], item["end"])
                    for item in raw_members
                ),
            )
            recipe = PathSweptGeometry(
                base_recipe,
                path,
                (source_face_id,),
                str(edit["frame_strategy"]),
            )
            _run_profile_transform_preflight(
                _preflight_derived_geometry,
                recipe,
            )
            summary = (
                f"沿 {len(path.members)} 段显式开放折线路径扫掠部件 "
                f"{part.name} 的 Profile {source_face_id}；"
                f"frame={recipe.frame_strategy}，生成 1 个实体 Part"
            )
            metadata = envelope(controller, "geometry-path-sweep")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = planned_geometry_edit_mode()
            proposal = create_profile_path_sweep_proposal(
                proposal_id=f"proposal-{suffix}",
                context=current_context(),
                part_id=part_id,
                base_recipe=base_recipe,
                source_face_id=source_face_id,
                path=path,
                frame_strategy=recipe.frame_strategy,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "确认后将 Profile 原子替换为路径扫掠实体，"
                    "并使旧网格、定义与结果失效",
                ),
                confirm_label="沿路径扫掠 Profile",
                extra_data={"geometry_edit_mode": edit_mode},
            )
        if operation == "add_line":
            if set(edit) != {"start", "end"}:
                raise ValueError("add_line fields do not match")
            draft = add_planar_line(part.geometry_recipe, **edit)
            summary = f"在部件 {part.name} 的平面草图中增加直线"
        elif operation == "add_arc":
            if set(edit) != {"start", "center", "end", "orientation"}:
                raise ValueError("add_arc fields do not match")
            draft = add_planar_arc(part.geometry_recipe, **edit)
            summary = f"在部件 {part.name} 的平面草图中增加圆弧"
        elif operation == "add_circle":
            if set(edit) != {"center_x", "center_y", "radius"}:
                raise ValueError("add_circle fields do not match")
            draft = add_planar_circle(part.geometry_recipe, **edit)
            summary = (
                f"在部件 {part.name} 的平面草图中增加圆："
                f"圆心 ({edit['center_x']}, {edit['center_y']})，"
                f"半径 {edit['radius']}"
            )
        elif operation == "add_rectangle":
            if set(edit) != {"x", "y", "width", "height"}:
                raise ValueError("add_rectangle fields do not match")
            draft = add_planar_rectangle(part.geometry_recipe, **edit)
            summary = (
                f"在部件 {part.name} 的平面草图中增加矩形轮廓："
                f"起点 ({edit['x']}, {edit['y']})，"
                f"尺寸 {edit['width']} × {edit['height']}"
            )
        elif operation == "add_polygon":
            if set(edit) != {"vertices"}:
                raise ValueError("add_polygon fields do not match")
            raw_vertices = edit["vertices"]
            if not isinstance(raw_vertices, list) or any(
                not isinstance(item, Mapping) or set(item) != {"x", "y"}
                for item in raw_vertices
            ):
                raise ValueError("vertices must contain x/y objects")
            vertices = tuple(
                (item["x"], item["y"]) for item in raw_vertices
            )
            draft = add_planar_polygon(
                part.geometry_recipe,
                vertices=vertices,
            )
            summary = (
                f"在部件 {part.name} 的平面草图中增加"
                f"{len(vertices)} 边闭合轮廓"
            )
        elif operation == "update_point":
            allowed = {"point_id", "x", "y"}
            if (
                not {"point_id"} <= set(edit) <= allowed
                or len(edit) == 1
            ):
                raise ValueError("update_point fields do not match")
            draft = update_planar_point(part.geometry_recipe, **edit)
            summary = f"更新部件 {part.name} 中的草图点 {edit['point_id']}"
        elif operation == "update_circle":
            allowed = {"circle_id", "center_x", "center_y", "radius"}
            if (
                not {"circle_id"} <= set(edit) <= allowed
                or len(edit) == 1
            ):
                raise ValueError("update_circle fields do not match")
            draft = update_planar_circle(part.geometry_recipe, **edit)
            summary = f"更新部件 {part.name} 中的圆 {edit['circle_id']}"
        elif operation == "update_line":
            allowed = {"line_id", "start", "end"}
            if not {"line_id"} <= set(edit) <= allowed or len(edit) == 1:
                raise ValueError("update_line fields do not match")
            draft = update_planar_line(part.geometry_recipe, **edit)
            summary = f"更新部件 {part.name} 中的直线 {edit['line_id']}"
        elif operation == "update_arc":
            allowed = {"arc_id", "start", "center", "end", "orientation"}
            if not {"arc_id"} <= set(edit) <= allowed or len(edit) == 1:
                raise ValueError("update_arc fields do not match")
            draft = update_planar_arc(part.geometry_recipe, **edit)
            summary = f"更新部件 {part.name} 中的圆弧 {edit['arc_id']}"
        elif operation == "delete_curves":
            if set(edit) != {"curve_ids"}:
                raise ValueError("delete_curves fields do not match")
            draft = delete_planar_curves(part.geometry_recipe, **edit)
            summary = f"从部件 {part.name} 的平面草图中删除直线或圆弧"
        elif operation == "delete_circles":
            if set(edit) != {"circle_ids"}:
                raise ValueError("delete_circles fields do not match")
            draft = delete_planar_circles(part.geometry_recipe, **edit)
            summary = (
                f"从部件 {part.name} 的平面草图中删除 "
                f"{len(edit['circle_ids'])} 个圆"
            )
        elif operation == "replace_circle_pattern":
            if set(edit) != {
                "target_circle_ids",
                "count",
                "start_center_x",
                "start_center_y",
                "spacing_x",
                "spacing_y",
                "radius",
            }:
                raise ValueError("replace_circle_pattern fields do not match")
            draft = replace_planar_circle_pattern(part.geometry_recipe, **edit)
            summary = (
                f"将部件 {part.name} 中的圆替换为 "
                f"{edit['count']} 孔线性阵列"
            )
        elif operation == "add_constraint":
            if set(edit) != {"constraint"}:
                raise ValueError("add_constraint fields do not match")
            draft = add_planar_constraint(part.geometry_recipe, **edit)
            summary = f"为部件 {part.name} 的平面草图增加约束"
        elif operation == "replace_constraint":
            if set(edit) != {"constraint_id", "constraint"}:
                raise ValueError("replace_constraint fields do not match")
            draft = replace_planar_constraint(part.geometry_recipe, **edit)
            summary = f"替换部件 {part.name} 的草图约束 {edit['constraint_id']}"
        elif operation == "delete_constraints":
            if set(edit) != {"constraint_ids"}:
                raise ValueError("delete_constraints fields do not match")
            draft = delete_planar_constraints(part.geometry_recipe, **edit)
            summary = f"从部件 {part.name} 的平面草图中删除约束"
        elif operation == "batch":
            if set(edit) != {"edits"}:
                raise ValueError("batch fields do not match")
            raw_edits = edit["edits"]
            if not isinstance(raw_edits, list):
                raise TypeError("batch edits must be an array")
            draft = apply_planar_edit_batch(
                part.geometry_recipe,
                edits=raw_edits,
            )
            summary = (
                f"批量修改部件 {part.name} 的平面草图："
                f"{len(raw_edits)} 个原子步骤"
            )
        elif operation == "translate":
            if not {"dx", "dy"} <= set(edit) <= {"dx", "dy", "dz"}:
                raise ValueError("translate fields do not match")
            base_draft = geometry_draft(part.geometry_recipe)
            draft = translate_geometry(base_draft, **edit)
            summary = f"平移部件 {part.name}"
        elif operation == "rotate":
            if set(edit) != {"axis", "angle_degrees"}:
                raise ValueError("rotate fields do not match")
            base_draft = geometry_draft(part.geometry_recipe)
            draft = rotate_geometry(base_draft, **edit)
            summary = f"旋转部件 {part.name}"
        else:
            raise ValueError("unsupported incremental geometry edit")
        metadata = envelope(controller, "geometry-edit")
        suffix = str(metadata.pop("identity_suffix"))
        edit_mode = planned_geometry_edit_mode()
        proposal = create_geometry_edit_proposal(
            proposal_id=f"proposal-{suffix}",
            context=current_context(),
            part_id=part_id,
            draft=draft,
            summary=summary,
            edit_mode=edit_mode,
            **metadata,
        )
        return proposal_outcome(
            proposal,
            summary=summary,
            impact=geometry_edit_impact(
                edit_mode,
                "确认后在当前模型中更新该部件，并使旧网格与结果失效",
            ),
            confirm_label="应用修改",
            extra_data={"geometry_edit_mode": edit_mode},
        )

    def prepare_geometry_edit_with_diagnostics(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        try:
            return prepare_geometry_edit(arguments, controller)
        except PlanarEditValidationError as error:
            data = error.to_provider_dict()
            data["retry_guidance"] = {
                "action": "revise_and_retry_same_geometry_edit",
                "present_confirmation": False,
                "ask_user_for_geometry_repair": False,
                "requirements": [
                    "trace the complete boundary in order",
                    "close every intended Profile",
                    "remove crossings, overlaps, and T-junctions",
                    "keep cutout Profiles inside a material Profile",
                ],
            }
            first = error.diagnostics[0]
            code = str(first["code"])
            affected = ", ".join(
                str(item) for item in first["affected_logical_ids"]
            )
            location = f" ({affected})" if affected else ""
            return AuthoringToolOutcome(
                "Exact planar Profile validation rejected the edit: "
                f"{code}{location}. Revise the contour before presenting a proposal.",
                data,
                ok=False,
            )

    def prepare_geometry_with_diagnostics(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        """Keep composite Profile failures on the same typed boundary."""

        geometry = arguments.get("geometry")
        kind = (
            str(geometry.get("kind", ""))
            if isinstance(geometry, Mapping)
            else ""
        )
        try:
            return prepare_geometry(arguments, controller)
        except _GeometryIntentMismatchError as error:
            message = str(error)
            return AuthoringToolOutcome(
                message,
                {
                    "diagnostic": {
                        "code": "geometry.intent-dimension-mismatch",
                        "message": message,
                        "retryable": True,
                        "submitted_kind": error.submitted_kind,
                        "expected_kinds": list(error.expected_kinds),
                    }
                },
                ok=False,
            )
        except (AuthoringContractError, TypeError, ValueError) as error:
            if kind in {
                "extruded_profiles",
                "extruded_path_slot_plate",
            }:
                return _profile_transform_error(error, operation="复合 Profile 拉伸")
            if kind == "path_swept_profile":
                return _profile_transform_error(
                    error,
                    operation="复合路径扫掠",
                    first_failed_member=_first_failed_path_member(
                        geometry.get("path") if isinstance(geometry, Mapping) else None
                    ),
                )
            raise

    def mesh_refinement_entities():
        snapshot = current_session().snapshot()
        part = next(
            (
                candidate
                for candidate in snapshot.parts
                if candidate.id == snapshot.active_part_id
                and not candidate.suppressed
            ),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError(
                "there is no active Part with editable mesh topology"
            )
        topology = describe_recipe_topology(part.geometry_recipe)
        if not topology.exact:
            raise AuthoringContractError(
                "active Part topology is not exact enough for local refinement"
            )
        entities = tuple(
            entity
            for entity in topology.selectable_entities()
            if entity.kind in {"point", "edge", "face"}
        )
        return part, entities

    def read_mesh_refinement_context(
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        part, entities = mesh_refinement_entities()
        visible = entities[:128]
        rows = []
        for entity in visible:
            falloff_references = ["global_size"]
            try:
                resolve_target_radius(
                    part.geometry_recipe,
                    LogicalEntityRef(entity.logical_id),
                )
            except ValueError:
                pass
            else:
                falloff_references.append("target_radius")
            rows.append(
                {
                    "logical_id": entity.logical_id,
                    "kind": entity.kind,
                    "semantic_role": entity.semantic_role,
                    "allowed_falloff_references": falloff_references,
                }
            )
        return AuthoringToolOutcome(
            f"已读取 {len(visible)} 个可用于局部加密的稳定逻辑实体。",
            {
                "part_id": str(part.id),
                "entities": rows,
                "entity_count": len(entities),
                "truncated": len(entities) > len(visible),
            },
        )

    def read_model_topology_context(
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        snapshot = current_session().snapshot()
        model = getattr(snapshot.artifact, "model", None)
        if (
            snapshot.source_kind != "native"
            or not snapshot.model_current
            or not snapshot.mesh_current
            or model is None
        ):
            raise AuthoringContractError(
                "model topology context requires one current native mesh"
            )
        entries: list[dict[str, object]] = []
        truncated = False
        for part in sorted(snapshot.parts, key=lambda item: str(item.id)):
            if part.suppressed or part.geometry_recipe is None:
                continue
            topology = describe_recipe_topology(part.geometry_recipe)
            if not topology.exact:
                raise AuthoringContractError(
                    f"Part {part.id} has no exact logical topology"
                )
            for entity in topology.selectable_entities():
                reference = LogicalEntityRef(
                    namespace_part_logical_id(
                        str(part.id),
                        entity.logical_id,
                    )
                )
                for mesh_kind in ("node", "edge", "face", "element"):
                    try:
                        materialized = mesh_references_for_logical_entities(
                            model,
                            (reference,),
                            mesh_kind=mesh_kind,
                        )
                    except ValueError:
                        continue
                    if len(entries) == 128:
                        truncated = True
                        break
                    entries.append(
                        {
                            "part_id": str(part.id),
                            "part_name": str(part.name),
                            "logical_id": entity.logical_id,
                            "kind": entity.kind,
                            "semantic_role": entity.semantic_role,
                            "mesh_kind": mesh_kind,
                            "matched_count": len(materialized),
                        }
                    )
                if truncated:
                    break
            if truncated:
                break
        if not entries:
            raise AuthoringContractError(
                "current native mesh exposes no materializable logical entities"
            )
        return AuthoringToolOutcome(
            f"已读取 {len(entries)} 项可物化的模型拓扑。",
            {
                "entries": entries,
                "entry_count": len(entries),
                "truncated": truncated,
            },
        )

    def prepare_mesh(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if not set(arguments) <= {"local_refinements"}:
            raise AuthoringContractError(
                "prepare_mesh_proposal accepts only local_refinements"
            )
        requirements = controller.collected_requirements("mesh")
        metadata = envelope(controller, "mesh")
        context = current_context()
        part_id = context.active_part_id
        if part_id is None:
            raise AuthoringContractError("there is no active Part to mesh")
        part, entities = mesh_refinement_entities()
        available_targets = {entity.logical_id for entity in entities}
        raw_refinements = arguments.get("local_refinements", [])
        if not isinstance(raw_refinements, list):
            raise TypeError("local_refinements must be an array")
        if len(raw_refinements) > 32:
            raise ValueError("local_refinements exceeds the 32-item bound")
        local_controls = []
        local_summary = []
        for raw_refinement in raw_refinements:
            if not isinstance(raw_refinement, Mapping):
                raise TypeError("each local refinement must be an object")
            refinement = dict(raw_refinement)
            if set(refinement) != {"target", "size", "falloff"}:
                raise ValueError("local refinement fields do not match")
            target = str(refinement["target"])
            if target not in available_targets:
                raise ValueError(
                    "local refinement target is not one current selectable "
                    "logical entity"
                )
            raw_falloff = refinement["falloff"]
            if not isinstance(raw_falloff, Mapping):
                raise TypeError("local refinement falloff must be an object")
            falloff = dict(raw_falloff)
            if set(falloff) != {
                "reference",
                "start_factor",
                "end_factor",
            }:
                raise ValueError("local refinement falloff fields do not match")
            if falloff["reference"] == "target_radius":
                try:
                    resolve_target_radius(
                        part.geometry_recipe,
                        LogicalEntityRef(target),
                    )
                except ValueError as error:
                    raise ValueError(
                        "target_radius falloff is unavailable for the selected "
                        "logical entity"
                    ) from error
            control = LocalMeshControl(
                LogicalEntityRef(target),
                refinement["size"],
                MeshSizeFalloff(
                    falloff["reference"],
                    falloff["start_factor"],
                    falloff["end_factor"],
                ),
            )
            local_controls.append(control)
            local_summary.append(
                {
                    "target": control.target.logical_id,
                    "size": control.size,
                    "falloff": {
                        "reference": control.falloff.reference,
                        "start_factor": control.falloff.start_factor,
                        "end_factor": control.falloff.end_factor,
                    },
                }
            )
        intent = MeshIntent(
            str(requirements["mesh_cell_shape"]),
            int(requirements["mesh_order"]),
            global_size=float(requirements["mesh_global_size"]),
            local_controls=tuple(local_controls),
            line_element_type=(
                str(requirements["line_element_type"])
                if part.dimension == 1
                else None
            ),
        )
        try:
            intent.validate_recipe_capability(part.geometry_recipe)
        except ValueError as error:
            message = str(error)
            if not message.startswith("mesh.hex.unsupported-shape:"):
                raise
            return AuthoringToolOutcome(
                message,
                {
                    "state": "failed",
                    "diagnostic_code": "mesh.hex.unsupported-shape",
                },
                ok=False,
            )
        suffix = str(metadata.pop("identity_suffix"))
        proposal = create_mesh_proposal(
            proposal_id=f"proposal-{suffix}",
            context=context,
            part_id=part_id,
            mesh_intent=intent,
            **metadata,
        )
        return proposal_outcome(
            proposal,
            summary=(
                f"划分 {requirements['mesh_order']} 阶"
                f"{requirements['mesh_cell_shape']}网格；全局尺寸 "
                f"{requirements['mesh_global_size']}；局部加密 "
                f"{len(local_controls)} 项"
                + (
                    f"；线单元 {requirements['line_element_type']}"
                    if part.dimension == 1
                    else ""
                )
            ),
            impact="确认后划分网格，成功时安装网格并刷新 GUI",
            confirm_label="开始划分",
            extra_data={
                "local_refinements": local_summary,
            },
        )

    def apply_model_definition(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) != {"action", "parameters"}:
            raise AuthoringContractError(
                "apply_model_definition requires action and parameters"
        )
        metadata = envelope(controller, "definition")
        suffix = str(metadata.pop("identity_suffix"))
        change = create_definition_change(
            patch_id=f"patch-{suffix}",
            proposal_id=f"proposal-{suffix}",
            context=current_context(),
            snapshot=current_session().snapshot(),
            action=arguments["action"],
            parameters=arguments["parameters"],
            **metadata,
        )
        if type(change.value) is AgentProposal:
            proposal = change.value
            return proposal_outcome(
                proposal,
                summary=str(proposal.display_summary["summary"]),
                impact=str(proposal.display_summary["impact"]),
                confirm_label=str(proposal.display_summary["confirm_label"]),
                extra_data={
                    "action": change.action,
                    "definition_object_type": change.resume_object_type,
                    "objects": list(
                        proposal.expected_changes.get("created_names", ())
                    ),
                },
            )
        patch = change.value
        provisional = AuthoringToolOutcome(
            str(patch.display_summary["summary"]),
            {
                "state": "succeeded",
                "action": arguments["action"],
                "patch_id": patch.patch_id,
                "undo_available": True,
                "objects": list(
                    patch.expected_changes.get("created_names", ())
                ),
                "gui_synchronized": True,
                "definition_object_type": change.resume_object_type,
            },
        )
        provider_safe_authoring_payload(provisional.data)
        applied = authoring_bridge.apply_automatic_patch(patch)
        final = AuthoringToolOutcome(
            str(patch.display_summary["summary"]),
            {
                "state": "succeeded",
                "action": arguments["action"],
                "patch_id": applied.patch.patch_id,
                "undo_available": applied.undo_available,
                "objects": list(
                    patch.expected_changes.get("created_names", ())
                ),
                "gui_synchronized": True,
                "definition_object_type": change.resume_object_type,
            },
        )
        provider_safe_authoring_payload(final.data)
        return final

    def run_preflight(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        step_name = resolved_step_name(arguments, operation="preflight")
        if not current_session().can_check(step_name):
            raise AuthoringContractError(
                "preflight requires a current runnable analysis step"
            )
        record = authoring_bridge.request_preflight(step_name)
        return AuthoringToolOutcome(
            "Native preflight was submitted through the existing GUI task.",
            {
                "request_id": record.request_id,
                "state": record.state.value,
                "passed": record.state is AgentPreflightState.PASSED,
            },
        )

    def prepare_solve(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        metadata = envelope(controller, "solve")
        suffix = str(metadata.pop("identity_suffix"))
        step_name = resolved_step_name(arguments, operation="solve")
        if not current_session().can_submit(step_name):
            raise AuthoringContractError(
                "solve requires a current validated runnable analysis step"
            )
        supplied_job_name = (
            current_session().next_run_name()
            if next_job_name is None
            else next_job_name()
        )
        if type(supplied_job_name) is not str or not supplied_job_name.strip():
            raise AuthoringContractError(
                "next_job_name must return a nonblank string"
            )
        proposal = create_solve_proposal(
            proposal_id=f"proposal-{suffix}",
            snapshot=current_session().snapshot(),
            step_name=step_name,
            job_name=supplied_job_name.strip(),
            **metadata,
        )
        return proposal_outcome(
            proposal,
            summary=f"提交 {step_name} 并绑定当前 validation stamp",
            impact="接受后后台执行当前已预检的线性静力模型",
            confirm_label="开始求解",
        )

    def request_project_save(
        _arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        metadata = envelope(controller, "project-save")
        suffix = str(metadata["identity_suffix"])
        context = current_context()
        preview = controller.preview_project_save_proposal(
            f"proposal-{suffix}",
            context,
        )
        provisional = AuthoringToolOutcome(
            "Project save is waiting for the local GUI control.",
            {
                "proposal_id": preview.proposal_id,
                "proposal_hash": preview.proposal_hash,
                "state": preview.state.value,
                "proposal_view": {
                    "proposal_id": preview.proposal_id,
                    "proposal_hash": preview.proposal_hash,
                    "proposal_kind": "project_save",
                    "title": "保存当前自主项目",
                    "summary": "保存当前已接受的模型状态",
                    "impact": "确认后调用本地项目保存；未确认草稿不会写入",
                    "confirm_label": "保存模型",
                    "target_document_id": preview.target_document_id,
                    "target_session_id": preview.target_session_id,
                    "base_session_revision": preview.base_session_revision,
                },
                "continuation_checkpoint": {
                    "session_id": str(metadata["agent_session_id"]),
                    "source_turn_id": str(metadata["turn_id"]),
                    "proposal_id": preview.proposal_id,
                    "proposal_hash": preview.proposal_hash,
                    "model_revision": preview.base_session_revision,
                    "proposal_kind": "project_save",
                },
            },
        )
        provider_safe_authoring_payload(provisional.data)
        controller.register_project_save_proposal(
            preview.proposal_id,
            context,
        )
        return provisional

    def create_native_model_document(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) - {"model_name"}:
            raise AuthoringContractError(
                "create_native_model_document has unknown fields"
            )
        if create_model_document is None:
            raise AuthoringContractError(
                "native model document creation is unavailable"
            )
        requested_name = arguments.get("model_name")
        if requested_name is not None and (
            type(requested_name) is not str
            or not requested_name.strip()
            or requested_name != requested_name.strip()
            or len(requested_name) > 160
        ):
            raise AuthoringContractError(
                "model_name must be a nonblank trimmed string"
            )
        previous = current_context().binding
        model_name = create_model_document(requested_name)
        if type(model_name) is not str or not model_name.strip():
            raise AuthoringContractError(
                "model document creation returned no model name"
            )
        context = current_context()
        if (
            context.binding.document_id == previous.document_id
            or context.binding.session_id == previous.session_id
        ):
            raise AuthoringContractError(
                "model document creation did not activate a new binding"
            )
        return AuthoringToolOutcome(
            f"Created and activated additional native model {model_name.strip()}.",
            {
                "state": "succeeded",
                "model_name": model_name.strip(),
                "target_document_id": context.binding.document_id,
                "target_session_id": context.binding.session_id,
                "model_revision": context.binding.session_revision,
                "preserved_existing_documents": True,
                "next_action": (
                    "read_authoring_context_then_prepare_requested_geometry"
                ),
            },
        )

    def read_deletable_objects(
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        catalog = deletable_object_catalog(current_session().snapshot(), limit=128)
        visible = catalog[:100]
        return AuthoringToolOutcome(
            "Current deletable native model objects read locally.",
            {
                "objects": [item.to_provider_dict() for item in visible],
                "count": len(visible),
                "truncated": len(catalog) > len(visible),
            },
        )

    def prepare_delete(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        allowed = {"object_type", "target_id", "step_name"}
        if set(arguments) - allowed:
            raise AuthoringContractError(
                "prepare_delete_proposal has unknown fields"
            )
        if not {"object_type", "target_id"}.issubset(arguments):
            raise AuthoringContractError(
                "prepare_delete_proposal requires object_type and target_id"
            )
        metadata = envelope(controller, "delete")
        suffix = str(metadata.pop("identity_suffix"))
        proposal, target = create_delete_proposal(
            proposal_id=f"proposal-{suffix}",
            context=current_context(),
            snapshot=current_session().snapshot(),
            object_type=arguments["object_type"],
            target_id=arguments["target_id"],
            step_name=arguments.get("step_name"),
            **metadata,
        )
        return proposal_outcome(
            proposal,
            summary=str(proposal.display_summary["summary"]),
            impact=str(proposal.display_summary["impact"]),
            confirm_label=str(proposal.display_summary["confirm_label"]),
            extra_data={"delete_object_type": target.object_type},
        )

    def read_editable_objects(
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        catalog = editable_object_catalog(current_session().snapshot(), limit=128)
        visible: list[object] = []
        for item in catalog[:100]:
            candidate = [
                *(entry.to_provider_dict() for entry in visible),
                item.to_provider_dict(),
            ]
            try:
                provider_safe_authoring_payload(
                    {
                        "objects": candidate,
                        "count": len(candidate),
                        "truncated": len(candidate) < len(catalog),
                    }
                )
            except ValueError:
                break
            visible.append(item)
        return AuthoringToolOutcome(
            "Current editable model objects read locally.",
            {
                "objects": [item.to_provider_dict() for item in visible],
                "count": len(visible),
                "truncated": len(catalog) > len(visible),
            },
        )

    def edit_model_object(
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        allowed = {"object_type", "target_id", "step_name", "changes"}
        if set(arguments) - allowed:
            raise AuthoringContractError(
                "edit_model_object has unknown fields"
            )
        if not {"object_type", "target_id", "changes"}.issubset(arguments):
            raise AuthoringContractError(
                "edit_model_object requires "
                "object_type, target_id, and changes"
            )
        metadata = envelope(controller, "edit")
        suffix = str(metadata.pop("identity_suffix"))
        patch, target = create_edit_patch(
            patch_id=f"patch-{suffix}",
            context=current_context(),
            snapshot=current_session().snapshot(),
            object_type=arguments["object_type"],
            target_id=arguments["target_id"],
            step_name=arguments.get("step_name"),
            changes=arguments["changes"],
            **metadata,
        )
        provisional = AuthoringToolOutcome(
            str(patch.display_summary["summary"]),
            {
                "state": "succeeded",
                "edit_object_type": target.object_type,
                "target_id": target.target_id,
                "patch_id": patch.patch_id,
                "undo_available": True,
                "gui_synchronized": True,
            },
        )
        provider_safe_authoring_payload(provisional.data)
        applied = authoring_bridge.apply_automatic_patch(patch)
        final = AuthoringToolOutcome(
            str(patch.display_summary["summary"]),
            {
                "state": "succeeded",
                "edit_object_type": target.object_type,
                "target_id": target.target_id,
                "patch_id": applied.patch.patch_id,
                "undo_available": applied.undo_available,
                "gui_synchronized": True,
            },
        )
        provider_safe_authoring_payload(final.data)
        return final

    def read_catalog(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) - {"run_id", "target"}:
            raise AuthoringContractError("result catalog has unknown fields")
        response = result_bridge.catalog(
            arguments.get("run_id"),
            target=arguments.get("target"),
        )
        return AuthoringToolOutcome(
            "Accepted result catalog read locally.",
            response.to_dict(),
            ok=response.ok,
        )

    def read_analysis_run_catalog(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) - {"cursor", "limit", "target"}:
            raise AuthoringContractError("run catalog has unknown fields")
        cursor = arguments.get("cursor", 0)
        limit = arguments.get("limit", 20)
        if type(cursor) is not int or cursor < 0:
            raise AuthoringContractError("cursor must be a non-negative integer")
        if (
            type(limit) is not int
            or limit < 1
            or limit > ANALYSIS_RUN_CATALOG_MAX_LIMIT
        ):
            raise AuthoringContractError("limit is outside the run catalog bound")
        target = arguments.get("target")
        if target is None and workspace_catalog_bridge is None:
            binding = current_context().binding
            target = {
                "document_id": binding.document_id,
                "session_id": binding.session_id,
            }
        catalog = result_bridge.analysis_runs(
            cursor=cursor,
            limit=limit,
            target=target,
        )
        return AuthoringToolOutcome(
            "Analysis run catalog read locally.",
            catalog.to_dict(),
        )

    def read_workspace_documents(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if arguments:
            raise AuthoringContractError(
                "read_workspace_documents accepts no arguments"
            )
        if workspace_catalog_bridge is None:
            raise AuthoringContractError("workspace catalog is unavailable")
        return AuthoringToolOutcome(
            "Workspace document catalog read locally.",
            workspace_catalog_bridge.catalog().to_dict(),
        )

    def read_geometry_feature_catalog(
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        snapshot = current_session().snapshot()
        active_parts = tuple(
            part
            for part in snapshot.parts
            if not part.suppressed and part.geometry_recipe is not None
        )
        source_parts = active_parts[:128]
        catalogs: list[dict[str, object]] = []
        for part in source_parts:
            item = feature_topology_catalog(
                part.geometry_recipe,
                part_id=str(part.id),
            )
            candidate = {
                "kind": "native_geometry_feature_catalog",
                "schema_version": 1,
                "session_revision": snapshot.session_revision,
                "parts": [*catalogs, item],
                "truncated": False,
                "omitted_part_count": 0,
            }
            try:
                provider_safe_authoring_payload(candidate)
            except ValueError:
                break
            catalogs.append(item)
        omitted = len(active_parts) - len(catalogs)
        data = {
            "kind": "native_geometry_feature_catalog",
            "schema_version": 1,
            "session_revision": snapshot.session_revision,
            "parts": catalogs,
            "truncated": omitted > 0,
            "omitted_part_count": omitted,
        }
        provider_safe_authoring_payload(data)
        return AuthoringToolOutcome(
            "Native geometry feature catalog read locally.",
            data,
        )

    def query_result(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        response = result_bridge.query(arguments)
        return AuthoringToolOutcome(
            "One accepted result scalar read locally.",
            response.to_dict(),
            ok=response.ok,
        )

    def compare_results(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        response = result_bridge.compare(arguments)
        return AuthoringToolOutcome(
            "Two accepted result scalars compared locally.",
            response.to_dict(),
            ok=response.ok,
        )

    def workspace_result_inventory() -> tuple[int, int]:
        if workspace_catalog_bridge is None:
            return 0, 0
        catalog = workspace_catalog_bridge.catalog()
        return (
            sum(item.run_count for item in catalog.documents),
            sum(item.result_count for item in catalog.documents),
        )

    controller = AuthoringWorkflowController(
        current_context,
        {
            "prepare_geometry_proposal": prepare_geometry_with_diagnostics,
            "prepare_planar_construction_proposal": prepare_planar_construction,
            "read_profile_transform_context": read_profile_transform_context,
            "prepare_profile_extrusion": prepare_profile_extrusion,
            "prepare_profile_revolution": prepare_profile_revolution,
            "prepare_profile_path_sweep": prepare_profile_path_sweep,
            "read_geometry_edit_context": read_geometry_edit_context,
            "prepare_geometry_edit": prepare_geometry_edit_with_diagnostics,
            "read_mesh_refinement_context": read_mesh_refinement_context,
            "read_model_topology_context": read_model_topology_context,
            "prepare_mesh_proposal": prepare_mesh,
            "apply_model_definition": apply_model_definition,
            "run_native_preflight": run_preflight,
            "prepare_solve_proposal": prepare_solve,
            "request_project_save": request_project_save,
            **(
                {}
                if create_model_document is None
                else {
                    "create_native_model_document": create_native_model_document
                }
            ),
            "read_deletable_objects": read_deletable_objects,
            "prepare_delete_proposal": prepare_delete,
            "read_editable_model_objects": read_editable_objects,
            "edit_model_object": edit_model_object,
            "read_accepted_result_catalog": read_catalog,
            "read_analysis_run_catalog": read_analysis_run_catalog,
            "read_geometry_feature_catalog": read_geometry_feature_catalog,
            "query_accepted_result": query_result,
            "compare_accepted_results": compare_results,
            **(
                {}
                if workspace_catalog_bridge is None
                else {"read_workspace_documents": read_workspace_documents}
            ),
        },
        workspace_result_inventory=(
            None
            if workspace_catalog_bridge is None
            else workspace_result_inventory
        ),
    )
    if authoring_bridge.context is not None:
        controller.observe_binding(current_context())
    return controller


__all__ = [
    "AppliedPatchRecord",
    "AppliedPatchState",
    "AgentPreflightRecord",
    "AgentPreflightState",
    "AgentPreflightTaskRequest",
    "AgentGeometryMutation",
    "AgentMeshTaskRequest",
    "AgentProposalPreview",
    "AgentSolveTaskRequest",
    "AgentAuthoringBridge",
    "AgentProposal",
    "AgentResultQueryBridge",
    "AuthoringWorkflowController",
    "AuthoringWorkflowStage",
    "BridgeReceipt",
    "ProposalState",
    "SessionResultQueryPort",
    "SessionGeometryAuthoringPort",
    "authoring_context_from_snapshot",
    "create_session_authoring_workflow_controller",
]
