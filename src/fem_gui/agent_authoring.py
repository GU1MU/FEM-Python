"""A1 GUI boundary for detached FEM Agent authoring.

The adapter reads a detached session snapshot into bounded DTOs.  The bridge
owns only those DTOs and an ``AuthoringPort``; it never stores or mutates a
``ModelSession``.
"""

from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Protocol

from fem import geometry as geometry_runtime
from fem.application import (
    ModelSession,
    UnitContext,
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
    add_planar_circle,
    add_planar_polygon,
    add_planar_rectangle,
    box_geometry,
    create_geometry_edit_proposal,
    create_geometry_proposal,
    create_profile_extrusion_proposal,
    create_profile_path_sweep_proposal,
    create_profile_revolution_proposal,
    cylinder_geometry,
    geometry_draft,
    feature_topology_catalog,
    geometry_recipe_from_payload,
    planar_geometry_catalog,
    planar_polygon_geometry,
    planar_sketch_geometry,
    rotate_geometry,
    translate_geometry,
    update_planar_circle,
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
    AcceptedResultSource,
    AgentResultAggregation,
    AgentResultCatalog,
    AgentResultCatalogResponse,
    AgentResultField,
    AgentResultLocation,
    AgentResultQuery,
    AgentResultQueryResponse,
    AgentResultScalar,
    AgentResultVariable,
    AgentResultQueryBridge,
)
from fem.geometry import (
    BooleanGeometry,
    BooleanLineageResolutionError,
    ExtrudedGeometry,
    LogicalEntityRef,
    MultiBodyGeometry,
    PathSweptGeometry,
    RevolvedGeometry,
    SketchGeometry,
    SketchCircle,
    SketchRectangle,
    WireMember,
    WireGeometry,
    WirePoint,
    describe_recipe_topology,
    namespace_part_logical_id,
    resolve_extrusion_source_faces,
    resolve_target_radius,
)
from fem.mesh.settings import LocalMeshControl, MeshSizeFalloff


class _SessionSnapshot(Protocol):
    session_id: str
    session_revision: int
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
    displayed_result: object | None
    mesh_current: bool
    unit_context: object | None


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

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="agent-extrusion-preflight",
    ) as executor:
        executor.submit(compile_all).result()


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

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="agent-derived-preflight",
    ) as executor:
        executor.submit(compile_one).result()


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


def authoring_context_from_snapshot(
    snapshot: _SessionSnapshot,
) -> AuthoringContext:
    """Copy a session projection into a bounded, provider-safe DTO."""

    session_id = str(snapshot.session_id)
    source_kind = "blank" if snapshot.source_kind is None else str(snapshot.source_kind)
    supported = source_kind in {"blank", "native"}
    binding = LocalModelBinding(
        document_id=f"document:{session_id}",
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
            if records and all(bool(getattr(item, "passed", False)) for item in records)
            else "blocked"
        )
    runs = tuple(getattr(snapshot, "runs", ()))
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
                else "当前 native 项目没有可编辑的作用域、边界条件或载荷"
            ),
        ),
        CapabilitySummary(
            "query_accepted_result",
            (
                supported
                and source_kind == "native"
                and snapshot.displayed_result is not None
            ),
            (
                None
                if (
                    supported
                    and source_kind == "native"
                    and snapshot.displayed_result is not None
                )
                else "结果查询需要当前已接受的 native 结果"
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
class AgentPreflightTaskRequest:
    """Exact metadata used to run an automatic local preflight."""

    request_id: str
    step_name: str
    base_session_revision: int
    session_id: str
    artifact_id: str
    model_revision: int


class SessionResultQueryPort:
    """A7 read-only adapter over the Session's exact accepted result provider."""

    def __init__(self, session: ModelSession) -> None:
        if type(session) is not ModelSession:
            raise TypeError("session must be ModelSession")
        self._session = session

    def catalog(self) -> AgentResultCatalogResponse:
        provider = self._session.current_result_provider()
        identity = self._session.current_result_identity()
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
        projection = self._session.projection_snapshot()
        units = projection.unit_context
        if projection.source_kind != "native" or units is None:
            return _result_catalog_failure(
                "result.catalog.units_unavailable",
                "A native project unit context is required for result values.",
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
                in {ResultVariable.U, ResultVariable.RF, ResultVariable.S}
            )
        )
        if not fields:
            return _result_catalog_failure(
                "result.catalog.no_supported_ready_fields",
                "The accepted result has no READY U, RF, or S fields.",
                clarification_required=True,
            )
        named_regions = tuple(projection.named_regions.values())
        nodal_regions = (
            "all_nodes",
            *tuple(
                region.name
                for region in named_regions[:127]
                if region.entity_kind in {"node", "edge", "face"}
            ),
        )
        element_regions = (
            "all_elements",
            *tuple(
                region.name
                for region in named_regions[:127]
                if region.entity_kind == "element"
            ),
        )
        catalog = AgentResultCatalog(
            source=_accepted_source(source),
            materialization_generation=generation,
            fields=fields,
            nodal_regions=tuple(dict.fromkeys(nodal_regions)),
            element_regions=tuple(dict.fromkeys(element_regions)),
        )
        if self._session.current_result_identity() != (source, generation):
            return _result_catalog_failure(
                "result.catalog.stale",
                "The accepted result changed before the catalog was returned.",
                retryable=True,
            )
        return AgentResultCatalogResponse.success(catalog)

    def query(self, request: AgentResultQuery) -> AgentResultQueryResponse:
        if type(request) is not AgentResultQuery:
            raise TypeError("request must be AgentResultQuery")
        provider = self._session.current_result_provider()
        identity = self._session.current_result_identity()
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

        projection = self._session.projection_snapshot()
        units = projection.unit_context
        if projection.source_kind != "native" or units is None:
            return _result_query_failure(
                "result.query.units_unavailable",
                "A native project unit context is required for result values.",
                clarification_required=True,
            )

        try:
            _require_published_result_region(projection, request)
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

        if self._session.current_result_identity() != (source, generation):
            return _result_query_failure(
                "result.query.stale",
                "The accepted result changed before the query completed.",
                retryable=True,
            )
        return AgentResultQueryResponse.success(scalar)


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
    projection: _SessionSnapshot,
    request: AgentResultQuery,
) -> None:
    if request.region in {"all_nodes", "all_elements"}:
        return
    regions = tuple(projection.named_regions.values())[:127]  # type: ignore[union-attr]
    matches = tuple(
        region for region in regions if region.name == request.region
    )
    if len(matches) != 1:
        raise _AgentResultQueryRejected(
            "result.query.region_not_published",
            "The requested named region is absent from the bounded result catalog.",
        )
    kind = matches[0].entity_kind
    if (
        request.variable
        in {
            AgentResultVariable.DISPLACEMENT,
            AgentResultVariable.REACTION_FORCE,
        }
        and kind not in {"node", "edge", "face"}
    ) or (
        request.variable is AgentResultVariable.STRESS
        and kind != "element"
    ):
        raise _AgentResultQueryRejected(
            "result.query.region_entity_unsupported",
            "The requested named region entity type does not support this variable.",
        )


def _native_result_query(
    provider: ResultProvider,
    field_key: FieldMaterializationKey,
    request: AgentResultQuery,
) -> NativeResultQuery:
    variable = request.variable
    region = request.region
    if variable in {
        AgentResultVariable.DISPLACEMENT,
        AgentResultVariable.REACTION_FORCE,
    }:
        if region == "all_elements":
            raise _AgentResultQueryRejected(
                "result.query.region_entity_unsupported",
                "Nodal U and RF queries cannot target all_elements.",
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
            "Stress S queries cannot target all_nodes.",
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
        AgentResultVariable.REACTION_FORCE: units.force,
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

    def set_record_listener(
        self,
        callback: Callable[[ProposalPortRecord], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("record listener must be callable")
        self._record_listener = callback

    def set_context(self, context: AuthoringContext) -> None:
        if type(context) is not AuthoringContext:
            raise AuthoringContractError("context must be AuthoringContext")
        self._context = context

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
        if (
            proposal.target_session_id != current_id
            or proposal.target_document_id != f"document:{current_id}"
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

            def commit_part_boolean() -> None:
                self._session.apply_part_boolean(
                    str(parameters["target_part_id"]),
                    str(parameters["tool_part_id"]),
                    str(parameters["operation"]),
                    str(parameters["result_name"]),
                    result_recipe=recipe,
                    expected_session_revision=proposal.base_session_revision,
                )

            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="agent-part-boolean-commit",
            ) as executor:
                executor.submit(commit_part_boolean).result()
            succeeded = replace(record, state=ProposalState.SUCCEEDED)
            self._records[proposal_id] = succeeded
            return succeeded
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

            def commit_body_boolean() -> None:
                self._session.apply_body_boolean(
                    str(parameters["part_id"]),
                    str(parameters["target_body_id"]),
                    str(parameters["tool_body_id"]),
                    str(parameters["operation"]),
                    str(parameters["result_name"]),
                    result=geometry,
                    expected_session_revision=proposal.base_session_revision,
                )

            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="agent-body-boolean-commit",
            ) as executor:
                executor.submit(commit_body_boolean).result()
            succeeded = replace(record, state=ProposalState.SUCCEEDED)
            self._records[proposal_id] = succeeded
            return succeeded
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
            if source_part is None or source_part.geometry_recipe != base_recipe:
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
            self._session.replace_part_with_extruded_siblings(
                part_id,
                recipes,
                expected_session_revision=proposal.base_session_revision,
            )
            succeeded = replace(record, state=ProposalState.SUCCEEDED)
            self._records[proposal_id] = succeeded
            return succeeded
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
            if source_part is None or source_part.geometry_recipe != base_recipe:
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
            def commit_derived_feature() -> None:
                self._session.replace_part_geometry(
                    part_id,
                    recipe,
                    expected_session_revision=proposal.base_session_revision,
                )

            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="agent-derived-commit",
            ) as executor:
                executor.submit(commit_derived_feature).result()
            succeeded = replace(record, state=ProposalState.SUCCEEDED)
            self._records[proposal_id] = succeeded
            return succeeded
        recipe = geometry_recipe_from_payload(parameters["recipe"])
        snapshot = self._session.snapshot()
        if operation.kind is OperationKind.REPLACE_PART_GEOMETRY:
            self._session.replace_part_geometry(
                str(parameters["part_id"]),
                recipe,
                expected_session_revision=proposal.base_session_revision,
            )
            succeeded = replace(record, state=ProposalState.SUCCEEDED)
            self._records[proposal_id] = succeeded
            return succeeded
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
        if (
            patch.target_session_id != snapshot.session_id
            or patch.target_document_id
            != f"document:{snapshot.session_id}"
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
        if has_results and authoring_mode not in {
            "direct_incremental",
            "direct_edit",
        }:
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
                "results": has_results,
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
        if (
            proposal.target_session_id != snapshot.session_id
            or proposal.target_document_id
            != f"document:{snapshot.session_id}"
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
        self._lifecycle_listener: (
            Callable[[AgentProposal, ProposalState, str], None] | None
        ) = None
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

    def bind_snapshot(
        self,
        snapshot: _SessionSnapshot,
    ) -> tuple[str, ...]:
        return self.bind_context(authoring_context_from_snapshot(snapshot))

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

    def register_proposal(self, proposal: AgentProposal) -> BridgeReceipt:
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
        self._require_live_target(proposal)
        record = self._port.present(proposal)
        self._records[proposal.proposal_id] = record
        self._idempotency[proposal.idempotency_key] = proposal.proposal_id
        return self._receipt(record)

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
) -> AuthoringWorkflowController:
    """Wire A1-A7 handlers to one GUI-owner A8 controller."""

    if type(session) is not ModelSession:
        raise TypeError("session must be exactly ModelSession")
    if type(authoring_bridge) is not AgentAuthoringBridge:
        raise TypeError("authoring_bridge must be AgentAuthoringBridge")
    if type(result_bridge) is not AgentResultQueryBridge:
        raise TypeError("result_bridge must be AgentResultQueryBridge")

    def current_context() -> AuthoringContext:
        context = authoring_bridge.context
        if context is None:
            raise AuthoringContractError("there is no current authoring binding")
        return context

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
            },
        }
        if extra_data is not None:
            data.update(dict(extra_data))
        provisional = AuthoringToolOutcome(summary, data)
        provider_safe_authoring_payload(provisional.data)
        receipt = authoring_bridge.register_proposal(proposal)
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
        return proposal_outcome(
            proposal,
            summary=proposal_summary,
            impact="确认后创建该几何并刷新 GUI",
            confirm_label="加入模型",
        )

    def read_geometry_edit_context(
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        part_id = str(arguments["part_id"])
        snapshot = session.snapshot()
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
                "add_circle",
                "add_rectangle",
                "add_polygon",
                "update_point",
                "update_circle",
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
        snapshot = session.snapshot()
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
                    result_part_id=session.next_native_part_id,
                    feature_id=session.next_part_boolean_feature_id,
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
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=(
                    "确认后创建一个 proven 结果 Part，抑制 target/tool 源 Part，"
                    "并使旧网格、定义与结果失效"
                ),
                confirm_label="执行精确 Part 布尔",
                extra_data={
                    "result_part_id": prepared.context.result_part_id,
                    "feature_id": prepared.context.feature_id,
                    "lineage_proven": True,
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
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=(
                    "确认后在同一 Part 内保留 target Body ID、消费 tool Body，"
                    "并使旧网格、定义与结果失效"
                ),
                confirm_label="执行精确 Body 布尔",
                extra_data={
                    "target_body_id": target_body_id,
                    "consumed_tool_body_id": tool_body_id,
                    "lineage_proven": True,
                },
            )
        if operation == "extrude_profiles":
            if set(edit) != {"source_face_ids", "height"}:
                raise ValueError("extrude_profiles fields do not match")
            base_recipe = part.geometry_recipe
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
            _preflight_profile_extrusions(recipes)
            summary = (
                f"选择式拉伸部件 {part.name} 的 {len(selection.face_ids)} 个 "
                f"Profile：高度 {_display_number(height)}，沿草图正法向，"
                f"生成 {len(selection.face_ids)} 个独立 Part"
            )
            metadata = envelope(controller, "geometry-edit")
            suffix = str(metadata.pop("identity_suffix"))
            proposal = create_profile_extrusion_proposal(
                proposal_id=f"proposal-{suffix}",
                context=current_context(),
                part_id=part_id,
                base_recipe=base_recipe,
                source_face_ids=selection.face_ids,
                height=height,
                summary=summary,
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=(
                    "确认后将选定 Profiles 原子转换为独立实体 Part，"
                    "并使旧网格、定义与结果失效"
                ),
                confirm_label="拉伸选定 Profiles",
            )
        if operation == "revolve_profile":
            if set(edit) != {"source_face_id", "axis", "angle_degrees"}:
                raise ValueError("revolve_profile fields do not match")
            base_recipe = part.geometry_recipe
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
            _preflight_derived_geometry(recipe)
            summary = (
                f"绕 {recipe.axis.upper()} 轴旋转扫掠部件 {part.name} 的 "
                f"Profile {source_face_id}：角度 {recipe.angle_degrees:g}°，"
                "生成 1 个实体 Part"
            )
            metadata = envelope(controller, "geometry-revolve")
            suffix = str(metadata.pop("identity_suffix"))
            proposal = create_profile_revolution_proposal(
                proposal_id=f"proposal-{suffix}",
                context=current_context(),
                part_id=part_id,
                base_recipe=base_recipe,
                source_face_id=source_face_id,
                axis=recipe.axis,
                angle_degrees=recipe.angle_degrees,
                summary=summary,
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=(
                    "确认后将 Profile 原子替换为旋转实体，"
                    "并使旧网格、定义与结果失效"
                ),
                confirm_label="旋转扫掠 Profile",
            )
        if operation == "path_sweep_profile":
            if set(edit) != {"source_face_id", "path", "frame_strategy"}:
                raise ValueError("path_sweep_profile fields do not match")
            base_recipe = part.geometry_recipe
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
            _preflight_derived_geometry(recipe)
            summary = (
                f"沿 {len(path.members)} 段显式开放折线路径扫掠部件 "
                f"{part.name} 的 Profile {source_face_id}；"
                f"frame={recipe.frame_strategy}，生成 1 个实体 Part"
            )
            metadata = envelope(controller, "geometry-path-sweep")
            suffix = str(metadata.pop("identity_suffix"))
            proposal = create_profile_path_sweep_proposal(
                proposal_id=f"proposal-{suffix}",
                context=current_context(),
                part_id=part_id,
                base_recipe=base_recipe,
                source_face_id=source_face_id,
                path=path,
                frame_strategy=recipe.frame_strategy,
                summary=summary,
                **metadata,
            )
            return proposal_outcome(
                proposal,
                summary=summary,
                impact=(
                    "确认后将 Profile 原子替换为路径扫掠实体，"
                    "并使旧网格、定义与结果失效"
                ),
                confirm_label="沿路径扫掠 Profile",
            )
        if operation == "add_circle":
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
        proposal = create_geometry_edit_proposal(
            proposal_id=f"proposal-{suffix}",
            context=current_context(),
            part_id=part_id,
            draft=draft,
            summary=summary,
            **metadata,
        )
        return proposal_outcome(
            proposal,
            summary=summary,
            impact="确认后原位更新该部件，并使旧网格与结果失效",
            confirm_label="应用修改",
        )

    def mesh_refinement_entities():
        snapshot = session.snapshot()
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
        snapshot = session.snapshot()
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
            snapshot=session.snapshot(),
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
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        steps = tuple(session.snapshot().steps)
        if len(steps) != 1:
            raise AuthoringContractError(
                "preflight requires exactly one current analysis step"
            )
        record = authoring_bridge.request_preflight(str(steps[0].name))
        return AuthoringToolOutcome(
            "Native preflight was submitted through the existing GUI task.",
            {
                "request_id": record.request_id,
                "state": record.state.value,
                "passed": record.state is AgentPreflightState.PASSED,
            },
        )

    def prepare_solve(
        _arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        metadata = envelope(controller, "solve")
        suffix = str(metadata.pop("identity_suffix"))
        steps = tuple(session.snapshot().steps)
        if len(steps) != 1:
            raise AuthoringContractError(
                "solve requires exactly one current analysis step"
            )
        step_name = str(steps[0].name)
        proposal = create_solve_proposal(
            proposal_id=f"proposal-{suffix}",
            snapshot=session.snapshot(),
            step_name=step_name,
            job_name=f"作业-{step_name.removeprefix('分析步-')}",
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
                },
            },
        )
        provider_safe_authoring_payload(provisional.data)
        controller.register_project_save_proposal(
            preview.proposal_id,
            context,
        )
        return provisional

    def read_deletable_objects(
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        catalog = deletable_object_catalog(session.snapshot(), limit=128)
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
            snapshot=session.snapshot(),
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
        catalog = editable_object_catalog(session.snapshot(), limit=128)
        visible = catalog[:100]
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
            snapshot=session.snapshot(),
            object_type=arguments["object_type"],
            target_id=arguments["target_id"],
            step_name=arguments.get("step_name"),
            changes=arguments["changes"],
            **metadata,
        )
        if patch.invalidation_impact.get("results") is True:
            proposal = AgentProposal.create(
                proposal_id=f"proposal-{suffix}",
                proposal_kind=ProposalKind.DESTRUCTIVE_EDIT,
                agent_session_id=patch.agent_session_id,
                turn_id=patch.turn_id,
                source_tool_call_ids=patch.source_tool_call_ids,
                target_document_id=patch.target_document_id,
                target_session_id=patch.target_session_id,
                base_session_revision=patch.base_session_revision,
                draft_revision=patch.draft_revision,
                operations=patch.operations,
                preconditions={
                    **patch.preconditions,
                    "accepted_result_confirmation_required": True,
                },
                expected_changes=patch.expected_changes,
                invalidation_impact=patch.invalidation_impact,
                display_summary={
                    "title": "编辑将使已有结果失效",
                    "summary": str(patch.display_summary["summary"]),
                    "impact": "已有验证、作业和结果将失效",
                    "confirm_label": "确认修改",
                },
            )
            return proposal_outcome(
                proposal,
                summary=str(proposal.display_summary["summary"]),
                impact=str(proposal.display_summary["impact"]),
                confirm_label=str(proposal.display_summary["confirm_label"]),
                extra_data={
                    "edit_object_type": target.object_type,
                    "target_id": target.target_id,
                },
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
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        response = result_bridge.catalog()
        return AuthoringToolOutcome(
            "Accepted result catalog read locally.",
            response.to_dict(),
            ok=response.ok,
        )

    def read_geometry_feature_catalog(
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        snapshot = session.snapshot()
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

    controller = AuthoringWorkflowController(
        current_context,
        {
            "prepare_geometry_proposal": prepare_geometry,
            "read_geometry_edit_context": read_geometry_edit_context,
            "prepare_geometry_edit": prepare_geometry_edit,
            "read_mesh_refinement_context": read_mesh_refinement_context,
            "read_model_topology_context": read_model_topology_context,
            "prepare_mesh_proposal": prepare_mesh,
            "apply_model_definition": apply_model_definition,
            "run_native_preflight": run_preflight,
            "prepare_solve_proposal": prepare_solve,
            "request_project_save": request_project_save,
            "read_deletable_objects": read_deletable_objects,
            "prepare_delete_proposal": prepare_delete,
            "read_editable_model_objects": read_editable_objects,
            "edit_model_object": edit_model_object,
            "read_accepted_result_catalog": read_catalog,
            "read_geometry_feature_catalog": read_geometry_feature_catalog,
            "query_accepted_result": query_result,
        },
    )
    controller.observe_binding(current_context())
    return controller


__all__ = [
    "AppliedPatchRecord",
    "AppliedPatchState",
    "AgentPreflightRecord",
    "AgentPreflightState",
    "AgentPreflightTaskRequest",
    "AgentMeshTaskRequest",
    "AgentSolveTaskRequest",
    "AgentAuthoringBridge",
    "BridgeReceipt",
    "SessionResultQueryPort",
    "SessionGeometryAuthoringPort",
    "authoring_context_from_snapshot",
    "create_session_authoring_workflow_controller",
]
