"""A1 GUI boundary for detached FEM Agent authoring.

The adapter reads a detached session snapshot into bounded DTOs.  The bridge
owns only those DTOs and an ``AuthoringPort``; it never stores or mutates a
``ModelSession``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Protocol

from fem.application import ModelSession, UnitContext
from fem.application.changes import SessionDelta
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
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
)
from fem_agent.definition_authoring import (
    inverse_operations_for_snapshot,
    require_non_destructive_a4_batch,
    scoped_definition_batch_from_operations,
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
    create_geometry_proposal,
    geometry_recipe_from_payload,
    plate_with_hole_geometry,
)
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_agent.incremental_authoring import (
    create_incremental_definition_patch,
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
from fem.geometry import LogicalEntityRef
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


def _bounded_count(value: object) -> int:
    try:
        return max(0, int(len(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


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
                    recipe_kind=(None if recipe is None else type(recipe).__name__),
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
    capabilities = (
        CapabilitySummary("read_authoring_context", supported, blocked_reason),
        CapabilitySummary("review_requirements", supported, blocked_reason),
        CapabilitySummary("build_agent_draft", supported, blocked_reason),
        CapabilitySummary("present_static_proposal", supported, blocked_reason),
        CapabilitySummary("draft_native_geometry", supported, blocked_reason),
        CapabilitySummary("commit_native_geometry", supported, blocked_reason),
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
                delta = self._session.apply_scoped_definition_batch(batch)
                self._project_definition_delta(delta)
            succeeded = replace(record, state=ProposalState.SUCCEEDED)
            self._records[proposal_id] = succeeded
            return succeeded
        operation = proposal.operations[0]
        parameters = operation.parameters
        recipe = geometry_recipe_from_payload(parameters["recipe"])
        snapshot = self._session.snapshot()
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
            return self._receipt(failed)
        finally:
            self._accepting_proposal_id = None
        self._records[proposal_id] = accepted
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
        receipt = authoring_bridge.register_proposal(proposal)
        data = {
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.proposal_hash,
            "state": receipt.state.value,
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
        }
        if extra_data is not None:
            data.update(dict(extra_data))
        return AuthoringToolOutcome(
            summary,
            data,
        )

    def prepare_geometry(
        _arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        requirements = controller.collected_requirements("geometry")
        metadata = envelope(controller, "geometry")
        context = current_context()
        draft = plate_with_hole_geometry(
            "实体-偏心孔板",
            width=float(requirements["plate_width"]),
            height=float(requirements["plate_height"]),
            hole_radius=float(requirements["hole_radius"]),
            hole_center=(
                float(requirements["hole_center_x"]),
                float(requirements["hole_center_y"]),
            ),
        )
        suffix = str(metadata.pop("identity_suffix"))
        proposal = create_geometry_proposal(
            proposal_id=f"proposal-{suffix}",
            context=context,
            draft=draft,
            part_function="偏心孔板",
            project_function=(
                "偏心孔板"
                if context.binding.source_kind == "blank"
                else None
            ),
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
            summary=(
                f"创建 {requirements['plate_width']} × "
                f"{requirements['plate_height']} "
                f"{requirements['length_unit']} 的偏心孔板；孔半径 "
                f"{requirements['hole_radius']}，孔心 "
                f"({requirements['hole_center_x']}, "
                f"{requirements['hole_center_y']})"
            ),
            impact="确认后创建该几何并刷新 GUI",
            confirm_label="加入模型",
        )

    def prepare_mesh(
        _arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        requirements = controller.collected_requirements("mesh")
        metadata = envelope(controller, "mesh")
        context = current_context()
        part_id = context.active_part_id
        if part_id is None:
            raise AuthoringContractError("there is no active Part to mesh")
        intent = MeshIntent(
            str(requirements["mesh_cell_shape"]),
            int(requirements["mesh_order"]),
            global_size=float(requirements["mesh_global_size"]),
            local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("edge:hole-loop"),
                    float(requirements["hole_mesh_size"]),
                    MeshSizeFalloff("target_radius", 0.0, 2.0),
                ),
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
                f"{requirements['mesh_global_size']}，孔边尺寸 "
                f"{requirements['hole_mesh_size']}"
            ),
            impact="确认后划分网格，成功时安装网格并刷新 GUI",
            confirm_label="开始划分",
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
        patch = create_incremental_definition_patch(
            patch_id=f"patch-{suffix}",
            context=current_context(),
            snapshot=session.snapshot(),
            action=arguments["action"],
            parameters=arguments["parameters"],
            **metadata,
        )
        applied = authoring_bridge.apply_automatic_patch(patch)
        return AuthoringToolOutcome(
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
            },
        )

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
        record = controller.register_project_save_proposal(
            f"proposal-{suffix}",
            context,
        )
        return AuthoringToolOutcome(
            "Project save is waiting for the local GUI control.",
            {
                "proposal_id": record.proposal_id,
                "proposal_hash": record.proposal_hash,
                "state": record.state.value,
                "proposal_view": {
                    "proposal_id": record.proposal_id,
                    "proposal_hash": record.proposal_hash,
                    "proposal_kind": "project_save",
                    "title": "保存当前自主项目",
                    "summary": "保存当前已接受的模型状态",
                    "impact": "确认后调用本地项目保存；未确认草稿不会写入",
                    "confirm_label": "保存模型",
                    "target_document_id": record.target_document_id,
                    "target_session_id": record.target_session_id,
                    "base_session_revision": record.base_session_revision,
                },
            },
        )

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
        applied = authoring_bridge.apply_automatic_patch(patch)
        return AuthoringToolOutcome(
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
