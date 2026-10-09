"""Read-only GUI adapter for accepted results and workspace result selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from fem.application import ModelSession, SessionSnapshot, UnitContext
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
    compare_result_scalars,
)
from fem_agent.workspace_catalog import WorkspaceDocumentIdentity

from .workspace import FEMWorkspace, WorkspaceDocument


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
                raise _WorkspaceResultResolutionError("target session is unavailable")
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
                materialization_generation=snapshot.result_generations.get(run.run_id),
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
        if provider.source != source or provider.snapshot.generation != generation:
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
                variable=AgentResultVariable(item.descriptor.field_id.variable.value),
                position=item.descriptor.field_id.position.value,
                components=item.descriptor.columns,
                unit=_result_unit(
                    units,
                    AgentResultVariable(item.descriptor.field_id.variable.value),
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
            (
                uses_displayed_result
                and session.projection_snapshot().displayed_result_run_id
                != target_run_id
            )
            or session.result_identity_for(target_run_id)
            != (
                source,
                generation,
            )
            or not self._target_is_current(resolved)
        ):
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
        if provider.source != source or provider.snapshot.generation != generation:
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
            target.session.result_identity_for(reference.expected_source.run_id)
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
                or generation != reference.expected_materialization_generation
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
            target.session.result_identity_for(reference.expected_source.run_id)
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
    projection: SessionSnapshot,
    source: ResultSourceKey,
    run_id: str,
) -> bool:
    artifact = getattr(projection, "artifact", None)
    return bool(
        run_id == projection.displayed_result_run_id
        and artifact is not None
        and str(getattr(artifact, "artifact_id", "")) == source.artifact_id
        and int(getattr(projection, "model_revision", -1)) == source.model_revision
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
    projection: SessionSnapshot,
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
            *(
                region.name
                for region in regions
                if region.entity_kind in {"node", "edge", "face"}
            ),
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
            () if region == "all_nodes" else provider.named_region_node_ids(region)
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
        () if region == "all_elements" else provider.named_region_element_ids(region)
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


__all__ = ["SessionResultQueryPort"]
