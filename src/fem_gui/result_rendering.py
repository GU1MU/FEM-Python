"""Prepare document result rendering and revision-neutral averaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from fem.application.results import (
    FieldPosition, FieldRequest, FieldState, NodalAveragingPolicy,
    ResultFieldId, ResultMaterializationPatch, ResultProvider, ResultVariable,
    ScalarFieldSelection, build_result_field_topology_template,
    prepare_result_export_snapshot, project_scalar_field_topology,
    project_scalar_field_topology_from_template,
)

from .document_tasks import DocumentTaskRouter, session_task_outcome
from .result_display_state import ResultAveragingRequest, ResultRenderingCache
from .task_controller import BackgroundTaskState, TaskApplyOutcome, TaskCompletion
from .visualization.result_renderer import ResultRenderPayload, build_result_render_payload
from .workers import TaskContext
from .workspace import WorkspaceDocument


@dataclass(frozen=True, slots=True)
class ResultPresentationPort:
    current_provider: Callable[[WorkspaceDocument], ResultProvider | None]
    current_selection: Callable[[WorkspaceDocument], ScalarFieldSelection | None]
    current_payload: Callable[[WorkspaceDocument], ResultRenderPayload | None]
    refresh_display: Callable[[WorkspaceDocument], None]
    can_present: Callable[[WorkspaceDocument], bool]
    defer: Callable[[Callable[[], None]], None]


class ResultPresentationService:
    """Own rendering preparation and caches without owning the viewport."""

    def __init__(self, tasks: DocumentTaskRouter, presentation: ResultPresentationPort) -> None:
        self.tasks = tasks
        self.presentation = presentation

    @staticmethod
    def _cache(document: WorkspaceDocument, provider: ResultProvider) -> ResultRenderingCache:
        cache = document.presentation_cache.result_rendering
        if (cache.source, cache.generation) != (provider.source, provider.snapshot.generation):
            cache.clear()
            cache.source = provider.source
            cache.generation = provider.snapshot.generation
        return cache

    @staticmethod
    def _availability(provider: ResultProvider, selection: ScalarFieldSelection):
        availability = provider.field_status(selection.field_key)
        if selection.component not in availability.descriptor.columns:
            raise ValueError("selection component is outside the field descriptor")
        return availability

    def deformation_scale(
        self,
        document: WorkspaceDocument,
        provider: ResultProvider,
        *,
        shape_mode: str | None = None,
        scale_mode: str | None = None,
        scale_value: float | None = None,
    ) -> float:
        shape = document.presentation_state.result_display.display.shape_mode if shape_mode is None else shape_mode
        if shape != "deformed":
            return 0.0
        mode = document.presentation_state.result_display.scale_mode if scale_mode is None else scale_mode
        requested_value = (
            float(document.presentation_state.result_display.scale_value if scale_value is None else scale_value)
            if mode == "custom"
            else 0.0
        )
        cache = self._cache(document, provider).deformation_scale
        if (
            cache is not None
            and cache[0] is provider.snapshot
            and cache[1] == shape
            and cache[2] == mode
            and cache[3] == requested_value
        ):
            return cache[4]
        if mode == "real":
            scale = 1.0
        elif mode == "custom":
            scale = requested_value
            if not np.isfinite(scale) or scale < 0.0:
                raise ValueError(
                    "custom deformation scale must be finite and non-negative"
                )
        elif mode != "auto":
            raise ValueError("unknown deformation scale mode")
        else:
            topology = provider.snapshot.topology
            coordinates = getattr(topology, "_node_coordinates", None)
            if coordinates is None:
                coordinates = topology.node_coordinates
            displacements = getattr(topology, "_nodal_displacements", None)
            if displacements is None:
                displacements = topology.nodal_displacements
            if len(coordinates) == 0:
                scale = 1.0
            else:
                span = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
                maximum = float(
                    np.max(np.linalg.norm(displacements, axis=1))
                )
                scale = (
                    1.0
                    if maximum <= 0.0 or span <= 0.0
                    else 0.1 * span / maximum
                )
        self._cache(document, provider).deformation_scale = (
            provider.snapshot,
            shape,
            mode,
            requested_value,
            scale,
        )
        return scale

    def build_render_payload(
        self,
        document: WorkspaceDocument,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
        *,
        shape_mode: str | None = None,
        scale_mode: str | None = None,
        scale_value: float | None = None,
    ) -> ResultRenderPayload:
        availability = self._availability(
            provider,
            selection,
        )
        if availability.state is not FieldState.READY:
            raise KeyError("only a READY catalog field can be rendered")
        export = prepare_result_export_snapshot(
            provider.snapshot,
            selection,
        )
        deformation_scale = self.deformation_scale(
            document, provider,
            shape_mode=shape_mode,
            scale_mode=scale_mode,
            scale_value=scale_value,
        )
        current_payload = self.presentation.current_payload(document)
        if current_payload is not None:
            current_topology = current_payload.topology
            if (
                current_topology.source == export.source
                and current_topology.materialization_generation
                == export.materialization_generation
                and current_topology.selection == export.selection
                and current_topology.deformation_scale == deformation_scale
            ):
                return current_payload
        cache = self._cache(document, provider).topology_template
        if (
            cache is not None
            and cache[0] is provider.snapshot
            and cache[1].matches(export, deformation_scale)
        ):
            topology = project_scalar_field_topology_from_template(
                export,
                cache[1],
                deformation_scale,
            )
        else:
            topology = project_scalar_field_topology(
                export,
                deformation_scale=deformation_scale,
            )
            self._cache(document, provider).topology_template = (
                provider.snapshot,
                build_result_field_topology_template(
                    topology,
                    export.field,
                ),
            )
        return build_result_render_payload(
            topology,
            reusable=self.presentation.current_payload(document),
        )

    def averaging_visual_selection(
        self,
        document: WorkspaceDocument,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> ScalarFieldSelection:
        request = selection.field_key.request
        field_id = request.field_id
        if (
            field_id.variable is not ResultVariable.S
            or field_id.position not in {
                FieldPosition.ELEMENT_NODAL,
                FieldPosition.RESOLVED_NODAL,
            }
        ):
            return selection
        visual_request = FieldRequest(
            field_id=ResultFieldId(
                ResultVariable.S,
                FieldPosition.RESOLVED_NODAL,
            ),
            averaging_policy=NodalAveragingPolicy(
                threshold_percent=float(
                    document.presentation_state.result_display.contour_options["averaging_threshold"]
                )
            ),
            gauss_order=request.gauss_order,
        )
        return ScalarFieldSelection(
            provider.resolve_request(visual_request),
            selection.component,
        )

    def visualization_provider(
        self,
        document: WorkspaceDocument,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> tuple[ResultProvider, ScalarFieldSelection]:
        visual_selection = self.averaging_visual_selection(
            document, provider,
            selection,
        )
        if visual_selection == selection:
            return provider, selection
        if (
            provider.field_status(visual_selection.field_key).state
            is FieldState.READY
        ):
            return provider, visual_selection
        cached = self._cache(document, provider).visualization
        if (
            cached is None
            or cached[0] != visual_selection.field_key
        ):
            return provider, selection
        if (
            cached[1].source != provider.source
            or cached[1].snapshot.generation
            != provider.snapshot.generation
        ):
            return provider, selection
        return (
            cached[1],
            visual_selection,
        )

    def apply_averaging(
        self, document: WorkspaceDocument, provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> None:
        if not self.tasks.is_live(document) or not self.presentation.can_present(document):
            return
        visual_selection = self.averaging_visual_selection(document, provider, selection)
        _render_provider, render_selection = self.visualization_provider(document, provider, selection)
        if visual_selection == selection or render_selection == visual_selection:
            self.presentation.refresh_display(document)
            return
        if document.task_controller.busy:
            cache = document.presentation_cache.result_rendering
            desired = ResultAveragingRequest(provider.source, provider.snapshot.generation, selection, visual_selection)
            if cache.averaging_request == desired:
                cache.averaging_refresh_requested = False
            else:
                self._schedule_refresh(document)
            self.tasks.presentation.set_status(
                "Result task running; nodal averaging will be applied when complete", 4000,
            )
            return
        self._begin_averaging(document, provider, selection, visual_selection)

    def _schedule_refresh(self, document: WorkspaceDocument) -> None:
        cache = document.presentation_cache.result_rendering
        cache.averaging_refresh_requested = True
        if cache.averaging_refresh_scheduled:
            return
        cache.averaging_refresh_scheduled = True
        session = document.session

        def refresh() -> None:
            current_cache = document.presentation_cache.result_rendering
            current_cache.averaging_refresh_scheduled = False
            requested = current_cache.averaging_refresh_requested
            current_cache.averaging_refresh_requested = False
            if (
                not requested or not self.tasks.is_live(document) or document.session is not session
                or not self.presentation.can_present(document)
            ):
                return
            provider = self.presentation.current_provider(document)
            selection = self.presentation.current_selection(document)
            if provider is not None and type(selection) is ScalarFieldSelection:
                self.apply_averaging(document, provider, selection)

        self.tasks.after_idle(document.task_controller, lambda: self.presentation.defer(refresh))

    def _begin_averaging(
        self, document: WorkspaceDocument, provider: ResultProvider,
        selection: ScalarFieldSelection, visual_selection: ScalarFieldSelection,
    ) -> None:
        session = document.session
        task = None
        title = "Failed to compute averaged nodal stress contours"
        request = ResultAveragingRequest(provider.source, provider.snapshot.generation, selection, visual_selection)

        def failed(message: str) -> None:
            session.accept_task_failed(task.token, message)
            if self.presentation.can_present(document):
                self.tasks.presentation.show_error(title, message)

        def cancelled() -> None:
            session.accept_task_cancelled(task.token)

        def terminal(record: TaskCompletion) -> None:
            cache = document.presentation_cache.result_rendering
            if cache.averaging_request is request:
                cache.averaging_request = None
            if record.state is BackgroundTaskState.CANCELLED:
                cache.averaging_refresh_requested = False
            if (
                record.state is BackgroundTaskState.SUCCEEDED
                and self.tasks.is_live(document) and document.session is session
                and session.current_result_identity() == (provider.source, provider.snapshot.generation)
            ):
                current_selection = self.presentation.current_selection(document)
                if type(current_selection) is ScalarFieldSelection and (
                    current_selection != selection
                    or self.averaging_visual_selection(document, provider, current_selection) != visual_selection
                ):
                    self._schedule_refresh(document)

        try:
            task = session.prepare_result_projection(provider.source.run_id)
            if (
                session.current_result_identity() != (provider.source, provider.snapshot.generation)
                or task.token.result_id != provider.source.result_id
                or dict(task.token.dependency_revisions).get("materialization_generation") != provider.snapshot.generation
            ):
                raise RuntimeError("result changed before averaging visualization started")
            document.presentation_cache.result_rendering.averaging_request = request

            def workload(context: TaskContext) -> ResultMaterializationPatch:
                context.report("Computing averaged nodal stress contours...")
                return provider.materialize((visual_selection.field_key,), cancellation=context)

            def apply_result(value: object) -> TaskApplyOutcome:
                if type(value) is not ResultMaterializationPatch:
                    raise TypeError("averaging visualization must return ResultMaterializationPatch")
                return session_task_outcome(session.accept_result_projection(task.token), value)

            def succeeded(value: object) -> None:
                delta, patch = value
                if not delta.accepted or delta.changed or delta.invalidated:
                    raise RuntimeError("averaging visualization receipt was not accepted")
                visual_provider = provider.apply(patch)
                if not any(field.key == visual_selection.field_key for field in visual_provider.snapshot.fields):
                    raise RuntimeError("averaging visualization field was not materialized")
                current_selection = self.presentation.current_selection(document)
                if (
                    session.current_result_identity() != (provider.source, provider.snapshot.generation)
                    or current_selection != selection
                    or self.averaging_visual_selection(document, provider, current_selection) != visual_selection
                ):
                    return
                self._cache(document, provider).visualization = (visual_selection.field_key, visual_provider)
                if self.presentation.can_present(document):
                    self.presentation.refresh_display(document)
                    self.tasks.presentation.set_status("Averaged nodal stress contours updated", 4000)

            started = self.tasks.start(
                workload, succeeded, title, failed,
                context=document, controller=document.task_controller,
                task_name="Averaged Nodal Stress Contours",
                on_cancelled=cancelled, on_inactive_cancelled=cancelled,
                on_inactive_failure=failed, on_inactive_success=succeeded,
                apply_result=apply_result, on_terminal=terminal,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            if document.presentation_cache.result_rendering.averaging_request is request:
                document.presentation_cache.result_rendering.averaging_request = None
            if task is not None:
                session.accept_task_failed(task.token, error)
            if self.presentation.can_present(document):
                self.tasks.presentation.show_error(title, str(error))
            return
        if not started:
            if document.presentation_cache.result_rendering.averaging_request is request:
                document.presentation_cache.result_rendering.averaging_request = None
            cancelled()
