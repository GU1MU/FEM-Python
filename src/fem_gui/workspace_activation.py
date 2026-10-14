"""Coordinate activation of a registered document and its shared presentation."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
import logging
from typing import Callable, Literal

from fem.application import MeshEntityRef, ModelSession, SessionSnapshot
from fem.application.results import ResultProvider, ScalarFieldSelection
from fem.geometry import LogicalEntityRef

from .inspection_service import InspectionService
from .result_display_state import ResultDisplayPreferences
from .task_controller import BackgroundTaskController
from .visualization.model_adapter import ArchiveModelView, ModelGeometry
from .visualization.selection import SelectionContextState, SelectionState
from .visualization.symbols import SymbolSettings
from .workspace import FEMWorkspace, WorkspaceDocument


@dataclass(frozen=True, slots=True)
class WorkspaceSceneControls:
    """Small control values read when rebuilding model and symbol scenes."""

    edges: bool
    nodes: bool
    node_labels: bool
    element_labels: bool
    symbols: bool


@dataclass(frozen=True, slots=True)
class WorkspacePresentationCheckpoint:
    """Shallow adapters and small visible values needed to recover a switch."""

    session: ModelSession
    document: SessionSnapshot
    task_controller: BackgroundTaskController
    visible_document_id: int | None
    applied_revision: int
    module_name: str
    scene_module_name: Literal["Geometry", "Model", "Results"]
    window_title: str
    result_navigation_visible: bool
    scene_controls: WorkspaceSceneControls
    geometry: ModelGeometry | None
    inspection_service: InspectionService | None
    result_provider: ResultProvider | None
    result_selection: ScalarFieldSelection | None
    step_name: str | None
    legacy_project_extension: bool
    result_archive_geometry: ModelGeometry | None
    result_archive_model_view: ArchiveModelView | None
    model_edges_visible: bool
    symbol_settings: SymbolSettings
    selection_context: SelectionContextState
    geometry_selection_mode: str
    selection: SelectionState
    geometry_selection: frozenset[LogicalEntityRef]
    mesh_selection: frozenset[MeshEntityRef]
    blank_result_preferences: ResultDisplayPreferences


@dataclass(frozen=True, slots=True)
class WorkspaceActivationPresentation:
    """Explicit window and scene operations used by the activation transaction."""

    visible_document: Callable[[], WorkspaceDocument | None]
    editor_active: Callable[[], bool]
    capture_window: Callable[[], WorkspacePresentationCheckpoint]
    capture_document: Callable[[WorkspaceDocument], None]
    bind_agent: Callable[[WorkspaceDocument, bool], None]
    restore_agent: Callable[[WorkspaceDocument | None, bool], None]
    scene_updates: Callable[[], AbstractContextManager[None]]
    clear_transients: Callable[[], None]
    prepare_document: Callable[[WorkspaceDocument], None]
    install_document: Callable[[WorkspaceDocument], None]
    restore_window: Callable[[WorkspaceDocument | None, WorkspacePresentationCheckpoint], None]
    restore_scene: Callable[[WorkspaceDocument | None, WorkspacePresentationCheckpoint], None]
    synchronize_active: Callable[[WorkspaceDocument | None], None]
    render: Callable[[], None]


class WorkspaceActivationCoordinator:
    """Own activation order while Session and the registry retain document state."""

    def __init__(
        self,
        workspace: FEMWorkspace,
        presentation: WorkspaceActivationPresentation,
    ) -> None:
        self.workspace = workspace
        self.presentation = presentation

    def _is_live(self, document: WorkspaceDocument) -> bool:
        try:
            return self.workspace.document(document.document_id) is document
        except (KeyError, TypeError, ValueError):
            return False

    def activate(
        self,
        document: WorkspaceDocument,
        *,
        from_agent_tool: bool = False,
    ) -> bool:
        if not self._is_live(document):
            return False
        view = self.presentation
        previous = view.visible_document()
        if previous is not None and not self._is_live(previous):
            previous = None
        if previous is document and self.workspace.active_document() is document:
            try:
                view.synchronize_active(document)
                return True
            except Exception:
                logging.exception("failed to refresh active workspace controls")
                return False
        if view.editor_active():
            return False

        checkpoint = None
        previous_state = None
        agent_binding_started = False
        scene_started = False
        try:
            # Preserve the original visible window without compatibility getters:
            # registering a first document may already select it in the registry.
            checkpoint = view.capture_window()
            if previous is not None:
                previous_state = replace(
                    previous.presentation_state,
                    result_display=previous.presentation_state.result_display.copy(),
                )
                view.capture_document(previous)
                previous_state = replace(
                    previous.presentation_state,
                    result_display=previous.presentation_state.result_display.copy(),
                )
            target_preferences = document.presentation_state.result_display.copy()
            target_rendering = replace(document.presentation_cache.result_rendering)
            agent_binding_started = True
            view.bind_agent(document, from_agent_tool)
            with view.scene_updates():
                scene_started = True
                view.clear_transients()
                self.workspace.activate(document)
                view.prepare_document(document)
                view.install_document(document)
            # Rendering and task/root presentation are part of successful activation.
            view.synchronize_active(document)
            view.render()
            return True
        except Exception:
            logging.exception("workspace document activation failed")

        if previous is not None and previous_state is not None and self._is_live(previous):
            previous.presentation_state = previous_state
        if agent_binding_started:
            document.presentation_state.result_display = target_preferences
            document.presentation_cache.result_rendering = target_rendering
        # Roll back to the actual former visible document, never the failed target
        # automatically selected by registration in an initially empty workspace.
        if previous is not None and self._is_live(previous):
            try:
                self.workspace.activate(previous)
            except Exception:
                logging.exception("failed to reactivate previous workspace document")
                previous = None
                self.workspace.deactivate()
        else:
            previous = None
            self.workspace.deactivate()
        if checkpoint is None:
            return False
        if scene_started:
            try:
                view.restore_window(previous, checkpoint)
            except Exception:
                logging.exception("failed to restore previous window controls")
        if agent_binding_started:
            try:
                view.restore_agent(previous, from_agent_tool)
            except Exception:
                logging.exception("failed to restore Agent document binding")
        if scene_started:
            try:
                with view.scene_updates():
                    view.restore_scene(previous, checkpoint)
            except Exception:
                logging.exception("failed to restore previous viewport after activation failure")
        try:
            view.synchronize_active(previous)
        except Exception:
            logging.exception("failed to restore active workspace controls")
        if scene_started:
            try:
                view.render()
            except Exception:
                logging.exception("failed to repaint previous viewport after activation failure")
        return False
