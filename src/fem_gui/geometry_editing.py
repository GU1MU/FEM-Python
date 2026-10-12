"""Document-owned interactive geometry editing and exact preview lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from typing import Callable, Literal

from fem.application import ModelSession, SessionDelta
from fem.application.face_sketch_boolean import FaceSketchBooleanResult
from fem.application.planar_boolean import StrictPlanarBooleanResult
from fem.geometry import (
    ExtrusionSourceResolutionError, FaceSketchBooleanDirection,
    FaceSketchBooleanGeometry, FaceSketchBooleanOperation, LogicalEntityRef,
    NATIVE_GEOMETRY_TYPES, SketchGeometry, SketchReferencePoint, WireGeometry,
    ResolvedFaceWorkplane,
    geometry_dimension, namespace_part_logical_id, part_id_from_logical_id,
    strip_part_logical_id,
)

from .document_tasks import DocumentTaskRouter
from .face_sketch_boolean_dialog import FaceSketchBooleanDialog, FaceSketchBooleanParameters
from .face_sketch_editor import FaceSketchBooleanFeatureRequest, FaceSupportedSketchController
from .geometry_preview import (
    GeometryPreview, build_face_sketch_boolean_committed_preview,
    build_geometry_preview, build_strict_planar_boolean_preview,
    build_strict_sketch_draft_preview, namespace_part_geometry_preview,
)
from .geometry_tasks import (
    PreparedFaceSketchWorkplane, prepare_face_sketch_workplane,
    prepare_face_boolean_preview, prepare_planar_boolean_preview,
)
from .planar_boolean import PlanarBooleanController, planar_reference_points
from .sketch_editor import SketchDraftController, SketchDraftValidationError
from .task_controller import BackgroundTaskState, TaskApplyOutcome, TaskCompletion
from .viewport_drafting import SketchDraftRenderData
from .widgets.planar_boolean_panel import PlanarBooleanPanel
from .widgets.sketch_editor_panel import SketchEditorPanel
from .widgets.wire_editor_panel import WireEditorPanel
from .wire_editor import WireDraftController, WireDraftValidationError
from .workspace import WorkspaceDocument
from .workers import TaskContext


SketchPurpose = Literal["geometry", "planar_boolean_tool", "face_sketch"]


@dataclass(frozen=True, slots=True)
class GeometryEditLaunch:
    document: WorkspaceDocument
    session: ModelSession
    session_id: str
    session_revision: int
    part_id: str | None
    part_revision: int | None
    source_recipe: object

    @classmethod
    def capture(cls, document: WorkspaceDocument) -> GeometryEditLaunch:
        session = document.session
        snapshot = session.projection_snapshot()
        part_id = snapshot.active_part_id
        return cls(
            document, session, snapshot.session_id, snapshot.session_revision,
            part_id, None if part_id is None else snapshot.part_revision(part_id),
            None if part_id is None else snapshot.part(part_id).geometry_recipe,
        )


@dataclass(slots=True)
class GeometryEditSession:
    launch: GeometryEditLaunch
    original_selection: tuple[LogicalEntityRef, ...]
    cancelled: bool = False
    committed: bool = False
    task_id: int | None = None
    continuation_queued: bool = False
    continuation: Callable[[], None] | None = None


@dataclass(slots=True)
class WireEditSession(GeometryEditSession):
    controller: WireDraftController = field(kw_only=True)
    original_recipe: object = None
    part_name: str | None = None


@dataclass(slots=True)
class SketchEditSession(GeometryEditSession):
    controller: SketchDraftController = field(kw_only=True)
    original_recipe: object = None
    part_name: str | None = None


@dataclass(frozen=True, slots=True)
class PlanarPreviewInputs:
    generation: int
    operation: str
    target_face_id: str
    tool_geometry: SketchGeometry
    tool_face_ids: tuple[str, ...]
    sketch_revision: int


@dataclass(slots=True)
class PlanarBooleanEditSession(GeometryEditSession):
    controller: PlanarBooleanController = field(kw_only=True)
    draft: SketchDraftController | None = None
    draft_visible: bool = False
    generation: int = 0
    preview_inputs: PlanarPreviewInputs | None = None
    preview_result: StrictPlanarBooleanResult | None = None


@dataclass(frozen=True, slots=True)
class FacePreviewInputs:
    generation: int
    parameters: FaceSketchBooleanParameters
    sketch_revision: int
    recipe: FaceSketchBooleanGeometry


@dataclass(slots=True)
class FaceSketchEditSession(GeometryEditSession):
    controller: FaceSupportedSketchController = field(kw_only=True)
    reference_points: tuple[SketchReferencePoint, ...] = ()
    operation: FaceSketchBooleanOperation | None = None
    draft_visible: bool = False
    dialog: FaceSketchBooleanDialog | None = None
    parameters: FaceSketchBooleanParameters | None = None
    generation: int = 0
    preview_inputs: FacePreviewInputs | None = None
    preview_result: FaceSketchBooleanResult | None = None


@dataclass(slots=True)
class SolidFaceSelectionSession(GeometryEditSession):
    operation: FaceSketchBooleanOperation = FaceSketchBooleanOperation.FUSE


@dataclass(slots=True)
class FaceWorkplaneRequest:
    launch: GeometryEditLaunch
    face_id: str
    task_id: int | None = None


@dataclass(frozen=True, slots=True)
class FaceBooleanDisplay:
    target: GeometryPreview
    result: FaceSketchBooleanResult
    inputs: FacePreviewInputs
    workplane: ResolvedFaceWorkplane


@dataclass(frozen=True, slots=True)
class GeometryEditPresentation:
    """Only the shared selection, scene and layout operations editors need."""

    active_document: Callable[[], WorkspaceDocument | None]
    selected_references: Callable[[], tuple[LogicalEntityRef, ...]]
    set_selection: Callable[[tuple[LogicalEntityRef, ...]], None]
    set_selection_mode: Callable[[str], None]
    project_delta: Callable[[WorkspaceDocument, SessionDelta, GeometryPreview | None], bool]
    restore_scene: Callable[[WorkspaceDocument], None]
    source_preview: Callable[[GeometryEditLaunch], GeometryPreview]
    show_preview: Callable[[GeometryPreview], None]
    show_reference: Callable[[GeometryPreview, str | None, str | None], None]
    show_face_preview: Callable[[FaceBooleanDisplay], None]
    draft_render_data: Callable[[], SketchDraftRenderData | None]
    enter_wire: Callable[[float | None], None]
    enter_sketch: Callable[[SketchPurpose, float | None], None]
    exit_editor: Callable[[], None]
    begin_face_prompt: Callable[[str], None]
    end_face_prompt: Callable[[], None]
    face_selection_ready: Callable[[bool], None]
    create_face_dialog: Callable[[], FaceSketchBooleanDialog]
    feature_requested: Callable[[FaceSketchBooleanFeatureRequest], None]
    confirm_discard: Callable[[str], bool]
    commit_allowed: Callable[[], bool]
    geometry_committed: Callable[[SessionDelta, str, int], None]
    set_status: Callable[[str, int], None]
    show_error: Callable[[str, str], None]
    refresh_actions: Callable[[], None]
    defer: Callable[[Callable[[], None]], None]
    shutting_down: Callable[[], bool]


class GeometryEditCoordinator:
    def __init__(
        self, tasks: DocumentTaskRouter, presentation: GeometryEditPresentation,
        wire_panel: WireEditorPanel, sketch_panel: SketchEditorPanel,
        planar_panel: PlanarBooleanPanel,
    ) -> None:
        self.tasks = tasks
        self.presentation = presentation
        self.wire_panel = wire_panel
        self.sketch_panel = sketch_panel
        self.planar_panel = planar_panel
        self._edit: GeometryEditSession | None = None
        self._face_request: FaceWorkplaneRequest | None = None
        self._face_cache: tuple[tuple[int, str, str, int, str], PreparedFaceSketchWorkplane | str] | None = None
        self._face_launch_queued = False
        self._lifecycle_generation = 0

    @property
    def edit_session(self) -> GeometryEditSession | None:
        return self._edit

    @property
    def active(self) -> bool:
        return self._edit is not None

    @property
    def wire_active(self) -> bool:
        return isinstance(self._edit, WireEditSession)

    @property
    def sketch_active(self) -> bool:
        state = self._edit
        return isinstance(state, SketchEditSession) or (
            isinstance(state, (PlanarBooleanEditSession, FaceSketchEditSession))
            and state.draft_visible
        )

    @property
    def boolean_active(self) -> bool:
        return isinstance(self._edit, (PlanarBooleanEditSession, FaceSketchEditSession, SolidFaceSelectionSession))

    @property
    def selecting_target(self) -> bool:
        state = self._edit
        return isinstance(state, SolidFaceSelectionSession) or (
            isinstance(state, PlanarBooleanEditSession)
            and state.controller.selecting_target
        )

    def _current(self, state: GeometryEditSession) -> bool:
        return self._edit is state and not state.cancelled and not state.committed and self._launch_current(state.launch)

    def _launch_current(self, launch: GeometryEditLaunch) -> bool:
        if (
            self.presentation.shutting_down()
            or not self.tasks.is_active(launch.document)
            or launch.document.session is not launch.session
        ):
            return False
        snapshot = launch.session.projection_snapshot()
        if snapshot.session_id != launch.session_id or snapshot.session_revision != launch.session_revision:
            return False
        if snapshot.active_part_id != launch.part_id:
            return False
        if launch.part_id is None:
            return True
        try:
            return (
                snapshot.part_revision(launch.part_id) == launch.part_revision
                and snapshot.part(launch.part_id).geometry_recipe == launch.source_recipe
            )
        except KeyError:
            return False

    def _new_launch(self) -> GeometryEditLaunch | None:
        document = self.presentation.active_document()
        if self._edit is not None or document is None or document.task_controller.busy or self.presentation.shutting_down():
            return None
        self._lifecycle_generation += 1
        return GeometryEditLaunch.capture(document)

    def _continue_after_idle(self, state: GeometryEditSession, callback: Callable[[], None]) -> None:
        state.continuation = callback
        if state.continuation_queued:
            return
        state.continuation_queued = True

        def continued() -> None:
            state.continuation_queued = False
            continuation = state.continuation
            state.continuation = None
            if self._current(state) and continuation is not None:
                continuation()

        self.tasks.after_idle(
            state.launch.document.task_controller,
            lambda: self.presentation.defer(continued),
        )

    def _start(
        self, state: GeometryEditSession, workload: Callable[[TaskContext], object], name: str,
        apply: Callable[[object], TaskApplyOutcome], success: Callable[[object], None],
        failure: Callable[[str], None], terminal: Callable[[TaskCompletion], None] | None = None,
    ) -> bool:
        document = state.launch.document
        started = self.tasks.start(
            workload, success, name, failure, context=document,
            controller=document.task_controller, task_name=name,
            apply_result=apply, on_terminal=terminal,
        )
        if started:
            state.task_id = document.task_controller.current_task_id
        return started

    def begin_wire(
        self, root: WireGeometry | None, *, original_recipe: object,
        part_name: str | None = None, display_size: float | None = None,
    ) -> None:
        launch = self._new_launch()
        if launch is None:
            return
        controller = WireDraftController(root=root) if root is not None else WireDraftController(name="Wire-1")
        state = WireEditSession(
            launch, self.presentation.selected_references(), controller=controller,
            original_recipe=original_recipe, part_name=part_name,
        )
        self._edit = state
        self.wire_panel.set_controller(controller, base_snapshot=controller.snapshot())
        self.presentation.enter_wire(display_size)
        self.presentation.set_status("Wire editor: add points and connect members", 0)
        self.presentation.refresh_actions()

    def begin_sketch(
        self, root: SketchGeometry | None, *, original_recipe: object,
        part_name: str | None = None, display_size: float | None = None,
    ) -> None:
        launch = self._new_launch()
        if launch is None:
            return
        controller = SketchDraftController(root=root) if root is not None else SketchDraftController(name="Sketch-1")
        self._edit = SketchEditSession(
            launch, self.presentation.selected_references(), controller=controller,
            original_recipe=original_recipe, part_name=part_name,
        )
        self.sketch_panel.set_controller(controller, base_snapshot=controller.snapshot())
        self.presentation.enter_sketch("geometry", display_size)
        self.presentation.set_status("2D sketch editor: draw closed contours", 0)
        self.presentation.refresh_actions()

    def _replace_root(self, recipe: object, root: object) -> object:
        from fem.geometry import BooleanGeometry, ExtrudedGeometry, MovedGeometry, PathSweptGeometry, RevolvedGeometry, RotatedGeometry
        if isinstance(recipe, (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry, PathSweptGeometry)):
            return replace(recipe, base=self._replace_root(recipe.base, root))
        if isinstance(recipe, BooleanGeometry):
            return replace(recipe, object_geometry=self._replace_root(recipe.object_geometry, root))
        return root

    def finish_wire(self) -> None:
        state = self._edit
        if not isinstance(state, WireEditSession):
            return
        try:
            root = state.controller.to_geometry()
            self._commit_draft(state, root)
        except (WireDraftValidationError, RuntimeError, TypeError, ValueError, KeyError) as error:
            self.wire_panel.show_status(f"geometry.part.edit-rejected: {error}")
            return

    def finish_sketch(self) -> None:
        state = self._edit
        if not self.sketch_active or state.cancelled:
            return
        controller = self.sketch_panel.controller
        try:
            root = controller.to_sketch_geometry()
            if not self._current(state):
                raise RuntimeError("Project changed; sketch not applied. Cancel and restart")
            if isinstance(state, FaceSketchEditSession):
                self.sketch_panel.end()
                state.draft_visible = False
                self.presentation.exit_editor()
                self._open_face_dialog(state, root)
            elif isinstance(state, PlanarBooleanEditSession):
                snapshot = controller.snapshot()
                state.controller.set_tool_recipe(
                    root, external_references=snapshot.external_references,
                    external_coincidences=snapshot.external_coincidences,
                    unresolved_reference_ids=snapshot.unresolved_reference_ids,
                )
                self.sketch_panel.end()
                state.draft_visible = False
                self.presentation.exit_editor()
                self.refresh_planar_preview(auto_commit=True)
            else:
                self._commit_draft(state, root)
        except (SketchDraftValidationError, ExtrusionSourceResolutionError, RuntimeError, TypeError, ValueError, KeyError) as error:
            self.sketch_panel.show_status(f"geometry.part.edit-rejected: {error}")

    def _commit_draft(self, state: WireEditSession | SketchEditSession, root: object) -> None:
        if not self._current(state):
            raise RuntimeError("Project changed; geometry not applied. Cancel and restart")
        if not self.presentation.commit_allowed() or not self._current(state):
            return
        launch = state.launch
        recipe = root if state.original_recipe is None else self._replace_root(state.original_recipe, root)
        preview = None
        if isinstance(state, SketchEditSession) and state.original_recipe is None:
            rendered = self.presentation.draft_render_data()
            local = (
                GeometryPreview(
                    rendered.points, rendered.faces, rendered.curves,
                    tuple(f"face:{value}" for value in rendered.face_ids),
                    tuple(f"edge:{value}" for value in rendered.curve_ids),
                    tuple(None if value is None else f"point:{value}" for value in rendered.point_ids),
                    "body:domain", 2,
                ) if rendered is not None and rendered.geometry_revision == state.controller.snapshot().revision and rendered.faces
                else build_strict_sketch_draft_preview(root, analysis=state.controller.derive_profiles())
            )
            preview = namespace_part_geometry_preview(launch.session.next_native_part_id, local)
        if state.original_recipe is None:
            delta = launch.session.add_native_part(recipe, name=state.part_name, expected_session_revision=launch.session_revision)
        else:
            if launch.part_id is None:
                raise RuntimeError("No current part")
            delta = launch.session.replace_part_geometry(
                launch.part_id, recipe, expected_part_revision=launch.part_revision,
                expected_session_revision=launch.session_revision,
            )
        if self._project_committed(state, delta, preview):
            self.presentation.set_status(
                "Wire created; select truss or beam elements"
                if isinstance(state, WireEditSession) else "2D sketch created", 6000,
            )

    def confirm_discard(self) -> bool:
        state = self._edit
        if state is None or state.cancelled or state.committed:
            return True
        if isinstance(state, WireEditSession) and state.controller.dirty:
            return self.presentation.confirm_discard("Wire")
        draft = None
        if isinstance(state, SketchEditSession):
            draft = state.controller
        elif isinstance(state, PlanarBooleanEditSession) and state.draft_visible:
            draft = state.draft
        elif isinstance(state, FaceSketchEditSession) and state.draft_visible:
            draft = state.controller.draft
        if draft is not None and draft.dirty:
            return self.presentation.confirm_discard("2D Sketch")
        return True

    def cancel(self, *, confirm: bool = True) -> None:
        state = self._edit
        if state is None or state.cancelled or (confirm and not self.confirm_discard()):
            return
        state.cancelled = True
        state.continuation = None
        self._close_views(state)
        controller = state.launch.document.task_controller
        if state.task_id is not None and controller.current_task_id == state.task_id:
            controller.request_cancel()

        generation = self._lifecycle_generation

        def restore() -> None:
            if self._lifecycle_generation != generation or self._edit is not None or self.presentation.shutting_down():
                return
            if self.tasks.is_active(state.launch.document):
                self.presentation.restore_scene(state.launch.document)
                self.presentation.set_selection(state.original_selection)
                self.presentation.set_status("Geometry editing canceled", 4000)
            self.presentation.refresh_actions()

        def cleanup(*, defer_restore: bool = True) -> None:
            if self._edit is not state:
                return
            self._edit = None
            if defer_restore and not self.presentation.shutting_down():
                self.presentation.defer(restore)

        if controller.busy:
            self.tasks.after_idle(controller, cleanup)
        else:
            cleanup(defer_restore=False)
            restore()

    def dispose_document(self, document: WorkspaceDocument) -> None:
        """Release hidden/noninteractive editors independently of UI continuations."""
        state = self._edit
        if state is not None and state.launch.document is document:
            if document.task_controller.busy:
                self.cancel(confirm=False)
            else:
                self._lifecycle_generation += 1
                self._release(state)
        if self._face_cache is not None and self._face_cache[0][0] == document.document_id:
            self._face_cache = None

    def _close_views(self, state: GeometryEditSession) -> None:
        if isinstance(state, WireEditSession):
            self.wire_panel.end()
        elif isinstance(state, SketchEditSession):
            self.sketch_panel.end()
        elif isinstance(state, PlanarBooleanEditSession):
            if state.draft_visible:
                self.sketch_panel.end()
            self.planar_panel.end()
        elif isinstance(state, FaceSketchEditSession):
            if state.draft_visible:
                self.sketch_panel.end()
            if state.dialog is not None:
                state.dialog.close_for_workflow()
                state.dialog.deleteLater()
                state.dialog = None
        self.presentation.end_face_prompt()
        self.presentation.exit_editor()

    def _release(self, state: GeometryEditSession) -> None:
        self._close_views(state)
        if self._edit is state:
            self._edit = None
        self.presentation.refresh_actions()

    def _project_committed(self, state: GeometryEditSession, delta: SessionDelta, preview: GeometryPreview | None) -> bool:
        """An accepted Session transaction is complete even if rendering fails."""
        state.committed = True
        projected = False
        try:
            projected = self.presentation.project_delta(state.launch.document, delta, preview)
        except Exception:
            logging.exception("Geometry committed but its display projection failed")
        finally:
            self._release(state)
        if not projected:
            try:
                self.presentation.restore_scene(state.launch.document)
                return True
            except Exception:
                logging.exception("Failed to rebuild the committed geometry display")
                self.presentation.set_status("Geometry committed, but UI refresh failed", 8000)
                return False
        return True

    def begin_boolean(self, operation: str) -> None:
        launch = self._new_launch()
        if launch is None or not isinstance(launch.source_recipe, NATIVE_GEOMETRY_TYPES):
            return
        dimension = geometry_dimension(launch.source_recipe)
        selection = self.presentation.selected_references()
        if dimension == 3:
            self._edit = SolidFaceSelectionSession(launch, selection, operation=FaceSketchBooleanOperation(operation))
        elif dimension == 2:
            initial = None
            if len(selection) == 1 and selection[0].kind == "face":
                try:
                    initial = self._local_reference(launch, selection[0]).logical_id
                except ValueError:
                    pass
            controller = PlanarBooleanController(launch.source_recipe, launch.session_revision, operation, target_face_id=initial)
            controller.request_target_selection()
            self._edit = PlanarBooleanEditSession(launch, selection, controller=controller)
            self.planar_panel.begin(controller)
            self.planar_panel.hide()
        else:
            return
        self.presentation.set_selection_mode("face")
        self.presentation.begin_face_prompt(operation)
        self.presentation.set_status("Select a target face", 0)
        if dimension == 3:
            self.refresh_face_prompt()
        else:
            self.presentation.face_selection_ready(initial is not None)
            if initial is not None:
                self.presentation.set_selection(selection)
        self.presentation.refresh_actions()

    @staticmethod
    def _local_reference(launch: GeometryEditLaunch, reference: LogicalEntityRef) -> LogicalEntityRef:
        owner = part_id_from_logical_id(reference.logical_id)
        if owner is None:
            return reference
        if owner != launch.part_id:
            raise ValueError("Select a face on the target part")
        return LogicalEntityRef(strip_part_logical_id(owner, reference.logical_id))

    def accept_reference(self, reference: LogicalEntityRef) -> bool:
        state = self._edit
        if not self.selecting_target:
            return False
        if state.cancelled:
            return True
        try:
            if reference.kind != "face":
                raise ValueError("Select a planar solid face")
            if isinstance(state, PlanarBooleanEditSession):
                if not self._current(state):
                    raise ValueError("Project changed; reopen the 2D boolean")
                state.controller.assign_reference(self._local_reference(state.launch, reference))
                state.controller.request_target_selection()
                self._invalidate_planar(state)
                self.presentation.set_selection((reference,))
                self.presentation.face_selection_ready(True)
                self.presentation.set_status("Target face selected; click OK to continue", 0)
            else:
                launch = state.launch
                if not self.tasks.is_active(launch.document) or launch.document.session is not launch.session:
                    raise ValueError("Target document changed")
                owner = part_id_from_logical_id(reference.logical_id)
                if owner is not None and owner != launch.session.projection_snapshot().active_part_id:
                    delta = launch.session.set_active_native_part(owner, expected_session_revision=launch.session.session_revision)
                    self.presentation.project_delta(launch.document, delta, None)
                    state.launch = GeometryEditLaunch.capture(launch.document)
                self.presentation.set_selection((reference,))
                self.retry_face_selection()
                self.refresh_face_prompt()
                self.presentation.refresh_actions()
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            self.presentation.face_selection_ready(False)
            self.presentation.set_status(str(error), 5000)
        return True

    def accept_box_selection(self, references: tuple[LogicalEntityRef, ...]) -> bool:
        if not self.selecting_target:
            return False
        if len(references) != 1 or references[0].kind != "face":
            self.presentation.set_status("Boolean operation allows only one target face at a time", 4000)
            return True
        return self.accept_reference(references[0])

    def confirm_target(self) -> None:
        state = self._edit
        if isinstance(state, PlanarBooleanEditSession):
            try:
                state.controller.confirm_target_selection()
            except ValueError as error:
                self.presentation.set_status(str(error), 3000)
                return
            self.presentation.end_face_prompt()
            self.edit_planar_tool()
        elif isinstance(state, SolidFaceSelectionSession):
            if not self.face_selection_valid():
                self.refresh_face_prompt()
                return
            self.presentation.end_face_prompt()
            self.begin_face_sketch()

    def request_planar_target(self) -> None:
        state = self._edit
        if not isinstance(state, PlanarBooleanEditSession) or state.cancelled:
            return
        state.controller.request_target_selection()
        self._invalidate_planar(state)
        self.presentation.begin_face_prompt(state.controller.operation)
        self.presentation.face_selection_ready(state.controller.target_face_id is not None)

    def clear_planar_target(self) -> None:
        state = self._edit
        if not isinstance(state, PlanarBooleanEditSession) or state.cancelled:
            return
        state.controller.clear_target()
        self._invalidate_planar(state)
        self.presentation.restore_scene(state.launch.document)
        self.presentation.set_selection(())
        self.presentation.set_selection_mode("face")
        self.presentation.face_selection_ready(False)
        self.planar_panel.refresh()

    def _invalidate_planar(self, state: PlanarBooleanEditSession) -> None:
        state.generation += 1
        state.preview_inputs = None
        state.preview_result = None
        self.planar_panel.set_preview_valid(False)

    def edit_planar_tool(self) -> None:
        state = self._edit
        if not isinstance(state, PlanarBooleanEditSession) or not self._current(state) or state.draft_visible:
            return
        controller = state.controller
        if controller.target_face_id is None or controller.selecting_target:
            self.request_planar_target()
            return
        self._invalidate_planar(state)
        if state.launch.document.task_controller.busy:
            self._continue_after_idle(state, self.edit_planar_tool)
            return
        if state.draft is None:
            if controller.tool_geometry is None:
                draft = SketchDraftController(name="Boolean Tool Sketch")
            else:
                draft = SketchDraftController(snapshot=replace(
                    SketchDraftController.snapshot_from_geometry(controller.tool_geometry),
                    external_references=controller.external_references,
                    external_coincidences=controller.external_coincidences,
                    unresolved_reference_ids=controller.unresolved_reference_ids,
                ))
            state.draft = draft
        draft = state.draft
        state.draft_visible = True
        preview = self.presentation.source_preview(state.launch)
        self.sketch_panel.set_controller(draft, base_snapshot=draft.snapshot())
        self.sketch_panel.set_reference_points(planar_reference_points(preview, controller.target_face_id, plane=draft.plane))
        self.presentation.enter_sketch("planar_boolean_tool", None)
        faces = tuple(face for face, identity in zip(preview.faces, preview.face_logical_ids, strict=True) if identity == controller.target_face_id)
        self.presentation.show_reference(
            GeometryPreview(
                preview.points, faces or preview.faces, (),
                (controller.target_face_id,) * len(faces or preview.faces),
            ), None, None,
        )
        self.presentation.set_status("Draw closed contours on the target XY plane", 0)
        self.presentation.refresh_actions()

    def delete_planar_tool(self) -> None:
        state = self._edit
        if not isinstance(state, PlanarBooleanEditSession) or state.cancelled:
            return
        state.controller.clear_tool()
        state.draft = None
        self._invalidate_planar(state)
        self.presentation.restore_scene(state.launch.document)
        self.planar_panel.refresh()

    def change_planar_operation(self, operation: str) -> None:
        state = self._edit
        if not isinstance(state, PlanarBooleanEditSession) or state.cancelled:
            return
        state.controller.set_operation(operation)
        self._invalidate_planar(state)
        if state.controller.ready:
            self.refresh_planar_preview()

    def _planar_inputs_current(self, state: PlanarBooleanEditSession, inputs: PlanarPreviewInputs) -> bool:
        controller = state.controller
        return (
            self._current(state) and state.generation == inputs.generation
            and controller.operation == inputs.operation and controller.target_face_id == inputs.target_face_id
            and controller.tool_geometry == inputs.tool_geometry and controller.tool_face_ids == inputs.tool_face_ids
            and (state.draft is None or state.draft.snapshot().revision == inputs.sketch_revision)
        )

    def refresh_planar_preview(self, *, auto_commit: bool = False) -> None:
        state = self._edit
        if not isinstance(state, PlanarBooleanEditSession) or not self._current(state) or not state.controller.ready:
            return
        if state.launch.document.task_controller.busy:
            self._continue_after_idle(state, lambda: self.refresh_planar_preview(auto_commit=auto_commit))
            return
        self._invalidate_planar(state)
        controller = state.controller
        inputs = PlanarPreviewInputs(
            state.generation, controller.operation, controller.target_face_id,
            controller.tool_geometry, controller.tool_face_ids,
            0 if state.draft is None else state.draft.snapshot().revision,
        )
        self.planar_panel.set_preview_running(True)

        def apply(value: object) -> TaskApplyOutcome:
            if type(value) is not StrictPlanarBooleanResult or not self._planar_inputs_current(state, inputs):
                return TaskApplyOutcome.stale("2D boolean preview is stale and was not applied")
            state.preview_result = value
            state.preview_inputs = inputs
            return TaskApplyOutcome.accepted(value)

        def succeeded(value: StrictPlanarBooleanResult) -> None:
            self.planar_panel.set_preview_running(False)
            self.planar_panel.set_preview_valid(True)
            if auto_commit:
                self.finish_planar_boolean()
            else:
                self.presentation.show_preview(namespace_part_geometry_preview(
                    state.launch.part_id,
                    build_strict_planar_boolean_preview(value.geometry, value.preview),
                ))

        def failed(message: str) -> None:
            if self._edit is state and not state.cancelled and state.generation == inputs.generation:
                self.planar_panel.set_preview_running(False)
                self.planar_panel.set_preview_valid(False)
                self.planar_panel.show_status(f"OCC planar boolean or topology validation failed; committed geometry is unchanged:\n{message}")
                if auto_commit:
                    self._continue_after_idle(state, self.edit_planar_tool)

        def terminal(record: TaskCompletion) -> None:
            if self._edit is state and not state.cancelled:
                self.planar_panel.set_preview_running(False)
                if (
                    state.generation == inputs.generation and auto_commit
                    and record.state in {BackgroundTaskState.CANCELLED, BackgroundTaskState.DISCARDED}
                ):
                    self._continue_after_idle(state, self.edit_planar_tool)

        if not self._start(
            state, lambda context: prepare_planar_boolean_preview(
                state.launch.source_recipe, inputs.operation, inputs.target_face_id,
                inputs.tool_geometry, inputs.tool_face_ids, context,
            ), "2D Boolean Preview", apply, succeeded, failed, terminal,
        ):
            self.planar_panel.set_preview_running(False)

    def finish_planar_boolean(self) -> None:
        state = self._edit
        if not isinstance(state, PlanarBooleanEditSession) or state.cancelled:
            return
        result, inputs = state.preview_result, state.preview_inputs
        if result is None or inputs is None:
            self.refresh_planar_preview()
            return
        if not self._planar_inputs_current(state, inputs):
            self._invalidate_planar(state)
            self.planar_panel.show_status("State changed; rebuild the preview")
            return
        if not self.presentation.commit_allowed() or not self._planar_inputs_current(state, inputs):
            self._continue_after_idle(state, self.edit_planar_tool)
            return
        launch = state.launch
        try:
            preview = namespace_part_geometry_preview(launch.part_id, build_strict_planar_boolean_preview(result.geometry, result.preview))
            prior_regions = len(launch.session.projection_snapshot().named_regions)
            delta = launch.session.replace_part_geometry(
                launch.part_id, result.geometry, expected_part_revision=launch.part_revision,
                expected_session_revision=launch.session_revision, authenticate_geometry=False,
            )
            displayed = self._project_committed(state, delta, preview)
            if displayed:
                self.presentation.geometry_committed(delta, "2D Boolean Result", prior_regions)
        except (RuntimeError, TypeError, ValueError, KeyError) as error:
            self.planar_panel.show_status(f"Commit rejected: {error}")
            self._continue_after_idle(state, self.edit_planar_tool)
            return
        references = tuple(
            LogicalEntityRef(namespace_part_logical_id(launch.part_id, entity.logical_id))
            for entity in result.proof.result_entities if entity.kind == "face"
        )
        self.presentation.set_selection(references)

    def _face_key(self) -> tuple[int, str, str, int, str] | None:
        document = self.presentation.active_document()
        selection = self.presentation.selected_references()
        if (
            document is None
            or (self._edit is not None and not isinstance(self._edit, SolidFaceSelectionSession))
            or len(selection) != 1 or selection[0].kind != "face"
        ):
            return None
        snapshot = document.session.projection_snapshot()
        launch = GeometryEditLaunch.capture(document)
        if snapshot.source_kind != "native" or launch.part_id is None:
            return None
        try:
            face_id = self._local_reference(launch, selection[0]).logical_id
            part = snapshot.part(launch.part_id)
            if part.suppressed or part.geometry_recipe is None or geometry_dimension(part.geometry_recipe) != 3:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        return document.document_id, launch.session_id, launch.part_id, launch.part_revision, face_id

    def face_selection_valid(self) -> bool:
        key = self._face_key()
        if self._face_cache is not None and self._face_cache[0] != key:
            self._face_cache = None
        if key is None or self.presentation.shutting_down():
            return False
        if self._face_cache is not None:
            return isinstance(self._face_cache[1], PreparedFaceSketchWorkplane)
        document = self.presentation.active_document()
        if document.task_controller.busy or self._face_request is not None:
            return False
        if not self._face_launch_queued:
            self._face_launch_queued = True
            self.presentation.defer(lambda: self._prepare_face_selection(self._face_key()))
        return False

    def retry_face_selection(self) -> None:
        if self._face_cache is not None and self._face_cache[0] == self._face_key() and isinstance(self._face_cache[1], str):
            self._face_cache = None

    def _prepare_face_selection(self, key: tuple[int, str, str, int, str] | None) -> None:
        self._face_launch_queued = False
        document = self.presentation.active_document()
        if key is None or document is None or self._face_key() != key or document.task_controller.busy or self.presentation.shutting_down():
            return
        request = FaceWorkplaneRequest(GeometryEditLaunch.capture(document), key[-1])
        self._face_request = request
        if isinstance(self._edit, SolidFaceSelectionSession):
            # The selected Part is authoritative when the guided face is checked.
            self._edit.launch = request.launch

        def apply(value: object) -> TaskApplyOutcome:
            if (
                self._face_request is not request or self._face_key() != key
                or self.presentation.shutting_down() or not self.tasks.is_active(document)
                or document.session is not request.launch.session
                or document.session.projection_snapshot().part(request.launch.part_id).geometry_recipe != request.launch.source_recipe
            ):
                return TaskApplyOutcome.stale("Workplane result is stale; sketch not opened")
            if type(value) is not PreparedFaceSketchWorkplane:
                raise TypeError("workplane preparation returned invalid data")
            self._face_cache = (key, value)
            return TaskApplyOutcome.accepted(value)

        def failed(message: str) -> None:
            if self._face_key() == key:
                self._face_cache = (key, message)

        def terminal(record: TaskCompletion) -> None:
            if record.state is BackgroundTaskState.CANCELLED:
                failed("Workplane check canceled; select the face again")
            if self._face_request is request:
                self._face_request = None

        if self.tasks.start(
            lambda context: prepare_face_sketch_workplane(request.launch.source_recipe, request.face_id, context),
            lambda value: self.refresh_face_prompt(), "Create Sketch on Face", failed,
            context=document, controller=document.task_controller,
            task_name="Resolve Face Sketch Workplane", apply_result=apply, on_terminal=terminal,
        ):
            request.task_id = document.task_controller.current_task_id
            if isinstance(self._edit, SolidFaceSelectionSession):
                self._edit.task_id = request.task_id
        else:
            self._face_request = None
            failed("Another background task is running")

    def refresh_face_prompt(self) -> None:
        if not isinstance(self._edit, SolidFaceSelectionSession) or self._edit.cancelled or self.presentation.shutting_down():
            return
        valid = self.face_selection_valid()
        self.presentation.face_selection_ready(valid)
        failure = None if self._face_cache is None else self._face_cache[1]
        self.presentation.set_status(
            "Target face selected; confirm to continue" if valid else
            f"Invalid planar face; select again: {failure}" if isinstance(failure, str) else
            "Resolving sketch workplane..." if self._face_request is not None or self._face_launch_queued else "Select a target face",
            4000 if isinstance(failure, str) else 0,
        )

    def begin_face_sketch(self) -> None:
        previous = self._edit
        if previous is not None and not isinstance(previous, SolidFaceSelectionSession):
            return
        if not self.face_selection_valid():
            return
        document = self.presentation.active_document()
        if document.task_controller.busy:
            return
        launch = GeometryEditLaunch.capture(document)
        prepared = self._face_cache[1]
        try:
            controller = FaceSupportedSketchController(
                document.session.projection_snapshot(), launch.part_id,
                prepared.workplane, reference_points=prepared.reference_points,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            self.presentation.show_error("Create Sketch on Face", str(error))
            return
        state = FaceSketchEditSession(
            launch, previous.original_selection if previous is not None else self.presentation.selected_references(),
            controller=controller, reference_points=prepared.reference_points,
            operation=None if previous is None else previous.operation,
        )
        self._lifecycle_generation += 1
        self._edit = state
        self._enter_face_sketch(state)

    def _enter_face_sketch(self, state: FaceSketchEditSession) -> None:
        if not self._current(state):
            return
        state.draft_visible = True
        draft = state.controller.draft
        self.sketch_panel.set_controller(draft, base_snapshot=draft.snapshot())
        self.sketch_panel.set_reference_points(state.reference_points, refresh_controller=False)
        self.presentation.enter_sketch("face_sketch", None)
        workplane = state.controller.workplane
        self.presentation.show_reference(self.presentation.source_preview(state.launch), workplane.support_face_id, workplane.target_body_id)
        self.presentation.set_status("Face sketch editor: draw contours, then create", 0)
        self.presentation.refresh_actions()

    def _open_face_dialog(self, state: FaceSketchEditSession, sketch: SketchGeometry) -> None:
        material_ids = tuple(profile.id for profile in state.controller.draft.derive_profiles().profiles if profile.is_material)
        previous = state.parameters
        selected = material_ids if previous is None else tuple(identity for identity in previous.participating_profile_ids if identity in material_ids)
        dialog = self.presentation.create_face_dialog()
        dialog.set_profiles(material_ids, selected_ids=selected or material_ids)
        defaults = dialog.parameters()
        if defaults is not None and (previous is not None or state.operation is not None):
            dialog.set_parameters(FaceSketchBooleanParameters(
                state.operation if state.operation is not None else previous.operation,
                previous.direction if previous is not None else
                FaceSketchBooleanDirection.INWARD
                if state.operation is FaceSketchBooleanOperation.CUT else defaults.direction,
                previous.distance if previous is not None else defaults.distance,
                selected or material_ids,
            ))
        if state.operation is not None:
            dialog.fix_operation(state.operation)
        state.dialog = dialog
        dialog.parametersChanged.connect(self.change_face_parameters)
        dialog.createFeatureRequested.connect(self.request_face_feature)
        dialog.returnSketchRequested.connect(self.return_to_face_sketch)
        dialog.cancelRequested.connect(lambda: self.cancel(confirm=False))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self.change_face_parameters(dialog.parameters())
        self.presentation.refresh_actions()

    def change_face_parameters(self, payload: object) -> None:
        state = self._edit
        if not isinstance(state, FaceSketchEditSession) or state.dialog is None or state.cancelled:
            return
        state.generation += 1
        state.preview_inputs = None
        state.preview_result = None
        state.parameters = payload if type(payload) is FaceSketchBooleanParameters else None
        state.dialog.set_preview_running(state.generation)
        if state.parameters is None:
            state.dialog.set_preview_invalid(state.generation, state.dialog.validation_reason())
            return
        self.refresh_face_preview()

    def _face_inputs_current(self, state: FaceSketchEditSession, inputs: FacePreviewInputs) -> bool:
        return (
            self._current(state) and state.dialog is not None and not state.draft_visible
            and state.generation == inputs.generation and state.parameters == inputs.parameters
            and state.controller.sketch_snapshot().revision == inputs.sketch_revision
            and state.controller.launch_is_current(state.launch.session.projection_snapshot())
            and state.controller.draft.to_sketch_geometry() == inputs.recipe.sketch
            and state.controller.sketch_snapshot().external_references == inputs.recipe.external_references
            and state.controller.sketch_snapshot().external_coincidences == inputs.recipe.external_coincidences
        )

    def refresh_face_preview(self) -> None:
        state = self._edit
        if (
            not isinstance(state, FaceSketchEditSession) or not self._current(state)
            or state.dialog is None or state.parameters is None or state.draft_visible
        ):
            return
        if state.preview_result is not None:
            return
        if state.launch.document.task_controller.busy:
            self._continue_after_idle(state, self.refresh_face_preview)
            return
        controller, parameters = state.controller, state.parameters
        launch, sketch_snapshot = controller.launch_snapshot, controller.sketch_snapshot()
        feature_ids = {str(record.payload.get("feature_id")) for record in launch.part.feature_history if record.payload.get("feature_id")}
        index = 1
        while f"FSB{index}" in feature_ids:
            index += 1
        kind = "face_sketch_boolean_fuse" if parameters.operation is FaceSketchBooleanOperation.FUSE else "face_sketch_boolean_cut"
        operation_index = 1 + sum(record.kind == kind for record in launch.part.feature_history)
        try:
            recipe = FaceSketchBooleanGeometry(
                launch.part.geometry_recipe, f"FSB{index}",
                f"Extrude Fuse-{operation_index}" if parameters.operation is FaceSketchBooleanOperation.FUSE else f"Extrude Cut-{operation_index}",
                launch.workplane.support_face_id, launch.workplane.strategy,
                controller.draft.to_sketch_geometry(), parameters.operation, parameters.direction,
                parameters.distance, parameters.participating_profile_ids,
                sketch_snapshot.external_references, sketch_snapshot.external_coincidences,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            state.dialog.set_preview_invalid(state.generation, str(error))
            return
        inputs = FacePreviewInputs(state.generation, parameters, sketch_snapshot.revision, recipe)
        state.dialog.set_preview_running(inputs.generation)

        def apply(value: object) -> TaskApplyOutcome:
            if (
                type(value) is not FaceSketchBooleanResult
                or not self._face_inputs_current(state, inputs)
                or replace(value.geometry, step_proofs=()) != inputs.recipe
            ):
                return TaskApplyOutcome.stale("Extrude boolean preview is stale and was not applied")
            state.preview_inputs, state.preview_result = inputs, value
            return TaskApplyOutcome.accepted(value)

        def succeeded(value: FaceSketchBooleanResult) -> None:
            self.presentation.show_face_preview(FaceBooleanDisplay(
                self.presentation.source_preview(state.launch), value,
                inputs, controller.workplane,
            ))
            state.dialog.set_preview_valid(inputs.generation)

        def failed(message: str) -> None:
            if self._current(state) and state.dialog is not None and state.generation == inputs.generation:
                state.dialog.set_preview_invalid(inputs.generation, f"Exact preview failed: {message}")

        def terminal(record: TaskCompletion) -> None:
            if self._current(state) and state.dialog is not None:
                if state.generation != inputs.generation:
                    self._continue_after_idle(state, self.refresh_face_preview)
                elif record.state is BackgroundTaskState.CANCELLED:
                    state.dialog.set_preview_invalid(inputs.generation, "Exact preview canceled; change parameters to retry")

        if not self._start(
            state, lambda context: prepare_face_boolean_preview(recipe, inputs.generation, context),
            "Exact Extrude Boolean Preview", apply, succeeded, failed, terminal,
        ):
            state.dialog.set_preview_invalid(inputs.generation, "Another background task is running; change parameters to retry")

    def request_face_feature(self, parameters: object, generation: int) -> None:
        state = self._edit
        if (
            not isinstance(state, FaceSketchEditSession)
            or state.preview_result is None or state.preview_inputs is None
            or parameters != state.parameters or generation != state.generation
        ):
            return
        if not self._face_inputs_current(state, state.preview_inputs) or not state.dialog.preview_is_valid:
            return
        self.presentation.feature_requested(FaceSketchBooleanFeatureRequest(
            state.controller.launch_snapshot, state.preview_result.geometry,
            state.controller.sketch_snapshot().revision, generation,
        ))

    def commit_face_feature(self, payload: object) -> None:
        state = self._edit
        if (
            not isinstance(state, FaceSketchEditSession)
            or type(payload) is not FaceSketchBooleanFeatureRequest
            or state.preview_result is None or state.preview_inputs is None
        ):
            return
        inputs, result, controller = state.preview_inputs, state.preview_result, state.controller
        launch = controller.launch_snapshot
        if (
            not self._face_inputs_current(state, inputs) or not state.dialog.preview_is_valid
            or payload.launch != launch or payload.geometry != result.geometry
            or replace(payload.geometry, step_proofs=()) != inputs.recipe
            or payload.sketch_revision != inputs.sketch_revision
            or payload.preview_generation != inputs.generation
        ):
            if state.dialog is not None:
                state.dialog.set_preview_invalid(state.generation, "State changed before commit; exact preview is stale. Regenerate the preview")
            return
        if not self.presentation.commit_allowed() or not self._face_inputs_current(state, inputs):
            return
        try:
            preview = namespace_part_geometry_preview(launch.part_id, build_face_sketch_boolean_committed_preview(payload.geometry, result.preview))
            delta = state.launch.session.commit_face_sketch_boolean(
                launch.part_id, launch.body_id, payload.geometry,
                expected_session_id=launch.session_id, expected_part_revision=launch.part_revision,
                expected_session_revision=launch.session_revision, expected_body_recipe=launch.body.recipe,
                expected_support_face_id=launch.workplane.support_face_id,
                expected_workplane_strategy=launch.workplane.strategy,
                expected_sketch_revision=inputs.sketch_revision, sketch_revision=controller.sketch_snapshot().revision,
                expected_preview_generation=inputs.generation, preview_generation=state.generation,
            )
            displayed = self._project_committed(state, delta, preview)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            state.dialog.set_preview_invalid(state.generation, f"Commit rejected: {error}")
            return
        name = state.launch.session.projection_snapshot().part(launch.part_id).feature_history[-1].name
        self.presentation.set_selection(())
        if displayed:
            self.presentation.set_status(f"{name} committed as a single undo unit", 6000)

    def return_to_face_sketch(self, payload: object) -> None:
        state = self._edit
        if not isinstance(state, FaceSketchEditSession) or type(payload) is not FaceSketchBooleanParameters or state.cancelled:
            return
        state.parameters = payload
        state.generation += 1
        state.preview_result = None
        state.preview_inputs = None
        if state.dialog is not None:
            state.dialog.close_for_workflow()
            state.dialog.deleteLater()
            state.dialog = None
        self._enter_face_sketch(state)
