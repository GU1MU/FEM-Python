"""Own native actors for transient sketch and wire displays; never render."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import logging
import math
from typing import Any

import numpy as np

from fem.geometry import SketchCircle, SketchLine, SketchPlane, SketchReferencePoint
from ..geometry_preview import GeometryPreview
from ..viewport_drafting import (
    SketchDraftRenderData,
    WireDraftRenderData,
    sketch_axis_local_endpoints,
    sketch_coordinate_label,
    sketch_curve_sample_count,
    sketch_grid_polydata,
    sketch_shape_preview_points,
    sketch_snap_label,
    wire_coordinate_label,
    wire_grid_layout,
    wire_grid_polydata,
)


@dataclass(frozen=True, slots=True)
class DraftBackend:
    """Native operations needed only for draft actor installation."""

    pyvista: Any
    add_mesh: Callable[..., Any]
    add_point_labels: Callable[..., Any]
    remove_actor: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class SketchDisplaySettings:
    spacing: float
    display_size: float | None
    grid_visible: bool
    show_work_plane_axes: bool
    show_profile_fill: bool
    show_point_ids: bool
    show_external_labels: bool


def _sketch_geometry_color(constraint_status: str) -> str:
    return {
        "under_constrained": "#1976a8",
        "fully_constrained": "#2e7d32",
        "redundant": "#ef6c00",
        "conflicting": "#c62828",
        "failed": "#c62828",
    }[constraint_status]


def _sketch_constraint_label_options(*, warning: bool) -> dict[str, object]:
    return {
        "font_size": 12,
        "bold": True,
        "shape": "rounded_rect",
        "shape_color": "#ffebee" if warning else "#fff8e1",
        "shape_opacity": 0.96,
        "margin": 5,
        "text_color": "#b71c1c" if warning else "#263238",
        "always_visible": True,
    }


def _sketch_fixed_constraint_label_options(*, warning: bool) -> dict[str, object]:
    """Return a compact style that leaves the constrained point unobscured."""

    return {
        "font_size": 9,
        "bold": False,
        "shape": None,
        "margin": 1,
        "text_color": "#b71c1c" if warning else "#455a64",
        "always_visible": True,
    }


class SketchScene:
    """Own sketch display actors and their curve-sampling stamp."""

    def __init__(self) -> None:
        self._backend: DraftBackend | None = None
        self._actors: dict[str, Any] = {}
        self._last_curve_sampling: tuple[SketchDraftRenderData, float] | None = None

    def bind_backend(self, backend: DraftBackend) -> None:
        if self._backend == backend:
            return
        self.detach()
        self._backend = backend

    def clear(self) -> None:
        for name in tuple(self._actors):
            self._remove_actor(name)
        self.invalidate_sampling()

    def detach(self) -> None:
        self.clear()
        self._backend = None

    def _remove_actor(self, name: str) -> None:
        actor = self._actors.pop(name, None)
        if actor is not None and self._backend is not None:
            try:
                self._backend.remove_actor(actor, reset_camera=False, render=False)
            except Exception:
                pass
        if name == "sketch_draft_curves":
            self.invalidate_sampling()

    def _add_mesh(self, *args, **kwargs):
        actor = self._backend.add_mesh(*args, render=False, **kwargs)
        actor.SetPickable(False)
        return actor

    def _add_point_labels(self, *args, **kwargs):
        actor = self._backend.add_point_labels(
            *args, show_points=False, render=False, **kwargs,
        )
        actor.SetPickable(False)
        return actor

    def invalidate_sampling(self) -> None:
        self._last_curve_sampling = None

    def clear_draft(self) -> None:
        for name in (
            "sketch_work_plane_grid",
            "sketch_work_plane_axis_0",
            "sketch_work_plane_axis_1",
            "sketch_work_plane_origin",
            "sketch_work_plane_axis_labels",
            "sketch_draft_faces",
            "sketch_draft_curves",
            "sketch_draft_points",
            "sketch_draft_point_labels",
            "sketch_reference_points",
            "sketch_reference_labels",
            "sketch_authoring_selection",
            "sketch_authoring_shape_preview",
            "sketch_authoring_hover_outline",
            "sketch_authoring_hover",
            "sketch_authoring_hover_label",
            "sketch_entity_hover",
            "sketch_constraint_selection_points",
            "sketch_constraint_selection_curves",
            "sketch_constraint_overlays",
            "sketch_fixed_constraint_overlays",
        ):
            self._remove_actor(name)

    def install(
        self, data: SketchDraftRenderData, settings: SketchDisplaySettings,
        reference_points: tuple[SketchReferencePoint, ...],
        support_points: tuple[tuple[float, float, float], ...],
    ) -> None:
        self.clear_draft()
        if self._backend is None:
            return
        pyvista = self._backend.pyvista
        coordinates = np.asarray(data.points, dtype=float).reshape((-1, 3))
        extent_points = (
            tuple(data.points)
            + support_points
            + tuple(point.position for point in reference_points)
        )
        geometry_color = _sketch_geometry_color(data.constraint_status)
        grid, grid_layout = sketch_grid_polydata(
            pyvista,
            data.plane,
            extent_points,
            settings.spacing,
            settings.display_size or 10.0,
        )
        if settings.grid_visible:
            self._actors["sketch_work_plane_grid"] = self._add_mesh(
                grid,
                color="#7f8f9f",
                opacity=0.35,
                line_width=1,
                name="sketch_work_plane_grid",
                reset_camera=False,
                pickable=False,
            )
        origin = np.asarray(data.plane.origin, dtype=float)
        label_points = [origin]
        axis_labels = ["O"]
        for index in range(2) if settings.show_work_plane_axes else ():
            local_start, local_end = sketch_axis_local_endpoints(
                grid_layout,
                index,
            )
            start = np.asarray(
                data.plane.to_global(*local_start),
                dtype=float,
            )
            end = np.asarray(
                data.plane.to_global(*local_end),
                dtype=float,
            )
            axis_line = pyvista.PolyData()
            axis_line.points = np.asarray((start, end), dtype=float)
            axis_line.lines = np.asarray((2, 0, 1), dtype=np.int64)
            actor_name = f"sketch_work_plane_axis_{index}"
            self._actors[actor_name] = self._add_mesh(
                axis_line,
                color=("#d9534f", "#45a049")[index],
                line_width=4,
                name=actor_name,
                reset_camera=False,
                pickable=False,
            )
            label_points.append(end)
            axis_labels.append(("U", "V")[index])
        if settings.show_work_plane_axes:
            self._actors["sketch_work_plane_origin"] = self._add_mesh(
                pyvista.PolyData(np.asarray((origin,), dtype=float)),
                color="#ff8c00",
                point_size=13,
                render_points_as_spheres=True,
                name="sketch_work_plane_origin",
                reset_camera=False,
                pickable=False,
            )
            self._actors["sketch_work_plane_axis_labels"] = (
                self._add_point_labels(
                    np.asarray(label_points, dtype=float),
                    axis_labels,
                    point_size=0,
                    font_size=11,
                    shape=None,
                    text_color="#ff8c00",
                    name="sketch_work_plane_axis_labels",
                    reset_camera=False,
                )
            )
        if data.faces and settings.show_profile_fill:
            surface = pyvista.PolyData()
            surface.points = coordinates
            surface.faces = np.hstack(
                tuple((len(face), *face) for face in data.faces)
            ).astype(np.int64)
            self._actors["sketch_draft_faces"] = self._add_mesh(
                surface,
                color="#80b9d8",
                opacity=0.28,
                show_edges=False,
                name="sketch_draft_faces",
                reset_camera=False,
                pickable=False,
            )
        if data.curves:
            curves = pyvista.PolyData()
            curves.points = coordinates
            curves.lines = np.hstack(
                tuple((len(curve), *curve) for curve in data.curves)
            ).astype(np.int64)
            self._actors["sketch_draft_curves"] = self._add_mesh(
                curves,
                color=geometry_color,
                line_width=3,
                name="sketch_draft_curves",
                reset_camera=False,
                pickable=False,
            )
        authoring_points = tuple(
            index
            for index, point_id in enumerate(data.point_ids)
            if point_id is not None
        )
        if authoring_points:
            point_coordinates = coordinates[np.asarray(authoring_points)]
            self._actors["sketch_draft_points"] = self._add_mesh(
                pyvista.PolyData(point_coordinates),
                color=geometry_color,
                point_size=10,
                render_points_as_spheres=True,
                name="sketch_draft_points",
                reset_camera=False,
                pickable=False,
            )
            if settings.show_point_ids:
                self._actors["sketch_draft_point_labels"] = (
                    self._add_point_labels(
                        point_coordinates,
                        tuple(str(index + 1) for index in range(len(authoring_points))),
                        point_size=0,
                        font_size=9,
                        shape=None,
                        text_color=geometry_color,
                        name="sketch_draft_point_labels",
                        reset_camera=False,
                    )
                )
        if reference_points:
            reference_coordinates = np.asarray(
                tuple(point.position for point in reference_points),
                dtype=float,
            )
            self._actors["sketch_reference_points"] = self._add_mesh(
                pyvista.PolyData(reference_coordinates),
                color="#d32f2f",
                point_size=13,
                render_points_as_spheres=True,
                name="sketch_reference_points",
                reset_camera=False,
                pickable=False,
            )
            labels = {
                "topology_vertex": "Vertex",
                "line_midpoint": "Midpoint",
                "circle_center": "Circle center",
                "arc_center": "Arc center",
                "face_center": "Face center",
            }
            if settings.show_external_labels:
                self._actors["sketch_reference_labels"] = (
                    self._add_point_labels(
                        reference_coordinates,
                        [
                            labels[point.derived_type.value]
                            for point in reference_points
                        ],
                        point_size=0,
                        font_size=9,
                        shape=None,
                        text_color="#d32f2f",
                        name="sketch_reference_labels",
                        reset_camera=False,
                    )
                )
        overlay_groups = (
            (
                "sketch_constraint_overlays",
                tuple(item for item in data.constraint_overlays if item.kind != "fixed"),
                _sketch_constraint_label_options,
            ),
            (
                "sketch_fixed_constraint_overlays",
                tuple(item for item in data.constraint_overlays if item.kind == "fixed"),
                _sketch_fixed_constraint_label_options,
            ),
        )
        for actor_name, overlays, style_options in overlay_groups:
            if not overlays:
                continue
            self._actors[actor_name] = self._add_point_labels(
                np.asarray(tuple(item.position for item in overlays)),
                tuple(item.text for item in overlays),
                point_size=0,
                name=actor_name,
                reset_camera=False,
                **style_options(warning=any(item.warning for item in overlays)),
            )
        selection = self._selection_polydata(data, coordinates)
        if selection is not None:
            self._actors["sketch_authoring_selection"] = self._add_mesh(
                selection,
                color="#ffb300",
                line_width=8,
                point_size=14,
                render_points_as_spheres=True,
                render_lines_as_tubes=True,
                opacity=1.0,
                name="sketch_authoring_selection",
                reset_camera=False,
                pickable=False,
            )

    def show_selection(self, data: SketchDraftRenderData | None) -> None:
        self._remove_actor("sketch_authoring_selection")
        if data is None or self._backend is None:
            return
        pyvista = self._backend.pyvista
        coordinates = np.asarray(data.points, dtype=float).reshape((-1, 3))
        selection = self._selection_polydata(data, coordinates)
        if selection is not None:
            self._actors["sketch_authoring_selection"] = self._add_mesh(
                selection,
                color="#ffb300",
                line_width=8,
                point_size=14,
                render_points_as_spheres=True,
                render_lines_as_tubes=True,
                opacity=1.0,
                name="sketch_authoring_selection",
                reset_camera=False,
                pickable=False,
            )

    def show_entity_hover(
        self, data: SketchDraftRenderData | None, hover: tuple[str, str] | None,
    ) -> None:
        self._remove_actor("sketch_entity_hover")
        if data is None or hover is None or self._backend is None:
            return
        pyvista = self._backend.pyvista
        kind, entity_id = hover
        hover_data = replace(
            data,
            selected_kind=kind,
            selected_id=entity_id,
            selected_ids=(entity_id,),
        )
        coordinates = np.asarray(data.points, dtype=float).reshape((-1, 3))
        geometry = self._selection_polydata(hover_data, coordinates)
        if geometry is not None:
            self._actors["sketch_entity_hover"] = self._add_mesh(
                geometry,
                color="#ffd54f",
                line_width=6,
                point_size=12,
                render_points_as_spheres=True,
                render_lines_as_tubes=True,
                opacity=0.95,
                name="sketch_entity_hover",
                reset_camera=False,
                pickable=False,
            )

    def show_constraint_selection(
        self, data: SketchDraftRenderData | None, targets: tuple[tuple[str, str], ...],
    ) -> None:
        actor_names = {
            "point": "sketch_constraint_selection_points",
            "curve": "sketch_constraint_selection_curves",
        }
        for actor_name in actor_names.values():
            self._remove_actor(actor_name)
        if data is None or not targets or self._backend is None:
            return
        pyvista = self._backend.pyvista
        coordinates = np.asarray(data.points, dtype=float).reshape((-1, 3))
        for kind, actor_name in actor_names.items():
            ids = tuple(
                entity_id
                for entity_kind, entity_id in targets
                if entity_kind == kind
            )
            if not ids:
                continue
            selection_data = replace(
                data,
                selected_kind=kind,
                selected_id=ids[0],
                selected_ids=ids,
            )
            geometry = self._selection_polydata(
                selection_data,
                coordinates,
            )
            if geometry is None:
                continue
            self._actors[actor_name] = self._add_mesh(
                geometry,
                color="#ffb300",
                line_width=8,
                point_size=14,
                render_points_as_spheres=True,
                render_lines_as_tubes=True,
                opacity=1.0,
                name=actor_name,
                reset_camera=False,
                pickable=False,
            )

    def refresh_curve_sampling(
        self, data: SketchDraftRenderData | None, world_per_pixel: float | None,
    ) -> bool:
        if (
            data is None or not data.analytic_curves or world_per_pixel is None
            or self._backend is None
        ):
            return False
        pyvista = self._backend.pyvista
        previous_sampling = self._last_curve_sampling
        if (
            previous_sampling is not None
            and previous_sampling[0] is data
            and "sketch_draft_curves" in self._actors
            and abs(world_per_pixel - previous_sampling[1])
            <= 0.02 * max(previous_sampling[1], 1.0e-12)
        ):
            return False
        point_map = {point.id: point for point in data.analytic_points}
        coordinates: list[tuple[float, float, float]] = []
        cells: list[tuple[int, ...]] = []

        def append_point(u: float, v: float) -> int:
            index = len(coordinates)
            coordinates.append(data.plane.to_global(u, v))
            return index

        for curve in data.analytic_curves:
            if isinstance(curve, SketchLine):
                start = point_map[curve.start_point_id]
                end = point_map[curve.end_point_id]
                cells.append(
                    (append_point(start.u, start.v), append_point(end.u, end.v))
                )
                continue
            center = point_map[curve.center_point_id]
            if isinstance(curve, SketchCircle):
                radius = curve.radius
                start_angle = 0.0
                sweep = math.tau
            else:
                start = point_map[curve.start_point_id]
                end = point_map[curve.end_point_id]
                radius = math.hypot(start.u - center.u, start.v - center.v)
                start_angle = math.atan2(start.v - center.v, start.u - center.u)
                end_angle = math.atan2(end.v - center.v, end.u - center.u)
                sweep = (
                    (end_angle - start_angle) % math.tau
                    if curve.orientation == "ccw"
                    else -((start_angle - end_angle) % math.tau)
                )
            count = sketch_curve_sample_count(
                radius,
                sweep,
                world_per_pixel,
            )
            cells.append(
                tuple(
                    append_point(
                        center.u + radius * math.cos(start_angle + sweep * index / count),
                        center.v + radius * math.sin(start_angle + sweep * index / count),
                    )
                    for index in range(count + 1)
                )
            )

        polydata = pyvista.PolyData()
        polydata.points = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        polydata.lines = np.hstack(
            tuple((len(cell), *cell) for cell in cells)
        ).astype(np.int64)
        self._remove_actor("sketch_draft_curves")
        self._actors["sketch_draft_curves"] = self._add_mesh(
            polydata,
            color=_sketch_geometry_color(data.constraint_status),
            line_width=3,
            name="sketch_draft_curves",
            reset_camera=False,
            pickable=False,
        )
        self._last_curve_sampling = (data, world_per_pixel)
        return True

    def _selection_polydata(
        self, data: SketchDraftRenderData, coordinates: np.ndarray,
    ):
        pyvista = self._backend.pyvista
        if not data.selected_ids:
            return None
        if data.selected_kind == "point":
            indices = tuple(
                index
                for index, point_id in enumerate(data.point_ids)
                if point_id in data.selected_ids
            )
            if indices:
                return pyvista.PolyData(coordinates[np.asarray(indices)])
        if data.selected_kind == "curve":
            cells = tuple(
                curve
                for curve, curve_id in zip(data.curves, data.curve_ids, strict=True)
                if curve_id in data.selected_ids
            )
            if not cells:
                return None
            result = pyvista.PolyData()
            result.points = coordinates
            result.lines = np.hstack(
                tuple((len(curve), *curve) for curve in cells)
            ).astype(np.int64)
            return result
        if data.selected_kind == "profile":
            cells = tuple(
                face
                for face, face_id in zip(
                    data.faces,
                    data.face_ids,
                    strict=True,
                )
                if face_id in data.selected_ids
            )
            if cells:
                result = pyvista.PolyData()
                result.points = coordinates
                result.faces = np.hstack(
                    tuple((len(face), *face) for face in cells)
                ).astype(np.int64)
                return result
        return None

    def clear_preview(self) -> None:
        for actor_name in (
            "sketch_authoring_shape_preview",
            "sketch_authoring_pending_outline",
            "sketch_authoring_pending_points",
            "sketch_authoring_pending_labels",
            "sketch_authoring_hover_outline",
            "sketch_authoring_hover",
            "sketch_authoring_hover_label",
        ):
            self._remove_actor(actor_name)

    def show_preview(
        self, data: SketchDraftRenderData | None, mode: str,
        pending_points: tuple[tuple[float, float, float], ...],
        preview_point: tuple[float, float, float] | None, snap_kind: str | None,
    ) -> None:
        self.clear_preview()
        if self._backend is None:
            return
        pyvista = self._backend.pyvista
        point = preview_point
        if mode == "arc" and pending_points:
            pending = np.asarray(pending_points, dtype=float)
            labels = tuple(
                ("1  Start", "2  Through point")[index]
                for index in range(min(len(pending), 2))
            )
            self._actors["sketch_authoring_pending_outline"] = (
                self._add_mesh(
                    pyvista.PolyData(pending),
                    color="#006d8f",
                    point_size=18,
                    render_points_as_spheres=True,
                    name="sketch_authoring_pending_outline",
                    reset_camera=False,
                    pickable=False,
                )
            )
            self._actors["sketch_authoring_pending_points"] = (
                self._add_mesh(
                    pyvista.PolyData(pending),
                    color="#40c4ff",
                    point_size=11,
                    render_points_as_spheres=True,
                    name="sketch_authoring_pending_points",
                    reset_camera=False,
                    pickable=False,
                )
            )
            self._actors["sketch_authoring_pending_labels"] = (
                self._add_point_labels(
                    pending[: len(labels)],
                    labels,
                    point_size=0,
                    font_size=12,
                    shape=None,
                    text_color="#006d8f",
                    name="sketch_authoring_pending_labels",
                    reset_camera=False,
                )
            )
        if point is None:
            return
        shape_points = sketch_shape_preview_points(
            mode,
            pending_points,
            point,
            (
                None
                if data is None
                else data.plane
            ),
        )
        if shape_points:
            shape = pyvista.PolyData()
            shape.points = np.asarray(shape_points, dtype=float)
            shape.lines = np.asarray(
                (len(shape_points), *range(len(shape_points))),
                dtype=np.int64,
            )
            self._actors["sketch_authoring_shape_preview"] = (
                self._add_mesh(
                    shape,
                    color="#00b8d4",
                    line_width=4,
                    name="sketch_authoring_shape_preview",
                    reset_camera=False,
                    pickable=False,
                )
            )
        label_point = np.asarray((point,), dtype=float)
        self._actors["sketch_authoring_hover_outline"] = (
            self._add_mesh(
                pyvista.PolyData(label_point),
                color="#e0a800",
                point_size=14,
                render_points_as_spheres=True,
                name="sketch_authoring_hover_outline",
                reset_camera=False,
                pickable=False,
            )
        )
        self._actors["sketch_authoring_hover"] = self._add_mesh(
            pyvista.PolyData(label_point),
            color="#ffd54f",
            point_size=9,
            render_points_as_spheres=True,
            name="sketch_authoring_hover",
            reset_camera=False,
            pickable=False,
        )
        coordinate_label = sketch_coordinate_label(
            point,
            (
                SketchPlane.xy()
                if data is None
                else data.plane
            ),
        )
        snap_label = sketch_snap_label(snap_kind)
        inference_labels = {
            "coincident": "Coincident",
            "point_on_curve": "Point on Curve",
            "horizontal": "Horizontal",
            "vertical": "Vertical",
        }
        inference = ", ".join(
            inference_labels.get(kind, kind)
            for kind in (
                data.inference_preview
                if data is not None
                else ()
            )
        )
        hover_text = f"{snap_label} {coordinate_label}" if snap_label else coordinate_label
        if mode == "arc":
            prompts = (
                "Click to set the start",
                "Click to set the arc through point",
                "Click to set the end",
            )
            hover_text = f"{prompts[min(len(pending_points), 2)]}  {hover_text}"
        if inference:
            hover_text += f"  Inference: {inference}"
        self._actors["sketch_authoring_hover_label"] = (
            self._add_point_labels(
                label_point,
                [hover_text],
                point_size=0,
                font_size=11,
                shape=None,
                text_color="#a66a00",
                name="sketch_authoring_hover_label",
                reset_camera=False,
            )
        )

    def clear_reference(self) -> None:
        self._remove_actor("sketch_reference_surface")
        self._remove_actor("sketch_support_face_highlight")

    def show_reference(
        self, preview: GeometryPreview, face_indices: tuple[int, ...],
        support_face_indices: tuple[int, ...],
    ) -> None:
        self.clear_reference()
        if self._backend is None:
            return
        pyvista = self._backend.pyvista
        surface = pyvista.PolyData()
        surface.points = np.asarray(preview.points, dtype=float)
        surface.faces = np.hstack(
            tuple(
                (len(preview.faces[index]), *preview.faces[index])
                for index in face_indices
            )
        ).astype(np.int64)
        actor = self._add_mesh(
            surface,
            color="#7194ab",
            opacity=0.32,
            smooth_shading=False,
            show_edges=True,
            edge_color="#527084",
            line_width=1,
            name="sketch_reference_surface",
            reset_camera=False,
        )
        actor.SetPickable(False)
        self._actors["sketch_reference_surface"] = actor
        support_faces = tuple(preview.faces[index] for index in support_face_indices)
        if support_faces:
            highlight = pyvista.PolyData()
            highlight.points = np.asarray(
                preview.points,
                dtype=float,
            )
            highlight.faces = np.hstack(
                tuple(
                    (len(face), *face)
                    for face in support_faces
                )
            ).astype(np.int64)
            support_actor = self._add_mesh(
                highlight,
                color="#f5a623",
                opacity=0.52,
                show_edges=True,
                edge_color="#ff8c00",
                line_width=4,
                name="sketch_support_face_highlight",
                reset_camera=False,
                pickable=False,
            )
            support_actor.SetPickable(False)
            self._actors["sketch_support_face_highlight"] = (
                support_actor
            )


class WireScene:
    """Own wire work-plane, draft, selection and cursor actors."""

    def __init__(self) -> None:
        self._backend: DraftBackend | None = None
        self._actors: dict[str, Any] = {}

    def bind_backend(self, backend: DraftBackend) -> None:
        if self._backend == backend:
            return
        self.detach()
        self._backend = backend

    def clear(self) -> None:
        for name in tuple(self._actors):
            self._remove_actor(name)

    def detach(self) -> None:
        self.clear()
        self._backend = None

    def _remove_actor(self, name: str) -> None:
        actor = self._actors.pop(name, None)
        if actor is not None and self._backend is not None:
            try:
                self._backend.remove_actor(actor, reset_camera=False, render=False)
            except Exception:
                pass

    def _add_mesh(self, *args, **kwargs):
        actor = self._backend.add_mesh(*args, render=False, **kwargs)
        actor.SetPickable(False)
        return actor

    def _add_point_labels(self, *args, **kwargs):
        actor = self._backend.add_point_labels(
            *args, show_points=False, render=False, **kwargs,
        )
        actor.SetPickable(False)
        return actor

    def install(
        self, data: WireDraftRenderData, *, work_plane: str, offset: float,
        spacing: float, display_size: float | None, selection: tuple[str, str] | None,
    ) -> None:
        if self._backend is None:
            return
        pyvista = self._backend.pyvista
        for name in (
            "wire_work_plane_grid",
            "wire_work_plane_axis_0",
            "wire_work_plane_axis_1",
            "wire_work_plane_origin",
            "wire_work_plane_axis_labels",
            "wire_draft_members",
            "wire_draft_points",
            "wire_pending_member_start",
            "wire_authoring_selection",
            "wire_authoring_selection_label",
            "wire_authoring_hover_outline",
            "wire_authoring_hover",
            "wire_authoring_hover_label",
        ):
            self._remove_actor(name)
        points = np.asarray(data.points, dtype=float).reshape((-1, 3))
        grid_layout = wire_grid_layout(
            points,
            work_plane,
            offset,
            spacing,
            display_size or 10.0,
        )
        axes = grid_layout.axes
        try:
            grid = wire_grid_polydata(pyvista, grid_layout)
            self._actors["wire_work_plane_grid"] = self._add_mesh(
                grid,
                color="#7f8f9f",
                opacity=0.45,
                line_width=2,
                name="wire_work_plane_grid",
                reset_camera=False,
                pickable=False,
            )
            center = np.asarray(grid_layout.center, dtype=float)
            origin = np.zeros(3, dtype=float)
            origin[axes[2]] = offset
            half_size = 0.5 * grid_layout.plane_size
            axis_colors = ("#d9534f", "#45a049", "#3b82c4")
            label_points = [origin]
            axis_labels = ["O"]
            for index, axis in enumerate(axes[:2]):
                start = origin.copy()
                end = origin.copy()
                start[axis] = center[axis] - half_size
                end[axis] = center[axis] + half_size
                axis_line = pyvista.PolyData()
                axis_line.points = np.asarray((start, end), dtype=float)
                axis_line.lines = np.asarray((2, 0, 1), dtype=np.int64)
                actor_name = f"wire_work_plane_axis_{index}"
                self._actors[actor_name] = self._add_mesh(
                    axis_line,
                    color=axis_colors[axis],
                    line_width=4,
                    name=actor_name,
                    reset_camera=False,
                    pickable=False,
                )
                label_points.append(end)
                axis_labels.append("XYZ"[axis])
            self._actors["wire_work_plane_origin"] = self._add_mesh(
                pyvista.PolyData(np.asarray((origin,), dtype=float)),
                color="#ff8c00",
                point_size=13,
                render_points_as_spheres=True,
                name="wire_work_plane_origin",
                reset_camera=False,
                pickable=False,
            )
            self._actors["wire_work_plane_axis_labels"] = (
                self._add_point_labels(
                    np.asarray(label_points, dtype=float),
                    axis_labels,
                    point_size=0,
                    font_size=11,
                    shape=None,
                    text_color="#ff8c00",
                    name="wire_work_plane_axis_labels",
                    reset_camera=False,
                )
            )
        except Exception:
            logging.exception("failed to render wire work plane")
        if data.members:
            line_cells = np.hstack(
                [np.asarray((2, *member), dtype=np.int64) for member in data.members]
            )
            members = pyvista.PolyData()
            members.points = points
            members.lines = line_cells
            members.cell_data["draft_member_index"] = np.arange(
                len(data.members), dtype=np.int64
            )
            self._actors["wire_draft_members"] = self._add_mesh(
                members,
                color="#334b5f",
                line_width=3,
                name="wire_draft_members",
                reset_camera=False,
                pickable=False,
            )
        if len(points):
            draft_points = pyvista.PolyData(points)
            draft_points.point_data["draft_point_index"] = np.arange(
                len(points), dtype=np.int64
            )
            self._actors["wire_draft_points"] = self._add_mesh(
                draft_points,
                color="#406f8f",
                point_size=10,
                render_points_as_spheres=True,
                name="wire_draft_points",
                reset_camera=False,
                pickable=False,
            )
            pending = data.pending_member_start
            if pending in data.point_names:
                pending_point = pyvista.PolyData(
                    points[[data.point_names.index(pending)]]
                )
                self._actors["wire_pending_member_start"] = self._add_mesh(
                    pending_point,
                    color="#d69a3a",
                    point_size=15,
                    render_points_as_spheres=True,
                    name="wire_pending_member_start",
                    reset_camera=False,
                    pickable=False,
                )
        selected = selection
        selection_label_point = None
        if selected is not None:
            kind, name = selected
            if kind == "point" and name in data.point_names:
                selected_points = points[[data.point_names.index(name)]]
                selected_point = pyvista.PolyData(selected_points)
                self._actors["wire_authoring_selection"] = self._add_mesh(
                    selected_point,
                    color="#ff8c00",
                    point_size=20,
                    render_points_as_spheres=True,
                    name="wire_authoring_selection",
                    reset_camera=False,
                    pickable=False,
                )
                selection_label_point = selected_points
            elif kind == "member" and name in data.member_names:
                member_index = data.member_names.index(name)
                start, end = data.members[member_index]
                selected_member = pyvista.PolyData()
                selected_member.points = points
                selected_member.lines = np.asarray(
                    (2, start, end),
                    dtype=np.int64,
                )
                self._actors["wire_authoring_selection"] = self._add_mesh(
                    selected_member,
                    color="#ff8c00",
                    line_width=8,
                    name="wire_authoring_selection",
                    reset_camera=False,
                    pickable=False,
                )
                selection_label_point = np.mean(
                    points[[start, end]],
                    axis=0,
                    keepdims=True,
                )
            if selection_label_point is not None:
                self._actors["wire_authoring_selection_label"] = (
                    self._add_point_labels(
                        selection_label_point,
                        [name],
                        point_size=0,
                        font_size=11,
                        shape=None,
                        text_color="#ff8c00",
                        name="wire_authoring_selection_label",
                        reset_camera=False,
                    )
                )

    def clear_hover(self) -> None:
        for actor_name in (
            "wire_authoring_hover_outline",
            "wire_authoring_hover",
            "wire_authoring_hover_label",
        ):
            self._remove_actor(actor_name)

    def show_hover(
        self, data: WireDraftRenderData | None, hovered: tuple[str, str] | None,
        preview_point: tuple[float, float, float] | None,
    ) -> tuple[np.ndarray, str] | None:
        self.clear_hover()
        if (
            data is None or self._backend is None
            or (hovered is None and preview_point is None)
        ):
            return None
        pyvista = self._backend.pyvista
        points = np.asarray(data.points, dtype=float).reshape((-1, 3))
        label_point = None
        label = None
        if preview_point is not None:
            label_point = np.asarray((preview_point,), dtype=float)
            label = wire_coordinate_label(preview_point)
            self._actors["wire_authoring_hover_outline"] = (
                self._add_mesh(
                    pyvista.PolyData(label_point),
                    color="#173443",
                    point_size=24,
                    render_points_as_spheres=True,
                    name="wire_authoring_hover_outline",
                    reset_camera=False,
                    pickable=False,
                )
            )
            self._actors["wire_authoring_hover"] = self._add_mesh(
                pyvista.PolyData(label_point),
                color="#00e5ff",
                point_size=16,
                render_points_as_spheres=True,
                name="wire_authoring_hover",
                reset_camera=False,
                pickable=False,
            )
        elif hovered is not None:
            kind, name = hovered
            label = name
            if kind == "point" and name in data.point_names:
                label_point = points[[data.point_names.index(name)]]
                self._actors["wire_authoring_hover_outline"] = (
                    self._add_mesh(
                        pyvista.PolyData(label_point),
                        color="#173443",
                        point_size=26,
                        render_points_as_spheres=True,
                        name="wire_authoring_hover_outline",
                        reset_camera=False,
                        pickable=False,
                    )
                )
                self._actors["wire_authoring_hover"] = self._add_mesh(
                    pyvista.PolyData(label_point),
                    color="#00e5ff",
                    point_size=18,
                    render_points_as_spheres=True,
                    name="wire_authoring_hover",
                    reset_camera=False,
                    pickable=False,
                )
            elif kind == "member" and name in data.member_names:
                member_index = data.member_names.index(name)
                start, end = data.members[member_index]
                hovered_member = pyvista.PolyData()
                hovered_member.points = points
                hovered_member.lines = np.asarray(
                    (2, start, end),
                    dtype=np.int64,
                )
                self._actors["wire_authoring_hover_outline"] = (
                    self._add_mesh(
                        hovered_member,
                        color="#173443",
                        line_width=10,
                        name="wire_authoring_hover_outline",
                        reset_camera=False,
                        pickable=False,
                    )
                )
                self._actors["wire_authoring_hover"] = self._add_mesh(
                    hovered_member,
                    color="#00e5ff",
                    line_width=6,
                    name="wire_authoring_hover",
                    reset_camera=False,
                    pickable=False,
                )
                label_point = np.mean(
                    points[[start, end]],
                    axis=0,
                    keepdims=True,
                )
        if label_point is None or label is None:
            return None
        return label_point, label

    def show_hover_label(self, anchor: np.ndarray, label: str) -> None:
        if self._backend is None:
            return
        self._actors["wire_authoring_hover_label"] = (
            self._add_point_labels(
                anchor,
                [label],
                point_size=0,
                font_size=11,
                shape=None,
                text_color="#00e5ff",
                always_visible=True,
                justification_horizontal="left",
                justification_vertical="bottom",
                name="wire_authoring_hover_label",
                reset_camera=False,
            )
        )
