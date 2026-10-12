"""Own constraint and load actors behind a narrow native backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .symbol_preparation import SymbolRenderData
from .symbols import (
    SymbolSettings,
    arc_points,
    camera_facing_offset,
    constraint_outward_direction,
    constraint_rotation_axes,
    constraint_symbol_dimensions,
    load_arrow_origins,
    load_symbol_length,
    rotation_lock_points,
)


_GRAVITY_SYMBOL_COLOR = "#FFD400"


@dataclass(frozen=True, slots=True)
class SymbolCameraMetrics:
    world_per_pixel: float | None = None
    position: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class SymbolBackend:
    """Native operations needed only for symbol actor installation."""

    pyvista: Any
    add_mesh: Callable[..., Any]
    add_arrows: Callable[..., Any]
    add_point_labels: Callable[..., Any]
    remove_actor: Callable[..., Any]


def _line_only_polydata(pyvista, points: np.ndarray, lines: Iterable[int]):
    """Build line cells without PyVista's implicit vertex cells."""

    dataset = pyvista.PolyData()
    dataset.points = points
    dataset.lines = np.asarray(tuple(lines), dtype=np.int64)
    dataset.set_active_scalars(None)
    return dataset


class SymbolScene:
    """Own display settings, camera stamps and symbol actors; never render."""

    def __init__(self) -> None:
        self.settings = SymbolSettings()
        self.visible = True
        self._sampling_density_override: str | None = None
        self._backend: SymbolBackend | None = None
        self._actors: dict[str, Any] = {}
        self._camera_metrics = SymbolCameraMetrics()
        self._last_scale: float | None = None
        self._last_camera_position: tuple[float, float, float] | None = None
        self._refreshing_camera = False
        self._label_background = "#ffffff"

    @property
    def sampling_density(self) -> str:
        return self._sampling_density_override or self.settings.sampling_density

    @property
    def has_actors(self) -> bool:
        return bool(self._actors)

    def set_sampling_density_override(self, density: str | None) -> None:
        if density not in {None, "low", "medium", "high"}:
            raise ValueError("Symbol sampling density must be low, medium, high, or None")
        self._sampling_density_override = density

    def set_visible(self, visible: bool) -> bool:
        self.visible = bool(visible)
        for actor in self._actors.values():
            actor.SetVisibility(self.visible)
        return bool(self._actors)

    def bind_backend(self, backend: SymbolBackend) -> None:
        if self._backend == backend:
            return
        self.detach()
        self._backend = backend

    def invalidate_camera(self) -> None:
        self._last_scale = None
        self._last_camera_position = None

    def clear(self) -> None:
        for name in tuple(self._actors):
            self._remove_actor(name)
        self.invalidate_camera()

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

    def _add_arrows(self, *args, **kwargs):
        actor = self._backend.add_arrows(*args, render=False, **kwargs)
        actor.SetPickable(False)
        return actor

    def _add_point_labels(self, *args, **kwargs):
        actor = self._backend.add_point_labels(
            *args, show_points=False, render=False, **kwargs,
        )
        actor.SetPickable(False)
        return actor

    @staticmethod
    def symbol_length(reference_length: float, camera: SymbolCameraMetrics, multiplier: float) -> float:
        base = reference_length
        world_per_pixel = camera.world_per_pixel
        if world_per_pixel is not None and world_per_pixel > 0.0:
            base = float(np.clip(base, 24.0 * world_per_pixel, 56.0 * world_per_pixel))
        return max(base * float(multiplier), 1.0e-9)

    def refresh_for_camera(
        self, data: SymbolRenderData, camera: SymbolCameraMetrics,
        display_points: np.ndarray, label_background: str,
    ) -> bool:
        if self._refreshing_camera or not self.visible or self._backend is None:
            return False
        scale = self.symbol_length(data.reference_length, camera, self.settings.scale)
        previous = self._last_scale
        same_scale = previous is not None and abs(scale - previous) <= 0.02 * max(previous, 1.0e-12)
        same_camera = (
            camera.position is not None and self._last_camera_position is not None
            and np.allclose(camera.position, self._last_camera_position)
        )
        if same_scale and same_camera:
            return False
        self._refreshing_camera = True
        try:
            self.install(data, camera, display_points, label_background)
            return True
        finally:
            self._refreshing_camera = False

    def install(
        self, data: SymbolRenderData, camera: SymbolCameraMetrics,
        display_points: np.ndarray, label_background: str,
    ) -> None:
        self.clear()
        if not self.visible or self._backend is None:
            return
        self._camera_metrics = camera
        self._label_background = label_background
        settings = self.settings
        glyph_scale = self.symbol_length(data.reference_length, camera, settings.scale)
        self._last_scale = glyph_scale
        self._last_camera_position = camera.position
        constraint_scale, constraint_radius = constraint_symbol_dimensions(glyph_scale)
        load_scale = load_symbol_length(glyph_scale)
        model_center = np.asarray(data.model_center)
        self._add_gravity_arrow(model_center, np.asarray(data.gravity), load_scale)
        translation_count = 3 if data.is_3d else 2
        constraint_points: list[np.ndarray] = []
        constraint_vectors: list[np.ndarray] = []
        rotation_points: list[np.ndarray] = []
        rotation_axes: list[np.ndarray] = []
        constraint_label_points: dict[int, np.ndarray] = {}
        constraint_labels_by_node: dict[int, list[str]] = {}
        for sample in data.constraints:
            node_id = sample.node_id
            base = np.asarray(sample.point)
            display_base = base + camera_facing_offset(base, camera.position, 0.04 * constraint_scale)
            for component, value in sample.components:
                name = (
                    f"R{component - translation_count + 1}"
                    if translation_count == 3 and component >= translation_count
                    else "R3" if component >= translation_count else f"U{component + 1}"
                )
                constraint_label_points[node_id] = display_base
                constraint_labels_by_node.setdefault(node_id, []).append(f"{name}={value:.6g}" if value else name)
                if component >= translation_count:
                    continue
                outward = constraint_outward_direction(base, model_center, component)
                tip = display_base + 0.08 * constraint_scale * outward
                constraint_points.append(tip + constraint_scale * outward)
                constraint_vectors.append(-outward)
            displayed_axes = constraint_rotation_axes(
                tuple(component for component, _value in sample.components),
                is_3d=data.is_3d, point=base, camera_position=camera.position,
            )
            rotation_points.extend(display_base for _axis in displayed_axes)
            rotation_axes.extend(displayed_axes)
        if constraint_points:
            cloud = self._backend.pyvista.PolyData(np.asarray(constraint_points))
            cloud["directions"] = np.asarray(constraint_vectors)
            cloud["constraint_scale"] = np.full(
                len(constraint_points), constraint_scale
            )
            marker = self._backend.pyvista.Arrow(
                start=(0.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0),
                tip_length=0.34,
                tip_radius=constraint_radius / constraint_scale,
                tip_resolution=16,
                shaft_radius=0.028,
                shaft_resolution=12,
            )
            glyphs = cloud.glyph(
                orient="directions",
                scale="constraint_scale",
                factor=1.0,
                geom=marker,
            )
            self._actors["constraints"] = self._add_mesh(
                glyphs,
                color=settings.constraint_color,
                lighting=False,
                name="constraints",
                reset_camera=False,
            )
        if rotation_points:
            self._add_rotation_constraint_symbols(
                rotation_points, rotation_axes, 0.36 * constraint_scale,
                settings.constraint_color, "constraint_rotations",
            )
        if settings.show_values and constraint_labels_by_node:
            node_ids = sorted(constraint_labels_by_node)
            label_points = [constraint_label_points[node_id] for node_id in node_ids]
            labels = [", ".join(constraint_labels_by_node[node_id]) for node_id in node_ids]
            self._actors["constraint_labels"] = self._add_point_labels(
                np.asarray(label_points), labels, point_size=0, font_size=9,
                shape_color=self._label_background,
                text_color=settings.constraint_color,
                name="constraint_labels", reset_camera=False,
            )
        if data.face_cells:
            region = self._backend.pyvista.PolyData(
                display_points, faces=np.asarray(data.face_cells, dtype=np.int64)
            )
            self._actors["load_regions"] = self._add_mesh(
                region, color=settings.load_color, opacity=0.11, show_edges=False,
                lighting=False, name="load_regions", reset_camera=False,
            )
        if data.edge_lines:
            region_edges = _line_only_polydata(
                self._backend.pyvista,
                display_points,
                data.edge_lines,
            )
            self._actors["load_region_edges"] = self._add_mesh(
                region_edges, color=settings.load_color, line_width=3,
                lighting=False, name="load_region_edges", reset_camera=False,
            )
        self._add_load_arrows(
            data.load_anchors,
            data.load_vectors,
            data.load_starts_at_anchor,
            data.load_labels,
            load_scale,
        )
        if len(data.moment_points):
            self._add_arc_symbols(
                data.moment_points, data.moment_axes, 0.65 * glyph_scale,
                settings.load_color, "load_moments",
            )
            if settings.show_values:
                self._actors["load_moment_labels"] = self._add_point_labels(
                    data.moment_points, data.moment_labels, point_size=0, font_size=9,
                    shape_color=self._label_background,
                    text_color=settings.load_color,
                    name="load_moment_labels", reset_camera=False,
                )

    def _add_arc_symbols(
        self,
        centers: Iterable[np.ndarray],
        axes: Iterable[np.ndarray],
        radius: float,
        color: str,
        name: str,
    ) -> None:
        points: list[np.ndarray] = []
        lines: list[int] = []
        heads: list[np.ndarray] = []
        tangents: list[np.ndarray] = []
        for center, axis in zip(centers, axes):
            arc = arc_points(center, axis, radius)
            if len(arc) < 2:
                continue
            offset = len(points)
            points.extend(arc)
            lines.extend((len(arc), *(offset + index for index in range(len(arc)))))
            axis_start = len(points)
            points.extend((center - 0.45 * radius * axis, center + 0.45 * radius * axis))
            lines.extend((2, axis_start, axis_start + 1))
            heads.append(arc[-1])
            tangents.append(arc[-1] - arc[-2])
        if not points:
            return
        poly = self._backend.pyvista.PolyData(np.asarray(points))
        poly.lines = np.asarray(lines, dtype=np.int64)
        self._actors[name] = self._add_mesh(
            poly, color=color, line_width=2, lighting=False,
            name=name, reset_camera=False,
        )
        head_name = f"{name[:-1]}_heads" if name.endswith("s") else f"{name}_heads"
        self._actors[head_name] = self._add_arrows(
            np.asarray(heads), np.asarray(tangents), mag=0.22 * radius,
            color=color, lighting=False, name=head_name, reset_camera=False,
        )


    def _add_rotation_constraint_symbols(
        self,
        centers: Iterable[np.ndarray],
        axes: Iterable[np.ndarray],
        radius: float,
        color: str,
        name: str,
    ) -> None:
        points: list[np.ndarray] = []
        lines: list[int] = []
        for center, axis in zip(centers, axes):
            ring, bars = rotation_lock_points(center, axis, radius)
            if len(ring) == 0:
                continue
            ring_start = len(points)
            points.extend(ring)
            lines.extend((len(ring), *(ring_start + index for index in range(len(ring)))))
            bars_start = len(points)
            points.extend(bars)
            lines.extend((2, bars_start, bars_start + 1))
            lines.extend((2, bars_start + 2, bars_start + 3))
        if not points:
            return
        poly = self._backend.pyvista.PolyData(np.asarray(points))
        poly.lines = np.asarray(lines, dtype=np.int64)
        self._actors[name] = self._add_mesh(
            poly, color=color, line_width=2, lighting=False,
            name=name, reset_camera=False,
        )


    def _add_load_arrows(
        self,
        anchors: np.ndarray,
        vectors: np.ndarray,
        starts_at_anchor: Sequence[bool],
        labels: Sequence[str],
        glyph_scale: float,
    ) -> None:
        if not len(vectors):
            return
        settings = self.settings
        display_vectors = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(display_vectors, axis=1)
        nonzero = norms > 0.0
        directions = np.zeros_like(display_vectors)
        directions[nonzero] = display_vectors[nonzero] / norms[nonzero, None]
        if settings.normalize_arrows:
            display_vectors = directions
        else:
            maximum = float(np.max(norms))
            if maximum > 0.0:
                factors = 0.55 + 0.45 * np.sqrt(norms / maximum)
                display_vectors = directions * factors[:, None]
        arrow_lengths = np.linalg.norm(display_vectors, axis=1)
        camera_position = self._camera_metrics.position
        anchor_points = np.asarray([
            np.asarray(anchor, dtype=float) + camera_facing_offset(
                anchor,
                camera_position,
                0.08 * glyph_scale,
            )
            for anchor in anchors
        ])
        displayed_lengths = arrow_lengths * glyph_scale
        display_origins = load_arrow_origins(
            anchor_points,
            directions,
            displayed_lengths,
            np.asarray(starts_at_anchor),
        )
        arrow_points = self._backend.pyvista.PolyData(display_origins)
        arrow_points["directions"] = directions
        arrow_points["arrow_scale"] = displayed_lengths
        arrow = self._backend.pyvista.Arrow(
            start=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
            tip_length=0.20, tip_radius=0.07, tip_resolution=8,
            shaft_radius=0.015, shaft_resolution=8,
        )
        arrow_glyphs = arrow_points.glyph(
            orient="directions", scale="arrow_scale", factor=1.0, geom=arrow,
        )
        self._actors["loads"] = self._add_mesh(
            arrow_glyphs, color=settings.load_color, lighting=False,
            name="loads", reset_camera=False,
        )
        if settings.show_values and len(anchors):
            existing = self._actors.get("load_labels")
            if existing is not None:
                self._remove_actor("load_labels")
            labelled = [index for index, label in enumerate(labels) if label]
            if labelled:
                self._actors["load_labels"] = self._add_point_labels(
                    display_origins[labelled], [labels[index] for index in labelled],
                    point_size=0, font_size=9,
                    shape_color=self._label_background,
                    text_color=settings.load_color, name="load_labels",
                    reset_camera=False,
                )


    def _add_gravity_arrow(
        self,
        center: np.ndarray,
        acceleration: np.ndarray,
        glyph_scale: float,
    ) -> None:
        magnitude = float(np.linalg.norm(acceleration))
        if magnitude <= 0.0 or self._backend is None:
            return
        direction = np.asarray(acceleration, dtype=float) / magnitude
        origin = np.asarray(center, dtype=float) - 0.5 * glyph_scale * direction
        actor = self._add_arrows(
            origin.reshape((1, 3)),
            direction.reshape((1, 3)),
            mag=glyph_scale,
            color=_GRAVITY_SYMBOL_COLOR,
            lighting=False,
            name="gravity",
            reset_camera=False,
        )
        self._actors["gravity"] = actor
