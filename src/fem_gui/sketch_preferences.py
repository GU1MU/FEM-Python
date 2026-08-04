"""Application-level preferences for planar sketch authoring."""

from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6.QtCore import QSettings


@dataclass(frozen=True, slots=True)
class SketchPreferences:
    grid_visible: bool = True
    grid_snap: bool = True
    grid_spacing: float = 0.1
    snap_sketch_points: bool = True
    snap_external_points: bool = True
    snap_midpoints: bool = True
    snap_centers: bool = True
    snap_intersections: bool = True
    screen_snap_tolerance: float = 9.0
    auto_merge_tolerance: float = 1.0e-6
    show_point_ids: bool = True
    show_external_labels: bool = True
    show_profile_fill: bool = True
    show_work_plane_axes: bool = True
    continuous_polyline: bool = True
    end_polyline_on_close: bool = False
    keep_tool_after_completion: bool = True
    confirm_cascade_delete: bool = True
    auto_constraints: bool = True

    def normalized(self) -> "SketchPreferences":
        spacing = float(self.grid_spacing)
        if not math.isfinite(spacing) or spacing <= 0.0:
            spacing = 0.1
        spacing = min(max(spacing, 0.001), 1.0e12)
        screen_tolerance = float(self.screen_snap_tolerance)
        if not math.isfinite(screen_tolerance) or screen_tolerance < 0.0:
            screen_tolerance = 9.0
        screen_tolerance = min(screen_tolerance, 100.0)
        merge_tolerance = float(self.auto_merge_tolerance)
        if not math.isfinite(merge_tolerance) or merge_tolerance < 0.0:
            merge_tolerance = 1.0e-6
        merge_tolerance = min(merge_tolerance, 1.0e6)
        return SketchPreferences(
            grid_visible=bool(self.grid_visible),
            grid_snap=bool(self.grid_snap),
            grid_spacing=spacing,
            snap_sketch_points=bool(self.snap_sketch_points),
            snap_external_points=bool(self.snap_external_points),
            snap_midpoints=bool(self.snap_midpoints),
            snap_centers=bool(self.snap_centers),
            snap_intersections=bool(self.snap_intersections),
            screen_snap_tolerance=screen_tolerance,
            auto_merge_tolerance=merge_tolerance,
            show_point_ids=bool(self.show_point_ids),
            show_external_labels=bool(self.show_external_labels),
            show_profile_fill=bool(self.show_profile_fill),
            show_work_plane_axes=bool(self.show_work_plane_axes),
            continuous_polyline=bool(self.continuous_polyline),
            end_polyline_on_close=bool(self.end_polyline_on_close),
            keep_tool_after_completion=bool(self.keep_tool_after_completion),
            confirm_cascade_delete=bool(self.confirm_cascade_delete),
            auto_constraints=bool(self.auto_constraints),
        )


_KEYS = {
    field: f"sketch/{field}"
    for field in SketchPreferences.__dataclass_fields__
}


def load_sketch_preferences(store: QSettings) -> SketchPreferences:
    defaults = SketchPreferences()
    values = {
        field: store.value(
            key,
            getattr(defaults, field),
            type=(float if isinstance(getattr(defaults, field), float) else bool),
        )
        for field, key in _KEYS.items()
    }
    return SketchPreferences(**values).normalized()


def save_sketch_preferences(store: QSettings, preferences: SketchPreferences) -> None:
    values = preferences.normalized()
    for field, key in _KEYS.items():
        store.setValue(key, getattr(values, field))


__all__ = [
    "SketchPreferences",
    "load_sketch_preferences",
    "save_sketch_preferences",
]
