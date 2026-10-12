"""Document-owned result preferences and rendering cache values."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from fem.application.results import (
    FieldMaterializationKey, ResultFieldTopologyTemplate, ResultProvider, ResultSourceKey,
    ScalarFieldSelection,
    ResultMaterializationSnapshot,
)

from .visualization.colormaps import ABAQUS_RAINBOW
from .visualization.contour_rendering import CONTOUR_EDGE_GEOMETRY, CONTOUR_RENDER_SHADED
from .visualization.scene import DisplayState


def default_contour_options() -> dict[str, object]:
    return {
        "manual": False, "minimum": 0.0, "maximum": 1.0,
        "levels": 12, "colormap": ABAQUS_RAINBOW,
        "style": "segmented", "legend": True,
        "render_mode": CONTOUR_RENDER_SHADED,
        "edge_mode": CONTOUR_EDGE_GEOMETRY,
        "edge_style": "solid", "edge_width": 1.0,
        "number_format": "scientific", "decimals": 2,
        "orientation": "vertical", "show_minimum": False,
        "show_maximum": False, "show_ids": False,
        "legend_font": "Arial", "legend_font_size": 14,
        "show_coordinate_system": True, "edges": True,
        "averaging_threshold": 75.0,
    }


@dataclass(slots=True)
class ResultDisplayPreferences:
    display: DisplayState = field(default_factory=DisplayState)
    scale_mode: str = "auto"
    scale_value: float = 1.0
    overlay_undeformed: bool = False
    contour_options: dict[str, object] = field(default_factory=default_contour_options)

    def copy(self) -> ResultDisplayPreferences:
        """Copy small preferences for activation rollback, preserving values."""

        return replace(self, contour_options=dict(self.contour_options))


@dataclass(slots=True)
class ResultRenderingCache:
    """Adapters and computation reuse owned by one document presentation."""

    source: ResultSourceKey | None = None
    generation: int | None = None
    visualization: tuple[FieldMaterializationKey, ResultProvider] | None = None
    topology_template: tuple[ResultMaterializationSnapshot, ResultFieldTopologyTemplate] | None = None
    deformation_scale: tuple[ResultMaterializationSnapshot, str, str, float, float] | None = None
    averaging_refresh_scheduled: bool = False
    averaging_refresh_requested: bool = False
    averaging_request: ResultAveragingRequest | None = None

    def clear(self) -> None:
        self.source = None
        self.generation = None
        self.visualization = None
        self.topology_template = None
        self.deformation_scale = None


@dataclass(frozen=True, slots=True)
class ResultAveragingRequest:
    source: ResultSourceKey
    generation: int
    selection: ScalarFieldSelection
    visual_selection: ScalarFieldSelection
