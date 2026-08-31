"""后处理场景中相互独立的形状和着色状态。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from typing import Literal

from .colormaps import DEFAULT_CUSTOM_COLOR_STOPS, normalize_color_stops


ShapeMode = Literal["undeformed", "deformed"]


@dataclass(frozen=True, slots=True)
class DisplayState:
    """描述当前几何形状和云图开关。"""

    shape_mode: ShapeMode = "undeformed"
    contour_enabled: bool = False


@dataclass(frozen=True, slots=True)
class ContourDisplayState:
    """视口云图、边线和注释的统一显示状态。"""

    manual: bool = False
    minimum: float = 0.0
    maximum: float = 1.0
    levels: int = 12
    colormap: str = "abaqus_rainbow"
    colormap_reverse: bool = False
    custom_color_stops: tuple[tuple[float, str], ...] = DEFAULT_CUSTOM_COLOR_STOPS
    style: str = "segmented"
    legend: bool = True
    edges: bool = True
    render_mode: str = "shaded"
    edge_mode: str = "geometry"
    edge_style: str = "solid"
    edge_width: float = 1.0
    number_format: str = "scientific"
    decimals: int = 2
    orientation: str = "vertical"
    show_minimum: bool = False
    show_maximum: bool = False
    show_ids: bool = False
    legend_font: str = "Arial"
    legend_font_size: int = 14
    legend_position: str = "auto"
    legend_label_count: str = "auto"
    legend_title: bool = True
    show_coordinate_system: bool = True
    show_node_labels: bool = False
    show_element_labels: bool = False
    averaging_threshold: float = 75.0
    # ``auto`` keeps the light/dark palette chosen by the viewport.  Explicit
    # colours belong to presentation state only and never affect result data.
    face_color: str = "auto"
    edge_color: str = "auto"
    model_opacity: float = 1.0
    deformed_color: str = "auto"
    deformed_line_style: str = "solid"
    deformed_opacity: float = 1.0
    undeformed_color: str = "auto"
    undeformed_line_style: str = "solid"
    undeformed_opacity: float = 0.65

    def __post_init__(self) -> None:
        for name in (
            "manual",
            "legend",
            "edges",
            "colormap_reverse",
            "show_minimum",
            "show_maximum",
            "show_ids",
            "legend_title",
            "show_coordinate_system",
            "show_node_labels",
            "show_element_labels",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name in (
            "minimum",
            "maximum",
            "edge_width",
            "averaging_threshold",
            "model_opacity",
            "deformed_opacity",
            "undeformed_opacity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(self.averaging_threshold) <= 100.0:
            raise ValueError("averaging_threshold must be from 0.0 through 100.0")
        for name in (
            "model_opacity",
            "deformed_opacity",
            "undeformed_opacity",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be from 0.0 through 1.0")
        for name in ("levels", "decimals", "legend_font_size"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an int")
        if self.levels < 1:
            raise ValueError("levels must be positive")
        if self.decimals < 0:
            raise ValueError("decimals must be non-negative")
        if self.legend_font_size < 1:
            raise ValueError("legend_font_size must be positive")
        normalized_stops = normalize_color_stops(self.custom_color_stops)
        if normalized_stops != self.custom_color_stops:
            object.__setattr__(self, "custom_color_stops", normalized_stops)
        for name in (
            "colormap",
            "style",
            "render_mode",
            "edge_mode",
            "edge_style",
            "number_format",
            "orientation",
            "legend_font",
            "legend_position",
            "legend_label_count",
            "face_color",
            "edge_color",
            "deformed_color",
            "deformed_line_style",
            "undeformed_color",
            "undeformed_line_style",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty string")

    @classmethod
    def from_mapping(
        cls,
        options: Mapping[str, object],
        *,
        base: "ContourDisplayState | None" = None,
    ) -> "ContourDisplayState":
        """Build a state from a UI/renderer mapping."""

        if not isinstance(options, Mapping):
            raise TypeError("options must be a mapping")
        values = (base or cls()).to_mapping()
        names = {item.name for item in fields(cls)}
        for key, value in options.items():
            if key in names:
                values[key] = value
        return cls(**values)

    def to_mapping(self) -> dict[str, object]:
        """Return the renderer-facing projection of this state."""

        return {item.name: getattr(self, item.name) for item in fields(self)}
