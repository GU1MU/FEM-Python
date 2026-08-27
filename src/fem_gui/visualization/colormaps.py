"""Project-owned contour color maps and palette definitions."""

from __future__ import annotations

import re
from colorsys import hsv_to_rgb
from itertools import pairwise
from math import isfinite

ABAQUS_RAINBOW = "abaqus_rainbow"
CUSTOM_COLORMAP = "custom"

# Keep the list in one place so the unified dialog and any future colour
# editors expose exactly the same choices.  The values are Matplotlib/PyVista
# names except for the two project-owned entries.
COLORMAP_CHOICES = (
    ("彩虹", ABAQUS_RAINBOW),
    ("涡轮彩色", "turbo"),
    ("维里迪斯", "viridis"),
    ("等离子", "plasma"),
    ("高温", "inferno"),
    ("岩浆", "magma"),
    ("色盲友好", "cividis"),
    ("冷暖", "coolwarm"),
    ("蓝-白-红", "RdBu"),
    ("光谱", "Spectral"),
    ("蓝色渐变", "Blues"),
    ("红色渐变", "Reds"),
    ("灰度", "gray"),
    ("自定义", CUSTOM_COLORMAP),
)

DEFAULT_CUSTOM_COLOR_STOPS = (
    (0.0, "#0000ff"),
    (0.5, "#00ff00"),
    (1.0, "#ff0000"),
)

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_PREVIEW_STOPS = {
    "turbo": ("#30123b", "#2474d8", "#18cfae", "#f4e61e", "#7a0403"),
    "viridis": ("#440154", "#31688e", "#35b779", "#fde725"),
    "plasma": ("#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"),
    "inferno": ("#000004", "#420a68", "#932667", "#dd513a", "#fcffa4"),
    "magma": ("#000004", "#3b0f70", "#8c2981", "#de4968", "#fcfdbf"),
    "cividis": ("#00224e", "#42566c", "#7c7b78", "#b8a96e", "#fee838"),
    "coolwarm": ("#3b4cc0", "#8db0fe", "#dddcdc", "#f4987a", "#b40426"),
    "RdBu": ("#053061", "#67a9cf", "#f7f7f7", "#ef8a62", "#67001f"),
    "Spectral": ("#5e4fa2", "#3288bd", "#e6f598", "#fdae61", "#9e0142"),
    "Blues": ("#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"),
    "Reds": ("#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"),
    "gray": ("#ffffff", "#bdbdbd", "#636363", "#000000"),
}


def abaqus_rainbow_colors(color_count: int) -> list[str]:
    """Return saturated Abaqus-style rainbow colors from low to high."""

    if color_count < 2:
        raise ValueError("color_count must be at least 2")
    return [
        _rgb_hex(
            hsv_to_rgb(
                (2.0 / 3.0) * (1.0 - index / (color_count - 1)),
                1.0,
                1.0,
            )
        )
        for index in range(color_count)
    ]


def normalize_color_stops(value: object) -> tuple[tuple[float, str], ...]:
    """Validate and canonically order editable colour stops."""

    if value is None:
        return DEFAULT_CUSTOM_COLOR_STOPS
    if isinstance(value, (str, bytes)):
        raise TypeError("custom colour stops must be a sequence")
    try:
        raw_stops = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("custom colour stops must be a sequence") from error
    if len(raw_stops) < 2:
        raise ValueError("custom colour map requires at least two colour stops")

    stops: list[tuple[float, str]] = []
    for raw_stop in raw_stops:
        if not isinstance(raw_stop, (tuple, list)) or len(raw_stop) != 2:
            raise ValueError("each colour stop must contain position and colour")
        position = float(raw_stop[0])
        colour = str(raw_stop[1]).strip()
        if not isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("colour stop positions must be from 0.0 through 1.0")
        if _HEX_COLOR.fullmatch(colour) is None:
            raise ValueError("colour stops must use #RRGGBB colours")
        stops.append((position, colour.lower()))
    stops.sort(key=lambda item: item[0])
    if any(left[0] == right[0] for left, right in pairwise(stops)):
        raise ValueError("colour stop positions must be unique")
    if stops[0][0] != 0.0 or stops[-1][0] != 1.0:
        raise ValueError("colour stops must start at 0.0 and end at 1.0")
    return tuple(stops)


def interpolate_color_stops(
    stops: object,
    color_count: int,
) -> list[str]:
    """Sample editable colour stops into a palette accepted by PyVista."""

    if color_count < 2:
        raise ValueError("color_count must be at least 2")
    normalized = normalize_color_stops(stops)
    result: list[str] = []
    for index in range(color_count):
        position = index / (color_count - 1)
        right_index = next(
            (
                stop_index
                for stop_index, stop in enumerate(normalized)
                if stop[0] >= position
            ),
            len(normalized) - 1,
        )
        if right_index == 0:
            result.append(normalized[0][1])
            continue
        left_position, left_colour = normalized[right_index - 1]
        right_position, right_colour = normalized[right_index]
        if right_position == left_position:
            fraction = 0.0
        else:
            fraction = (position - left_position) / (right_position - left_position)
        result.append(_interpolate_hex(left_colour, right_colour, fraction))
    return result


def preview_contour_colors(
    name: str,
    color_count: int = 9,
    *,
    reverse: bool = False,
    custom_color_stops: object = None,
) -> list[str]:
    """Return representative colours for the dialog preview."""

    canonical_name = ABAQUS_RAINBOW if name == "jet" else name
    if canonical_name == CUSTOM_COLORMAP:
        colours = interpolate_color_stops(
            custom_color_stops or DEFAULT_CUSTOM_COLOR_STOPS,
            color_count,
        )
    elif canonical_name == ABAQUS_RAINBOW:
        colours = abaqus_rainbow_colors(color_count)
    else:
        preview_stops = _PREVIEW_STOPS.get(
            canonical_name,
            _PREVIEW_STOPS["viridis"],
        )
        colours = interpolate_color_stops(
            tuple(
                (index / (len(preview_stops) - 1), colour)
                for index, colour in enumerate(
                    preview_stops
                )
            ),
            color_count,
        )
    return list(reversed(colours)) if reverse else colours


def resolve_contour_colormap(
    name: str,
    color_count: int,
    *,
    reverse: bool = False,
    custom_color_stops: object = None,
) -> str | list[str]:
    """Resolve project colour-map names to values accepted by PyVista."""

    canonical_name = ABAQUS_RAINBOW if name == "jet" else name
    if canonical_name == CUSTOM_COLORMAP:
        colours = interpolate_color_stops(
            custom_color_stops or DEFAULT_CUSTOM_COLOR_STOPS,
            color_count,
        )
        return list(reversed(colours)) if reverse else colours
    if canonical_name == ABAQUS_RAINBOW:
        colours = abaqus_rainbow_colors(color_count)
        return list(reversed(colours)) if reverse else colours
    if canonical_name.endswith("_r"):
        return canonical_name[:-2] if reverse else canonical_name
    return f"{canonical_name}_r" if reverse else canonical_name


def _interpolate_hex(left: str, right: str, fraction: float) -> str:
    left_rgb = tuple(int(left[index : index + 2], 16) for index in (1, 3, 5))
    right_rgb = tuple(int(right[index : index + 2], 16) for index in (1, 3, 5))
    return "#{}".format(
        "".join(
            f"{round(left_channel + (right_channel - left_channel) * fraction):02x}"
            for left_channel, right_channel in zip(left_rgb, right_rgb)
        )
    )


def _rgb_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        *(round(channel * 255.0) for channel in rgb)
    )


__all__ = [
    "ABAQUS_RAINBOW",
    "COLORMAP_CHOICES",
    "CUSTOM_COLORMAP",
    "DEFAULT_CUSTOM_COLOR_STOPS",
    "abaqus_rainbow_colors",
    "interpolate_color_stops",
    "normalize_color_stops",
    "preview_contour_colors",
    "resolve_contour_colormap",
]
