"""Project-owned contour color maps."""

from __future__ import annotations

from colorsys import hsv_to_rgb

ABAQUS_RAINBOW = "abaqus_rainbow"


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


def resolve_contour_colormap(name: str, color_count: int) -> str | list[str]:
    """Resolve project color-map names to values accepted by PyVista."""

    if name in {ABAQUS_RAINBOW, "jet"}:
        return abaqus_rainbow_colors(color_count)
    return name


def _rgb_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        *(round(channel * 255.0) for channel in rgb)
    )


__all__ = [
    "ABAQUS_RAINBOW",
    "abaqus_rainbow_colors",
    "resolve_contour_colormap",
]
