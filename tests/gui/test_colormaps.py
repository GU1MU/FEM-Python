from __future__ import annotations

from fem_gui.visualization.colormaps import (
    ABAQUS_RAINBOW,
    CUSTOM_COLORMAP,
    abaqus_rainbow_colors,
    interpolate_color_stops,
    preview_contour_colors,
    resolve_contour_colormap,
)


def test_abaqus_rainbow_uses_discrete_saturated_blue_to_red_bands() -> None:
    colors = abaqus_rainbow_colors(12)

    assert len(colors) == 12
    assert len(set(colors)) == 12
    assert colors[0] == "#0000ff"
    assert colors[-1] == "#ff0000"


def test_abaqus_rainbow_resolver_supports_legacy_jet_name() -> None:
    expected = abaqus_rainbow_colors(12)

    assert resolve_contour_colormap(ABAQUS_RAINBOW, 12) == expected
    assert resolve_contour_colormap("jet", 12) == expected
    assert resolve_contour_colormap("viridis", 12) == "viridis"


def test_builtin_colormap_can_be_reversed_without_changing_palette_identity() -> None:
    assert resolve_contour_colormap("viridis", 12, reverse=True) == "viridis_r"
    assert resolve_contour_colormap("viridis_r", 12, reverse=True) == "viridis"


def test_custom_color_stops_are_interpolated_for_continuous_and_banded_plots() -> None:
    stops = ((0.0, "#0000ff"), (0.5, "#ffffff"), (1.0, "#ff0000"))

    assert interpolate_color_stops(stops, 3) == [
        "#0000ff",
        "#ffffff",
        "#ff0000",
    ]
    assert resolve_contour_colormap(
        CUSTOM_COLORMAP,
        3,
        custom_color_stops=stops,
    ) == ["#0000ff", "#ffffff", "#ff0000"]
    assert preview_contour_colors(
        CUSTOM_COLORMAP,
        3,
        reverse=True,
        custom_color_stops=stops,
    ) == ["#ff0000", "#ffffff", "#0000ff"]
