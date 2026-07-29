from __future__ import annotations

from fem_gui.visualization.colormaps import (
    ABAQUS_RAINBOW,
    abaqus_rainbow_colors,
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
