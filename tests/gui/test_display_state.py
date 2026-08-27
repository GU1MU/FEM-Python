from __future__ import annotations

import pytest

from fem_gui.visualization.scene import ContourDisplayState


def test_contour_display_state_round_trips_renderer_options() -> None:
    state = ContourDisplayState.from_mapping(
        {
            "edge_mode": "feature",
            "edge_style": "dashed",
            "edge_width": 2.5,
            "legend": False,
            "show_coordinate_system": False,
            "show_node_labels": True,
            "show_element_labels": True,
            "averaging_threshold": 25.0,
            # UI-only options are intentionally ignored at this boundary.
            "range_mode": "global_step",
        }
    )

    assert state.edge_mode == "feature"
    assert state.edge_style == "dashed"
    assert state.edge_width == 2.5
    assert not state.legend
    assert not state.show_coordinate_system
    assert state.show_node_labels
    assert state.show_element_labels
    assert state.averaging_threshold == 25.0
    mapping = state.to_mapping()
    assert mapping["show_node_labels"] is True
    assert mapping["show_element_labels"] is True
    assert "range_mode" not in mapping


def test_contour_display_state_rejects_invalid_averaging_threshold() -> None:
    with pytest.raises(ValueError, match="averaging_threshold"):
        ContourDisplayState(averaging_threshold=101.0)


def test_contour_display_state_round_trips_colour_palette_options() -> None:
    state = ContourDisplayState.from_mapping(
        {
            "colormap": "custom",
            "colormap_reverse": True,
            "custom_color_stops": [
                [1.0, "#ff0000"],
                [0.0, "#0000ff"],
                [0.5, "#ffffff"],
            ],
        }
    )

    assert state.colormap == "custom"
    assert state.colormap_reverse
    assert state.custom_color_stops == (
        (0.0, "#0000ff"),
        (0.5, "#ffffff"),
        (1.0, "#ff0000"),
    )
    assert state.to_mapping()["custom_color_stops"] == state.custom_color_stops
