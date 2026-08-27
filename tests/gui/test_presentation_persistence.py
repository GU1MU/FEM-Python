from __future__ import annotations

from fem.results import (
    FieldPosition,
    FieldRequest,
    FieldMaterializationKey,
    ResultFieldId,
    ResultVariable,
    ScalarFieldSelection,
)
from fem_gui.presentation_persistence import (
    apply_presentation_payload,
    encode_presentation_state,
    load_presentation_state,
    save_presentation_state,
)
from fem_gui.visualization.scene import DisplayState
from fem_gui.workspace import DocumentPresentationState


class _Settings:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def setValue(self, key: str, value: str) -> None:
        self.values[key] = value

    def value(self, key: str, default=None):
        return self.values.get(key, default)


def test_presentation_state_round_trips_without_project_codec() -> None:
    selection = ScalarFieldSelection(
        FieldMaterializationKey(
            FieldRequest(
                ResultFieldId(ResultVariable.U, FieldPosition.NODE),
            ),
            1,
        ),
        "U1",
    )
    state = DocumentPresentationState(
        module_name="结果",
        step_name="Dynamic-1",
        result_selection=selection,
        result_frame_index=4,
        display_state=DisplayState("deformed", True),
        result_scale_mode="custom",
        result_scale_value=2.5,
        contour_options={"levels": 24, "custom_color_stops": ((0.0, "#00f"),)},
        overlay_undeformed=True,
        display_groups={"外壳": {"element_ids": (1, 3), "exclude": False}},
        active_display_group="外壳",
        view_cut_settings={"enabled": True, "axis": "z", "offset": 0.2, "invert": False},
        animation_settings={"frame_step": 2, "playback_mode": "pingpong"},
    )
    settings = _Settings()
    path = "presentation-model.femproj"

    assert save_presentation_state(settings, path, state)
    payload = load_presentation_state(settings, path)
    assert payload is not None
    restored = DocumentPresentationState()
    apply_presentation_payload(restored, payload)

    assert restored.module_name == state.module_name
    assert restored.step_name == state.step_name
    assert restored.result_selection == state.result_selection
    assert restored.result_frame_index == 4
    assert restored.display_state == state.display_state
    assert restored.result_scale_mode == "custom"
    assert restored.result_scale_value == 2.5
    assert restored.display_groups == {
        "外壳": {"element_ids": [1, 3], "exclude": False}
    }
    assert restored.view_cut_settings == {
        "planes": {
            "x": {"enabled": False, "offset": 0.0, "invert": False},
            "y": {"enabled": False, "offset": 0.0, "invert": False},
            "z": {"enabled": True, "offset": 0.2, "invert": False},
        }
    }
    assert encode_presentation_state(restored)["version"] == 1
