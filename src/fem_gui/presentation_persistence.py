"""Persist GUI-only result presentation state outside the project schema.

Presentation settings are deliberately kept out of the FEM project codec.  A
project remains a numerical/model document, while this small side store lets
the GUI reopen the last result frame and display configuration for a known
file path without coupling the solver to Qt or to a versioned legacy schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from fem.results import (
    FieldPosition,
    FieldRequest,
    FieldMaterializationKey,
    NodalAveragingPolicy,
    ResultFieldId,
    ResultVariable,
    ScalarFieldSelection,
)

from .visualization.scene import DisplayState
from .view_cut_state import normalize_view_cut_settings
from .workspace import DocumentPresentationState, canonical_path


_SETTINGS_PREFIX = "presentationState/v1"


def presentation_settings_key(path: str | Path) -> str:
    """Return a stable, non-path-leaking QSettings key for one document."""

    normalized = canonical_path(path)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{_SETTINGS_PREFIX}/{digest}"


def save_presentation_state(
    settings: object,
    path: str | Path,
    state: DocumentPresentationState,
) -> bool:
    """Store one validated JSON payload and return whether it was accepted."""

    if not isinstance(state, DocumentPresentationState):
        raise TypeError("state must be DocumentPresentationState")
    try:
        payload = encode_presentation_state(state)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        setter = getattr(settings, "setValue")
        setter(presentation_settings_key(path), encoded)
        sync = getattr(settings, "sync", None)
        if callable(sync):
            sync()
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    return True


def load_presentation_state(
    settings: object,
    path: str | Path,
) -> dict[str, object] | None:
    """Read one side-store payload, ignoring malformed/stale GUI settings."""

    try:
        getter = getattr(settings, "value")
        raw = getter(presentation_settings_key(path), None)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if isinstance(raw, Mapping):
        payload: object = dict(raw)
    elif isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    else:
        return None
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        return None
    return dict(payload)


def encode_presentation_state(
    state: DocumentPresentationState,
) -> dict[str, object]:
    """Convert the mutable GUI record to a JSON-compatible mapping."""

    if not isinstance(state, DocumentPresentationState):
        raise TypeError("state must be DocumentPresentationState")
    display = state.display_state
    display_payload = None
    if isinstance(display, DisplayState):
        display_payload = {
            "shape_mode": display.shape_mode,
            "contour_enabled": display.contour_enabled,
        }
    selection = _encode_selection(state.result_selection)
    payload: dict[str, object] = {
        "version": 1,
        "module_name": state.module_name,
        "step_name": state.step_name,
        "result_selection": selection,
        "result_frame_index": int(state.result_frame_index),
        "display_state": display_payload,
        "result_scale_mode": str(state.result_scale_mode),
        "result_scale_value": float(state.result_scale_value),
        "contour_options": deepcopy(state.contour_options),
        "overlay_undeformed": bool(state.overlay_undeformed),
        "display_groups": deepcopy(state.display_groups),
        "active_display_group": state.active_display_group,
        "view_cut_settings": normalize_view_cut_settings(
            state.view_cut_settings
        ),
        "animation_settings": deepcopy(state.animation_settings),
    }
    return _json_ready(payload)


def apply_presentation_payload(
    state: DocumentPresentationState,
    payload: Mapping[str, object],
) -> DocumentPresentationState:
    """Apply a side-store payload to an existing in-memory state record."""

    if not isinstance(state, DocumentPresentationState):
        raise TypeError("state must be DocumentPresentationState")
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise ValueError("unsupported presentation state version")

    module_name = payload.get("module_name")
    if module_name is None or isinstance(module_name, str):
        state.module_name = module_name
    step_name = payload.get("step_name")
    if step_name is None or isinstance(step_name, str):
        state.step_name = step_name

    frame = payload.get("result_frame_index")
    if type(frame) is int and frame >= 0:
        state.result_frame_index = frame
    display = _decode_display_state(payload.get("display_state"))
    if display is not None:
        state.display_state = display
    selection = _decode_selection(payload.get("result_selection"))
    if selection is not None:
        state.result_selection = selection
    scale_mode = payload.get("result_scale_mode")
    if scale_mode in {"auto", "real", "custom"}:
        state.result_scale_mode = str(scale_mode)
    scale_value = payload.get("result_scale_value")
    if _finite_number(scale_value):
        state.result_scale_value = float(scale_value)
    for name in ("contour_options", "display_groups", "animation_settings"):
        value = payload.get(name)
        if isinstance(value, Mapping):
            setattr(state, name, deepcopy(dict(value)))
    if "view_cut_settings" in payload:
        state.view_cut_settings = normalize_view_cut_settings(
            payload.get("view_cut_settings")
        )
    overlay = payload.get("overlay_undeformed")
    if type(overlay) is bool:
        state.overlay_undeformed = overlay
    active_group = payload.get("active_display_group")
    if active_group is None or isinstance(active_group, str):
        state.active_display_group = active_group
    return state


def _encode_selection(value: object) -> dict[str, object] | None:
    if not isinstance(value, ScalarFieldSelection):
        return None
    request = value.field_key.request
    field_id = request.field_id
    policy = request.averaging_policy
    return {
        "variable": field_id.variable.value,
        "position": field_id.position.value,
        "section_point_number": field_id.section_point_number,
        "component": value.component,
        "recovery_contract": value.field_key.recovery_contract,
        "gauss_order": request.gauss_order,
        "averaging_policy": (
            None
            if policy is None
            else {
                "threshold_percent": policy.threshold_percent,
                "preserve_region_boundaries": policy.preserve_region_boundaries,
            }
        ),
    }


def _decode_selection(value: object) -> ScalarFieldSelection | None:
    if not isinstance(value, Mapping):
        return None
    try:
        variable = ResultVariable(str(value["variable"]))
        position = FieldPosition(str(value["position"]))
        section_point = value.get("section_point_number")
        if section_point is not None:
            section_point = int(section_point)
        field_id = ResultFieldId(variable, position, section_point)
        raw_policy = value.get("averaging_policy")
        policy = None
        if isinstance(raw_policy, Mapping):
            policy = NodalAveragingPolicy(
                threshold_percent=float(raw_policy.get("threshold_percent", 75.0)),
                preserve_region_boundaries=bool(
                    raw_policy.get("preserve_region_boundaries", True)
                ),
            )
        raw_gauss = value.get("gauss_order")
        gauss_order = None if raw_gauss is None else int(raw_gauss)
        request = FieldRequest(
            field_id,
            averaging_policy=policy,
            gauss_order=gauss_order,
        )
        key = FieldMaterializationKey(
            request,
            int(value.get("recovery_contract", 1)),
        )
        component = value.get("component")
        if not isinstance(component, str) or not component.strip():
            return None
        return ScalarFieldSelection(key, component)
    except (KeyError, TypeError, ValueError):
        return None


def _decode_display_state(value: object) -> DisplayState | None:
    if not isinstance(value, Mapping):
        return None
    shape = value.get("shape_mode")
    contour = value.get("contour_enabled")
    if shape not in {"undeformed", "deformed"} or type(contour) is not bool:
        return None
    return DisplayState(shape, contour)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def _json_ready(value: object) -> Any:
    """Normalize nested tuples/enums while rejecting unserializable objects."""

    if isinstance(value, Enum):
        return _json_ready(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("presentation state contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    raise TypeError(
        f"presentation state contains unsupported value {type(value).__name__}"
    )


__all__ = [
    "apply_presentation_payload",
    "encode_presentation_state",
    "load_presentation_state",
    "presentation_settings_key",
    "save_presentation_state",
]
