"""Canonical presentation state for axis-aligned result view cuts."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite


VIEW_CUT_AXES = ("x", "y", "z")


def _default_plane() -> dict[str, object]:
    return {
        "enabled": False,
        "offset": 0.0,
        "invert": False,
    }


def default_view_cut_settings() -> dict[str, object]:
    """Return a detached empty set of X/Y/Z view-cut planes."""

    return {
        "planes": {
            axis: _default_plane()
            for axis in VIEW_CUT_AXES
        }
    }


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if isfinite(result) else default


def normalize_view_cut_settings(value: object) -> dict[str, object]:
    """Normalize the one canonical view-cut shape used by the GUI and viewport.

    The old single-plane mapping is accepted only at this boundary so saved
    presentation state can be read once and immediately represented as three
    independent axis planes.
    """

    raw = value if isinstance(value, Mapping) else {}
    raw_planes = raw.get("planes")
    planes: dict[str, dict[str, object]] = {}
    if isinstance(raw_planes, Mapping):
        for axis in VIEW_CUT_AXES:
            raw_plane = raw_planes.get(axis)
            raw_plane = raw_plane if isinstance(raw_plane, Mapping) else {}
            planes[axis] = {
                "enabled": bool(raw_plane.get("enabled", False)),
                "offset": _finite_float(raw_plane.get("offset", 0.0)),
                "invert": bool(raw_plane.get("invert", False)),
            }
    else:
        # Read the previous single-axis representation at the boundary only.
        axis = str(raw.get("axis", "z")).strip().casefold()
        if axis not in VIEW_CUT_AXES:
            axis = "z"
        for candidate in VIEW_CUT_AXES:
            planes[candidate] = _default_plane()
        planes[axis] = {
            "enabled": bool(raw.get("enabled", False)),
            "offset": _finite_float(raw.get("offset", 0.0)),
            "invert": bool(raw.get("invert", False)),
        }
    return {"planes": planes}


def view_cut_has_user_data(value: object) -> bool:
    """Return whether any cut plane is active or has a non-default setting."""

    normalized = normalize_view_cut_settings(value)
    planes = normalized["planes"]
    assert isinstance(planes, Mapping)
    return any(
        bool(planes[axis].get("enabled", False))
        or float(planes[axis].get("offset", 0.0)) != 0.0
        or bool(planes[axis].get("invert", False))
        for axis in VIEW_CUT_AXES
    )

