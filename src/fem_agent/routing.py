"""Deterministic routing hints for high-value native geometry requests.

The provider remains responsible for selecting and calling tools.  This module
only classifies a small, explicit vocabulary and returns bounded metadata that
the engine can place beside the typed authoring snapshot.  It deliberately
does not inspect a ``ModelSession`` or infer any CAD identity from the user's
text.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_PROBE_TOOL = "read_geometry_edit_context"
_PREPARE_TOOL = "prepare_geometry_edit"
_TRANSFORM_DIMENSION = 2

_PATH_MARKERS = (
    "沿路径扫掠",
    "按路径扫掠",
    "沿着路径扫掠",
    "路径扫掠",
    "path sweep",
    "sweep along path",
    "sweep along the path",
)
_REVOLVE_MARKERS = (
    "旋转扫掠",
    "绕轴旋转",
    "绕 x 轴",
    "绕 y 轴",
    "绕 z 轴",
    "绕x轴",
    "绕y轴",
    "绕z轴",
    "revolve",
    "revolution",
)
_EXTRUDE_MARKERS = (
    "拉伸",
    "挤出",
    "加厚",
    "加厚成3d",
    "加厚成 3d",
    "extrude",
    "extrusion",
)
_SWEEP_MARKERS = ("扫掠", "sweep", "swept")
_MESH_MARKERS = (
    "网格",
    "划分网格",
    "生成网格",
    "六面体",
    "hex mesh",
    "hexes",
    "mesh",
    "meshing",
)
_ARBITRARY_MARKERS = (
    "尺寸任意",
    "大小任意",
    "高度任意",
    "厚度任意",
    "任意尺寸",
    "任意大小",
    "任意高度",
    "any size",
    "any dimension",
    "any height",
    "whatever size",
)
_ARBITRARY_ENGLISH = re.compile(
    r"(?ix)(?:size|dimension|height|thickness)\s+(?:may|can|is)\s+"
    r"(?:be\s+)?(?:arbitrary|任意|随意)"
)
_DIMENSION_VALUE = re.compile(
    r"(?ix)(?:\d+(?:\.\d+)?)\s*(?:mm|毫米|cm|厘米|m|米|in|inch|英寸)"
)
_HEIGHT_VALUE = re.compile(
    r"(?ix)(?:height|thickness|length|高度|厚度|长度|加厚|厚)\s*(?:to|=|:|为|到)?\s*"
    r"(?:\+?\d+(?:\.\d+)?)"
)
_EXTRUDE_BY_VALUE = re.compile(
    r"(?ix)\bextrud\w*\b[^\n]{0,32}?\b(?:by|to)\s*\+?\d+(?:\.\d+)?"
)
_REVOLVE_ANGLE = re.compile(
    r"(?ix)(?:(?:angle|角度|旋转)\s*(?:=|:|为|至|到)?\s*)?"
    r"(?:\+?\d+(?:\.\d+)?)\s*(?:°|度|deg(?:rees?)?)"
)
_PATH_LABEL = r"(?:[A-Z][A-Za-z0-9_.:]*|点[A-Za-z0-9_.:]+)"
_PATH_LINK = re.compile(
    rf"{_PATH_LABEL}\s*(?:-|→|->|到|至)\s*{_PATH_LABEL}"
)
_PATH_POINT_PAIR = re.compile(
    r"\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?"
    r"(?:\s*,\s*[-+]?\d+(?:\.\d+)?)?\s*\)"
)


@dataclass(frozen=True, slots=True)
class GeometryRouteHint:
    """Provider-safe metadata for one explicitly classified user request."""

    requested_operation: str
    target_part_dimension: int | None = _TRANSFORM_DIMENSION
    required_probe_tool: str | None = _PROBE_TOOL
    required_prepare_tool: str | None = _PREPARE_TOOL
    mesh_prerequisite: bool = False
    missing_fields: tuple[str, ...] = ()
    allow_arbitrary_size: bool = False
    intent_kind: str = "transform"

    def __post_init__(self) -> None:
        if not isinstance(self.requested_operation, str) or not self.requested_operation:
            raise ValueError("requested_operation must be non-empty")
        if isinstance(self.target_part_dimension, bool) or self.target_part_dimension not in {
            None,
            1,
            2,
            3,
        }:
            raise ValueError("target_part_dimension must be 1, 2, 3, or null")
        if self.required_probe_tool is not None and not isinstance(
            self.required_probe_tool, str
        ):
            raise TypeError("required_probe_tool must be a string or null")
        if self.required_prepare_tool is not None and not isinstance(
            self.required_prepare_tool, str
        ):
            raise TypeError("required_prepare_tool must be a string or null")
        if type(self.mesh_prerequisite) is not bool:
            raise TypeError("mesh_prerequisite must be boolean")
        if type(self.allow_arbitrary_size) is not bool:
            raise TypeError("allow_arbitrary_size must be boolean")
        normalized = tuple(str(item) for item in self.missing_fields)
        if any(not item for item in normalized):
            raise ValueError("missing_fields entries must be non-empty")
        object.__setattr__(self, "missing_fields", normalized)

    @property
    def is_transform(self) -> bool:
        return self.intent_kind == "transform"

    @property
    def is_meshing(self) -> bool:
        return self.intent_kind == "meshing"

    @property
    def is_ambiguous(self) -> bool:
        return self.intent_kind == "ambiguous"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, bounded route contract."""

        return {
            "requested_operation": self.requested_operation,
            "target_part_dimension": self.target_part_dimension,
            "required_probe_tool": self.required_probe_tool,
            "required_prepare_tool": self.required_prepare_tool,
            "mesh_prerequisite": self.mesh_prerequisite,
            "missing_fields": list(self.missing_fields),
            "allow_arbitrary_size": self.allow_arbitrary_size,
            "intent_kind": self.intent_kind,
        }

    to_provider_dict = to_dict


def geometry_route_hint(text: str) -> GeometryRouteHint | None:
    """Classify one narrow geometry/mesh intent without executing anything.

    Classification is intentionally conservative: a bare ``sweep`` returns an
    explicit clarification hint, while only path/revolve vocabulary creates a
    transform route.  ``mesh`` vocabulary wins when no explicit transform is
    present, keeping swept meshing separate from path sweeping.
    """

    if not isinstance(text, str) or not text.strip():
        return None
    normalized = " ".join(text.casefold().split())
    compact = normalized.replace(" ", "")
    arbitrary = any(
        marker in normalized or marker in compact for marker in _ARBITRARY_MARKERS
    ) or bool(_ARBITRARY_ENGLISH.search(normalized))
    arbitrary = arbitrary or (
        "任意" in compact and any(
            marker in compact for marker in ("尺寸", "大小", "高度", "厚度")
        )
    )

    explicit_path = _contains_explicit_path(text)
    has_path = any(marker in normalized or marker in compact for marker in _PATH_MARKERS)
    has_path = has_path or any(
        marker in compact
        for marker in ("沿路径", "沿着路径", "按路径", "alongpath")
    )
    has_revolve = any(marker in normalized or marker in compact for marker in _REVOLVE_MARKERS)
    has_extrude = any(marker in normalized or marker in compact for marker in _EXTRUDE_MARKERS)
    has_sweep = any(marker in normalized or marker in compact for marker in _SWEEP_MARKERS)
    has_mesh = any(marker in normalized or marker in compact for marker in _MESH_MARKERS)
    has_path = has_path or bool(
        has_sweep
        and explicit_path
        and re.search(r"(?i)\balong\b", normalized)
    )

    # A mesh request is not a geometry sweep unless the user supplied an
    # explicit path/revolve/extrude operation in the same request.
    if has_mesh and has_sweep and not (has_path or has_revolve or has_extrude):
        return GeometryRouteHint(
            "swept_mesh",
            target_part_dimension=3,
            required_probe_tool=None,
            required_prepare_tool=None,
            mesh_prerequisite=False,
            intent_kind="meshing",
        )

    if has_path:
        missing = () if explicit_path else ("path",)
        return GeometryRouteHint(
            "path_sweep_profile",
            missing_fields=missing,
            allow_arbitrary_size=arbitrary,
        )

    if has_revolve:
        missing: list[str] = []
        if not _contains_explicit_axis(normalized):
            missing.append("axis")
        if not arbitrary and not _REVOLVE_ANGLE.search(normalized):
            missing.append("angle_degrees")
        return GeometryRouteHint(
            "revolve_profile",
            missing_fields=tuple(missing),
            allow_arbitrary_size=arbitrary,
        )

    if has_extrude:
        has_height = bool(
            _DIMENSION_VALUE.search(normalized)
            or _HEIGHT_VALUE.search(normalized)
            or _EXTRUDE_BY_VALUE.search(normalized)
        )
        return GeometryRouteHint(
            "extrude_profiles",
            missing_fields=() if has_height or arbitrary else ("height",),
            allow_arbitrary_size=arbitrary,
        )

    if has_sweep:
        return GeometryRouteHint(
            "sweep",
            target_part_dimension=None,
            required_probe_tool=None,
            required_prepare_tool=None,
            mesh_prerequisite=False,
            missing_fields=("sweep_type",),
            intent_kind="ambiguous",
        )

    return None


def _contains_explicit_axis(normalized: str) -> bool:
    if re.search(
        r"(?ix)\b(?:around|about)\s+(?:the\s+)?[xyz]\s*(?:-\s*)?axis\b",
        normalized,
    ):
        return True
    if re.search(r"(?ix)\baxis\s*[:=]?\s*[xyz]\b", normalized):
        return True
    if re.search(r"(?ix)\b[xyz]\s*-\s*axis\b", normalized):
        return True
    return bool(re.search(r"(?ix)绕\s*[xyz]\s*轴", normalized))


def _contains_explicit_path(text: str) -> bool:
    normalized = " ".join(text.split())
    lowered = normalized.casefold()
    if _PATH_LINK.search(normalized) is not None:
        return True
    if len(_PATH_POINT_PAIR.findall(normalized)) >= 2:
        return True
    for marker in ("path:", "path=", "路径:", "路径=", "路径为"):
        index = lowered.find(marker)
        if index < 0:
            continue
        tail = normalized[index + len(marker) :]
        if _PATH_LINK.search(tail) is not None:
            return True
        if len(_PATH_POINT_PAIR.findall(tail)) >= 2:
            return True
    return False


# Descriptive aliases make the contract discoverable without coupling callers
# to one helper spelling.
detect_geometry_route = geometry_route_hint
detect_geometry_intent = geometry_route_hint
parse_geometry_intent = geometry_route_hint
route_hint_for_message = geometry_route_hint


__all__ = [
    "GeometryRouteHint",
    "detect_geometry_intent",
    "detect_geometry_route",
    "geometry_route_hint",
    "parse_geometry_intent",
    "route_hint_for_message",
]
