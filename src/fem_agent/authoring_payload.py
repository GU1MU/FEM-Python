"""Provider-safe JSON projections and authoring message validation."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import re


_ABSOLUTE_PATH = re.compile(
    r"""(?ix)
    (?:
        [a-z]:[\\/]
        |
        \\\\[^\\/\s]+[\\/][^\\/\s]+
        |
        (?<![a-z0-9])/(?:[^/\s]+/)+[^/\s]*
    )
    """
)
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "nodes",
        "node_ids",
        "node_coordinates",
        "coordinates",
        "elements",
        "element_ids",
        "connectivity",
        "records",
        "result_arrays",
        "result_provider",
        "model_session",
        "model_patch",
        "qt_object",
        "vtk",
        "gmsh_model",
        "raw_patch",
        "patch_before",
        "patch_after",
        "source_path",
        "absolute_path",
        "api_key",
        "credential",
    }
)
# Rich planar-boolean catalogs (entities + planar feature geometry) can
# approach 40 KB once wrapped in the multi-part envelope; 64 KiB keeps one
# worst-case catalog provider-safe while the read handler still truncates
# additional Parts instead of exceeding the budget.
_MAX_PROVIDER_PAYLOAD_BYTES = 65_536
# Aligned with the feature/topology catalog bound so a full catalog entity
# array survives the provider payload sanitization pass.
_MAX_PROVIDER_COLLECTION_ITEMS = 256
_MAX_PROVIDER_DEPTH = 12


def provider_safe_authoring_payload(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate and copy one bounded JSON object before Provider exposure."""

    if not isinstance(value, Mapping):
        raise TypeError("authoring payload must be an object")
    normalized = _safe_json_value(dict(value), depth=0)
    assert isinstance(normalized, dict)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_PROVIDER_PAYLOAD_BYTES:
        raise ValueError("authoring payload exceeds the provider-safe byte budget")
    return normalized


def _provider_safe_summary(value: str) -> str:
    normalized = value.strip()
    encoded = normalized.encode("utf-8")
    if len(encoded) > 2048:
        raise ValueError("authoring summary exceeds the provider-safe byte budget")
    if _ABSOLUTE_PATH.search(normalized):
        raise ValueError("absolute paths are local-only")
    lowered = normalized.casefold()
    if any(
        marker in lowered
        for marker in ("api_key", "apikey", "password", "secret", "credential")
    ):
        raise ValueError("credential material is local-only")
    return normalized


def _provider_safe_error_detail(value: str) -> str:
    """Return bounded exception detail, or "" when it carries local-only content.

    Mirrors the ``_provider_safe_summary`` rules without raising, so the
    generic rejection handler can fall back to the masked placeholder when an
    exception message embeds a filesystem path or credential marker.
    """

    normalized = " ".join(value.split())[:320]
    if not normalized:
        return ""
    if _ABSOLUTE_PATH.search(normalized):
        return ""
    lowered = normalized.casefold()
    if any(
        marker in lowered
        for marker in ("api_key", "apikey", "password", "secret", "credential")
    ):
        return ""
    return normalized


def _safe_json_value(value: object, *, depth: int) -> object:
    if depth > _MAX_PROVIDER_DEPTH:
        raise ValueError("authoring payload exceeds the nesting budget")
    if isinstance(value, Mapping):
        if len(value) > _MAX_PROVIDER_COLLECTION_ITEMS:
            raise ValueError("authoring payload object exceeds the item budget")
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("authoring payload keys must be strings")
            if key.casefold() in _FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"authoring payload field {key!r} is local-only")
            result[key] = _safe_json_value(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_PROVIDER_COLLECTION_ITEMS:
            raise ValueError("authoring payload array exceeds the item budget")
        return [_safe_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError("authoring payload text exceeds the item budget")
        if _ABSOLUTE_PATH.search(value):
            raise ValueError("absolute paths are local-only")
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("authoring payload must contain finite JSON values only")
