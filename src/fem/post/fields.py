"""Neutral, deterministic identities shared by result-field consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np


MATERIAL_SIGNATURE_KEY = "_stress_material_signature"
SECTION_SIGNATURE_KEY = "_stress_section_signature"


@dataclass(frozen=True, slots=True, init=False)
class ResultRegionSignature:
    """Canonical finite-JSON identity for one material or section signature.

    Instances are intentionally created only by
    :func:`make_result_region_signature` or
    :func:`decode_result_region_signature`.  This keeps arbitrary text from
    entering equality, hashing, sorting, CSV, or VTK identities.
    """

    canonical_json: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "ResultRegionSignature must be created by "
            "make_result_region_signature() or "
            "decode_result_region_signature()"
        )

    def __copy__(self) -> ResultRegionSignature:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> ResultRegionSignature:
        del memo
        return self

    @classmethod
    def _from_canonical_json(cls, canonical_json: str) -> ResultRegionSignature:
        instance = object.__new__(cls)
        object.__setattr__(instance, "canonical_json", canonical_json)
        return instance


@dataclass(frozen=True, slots=True)
class ResultRegionKey:
    """Material and section signatures forming one hard result boundary."""

    material_signature: ResultRegionSignature
    section_signature: ResultRegionSignature

    def __post_init__(self) -> None:
        if type(self.material_signature) is not ResultRegionSignature:
            raise TypeError("material_signature must be ResultRegionSignature")
        if type(self.section_signature) is not ResultRegionSignature:
            raise TypeError("section_signature must be ResultRegionSignature")


def make_result_region_signature(payload: Any) -> ResultRegionSignature:
    """Deep-own a strict finite JSON value and return its canonical identity."""

    owned = _clone_json_value(payload, path="$", ancestors=set())
    return ResultRegionSignature._from_canonical_json(_dump_canonical(owned))


def decode_result_region_signature(
    canonical_json: str,
) -> ResultRegionSignature:
    """Decode only text already in the unique canonical JSON representation."""

    _load_canonical(canonical_json, label="result region signature")
    return ResultRegionSignature._from_canonical_json(canonical_json)


def encode_result_region_key(region_key: ResultRegionKey) -> str:
    """Return the unique canonical text projection of one region key."""

    if type(region_key) is not ResultRegionKey:
        raise TypeError("region_key must be ResultRegionKey")
    material = _load_canonical(
        region_key.material_signature.canonical_json,
        label="material signature",
    )
    section = _load_canonical(
        region_key.section_signature.canonical_json,
        label="section signature",
    )
    return _dump_canonical(
        {
            "material": material,
            "section": section,
        }
    )


def decode_result_region_key(canonical_json: str) -> ResultRegionKey:
    """Decode a canonical region-key projection without accepting aliases."""

    payload = _load_canonical(canonical_json, label="result region key")
    if type(payload) is not dict:
        raise ValueError("result region key must be a JSON object")
    if set(payload) != {"material", "section"}:
        raise ValueError("result region key must contain exactly material and section")
    return ResultRegionKey(
        material_signature=make_result_region_signature(payload["material"]),
        section_signature=make_result_region_signature(payload["section"]),
    )


def result_region_sort_key(region_key: ResultRegionKey) -> tuple[str, str]:
    """Return the sole deterministic ordering key for result regions."""

    if type(region_key) is not ResultRegionKey:
        raise TypeError("region_key must be ResultRegionKey")
    return (
        region_key.material_signature.canonical_json,
        region_key.section_signature.canonical_json,
    )


def result_region_key_for_element(element: Any) -> ResultRegionKey:
    """Interpret and deep-own the canonical result-region identity of an element.

    Explicit assignment signatures take precedence.  Elements without those
    signatures retain the historical material-name, material-id, or effective
    ``E``/``nu``/``rho`` identity and the remaining section properties.
    """

    props = dict(getattr(element, "props", {}))
    material_signature = props.get(MATERIAL_SIGNATURE_KEY)
    if material_signature is None:
        if "material" in props:
            material_signature = ("material", props["material"])
        elif "material_id" in props:
            material_signature = ("material_id", props["material_id"])
        else:
            material_signature = (
                "effective",
                tuple(
                    (name, props[name])
                    for name in ("E", "nu", "rho")
                    if name in props
                ),
            )

    section_signature = props.get(SECTION_SIGNATURE_KEY)
    if section_signature is None:
        excluded = {
            MATERIAL_SIGNATURE_KEY,
            SECTION_SIGNATURE_KEY,
            "material",
            "material_id",
            "E",
            "nu",
            "rho",
            "section_type",
        }
        raw_frame_field = props.get("beam_frame_field")
        if raw_frame_field is not None:
            from fem.elements import BeamFrameField

            if type(raw_frame_field) is BeamFrameField:
                excluded.update(
                    {
                        "beam_frame_field",
                        "beam_frame_field_reference",
                    }
                )
        section_properties = {
            key: value for key, value in props.items() if key not in excluded
        }
        section_signature = (
            "section",
            props.get("section_type"),
            section_properties,
        )

    return _result_region_key_from_compatible_signatures(
        material_signature,
        section_signature,
    )


def _result_region_key_from_compatible_signatures(
    material_signature: Any,
    section_signature: Any,
) -> ResultRegionKey:
    return ResultRegionKey(
        _coerce_compatible_region_signature(material_signature),
        _coerce_compatible_region_signature(section_signature),
    )


def _coerce_compatible_region_signature(value: Any) -> ResultRegionSignature:
    if type(value) is ResultRegionSignature:
        return value
    compatible = _compatible_signature_json(
        value,
        path="$",
        ancestors=set(),
    )
    return make_result_region_signature(compatible)


def _compatible_signature_json(
    value: Any,
    *,
    path: str,
    ancestors: set[int],
) -> Any:
    """Translate historical tuple signatures without repr/hash fallbacks."""

    if isinstance(value, np.generic):
        return _compatible_signature_json(
            value.item(),
            path=path,
            ancestors=ancestors,
        )
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} contains a cyclic result-region signature")
        ancestors.add(identity)
        try:
            converted: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(
                        "result-region signature mapping keys must be strings"
                    )
                converted[key] = _compatible_signature_json(
                    item,
                    path=f"{path}.{key}",
                    ancestors=ancestors,
                )
            return converted
        finally:
            ancestors.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} contains a cyclic result-region signature")
        ancestors.add(identity)
        try:
            return [
                _compatible_signature_json(
                    item,
                    path=f"{path}[{index}]",
                    ancestors=ancestors,
                )
                for index, item in enumerate(value)
            ]
        finally:
            ancestors.remove(identity)
    raise TypeError(
        "result-region signatures must contain only finite JSON values"
    )


def _clone_json_value(
    value: Any,
    *,
    path: str,
    ancestors: set[int],
) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int}:
        return value
    if value_type is str:
        _utf8_bytes(value, label=path)
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if value_type is list:
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} contains a cyclic JSON value")
        ancestors.add(identity)
        try:
            return [
                _clone_json_value(
                    item,
                    path=f"{path}[{index}]",
                    ancestors=ancestors,
                )
                for index, item in enumerate(value)
            ]
        finally:
            ancestors.remove(identity)
    if value_type is dict:
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} contains a cyclic JSON value")
        ancestors.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} JSON object keys must be strings")
                _utf8_bytes(key, label=f"{path} object key")
                result[key] = _clone_json_value(
                    item,
                    path=f"{path}.{key}",
                    ancestors=ancestors,
                )
            return result
        finally:
            ancestors.remove(identity)
    raise TypeError(
        f"{path} contains unsupported JSON value type {value_type.__name__}"
    )


def _dump_canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_canonical(canonical_json: str, *, label: str) -> Any:
    encoded = _canonical_text_bytes(canonical_json, label=label)
    try:
        decoded = json.loads(
            canonical_json,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {error}") from error
    owned = _clone_json_value(decoded, path="$", ancestors=set())
    reencoded = _utf8_bytes(_dump_canonical(owned), label=label)
    if reencoded != encoded:
        raise ValueError(f"{label} is not in canonical JSON form")
    return owned


def _canonical_text_bytes(value: Any, *, label: str) -> bytes:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value:
        raise ValueError(f"{label} must not be empty")
    if value.startswith("\ufeff"):
        raise ValueError(f"{label} must not start with a UTF-8 BOM")
    return _utf8_bytes(value, label=label)


def _utf8_bytes(value: str, *, label: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must be valid UTF-8 text") from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


__all__ = [
    "MATERIAL_SIGNATURE_KEY",
    "ResultRegionKey",
    "ResultRegionSignature",
    "SECTION_SIGNATURE_KEY",
    "decode_result_region_key",
    "decode_result_region_signature",
    "encode_result_region_key",
    "make_result_region_signature",
    "result_region_key_for_element",
    "result_region_sort_key",
]
