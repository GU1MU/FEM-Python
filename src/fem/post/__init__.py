"""Pure post-processing algorithms, loaded without result-domain cycles."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from .fields import (
    ResultRegionKey,
    ResultRegionSignature,
    decode_result_region_key,
    decode_result_region_signature,
    encode_result_region_key,
    make_result_region_signature,
    result_region_key_for_element,
    result_region_sort_key,
)

__all__ = [
    "ResultRegionKey",
    "ResultRegionSignature",
    "decode_result_region_key",
    "decode_result_region_signature",
    "displacement",
    "encode_result_region_key",
    "fields",
    "make_result_region_signature",
    "path",
    "polar",
    "result_region_key_for_element",
    "result_region_sort_key",
    "stress",
    "vtk",
]


def __getattr__(name: str) -> ModuleType:
    if name not in {"displacement", "fields", "path", "polar", "stress", "vtk"}:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
