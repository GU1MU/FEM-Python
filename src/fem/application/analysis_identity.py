"""Stable application identities for analysis objects."""

from __future__ import annotations

import unicodedata
from copy import deepcopy
from dataclasses import replace
from typing import Any, Iterable, Sequence


ANALYSIS_OBJECT_COLLECTIONS = (
    "boundaries",
    "cloads",
    "edge_loads",
    "surface_loads",
    "line_loads",
    "body_loads",
    "gravity_loads",
    "outputs",
)
_LOAD_COLLECTIONS = frozenset(ANALYSIS_OBJECT_COLLECTIONS[1:-1])


def analysis_object_namespace(collection: str) -> str:
    if collection == "boundaries":
        return "boundaries"
    if collection == "outputs":
        return "outputs"
    if collection in _LOAD_COLLECTIONS:
        return "loads"
    raise ValueError(f"unknown analysis collection {collection!r}")


def compatibility_analysis_name(
    step_name: str,
    collection: str,
    index: int,
) -> str:
    """Return one deterministic `{type}-{function}` legacy display identity."""

    if type(step_name) is not str or not step_name.strip():
        raise ValueError("analysis step name must be nonblank")
    if type(index) is not int or index < 0:
        raise ValueError("analysis object index must be a non-negative integer")
    object_type = {
        "boundaries": "位移",
        "outputs": "结果请求",
    }.get(collection, "载荷")
    collection_label = {
        "boundaries": "位移",
        "cloads": "节点",
        "edge_loads": "二维边",
        "surface_loads": "三维面",
        "line_loads": "线",
        "body_loads": "体力",
        "gravity_loads": "重力",
        "outputs": "输出",
    }.get(collection)
    if collection_label is None:
        raise ValueError(f"unknown analysis collection {collection!r}")
    step_label = unicodedata.normalize("NFKC", step_name.strip())
    return f"{object_type}-兼容-{step_label}-{collection_label}-{index + 1}"


def with_compatibility_analysis_names(
    steps: Sequence[Any],
) -> tuple[Any, ...]:
    """Copy steps and fill only anonymous analysis-object identities."""

    owned = deepcopy(tuple(steps))
    occupied: dict[str, set[str]] = {
        "boundaries": set(),
        "loads": set(),
        "outputs": set(),
    }
    for step in owned:
        for collection in ANALYSIS_OBJECT_COLLECTIONS:
            namespace = analysis_object_namespace(collection)
            for item in tuple(getattr(step, collection)):
                name = getattr(item, "name", None)
                if name is not None:
                    occupied[namespace].add(
                        unicodedata.normalize("NFKC", name).casefold()
                    )
    for step in owned:
        for collection in ANALYSIS_OBJECT_COLLECTIONS:
            values = tuple(getattr(step, collection))
            namespace = analysis_object_namespace(collection)
            migrated = []
            for index, item in enumerate(values):
                if getattr(item, "name", None) is not None:
                    migrated.append(item)
                    continue
                base = compatibility_analysis_name(
                    str(step.name),
                    collection,
                    index,
                )
                candidate = base
                suffix = 2
                while (
                    unicodedata.normalize("NFKC", candidate).casefold()
                    in occupied[namespace]
                ):
                    candidate = f"{base}-{suffix}"
                    suffix += 1
                occupied[namespace].add(
                    unicodedata.normalize("NFKC", candidate).casefold()
                )
                migrated.append(replace(item, name=candidate))
            setattr(
                step,
                collection,
                tuple(migrated),
            )
    validate_analysis_object_names(owned, require_all=True)
    return owned


def without_analysis_object_names(
    steps: Sequence[Any],
) -> tuple[Any, ...]:
    """Copy steps and erase application identities for an older codec."""

    owned = deepcopy(tuple(steps))
    for step in owned:
        for collection in ANALYSIS_OBJECT_COLLECTIONS:
            setattr(
                step,
                collection,
                tuple(
                    replace(item, name=None)
                    for item in tuple(getattr(step, collection))
                ),
            )
    return owned


def validate_analysis_object_names(
    steps: Iterable[Any],
    *,
    require_all: bool,
) -> None:
    """Validate stable uniqueness within boundary/load/output namespaces."""

    occupied: dict[str, set[str]] = {
        "boundaries": set(),
        "loads": set(),
        "outputs": set(),
    }
    for step in steps:
        for collection in ANALYSIS_OBJECT_COLLECTIONS:
            namespace = analysis_object_namespace(collection)
            for item in tuple(getattr(step, collection)):
                name = getattr(item, "name", None)
                if name is None:
                    if require_all:
                        raise ValueError(
                            f"{collection} analysis object has no stable name"
                        )
                    continue
                if type(name) is not str or not name.strip():
                    raise ValueError("analysis object name must be nonblank")
                key = unicodedata.normalize("NFKC", name).casefold()
                if key in occupied[namespace]:
                    raise ValueError(
                        f"duplicate analysis object name {name!r} "
                        f"in {namespace}"
                    )
                occupied[namespace].add(key)


__all__ = [
    "ANALYSIS_OBJECT_COLLECTIONS",
    "analysis_object_namespace",
    "compatibility_analysis_name",
    "validate_analysis_object_names",
    "with_compatibility_analysis_names",
    "without_analysis_object_names",
]
