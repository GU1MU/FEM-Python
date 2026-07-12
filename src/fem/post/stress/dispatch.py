from __future__ import annotations

from typing import Any, Iterable


ELEMENT_STRESS_KEYS = {
    "truss2",
    "tri3",
    "tri6",
    "quad4",
    "quad8",
    "hex8",
    "hex20",
    "tet4",
    "tet10",
}
NODAL_STRESS_KEYS = {
    "tri3",
    "tri6",
    "quad4",
    "quad8",
    "hex8",
    "hex20",
    "tet4",
    "tet10",
}
TYPE_GROUPS = {
    "truss2": "line",
    "tri3": "plane",
    "tri6": "plane",
    "quad4": "plane",
    "quad8": "plane",
    "hex8": "solid",
    "hex20": "solid",
    "tet4": "solid",
    "tet10": "solid",
}


def resolve_type_keys(mesh: Any, element_type: str | None) -> tuple[str, ...]:
    """Resolve normalized stress exporter keys while preserving mesh element order."""
    if element_type is not None:
        type_key = type_key_from_name(element_type)
        if type_key is None:
            raise ValueError(f"Unsupported stress element type: {element_type!r}")
        return (type_key,)

    type_keys: list[str] = []
    seen: set[str] = set()
    for elem in mesh.elements:
        type_key = type_key_from_name(elem.type)
        if type_key is None:
            raise ValueError(f"Unsupported stress element type: {elem.type!r}")
        if type_key not in seen:
            seen.add(type_key)
            type_keys.append(type_key)

    if not type_keys:
        raise ValueError("Cannot infer stress element type from mesh")
    return tuple(type_keys)


def stress_group_for_keys(type_keys: Iterable[str]) -> str:
    """Return one compatible stress group for a collection of type keys."""
    groups = set()
    for key in type_keys:
        if key not in TYPE_GROUPS:
            raise ValueError(f"Unsupported stress element type key: {key!r}")
        groups.add(TYPE_GROUPS[key])
    if len(groups) != 1:
        raise ValueError(
            f"Mixed stress export requires compatible element groups, got {sorted(groups)}"
        )
    return groups.pop()


def element_stress_supported(type_keys: Iterable[str]) -> bool:
    """Return whether all type keys support element stress export."""
    keys = tuple(type_keys)
    return bool(keys) and all(key in ELEMENT_STRESS_KEYS for key in keys)


def nodal_stress_supported(type_keys: Iterable[str]) -> bool:
    """Return whether all type keys support nodal stress export."""
    keys = tuple(type_keys)
    return bool(keys) and all(key in NODAL_STRESS_KEYS for key in keys)


def default_gauss_order(type_key: str) -> int | None:
    """Return the default nodal stress extrapolation order for one type key."""
    if type_key in {"quad4", "hex8"}:
        return 2
    if type_key in {"tri6", "quad8", "hex20"}:
        return 3
    return None


def type_key_from_name(element_type: Any) -> str | None:
    """Normalize mesh element type names to stress exporter keys."""
    return {
        "truss2": "truss2",
        "tri6plane": "tri6", "tri6": "tri6", "cps6": "tri6", "cpe6": "tri6",
        "tri3plane": "tri3", "tri3": "tri3", "cps3": "tri3", "cpe3": "tri3",
        "quad4plane": "quad4", "quad4": "quad4", "cps4": "quad4", "cpe4": "quad4",
        "quad8plane": "quad8", "quad8": "quad8", "cps8": "quad8", "cpe8": "quad8",
        "hex20": "hex20", "c3d20": "hex20",
        "hex8": "hex8", "c3d8": "hex8",
        "tet10": "tet10", "c3d10": "tet10",
        "tet4": "tet4", "c3d4": "tet4",
    }.get(str(element_type).casefold())
