"""A small, explicit registry for the material editor behaviors.

The registry is intentionally limited to behaviors backed by the current
solver.  The editor, validation layer, and material compiler can grow by
registering one new behavior specification instead of adding unrelated GUI
branches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fem.model import MaterialBehavior, MaterialDefinition


ELASTIC_BEHAVIOR_ID = "mechanical.elasticity.isotropic"
PLASTIC_BEHAVIOR_ID = "mechanical.plasticity.j2"
DENSITY_BEHAVIOR_ID = "general.density"
NEO_HOOKEAN_BEHAVIOR_ID = "mechanical.hyperelasticity.neo_hookean"


@dataclass(frozen=True, slots=True)
class MaterialBehaviorSpec:
    """Editor metadata for one solver-backed material behavior."""

    behavior_id: str
    label: str
    menu_path: tuple[str, ...]
    constitutive_model: str | None = None


MATERIAL_BEHAVIOR_SPECS: tuple[MaterialBehaviorSpec, ...] = (
    MaterialBehaviorSpec(
        DENSITY_BEHAVIOR_ID,
        "密度",
        ("常规", "密度"),
    ),
    MaterialBehaviorSpec(
        ELASTIC_BEHAVIOR_ID,
        "线弹性",
        ("力学", "弹性", "线弹性"),
        "linear_elastic",
    ),
    MaterialBehaviorSpec(
        PLASTIC_BEHAVIOR_ID,
        "塑性",
        ("力学", "塑性", "塑性"),
        "j2_plasticity",
    ),
    MaterialBehaviorSpec(
        NEO_HOOKEAN_BEHAVIOR_ID,
        "超弹性（Neo Hooke）",
        ("力学", "弹性", "超弹性", "Neo Hooke"),
        "neo_hookean",
    ),
)

_SPEC_BY_ID = {item.behavior_id: item for item in MATERIAL_BEHAVIOR_SPECS}


def material_behavior_specs() -> tuple[MaterialBehaviorSpec, ...]:
    """Return the supported behaviors in material-editor menu order."""

    return MATERIAL_BEHAVIOR_SPECS


def material_behavior_spec(behavior_id: str) -> MaterialBehaviorSpec:
    normalized = str(behavior_id).strip().casefold()
    try:
        return _SPEC_BY_ID[normalized]
    except KeyError as error:
        raise KeyError(f"unsupported material behavior {behavior_id!r}") from error


def material_behavior_ids(material: MaterialDefinition) -> tuple[str, ...]:
    """Return unique behavior IDs without inferring from arbitrary properties."""

    ids = tuple(item.behavior_id for item in material.behaviors)
    return tuple(dict.fromkeys(ids))


def material_behavior_diagnostics(
    material: MaterialDefinition | Sequence[MaterialBehavior],
) -> tuple[str, ...]:
    """Return user-facing composition errors for a material behavior set."""

    behaviors = (
        tuple(material.behaviors)
        if isinstance(material, MaterialDefinition)
        else tuple(material)
    )
    ids = {item.behavior_id for item in behaviors}
    diagnostics: list[str] = []
    unknown = sorted(ids - set(_SPEC_BY_ID))
    if unknown:
        diagnostics.append(f"包含未注册的材料行为：{', '.join(unknown)}")
    if len(ids) != len(behaviors):
        diagnostics.append("同一种材料行为不能重复定义")
    if PLASTIC_BEHAVIOR_ID in ids and ELASTIC_BEHAVIOR_ID not in ids:
        diagnostics.append("塑性行为需要同时定义线弹性行为")
    if NEO_HOOKEAN_BEHAVIOR_ID in ids and (
        ELASTIC_BEHAVIOR_ID in ids or PLASTIC_BEHAVIOR_ID in ids
    ):
        diagnostics.append("Neo-Hookean 超弹性不能与线弹性或塑性同时定义")
    return tuple(diagnostics)


def behavior_parameter_mapping(
    behaviors: Sequence[MaterialBehavior],
) -> dict[str, object]:
    """Flatten registered behavior parameters for the current solver boundary."""

    values: dict[str, object] = {}
    for behavior in behaviors:
        values.update(dict(behavior.parameters))
    return values


def behavior_summary(material: MaterialDefinition) -> str:
    """Render the compact behavior summary used by the material manager."""

    labels = [
        material_behavior_spec(item.behavior_id).label
        for item in material.behaviors
        if item.behavior_id in _SPEC_BY_ID
    ]
    if not labels:
        return "未定义"
    return "、".join(labels)


__all__ = [
    "DENSITY_BEHAVIOR_ID",
    "ELASTIC_BEHAVIOR_ID",
    "MATERIAL_BEHAVIOR_SPECS",
    "MaterialBehaviorSpec",
    "NEO_HOOKEAN_BEHAVIOR_ID",
    "PLASTIC_BEHAVIOR_ID",
    "behavior_parameter_mapping",
    "behavior_summary",
    "material_behavior_diagnostics",
    "material_behavior_ids",
    "material_behavior_spec",
    "material_behavior_specs",
]
