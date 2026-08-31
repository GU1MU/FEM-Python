"""Material models and constitutive response contracts."""

from __future__ import annotations

from . import linear_elastic
from .contracts import (
    KinematicMeasure,
    MaterialModel,
    MaterialPointInput,
    MaterialResponse,
    StressMeasure,
)
from .elastic import (
    FiniteStrainElasticMaterial,
    GreenLagrangeElasticMaterial,
    SmallStrainElasticMaterial,
)
from .small_strain_j2 import SmallStrainJ2PlasticityMaterial
from .hyperelastic import NeoHookeanMaterial
from .finite_strain_j2 import (
    J2_ALGORITHMS,
    J2Algorithm,
    J2PlasticityMaterial,
    resolve_j2_algorithm,
)
from .behavior_registry import (
    DENSITY_BEHAVIOR_ID,
    ELASTIC_BEHAVIOR_ID,
    MATERIAL_BEHAVIOR_SPECS,
    NEO_HOOKEAN_BEHAVIOR_ID,
    PLASTIC_BEHAVIOR_ID,
    MaterialBehaviorSpec,
    behavior_parameter_mapping,
    behavior_summary,
    material_behavior_diagnostics,
    material_behavior_ids,
    material_behavior_spec,
    material_behavior_specs,
)

__all__ = [
    "GreenLagrangeElasticMaterial",
    "FiniteStrainElasticMaterial",
    "SmallStrainElasticMaterial",
    "SmallStrainJ2PlasticityMaterial",
    "NeoHookeanMaterial",
    "J2Algorithm",
    "J2_ALGORITHMS",
    "J2PlasticityMaterial",
    "KinematicMeasure",
    "MaterialModel",
    "MaterialPointInput",
    "MaterialResponse",
    "StressMeasure",
    "linear_elastic",
    "resolve_j2_algorithm",
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
