"""Registered analysis capabilities used by the analysis compiler.

The compiler owns composition, while this module owns the concrete factories
available for one procedure.  Adding a new element/kinematics/material
combination therefore extends the registry instead of adding another branch
to ``analysis.compiler``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fem.elements import canonical_element_type, get_element_definition
from fem.elements.contracts import ElementDefinition
from fem.materials import (
    FiniteStrainElasticMaterial,
    GreenLagrangeElasticMaterial,
    J2PlasticityMaterial,
    MaterialModel,
    NeoHookeanMaterial,
    SmallStrainElasticMaterial,
    SmallStrainJ2PlasticityMaterial,
    resolve_j2_algorithm,
)
from fem.model import GeometryMode
from fem.physics.contracts import PhysicsOperator
from fem.physics.mechanics import (
    ContinuumMechanicsOperator,
    TotalLagrangianKinematics,
)
from fem.physics.mechanics.kinematics import KinematicsModel
from fem.physics.mechanics.operators.plane import PlaneProperties


MaterialFactory = Callable[
    [Mapping[str, Any], Mapping[str, Any], int, str],
    MaterialModel,
]
DefinitionFactory = Callable[[], ElementDefinition]
KinematicsFactory = Callable[[Mapping[str, Any]], KinematicsModel]
OperatorFactory = Callable[
    [ElementDefinition, KinematicsModel, Mapping[str, Any]],
    PhysicsOperator,
]


@dataclass(frozen=True, slots=True)
class NonlinearStaticCapability:
    """Factory set for one element family in nonlinear static analysis."""

    canonical_type: str
    definition_factory: DefinitionFactory
    kinematics_factory: KinematicsFactory
    operator_factory: OperatorFactory
    material_factory: MaterialFactory

    def __post_init__(self) -> None:
        canonical = canonical_element_type(self.canonical_type)
        object.__setattr__(self, "canonical_type", canonical)

    def build_operator(self, options: Mapping[str, Any]) -> PhysicsOperator:
        definition = self.definition_factory()
        kinematics = self.kinematics_factory(options)
        operator = self.operator_factory(definition, kinematics, options)
        if not isinstance(operator, PhysicsOperator):
            raise TypeError(
                f"nonlinear capability {self.canonical_type!r} returned "
                "an invalid PhysicsOperator"
            )
        return operator

    def build_material(
        self,
        properties: Mapping[str, Any],
        options: Mapping[str, Any],
        *,
        element_id: int,
        element_type: str,
    ) -> MaterialModel:
        material = self.material_factory(
            properties,
            options,
            int(element_id),
            str(element_type),
        )
        if not isinstance(material, MaterialModel):
            raise TypeError(
                f"nonlinear capability {self.canonical_type!r} returned "
                "an invalid MaterialModel"
            )
        return material


class NonlinearStaticCapabilityRegistry:
    """Immutable lookup table for procedure-specific numerical capabilities."""

    def __init__(self, capabilities: tuple[NonlinearStaticCapability, ...]):
        entries = tuple(capabilities)
        if not entries:
            raise ValueError("nonlinear static capability registry must not be empty")
        by_type: dict[str, NonlinearStaticCapability] = {}
        for capability in entries:
            if type(capability) is not NonlinearStaticCapability:
                raise TypeError("capabilities must contain NonlinearStaticCapability")
            key = capability.canonical_type.casefold()
            if key in by_type:
                raise ValueError(
                    f"duplicate nonlinear capability {capability.canonical_type!r}"
                )
            by_type[key] = capability
        self._capabilities = entries
        self._by_type = by_type

    def resolve(self, element_type: str) -> NonlinearStaticCapability:
        canonical = canonical_element_type(element_type)
        try:
            return self._by_type[canonical.casefold()]
        except KeyError as exc:
            raise NotImplementedError(
                f"nonlinear static analysis does not support element type {element_type!r}"
            ) from exc

    def entries(self) -> tuple[NonlinearStaticCapability, ...]:
        return self._capabilities


def _total_lagrangian_kinematics(options: Mapping[str, Any]) -> KinematicsModel:
    mode = options.get("geometry_mode", GeometryMode.FINITE_STRAIN)
    if mode is GeometryMode.SMALL_STRAIN or str(mode).casefold() == GeometryMode.SMALL_STRAIN.value:
        from fem.physics.mechanics import SmallStrainKinematics

        return SmallStrainKinematics()
    return TotalLagrangianKinematics()


def _definition_factory(element_type: str) -> DefinitionFactory:
    """Return the one canonical definition lookup for an element type.

    The analysis registry selects a procedure capability; it does not own a
    second table of reference-element classes.  Geometry identity and
    interpolation remain owned by ``fem.elements.registry``.
    """

    canonical = canonical_element_type(element_type)

    def build() -> ElementDefinition:
        return get_element_definition(canonical)

    return build


def _continuum_operator(
    definition: ElementDefinition,
    kinematics: KinematicsModel,
    options: Mapping[str, Any],
) -> PhysicsOperator:
    del options
    return ContinuumMechanicsOperator(
        kinematics=kinematics,
        definition=definition,
    )


def _plane_material(
    properties: Mapping[str, Any],
    options: Mapping[str, Any],
    element_id: int,
    element_type: str,
) -> MaterialModel:
    try:
        constitutive_model = str(properties["constitutive_model"]).strip().casefold()
    except KeyError as exc:
        raise ValueError(
            "compiled material properties must declare constitutive_model"
        ) from exc
    if constitutive_model == "neo_hookean":
        try:
            C10 = float(properties["C10"])
            D1 = float(properties["D1"])
        except KeyError as exc:
            raise KeyError("plane Neo-Hookean material requires C10 and D1") from exc
        plane_type = properties.get(
            "plane_type",
            "strain"
            if str(element_type).strip().upper().startswith("CPE")
            else "stress",
        )
        if _is_small_strain(options):
            raise ValueError("Neo-Hookean material requires finite-strain kinematics")
        return NeoHookeanMaterial(C10, D1, plane_type=str(plane_type))
    plane = PlaneProperties.from_mapping(
        properties,
        element_id=element_id,
        element_type=element_type,
        label="plane section",
    )
    if constitutive_model == "linear_elastic":
        if _is_small_strain(options):
            return SmallStrainElasticMaterial(
                plane.E,
                plane.nu,
                plane.plane_type,
            )
        return GreenLagrangeElasticMaterial(
            plane.E,
            plane.nu,
            plane.plane_type,
        )
    if constitutive_model != "j2_plasticity":
        raise NotImplementedError(
            f"unsupported plane constitutive model {constitutive_model!r}"
        )
    if _is_small_strain(options):
        return SmallStrainJ2PlasticityMaterial(
            plane.E,
            plane.nu,
            float(properties["yield_stress"]),
            float(properties.get("hardening_modulus", 0.0)),
            plane_type=plane.plane_type,
        )
    algorithm = resolve_j2_algorithm(
        properties.get(
            "algorithm",
            options.get("material_algorithm", "hencky"),
        )
    )
    return J2PlasticityMaterial(
        plane.E,
        plane.nu,
        float(properties["yield_stress"]),
        float(properties.get("hardening_modulus", 0.0)),
        algorithm=algorithm,
    )


def _solid_material(
    properties: Mapping[str, Any],
    options: Mapping[str, Any],
    element_id: int,
    element_type: str,
) -> MaterialModel:
    del element_id, element_type
    try:
        constitutive_model = str(properties["constitutive_model"]).strip().casefold()
    except KeyError as exc:
        raise ValueError(
            "compiled material properties must declare constitutive_model"
        ) from exc
    if constitutive_model == "neo_hookean":
        try:
            C10 = float(properties["C10"])
            D1 = float(properties["D1"])
        except KeyError as exc:
            raise KeyError("solid Neo-Hookean material requires C10 and D1") from exc
        if _is_small_strain(options):
            raise ValueError("Neo-Hookean material requires finite-strain kinematics")
        return NeoHookeanMaterial(C10, D1)
    try:
        E = float(properties["E"])
        nu = float(properties["nu"])
    except KeyError as exc:
        raise KeyError("three-dimensional solid material requires E and nu") from exc
    if constitutive_model == "linear_elastic":
        if _is_small_strain(options):
            return SmallStrainElasticMaterial(E, nu)
        return FiniteStrainElasticMaterial(E, nu)
    if constitutive_model != "j2_plasticity":
        raise NotImplementedError(
            f"unsupported solid constitutive model {constitutive_model!r}"
        )
    if _is_small_strain(options):
        return SmallStrainJ2PlasticityMaterial(
            E,
            nu,
            float(properties["yield_stress"]),
            float(properties.get("hardening_modulus", 0.0)),
        )
    algorithm = resolve_j2_algorithm(
        properties.get(
            "algorithm",
            options.get("material_algorithm", "hencky"),
        )
    )
    return J2PlasticityMaterial(
        E,
        nu,
        float(properties["yield_stress"]),
        float(properties.get("hardening_modulus", 0.0)),
        algorithm=algorithm,
    )


DEFAULT_NONLINEAR_STATIC_CAPABILITIES = NonlinearStaticCapabilityRegistry(
    (
        NonlinearStaticCapability(
            canonical_type="Quad4",
            definition_factory=_definition_factory("Quad4"),
            kinematics_factory=_total_lagrangian_kinematics,
            operator_factory=_continuum_operator,
            material_factory=_plane_material,
        ),
        NonlinearStaticCapability(
            canonical_type="Tri3",
            definition_factory=_definition_factory("Tri3"),
            kinematics_factory=_total_lagrangian_kinematics,
            operator_factory=_continuum_operator,
            material_factory=_plane_material,
        ),
        NonlinearStaticCapability(
            canonical_type="Quad8",
            definition_factory=_definition_factory("Quad8"),
            kinematics_factory=_total_lagrangian_kinematics,
            operator_factory=_continuum_operator,
            material_factory=_plane_material,
        ),
        NonlinearStaticCapability(
            canonical_type="Tri6",
            definition_factory=_definition_factory("Tri6"),
            kinematics_factory=_total_lagrangian_kinematics,
            operator_factory=_continuum_operator,
            material_factory=_plane_material,
        ),
        NonlinearStaticCapability(
            canonical_type="Hex8",
            definition_factory=_definition_factory("Hex8"),
            kinematics_factory=_total_lagrangian_kinematics,
            operator_factory=_continuum_operator,
            material_factory=_solid_material,
        ),
        NonlinearStaticCapability(
            canonical_type="Hex20",
            definition_factory=_definition_factory("Hex20"),
            kinematics_factory=_total_lagrangian_kinematics,
            operator_factory=_continuum_operator,
            material_factory=_solid_material,
        ),
        NonlinearStaticCapability(
            canonical_type="Tet4",
            definition_factory=_definition_factory("Tet4"),
            kinematics_factory=_total_lagrangian_kinematics,
            operator_factory=_continuum_operator,
            material_factory=_solid_material,
        ),
        NonlinearStaticCapability(
            canonical_type="Tet10",
            definition_factory=_definition_factory("Tet10"),
            kinematics_factory=_total_lagrangian_kinematics,
            operator_factory=_continuum_operator,
            material_factory=_solid_material,
        ),
    )
)


def nonlinear_static_capability_for(
    element_type: str,
) -> NonlinearStaticCapability:
    """Resolve one element type from the default nonlinear registry."""

    return DEFAULT_NONLINEAR_STATIC_CAPABILITIES.resolve(element_type)


def nonlinear_static_capabilities() -> tuple[NonlinearStaticCapability, ...]:
    """Return all registered nonlinear static capabilities."""

    return DEFAULT_NONLINEAR_STATIC_CAPABILITIES.entries()


def _is_small_strain(options: Mapping[str, Any]) -> bool:
    mode = options.get("geometry_mode", GeometryMode.FINITE_STRAIN)
    return mode is GeometryMode.SMALL_STRAIN or str(mode).casefold() == (
        GeometryMode.SMALL_STRAIN.value
    )


__all__ = [
    "DEFAULT_NONLINEAR_STATIC_CAPABILITIES",
    "NonlinearStaticCapability",
    "NonlinearStaticCapabilityRegistry",
    "nonlinear_static_capabilities",
    "nonlinear_static_capability_for",
]
