"""Elastic material models for the v2 core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import linear_elastic
from .contracts import (
    KinematicMeasure,
    MaterialPointInput,
    MaterialResponse,
    StressMeasure,
)


@dataclass(frozen=True, slots=True)
class SmallStrainElasticMaterial:
    """Isotropic elastic response in the infinitesimal-strain measure."""

    E: float
    nu: float
    plane_type: str | None = None

    def __post_init__(self) -> None:
        E = float(self.E)
        nu = float(self.nu)
        if not np.isfinite(E) or E <= 0.0:
            raise ValueError("E must be finite and > 0")
        if not np.isfinite(nu) or not -1.0 < nu < 0.5:
            raise ValueError("nu must be finite and in (-1, 0.5)")
        plane_type = self.plane_type
        if plane_type is not None:
            plane_type = str(plane_type).casefold()
            if not (
                plane_type.startswith("stress")
                or plane_type.startswith("strain")
            ):
                raise ValueError("plane_type must be stress or strain")
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "plane_type", plane_type)

    def initial_state(self) -> dict[str, object]:
        return {}

    def evaluate(self, point: MaterialPointInput) -> MaterialResponse:
        strain = _small_strain_tensor(point.kinematics)
        need_tangent = bool(
            point.context.parameters.get("_fem_need_tangent", True)
        )
        capture_outputs = bool(
            point.context.parameters.get("_fem_capture_outputs", True)
        )
        if strain.shape == (2, 2):
            if self.plane_type is None:
                raise ValueError("2D small-strain material requires plane_type")
            constitutive = linear_elastic.plane_matrix(
                self.E,
                self.nu,
                self.plane_type,
            )
            strain_vector = np.array(
                [strain[0, 0], strain[1, 1], 2.0 * strain[0, 1]],
                dtype=float,
            )
            stress_vector = constitutive @ strain_vector
            stress = np.zeros((3, 3), dtype=float)
            stress[0, 0], stress[1, 1], stress[0, 1] = (
                stress_vector[0],
                stress_vector[1],
                stress_vector[2],
            )
            stress[1, 0] = stress[0, 1]
            if self.plane_type.startswith("strain"):
                lame = self.E * self.nu / (
                    (1.0 + self.nu) * (1.0 - 2.0 * self.nu)
                )
                stress[2, 2] = lame * (strain[0, 0] + strain[1, 1])
            tangent = np.zeros((6, 6), dtype=float)
            active = (0, 1, 3)
            tangent[np.ix_(active, active)] = constitutive
        elif strain.shape == (3, 3):
            constitutive = linear_elastic.solid_3d_matrix(self.E, self.nu)
            stress_vector = constitutive @ _engineering_strain(strain)
            stress = _stress_from_engineering(stress_vector)
            tangent = constitutive
        else:
            raise ValueError("small strain must be a 2x2 or 3x3 tensor")
        if not need_tangent:
            tangent = np.zeros_like(tangent)
        return MaterialResponse(
            stress=stress,
            tangent=tangent,
            stress_measure=StressMeasure.CAUCHY,
            tangent_input=KinematicMeasure.SMALL_STRAIN,
            outputs=(
                {
                    "cauchy_stress": stress,
                    "return_algorithm": "elastic",
                }
                if capture_outputs
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class GreenLagrangeElasticMaterial:
    """Isotropic elastic response in Green--Lagrange/second-Piola form."""

    E: float
    nu: float
    plane_type: str = "stress"

    def __post_init__(self) -> None:
        E = float(self.E)
        nu = float(self.nu)
        if not np.isfinite(E) or E <= 0.0:
            raise ValueError("E must be finite and > 0")
        if not np.isfinite(nu) or not -1.0 < nu < 0.5:
            raise ValueError("nu must be finite and in (-1, 0.5)")
        normalized = str(self.plane_type).casefold()
        if not (normalized.startswith("stress") or normalized.startswith("strain")):
            raise ValueError("plane_type must be stress or strain")
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "plane_type", normalized)

    def initial_state(self) -> dict[str, object]:
        return {}

    def evaluate(self, point: MaterialPointInput) -> MaterialResponse:
        kinematics = point.kinematics
        need_tangent = bool(
            point.context.parameters.get("_fem_need_tangent", True)
        )
        capture_outputs = bool(
            point.context.parameters.get("_fem_capture_outputs", True)
        )
        deformation_gradient = np.asarray(
            kinematics.deformation_gradient,
            dtype=float,
        )
        if deformation_gradient.shape != (2, 2):
            raise ValueError("GreenLagrangeElasticMaterial currently requires a 2D strain")
        strain = 0.5 * (
            deformation_gradient.T @ deformation_gradient - np.eye(2)
        )
        tangent = linear_elastic.plane_matrix(self.E, self.nu, self.plane_type)
        strain_vector = np.array(
            [strain[0, 0], strain[1, 1], 2.0 * strain[0, 1]],
            dtype=float,
        )
        stress_vector = tangent @ strain_vector
        second_piola = np.array(
            [[stress_vector[0], stress_vector[2]], [stress_vector[2], stress_vector[1]]],
            dtype=float,
        )
        first_piola = deformation_gradient @ second_piola
        first_piola_3d = np.zeros((3, 3), dtype=float)
        first_piola_3d[:2, :2] = first_piola
        second_piola_3d = np.zeros((3, 3), dtype=float)
        second_piola_3d[:2, :2] = second_piola
        if self.plane_type.startswith("strain"):
            lame = self.E * self.nu / (
                (1.0 + self.nu) * (1.0 - 2.0 * self.nu)
            )
            second_piola_3d[2, 2] = lame * float(strain[0, 0] + strain[1, 1])
            first_piola_3d[2, 2] = second_piola_3d[2, 2]

        tangent_3d = (
            _total_lagrangian_tangent(
                deformation_gradient,
                second_piola,
                tangent,
            )
            if need_tangent
            else np.zeros((9, 9), dtype=float)
        )
        kirchhoff = first_piola_3d @ _embed_2d(deformation_gradient).T
        return MaterialResponse(
            stress=first_piola_3d,
            tangent=tangent_3d,
            stress_measure=StressMeasure.FIRST_PIOLA,
            tangent_input=KinematicMeasure.DEFORMATION_GRADIENT,
            outputs=(
                {
                    "second_piola_stress": second_piola_3d,
                    "kirchhoff_stress": kirchhoff,
                    "return_algorithm": "elastic",
                }
                if capture_outputs
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class FiniteStrainElasticMaterial:
    """Three-dimensional isotropic Green--Lagrange elastic material."""

    E: float
    nu: float

    def __post_init__(self) -> None:
        E = float(self.E)
        nu = float(self.nu)
        if not np.isfinite(E) or E <= 0.0:
            raise ValueError("E must be finite and > 0")
        if not np.isfinite(nu) or not -1.0 < nu < 0.5:
            raise ValueError("nu must be finite and in (-1, 0.5)")
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "nu", nu)

    def initial_state(self) -> dict[str, object]:
        return {}

    def evaluate(self, point: MaterialPointInput) -> MaterialResponse:
        deformation_gradient = np.asarray(
            point.kinematics.deformation_gradient,
            dtype=float,
        )
        need_tangent = bool(
            point.context.parameters.get("_fem_need_tangent", True)
        )
        capture_outputs = bool(
            point.context.parameters.get("_fem_capture_outputs", True)
        )
        if deformation_gradient.shape != (3, 3):
            raise ValueError(
                "FiniteStrainElasticMaterial requires a 3D deformation gradient"
            )
        green_lagrange = 0.5 * (
            deformation_gradient.T @ deformation_gradient - np.eye(3)
        )
        lame = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        shear = self.E / (2.0 * (1.0 + self.nu))
        second_piola = (
            lame * np.trace(green_lagrange) * np.eye(3)
            + 2.0 * shear * green_lagrange
        )
        first_piola = deformation_gradient @ second_piola
        tangent = (
            _three_dimensional_tangent(
                deformation_gradient,
                second_piola,
                lame,
                shear,
            )
            if need_tangent
            else np.zeros((9, 9), dtype=float)
        )
        kirchhoff = first_piola @ deformation_gradient.T
        return MaterialResponse(
            stress=first_piola,
            tangent=tangent,
            stress_measure=StressMeasure.FIRST_PIOLA,
            tangent_input=KinematicMeasure.DEFORMATION_GRADIENT,
            outputs=(
                {
                    "second_piola_stress": second_piola,
                    "kirchhoff_stress": kirchhoff,
                    "return_algorithm": "elastic",
                }
                if capture_outputs
                else {}
            ),
        )


def _embed_2d(value: np.ndarray) -> np.ndarray:
    result = np.eye(3, dtype=float)
    result[:2, :2] = value
    return result


def _total_lagrangian_tangent(
    deformation_gradient: np.ndarray,
    second_piola: np.ndarray,
    constitutive: np.ndarray,
) -> np.ndarray:
    """Build ``dP/dF`` from the plane ``dS/dE`` matrix."""

    tangent = np.zeros((9, 9), dtype=float)
    for row_component in range(2):
        for row_direction in range(2):
            delta_f = np.zeros((2, 2), dtype=float)
            delta_f[row_component, row_direction] = 1.0
            delta_e = 0.5 * (
                delta_f.T @ deformation_gradient
                + deformation_gradient.T @ delta_f
            )
            delta_e_vector = np.array(
                [delta_e[0, 0], delta_e[1, 1], 2.0 * delta_e[0, 1]],
                dtype=float,
            )
            delta_s_vector = constitutive @ delta_e_vector
            delta_s = np.array(
                [
                    [delta_s_vector[0], delta_s_vector[2]],
                    [delta_s_vector[2], delta_s_vector[1]],
                ],
                dtype=float,
            )
            delta_p = delta_f @ second_piola + deformation_gradient @ delta_s
            column = 3 * row_component + row_direction
            for output_component in range(2):
                for output_direction in range(2):
                    row = 3 * output_component + output_direction
                    tangent[row, column] = delta_p[output_component, output_direction]
    return tangent


def _three_dimensional_tangent(
    deformation_gradient: np.ndarray,
    second_piola: np.ndarray,
    lame: float,
    shear: float,
) -> np.ndarray:
    """Return the material/geometric ``dP/dF`` tangent for 3D elasticity."""

    tangent = np.zeros((9, 9), dtype=float)
    for row_component in range(3):
        for row_direction in range(3):
            delta_f = np.zeros((3, 3), dtype=float)
            delta_f[row_component, row_direction] = 1.0
            delta_e = 0.5 * (
                delta_f.T @ deformation_gradient
                + deformation_gradient.T @ delta_f
            )
            delta_s = (
                lame * np.trace(delta_e) * np.eye(3)
                + 2.0 * shear * delta_e
            )
            delta_p = delta_f @ second_piola + deformation_gradient @ delta_s
            column = 3 * row_component + row_direction
            tangent[:, column] = delta_p.reshape(9)
    return tangent


def _small_strain_tensor(kinematics: object) -> np.ndarray:
    value = getattr(kinematics, "small_strain", None)
    if value is None:
        raise ValueError("small-strain material requires kinematics.small_strain")
    strain = np.asarray(value, dtype=float)
    if strain.shape not in {(2, 2), (3, 3)}:
        raise ValueError("small_strain must be a 2x2 or 3x3 tensor")
    return strain


def _engineering_strain(strain: np.ndarray) -> np.ndarray:
    return np.array(
        [
            strain[0, 0],
            strain[1, 1],
            strain[2, 2],
            2.0 * strain[0, 1],
            2.0 * strain[0, 2],
            2.0 * strain[1, 2],
        ],
        dtype=float,
    )


def _stress_from_engineering(stress: np.ndarray) -> np.ndarray:
    vector = np.asarray(stress, dtype=float)
    if vector.shape != (6,):
        raise ValueError("engineering stress must have shape (6,)")
    return np.array(
        [
            [vector[0], vector[3], vector[4]],
            [vector[3], vector[1], vector[5]],
            [vector[4], vector[5], vector[2]],
        ],
        dtype=float,
    )


__all__ = [
    "FiniteStrainElasticMaterial",
    "GreenLagrangeElasticMaterial",
    "SmallStrainElasticMaterial",
]
