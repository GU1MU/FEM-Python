"""Stateless finite-strain J2 material models for the v2 core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .contracts import (
    KinematicMeasure,
    MaterialPointInput,
    MaterialResponse,
    StressMeasure,
)


_ZERO_TENSOR_3 = np.zeros((3, 3), dtype=float)
_ZERO_TENSOR_3.flags.writeable = False
_IDENTITY_TENSOR_3 = np.eye(3, dtype=float)
_IDENTITY_TENSOR_3.flags.writeable = False
from .j2_algorithms import (
    _analytic_elastic_dP_dF,
    _analytic_hencky_dP_dF,
    _analytic_plastic_dP_dF,
    _hencky_j2_response_batch,
    _hencky_j2_response,
    _implicit_joint_dP_dF,
    _multiplicative_j2_response,
)


J2Algorithm = Literal["hencky", "multiplicative"]
J2_ALGORITHMS: tuple[J2Algorithm, ...] = ("hencky", "multiplicative")


def resolve_j2_algorithm(value: object) -> J2Algorithm:
    """Normalize the internal return-mapping algorithm selection."""

    if not isinstance(value, str):
        raise ValueError("J2 algorithm must be 'hencky' or 'multiplicative'")
    normalized = value.strip().casefold()
    # ``formal_j2`` is accepted only while old project files are migrated.
    if normalized == "formal_j2":
        normalized = "multiplicative"
    if normalized not in J2_ALGORITHMS:
        raise ValueError("J2 algorithm must be 'hencky' or 'multiplicative'")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class J2PlasticityMaterial:
    """J2 plasticity response with history supplied by ``StateManager``."""

    E: float
    nu: float
    yield_stress: float
    hardening_modulus: float = 0.0
    algorithm: J2Algorithm = "hencky"

    def __post_init__(self) -> None:
        E = float(self.E)
        nu = float(self.nu)
        yield_stress = float(self.yield_stress)
        hardening = float(self.hardening_modulus)
        if not np.isfinite(E) or E <= 0.0:
            raise ValueError("E must be finite and > 0")
        if not np.isfinite(nu) or not -1.0 < nu < 0.5:
            raise ValueError("nu must be finite and in (-1, 0.5)")
        if not np.isfinite(yield_stress) or yield_stress < 0.0:
            raise ValueError("yield_stress must be finite and >= 0")
        if not np.isfinite(hardening) or hardening < 0.0:
            raise ValueError("hardening_modulus must be finite and >= 0")
        algorithm = resolve_j2_algorithm(self.algorithm)
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "yield_stress", yield_stress)
        object.__setattr__(self, "hardening_modulus", hardening)
        object.__setattr__(self, "algorithm", algorithm)

    def initial_state(self) -> dict[str, object]:
        if self.algorithm == "hencky":
            return {
                "plastic_log_strain": np.zeros((3, 3), dtype=float),
                "equivalent_plastic_strain": 0.0,
            }
        return {
            "plastic_deformation_gradient": np.eye(3, dtype=float),
            "equivalent_plastic_strain": 0.0,
        }

    def evaluate_batch(
        self,
        deformation_gradients: np.ndarray,
        committed_plastic_log_strains: np.ndarray,
        committed_equivalent_plastic_strains: np.ndarray,
        *,
        need_tangent: bool,
        tangent_columns: tuple[int, ...] | None = None,
    ):
        """Evaluate a compatible Hencky point block with one array kernel."""

        if self.algorithm != "hencky":
            raise NotImplementedError(
                "batched finite-strain evaluation currently supports Hencky J2 only"
            )
        return _hencky_j2_response_batch(
            deformation_gradients,
            committed_plastic_log_strains,
            committed_equivalent_plastic_strains,
            self.E,
            self.nu,
            self.yield_stress,
            self.hardening_modulus,
            need_tangent=need_tangent,
            tangent_columns=tangent_columns,
        )

    def evaluate(self, point: MaterialPointInput) -> MaterialResponse:
        F = _embedded_deformation_gradient(point.kinematics.deformation_gradient)
        state = point.committed_state
        alpha = float(state.get("equivalent_plastic_strain", 0.0))
        need_tangent = bool(
            point.context.parameters.get("_fem_need_tangent", True)
        )
        capture_outputs = bool(
            point.context.parameters.get("_fem_capture_outputs", True)
        )
        if self.algorithm == "hencky":
            plastic_log_strain_value = state.get("plastic_log_strain")
            if plastic_log_strain_value is None:
                plastic_log_strain_value = _ZERO_TENSOR_3
            plastic_log_strain = np.asarray(plastic_log_strain_value, dtype=float)
            response = _hencky_j2_response(
                F,
                plastic_log_strain,
                alpha,
                self.E,
                self.nu,
                self.yield_stress,
                self.hardening_modulus,
            )
            tangent = (
                _analytic_hencky_dP_dF(
                    F,
                    plastic_log_strain,
                    alpha,
                    self.E,
                    self.nu,
                    self.yield_stress,
                    self.hardening_modulus,
                    response=response,
                )
                if need_tangent
                else np.zeros((9, 9), dtype=float)
            )
            trial_state = {
                "plastic_log_strain": response.plastic_log_strain,
                "equivalent_plastic_strain": response.equivalent_plastic_strain,
            }
            outputs = (
                {
                    "second_piola_stress": response.second_piola_stress,
                    "kirchhoff_stress": response.kirchhoff_stress,
                    "return_algorithm": (
                        "elastic"
                        if response.equivalent_plastic_strain <= alpha + 1.0e-14
                        else "radial_return"
                    ),
                    "tangent_algorithm": "analytic_objective",
                }
                if capture_outputs
                else {}
            )
            return MaterialResponse(
                stress=response.first_piola_stress,
                tangent=tangent,
                stress_measure=StressMeasure.FIRST_PIOLA,
                tangent_input=KinematicMeasure.DEFORMATION_GRADIENT,
                trial_state=trial_state,
                outputs=outputs,
            )

        plastic_gradient_value = state.get("plastic_deformation_gradient")
        if plastic_gradient_value is None:
            plastic_gradient_value = _IDENTITY_TENSOR_3
        plastic_gradient = np.asarray(plastic_gradient_value, dtype=float)
        response = _multiplicative_j2_response(
            F,
            plastic_gradient,
            alpha,
            self.E,
            self.nu,
            self.yield_stress,
            self.hardening_modulus,
        )
        if not need_tangent:
            tangent = np.zeros((9, 9), dtype=float)
        elif response.plastic_increment <= 1.0e-14:
            tangent = _analytic_elastic_dP_dF(F, plastic_gradient, self.E, self.nu)
        elif response.iterated_direction:
            tangent = _implicit_joint_dP_dF(
                F,
                plastic_gradient,
                alpha,
                self.E,
                self.nu,
                self.yield_stress,
                self.hardening_modulus,
                response.plastic_increment_tensor,
            )
        else:
            tangent = _analytic_plastic_dP_dF(
                F,
                plastic_gradient,
                alpha,
                self.E,
                self.nu,
                self.yield_stress,
                self.hardening_modulus,
            )
        outputs = (
            {
                "second_piola_stress": response.second_piola_stress,
                "kirchhoff_stress": response.kirchhoff_stress,
                "return_algorithm": (
                    "elastic"
                    if response.plastic_increment <= 1.0e-14
                    else (
                        "joint_return"
                        if response.iterated_direction
                        else "fixed_return"
                    )
                ),
                "tangent_algorithm": (
                    "analytic_elastic"
                    if response.plastic_increment <= 1.0e-14
                    else (
                        "implicit_joint_return"
                        if response.iterated_direction
                        else "analytic_fixed_direction"
                    )
                ),
            }
            if capture_outputs
            else {}
        )
        return MaterialResponse(
            stress=response.first_piola_stress,
            tangent=tangent,
            stress_measure=StressMeasure.FIRST_PIOLA,
            tangent_input=KinematicMeasure.DEFORMATION_GRADIENT,
            trial_state={
                "plastic_deformation_gradient": response.plastic_deformation_gradient,
                "equivalent_plastic_strain": response.equivalent_plastic_strain,
            },
            outputs=outputs,
        )


def _embedded_deformation_gradient(value: np.ndarray) -> np.ndarray:
    F = np.asarray(value, dtype=float)
    if F.shape == (3, 3):
        return np.array(F, dtype=float, copy=True)
    if F.shape != (2, 2):
        raise ValueError("finite-strain J2 requires a 2x2 or 3x3 deformation gradient")
    embedded = np.eye(3, dtype=float)
    embedded[:2, :2] = F
    return embedded


__all__ = [
    "J2_ALGORITHMS",
    "J2Algorithm",
    "J2PlasticityMaterial",
    "resolve_j2_algorithm",
]
