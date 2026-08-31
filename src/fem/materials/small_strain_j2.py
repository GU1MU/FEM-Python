"""Infinitesimal-strain J2 plasticity with transactional return mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import (
    KinematicMeasure,
    MaterialPointInput,
    MaterialResponse,
    StressMeasure,
)


_ZERO_TENSOR_3 = np.zeros((3, 3), dtype=float)
_ZERO_TENSOR_3.flags.writeable = False
from .elastic import _engineering_strain, _stress_from_engineering
from . import linear_elastic


@dataclass(frozen=True, slots=True)
class SmallStrainJ2PlasticityMaterial:
    """Associative von-Mises plasticity in the small-strain configuration.

    The material stores only committed plastic strain and equivalent plastic
    strain.  Every evaluation is a pure trial computation; the assembler's
    state manager decides when the returned state is committed.
    """

    E: float
    nu: float
    yield_stress: float
    hardening_modulus: float = 0.0
    plane_type: str | None = None
    algorithm: str = "radial_return"

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
        plane_type = self.plane_type
        if plane_type is not None:
            plane_type = str(plane_type).casefold()
            if not (
                plane_type.startswith("stress")
                or plane_type.startswith("strain")
            ):
                raise ValueError("plane_type must be stress or strain")
        algorithm = str(self.algorithm).strip().casefold()
        if algorithm != "radial_return":
            raise ValueError("small-strain J2 algorithm must be radial_return")
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "yield_stress", yield_stress)
        object.__setattr__(self, "hardening_modulus", hardening)
        object.__setattr__(self, "plane_type", plane_type)
        object.__setattr__(self, "algorithm", algorithm)

    def initial_state(self) -> dict[str, object]:
        return {
            "plastic_strain": np.zeros((3, 3), dtype=float),
            "equivalent_plastic_strain": 0.0,
        }

    def evaluate(self, point: MaterialPointInput) -> MaterialResponse:
        strain = _input_small_strain(point.kinematics)
        committed = point.committed_state
        update = _response_for_strain(
            strain,
            committed,
            E=self.E,
            nu=self.nu,
            yield_stress=self.yield_stress,
            hardening_modulus=self.hardening_modulus,
            plane_type=self.plane_type,
        )
        need_tangent = bool(
            point.context.parameters.get("_fem_need_tangent", True)
        )
        capture_outputs = bool(
            point.context.parameters.get("_fem_capture_outputs", True)
        )
        tangent = (
            _consistent_tangent(
                strain,
                committed,
                E=self.E,
                nu=self.nu,
                yield_stress=self.yield_stress,
                hardening_modulus=self.hardening_modulus,
                plane_type=self.plane_type,
            )
            if need_tangent
            else np.zeros((6, 6), dtype=float)
        )
        algorithm = "elastic" if update.plastic_increment <= 1.0e-14 else self.algorithm
        return MaterialResponse(
            stress=update.stress,
            tangent=tangent,
            stress_measure=StressMeasure.CAUCHY,
            tangent_input=KinematicMeasure.SMALL_STRAIN,
            trial_state={
                "plastic_strain": update.plastic_strain,
                "equivalent_plastic_strain": update.equivalent_plastic_strain,
            },
            outputs=(
                {
                    "cauchy_stress": update.stress,
                    "small_strain": strain,
                    "equivalent_plastic_strain": update.equivalent_plastic_strain,
                    "return_algorithm": algorithm,
                    "tangent_algorithm": (
                        "numerical_plane_stress_return"
                        if strain.shape == (2, 2)
                        and self.plane_type is not None
                        and self.plane_type.startswith("stress")
                        else "analytic_radial_return"
                    ),
                }
                if capture_outputs
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class _ReturnUpdate:
    stress: np.ndarray
    plastic_strain: np.ndarray
    equivalent_plastic_strain: float
    plastic_increment: float


def _response_for_strain(
    strain: np.ndarray,
    committed: Any,
    *,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
    plane_type: str | None,
) -> _ReturnUpdate:
    if strain.shape == (3, 3):
        return _radial_return(
            strain,
            committed,
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
        )
    if strain.shape != (2, 2) or plane_type is None:
        raise ValueError("2D small-strain J2 requires plane_type")
    base = np.zeros((3, 3), dtype=float)
    base[:2, :2] = strain
    if plane_type.startswith("strain"):
        return _radial_return(
            base,
            committed,
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
        )

    # Plane stress is a constrained three-dimensional return problem.  Solve
    # the out-of-plane strain against sigma_33 while keeping the committed
    # state fixed for every Newton trial.
    e33 = -nu / max(1.0 - nu, 1.0e-12) * float(strain[0, 0] + strain[1, 1])
    update = None
    for _ in range(30):
        base[2, 2] = e33
        update = _radial_return(
            base,
            committed,
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
        )
        residual = float(update.stress[2, 2])
        scale = max(1.0, abs(update.stress[0, 0]), abs(update.stress[1, 1]))
        if abs(residual) <= 1.0e-11 * scale:
            break
        step = 1.0e-8 * max(1.0, abs(e33))
        base[2, 2] = e33 + step
        plus = _radial_return(
            base,
            committed,
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
        )
        base[2, 2] = e33 - step
        minus = _radial_return(
            base,
            committed,
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
        )
        derivative = float(plus.stress[2, 2] - minus.stress[2, 2]) / (2.0 * step)
        if abs(derivative) <= 1.0e-14:
            raise ValueError("plane-stress J2 return mapping has a zero sigma33 tangent")
        e33 -= residual / derivative
    else:
        raise ValueError("plane-stress J2 return mapping did not converge")
    if update is None:
        raise RuntimeError("plane-stress J2 return mapping produced no state")
    return update


def _radial_return(
    strain: np.ndarray,
    committed: Any,
    *,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
) -> _ReturnUpdate:
    full_strain = np.asarray(strain, dtype=float)
    if full_strain.shape != (3, 3):
        raise ValueError("radial return requires a 3x3 strain tensor")
    plastic_value = committed.get("plastic_strain")
    if plastic_value is None:
        plastic_value = _ZERO_TENSOR_3
    plastic = np.asarray(plastic_value, dtype=float)
    if plastic.shape != (3, 3) or not np.all(np.isfinite(plastic)):
        raise ValueError("committed plastic_strain must be a finite 3x3 tensor")
    alpha = float(committed.get("equivalent_plastic_strain", 0.0))
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("committed equivalent_plastic_strain must be finite and >= 0")
    tangent = linear_elastic.solid_3d_matrix(E, nu)
    trial_stress = _stress_from_engineering(
        tangent @ (_engineering_strain(full_strain) - _engineering_strain(plastic))
    )
    mean = float(np.trace(trial_stress) / 3.0)
    deviatoric = trial_stress - mean * np.eye(3)
    shear = E / (2.0 * (1.0 + nu))
    equivalent = float(np.sqrt(max(0.0, 1.5 * np.sum(deviatoric * deviatoric))))
    yield_limit = yield_stress + hardening_modulus * alpha
    yield_function = equivalent - yield_limit
    if yield_function <= 1.0e-12 * max(1.0, yield_limit):
        return _ReturnUpdate(
            stress=trial_stress,
            plastic_strain=np.array(plastic, copy=True),
            equivalent_plastic_strain=alpha,
            plastic_increment=0.0,
        )
    if equivalent <= 1.0e-14:
        raise ValueError("J2 return mapping found a plastic state with zero deviatoric stress")
    direction = 1.5 * deviatoric / equivalent
    increment = yield_function / (3.0 * shear + hardening_modulus)
    updated_plastic = plastic + increment * direction
    stress = trial_stress - 2.0 * shear * increment * direction
    return _ReturnUpdate(
        stress=stress,
        plastic_strain=updated_plastic,
        equivalent_plastic_strain=alpha + increment,
        plastic_increment=increment,
    )


def _numerical_tangent(
    strain: np.ndarray,
    committed: Any,
    *,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
    plane_type: str | None,
) -> np.ndarray:
    active = (0, 1, 3) if strain.shape == (2, 2) else tuple(range(6))
    baseline = _engineering_input(strain)
    tangent = np.zeros((6, 6), dtype=float)
    for column in active:
        step = 1.0e-7 * max(1.0, abs(float(baseline[active.index(column)])))
        plus_values = baseline.copy()
        minus_values = baseline.copy()
        local_index = active.index(column)
        plus_values[local_index] += step
        minus_values[local_index] -= step
        plus = _response_for_strain(
            _strain_from_engineering_input(plus_values, strain.shape),
            committed,
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
            plane_type=plane_type,
        )
        minus = _response_for_strain(
            _strain_from_engineering_input(minus_values, strain.shape),
            committed,
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
            plane_type=plane_type,
        )
        stress_plus = _engineering_stress(plus.stress)
        stress_minus = _engineering_stress(minus.stress)
        tangent[:, column] = (stress_plus - stress_minus) / (2.0 * step)
    return tangent


def _consistent_tangent(
    strain: np.ndarray,
    committed: Any,
    *,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
    plane_type: str | None,
) -> np.ndarray:
    """Return the radial-return tangent without perturbing every component.

    Plane stress keeps the existing constrained numerical tangent because its
    out-of-plane strain is itself solved by a local scalar iteration.  The
    3-D and plane-strain paths use the exact derivative of the same radial
    return mapping and therefore preserve the constitutive response.
    """

    if strain.shape == (2, 2) and plane_type is not None and plane_type.startswith(
        "stress"
    ):
        return _numerical_tangent(
            strain,
            committed,
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
            plane_type=plane_type,
        )

    full_strain = np.zeros((3, 3), dtype=float)
    if strain.shape == (2, 2):
        full_strain[:2, :2] = strain
    elif strain.shape == (3, 3):
        full_strain[:, :] = strain
    else:
        raise ValueError("small-strain tangent requires a 2x2 or 3x3 strain")
    full = _radial_return(
        full_strain,
        committed,
        E=E,
        nu=nu,
        yield_stress=yield_stress,
        hardening_modulus=hardening_modulus,
    )
    full_tangent = _radial_return_tangent(
        full_strain,
        committed,
        full,
        E=E,
        nu=nu,
        yield_stress=yield_stress,
        hardening_modulus=hardening_modulus,
    )
    if strain.shape == (3, 3):
        return full_tangent
    tangent = np.zeros((6, 6), dtype=float)
    active = (0, 1, 3)
    tangent[:, active] = full_tangent[:, active]
    return tangent


def _radial_return_tangent(
    strain: np.ndarray,
    committed: Any,
    update: _ReturnUpdate,
    *,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
) -> np.ndarray:
    """Differentiate the same fixed-committed-state radial return update."""

    elastic = linear_elastic.solid_3d_matrix(E, nu)
    if update.plastic_increment <= 1.0e-14:
        return elastic
    plastic_value = committed.get("plastic_strain")
    plastic = (
        _ZERO_TENSOR_3
        if plastic_value is None
        else np.asarray(plastic_value, dtype=float)
    )
    alpha = float(committed.get("equivalent_plastic_strain", 0.0))
    trial_stress = _stress_from_engineering(
        elastic @ (_engineering_strain(strain) - _engineering_strain(plastic))
    )
    trial_deviator = _deviatoric(trial_stress)
    equivalent = float(np.sqrt(1.5 * np.sum(trial_deviator * trial_deviator)))
    if equivalent <= 1.0e-14:
        return _numerical_tangent(
            _engineering_to_tensor(_engineering_strain(strain)),
            committed,
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
            plane_type=None,
        )
    shear = E / (2.0 * (1.0 + nu))
    direction = 1.5 * trial_deviator / equivalent

    derivative_basis = np.eye(6, dtype=float)
    derivative_trial = np.asarray(
        [
            _stress_from_engineering(elastic @ derivative_basis[column])
            for column in range(6)
        ],
        dtype=float,
    )
    derivative_trial_deviator = derivative_trial - (
        np.trace(derivative_trial, axis1=1, axis2=2)[:, None, None] / 3.0
    ) * np.eye(3)
    derivative_equivalent = (
        1.5
        * np.einsum("ij,cij->c", trial_deviator, derivative_trial_deviator)
        / equivalent
    )
    derivative_direction = 1.5 * (
        derivative_trial_deviator / equivalent
        - trial_deviator[None, :, :]
        * derivative_equivalent[:, None, None]
        / equivalent**2
    )
    derivative_gamma = derivative_equivalent / (3.0 * shear + hardening_modulus)
    derivative_stress = derivative_trial - 2.0 * shear * (
        derivative_gamma[:, None, None] * direction[None, :, :]
        + update.plastic_increment * derivative_direction
    )
    result = np.asarray(
        [_engineering_stress(value) for value in derivative_stress],
        dtype=float,
    ).T
    del alpha, yield_stress
    return result


def _deviatoric(value: np.ndarray) -> np.ndarray:
    return value - np.trace(value) / 3.0 * np.eye(3)


def _engineering_to_tensor(value: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [value[0], 0.5 * value[3], 0.5 * value[4]],
            [0.5 * value[3], value[1], 0.5 * value[5]],
            [0.5 * value[4], 0.5 * value[5], value[2]],
        ],
        dtype=float,
    )


def _input_small_strain(kinematics: Any) -> np.ndarray:
    value = getattr(kinematics, "small_strain", None)
    if value is None:
        raise ValueError("small-strain J2 requires kinematics.small_strain")
    strain = np.asarray(value, dtype=float)
    if strain.shape not in {(2, 2), (3, 3)} or not np.all(np.isfinite(strain)):
        raise ValueError("small_strain must be a finite 2x2 or 3x3 tensor")
    return strain


def _engineering_input(strain: np.ndarray) -> np.ndarray:
    if strain.shape == (2, 2):
        return np.array(
            [strain[0, 0], strain[1, 1], 2.0 * strain[0, 1]],
            dtype=float,
        )
    return _engineering_strain(strain)


def _strain_from_engineering_input(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if shape == (2, 2):
        return np.array(
            [[values[0], 0.5 * values[2]], [0.5 * values[2], values[1]]],
            dtype=float,
        )
    return np.array(
        [
            [values[0], 0.5 * values[3], 0.5 * values[4]],
            [0.5 * values[3], values[1], 0.5 * values[5]],
            [0.5 * values[4], 0.5 * values[5], values[2]],
        ],
        dtype=float,
    )


def _engineering_stress(stress: np.ndarray) -> np.ndarray:
    value = np.asarray(stress, dtype=float)
    return np.array(
        [
            value[0, 0],
            value[1, 1],
            value[2, 2],
            value[0, 1],
            value[0, 2],
            value[1, 2],
        ],
        dtype=float,
    )


__all__ = ["SmallStrainJ2PlasticityMaterial"]
