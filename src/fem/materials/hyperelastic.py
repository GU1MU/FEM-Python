"""Minimal finite-strain Neo-Hookean material response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import (
    KinematicMeasure,
    MaterialPointInput,
    MaterialResponse,
    StressMeasure,
)


@dataclass(frozen=True, slots=True)
class NeoHookeanMaterial:
    """Compressible isotropic Neo-Hookean material.

    ``C10`` and ``D1`` follow the Abaqus material-editor convention.  The
    current implementation uses a numerical material tangent so the same
    response contract works for 2D plane and 3D total-Lagrangian operators.
    """

    C10: float
    D1: float
    plane_type: str | None = None

    def __post_init__(self) -> None:
        C10 = float(self.C10)
        D1 = float(self.D1)
        if not np.isfinite(C10) or C10 <= 0.0:
            raise ValueError("C10 must be finite and > 0")
        if not np.isfinite(D1) or D1 <= 0.0:
            raise ValueError("D1 must be finite and > 0")
        plane_type = self.plane_type
        if plane_type is not None:
            plane_type = str(plane_type).casefold()
            if not (
                plane_type.startswith("stress")
                or plane_type.startswith("strain")
            ):
                raise ValueError("plane_type must be stress or strain")
        object.__setattr__(self, "C10", C10)
        object.__setattr__(self, "D1", D1)
        object.__setattr__(self, "plane_type", plane_type)

    @property
    def shear_modulus(self) -> float:
        return 2.0 * self.C10

    @property
    def bulk_modulus(self) -> float:
        return 2.0 / self.D1

    def initial_state(self) -> dict[str, object]:
        return {}

    def evaluate(self, point: MaterialPointInput) -> MaterialResponse:
        raw = np.asarray(point.kinematics.deformation_gradient, dtype=float)
        need_tangent = bool(
            point.context.parameters.get("_fem_need_tangent", True)
        )
        capture_outputs = bool(
            point.context.parameters.get("_fem_capture_outputs", True)
        )
        if raw.shape == (3, 3):
            deformation_gradient = raw
            first_piola = _neo_hookean_first_piola(
                deformation_gradient,
                self.shear_modulus,
                self.bulk_modulus,
            )
            tangent = (
                _numerical_tangent(
                    deformation_gradient,
                    lambda value: _neo_hookean_first_piola(
                        value,
                        self.shear_modulus,
                        self.bulk_modulus,
                    ),
                )
                if need_tangent
                else np.zeros((9, 9), dtype=float)
            )
        elif raw.shape == (2, 2):
            if self.plane_type is None:
                raise ValueError("2D Neo-Hookean material requires plane_type")
            deformation_gradient, first_piola = _plane_response(
                raw,
                self.plane_type,
                self.shear_modulus,
                self.bulk_modulus,
            )
            tangent = (
                _plane_tangent(
                    raw,
                    self.plane_type,
                    self.shear_modulus,
                    self.bulk_modulus,
                )
                if need_tangent
                else np.zeros((9, 9), dtype=float)
            )
        else:
            raise ValueError(
                "Neo-Hookean material requires a 2x2 or 3x3 deformation gradient"
            )

        determinant = float(np.linalg.det(deformation_gradient))
        inverse = np.linalg.inv(deformation_gradient)
        kirchhoff = first_piola @ deformation_gradient.T
        cauchy = kirchhoff / determinant
        second_piola = inverse @ first_piola
        outputs = (
            {
                "first_piola_stress": first_piola,
                "second_piola_stress": second_piola,
                "kirchhoff_stress": kirchhoff,
                "cauchy_stress": cauchy,
                "return_algorithm": "neo_hookean",
            }
            if capture_outputs
            else {}
        )
        return MaterialResponse(
            stress=first_piola,
            tangent=tangent,
            stress_measure=StressMeasure.FIRST_PIOLA,
            tangent_input=KinematicMeasure.DEFORMATION_GRADIENT,
            outputs=outputs,
        )


def _neo_hookean_first_piola(
    deformation_gradient: np.ndarray,
    shear_modulus: float,
    bulk_modulus: float,
) -> np.ndarray:
    F = np.asarray(deformation_gradient, dtype=float)
    if F.shape != (3, 3):
        raise ValueError("Neo-Hookean 3D response requires a 3x3 gradient")
    determinant = float(np.linalg.det(F))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("deformation gradient determinant must be > 0")
    inverse_transpose = np.linalg.inv(F).T
    invariant = float(np.trace(F.T @ F))
    isochoric_factor = determinant ** (-2.0 / 3.0)
    deviatoric = isochoric_factor * (
        F - invariant / 3.0 * inverse_transpose
    )
    volumetric = (
        bulk_modulus
        * (determinant - 1.0)
        * determinant
        * inverse_transpose
    )
    return shear_modulus * deviatoric + volumetric


def _embed_plane_gradient(value: np.ndarray, out_of_plane: float) -> np.ndarray:
    result = np.eye(3, dtype=float)
    result[:2, :2] = np.asarray(value, dtype=float)
    result[2, 2] = float(out_of_plane)
    return result


def _plane_response(
    value: np.ndarray,
    plane_type: str,
    shear_modulus: float,
    bulk_modulus: float,
) -> tuple[np.ndarray, np.ndarray]:
    if plane_type.startswith("strain"):
        F = _embed_plane_gradient(value, 1.0)
        return F, _neo_hookean_first_piola(F, shear_modulus, bulk_modulus)

    out_of_plane = 1.0
    for _ in range(40):
        F = _embed_plane_gradient(value, out_of_plane)
        response = _neo_hookean_first_piola(F, shear_modulus, bulk_modulus)
        residual = float(response[2, 2])
        scale = max(1.0, float(np.linalg.norm(response)))
        if abs(residual) <= 1.0e-10 * scale:
            return F, response
        increment = 1.0e-7 * max(1.0, abs(out_of_plane))
        plus = _embed_plane_gradient(value, out_of_plane + increment)
        minus = _embed_plane_gradient(value, max(1.0e-10, out_of_plane - increment))
        derivative = (
            _neo_hookean_first_piola(plus, shear_modulus, bulk_modulus)[2, 2]
            - _neo_hookean_first_piola(minus, shear_modulus, bulk_modulus)[2, 2]
        ) / (2.0 * increment)
        if not np.isfinite(derivative) or abs(derivative) <= 1.0e-14:
            break
        out_of_plane = max(1.0e-10, out_of_plane - residual / derivative)
    raise ValueError("plane-stress Neo-Hookean return did not converge")


def _numerical_tangent(
    value: np.ndarray,
    response: object,
) -> np.ndarray:
    F = np.asarray(value, dtype=float)
    tangent = np.zeros((9, 9), dtype=float)
    for column in range(9):
        row, direction = divmod(column, 3)
        step = 1.0e-7 * max(1.0, abs(float(F[row, direction])))
        plus = np.array(F, copy=True)
        minus = np.array(F, copy=True)
        plus[row, direction] += step
        minus[row, direction] -= step
        tangent[:, column] = (
            np.asarray(response(plus), dtype=float).reshape(9)
            - np.asarray(response(minus), dtype=float).reshape(9)
        ) / (2.0 * step)
    return tangent


def _plane_tangent(
    value: np.ndarray,
    plane_type: str,
    shear_modulus: float,
    bulk_modulus: float,
) -> np.ndarray:
    tangent = np.zeros((9, 9), dtype=float)
    F = np.asarray(value, dtype=float)
    active = (0, 1, 3, 4)

    def response(plane_gradient: np.ndarray) -> np.ndarray:
        _, result = _plane_response(
            plane_gradient,
            plane_type,
            shear_modulus,
            bulk_modulus,
        )
        return result

    for column in active:
        row, direction = divmod(column, 3)
        step = 1.0e-7 * max(1.0, abs(float(F[row, direction])))
        plus = np.array(F, copy=True)
        minus = np.array(F, copy=True)
        plus[row, direction] += step
        minus[row, direction] -= step
        tangent[:, column] = (
            response(plus).reshape(9) - response(minus).reshape(9)
        ) / (2.0 * step)
    return tangent


__all__ = ["NeoHookeanMaterial"]
