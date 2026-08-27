"""Internal J2 return-mapping algorithms used by Materials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True, slots=True)
class _HenckyJ2Response:
    first_piola_stress: np.ndarray
    second_piola_stress: np.ndarray
    kirchhoff_stress: np.ndarray
    plastic_log_strain: np.ndarray
    equivalent_plastic_strain: float
    rotation: np.ndarray | None = None
    stretch: np.ndarray | None = None
    stretch_eigenvalues: np.ndarray | None = None
    stretch_eigenvectors: np.ndarray | None = None
    inverse_deformation_gradient: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _HenckyJ2BatchResponse:
    first_piola_stress: np.ndarray
    second_piola_stress: np.ndarray
    kirchhoff_stress: np.ndarray
    plastic_log_strain: np.ndarray
    equivalent_plastic_strain: np.ndarray
    tangent: np.ndarray


@dataclass(frozen=True, slots=True)
class _MultiplicativeJ2Response:
    first_piola_stress: np.ndarray
    second_piola_stress: np.ndarray
    kirchhoff_stress: np.ndarray
    plastic_deformation_gradient: np.ndarray
    equivalent_plastic_strain: float
    yield_function: float
    plastic_increment: float
    iterated_direction: bool
    plastic_increment_tensor: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _MultiplicativeElasticResponse:
    first_piola_stress: np.ndarray
    second_piola_stress: np.ndarray
    kirchhoff_stress: np.ndarray
    mandel_stress: np.ndarray
    equivalent_stress: float


def _multiplicative_j2_response(
    deformation_gradient: np.ndarray,
    committed_plastic_deformation_gradient: np.ndarray,
    committed_equivalent_plastic_strain: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
) -> _MultiplicativeJ2Response:
    F = _validated_deformation_gradient(deformation_gradient)
    Fp = _validated_deformation_gradient(committed_plastic_deformation_gradient)
    trial = _multiplicative_elastic_response(F, Fp, E, nu)
    yield_limit = yield_stress + hardening_modulus * committed_equivalent_plastic_strain
    yield_function = trial.equivalent_stress - yield_limit
    if yield_function <= 1.0e-12:
        return _MultiplicativeJ2Response(
            first_piola_stress=trial.first_piola_stress,
            second_piola_stress=trial.second_piola_stress,
            kirchhoff_stress=trial.kirchhoff_stress,
            plastic_deformation_gradient=Fp.copy(),
            equivalent_plastic_strain=float(committed_equivalent_plastic_strain),
            yield_function=yield_function,
            plastic_increment=0.0,
            iterated_direction=False,
            plastic_increment_tensor=None,
        )

    direction = 1.5 * _deviatoric(trial.mandel_stress) / trial.equivalent_stress
    iterated_direction = False
    try:
        delta_gamma = _solve_multiplicative_plastic_increment(
            F,
            Fp,
            committed_equivalent_plastic_strain,
            direction,
            trial.equivalent_stress,
            E,
            nu,
            yield_stress,
            hardening_modulus,
        )
        plastic_deformation_gradient = (
            _symmetric_matrix_exponential(delta_gamma * direction) @ Fp
        )
        updated = _multiplicative_elastic_response(
            F,
            plastic_deformation_gradient,
            E,
            nu,
        )
    except RuntimeError:
        try:
            (
                delta_gamma,
                plastic_deformation_gradient,
                updated,
                plastic_increment_tensor,
            ) = _solve_iterated_multiplicative_return(
                F,
                Fp,
                committed_equivalent_plastic_strain,
                direction,
                trial.equivalent_stress,
                E,
                nu,
                yield_stress,
                hardening_modulus,
            )
        except RuntimeError as iterated_direction_error:
            raise RuntimeError(
                "constitutive return mapping failed: "
                f"{iterated_direction_error}"
            ) from iterated_direction_error
        iterated_direction = True
    equivalent_plastic_strain = (
        float(committed_equivalent_plastic_strain) + delta_gamma
    )
    return _MultiplicativeJ2Response(
        first_piola_stress=updated.first_piola_stress,
        second_piola_stress=updated.second_piola_stress,
        kirchhoff_stress=updated.kirchhoff_stress,
        plastic_deformation_gradient=plastic_deformation_gradient,
        equivalent_plastic_strain=equivalent_plastic_strain,
        yield_function=(
            updated.equivalent_stress
            - yield_stress
            - hardening_modulus * equivalent_plastic_strain
        ),
        plastic_increment=delta_gamma,
        iterated_direction=iterated_direction,
        plastic_increment_tensor=(
            plastic_increment_tensor
            if iterated_direction
            else delta_gamma * direction
        ),
    )


def _multiplicative_elastic_response(
    deformation_gradient: np.ndarray,
    plastic_deformation_gradient: np.ndarray,
    E: float,
    nu: float,
) -> _MultiplicativeElasticResponse:
    F = _validated_deformation_gradient(deformation_gradient)
    Fp = _validated_deformation_gradient(plastic_deformation_gradient)
    elastic_deformation_gradient = F @ np.linalg.inv(Fp)
    elastic_jacobian = float(np.linalg.det(elastic_deformation_gradient))
    if elastic_jacobian <= 0.0:
        raise ValueError("elastic deformation gradient determinant must be > 0")

    shear = E / (2.0 * (1.0 + nu))
    lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    left_cauchy_green = elastic_deformation_gradient @ elastic_deformation_gradient.T
    right_cauchy_green = elastic_deformation_gradient.T @ elastic_deformation_gradient
    kirchhoff = (
        shear * (left_cauchy_green - np.eye(3))
        + lame * np.log(elastic_jacobian) * np.eye(3)
    )
    mandel = (
        shear * (right_cauchy_green - np.eye(3))
        + lame * np.log(elastic_jacobian) * np.eye(3)
    )
    trial_deviator = _deviatoric(mandel)
    equivalent_stress = float(
        np.sqrt(1.5 * np.sum(trial_deviator * trial_deviator))
    )
    inverse_f = np.linalg.inv(F)
    return _MultiplicativeElasticResponse(
        first_piola_stress=kirchhoff @ inverse_f.T,
        second_piola_stress=float(np.linalg.det(F)) * inverse_f @ kirchhoff @ inverse_f.T,
        kirchhoff_stress=kirchhoff,
        mandel_stress=mandel,
        equivalent_stress=equivalent_stress,
    )


def _solve_multiplicative_plastic_increment(
    deformation_gradient: np.ndarray,
    committed_plastic_deformation_gradient: np.ndarray,
    committed_equivalent_plastic_strain: float,
    direction: np.ndarray,
    trial_equivalent_stress: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
) -> float:
    """Solve the scalar consistency equation for a fixed trial direction."""

    shear = E / (2.0 * (1.0 + nu))
    trial_limit = yield_stress + hardening_modulus * committed_equivalent_plastic_strain
    initial_residual = trial_equivalent_stress - trial_limit

    def residual(delta_gamma: float) -> float:
        plastic_gradient = (
            _symmetric_matrix_exponential(delta_gamma * direction)
            @ committed_plastic_deformation_gradient
        )
        state = _multiplicative_elastic_response(
            deformation_gradient,
            plastic_gradient,
            E,
            nu,
        )
        return state.equivalent_stress - (
            yield_stress
            + hardening_modulus
            * (committed_equivalent_plastic_strain + delta_gamma)
        )

    lower = 0.0
    initial_upper = max(
        initial_residual / (3.0 * shear + hardening_modulus),
        1.0e-12,
    )
    upper = initial_upper
    bracketed = False
    for _ in range(30):
        interval_start = 0.0 if upper == initial_upper else upper * 0.5
        previous_gamma = interval_start
        previous_residual = (
            initial_residual if interval_start == 0.0 else residual(interval_start)
        )
        for fraction in np.linspace(1.0 / 16.0, 1.0, 16):
            candidate = interval_start + (upper - interval_start) * fraction
            candidate_residual = residual(candidate)
            if previous_residual > 0.0 and candidate_residual <= 0.0:
                lower = previous_gamma
                upper = candidate
                bracketed = True
                break
            previous_gamma = candidate
            previous_residual = candidate_residual
        if bracketed:
            break
        if upper >= 2.0:
            break
        upper *= 2.0
    if not bracketed:
        raise RuntimeError("multiplicative J2 return mapping failed to bracket root")

    delta_gamma = min(
        max(initial_residual / (3.0 * shear + hardening_modulus), lower),
        upper,
    )
    tolerance = 1.0e-10 * max(1.0, trial_limit)
    for _ in range(60):
        current_residual = residual(delta_gamma)
        if abs(current_residual) <= tolerance:
            return float(delta_gamma)
        if current_residual > 0.0:
            lower = delta_gamma
        else:
            upper = delta_gamma

        step = 1.0e-7 * max(1.0, abs(delta_gamma))
        left = max(lower, delta_gamma - step)
        right = min(upper, delta_gamma + step)
        derivative = (residual(right) - residual(left)) / (right - left)
        if not np.isfinite(derivative) or derivative >= 0.0:
            delta_gamma = 0.5 * (lower + upper)
            continue
        candidate = delta_gamma - current_residual / derivative
        if not lower < candidate < upper:
            candidate = 0.5 * (lower + upper)
        delta_gamma = candidate
    raise RuntimeError("multiplicative J2 return mapping did not converge")


def _solve_iterated_multiplicative_return(
    deformation_gradient: np.ndarray,
    committed_plastic_deformation_gradient: np.ndarray,
    committed_equivalent_plastic_strain: float,
    trial_direction: np.ndarray,
    trial_equivalent_stress: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
) -> tuple[float, np.ndarray, _MultiplicativeElasticResponse, np.ndarray]:
    """Solve a local return with a symmetric trace-free flow increment."""

    initial_gamma = max(
        (
            trial_equivalent_stress
            - yield_stress
            - hardening_modulus * committed_equivalent_plastic_strain
        )
        / (3.0 * E / (2.0 * (1.0 + nu)) + hardening_modulus),
        1.0e-10,
    )
    initial_increment = initial_gamma * trial_direction
    initial_vector = np.array(
        [
            initial_increment[0, 0],
            initial_increment[1, 1],
            initial_increment[0, 1],
            initial_increment[0, 2],
            initial_increment[1, 2],
            initial_gamma,
        ],
        dtype=float,
    )
    stress_scale = max(
        1.0,
        yield_stress + hardening_modulus * committed_equivalent_plastic_strain,
    )

    def increment_from_vector(vector: np.ndarray) -> np.ndarray:
        increment = np.array(
            [
                [vector[0], vector[2], vector[3]],
                [vector[2], vector[1], vector[4]],
                [vector[3], vector[4], -vector[0] - vector[1]],
            ],
            dtype=float,
        )
        return 0.5 * (increment + increment.T)

    def residual(vector: np.ndarray) -> np.ndarray:
        increment = increment_from_vector(vector)
        gamma = float(vector[5])
        plastic_gradient = (
            _symmetric_matrix_exponential(increment)
            @ committed_plastic_deformation_gradient
        )
        state = _multiplicative_elastic_response(
            deformation_gradient,
            plastic_gradient,
            E,
            nu,
        )
        if state.equivalent_stress <= 1.0e-14:
            flow = _deviatoric(increment)
        else:
            flow = _deviatoric(increment) - gamma * 1.5 * (
                _deviatoric(state.mandel_stress) / state.equivalent_stress
            )
        consistency = (
            state.equivalent_stress
            - yield_stress
            - hardening_modulus
            * (committed_equivalent_plastic_strain + gamma)
        ) / stress_scale
        return np.array(
            [flow[0, 0], flow[1, 1], flow[0, 1], flow[0, 2], flow[1, 2], consistency],
            dtype=float,
        )

    lower = np.full(6, -np.inf, dtype=float)
    lower[5] = 0.0
    upper = np.full(6, np.inf, dtype=float)
    # Keep the plastic multiplier comfortably away from the artificial bound.
    # The bound also affects least_squares scaling; 2.0 can make strong
    # plastic-return states stagnate even when a valid root is nearby.
    upper[5] = max(5.0, 10.0 * initial_gamma)
    result = least_squares(
        residual,
        initial_vector,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=500,
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
    )
    if not result.success or np.linalg.norm(residual(result.x), ord=np.inf) > 1.0e-8:
        raise RuntimeError("iterated multiplicative return did not converge")
    increment = increment_from_vector(result.x)
    plastic_gradient = (
        _symmetric_matrix_exponential(increment)
        @ committed_plastic_deformation_gradient
    )
    state = _multiplicative_elastic_response(
        deformation_gradient,
        plastic_gradient,
        E,
        nu,
    )
    return float(result.x[5]), plastic_gradient, state, increment


def _symmetric_matrix_exponential(value: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    result = eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _analytic_elastic_dP_dF(
    deformation_gradient: np.ndarray,
    plastic_deformation_gradient: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """Return the exact elastic ``dP/dF`` at fixed ``Fp``."""

    F = _validated_deformation_gradient(deformation_gradient)
    Fp = _validated_deformation_gradient(plastic_deformation_gradient)
    Fe = F @ np.linalg.inv(Fp)
    Fe_inverse = np.linalg.inv(Fe)
    F_inverse_transpose = np.linalg.inv(F).T
    elastic = _multiplicative_elastic_response(F, Fp, E, nu)
    shear = E / (2.0 * (1.0 + nu))
    lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    tangent = np.zeros((9, 9), dtype=float)
    for column in range(9):
        index = np.unravel_index(column, (3, 3))
        delta_f = np.zeros((3, 3), dtype=float)
        delta_f[index] = 1.0
        delta_fe = delta_f @ np.linalg.inv(Fp)
        delta_b = delta_fe @ Fe.T + Fe @ delta_fe.T
        delta_log_j = np.trace(Fe_inverse @ delta_fe)
        delta_tau = shear * delta_b + lame * delta_log_j * np.eye(3)
        delta_p = (
            delta_tau @ F_inverse_transpose
            - elastic.kirchhoff_stress
            @ F_inverse_transpose
            @ delta_f.T
            @ F_inverse_transpose
        )
        tangent[:, column] = delta_p.reshape(9)
    return tangent


def _analytic_plastic_dP_dF(
    deformation_gradient: np.ndarray,
    committed_plastic_deformation_gradient: np.ndarray,
    committed_equivalent_plastic_strain: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
) -> np.ndarray:
    """Return the algorithmic tangent for the current fixed-direction return."""

    F = _validated_deformation_gradient(deformation_gradient)
    Fp_n = _validated_deformation_gradient(committed_plastic_deformation_gradient)
    trial = _multiplicative_elastic_response(F, Fp_n, E, nu)
    direction = 1.5 * _deviatoric(trial.mandel_stress) / trial.equivalent_stress
    delta_gamma = _solve_multiplicative_plastic_increment(
        F,
        Fp_n,
        committed_equivalent_plastic_strain,
        direction,
        trial.equivalent_stress,
        E,
        nu,
        yield_stress,
        hardening_modulus,
    )
    exponential_argument = delta_gamma * direction
    exponential = _symmetric_matrix_exponential(exponential_argument)
    Fp = exponential @ Fp_n
    inverse_f_transpose = np.linalg.inv(F).T
    inverse_fp_n = np.linalg.inv(Fp_n)
    inverse_fp = np.linalg.inv(Fp)
    Fe = F @ inverse_fp
    final = _multiplicative_elastic_response(F, Fp, E, nu)
    derivative_exp_gamma = _symmetric_matrix_exponential_frechet(
        exponential_argument,
        direction,
    )
    dFp_gamma = derivative_exp_gamma @ Fp_n
    final_q = final.equivalent_stress

    tangent = np.zeros((9, 9), dtype=float)
    for column in range(9):
        index = np.unravel_index(column, (3, 3))
        delta_f = np.zeros((3, 3), dtype=float)
        delta_f[index] = 1.0

        delta_fe_trial = delta_f @ inverse_fp_n
        _, delta_mandel_trial, delta_q_trial = _neo_hooke_variation(
            F @ inverse_fp_n,
            delta_fe_trial,
            E,
            nu,
            trial.mandel_stress,
            trial.equivalent_stress,
        )
        trial_deviator = _deviatoric(trial.mandel_stress)
        delta_trial_deviator = _deviatoric(delta_mandel_trial)
        delta_direction = 1.5 * (
            delta_trial_deviator / trial.equivalent_stress
            - trial_deviator * delta_q_trial / trial.equivalent_stress**2
        )

        delta_exp_f = _symmetric_matrix_exponential_frechet(
            exponential_argument,
            delta_gamma * delta_direction,
        )
        delta_fp_f = delta_exp_f @ Fp_n
        delta_fe_f = delta_f @ inverse_fp - Fe @ delta_fp_f @ inverse_fp
        delta_fe_gamma = -Fe @ dFp_gamma @ inverse_fp

        delta_tau_f, _, delta_q_f = _neo_hooke_variation(
            Fe,
            delta_fe_f,
            E,
            nu,
            final.mandel_stress,
            final_q,
        )
        delta_tau_gamma, _, delta_q_gamma = _neo_hooke_variation(
            Fe,
            delta_fe_gamma,
            E,
            nu,
            final.mandel_stress,
            final_q,
        )
        consistency_derivative = delta_q_gamma - hardening_modulus
        if abs(consistency_derivative) <= 1.0e-14:
            raise RuntimeError("multiplicative J2 tangent consistency derivative is singular")
        delta_gamma_f = -delta_q_f / consistency_derivative
        delta_tau = delta_tau_f + delta_tau_gamma * delta_gamma_f
        delta_p = (
            delta_tau @ inverse_f_transpose
            - final.kirchhoff_stress
            @ inverse_f_transpose
            @ delta_f.T
            @ inverse_f_transpose
        )
        tangent[:, column] = delta_p.reshape(9)
    return tangent


def _neo_hooke_variation(
    elastic_deformation_gradient: np.ndarray,
    delta_elastic_deformation_gradient: np.ndarray,
    E: float,
    nu: float,
    mandel_stress: np.ndarray,
    equivalent_stress: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``dτ``, ``dM`` and ``dq`` for a perturbation of ``Fe``."""

    Fe = elastic_deformation_gradient
    delta_fe = delta_elastic_deformation_gradient
    shear = E / (2.0 * (1.0 + nu))
    lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    inverse_fe = np.linalg.inv(Fe)
    delta_log_j = np.trace(inverse_fe @ delta_fe)
    delta_left_cauchy_green = delta_fe @ Fe.T + Fe @ delta_fe.T
    delta_right_cauchy_green = delta_fe.T @ Fe + Fe.T @ delta_fe
    delta_tau = shear * delta_left_cauchy_green + lame * delta_log_j * np.eye(3)
    delta_mandel = shear * delta_right_cauchy_green + lame * delta_log_j * np.eye(3)
    if equivalent_stress <= 1.0e-14:
        delta_q = 0.0
    else:
        deviator = _deviatoric(mandel_stress)
        delta_q = 1.5 * np.sum(deviator * delta_mandel) / equivalent_stress
    return delta_tau, delta_mandel, float(delta_q)


def _symmetric_matrix_exponential_frechet(
    value: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Return the Frechet derivative of ``exp(value)`` in ``direction``."""

    symmetric_value = 0.5 * (value + value.T)
    symmetric_direction = 0.5 * (direction + direction.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_value)
    transformed_direction = eigenvectors.T @ symmetric_direction @ eigenvectors
    divided_difference = np.empty((3, 3), dtype=float)
    for row in range(3):
        for column in range(3):
            difference = eigenvalues[row] - eigenvalues[column]
            if abs(difference) <= 1.0e-12:
                divided_difference[row, column] = np.exp(eigenvalues[row])
            else:
                divided_difference[row, column] = (
                    np.exp(eigenvalues[row]) - np.exp(eigenvalues[column])
                ) / difference
    result = eigenvectors @ (divided_difference * transformed_direction) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _trace_free_increment_from_vector(vector: np.ndarray) -> np.ndarray:
    increment = np.array(
        [
            [vector[0], vector[2], vector[3]],
            [vector[2], vector[1], vector[4]],
            [vector[3], vector[4], -vector[0] - vector[1]],
        ],
        dtype=float,
    )
    return 0.5 * (increment + increment.T)


def _joint_return_residual(
    deformation_gradient: np.ndarray,
    committed_plastic_deformation_gradient: np.ndarray,
    committed_equivalent_plastic_strain: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
    vector: np.ndarray,
    stress_scale: float,
) -> tuple[np.ndarray, np.ndarray, _MultiplicativeElasticResponse]:
    increment = _trace_free_increment_from_vector(vector)
    gamma = float(vector[5])
    plastic_gradient = (
        _symmetric_matrix_exponential(increment)
        @ committed_plastic_deformation_gradient
    )
    state = _multiplicative_elastic_response(
        deformation_gradient,
        plastic_gradient,
        E,
        nu,
    )
    if state.equivalent_stress <= 1.0e-14:
        flow = _deviatoric(increment)
    else:
        flow = _deviatoric(increment) - gamma * 1.5 * (
            _deviatoric(state.mandel_stress) / state.equivalent_stress
        )
    consistency = (
        state.equivalent_stress
        - yield_stress
        - hardening_modulus
        * (committed_equivalent_plastic_strain + gamma)
    ) / stress_scale
    residual = np.array(
        [flow[0, 0], flow[1, 1], flow[0, 1], flow[0, 2], flow[1, 2], consistency],
        dtype=float,
    )
    return residual, plastic_gradient, state


def _implicit_joint_dP_dF(
    deformation_gradient: np.ndarray,
    committed_plastic_deformation_gradient: np.ndarray,
    committed_equivalent_plastic_strain: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
    plastic_increment_tensor: np.ndarray | None,
) -> np.ndarray:
    """Return a local implicit algorithmic tangent for joint flow return."""

    if plastic_increment_tensor is None:
        raise RuntimeError("joint return tangent requires a plastic increment tensor")
    F = _validated_deformation_gradient(deformation_gradient)
    Fp_n = _validated_deformation_gradient(committed_plastic_deformation_gradient)
    increment = np.asarray(plastic_increment_tensor, dtype=float)
    vector = np.array(
        [
            increment[0, 0],
            increment[1, 1],
            increment[0, 1],
            increment[0, 2],
            increment[1, 2],
            np.nan,
        ],
        dtype=float,
    )
    gamma = float(np.sqrt(np.sum(increment * increment) / 1.5))
    vector[5] = gamma
    stress_scale = max(
        1.0,
        yield_stress + hardening_modulus * committed_equivalent_plastic_strain,
    )

    def residual_at(F_value: np.ndarray, local_vector: np.ndarray) -> np.ndarray:
        return _joint_return_residual(
            F_value,
            Fp_n,
            committed_equivalent_plastic_strain,
            E,
            nu,
            yield_stress,
            hardening_modulus,
            local_vector,
            stress_scale,
        )[0]

    def stress_at(F_value: np.ndarray, local_vector: np.ndarray) -> np.ndarray:
        return _joint_return_residual(
            F_value,
            Fp_n,
            committed_equivalent_plastic_strain,
            E,
            nu,
            yield_stress,
            hardening_modulus,
            local_vector,
            stress_scale,
        )[2].first_piola_stress.reshape(9)

    local_jacobian = np.zeros((6, 6), dtype=float)
    for column in range(6):
        step = 1.0e-7 * max(1.0, abs(vector[column]))
        plus = vector.copy()
        minus = vector.copy()
        plus[column] += step
        minus[column] -= step
        local_jacobian[:, column] = (
            residual_at(F, plus) - residual_at(F, minus)
        ) / (2.0 * step)

    deformation_jacobian = np.zeros((6, 9), dtype=float)
    stress_deformation = np.zeros((9, 9), dtype=float)
    stress_local = np.zeros((9, 6), dtype=float)
    for column in range(9):
        index = np.unravel_index(column, (3, 3))
        step = 1.0e-7 * max(1.0, abs(F[index]))
        plus = F.copy()
        minus = F.copy()
        plus[index] += step
        minus[index] -= step
        deformation_jacobian[:, column] = (
            residual_at(plus, vector) - residual_at(minus, vector)
        ) / (2.0 * step)
        stress_deformation[:, column] = (
            stress_at(plus, vector) - stress_at(minus, vector)
        ) / (2.0 * step)
    for column in range(6):
        step = 1.0e-7 * max(1.0, abs(vector[column]))
        plus = vector.copy()
        minus = vector.copy()
        plus[column] += step
        minus[column] -= step
        stress_local[:, column] = (
            stress_at(F, plus) - stress_at(F, minus)
        ) / (2.0 * step)

    try:
        local_response = np.linalg.solve(local_jacobian, -deformation_jacobian)
    except np.linalg.LinAlgError:
        return _numerical_multiplicative_dP_dF(
            F,
            Fp_n,
            committed_equivalent_plastic_strain,
            E,
            nu,
            yield_stress,
            hardening_modulus,
        )
    tangent = stress_deformation + stress_local @ local_response
    if not np.all(np.isfinite(tangent)):
        return _numerical_multiplicative_dP_dF(
            F,
            Fp_n,
            committed_equivalent_plastic_strain,
            E,
            nu,
            yield_stress,
            hardening_modulus,
        )
    return tangent


def _numerical_multiplicative_dP_dF(
    deformation_gradient: np.ndarray,
    committed_plastic_deformation_gradient: np.ndarray,
    committed_equivalent_plastic_strain: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
) -> np.ndarray:
    """Return a full-return finite-difference tangent for the fallback path."""

    F = _validated_deformation_gradient(deformation_gradient)
    tangent = np.zeros((9, 9), dtype=float)
    for column in range(9):
        index = np.unravel_index(column, (3, 3))
        step = 1.0e-7 * max(1.0, abs(F[index]))
        plus = F.copy()
        minus = F.copy()
        plus[index] += step
        minus[index] -= step
        response_plus = _multiplicative_j2_response(
            plus,
            committed_plastic_deformation_gradient,
            committed_equivalent_plastic_strain,
            E,
            nu,
            yield_stress,
            hardening_modulus,
        )
        response_minus = _multiplicative_j2_response(
            minus,
            committed_plastic_deformation_gradient,
            committed_equivalent_plastic_strain,
            E,
            nu,
            yield_stress,
            hardening_modulus,
        )
        tangent[:, column] = (
            response_plus.first_piola_stress.reshape(9)
            - response_minus.first_piola_stress.reshape(9)
        ) / (2.0 * step)
    return tangent


def _hencky_j2_response(
    deformation_gradient: np.ndarray,
    committed_plastic_log_strain: np.ndarray,
    committed_equivalent_plastic_strain: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
) -> _HenckyJ2Response:
    F = _validated_deformation_gradient(deformation_gradient)
    rotation, stretch, jacobian = _polar_decomposition(F)
    stretch_eigenvalues, stretch_eigenvectors = np.linalg.eigh(stretch)
    if np.any(stretch_eigenvalues <= 0.0):
        raise ValueError("right stretch must be positive definite")
    principal_log_strain = (
        stretch_eigenvectors
        @ np.diag(np.log(stretch_eigenvalues))
        @ stretch_eigenvectors.T
    )
    bulk = E / (3.0 * (1.0 - 2.0 * nu))
    shear = E / (2.0 * (1.0 + nu))

    elastic_trial = principal_log_strain - committed_plastic_log_strain
    trial_kirchhoff = bulk * np.trace(elastic_trial) * np.eye(3)
    trial_kirchhoff += 2.0 * shear * _deviatoric(elastic_trial)
    trial_deviator = _deviatoric(trial_kirchhoff)
    trial_equivalent = float(
        np.sqrt(1.5 * np.sum(trial_deviator * trial_deviator))
    )
    yield_limit = yield_stress + hardening_modulus * committed_equivalent_plastic_strain
    yield_function = trial_equivalent - yield_limit

    if yield_function <= 1.0e-12:
        plastic_log_strain = np.array(
            committed_plastic_log_strain,
            dtype=float,
            copy=True,
        )
        equivalent_plastic_strain = float(committed_equivalent_plastic_strain)
        kirchhoff_hat = trial_kirchhoff
    else:
        direction = 1.5 * trial_deviator / trial_equivalent
        delta_gamma = yield_function / (3.0 * shear + hardening_modulus)
        plastic_log_strain = (
            np.asarray(committed_plastic_log_strain, dtype=float)
            + delta_gamma * direction
        )
        equivalent_plastic_strain = (
            float(committed_equivalent_plastic_strain) + delta_gamma
        )
        kirchhoff_hat = trial_kirchhoff - 2.0 * shear * delta_gamma * direction

    kirchhoff = rotation @ kirchhoff_hat @ rotation.T
    inverse_f = np.linalg.inv(F)
    first_piola = kirchhoff @ inverse_f.T
    second_piola = jacobian * inverse_f @ kirchhoff @ inverse_f.T
    return _HenckyJ2Response(
        first_piola_stress=first_piola,
        second_piola_stress=second_piola,
        kirchhoff_stress=kirchhoff,
        plastic_log_strain=0.5 * (plastic_log_strain + plastic_log_strain.T),
        equivalent_plastic_strain=equivalent_plastic_strain,
        rotation=rotation,
        stretch=stretch,
        stretch_eigenvalues=stretch_eigenvalues,
        stretch_eigenvectors=stretch_eigenvectors,
        inverse_deformation_gradient=inverse_f,
    )


def _numerical_dP_dF(
    deformation_gradient: np.ndarray,
    committed_plastic_log_strain: np.ndarray,
    committed_equivalent_plastic_strain: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
) -> np.ndarray:
    """Return a fixed-committed-state central-difference ``dP/dF``."""

    F = _validated_deformation_gradient(deformation_gradient)
    tangent = np.zeros((9, 9), dtype=float)
    for column in range(9):
        index = np.unravel_index(column, (3, 3))
        step = 1.0e-7 * max(1.0, abs(F[index]))
        plus = F.copy()
        minus = F.copy()
        plus[index] += step
        minus[index] -= step
        response_plus = _hencky_j2_response(
            plus,
            committed_plastic_log_strain,
            committed_equivalent_plastic_strain,
            E,
            nu,
            yield_stress,
            hardening_modulus,
        )
        response_minus = _hencky_j2_response(
            minus,
            committed_plastic_log_strain,
            committed_equivalent_plastic_strain,
            E,
            nu,
            yield_stress,
            hardening_modulus,
        )
        tangent[:, column] = (
            response_plus.first_piola_stress.reshape(9)
            - response_minus.first_piola_stress.reshape(9)
        ) / (2.0 * step)
    return tangent


def _analytic_hencky_dP_dF(
    deformation_gradient: np.ndarray,
    committed_plastic_log_strain: np.ndarray,
    committed_equivalent_plastic_strain: float,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
    *,
    response: _HenckyJ2Response | None = None,
) -> np.ndarray:
    """Return the objective algorithmic ``dP/dF`` for the Hencky model.

    The Hencky return mapping is unchanged.  This routine differentiates its
    existing polar/logarithmic-strain operations directly, so one material
    point no longer evaluates eighteen full central-difference responses for
    a nine-column tangent.  The committed plastic logarithmic strain remains
    fixed while differentiating, exactly as in ``_numerical_dP_dF``.
    """

    F = _validated_deformation_gradient(deformation_gradient)
    committed = np.asarray(committed_plastic_log_strain, dtype=float)
    if committed.shape != (3, 3) or not np.all(np.isfinite(committed)):
        raise ValueError("committed_plastic_log_strain must be a finite 3x3 array")
    if response is None:
        response = _hencky_j2_response(
            F,
            committed,
            committed_equivalent_plastic_strain,
            E,
            nu,
            yield_stress,
            hardening_modulus,
        )

    rotation = response.rotation
    stretch = response.stretch
    eigenvalues = response.stretch_eigenvalues
    eigenvectors = response.stretch_eigenvectors
    if (
        rotation is None
        or stretch is None
        or eigenvalues is None
        or eigenvectors is None
    ):
        rotation, stretch, _jacobian = _polar_decomposition(F)
        eigenvalues, eigenvectors = np.linalg.eigh(stretch)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("right stretch must be positive definite")
    log_stretch = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T

    bulk = E / (3.0 * (1.0 - 2.0 * nu))
    shear = E / (2.0 * (1.0 + nu))
    trial_log_strain = log_stretch - committed
    identity = np.eye(3, dtype=float)
    trial_kirchhoff = _hencky_isotropic_stress(
        trial_log_strain,
        bulk,
        shear,
    )
    trial_deviator = _deviatoric(trial_kirchhoff)
    trial_equivalent = float(
        np.sqrt(1.5 * np.sum(trial_deviator * trial_deviator))
    )
    plastic_increment = float(
        response.equivalent_plastic_strain
        - committed_equivalent_plastic_strain
    )
    plastic = plastic_increment > 1.0e-14
    if plastic:
        if trial_equivalent <= 1.0e-14:
            return _numerical_dP_dF(
                F,
                committed,
                committed_equivalent_plastic_strain,
                E,
                nu,
                yield_stress,
                hardening_modulus,
            )
        direction = 1.5 * trial_deviator / trial_equivalent
    else:
        direction = np.zeros((3, 3), dtype=float)

    dF = np.eye(9, dtype=float).reshape(9, 3, 3)
    rotated_dF = np.einsum("ij,cjk->cik", rotation.T, dF)
    rotated_dF_principal = np.einsum(
        "ij,cjk,kl->cil",
        eigenvectors.T,
        rotated_dF,
        eigenvectors,
    )
    omega_principal = (
        rotated_dF_principal - rotated_dF_principal.transpose(0, 2, 1)
    ) / (eigenvalues[:, None] + eigenvalues[None, :])
    omega = np.einsum(
        "ij,cjk,kl->cil",
        eigenvectors,
        omega_principal,
        eigenvectors.T,
    )
    d_stretch = rotated_dF - np.einsum(
        "cij,jk->cik",
        omega,
        stretch,
    )
    d_stretch = 0.5 * (d_stretch + d_stretch.transpose(0, 2, 1))

    d_stretch_principal = np.einsum(
        "ij,cjk,kl->cil",
        eigenvectors.T,
        d_stretch,
        eigenvectors,
    )
    log_divided_difference = np.empty((3, 3), dtype=float)
    for row in range(3):
        for column in range(3):
            difference = eigenvalues[row] - eigenvalues[column]
            if abs(difference) <= 1.0e-10 * max(1.0, eigenvalues[row]):
                log_divided_difference[row, column] = 1.0 / eigenvalues[row]
            else:
                log_divided_difference[row, column] = (
                    np.log(eigenvalues[row]) - np.log(eigenvalues[column])
                ) / difference
    d_log_stretch = np.einsum(
        "ij,cjk,kl->cil",
        eigenvectors,
        log_divided_difference[None, :, :] * d_stretch_principal,
        eigenvectors.T,
    )
    d_rotation = np.einsum("ij,cjk->cik", rotation, omega)

    d_trial = _hencky_isotropic_stress_derivative(
        d_log_stretch,
        bulk,
        shear,
    )
    d_trial_deviator = d_trial - (
        np.trace(d_trial, axis1=1, axis2=2)[:, None, None] / 3.0
    ) * identity
    if plastic:
        d_equivalent = (
            1.5
            * np.einsum("ij,cij->c", trial_deviator, d_trial_deviator)
            / trial_equivalent
        )
        d_direction = 1.5 * (
            d_trial_deviator / trial_equivalent
            - trial_deviator[None, :, :]
            * d_equivalent[:, None, None]
            / trial_equivalent**2
        )
        d_gamma = d_equivalent / (3.0 * shear + hardening_modulus)
        d_kirchhoff_hat = d_trial - 2.0 * shear * (
            d_gamma[:, None, None] * direction[None, :, :]
            + plastic_increment * d_direction
        )
    else:
        d_kirchhoff_hat = d_trial

    kirchhoff_hat = (
        rotation.T
        @ np.asarray(response.kirchhoff_stress, dtype=float)
        @ rotation
    )
    d_kirchhoff = (
        np.einsum(
            "cij,jk,kl->cil",
            d_rotation,
            kirchhoff_hat,
            rotation.T,
        )
        + np.einsum(
            "ij,cjk,kl->cil",
            rotation,
            d_kirchhoff_hat,
            rotation.T,
        )
        + np.einsum(
            "ij,jk,clk->cil",
            rotation,
            kirchhoff_hat,
            d_rotation,
        )
    )
    inverse_f = response.inverse_deformation_gradient
    if inverse_f is None:
        inverse_f = np.linalg.inv(F)
    inverse_f_transpose = inverse_f.T
    first_term = np.einsum(
        "cij,jk->cik",
        d_kirchhoff,
        inverse_f_transpose,
    )
    second_term = np.einsum(
        "ij,jk,ckl,lm->cim",
        np.asarray(response.kirchhoff_stress, dtype=float),
        inverse_f_transpose,
        dF.transpose(0, 2, 1),
        inverse_f_transpose,
    )
    d_first_piola = first_term - second_term
    return d_first_piola.reshape(9, 9).T


def _hencky_j2_response_batch(
    deformation_gradients: np.ndarray,
    committed_plastic_log_strains: np.ndarray,
    committed_equivalent_plastic_strains: np.ndarray,
    E: float,
    nu: float,
    yield_stress: float,
    hardening_modulus: float,
    *,
    need_tangent: bool,
    tangent_columns: tuple[int, ...] | None = None,
) -> _HenckyJ2BatchResponse:
    """Evaluate a block of Hencky points with shared batched linear algebra."""

    F = np.asarray(deformation_gradients, dtype=float)
    committed = np.asarray(committed_plastic_log_strains, dtype=float)
    alpha = np.asarray(committed_equivalent_plastic_strains, dtype=float)
    if F.ndim != 3 or F.shape[1:] != (3, 3):
        raise ValueError("deformation_gradients must have shape (n, 3, 3)")
    if committed.shape != F.shape:
        raise ValueError("committed_plastic_log_strains must match deformation_gradients")
    if alpha.shape != (F.shape[0],):
        raise ValueError("committed_equivalent_plastic_strains must match points")
    determinants = np.linalg.det(F)
    if np.any(~np.isfinite(F)) or np.any(determinants <= 0.0):
        raise ValueError("deformation_gradients must have positive finite determinants")

    left, singular_values, right_transpose = np.linalg.svd(F)
    right = right_transpose.transpose(0, 2, 1)
    stretch = (right * singular_values[:, None, :]) @ right_transpose
    rotation = left @ right_transpose
    stretch_eigenvalues, stretch_eigenvectors = np.linalg.eigh(stretch)
    if np.any(stretch_eigenvalues <= 0.0):
        raise ValueError("right stretch must be positive definite")
    log_stretch = (
        stretch_eigenvectors
        * np.log(stretch_eigenvalues)[:, None, :]
    ) @ stretch_eigenvectors.transpose(0, 2, 1)

    bulk = E / (3.0 * (1.0 - 2.0 * nu))
    shear = E / (2.0 * (1.0 + nu))
    identity = np.eye(3, dtype=float)
    trial_log_strain = log_stretch - committed
    trial_kirchhoff = (
        bulk
        * np.trace(trial_log_strain, axis1=1, axis2=2)[:, None, None]
        * identity
        + 2.0 * shear * (
            trial_log_strain
            - np.trace(trial_log_strain, axis1=1, axis2=2)[:, None, None]
            * identity
            / 3.0
        )
    )
    trial_deviator = trial_kirchhoff - (
        np.trace(trial_kirchhoff, axis1=1, axis2=2)[:, None, None] / 3.0
    ) * identity
    trial_equivalent = np.sqrt(
        np.maximum(0.0, 1.5 * np.sum(trial_deviator * trial_deviator, axis=(1, 2)))
    )
    yield_limit = yield_stress + hardening_modulus * alpha
    plastic = trial_equivalent - yield_limit > 1.0e-12
    safe_equivalent = np.maximum(trial_equivalent, 1.0e-14)
    direction = 1.5 * trial_deviator / safe_equivalent[:, None, None]
    plastic_increment = np.where(
        plastic,
        (trial_equivalent - yield_limit) / (3.0 * shear + hardening_modulus),
        0.0,
    )
    plastic_log_strain = committed + plastic_increment[:, None, None] * direction
    plastic_log_strain = np.where(
        plastic[:, None, None],
        plastic_log_strain,
        committed,
    )
    equivalent_plastic_strain = alpha + plastic_increment
    kirchhoff_hat = trial_kirchhoff - (
        2.0 * shear * plastic_increment[:, None, None] * direction
    )
    kirchhoff = rotation @ kirchhoff_hat @ rotation.transpose(0, 2, 1)
    inverse_f = _batch_inverse_3x3(F, determinants)
    first_piola = kirchhoff @ inverse_f.transpose(0, 2, 1)
    second_piola = (
        determinants[:, None, None]
        * inverse_f
        @ kirchhoff
        @ inverse_f.transpose(0, 2, 1)
    )
    if not need_tangent:
        tangent = np.zeros((F.shape[0], 9, 9), dtype=float)
    else:
        tangent_columns_value = (
            tuple(range(9))
            if tangent_columns is None
            else tuple(int(column) for column in tangent_columns)
        )
        tangent = np.zeros((F.shape[0], 9, 9), dtype=float)
        tangent[:, :, tangent_columns_value] = _analytic_hencky_batch_tangent(
            F,
            rotation,
            stretch,
            stretch_eigenvalues,
            stretch_eigenvectors,
            inverse_f,
            trial_deviator,
            trial_equivalent,
            direction,
            plastic_increment,
            plastic,
            kirchhoff_hat,
            bulk,
            shear,
            hardening_modulus,
            identity,
            tangent_columns=tangent_columns,
        )
    return _HenckyJ2BatchResponse(
        first_piola_stress=first_piola,
        second_piola_stress=second_piola,
        kirchhoff_stress=kirchhoff,
        plastic_log_strain=plastic_log_strain,
        equivalent_plastic_strain=equivalent_plastic_strain,
        tangent=tangent,
    )


def _batch_inverse_3x3(
    value: np.ndarray,
    determinants: np.ndarray,
) -> np.ndarray:
    """Invert a block of 3x3 tensors without one LAPACK dispatch per point."""

    a = value[:, 0, 0]
    b = value[:, 0, 1]
    c = value[:, 0, 2]
    d = value[:, 1, 0]
    e = value[:, 1, 1]
    f = value[:, 1, 2]
    g = value[:, 2, 0]
    h = value[:, 2, 1]
    i = value[:, 2, 2]
    inverse = np.empty_like(value)
    inverse[:, 0, 0] = (e * i - f * h) / determinants
    inverse[:, 0, 1] = (c * h - b * i) / determinants
    inverse[:, 0, 2] = (b * f - c * e) / determinants
    inverse[:, 1, 0] = (f * g - d * i) / determinants
    inverse[:, 1, 1] = (a * i - c * g) / determinants
    inverse[:, 1, 2] = (c * d - a * f) / determinants
    inverse[:, 2, 0] = (d * h - e * g) / determinants
    inverse[:, 2, 1] = (b * g - a * h) / determinants
    inverse[:, 2, 2] = (a * e - b * d) / determinants
    return inverse


def _analytic_hencky_batch_tangent(
    F: np.ndarray,
    rotation: np.ndarray,
    stretch: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    inverse_f: np.ndarray,
    trial_deviator: np.ndarray,
    trial_equivalent: np.ndarray,
    direction: np.ndarray,
    plastic_increment: np.ndarray,
    plastic: np.ndarray,
    kirchhoff_hat: np.ndarray,
    bulk: float,
    shear: float,
    hardening_modulus: float,
    identity: np.ndarray,
    *,
    tangent_columns: tuple[int, ...] | None,
) -> np.ndarray:
    """Batched form of ``_analytic_hencky_dP_dF``."""

    columns = (
        tuple(range(9))
        if tangent_columns is None
        else tuple(int(column) for column in tangent_columns)
    )
    if not columns or any(column < 0 or column >= 9 for column in columns):
        raise ValueError("tangent_columns must contain valid deformation-gradient columns")
    dF = np.eye(9, dtype=float)[np.asarray(columns, dtype=int)].reshape(
        len(columns),
        3,
        3,
    )
    rotated_dF = rotation.transpose(0, 2, 1)[:, None] @ dF[None]
    rotated_dF_principal = np.einsum(
        "pij,pkjl,plm->pkim",
        eigenvectors.transpose(0, 2, 1),
        rotated_dF,
        eigenvectors,
    )
    omega_principal = (
        rotated_dF_principal - rotated_dF_principal.transpose(0, 1, 3, 2)
    ) / (
        eigenvalues[:, None, :, None]
        + eigenvalues[:, None, None, :]
    )
    omega = eigenvectors[:, None] @ omega_principal @ eigenvectors.transpose(0, 2, 1)[:, None]
    d_stretch = rotated_dF - omega @ stretch[:, None]
    d_stretch = 0.5 * (d_stretch + d_stretch.transpose(0, 1, 3, 2))
    d_stretch_principal = (
        eigenvectors.transpose(0, 2, 1)[:, None]
        @ d_stretch
        @ eigenvectors[:, None]
    )
    difference = eigenvalues[:, :, None] - eigenvalues[:, None, :]
    numerator = np.log(eigenvalues)[:, :, None] - np.log(eigenvalues)[:, None, :]
    log_divided_difference = np.empty_like(difference)
    close = np.abs(difference) <= 1.0e-10 * np.maximum(
        1.0,
        eigenvalues[:, :, None],
    )
    np.divide(numerator, difference, out=log_divided_difference, where=~close)
    log_divided_difference[close] = np.broadcast_to(
        1.0 / eigenvalues[:, :, None],
        difference.shape,
    )[close]
    d_log_stretch = (
        eigenvectors[:, None]
        @ (log_divided_difference[:, None] * d_stretch_principal)
        @ eigenvectors.transpose(0, 2, 1)[:, None]
    )
    d_rotation = rotation[:, None] @ omega
    d_trial = (
        bulk
        * np.trace(d_log_stretch, axis1=2, axis2=3)[:, :, None, None]
        * identity
        + 2.0 * shear * (
            d_log_stretch
            - np.trace(d_log_stretch, axis1=2, axis2=3)[:, :, None, None]
            * identity
            / 3.0
        )
    )
    d_trial_deviator = d_trial - (
        np.trace(d_trial, axis1=2, axis2=3)[:, :, None, None] / 3.0
    ) * identity
    safe_equivalent = np.maximum(trial_equivalent, 1.0e-14)
    d_equivalent = (
        1.5
        * np.sum(
            trial_deviator[:, None] * d_trial_deviator,
            axis=(2, 3),
        )
        / safe_equivalent[:, None]
    )
    d_direction = 1.5 * (
        d_trial_deviator / safe_equivalent[:, None, None, None]
        - trial_deviator[:, None, :, :]
        * d_equivalent[:, :, None, None]
        / safe_equivalent[:, None, None, None] ** 2
    )
    d_gamma = d_equivalent / (3.0 * shear + hardening_modulus)
    d_kirchhoff_hat = d_trial - 2.0 * shear * (
        d_gamma[:, :, None, None] * direction[:, None, :, :]
        + plastic_increment[:, None, None, None] * d_direction
    )
    d_kirchhoff_hat = np.where(
        plastic[:, None, None, None],
        d_kirchhoff_hat,
        d_trial,
    )
    d_kirchhoff = (
        d_rotation @ kirchhoff_hat[:, None] @ rotation.transpose(0, 2, 1)[:, None]
        + rotation[:, None] @ d_kirchhoff_hat @ rotation.transpose(0, 2, 1)[:, None]
        + rotation[:, None]
        @ kirchhoff_hat[:, None]
        @ d_rotation.transpose(0, 1, 3, 2)
    )
    inverse_f_transpose = inverse_f.transpose(0, 2, 1)
    first_term = d_kirchhoff @ inverse_f_transpose[:, None]
    kirchhoff = rotation @ kirchhoff_hat @ rotation.transpose(0, 2, 1)
    second_term = (
        kirchhoff[:, None] @ inverse_f_transpose[:, None]
        @ dF.transpose(0, 2, 1)[None]
        @ inverse_f_transpose[:, None]
    )
    d_first_piola = first_term - second_term
    return d_first_piola.reshape(F.shape[0], len(columns), 9).transpose(0, 2, 1)


def _hencky_isotropic_stress(
    logarithmic_strain: np.ndarray,
    bulk: float,
    shear: float,
) -> np.ndarray:
    return bulk * np.trace(logarithmic_strain) * np.eye(3) + 2.0 * shear * _deviatoric(
        logarithmic_strain
    )


def _hencky_isotropic_stress_derivative(
    derivative: np.ndarray,
    bulk: float,
    shear: float,
) -> np.ndarray:
    trace = np.trace(derivative, axis1=1, axis2=2)
    identity = np.eye(3, dtype=float)
    return bulk * trace[:, None, None] * identity + 2.0 * shear * (
        derivative - trace[:, None, None] * identity / 3.0
    )


def _validated_deformation_gradient(value: Any) -> np.ndarray:
    """Validate a deformation gradient without copying read-only consumers."""

    F = np.asarray(value, dtype=float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)):
        raise ValueError("deformation_gradient must be a finite 3x3 array")
    determinant = float(np.linalg.det(F))
    if determinant <= 0.0:
        raise ValueError("deformation_gradient determinant must be > 0")
    return F


def _polar_decomposition(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    left, singular_values, right_transpose = np.linalg.svd(F)
    if np.any(singular_values <= 0.0):
        raise ValueError("deformation_gradient must be nonsingular")
    stretch = (
        right_transpose.T
        @ np.diag(singular_values)
        @ right_transpose
    )
    rotation = (
        F
        @ right_transpose.T
        @ np.diag(1.0 / singular_values)
        @ right_transpose
    )
    return rotation, stretch, float(np.linalg.det(F))


def _symmetric_log(value: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("right stretch must be positive definite")
    result = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _deviatoric(value: np.ndarray) -> np.ndarray:
    return value - np.trace(value) / 3.0 * np.eye(3)


def _positive_value(value: Any, name: str) -> float:
    scalar = _finite_material_scalar(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return scalar


def _nonnegative_value(value: Any, name: str) -> float:
    scalar = _finite_material_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return scalar


def _poisson_ratio(value: Any) -> float:
    scalar = _finite_material_scalar(value, "nu")
    if not -1.0 < scalar < 0.5:
        raise ValueError("nu must satisfy -1 < nu < 0.5")
    return scalar


def _finite_material_scalar(value: Any, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar
