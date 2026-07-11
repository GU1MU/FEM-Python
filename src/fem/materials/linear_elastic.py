from __future__ import annotations

import numpy as np

from ..core.model import MaterialDefinition


def _positive_finite(value: float, name: str) -> float:
    """Return a finite, strictly positive material constant."""
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return converted


def _poisson_ratio(value: float) -> float:
    """Return a finite isotropic Poisson ratio in its admissible range."""
    converted = float(value)
    if not np.isfinite(converted) or not -1.0 < converted < 0.5:
        raise ValueError(
            f"nu must be finite and satisfy -1 < nu < 0.5, got {value!r}"
        )
    return converted


def _density(value: float) -> float:
    """Return a finite, non-negative mass density."""
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"rho must be finite and >= 0, got {value!r}")
    return converted


def _elastic_constants(E: float, nu: float) -> tuple[float, float]:
    """Return validated isotropic elastic constants."""
    return _positive_finite(E, "E"), _poisson_ratio(nu)


def material(
    name: str,
    E: float,
    nu: float,
    rho: float | None = None,
    **properties,
) -> MaterialDefinition:
    """Return a named linear elastic material."""
    data = dict(properties)
    data["E"], data["nu"] = _elastic_constants(E, nu)
    if rho is not None:
        data["rho"] = _density(rho)
    return MaterialDefinition(str(name), data)


def plane_stress_matrix(E: float, nu: float) -> np.ndarray:
    """Return isotropic plane stress constitutive matrix."""
    E, nu = _elastic_constants(E, nu)
    coef = E / (1.0 - nu ** 2)
    return coef * np.array([
        [1.0,    nu,           0.0],
        [nu,     1.0,          0.0],
        [0.0,    0.0, (1.0 - nu) / 2.0],
    ], dtype=float)


def plane_strain_matrix(E: float, nu: float) -> np.ndarray:
    """Return isotropic plane strain constitutive matrix."""
    E, nu = _elastic_constants(E, nu)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    return np.array([
        [lam + 2.0 * mu, lam,            0.0],
        [lam,            lam + 2.0 * mu, 0.0],
        [0.0,            0.0,            mu],
    ], dtype=float)


def plane_matrix(E: float, nu: float, plane_type: str) -> np.ndarray:
    """Return plane stress or plane strain constitutive matrix."""
    pt = str(plane_type).lower()
    if pt.startswith("stress"):
        return plane_stress_matrix(E, nu)
    if pt.startswith("strain"):
        return plane_strain_matrix(E, nu)
    raise ValueError(
        f"plane_type {plane_type!r} is invalid; expected 'stress' or 'strain'"
    )


def solid_3d_matrix(E: float, nu: float) -> np.ndarray:
    """Return isotropic 3D constitutive matrix."""
    E, nu = _elastic_constants(E, nu)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    return np.array([
        [lam + 2.0 * mu, lam,            lam,            0.0, 0.0, 0.0],
        [lam,            lam + 2.0 * mu, lam,            0.0, 0.0, 0.0],
        [lam,            lam,            lam + 2.0 * mu, 0.0, 0.0, 0.0],
        [0.0,            0.0,            0.0,            mu,  0.0, 0.0],
        [0.0,            0.0,            0.0,            0.0, mu,  0.0],
        [0.0,            0.0,            0.0,            0.0, 0.0, mu],
    ], dtype=float)
