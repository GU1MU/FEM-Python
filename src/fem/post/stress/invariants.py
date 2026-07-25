from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StressInvariants:
    """Derived scalar values for one complete stress tensor."""

    mises: float
    max_principal: float
    mid_principal: float
    min_principal: float


def plane_out_of_plane_stress(
    sig_x: float,
    sig_y: float,
    plane_type: str = "stress",
    nu: float = 0.0,
) -> float:
    """Return S33 for plane stress or plane strain."""
    if str(plane_type).lower().startswith("strain"):
        return float(nu) * (float(sig_x) + float(sig_y))
    return 0.0


def complete_plane_components(
    components,
    plane_type: str = "stress",
    nu: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return canonical S11, S22, S33, S12 plane components."""
    sig_x, sig_y, tau_xy = (float(value) for value in components)
    return (
        sig_x,
        sig_y,
        plane_out_of_plane_stress(sig_x, sig_y, plane_type, nu),
        tau_xy,
    )


def derive_stress_invariants(
    components,
    component_names,
) -> StressInvariants:
    """Calculate Mises and principal stresses from named tensor components."""
    values = {
        str(name).upper(): float(value)
        for name, value in zip(component_names, components)
    }
    aliases = {
        "SIG_X": "S11",
        "SIG_Y": "S22",
        "SIG_Z": "S33",
        "TAU_XY": "S12",
        "TAU_YZ": "S23",
        "TAU_ZX": "S13",
    }
    for source, target in aliases.items():
        if source in values and target not in values:
            values[target] = values[source]

    s11 = values.get("S11", 0.0)
    s22 = values.get("S22", 0.0)
    s33 = values.get("S33", 0.0)
    s12 = values.get("S12", 0.0)
    s13 = values.get("S13", 0.0)
    s23 = values.get("S23", 0.0)
    tensor = np.array(
        [
            [s11, s12, s13],
            [s12, s22, s23],
            [s13, s23, s33],
        ],
        dtype=float,
    )
    principal = np.linalg.eigvalsh(tensor)
    return StressInvariants(
        mises=von_mises_3d(s11, s22, s33, s12, s23, s13),
        max_principal=float(principal[2]),
        mid_principal=float(principal[1]),
        min_principal=float(principal[0]),
    )


def von_mises_plane(
    sig_x: float,
    sig_y: float,
    tau_xy: float,
    plane_type: str = "stress",
    nu: float = 0.0,
) -> float:
    """Return plane stress or plane strain von Mises stress."""
    sig_x = float(sig_x)
    sig_y = float(sig_y)
    tau_xy = float(tau_xy)
    sig_z = plane_out_of_plane_stress(sig_x, sig_y, plane_type, nu)
    return von_mises_3d(sig_x, sig_y, sig_z, tau_xy, 0.0, 0.0)


def von_mises_3d(
    sig_x: float,
    sig_y: float,
    sig_z: float,
    tau_xy: float,
    tau_yz: float,
    tau_zx: float,
) -> float:
    """Return 3D von Mises stress."""
    return float(np.sqrt(
        0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
        + 3.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
    ))
