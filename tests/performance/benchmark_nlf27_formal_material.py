"""Measure NLF-27 constitutive and Quad4 performance baselines.

Run from the repository root::

    python tests/performance/benchmark_nlf27_formal_material.py
    python tests/performance/benchmark_nlf27_formal_material.py --repeats 10

The benchmark reports samples rather than enforcing a wall-clock threshold.
Hardware, Python, NumPy and SciPy versions are part of the reported metadata.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import numpy as np

import fem.materials.finite_strain as finite_strain
from fem.elements.quad4 import Quad4Kernel
from fem.materials import (
    FiniteStrainMaterialPointContext,
    HenckyJ2PlasticityMaterialPoint,
    MultiplicativeJ2PlasticityMaterialPoint,
)


def _deformation(strain: float, shear: float = 0.0) -> np.ndarray:
    F = np.diag(
        [
            np.exp(strain),
            np.exp(-0.5 * strain),
            np.exp(-0.5 * strain),
        ]
    )
    F[1, 0] = shear
    return F


def _context(F: np.ndarray) -> FiniteStrainMaterialPointContext:
    return FiniteStrainMaterialPointContext(deformation_gradient=F)


def _formal_points() -> tuple[MultiplicativeJ2PlasticityMaterialPoint, ...]:
    return tuple(
        MultiplicativeJ2PlasticityMaterialPoint(210.0, 0.3, 0.01, 0.1)
        for _ in range(4)
    )


def _hencky_points() -> tuple[HenckyJ2PlasticityMaterialPoint, ...]:
    return tuple(
        HenckyJ2PlasticityMaterialPoint(210.0, 0.3, 0.01, 0.1)
        for _ in range(4)
    )


def _formal_material_trial() -> Any:
    point = MultiplicativeJ2PlasticityMaterialPoint(100.0, 0.3, 0.5, 10.0)
    return point.evaluate_trial(_context(_deformation(0.02, 0.001)))


def _forced_joint_material_trial() -> Any:
    original = finite_strain._solve_multiplicative_plastic_increment

    def force_joint(*args: Any, **kwargs: Any) -> float:
        raise RuntimeError("benchmark fallback")

    finite_strain._solve_multiplicative_plastic_increment = force_joint
    try:
        point = MultiplicativeJ2PlasticityMaterialPoint(100.0, 0.3, 0.5, 10.0)
        return point.evaluate_trial(_context(_deformation(0.02, 0.001)))
    finally:
        finite_strain._solve_multiplicative_plastic_increment = original


def _formal_quad4() -> Any:
    displacement = np.array(
        [[0.0, 0.0], [0.002, 0.0005], [0.002, 0.0], [0.0, 0.0]],
        dtype=float,
    )
    return Quad4Kernel().finite_strain_internal_force_and_tangent(
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=float,
        ),
        displacement,
        _formal_points(),
    )


def _hencky_quad4() -> Any:
    displacement = np.array(
        [[0.0, 0.0], [0.002, 0.0005], [0.002, 0.0], [0.0, 0.0]],
        dtype=float,
    )
    return Quad4Kernel().finite_strain_internal_force_and_tangent(
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=float,
        ),
        displacement,
        _hencky_points(),
    )


def _measure(
    function: Callable[[], Any],
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        samples.append(perf_counter() - started)
    return {
        "samples_seconds": samples,
        "median_seconds": median(samples),
        "minimum_seconds": min(samples),
    }


def run(repeats: int = 5, warmup: int = 1) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "repeats": repeats,
        "warmup": warmup,
        "benchmarks": {
            "formal_material_fixed_direction": _measure(
                _formal_material_trial,
                repeats,
                warmup,
            ),
            "formal_material_joint_fallback": _measure(
                _forced_joint_material_trial,
                repeats,
                warmup,
            ),
            "formal_quad4": _measure(_formal_quad4, repeats, warmup),
            "hencky_quad4": _measure(_hencky_quad4, repeats, warmup),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.warmup < 0:
        raise SystemExit("--repeats must be >= 1 and --warmup must be >= 0")
    print(json.dumps(run(arguments.repeats, arguments.warmup), indent=2))


if __name__ == "__main__":
    main()
