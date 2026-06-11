from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from .invariants import von_mises_3d


SOLID_STRESS_NAMES = ("sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx")
_TINY = 1.0e-12


@dataclass(frozen=True)
class StressAveragingPolicy:
    """Configuration for region-aware solid nodal stress averaging."""
    threshold: float = 75.0


@dataclass(frozen=True)
class RegionKey:
    """Averaging boundary key for solid nodal stress contributions."""
    material: Hashable = ""
    section: Hashable = ""
    element_type: Hashable = ""


@dataclass(frozen=True)
class ElementNodalContribution:
    """One element-local nodal stress contribution."""
    source_elem_id: int
    source_local_node: int
    original_node_id: int
    region_key: RegionKey
    x: float
    y: float
    z: float
    stress: Sequence[float]

    @property
    def stress_array(self) -> np.ndarray:
        """Return stress as a six-component array."""
        values = np.asarray(self.stress, dtype=float).ravel()
        if values.shape[0] != 6:
            raise ValueError(f"solid stress must have 6 components, got {values.shape[0]}")
        return values

    @property
    def mises(self) -> float:
        """Return von Mises stress for this contribution."""
        return _mises_from_components(self.stress_array)


@dataclass(frozen=True)
class AveragedNodalStressRow:
    """Region-aware averaged nodal stress row for CSV/VTK export."""
    source_elem_id: int
    source_local_node: int
    original_node_id: int
    region_id: int
    cluster_id: int
    material_id: int
    section_id: int
    element_type_id: int
    x: float
    y: float
    z: float
    stress: tuple[float, float, float, float, float, float]
    mises: float


def average_solid_nodal_contributions(
    contributions: Sequence[ElementNodalContribution],
    policy: StressAveragingPolicy | None = None,
) -> list[AveragedNodalStressRow]:
    """Return region-aware averaged solid nodal stress rows."""
    policy = policy or StressAveragingPolicy()
    grouped = _group_contributions(contributions)
    region_mises_ranges = _region_mises_ranges(contributions)
    encoders = _RegionEncoders.from_contributions(contributions)
    rows: list[AveragedNodalStressRow] = []

    for (original_node_id, region_key), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], repr(item[0][1]))
    ):
        clusters = _cluster_by_mises(group, policy.threshold, region_mises_ranges[region_key])
        for cluster_id, cluster in enumerate(clusters):
            mean_stress = np.mean([item.stress_array for item in cluster], axis=0)
            mean_stress_tuple = tuple(float(value) for value in mean_stress)
            mises = _mises_from_components(mean_stress)
            for item in sorted(cluster, key=lambda c: (c.source_elem_id, c.source_local_node)):
                rows.append(
                    AveragedNodalStressRow(
                        source_elem_id=int(item.source_elem_id),
                        source_local_node=int(item.source_local_node),
                        original_node_id=int(original_node_id),
                        region_id=encoders.region_id(region_key),
                        cluster_id=cluster_id,
                        material_id=encoders.material_id(region_key.material),
                        section_id=encoders.section_id(region_key.section),
                        element_type_id=encoders.element_type_id(region_key.element_type),
                        x=float(item.x),
                        y=float(item.y),
                        z=float(item.z),
                        stress=mean_stress_tuple,
                        mises=mises,
                    )
                )

    return rows


def _group_contributions(
    contributions: Sequence[ElementNodalContribution],
) -> dict[tuple[int, RegionKey], list[ElementNodalContribution]]:
    grouped: dict[tuple[int, RegionKey], list[ElementNodalContribution]] = {}
    for contribution in contributions:
        key = (int(contribution.original_node_id), contribution.region_key)
        grouped.setdefault(key, []).append(contribution)
    return grouped


def _region_mises_ranges(
    contributions: Sequence[ElementNodalContribution],
) -> dict[RegionKey, tuple[float, float]]:
    grouped: dict[RegionKey, list[float]] = {}
    for contribution in contributions:
        grouped.setdefault(contribution.region_key, []).append(contribution.mises)
    return {
        key: (min(values), max(values))
        for key, values in grouped.items()
    }


def _cluster_by_mises(
    contributions: Sequence[ElementNodalContribution],
    threshold: float,
    region_mises_range: tuple[float, float],
) -> list[list[ElementNodalContribution]]:
    if not contributions:
        return []
    if threshold < 0.0:
        raise ValueError("averaging threshold must be non-negative")

    ordered = sorted(contributions, key=lambda item: (item.source_elem_id, item.source_local_node))
    if _mises_variation(ordered, region_mises_range) <= threshold:
        return [ordered]
    return [[contribution] for contribution in ordered]


def _mises_variation(
    contributions: Sequence[ElementNodalContribution],
    region_mises_range: tuple[float, float],
) -> float:
    mises_values = [item.mises for item in contributions]
    lo = min(mises_values)
    hi = max(mises_values)
    region_lo, region_hi = region_mises_range
    denom = region_hi - region_lo
    if abs(denom) <= _TINY:
        return 0.0 if abs(hi - lo) <= _TINY else float("inf")
    return 100.0 * (hi - lo) / denom


def _mises_from_components(stress: Sequence[float]) -> float:
    sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = np.asarray(stress, dtype=float)
    return float(von_mises_3d(sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx))


@dataclass(frozen=True)
class _RegionEncoders:
    region_ids: dict[RegionKey, int]
    material_ids: dict[Hashable, int]
    section_ids: dict[Hashable, int]
    element_type_ids: dict[Hashable, int]

    @classmethod
    def from_contributions(
        cls,
        contributions: Sequence[ElementNodalContribution],
    ) -> "_RegionEncoders":
        regions = sorted({item.region_key for item in contributions}, key=repr)
        materials = sorted({item.region_key.material for item in contributions}, key=repr)
        sections = sorted({item.region_key.section for item in contributions}, key=repr)
        element_types = sorted({item.region_key.element_type for item in contributions}, key=repr)
        return cls(
            region_ids={key: idx + 1 for idx, key in enumerate(regions)},
            material_ids={key: idx + 1 for idx, key in enumerate(materials)},
            section_ids={key: idx + 1 for idx, key in enumerate(sections)},
            element_type_ids={key: idx + 1 for idx, key in enumerate(element_types)},
        )

    def region_id(self, key: RegionKey) -> int:
        return self.region_ids[key]

    def material_id(self, key: Hashable) -> int:
        return self.material_ids[key]

    def section_id(self, key: Hashable) -> int:
        return self.section_ids[key]

    def element_type_id(self, key: Hashable) -> int:
        return self.element_type_ids[key]
