from __future__ import annotations

import csv
from typing import Dict, List, Optional

from ..materials import linear_elastic as linear_elastic_material


def read(path: str) -> Dict[int, Dict[str, str]]:
    """Read material CSV into a dict keyed by material_id."""
    materials: Dict[int, Dict[str, str]] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not row:
                continue

            mid_raw = row.get("material_id")
            if mid_raw is None or mid_raw.strip() == "":
                continue

            mid = int(mid_raw)
            materials[mid] = {k: (v.strip() if v is not None else "") for k, v in row.items()}

    return materials


def linear_elastic(path: str, name: str):
    """Read one named linear elastic material definition from a CSV catalog."""

    row = _get_material_by_name(read(path), name)
    E = _require_float_from_material(row, ["E"], name)
    nu = _require_float_from_material(row, ["nu", "poisson"], name)
    rho = _get_float_from_material(row, ["rho"])
    material_name = row.get("name", name) or name
    return linear_elastic_material.material(material_name, E=E, nu=nu, rho=rho)


def _get_material_by_name(
    materials: Dict[int, Dict[str, str]],
    name: str,
) -> Dict[str, str]:
    for row in materials.values():
        if row.get("name", "").strip() == name:
            return row
    raise KeyError(f"material {name} is not defined")


def _require_float_from_material(
    mat_row: Dict[str, str],
    keys: List[str],
    name: str,
) -> float:
    value = _get_float_from_material(mat_row, keys)
    if value is None:
        raise KeyError(f"material {name} is missing {'/'.join(keys)}")
    return value


def _get_float_from_material(
    mat_row: Dict[str, str],
    keys: List[str],
) -> Optional[float]:

    # 做一个 key.lower() -> 原始 key 的映射，方便大小写不敏感
    lower_map = {k.lower(): k for k in mat_row.keys()}

    for key in keys:
        kl = key.lower()
        if kl in lower_map:
            raw = mat_row[lower_map[kl]]
            if raw == "":
                continue
            try:
                return float(raw)
            except ValueError:
                continue
    return None
