from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PURLIN_IDS = tuple(range(109, 181))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _moments(snapshot: dict[str, Any]) -> dict[tuple[int, int], float]:
    return {
        (int(row["element_id"]), int(row["local_node"])): float(
            row["components"]["MZ"]
        )
        for row in snapshot["section_results"]
        if int(row["element_id"]) in PURLIN_IDS
        and int(row["section_point"]["number"]) == 1
    }


def _statistics(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {
        "mean": sum(values) / len(values),
        "median": median,
        "minimum": min(values),
        "maximum": max(values),
        "maximum_absolute": max(abs(value) for value in values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--phase2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    phase2 = json.loads(args.phase2.read_text(encoding="utf-8"))
    baseline_moments = _moments(baseline)
    phase2_moments = _moments(phase2)
    expected = {(element_id, end) for element_id in PURLIN_IDS for end in (1, 2)}
    if set(baseline_moments) != expected or set(phase2_moments) != expected:
        raise ValueError("purlin snapshot identity is incomplete")

    rows = []
    for element_id in PURLIN_IDS:
        old = [baseline_moments[element_id, end] for end in (1, 2)]
        new = [phase2_moments[element_id, end] for end in (1, 2)]
        rows.append(
            {
                "element_id": element_id,
                "baseline_end_mz": old,
                "phase2_end_mz": new,
                "baseline_midpoint_bias": sum(old) / 2.0,
                "phase2_midpoint_bias": sum(new) / 2.0,
                "baseline_same_sign_3_6_to_3_8_knm": (
                    old[0] * old[1] > 0.0
                    and all(3600.0 <= abs(value) <= 3800.0 for value in old)
                ),
                "phase2_same_sign_3_6_to_3_8_knm": (
                    new[0] * new[1] > 0.0
                    and all(3600.0 <= abs(value) <= 3800.0 for value in new)
                ),
            }
        )

    old_bias = [row["baseline_midpoint_bias"] for row in rows]
    new_bias = [row["phase2_midpoint_bias"] for row in rows]
    payload = {
        "schema": "fem-python-b31-phase2-purlin-bias-v1",
        "inputs": {
            "baseline": {
                "path": str(args.baseline.resolve()),
                "sha256": _sha256(args.baseline),
            },
            "phase2": {
                "path": str(args.phase2.resolve()),
                "sha256": _sha256(args.phase2),
            },
        },
        "element_count": len(rows),
        "baseline_midpoint_bias": _statistics(old_bias),
        "phase2_midpoint_bias": _statistics(new_bias),
        "baseline_abs_midpoint_at_least_3knm_count": sum(
            abs(value) >= 3000.0 for value in old_bias
        ),
        "phase2_abs_midpoint_at_least_3knm_count": sum(
            abs(value) >= 3000.0 for value in new_bias
        ),
        "baseline_same_sign_3_6_to_3_8_knm_count": sum(
            row["baseline_same_sign_3_6_to_3_8_knm"] for row in rows
        ),
        "phase2_same_sign_3_6_to_3_8_knm_count": sum(
            row["phase2_same_sign_3_6_to_3_8_knm"] for row in rows
        ),
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
