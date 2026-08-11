from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PHASE_DIR = Path(__file__).resolve().parent
REPOSITORY = PHASE_DIR.parents[3]
sys.path.insert(0, str(REPOSITORY / "src"))

from fem.io import inp  # noqa: E402


CASES = {
    101: "default-n1",
    102: "orientation-node-over-section-n1",
    103: "node-normal-over-generated-normal",
    104: "element-normal-over-node-normal",
    105: "reversed-connectivity",
    201: "averaged-normal-0deg",
    202: "averaged-normal-10deg",
    203: "averaged-normal-15deg",
    301: "non-clique-split-0deg",
    302: "non-clique-split-10deg",
    303: "non-clique-split-30deg",
    401: "disjoint-group-0deg",
    402: "disjoint-group-10deg",
    403: "disjoint-group-40deg",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = PHASE_DIR / "phase3_frame_sources.inp"
    oracle_path = PHASE_DIR / "abaqus_frame_oracle.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    result = inp.read_with_report(source)

    rows = []
    program_frames = {}
    maximum = 0.0
    for element in sorted(result.model.mesh.elements, key=lambda item: item.id):
        element_id = int(element.id)
        field = element.props["beam_frame_field"]
        program = np.asarray(field.rotation_at_fraction(0.5), dtype=float)
        abaqus = np.asarray(oracle["frames"][str(element_id)], dtype=float)
        error = float(np.max(np.abs(program - abaqus)))
        maximum = max(maximum, error)
        program_frames[str(element_id)] = {
            "case": CASES[element_id],
            "start": field.start.rotation.tolist(),
            "midpoint": program.tolist(),
            "end": field.end.rotation.tolist(),
        }
        rows.append(
            {
                "element_id": element_id,
                "case": CASES[element_id],
                "max_direction_cosine_absolute_error": error,
            }
        )

    (PHASE_DIR / "program_frame_snapshot.json").write_text(
        json.dumps(program_frames, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = {
        "passed": maximum <= 1.0e-10,
        "threshold": 1.0e-10,
        "max_direction_cosine_absolute_error": maximum,
        "cases": rows,
        "inputs": {
            "abaqus_oracle": {
                "path": oracle_path.name,
                "sha256": _sha256(oracle_path),
            },
            "inp": {"path": source.name, "sha256": _sha256(source)},
        },
        "notices": [notice.code for notice in result.notices],
    }
    (PHASE_DIR / "frame_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not gate["passed"]:
        raise SystemExit(
            "frame oracle failed: {:.17g} > {:.17g}".format(
                maximum,
                gate["threshold"],
            )
        )


if __name__ == "__main__":
    main()
