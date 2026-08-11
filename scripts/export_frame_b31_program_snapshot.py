"""Export one FEM-Python Beam2 solve as a deterministic validation snapshot.

This is the program-side companion to ``compare_frame_b31_odb.py``.  It is
deliberately a command-line engineering tool, not a pytest helper.  The current
Beam2 public stress recovery is located at element section ends; that identity
is retained in the snapshot and must not be compared with Abaqus integration
point values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.elements.beam_section import parse_beam2_section
from fem.io import inp
from fem.post.stress.beam import recover_section_stress
from fem.solvers import static_linear


SCRIPT_VERSION = "1.0.0"
SNAPSHOT_SCHEMA = "fem-python-b31-validation-snapshot-v1"
SECTION_TYPE_NAMES = {
    "rectangle": "RECT",
    "solid_circle": "CIRC",
    "hollow_circle": "THICK PIPE",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outside_data(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    data = (root / "data").resolve()
    if resolved == data or data in resolved.parents:
        raise ValueError("validation output must be outside the repository data directory")
    return resolved


def _vector_totals(mesh: Any, values: np.ndarray) -> dict[str, list[float]]:
    force = np.zeros(3, dtype=float)
    moment = np.zeros(3, dtype=float)
    lookup = {int(node.id): node for node in mesh.nodes}
    for node_id in mesh.node_ids:
        dofs = tuple(mesh.node_dofs(node_id))
        nodal_force = np.asarray(values[list(dofs[:3])], dtype=float)
        nodal_moment = np.asarray(values[list(dofs[3:6])], dtype=float)
        node = lookup[int(node_id)]
        coordinates = np.asarray((node.x, node.y, node.z), dtype=float)
        force += nodal_force
        moment += nodal_moment + np.cross(coordinates, nodal_force)
    return {
        "force": [float(value) for value in force],
        "moment_about_origin": [float(value) for value in moment],
    }


def _section_metadata(section: Any) -> dict[str, Any]:
    dimensions = {
        name: float(value)
        for name in (
            "radius",
            "outer_radius",
            "inner_radius",
            "height",
            "width",
        )
        if (value := getattr(section, name)) is not None
    }
    return {
        "type": SECTION_TYPE_NAMES[section.section_type],
        "dimensions": dimensions,
    }


def build_snapshot(
    input_path: Path,
    *,
    step: str | int | None = None,
    length_unit: str = "m",
    force_unit: str = "N",
    stress_unit: str = "Pa",
) -> dict[str, Any]:
    """Solve ``input_path`` and return the canonical validation snapshot."""

    source = input_path.resolve()
    model = inp.read(source)
    result = static_linear.solve(model, step=step)
    mesh = result.model.mesh
    if mesh.dofs_per_node != 6:
        raise ValueError("B31 validation snapshots require six DOFs per node")

    nodes = []
    lookup = {int(node.id): node for node in mesh.nodes}
    for node_id in sorted(int(value) for value in mesh.node_ids):
        node = lookup[node_id]
        dofs = tuple(mesh.node_dofs(node_id))
        nodes.append(
            {
                "node_id": node_id,
                "coordinates": [float(node.x), float(node.y), float(node.z)],
                "U": [float(result.U[dof]) for dof in dofs[:3]],
                "UR": [float(result.U[dof]) for dof in dofs[3:6]],
                "RF": [float(result.reactions[dof]) for dof in dofs[:3]],
                "RM": [float(result.reactions[dof]) for dof in dofs[3:6]],
            }
        )

    recovery = recover_section_stress(result)
    end_by_identity = {
        (row.element_id, row.local_node): row
        for row in recovery.section_end.rows
    }
    element_lookup = {int(element.id): element for element in mesh.elements}
    section_by_element = {
        element_id: parse_beam2_section(element.props)
        for element_id, element in element_lookup.items()
    }
    section_results = []
    for point_field in sorted(
        recovery.section_points,
        key=lambda field: field.point_number,
    ):
        for row in sorted(
            point_field.rows,
            key=lambda item: (item.element_id, item.local_node),
        ):
            end = end_by_identity[(row.element_id, row.local_node)]
            section_results.append(
                {
                    "position": "SECTION_END",
                    "element_id": row.element_id,
                    "local_node": row.local_node,
                    "node_id": row.node_id,
                    "integration_point": None,
                    "section": _section_metadata(
                        section_by_element[row.element_id]
                    ),
                    "section_point": {
                        "number": row.section_point.number,
                        "local_y": row.section_point.local_y,
                        "local_z": row.section_point.local_z,
                    },
                    "components": {
                        "S11": row.s11,
                        "S12": row.s12,
                        "Mises": row.mises,
                        "MaxPrincipal": row.max_principal,
                        "N": end.N,
                        "VY": end.Vy,
                        "VZ": end.Vz,
                        "MY": end.My,
                        "MZ": end.Mz,
                        "T": end.T,
                    },
                }
            )

    boundary = boundary_for_step(result.model, result.step)
    applied = build_load_vector(mesh, boundary)
    selected_step = getattr(result.step, "name", None)
    return {
        "schema": SNAPSHOT_SCHEMA,
        "producer": {
            "name": "export_frame_b31_program_snapshot.py",
            "version": SCRIPT_VERSION,
        },
        "input": {"path": str(source), "sha256": _sha256(source)},
        "step": selected_step,
        "units": {
            "length": length_unit,
            "force": force_unit,
            "stress": stress_unit,
            "rotation": "rad",
        },
        "positions": {
            "nodal": "NODE",
            "section_results": "SECTION_END",
            "integration_point_available": False,
        },
        "counts": {
            "nodes": len(nodes),
            "elements": len(element_lookup),
            "section_result_rows": len(section_results),
        },
        "totals": {
            "applied": _vector_totals(mesh, applied),
            "reaction": _vector_totals(mesh, result.reactions),
        },
        "nodes": nodes,
        "section_results": section_results,
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve an INP and export a FEM-Python B31 validation snapshot."
    )
    parser.add_argument("--inp", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step", default=None)
    parser.add_argument("--length-unit", default="m")
    parser.add_argument("--force-unit", default="N")
    parser.add_argument("--stress-unit", default="Pa")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    root = Path(__file__).resolve().parents[1]
    output = _outside_data(args.output, root)
    payload = build_snapshot(
        args.inp,
        step=args.step,
        length_unit=args.length_unit,
        force_unit=args.force_unit,
        stress_unit=args.stress_unit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    output.write_text(text + "\n", encoding="utf-8")
    print("Program validation snapshot: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
