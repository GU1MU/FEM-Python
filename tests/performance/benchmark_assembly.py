from __future__ import annotations

import argparse
import gc
import json
import tracemalloc
from time import perf_counter

from fem.assemble import assemble_global_stiffness_sparse
from fem.core.mesh import Element2D, Mesh2D, Node2D


def _make_medium_mixed_plane_mesh(cell_count):
    nodes = [
        Node2D(
            id=row * (cell_count + 1) + column + 1,
            x=float(column),
            y=float(row),
        )
        for row in range(cell_count + 1)
        for column in range(cell_count + 1)
    ]
    elements = []
    element_id = 1
    properties = {
        "E": 210.0,
        "nu": 0.3,
        "thickness": 1.0,
        "plane_type": "stress",
    }
    for row in range(cell_count):
        for column in range(cell_count):
            lower_left = row * (cell_count + 1) + column + 1
            lower_right = lower_left + 1
            upper_left = lower_left + cell_count + 1
            upper_right = upper_left + 1
            if (row + column) % 2 == 0:
                elements.append(
                    Element2D(
                        element_id,
                        [
                            lower_left,
                            lower_right,
                            upper_right,
                            upper_left,
                        ],
                        "Quad4",
                        dict(properties),
                    )
                )
                element_id += 1
            else:
                elements.extend(
                    (
                        Element2D(
                            element_id,
                            [lower_left, lower_right, upper_right],
                            "Tri3",
                            dict(properties),
                        ),
                        Element2D(
                            element_id + 1,
                            [lower_left, upper_right, upper_left],
                            "Tri3",
                            dict(properties),
                        ),
                    )
                )
                element_id += 2
    return Mesh2D(nodes, elements)


def run(cell_count: int) -> dict[str, float | int]:
    mesh = _make_medium_mixed_plane_mesh(cell_count)
    gc.collect()
    tracemalloc.start()
    try:
        started = perf_counter()
        stiffness = assemble_global_stiffness_sparse(mesh)
        elapsed = perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "elements": len(mesh.elements),
        "dofs": mesh.num_dofs,
        "nnz": stiffness.nnz,
        "assembly_seconds": elapsed,
        "assembly_peak_bytes": peak_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure sparse assembly time and peak traced memory."
    )
    parser.add_argument("--cells", type=int, default=12)
    arguments = parser.parse_args()
    if arguments.cells < 1:
        parser.error("--cells must be positive")
    print(json.dumps(run(arguments.cells), ensure_ascii=False))


if __name__ == "__main__":
    main()
