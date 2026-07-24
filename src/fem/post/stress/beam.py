from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...boundary.step import boundary_for_step
from ...elements import get_element_kernel
from ...elements.beam_section import axial_stress_extrema, parse_beam2_section
from .._paths import prepare_output_path


BEAM2_NODAL_STRESS_HEADER = (
    "node_id",
    "x",
    "y",
    "z",
    "axial_stress_max",
    "axial_stress_min",
    "axial_stress_abs_max",
)


@dataclass(frozen=True)
class Beam2NodalStress:
    """Axial-stress extrema enveloped at one mesh node."""

    node_id: int
    maximum: float
    minimum: float
    absolute_maximum: float


def nodal_envelope(result: Any) -> tuple[Beam2NodalStress, ...]:
    """Recover Beam2 end stresses and envelope them in mesh node order."""
    mesh = result.model.mesh
    lookup = {int(node.id): node for node in mesh.nodes}
    boundary = boundary_for_step(result.model, result.step)
    line_loads_by_element: dict[int, list[Any]] = {}
    for load in boundary.line_loads:
        line_loads_by_element.setdefault(int(load.elem_id), []).append(load)

    contributions: dict[int, list[tuple[float, float]]] = {
        int(node_id): [] for node_id in mesh.node_ids
    }
    for elem in mesh.elements:
        if str(elem.type).casefold() != "beam2":
            raise ValueError(
                "Beam2 nodal stress recovery requires only Beam2 elements, "
                f"got {elem.type!r}"
            )
        kernel = get_element_kernel(elem.type)
        local_load = np.zeros(12, dtype=float)
        for load in line_loads_by_element.get(int(elem.id), ()):
            local_load += kernel.local_line_load(
                mesh,
                elem,
                load.vector,
                load.coordinate_system,
                lookup,
            )
        end_actions = kernel.local_end_actions(
            mesh,
            elem,
            result.U,
            local_load,
            lookup,
        )
        section = parse_beam2_section(elem.props)
        for node_id, (axial_force, moment_y, moment_z) in zip(
            elem.node_ids,
            end_actions,
        ):
            maximum, minimum, _ = axial_stress_extrema(
                section,
                axial_force,
                moment_y,
                moment_z,
            )
            contributions[int(node_id)].append((maximum, minimum))

    rows: list[Beam2NodalStress] = []
    for node_id in mesh.node_ids:
        node_contributions = contributions[int(node_id)]
        if node_contributions:
            maximum = max(values[0] for values in node_contributions)
            minimum = min(values[1] for values in node_contributions)
        else:
            maximum = minimum = 0.0
        rows.append(
            Beam2NodalStress(
                node_id=int(node_id),
                maximum=float(maximum),
                minimum=float(minimum),
                absolute_maximum=float(max(abs(maximum), abs(minimum))),
            )
        )
    return tuple(rows)


def absolute_maximum(result: Any) -> float:
    """Return the maximum absolute stress from the Beam2 nodal envelope."""
    return max(
        (row.absolute_maximum for row in nodal_envelope(result)),
        default=0.0,
    )


def export_nodal(result: Any, path: str) -> None:
    """Write the Beam2 nodal axial-stress envelope CSV."""
    mesh = result.model.mesh
    lookup = {int(node.id): node for node in mesh.nodes}
    output_path = prepare_output_path(path)
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(BEAM2_NODAL_STRESS_HEADER)
        for row in nodal_envelope(result):
            node = lookup[row.node_id]
            writer.writerow(
                [
                    row.node_id,
                    node.x,
                    node.y,
                    node.z,
                    row.maximum,
                    row.minimum,
                    row.absolute_maximum,
                ]
            )
