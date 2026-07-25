from __future__ import annotations

from collections.abc import Sequence

from tests.helpers.file_builders import write_inp


def write_perforated_plate_style_inp(
    directory,
    filename: str,
    step_lines: Sequence[str],
    *,
    section_data: Sequence[str] = (),
):
    """Write a small Abaqus model with perforated-plate-style surface setup."""
    return write_inp(
        directory,
        filename,
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 2., 0.",
            "4, 0., 1.",
            "5, 1., 1.",
            "6, 2., 1.",
            "*Element, type=CPS4, elset=SOLID",
            "1, 1,2,5,4",
            "2, 2,3,6,5",
            "*Nset, nset=Set-left",
            "1,4",
            "*Nset, nset=Set-right",
            "3,6",
            "*Elset, elset=SOLID",
            "1,2",
            "*Elset, elset=_Surf-right_S2, internal",
            "2",
            "*Surface, type=ELEMENT, name=Surf-right",
            "_Surf-right_S2, S2",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            *section_data,
            "*Boundary",
            "Set-left, 1, 2, 0.",
            "*Step, name=Step-1, nlgeom=NO",
            "*Static",
            *step_lines,
            "*End Step",
        ],
    )


def write_hex20_block_inp(directory, filename: str):
    """Write one constrained and loaded quadratic brick for Agent E2E tests."""

    coordinates = (
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
        (0, -1, -1),
        (1, 0, -1),
        (0, 1, -1),
        (-1, 0, -1),
        (0, -1, 1),
        (1, 0, 1),
        (0, 1, 1),
        (-1, 0, 1),
        (-1, -1, 0),
        (1, -1, 0),
        (1, 1, 0),
        (-1, 1, 0),
    )
    return write_inp(
        directory,
        filename,
        [
            "*Node",
            *[
                f"{node_id}, {x}, {y}, {z}"
                for node_id, (x, y, z) in enumerate(
                    coordinates,
                    start=1,
                )
            ],
            "*Element, type=C3D20, elset=SOLID",
            "1, " + ",".join(str(node_id) for node_id in range(1, 21)),
            "*Nset, nset=Set-fixed",
            "1,4,5,8,12,16,17,20",
            "*Nset, nset=Set-loaded",
            "2,3,6,7,10,14,18,19",
            "*Surface, type=ELEMENT, name=Surf-loaded",
            "SOLID, S4",
            "*Material, name=STEEL",
            "*Elastic",
            "206000., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Boundary",
            "Set-fixed, 1, 3, 0.",
            "*Step, name=Step-1, nlgeom=NO",
            "*Static",
            "*Cload",
            "Set-loaded, 1, 10.",
            "*End Step",
        ],
    )
