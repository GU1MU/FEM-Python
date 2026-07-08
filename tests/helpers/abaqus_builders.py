from __future__ import annotations

from collections.abc import Sequence

from tests.helpers.file_builders import write_inp


def write_perforated_plate_style_inp(directory, filename: str, step_lines: Sequence[str]):
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
            "*Boundary",
            "Set-left, 1, 2, 0.",
            "*Step, name=Step-1, nlgeom=NO",
            "*Static",
            *step_lines,
            "*End Step",
        ],
    )
