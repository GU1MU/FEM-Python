from __future__ import annotations

import pytest

from tests.helpers.file_builders import write_inp


@pytest.fixture
def gui_inp_path(tmp_path):
    """生成可稳定求解的小型平面应力验收模型。"""
    return write_inp(
        tmp_path,
        "gui_plate.inp",
        [
            "*Heading",
            "GUI plate",
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 1., 1.",
            "4, 0., 1.",
            "*Element, type=CPS4, elset=SOLID",
            "1, 1, 2, 3, 4",
            "*Nset, nset=LEFT",
            "1, 4",
            "*Nset, nset=RIGHT",
            "2, 3",
            "*Elset, elset=SOLID",
            "1",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Boundary",
            "LEFT, 1, 2, 0.",
            "*Step, name=Static-1",
            "*Static",
            "*Cload",
            "RIGHT, 1, 10.",
            "*Output, field",
            "*Node Output",
            "U, RF",
            "*Element Output",
            "S",
            "*End Step",
        ],
    )
