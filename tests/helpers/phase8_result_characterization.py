from __future__ import annotations

import numpy as np

from fem.core.mesh import (
    Element2D,
    Element3D,
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
)
from fem.core.model import FEMModel
from fem.core.result import ModelResult


def make_continuum_nodal_semantics_result() -> ModelResult:
    """Build a three-element, two-region fan with one isolated mesh node."""

    common = {
        "E": 100.0,
        "nu": 0.0,
        "plane_type": "stress",
        "thickness": 1.0,
    }
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
            Node2D(6, 2.0, 0.0),
            Node2D(7, 0.0, -2.0),
            Node2D(8, 3.0, 3.0),
        ],
        elements=[
            Element2D(
                1,
                [1, 2, 3],
                "Tri3",
                {**common, "material_id": 1},
            ),
            Element2D(
                2,
                [1, 4, 5],
                "Tri3",
                {**common, "material_id": 1},
            ),
            Element2D(
                3,
                [1, 7, 6],
                "Tri3",
                {**common, "material_id": 2},
            ),
        ],
    )
    displacement = np.zeros(mesh.num_dofs, dtype=float)
    displacement[mesh.global_dof(2, 0)] = 0.1
    displacement[mesh.global_dof(4, 0)] = -0.3
    displacement[mesh.global_dof(6, 0)] = 1.0
    model = FEMModel(mesh=mesh, name="phase8-continuum-oracle")
    return ModelResult(
        model,
        None,
        displacement,
        np.zeros(mesh.num_dofs, dtype=float),
    )


def make_truss_field_characterization_result() -> ModelResult:
    """Build a literal 10-unit axial-stress Truss2 result."""

    mesh = Mesh3D(
        nodes=[
            Node3D(10, 0.0, 0.0, 0.0),
            Node3D(20, 2.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(
                30,
                [10, 20],
                "Truss2",
                {"E": 100.0, "area": 2.0},
            )
        ],
    )
    model = FEMModel(mesh=mesh, name="phase8-truss-oracle")
    return ModelResult(
        model,
        None,
        np.asarray([0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )


def make_beam_field_characterization_result() -> ModelResult:
    """Build a Beam2 result with 10 axial stress and one bending increment."""

    mesh = Mesh3D(
        nodes=[
            Node3D(10, 0.0, 0.0, 0.0),
            Node3D(20, 2.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(
                30,
                [10, 20],
                "Beam2",
                {
                    "E": 100.0,
                    "nu": 0.25,
                    "section_type": "rectangle",
                    "height": 2.0,
                    "width": 1.0,
                },
            )
        ],
        dofs_per_node=6,
    )
    model = FEMModel(mesh=mesh, name="phase8-beam-oracle")
    return ModelResult(
        model,
        None,
        np.asarray([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.02,
            0.0,
            0.0,
            0.0,
            0.02,
        ]),
        np.arange(1.0, 13.0),
    )
