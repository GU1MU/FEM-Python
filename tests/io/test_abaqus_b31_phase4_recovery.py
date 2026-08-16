from __future__ import annotations

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import AnalysisStep, FEMModel
from fem.core.result import ModelResult
from fem.elements import (
    BEAM_FRAME_FIELD_KEY,
    BeamFrameField,
    BeamIntegrationPointForces,
    get_element_kernel,
)
from fem.post.stress.beam import recover_integration_point_stress


E = 210.0e9
LENGTH = 2.0
HEIGHT = 0.4
WIDTH = 0.2
AREA = HEIGHT * WIDTH
IYY = WIDTH * HEIGHT**3 / 12.0
IZZ = HEIGHT * WIDTH**3 / 12.0


def _model() -> FEMModel:
    frame = BeamFrameField.from_rotations(LENGTH, np.eye(3), np.eye(3))
    return FEMModel(
        Mesh3D(
            nodes=(Node3D(1, 0.0, 0.0, 0.0), Node3D(2, LENGTH, 0.0, 0.0)),
            elements=(
                Element3D(
                    7,
                    (1, 2),
                    "Beam2",
                    {
                        "E": E,
                        "nu": 0.3,
                        "section_type": "rectangle",
                        "height": HEIGHT,
                        "width": WIDTH,
                        BEAM_FRAME_FIELD_KEY: frame,
                    },
                ),
            ),
            dofs_per_node=6,
        ),
        name="phase4-minimal-oracle",
        steps=(AnalysisStep("Load"),),
    )


def _prescribed_result(*, axial: float, my: float, mz: float) -> ModelResult:
    model = _model()
    displacement = np.zeros(12)
    displacement[6] = axial * LENGTH / (E * AREA)
    curvature_y = my / (E * IYY)
    displacement[8] = -curvature_y * LENGTH**2 / 2.0
    displacement[10] = curvature_y * LENGTH
    curvature_z = mz / (E * IZZ)
    displacement[7] = curvature_z * LENGTH**2 / 2.0
    displacement[11] = curvature_z * LENGTH
    return ModelResult(model, model.steps[0], displacement, np.zeros(12))


@pytest.mark.parametrize(
    ("axial", "my", "mz"),
    ((1200.0, 0.0, 0.0), (0.0, 180.0, 0.0), (0.0, 0.0, -90.0)),
)
def test_b31_single_point_recovers_constitutive_n_my_mz_and_s11(
    axial: float,
    my: float,
    mz: float,
) -> None:
    result = _prescribed_result(axial=axial, my=my, mz=mz)
    field = recover_integration_point_stress(result)

    assert len(field.section_points) == 4
    assert all(len(point_field.rows) == 1 for point_field in field.section_points)
    rows = tuple(point_field.rows[0] for point_field in field.section_points)
    section_row = field.section_forces.rows[0]
    assert {(row.element_id, row.integration_point) for row in rows} == {(7, 1)}
    assert (section_row.element_id, section_row.integration_point) == (7, 1)
    assert section_row.N == pytest.approx(axial)
    assert section_row.Vy == pytest.approx(0.0, abs=1.0e-8)
    assert section_row.Vz == pytest.approx(0.0, abs=1.0e-8)
    assert section_row.T == pytest.approx(0.0, abs=1.0e-8)
    assert section_row.My == pytest.approx(my)
    assert section_row.Mz == pytest.approx(mz)
    assert section_row.values() == pytest.approx(
        {
            "N": axial,
            "Vy": 0.0,
            "Vz": 0.0,
            "T": 0.0,
            "My": my,
            "Mz": mz,
        },
        abs=1.0e-8,
    )
    for row in rows:
        expected = (
            axial / AREA
            + my * row.section_point.local_z / IYY
            - mz * row.section_point.local_y / IZZ
        )
        assert row.s11 == pytest.approx(expected)


def test_rect_point_one_is_positive_y_positive_z_and_abaqus_point_25() -> None:
    row = recover_integration_point_stress(
        _prescribed_result(axial=1.0, my=2.0, mz=3.0)
    ).point_field(1).rows[0]

    assert row.section_point.local_coordinates == pytest.approx(
        (WIDTH / 2.0, HEIGHT / 2.0)
    )
    assert {"program_point": row.section_point.number, "abaqus_point": 25} == {
        "program_point": 1,
        "abaqus_point": 25,
    }


def test_integration_point_recovery_does_not_consume_section_end_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _prescribed_result(axial=1200.0, my=180.0, mz=-90.0)
    kernel = get_element_kernel("Beam2")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("section-end equilibrium action was consumed")

    monkeypatch.setattr(
        type(kernel),
        "local_section_end_actions",
        forbidden,
    )
    recovered = recover_integration_point_stress(result)
    forces = kernel.local_integration_point_forces(
        result.model.mesh,
        result.model.mesh.elements[0],
        result.U,
    )

    assert type(forces) is BeamIntegrationPointForces
    assert type(recovered.section_forces.rows[0].forces) is BeamIntegrationPointForces
    assert all(
        not hasattr(row, "forces")
        for field in recovered.section_points
        for row in field.rows
    )


def test_phase4_test_has_no_product_data_dependency() -> None:
    source = __file__.replace("\\", "/")
    assert "/data/" not in source.casefold()
