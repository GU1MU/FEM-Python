from __future__ import annotations

import numpy as np
import pytest

from fem.analysis.compilation.sections import resolve_section_properties
from fem.application import run_static_preflight, solve_analysis
from fem.materials import (
    ELASTIC_BEHAVIOR_ID,
    NEO_HOOKEAN_BEHAVIOR_ID,
    PLASTIC_BEHAVIOR_ID,
    MaterialPointInput,
    NeoHookeanMaterial,
    material_behavior_diagnostics,
)
from fem.model import (
    ElementSet,
    FEMModel,
    MaterialBehavior,
    MaterialDefinition,
    NodeSet,
    SectionAssignment,
    authoring as steps,
)
from fem.physics.mechanics.kinematics import KinematicPoint
from tests.helpers.mesh_builders import make_unit_hex8_mesh


def _point(deformation_gradient: np.ndarray) -> KinematicPoint:
    dimension = int(deformation_gradient.shape[0])
    return KinematicPoint(
        reference_shape_gradients=np.zeros((dimension, 1)),
        deformation_gradient=deformation_gradient,
        green_lagrange_strain=0.5
        * (deformation_gradient.T @ deformation_gradient - np.eye(dimension)),
        reference_jacobian_determinant=1.0,
    )


def test_material_definition_materializes_explicit_neo_hookean_behavior() -> None:
    material = MaterialDefinition(
        "Rubber",
        {"C10": 2.0, "D1": 4.0},
        constitutive_model="neo_hookean",
    )

    assert material.constitutive_model == "neo_hookean"
    assert tuple(item.behavior_id for item in material.behaviors) == (
        NEO_HOOKEAN_BEHAVIOR_ID,
    )
    assert material.properties["C10"] == 2.0
    assert material.properties["D1"] == 4.0


def test_neo_hookean_returns_zero_identity_stress_and_finite_tangent() -> None:
    identity = np.eye(3)
    material = NeoHookeanMaterial(2.0, 4.0)
    response = material.evaluate(
        MaterialPointInput(kinematics=_point(identity))
    )

    assert response.stress.shape == (3, 3)
    assert response.tangent.shape == (9, 9)
    np.testing.assert_allclose(response.stress, 0.0, atol=1.0e-12)
    assert np.all(np.isfinite(response.tangent))
    assert response.outputs["return_algorithm"] == "neo_hookean"


@pytest.mark.parametrize("plane_type", ("stress", "strain"))
def test_neo_hookean_supports_plane_response(plane_type: str) -> None:
    deformation_gradient = np.array([[1.08, 0.02], [0.0, 0.96]])
    material = NeoHookeanMaterial(1.5, 3.0, plane_type=plane_type)
    response = material.evaluate(
        MaterialPointInput(kinematics=_point(deformation_gradient))
    )

    assert response.stress.shape == (3, 3)
    assert response.tangent.shape == (9, 9)
    assert np.all(np.isfinite(response.stress))
    assert np.all(np.isfinite(response.tangent))


def test_neo_hookean_section_validation_uses_c10_and_d1() -> None:
    resolved = resolve_section_properties(
        "Tet4",
        {"C10": 1.0, "D1": 2.0},
        "solid",
        {},
        constitutive_model="neo_hookean",
    )

    assert resolved.effective_properties["C10"] == 1.0
    assert resolved.effective_properties["D1"] == 2.0


def test_material_behavior_composition_is_explicit() -> None:
    elastic = MaterialBehavior(ELASTIC_BEHAVIOR_ID, {"E": 100.0, "nu": 0.3})
    plastic = MaterialBehavior(PLASTIC_BEHAVIOR_ID, {"yield_stress": 1.0})
    hyperelastic = MaterialBehavior(
        NEO_HOOKEAN_BEHAVIOR_ID,
        {"C10": 1.0, "D1": 2.0},
    )

    assert not material_behavior_diagnostics((elastic, plastic))
    diagnostics = material_behavior_diagnostics((elastic, hyperelastic))
    assert diagnostics == (
        "Neo-Hookean 超弹性不能与线弹性或塑性同时定义",
    )


def test_neo_hookean_is_bound_through_the_nonlinear_static_solver() -> None:
    model = FEMModel(
        mesh=make_unit_hex8_mesh(),
        name="neo_hookean_smoke",
        node_sets={
            "fixed": NodeSet("fixed", (1, 2, 3, 4)),
            "loaded": NodeSet("loaded", (5, 6, 7, 8)),
        },
        element_sets={"block": ElementSet("block", (1,))},
        materials={
            "rubber": MaterialDefinition(
                "rubber",
                {"C10": 100.0, "D1": 1.0},
                constitutive_model="neo_hookean",
            )
        },
        sections=[SectionAssignment("block", "rubber", "solid", {})],
    )
    step = steps.static("neo-load", NLGEOM=True)
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.nodal_load(step, "loaded", component=1, value=0.01)
    steps.add(model, step)

    assert run_static_preflight(model, "neo-load").passed
    result = solve_analysis(model, "neo-load")

    assert result.outputs["material"] == "neo_hookean"
    assert np.all(np.isfinite(result.U))
