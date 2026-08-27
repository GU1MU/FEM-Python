from __future__ import annotations

import numpy as np
import pytest

from fem.analysis import (
    ExecutionStrategy,
    linear_static,
    resolve_analysis_request,
    resolve_execution_plan,
)
from fem.application import (
    analysis_solver_kind,
    safe_static_preflight,
    solve_analysis,
)
from fem.materials import SmallStrainJ2PlasticityMaterial
from fem.model import (
    ElementSet,
    FEMModel,
    GeometryMode,
    MaterialDefinition,
    NodeSet,
    SectionAssignment,
    StaticFormulation,
)
from fem.model import authoring as steps
from fem.results import (
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
)
from tests.architecture.test_nlf30_gui_workflow import (
    _make_nonlinear_workflow_model,
)
from tests.helpers.mesh_builders import make_hex8_stiffness_mesh


def _replace_with_small_strain_step(model, *, load: float) -> None:
    model.steps.clear()
    step = steps.static("small", formulation=StaticFormulation.LINEAR)
    steps.displacement(step, "fixed", components=(1, 2))
    steps.nodal_load(step, "loaded", component=1, value=load)
    steps.add(model, step)


def _make_small_strain_hex8_j2_model() -> FEMModel:
    model = FEMModel(
        mesh=make_hex8_stiffness_mesh(),
        name="small-strain-hex8-j2",
        node_sets={
            "fixed": NodeSet("fixed", (1, 4, 5, 8)),
            "loaded": NodeSet("loaded", (2, 3, 6, 7)),
        },
        element_sets={"solid": ElementSet("solid", (1,))},
        materials={
            "steel": MaterialDefinition(
                "steel",
                {
                    "E": 210.0,
                    "nu": 0.3,
                    "yield_stress": 0.01,
                    "hardening_modulus": 0.1,
                },
                constitutive_model="j2_plasticity",
            )
        },
        sections=[SectionAssignment("solid", "steel", "solid")],
    )
    step = steps.static("small-3d", formulation=StaticFormulation.LINEAR)
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.nodal_load(step, "loaded", component=1, value=0.1)
    steps.add(model, step)
    return model


def test_small_strain_j2_is_not_downgraded_to_linear_elasticity() -> None:
    model = _make_nonlinear_workflow_model(
        yield_stress=0.01,
        hardening_modulus=0.1,
        load=0.1,
    )
    _replace_with_small_strain_step(model, load=0.1)
    request = resolve_analysis_request(model.steps[0])
    plan = resolve_execution_plan(model, request)

    assert request.geometry_mode is GeometryMode.SMALL_STRAIN
    assert not model.steps[0].metadata.get("NLGEOM", False)
    assert plan.strategy is ExecutionStrategy.INCREMENTAL_NEWTON
    assert analysis_solver_kind(model, "small") == "nonlinear_static"

    result = solve_analysis(model, "small")

    points = result.outputs["integration_points"]
    assert result.outputs["geometry_mode"] == "small_strain"
    assert result.outputs["material_algorithm"] == "radial_return"
    assert np.all(points["equivalent_plastic_strain"] > 0.0)
    assert isinstance(
        result.outputs["material_algorithm"],
        str,
    )


def test_small_strain_elasticity_keeps_the_direct_linear_optimization() -> None:
    model = _make_nonlinear_workflow_model(yield_stress=None, load=1.0e-4)
    _replace_with_small_strain_step(model, load=1.0e-4)
    request = resolve_analysis_request(model.steps[0])
    plan = resolve_execution_plan(model, request)

    assert plan.strategy is ExecutionStrategy.DIRECT_LINEAR
    assert analysis_solver_kind(model, "small") == "linear_static"
    result = solve_analysis(model, "small")
    assert "integration_points" not in result.outputs


def test_small_strain_j2_publishes_material_state_without_finite_strain_fields() -> None:
    model = _make_nonlinear_workflow_model(
        yield_stress=0.01,
        hardening_modulus=0.1,
        load=0.1,
    )
    _replace_with_small_strain_step(model, load=0.1)
    result = solve_analysis(model, "small")
    source = ResultSourceKey(
        result_id="small-strain-j2",
        session_id="test-session",
        artifact_id="test-artifact",
        model_revision=1,
        step_name="small",
        run_id="run-1",
    )
    provider = build_result_provider(source, result)
    assert not any(
        availability.key.request.field_id.variable is ResultVariable.E
        for availability in provider.catalog().fields
    )
    key = provider.resolve_request(
        FieldRequest(
            ResultFieldId(
                ResultVariable.PEEQ,
                FieldPosition.INTEGRATION_POINT,
            )
        )
    )
    updated = provider.advance(provider.materialize((key,)))
    assert updated.field(key).values.shape == (4, 1)
    assert np.all(updated.field(key).values > 0.0)


def test_small_strain_j2_material_uses_explicit_radial_return_algorithm() -> None:
    material = SmallStrainJ2PlasticityMaterial(210.0, 0.3, 0.01, 0.1)
    assert material.algorithm == "radial_return"
    assert set(material.initial_state()) == {
        "plastic_strain",
        "equivalent_plastic_strain",
    }


def test_preflight_publishes_the_material_aware_execution_plan() -> None:
    model = _make_nonlinear_workflow_model(
        yield_stress=0.01,
        hardening_modulus=0.1,
        load=0.1,
    )
    _replace_with_small_strain_step(model, load=0.1)

    report = safe_static_preflight(model, "small")

    assert report.passed
    assert report.facts.geometry_mode == "small_strain"
    assert report.facts.execution_strategy == "incremental_newton"
    assert report.facts.material_models == ("j2_plasticity",)


def test_direct_linear_kernel_rejects_material_history_instead_of_dropping_it() -> None:
    model = _make_nonlinear_workflow_model(
        yield_stress=0.01,
        hardening_modulus=0.1,
        load=0.1,
    )
    _replace_with_small_strain_step(model, load=0.1)

    with pytest.raises(ValueError, match="unified static executor"):
        linear_static.solve(model, "small")


def test_small_strain_hex8_j2_materializes_captured_3d_stress() -> None:
    model = _make_small_strain_hex8_j2_model()

    result = solve_analysis(model, "small-3d")
    integration = result.outputs["integration_points"]

    assert result.outputs["geometry_mode"] == "small_strain"
    assert integration["cauchy_stress"].shape == (8, 3, 3)
    assert np.all(np.isfinite(integration["cauchy_stress"]))

    source = ResultSourceKey(
        result_id="small-3d",
        session_id="test-session",
        artifact_id="test-artifact",
        model_revision=1,
        step_name="small-3d",
        run_id="run-1",
    )
    provider = build_result_provider(source, result)
    key = provider.resolve_request(
        FieldRequest(
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.INTEGRATION_POINT,
            )
        )
    )
    updated = provider.advance(provider.materialize((key,)))
    assert updated.field(key).values.shape == (8, 10)
    assert np.all(np.isfinite(updated.field(key).values))
