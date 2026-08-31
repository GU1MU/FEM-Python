from __future__ import annotations

import pytest

from fem.analysis import (
    AnalysisPath,
    AnalysisExecutor,
    DEFAULT_ANALYSIS_EXECUTOR,
    compile_analysis,
    resolve_analysis_request,
)
from fem.materials import GreenLagrangeElasticMaterial
from tests.architecture.test_arch_v2_core import _model


def test_registered_executor_is_the_single_procedure_boundary() -> None:
    assert isinstance(DEFAULT_ANALYSIS_EXECUTOR, AnalysisExecutor)
    request = resolve_analysis_request(_model().steps[0])
    assert DEFAULT_ANALYSIS_EXECUTOR.procedure_for(request).path is AnalysisPath.NONLINEAR_STATIC


def test_compiled_analysis_contains_only_detached_runtime_objects() -> None:
    model = _model()
    request = resolve_analysis_request(model.steps[0])
    compiled = compile_analysis(model, request)

    assert compiled.request.step is not model.steps[0]
    assert not hasattr(compiled.request.step, "metadata")
    assert compiled.bindings[0].entity is not model.mesh.elements[0]
    assert compiled.bindings[0].entity.props["E"] == 210.0
    assert compiled.reference_load.flags.writeable is False

    model.mesh.elements[0].props["E"] = 999.0
    assert compiled.bindings[0].entity.props["E"] == 210.0
    with pytest.raises(TypeError):
        compiled.constraints.prescribed_values[0] = 1.0


def test_constitutive_model_is_not_inferred_from_yield_properties() -> None:
    model = _model()
    model.mesh.elements[0].props.update(
        yield_stress=0.01,
        hardening_modulus=0.1,
    )
    compiled = compile_analysis(
        model,
        resolve_analysis_request(model.steps[0]),
    )

    material = compiled.bindings[0].resources["material"]
    assert isinstance(material, GreenLagrangeElasticMaterial)
