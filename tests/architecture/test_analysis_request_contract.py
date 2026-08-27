import pytest

from fem.analysis import (
    AnalysisRequest,
    NonlinearStaticAnalysis,
    resolve_analysis_request,
)
from fem.model import AnalysisStep
from fem.model import StaticFormulation, StaticStepControls


def test_analysis_request_captures_typed_formulation_once() -> None:
    step = AnalysisStep(
        "nonlinear",
        procedure="static",
        formulation=StaticFormulation.NONLINEAR,
    )

    request = resolve_analysis_request(step)

    assert request.step is not step
    assert request.step.formulation is StaticFormulation.NONLINEAR
    assert request.formulation is StaticFormulation.NONLINEAR
    assert request.controls == StaticStepControls()


def test_analysis_request_rejects_untyped_formulation() -> None:
    step = resolve_analysis_request(AnalysisStep("step")).step
    with pytest.raises(TypeError, match="formulation"):
        AnalysisRequest(step=step, formulation="nonlinear_static")


def test_analysis_request_requires_a_detached_step_snapshot() -> None:
    controls = StaticStepControls(initial_increment=0.5)
    with pytest.raises(TypeError, match="AnalysisStepSnapshot"):
        AnalysisRequest(None, controls, (0.0, 1.0))


def test_procedure_request_preserves_step_options_and_uses_explicit_controls() -> None:
    step = AnalysisStep(
        "nonlinear",
        procedure="static",
        formulation=StaticFormulation.NONLINEAR,
    )
    controls = StaticStepControls(initial_increment=0.25)

    request = NonlinearStaticAnalysis(controls=controls).request(None, step)

    assert request.step is not step
    assert request.formulation is StaticFormulation.NONLINEAR
    assert request.controls == controls


def test_typed_controls_do_not_parse_invalid_metadata() -> None:
    controls = StaticStepControls(initial_increment=0.5)
    step = AnalysisStep(
        "nonlinear",
        procedure="static",
        formulation=StaticFormulation.NONLINEAR,
        controls=controls,
    )
    step.metadata["initial_increment"] = 0.0

    request = resolve_analysis_request(step)

    assert request.controls == controls
