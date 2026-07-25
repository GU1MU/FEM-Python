from fem.core.model import NodeSet

from fem_agent.diagnostics import DiagnosticCode
from fem_agent.tools.validation import validate_analysis
from tests.helpers.model_builders import make_static_pull_truss_model


def test_validate_analysis_accepts_valid_static_model_without_mutation():
    model = make_static_pull_truss_model()
    original_steps = tuple(model.steps)
    original_node_sets = dict(model.node_sets)

    assert validate_analysis(model, "pull") == ()
    assert tuple(model.steps) == original_steps
    assert model.node_sets == original_node_sets


def test_validate_analysis_normalizes_kernel_validation_failure():
    model = make_static_pull_truss_model()
    model.node_sets["TIP"] = NodeSet("TIP", (999,))

    diagnostics = validate_analysis(model, model.steps[0])

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code == DiagnosticCode.INVALID_MODEL.value
    assert diagnostic.source == "fem.validation"
    assert "missing node 999" in diagnostic.message
    assert diagnostic.remediation


def test_validate_analysis_classifies_unsupported_step_semantics():
    model = make_static_pull_truss_model()
    model.steps[0].procedure = "dynamic"

    diagnostics = validate_analysis(model, "pull")

    assert len(diagnostics) == 1
    assert diagnostics[0].code == DiagnosticCode.UNSUPPORTED_PROCEDURE.value
    assert diagnostics[0].step == "pull"
    assert "requires static" in diagnostics[0].message


def test_validate_analysis_rejects_geometric_nonlinearity():
    model = make_static_pull_truss_model()
    model.steps[0].metadata["NLGEOM"] = "YES"

    diagnostics = validate_analysis(model, "pull")

    assert diagnostics[0].code == DiagnosticCode.UNSUPPORTED_PROCEDURE.value
    assert "geometric nonlinearity" in diagnostics[0].message


def test_validate_analysis_reports_unknown_step_as_invalid_model():
    diagnostics = validate_analysis(make_static_pull_truss_model(), "missing")

    assert diagnostics[0].code == DiagnosticCode.INVALID_MODEL.value
    assert "missing" in diagnostics[0].message
