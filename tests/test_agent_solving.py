from fem import abaqus

from fem_agent.schemas import DiagnosticSeverity
from fem_agent.tools.solving import solve_analysis
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp


def test_solve_analysis_matches_direct_solver_success_shape(tmp_path):
    path = write_perforated_plate_style_inp(
        tmp_path,
        "solve_service.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    model = abaqus.read(path)
    step = model.steps[-1]

    outcome = solve_analysis(model, step)

    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.step is step
    assert outcome.elapsed_seconds >= 0


def test_solve_analysis_normalizes_a_singular_model(tmp_path):
    path = write_perforated_plate_style_inp(
        tmp_path,
        "singular_service.inp",
        ("*Cload", "Set-right, 1, 10."),
    )
    model = abaqus.read(path)
    model.steps[0].boundaries = ()
    step = model.steps[-1]

    outcome = solve_analysis(model, step)

    assert outcome.result is None
    assert outcome.diagnostics[0].code == "SOLVER_FAILED"
    assert outcome.diagnostics[0].severity == DiagnosticSeverity.ERROR
