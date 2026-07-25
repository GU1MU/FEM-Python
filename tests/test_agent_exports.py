from dataclasses import replace
import hashlib
from pathlib import Path

from fem_agent.diagnostics import DiagnosticCode
from fem_agent.schemas import ExportFormat, ResourceLimits
from fem_agent.tools.exports import export_results
from tests.helpers.mesh_builders import make_tri3_stiffness_mesh
from tests.helpers.result_builders import make_zero_result


def _run_paths(tmp_path: Path):
    run_directory = tmp_path / "sessions" / "session-1" / "runs" / "run-1"
    exports_directory = run_directory / "exports"
    exports_directory.mkdir(parents=True)
    return run_directory.resolve(), exports_directory.resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_csv_export_registers_every_actual_file_and_digest(tmp_path):
    run_directory, exports_directory = _run_paths(tmp_path)
    result = make_zero_result(make_tri3_stiffness_mesh(), "plate")

    outcome = export_results(
        result,
        (ExportFormat.CSV,),
        run_id="run-1",
        run_directory=run_directory,
        exports_directory=exports_directory,
    )

    assert outcome.ok is True
    assert outcome.diagnostics == ()
    assert len(outcome.artifacts) == 3
    actual_names = {
        path.name for path in exports_directory.iterdir() if path.is_file()
    }
    artifact_names = {
        artifact.display_path.removeprefix("exports/")
        for artifact in outcome.artifacts
    }
    assert artifact_names == actual_names
    assert any(name.endswith("_nodal_displacement.csv") for name in actual_names)
    assert any(name.endswith("_element_stress.csv") for name in actual_names)
    assert any(name.endswith("_nodal_stress.csv") for name in actual_names)
    for artifact in outcome.artifacts:
        assert artifact.display_path == f"exports/{Path(artifact.display_path).name}"
        path = run_directory / artifact.display_path
        assert artifact.sha256 == _sha256(path)
        assert artifact.size_bytes == path.stat().st_size
        assert artifact.kind == "csv"
    assert not any(
        path.name.startswith(".fem-agent-export-")
        for path in exports_directory.iterdir()
    )


def test_vtk_export_registers_vtk_and_materialized_csv_dependencies(tmp_path):
    run_directory, exports_directory = _run_paths(tmp_path)

    outcome = export_results(
        make_zero_result(make_tri3_stiffness_mesh(), "plate"),
        (ExportFormat.VTK,),
        run_id="run-1",
        run_directory=run_directory,
        exports_directory=exports_directory,
    )

    assert outcome.ok is True
    suffixes = {Path(item.display_path).suffix for item in outcome.artifacts}
    assert suffixes == {".csv", ".vtk"}
    assert len(outcome.artifacts) == len(tuple(exports_directory.iterdir()))


def test_export_rejects_directory_outside_active_run(tmp_path):
    run_directory, _exports_directory = _run_paths(tmp_path)
    outside = tmp_path / "outside" / "exports"
    outside.mkdir(parents=True)

    outcome = export_results(
        make_zero_result(make_tri3_stiffness_mesh(), "plate"),
        (ExportFormat.CSV,),
        run_id="run-1",
        run_directory=run_directory,
        exports_directory=outside.resolve(),
    )

    assert outcome.ok is False
    assert outcome.artifacts == ()
    assert outcome.diagnostics[0].code == DiagnosticCode.EXPORT_FAILED.value
    assert tuple(outside.iterdir()) == ()


def test_export_never_overwrites_existing_run_artifact(tmp_path):
    run_directory, exports_directory = _run_paths(tmp_path)
    existing = exports_directory / "result-run-1_nodal_displacement.csv"
    existing.write_text("sentinel", encoding="utf-8")

    outcome = export_results(
        make_zero_result(make_tri3_stiffness_mesh(), "plate"),
        (ExportFormat.CSV,),
        run_id="run-1",
        run_directory=run_directory,
        exports_directory=exports_directory,
    )

    assert outcome.ok is False
    assert outcome.artifacts == ()
    assert existing.read_text(encoding="utf-8") == "sentinel"
    assert {path.name for path in exports_directory.iterdir()} == {existing.name}


def test_export_uses_safe_run_prefix_instead_of_result_names(tmp_path):
    run_directory, exports_directory = _run_paths(tmp_path)
    result = make_zero_result(make_tri3_stiffness_mesh(), "../../escape")
    result.name = "..\\..\\escape"

    outcome = export_results(
        result,
        (ExportFormat.VTK,),
        run_id="run-1",
        run_directory=run_directory,
        exports_directory=exports_directory,
    )

    assert outcome.ok is True
    assert all(".." not in artifact.display_path for artifact in outcome.artifacts)
    assert all(
        (run_directory / artifact.display_path).resolve().parent
        == exports_directory
        for artifact in outcome.artifacts
    )
    assert not (tmp_path / "escape.vtk").exists()


def test_export_enforces_file_limit_before_committing(tmp_path):
    run_directory, exports_directory = _run_paths(tmp_path)
    limits = replace(ResourceLimits(), max_output_files=1)

    outcome = export_results(
        make_zero_result(make_tri3_stiffness_mesh(), "plate"),
        (ExportFormat.CSV,),
        run_id="run-1",
        run_directory=run_directory,
        exports_directory=exports_directory,
        resource_limits=limits,
    )

    assert outcome.ok is False
    assert outcome.artifacts == ()
    assert tuple(exports_directory.iterdir()) == ()
    assert "limit is 1" in outcome.diagnostics[0].message
