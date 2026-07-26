from __future__ import annotations

from datetime import datetime, timezone
import importlib
import importlib.util
from pathlib import Path

from fem.application import (
    AnalysisRun,
    ProjectSaveSnapshot,
    ResultProvenance,
    ResultRecord,
    RunStatus,
    SessionSnapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def test_analysis_jobs_module_is_removed_and_canonical_types_import() -> None:
    assert importlib.util.find_spec("fem_gui.analysis_jobs") is None
    assert AnalysisRun.__module__ == "fem.application.runs"
    assert RunStatus.__module__ == "fem.application.runs"


def test_run_result_and_save_aliases_are_absent() -> None:
    assert "COMPLETED" not in RunStatus.__members__
    assert RunStatus("succeeded") is RunStatus.SUCCEEDED
    assert "model_result" not in ResultRecord.__dict__
    assert "project" not in ProjectSaveSnapshot.__dict__
    assert "result" in ResultRecord.__dataclass_fields__
    assert "snapshot" in ProjectSaveSnapshot.__dict__


def test_session_snapshot_legacy_aliases_are_absent() -> None:
    removed = {
        "native_project_path",
        "jobs",
        "material_definitions",
        "section_definitions",
        "region_assignments",
        "analysis_definitions",
        "result",
        "revision",
    }
    canonical = {
        "project_path",
        "runs",
        "materials",
        "sections",
        "assignments",
        "steps",
        "displayed_result",
        "session_revision",
    }

    assert removed.isdisjoint(SessionSnapshot.__dict__)
    assert canonical.issubset(SessionSnapshot.__dataclass_fields__)


def test_result_record_keeps_raw_result_and_complete_provenance() -> None:
    raw = object()
    provenance = ResultProvenance(
        session_id="session",
        artifact_id="artifact",
        model_revision=7,
        step_name="Load",
        run_id="run",
    )
    record = ResultRecord(
        "result",
        provenance,
        raw,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert record.result is raw
    assert record.provenance == provenance
    assert record.provenance.run_id == "run"
    assert record.provenance.model_revision == 7
    assert record.provenance.step_name == "Load"


def test_retired_analysis_job_imports_do_not_remain_in_source() -> None:
    offenders = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if (
            "fem_gui.analysis_jobs" in source
            or "from .analysis_jobs import" in source
        ):
            offenders.append(path.relative_to(PROJECT_ROOT))

    assert offenders == []


def test_versioned_v1_compatibility_modules_remain_importable() -> None:
    assert importlib.import_module("fem.io.project_v1")
    assert importlib.import_module("fem.io.project_migration")
