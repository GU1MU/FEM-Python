"""Machine checks for the Phase 0 contract and usage ledger artifacts."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = PROJECT_ROOT / "docs" / "2026-08-04-abaqus-b31-inp-phase-0-freeze.md"
LEDGER_PATH = PROJECT_ROOT / "docs" / "2026-08-04-abaqus-b31-inp-usage-ledger.json"


def test_phase0_freeze_records_target_api_errors_orientation_terms_and_isolation() -> None:
    text = FREEZE_PATH.read_text(encoding="utf-8")

    for token in (
        "fem.io.inp.read",
        "fem.io.inp.read_with_report",
        "InpImportResult",
        "InpInputError",
        "InpParseError",
        "InpBuildError",
        "UnsupportedInpFeatureError",
        "default n1",
        "section n1",
        "orientation node",
        "nodal normal",
        "generated normal",
        "effective element frame",
        "element-end frame",
        "data",
        "tmp_path",
    ):
        assert token in text


def test_phase0_usage_ledger_is_complete_for_current_callers_and_data_free() -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))

    assert ledger["schema_version"] == 1
    assert ledger["phase"] == 0
    assert set(ledger["coverage"]) == {
        "gui",
        "agent",
        "application",
        "tests",
        "examples",
        "io",
    }
    assert "data" in ledger["excluded_paths"]
    assert ledger["entries"]

    caller_paths: set[str] = set()
    for entry in ledger["entries"]:
        assert entry["symbol"]
        assert entry["definition"]
        assert entry["planned_disposition"]
        assert entry["callers"]
        for caller in entry["callers"]:
            path_text = caller["path"]
            assert "data" not in path_text.casefold()
            path = PROJECT_ROOT / path_text
            assert path.is_file(), path_text
            assert caller["area"] in ledger["coverage"]
            caller_paths.add(path_text)

    assert "src/fem_gui/main_window.py" in caller_paths
    assert "src/fem_agent/tools/importing.py" in caller_paths
    assert "tests/application/test_preflight.py" in caller_paths
    assert "examples/cantilever_beam.py" in caller_paths
    assert "tests/test_io.py" in caller_paths
    assert all("data" not in path.casefold() for path in caller_paths)
