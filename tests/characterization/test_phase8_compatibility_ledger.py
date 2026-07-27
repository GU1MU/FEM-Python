from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "phase8"
    / "result_compatibility_ledger.json"
)
ENTRY_KEYS = {
    "id",
    "symbol",
    "visibility",
    "current_callers",
    "target_batch",
    "disposition",
    "known_divergences",
    "golden_tests",
}
TARGET_BATCHES = {
    *(f"Batch {number}" for number in range(1, 7)),
    "Deferred: Agent Integration",
}
VISIBILITIES = {
    "public",
    "legacy_public",
    "gui_legacy",
    "deferred_public",
}


def _assert_test_node_exists(reference: str) -> None:
    path_text, separator, node_name = reference.partition("::")
    assert separator == "::"
    assert node_name.startswith("test_")
    path = PROJECT_ROOT / path_text
    assert path.is_file(), reference
    assert f"def {node_name}(" in path.read_text(encoding="utf-8"), reference


def test_phase8_result_compatibility_ledger_is_machine_readable_and_complete() -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))

    assert ledger["schema_version"] == 1
    assert ledger["phase"] == 8
    entries = ledger["entries"]
    assert entries
    assert len({entry["id"] for entry in entries}) == len(entries)
    assert len({entry["symbol"] for entry in entries}) == len(entries)

    for entry in entries:
        assert set(entry) == ENTRY_KEYS
        assert entry["visibility"] in VISIBILITIES
        assert entry["target_batch"] in TARGET_BATCHES
        assert entry["current_callers"]
        assert entry["disposition"]
        assert isinstance(entry["known_divergences"], list)
        assert entry["golden_tests"]
        for caller in entry["current_callers"]:
            assert (PROJECT_ROOT / caller).is_file(), (entry["id"], caller)
        for reference in entry["golden_tests"]:
            _assert_test_node_exists(reference)

    symbols = {entry["symbol"] for entry in entries}
    assert {
        "fem.post.stress.StressRecovery",
        "fem.post.stress.field.resolve",
        "fem.post.vtk.export.from_csv",
        "fem.post.vtk.export.from_result",
        "fem_agent.tools.results.query_results",
        "fem_agent.tools.exports.export_results",
    }.issubset(symbols)
    assert symbols.isdisjoint(
        {
            "fem_gui.visualization.result_adapter.ResultData",
            "fem_gui.visualization.result_adapter.build_result_data",
            "fem_gui.visualization.result_adapter._line_stress",
            "fem_gui.visualization.stress_adapter.build_stress_render_geometry",
            "fem_gui.visualization.csv_export.export_field_csv",
        }
    )
