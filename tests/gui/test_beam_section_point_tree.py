import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem.application.results import build_archived_result_provider
from fem.io import load_result_archive, save_result_archive
from fem_gui.result_presentation import result_provider_section_point_labels
from fem_gui.widgets.result_tree import ResultTree
from tests.helpers.beam_section_builders import (
    _SECTION_CASES, _solve_and_request_stress, _archive_snapshot,
)


@pytest.mark.parametrize(
    ("section_type", "dimensions"), _SECTION_CASES,
    ids=["rectangle", "solid-circle", "hollow-circle"],
)
def test_archived_section_points_and_resultants_have_chinese_tree_labels(section_type, dimensions, tmp_path: Path):
    application = QApplication.instance() or QApplication([])
    _, provider, outcome = _solve_and_request_stress(section_type, dimensions)
    archive_path = tmp_path / "beam.femres"
    save_result_archive(archive_path, _archive_snapshot(provider, outcome, section_type))
    archived = build_archived_result_provider(load_result_archive(archive_path).snapshot)
    labels = result_provider_section_point_labels(archived)
    assert labels == ({1: "右上", 2: "左上", 3: "左下", 4: "右下"} if section_type == "rectangle" else {})
    tree = ResultTree()
    try:
        tree.set_catalog("Load", archived.catalog(), section_point_labels=labels)
        step = tree.topLevelItem(0).child(0)
        variables = {step.child(index).text(0): step.child(index) for index in range(step.childCount())}
        stress = variables["应力 S"]
        expected = ("右上", "左上", "左下", "右下") if section_type == "rectangle" else (
            "截面点 1", "截面点 2", "截面点 3", "截面点 4",
        )
        assert tuple(stress.child(index).text(0) for index in range(stress.childCount())) == expected
        for label, components in [
            ("截面力 SF（积分点）", ("N", "Vy", "Vz")),
            ("截面矩 SM（积分点）", ("T", "My", "Mz")),
        ]:
            item = variables[label]
            assert tuple(item.child(index).text(0) for index in range(item.childCount())) == components
    finally:
        tree.close()
        tree.deleteLater()
        application.processEvents()
