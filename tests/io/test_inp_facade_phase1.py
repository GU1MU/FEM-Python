from __future__ import annotations

from pathlib import Path

import pytest

from fem.io import inp


def _write_b31(path: Path, *, include_preprint: bool) -> None:
    lines = ["*Heading", "Phase 1 facade test"]
    if include_preprint:
        lines.append(
            "*Preprint, echo=NO, history=NO, model=NO, contact=NO"
        )
    lines.extend(
        (
            "*Node",
            "1, 0.0, 0.0, 0.0",
            "2, 1.0, 0.0, 0.0",
            "*Element, type=B31, elset=BEAM",
            "1, 1, 2",
            "*Material, name=STEEL",
            "*Elastic",
            "2.10E11, 0.30",
            "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
            "0.20, 0.10",
            "0.0, 1.0, 0.0",
            "*Step, name=STATIC",
            "*Static",
            "1.0, 1.0",
            "*Boundary",
            "1, 1, 6, 0.0",
            "*End Step",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_complete_inp_facade_is_model_equivalent_with_harmless_keywords(
    tmp_path: Path,
) -> None:
    with_preprint = tmp_path / "with-preprint.inp"
    without_preprint = tmp_path / "without-preprint.inp"
    _write_b31(with_preprint, include_preprint=True)
    _write_b31(without_preprint, include_preprint=False)

    reported = inp.read_with_report(with_preprint)
    plain = inp.read(with_preprint)
    baseline = inp.read(without_preprint)

    assert isinstance(reported, inp.InpImportResult)
    assert isinstance(reported.notices, tuple)
    assert isinstance(reported.notices[0], inp.InpImportNotice)
    assert reported.source_summary is not None
    assert reported.model.mesh.nodes == baseline.mesh.nodes
    assert reported.model.mesh.elements == baseline.mesh.elements
    assert reported.model.node_sets == baseline.node_sets
    assert reported.model.element_sets == baseline.element_sets
    assert reported.model.materials == baseline.materials
    assert reported.model.sections == baseline.sections
    assert reported.model.steps == baseline.steps
    assert plain.mesh.nodes == reported.model.mesh.nodes
    assert plain.mesh.elements == reported.model.mesh.elements

    occurrences = reported.source_summary.occurrences
    assert tuple(occurrence.name for occurrence in occurrences[:2]) == (
        "heading",
        "preprint",
    )
    assert occurrences[0].category is inp.InpKeywordCategory.HARMLESS_IGNORED
    assert occurrences[1].category is inp.InpKeywordCategory.HARMLESS_IGNORED
    assert occurrences[1].params == (
        ("echo", "NO"),
        ("history", "NO"),
        ("model", "NO"),
        ("contact", "NO"),
    )
    assert all(
        occurrence.location.path == with_preprint
        for occurrence in occurrences
    )


def test_facade_errors_are_public_typed_and_source_located(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.inp"
    _write_b31(path, include_preprint=True)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "*Include, input=unsupported.inp\n",
        encoding="utf-8",
    )

    with pytest.raises(inp.UnsupportedInpFeatureError) as caught:
        inp.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.line.keyword_unsupported"
    assert error.path == path
    assert error.line > 0
    assert error.keyword == "include"
    assert error.locations
    assert error.remediation
    assert str(error)


def test_harmless_keyword_duplicate_options_remain_parse_errors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate-preprint.inp"
    _write_b31(path, include_preprint=False)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "*Heading\nPhase 1 facade test",
        "*Heading\nPhase 1 facade test\n*Preprint, echo=NO, echo=YES",
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(inp.InpParseError) as caught:
        inp.read_with_report(path)

    assert caught.value.code == "abaqus.keyword.parameter_duplicate"
    assert caught.value.path == path
    assert caught.value.keyword == "preprint"
    assert caught.value.remediation


def test_gui_import_production_path_uses_only_the_public_facade() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source = (
        project_root / "src" / "fem_gui" / "main_window.py"
    ).read_text(encoding="utf-8")

    assert "from fem.io.inp import read_with_report" in source
    assert "fem.abaqus" not in source
    assert "parse_file" not in source
    assert "build_model_with_report" not in source
