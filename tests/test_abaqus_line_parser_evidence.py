"""Public source-evidence checks for the INP facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from fem.io import inp


def _write(tmp_path: Path, name: str, lines: tuple[str, ...]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _base(*extra: str) -> tuple[str, ...]:
    return (
        "*Heading",
        "Public evidence test",
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "*Element, type=B31, elset=BEAM",
        "1, 1, 2",
        "*Material, name=STEEL",
        "*Elastic",
        "210000D0, 0.3",
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.2, 0.1",
        "0., 1., 0.",
        *extra,
    )


def test_source_summary_preserves_keyword_identity_and_locations(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "evidence.inp",
        _base(
            "*Step, name=LOAD",
            "*Static",
            "1., 1.",
            "*End Step",
        ),
    )

    result = inp.read_with_report(path)

    assert result.source_summary is not None
    occurrences = result.source_summary.occurrences
    assert tuple(item.name for item in occurrences[:2]) == ("heading", "node")
    assert all(item.location.path == path for item in occurrences)
    assert next(item for item in occurrences if item.name == "step").params == (
        ("name", "LOAD"),
    )
    assert next(item for item in occurrences if item.name == "step").location.line > 0


def test_public_model_retains_decoded_material_values(tmp_path: Path) -> None:
    path = _write(tmp_path, "material.inp", _base())

    result = inp.read_with_report(path)

    assert result.model.materials["STEEL"].properties == {
        "E": 210000.0,
        "nu": 0.3,
    }


def test_public_model_retains_output_source_evidence(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "output.inp",
        _base(
            "*Step, name=LOAD",
            "*Static",
            "*Output, FIELD, VARIABLE=PRESELECT",
            "*Node Output, NSET=ALL",
            "U, RF",
            "*End Step",
        ),
    )

    result = inp.read_with_report(path)
    request = result.model.steps[-1].outputs[1]

    assert request.variables == ("U", "RF")
    assert request.source_evidence is not None
    assert request.source_evidence.parent_parameters == (
        ("variable", "PRESELECT"),
    )
    assert request.source_evidence.child_parameters == (("nset", "ALL"),)


def test_orphan_material_property_is_a_public_parse_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "orphan.inp", ("*Elastic", "210000., 0.3"))

    with pytest.raises(inp.InpParseError, match="must follow"):
        inp.read_with_report(path)
