from __future__ import annotations

from pathlib import Path

import pytest

from fem.abaqus.deck import AbaqusMaterial, AbaqusSection, AbaqusStep
from fem.abaqus.parser import parse_file


def _write_deck(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_material_retains_every_keyword_and_decoded_data_record(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "material_evidence.inp",
        [
            "*Material, name=STEEL",
            "*Elastic",
            "210000D0, 0.3, 12D0, -2.5e-1",
            "205000., 0.29, 25.",
            "*Elastic",
            "200000., 0.28, 50.",
            "*Density",
            "7.85D-9, 293.",
            "*Density",
            "8.0e-9, 303., 2.",
        ],
    )

    material = parse_file(path).materials["STEEL"]

    assert material.elastic_keyword_count == 2
    assert material.density_keyword_count == 2
    assert len(material.elastic_records) == 3
    assert len(material.density_records) == 2
    assert material.elastic_records[0].fields == (
        "210000D0",
        "0.3",
        "12D0",
        "-2.5e-1",
    )
    assert material.elastic_records[0].values == pytest.approx(
        (210000.0, 0.3, 12.0, -0.25)
    )
    assert material.elastic_records[0].raw == (
        "210000D0, 0.3, 12D0, -2.5e-1"
    )
    assert material.elastic_records[0].location is not None
    assert material.elastic_records[0].location.line == 3
    assert material.density_records[1].fields == (
        "8.0e-9",
        "303.",
        "2.",
    )
    assert material.density_records[1].values == pytest.approx(
        (8.0e-9, 303.0, 2.0)
    )
    assert material.properties == {
        "E": 200000.0,
        "nu": 0.28,
        "rho": 8.0e-9,
    }


def test_step_retains_keyword_and_procedure_evidence(tmp_path: Path) -> None:
    path = _write_deck(
        tmp_path,
        "step_evidence.inp",
        [
            "*Boundary",
            "1, 1",
            "*Step, name=LOAD",
            "*Static",
            "1., 1.",
            "*Static",
            "0.5, 1.",
            "*End Step",
        ],
    )

    initial, load = parse_file(path).steps

    assert initial.name == "Initial"
    assert initial.keyword_location is None
    assert initial.procedure_present is False
    assert initial.procedure_location is None
    assert initial.procedure_count == 0
    assert load.keyword_location is not None
    assert load.keyword_location.line == 3
    assert load.keyword_location.keyword == "step"
    assert load.procedure_present is True
    assert load.procedure_location is not None
    assert load.procedure_location.line == 6
    assert load.procedure_location.keyword == "static"
    assert load.procedure_count == 2


def test_section_distinguishes_missing_and_defined_empty_target(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "section_target_evidence.inp",
        [
            "*Elset, elset=EMPTY",
            "*Solid Section, elset=EMPTY, material=STEEL",
            "*Heading",
            "*Solid Section, elset=MISSING, material=STEEL",
        ],
    )

    empty_target, missing_target = parse_file(path).sections

    assert empty_target.element_set == "EMPTY"
    assert empty_target.element_ids == ()
    assert empty_target.target_was_defined is True
    assert missing_target.element_set == "MISSING"
    assert missing_target.element_ids == ()
    assert missing_target.target_was_defined is False


def test_node_raw_fields_keep_explicit_blank_normal_columns(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "node_evidence.inp",
        [
            "*Node",
            "1, 0., 1., 2., , 3.0, ",
            "2, 4., 5.",
            "3, 6., 7., , ",
        ],
    )

    deck = parse_file(path)
    first = deck.node_records[1]
    second = deck.node_records[2]
    third = deck.node_records[3]

    assert first.coordinates == (0.0, 1.0, 2.0)
    assert first.raw_fields == (
        "1",
        "0.",
        "1.",
        "2.",
        "",
        "3.0",
        "",
    )
    assert first.extra_fields == ("", "3.0", "")
    assert first.location is not None
    assert first.location.line == 2
    assert second.coordinates == (4.0, 5.0, 0.0)
    assert second.raw_fields == ("2", "4.", "5.")
    assert second.extra_fields == ()
    assert third.coordinates == (6.0, 7.0, 0.0)
    assert third.raw_fields == ("3", "6.", "7.", "", "")
    assert third.extra_fields == ("",)


def test_evidence_fields_keep_legacy_dto_construction_defaults() -> None:
    material = AbaqusMaterial("STEEL")
    section = AbaqusSection("SET", "STEEL")
    step = AbaqusStep("Initial")

    assert material.elastic_records == []
    assert material.density_records == []
    assert material.elastic_keyword_count == 0
    assert material.density_keyword_count == 0
    assert section.target_was_defined is False
    assert step.keyword_location is None
    assert step.procedure_present is False
    assert step.procedure_location is None
    assert step.procedure_count == 0


@pytest.mark.parametrize("keyword", ("*Elastic", "*Density"))
def test_material_property_keyword_requires_material_even_without_data(
    tmp_path: Path,
    keyword: str,
) -> None:
    path = _write_deck(
        tmp_path,
        "orphan_material_property.inp",
        [keyword, "*Heading"],
    )

    with pytest.raises(ValueError, match="must follow"):
        parse_file(path)
