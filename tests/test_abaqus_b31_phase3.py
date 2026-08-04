"""Public-facade coverage for B31 orientation source projection."""

from __future__ import annotations

from pathlib import Path

import pytest

from fem.application import RegionRef, resolve_effective_beam_frames
from fem.io import inp


def _write_deck(tmp_path: Path, name: str, lines: tuple[str, ...]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _beam_lines(
    *,
    nodes: tuple[str, ...] = (
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
    ),
    connectivity: str = "1, 1, 2",
    section_n1: str = "0., 1., 0.",
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    return (
        "*Heading",
        "Phase 3 public facade",
        "*Node",
        *nodes,
        "*Element, type=B31, elset=BEAM",
        connectivity,
        "*Material, name=STEEL",
        "*Elastic",
        "210000., 0.3",
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.2, 0.1",
        section_n1,
        *extra,
    )


def _frames(model):
    report = resolve_effective_beam_frames(
        model,
        RegionRef("element_set", "BEAM"),
    )
    assert report.passed
    return report


def test_orientation_node_is_source_evidence_and_not_a_beam_dof(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "orientation_node.inp",
        _beam_lines(
            nodes=(
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 0., 0., 1.",
            ),
            connectivity="1, 1, 2, 3",
            extra=("*Nset, nset=ALL_NODES", "1, 2, 3"),
        ),
    )

    result = inp.read_with_report(path)

    assert tuple(node.id for node in result.model.mesh.nodes) == (1, 2)
    assert tuple(result.model.mesh.elements[0].node_ids) == (1, 2)
    assert result.model.mesh.num_dofs == 12
    assert result.model.node_sets["ALL_NODES"].node_ids == (1, 2)
    assert _frames(result.model).frames[0].local_y == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    assert result.source_summary is not None
    element = next(
        occurrence
        for occurrence in result.source_summary.occurrences
        if occurrence.name == "element"
    )
    assert element.location.path == path


def test_node_extra_normal_is_consumed_by_public_import_and_d_exponents(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "node_normal.inp",
        _beam_lines(
            nodes=(
                "1, 0D0, 0D0, 0D0, 0D0, 0D0, 1D0",
                "2, 1D0, 0D0, 0D0, 0D0, 0D0, 1D0",
            ),
        ),
    )

    result = inp.read_with_report(path)
    assert _frames(result.model).frames[0].local_y == pytest.approx(
        (0.0, 1.0, 0.0)
    )
    assert result.source_summary is not None
    assert any(
        occurrence.name == "node"
        and occurrence.location.path == path
        for occurrence in result.source_summary.occurrences
    )


def test_element_normal_precedes_node_normal_and_preserves_normal_source(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "explicit_normal.inp",
        _beam_lines(
            nodes=(
                "1, 0., 0., 0., 0., 1., 0.",
                "2, 1., 0., 0., 0., 1., 0.",
            ),
            extra=(
                "*Normal, type=ELEMENT",
                "1, 1, 0., 0., 1.",
                "1, 2, 0., 0., 1.",
            ),
        ),
    )

    result = inp.read_with_report(path)
    assert _frames(result.model).frames[0].local_y == pytest.approx(
        (0.0, 1.0, 0.0)
    )
    assert result.source_summary is not None
    normal = next(
        occurrence
        for occurrence in result.source_summary.occurrences
        if occurrence.name == "normal"
    )
    assert normal.location.keyword == "normal"
    assert normal.location.path == path


def test_element_end_normal_variation_reaches_the_core_frame_field(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "varying_end_normals.inp",
        _beam_lines(
            extra=(
                "*Normal, type=ELEMENT",
                "1, 1, 0., 0., 1.",
                "1, 2, 0., 1., 0.",
            ),
        ),
    )

    result = inp.read_with_report(path)
    field = result.model.mesh.elements[0].props["beam_frame_field"]
    assert not field.is_constant
    assert _frames(result.model).frame_fields[0] == field


def test_malformed_orientation_source_fails_through_public_error_family(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "missing_orientation_node.inp",
        _beam_lines(connectivity="1, 1, 2, 99"),
    )

    with pytest.raises(inp.InpBuildError) as caught:
        inp.read_with_report(path)

    assert caught.value.code == "abaqus.b31.orientation_node_missing"
    assert caught.value.path == path
    assert caught.value.locations


def test_invalid_normal_record_is_a_located_public_parse_error(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "invalid_normal.inp",
        _beam_lines(extra=("*Normal, type=ELEMENT", "1, 1, 0., 0.")),
    )

    with pytest.raises(inp.InpParseError) as caught:
        inp.read_with_report(path)

    assert caught.value.code == "abaqus.b31.normal.record_shape"
    assert caught.value.keyword == "normal"
    assert caught.value.path == path


def test_normal_targeting_non_b31_is_a_public_unsupported_error(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "normal_non_b31.inp",
        (
            "*Heading",
            "Phase 3 non-B31 normal",
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "*Element, type=T3D2",
            "1, 1, 2",
            "*Normal, type=ELEMENT",
            "1, 1, 0., 0., 1.",
        ),
    )

    with pytest.raises(inp.UnsupportedInpFeatureError) as caught:
        inp.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.normal.element_type_unsupported"
    assert error.path == path
    assert error.keyword == "normal"
    assert error.line is not None
    assert error.locations


def test_normal_targeting_unknown_element_has_public_build_evidence(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "normal_unknown_element.inp",
        _beam_lines(
            extra=(
                "*Normal, type=ELEMENT",
                "99, 1, 0., 0., 1.",
            ),
        ),
    )

    with pytest.raises(inp.InpBuildError) as caught:
        inp.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.b31.normal.element_missing"
    assert error.path == path
    assert error.keyword == "normal"
    assert error.line is not None
    assert error.record == ("99", "1", "0.", "0.", "1.")
    assert error.locations


def test_empty_node_normal_components_have_a_public_build_code(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "node_normal_empty.inp",
        _beam_lines(
            nodes=(
                "1, 0., 0., , , 0., 1.",
                "2, 1., 0., 0.",
            ),
        ),
    )

    with pytest.raises(inp.InpBuildError) as caught:
        inp.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.b31.node_normal_empty"
    assert error.path == path
    assert error.keyword == "node"
    assert error.line == 4
    assert error.locations


def test_incomplete_node_normal_components_have_a_distinct_public_code(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "node_normal_incomplete.inp",
        _beam_lines(
            nodes=(
                "1, 0., 0., , 0., 1.",
                "2, 1., 0., 0.",
            ),
        ),
    )

    with pytest.raises(inp.InpBuildError) as caught:
        inp.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.b31.node_normal_shape"
    assert error.path == path
    assert error.keyword == "node"
    assert error.line == 4
    assert error.locations


def test_normal_invalid_local_end_has_public_build_location(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "normal_invalid_local_end.inp",
        _beam_lines(
            extra=(
                "*Normal, type=ELEMENT",
                "1, 99, 0., 0., 1.",
            ),
        ),
    )

    with pytest.raises(inp.InpBuildError) as caught:
        inp.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.b31.normal.local_end_invalid"
    assert error.path == path
    assert error.keyword == "normal"
    assert error.line is not None
    assert error.locations


def test_nonfinite_normal_component_has_a_public_parse_code_and_location(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "normal_nonfinite.inp",
        _beam_lines(
            extra=(
                "*Normal, type=ELEMENT",
                "1, 1, 1e999, 0., 1.",
            ),
        ),
    )

    with pytest.raises(inp.InpParseError) as caught:
        inp.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.real.nonfinite"
    assert error.path == path
    assert error.keyword == "normal"
    assert error.line is not None
    assert error.locations


def test_empty_normal_record_has_a_public_parse_code_and_location(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "normal_empty_record.inp",
        _beam_lines(extra=("*Normal, type=ELEMENT", "")),
    )

    with pytest.raises(inp.InpParseError) as caught:
        inp.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.b31.normal.record_shape"
    assert error.path == path
    assert error.keyword == "normal"
    assert error.line is not None
    assert error.locations
