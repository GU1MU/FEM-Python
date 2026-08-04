from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fem import abaqus
from fem.application import RegionRef, resolve_effective_beam_frames
from fem.abaqus.deck import AbaqusElementEndIdentity
from fem.abaqus.parser import parse_file


def _write_deck(tmp_path: Path, name: str, lines: list[str]) -> Path:
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
) -> list[str]:
    return [
        "*Heading",
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
    ]


def test_orientation_node_is_typed_and_excluded_from_beam_mesh(tmp_path: Path) -> None:
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
            section_n1="0., 1., 0.",
            extra=("*Nset, nset=ALL_NODES", "1, 2, 3"),
        ),
    )

    deck = parse_file(path)
    element = deck.elements[0]
    before_build = deepcopy(deck)

    assert element.node_ids == (1, 2)
    assert element.structural_node_ids == (1, 2)
    assert element.additional_orientation_node_id == 3
    assert element.raw_fields == ("1", "1", "2", "3")

    result = abaqus.build_model_with_report(deck)
    mesh = result.model.mesh
    assert tuple(node.id for node in mesh.nodes) == (1, 2)
    assert tuple(mesh.elements[0].node_ids) == (1, 2)
    assert mesh.num_nodes == 2
    assert mesh.num_dofs == 12
    assert result.model.node_sets["ALL_NODES"].node_ids == (1, 2)
    assert deck == before_build

    frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "BEAM"),
    )
    assert frames.passed
    assert frames.frames[0].local_y == pytest.approx((0.0, 0.0, 1.0))


def test_node_extra_normal_is_typed_with_source_span_and_abaqus_real_syntax(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "node_normal.inp",
        _beam_lines(
            nodes=(
                "1, 0D0, 0D0, , 0D0, 0D0, 1D0",
                "2, 1D0, 0D0, , 0D0, 0D0, 1D0",
            ),
        ),
    )

    deck = parse_file(path)
    first = deck.node_records[1]
    second = deck.node_records[2]

    assert first.coordinates == (0.0, 0.0, 0.0)
    assert first.extra_fields == ("0D0", "0D0", "1D0")
    assert first.normal is not None
    assert first.normal.node_id == 1
    assert first.normal.vector == pytest.approx((0.0, 0.0, 1.0))
    assert first.normal.raw == "1, 0D0, 0D0, , 0D0, 0D0, 1D0"
    assert first.normal.span is not None
    assert first.normal.span.start.keyword == "node"
    assert first.normal.span.start.line == 3
    assert second.normal is not None

    result = abaqus.build_model_with_report(deck)
    frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "BEAM"),
    )
    assert frames.passed
    assert frames.frames[0].local_y == pytest.approx((0.0, 1.0, 0.0))


def test_element_normal_takes_precedence_over_node_normal_and_keeps_identity(
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

    deck = parse_file(path)
    assert len(deck.normal_records) == 2
    assert deck.normal_records[0].element_target == 1
    assert deck.normal_records[0].node_target == 1
    assert deck.normal_records[0].identities == (
        deck.normal_records[0].identity,
    )
    assert deck.normal_records[0].identity is not None
    assert deck.normal_records[0].identity.local_end == 1
    assert deck.normal_records[0].source_span.start.keyword == "normal"
    assert deck.normal_records[0].source_span.start.line == 14

    result = abaqus.build_model_with_report(deck)
    frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "BEAM"),
    )
    assert frames.passed
    # The explicit n2=(0,0,1) wins over the node n2=(0,1,0), yielding n1=y.
    assert frames.frames[0].local_y == pytest.approx((0.0, 1.0, 0.0))


def test_node_extra_and_element_normal_share_comparable_typed_target_identity(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "equivalent_normal_sources.inp",
        _beam_lines(
            nodes=(
                "1, 0., 0., 0., 0., 0., 1.",
                "2, 1., 0., 0., 0., 0., 1.",
            ),
            extra=(
                "*Normal, type=ELEMENT",
                "1, 1, 0., 0., 1.",
                "1, 2, 0., 0., 1.",
            ),
        ),
    )

    deck = parse_file(path)
    node_source = deck.node_records[1].normal
    element_source = deck.normal_records[0]

    assert node_source is not None
    assert element_source.identity == AbaqusElementEndIdentity(1, 1, 1)
    assert node_source.node_id == element_source.node_id
    assert node_source.node_id == element_source.identity.node_id
    assert node_source.vector == element_source.vector

    result = abaqus.build_model_with_report(deck)
    frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "BEAM"),
    )
    assert frames.passed


def test_different_element_end_frames_return_typed_phase5_capability_error(
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

    deck = parse_file(path)
    with pytest.raises(abaqus.UnsupportedAbaqusFeatureError) as caught:
        abaqus.build_model_with_report(deck)

    error = caught.value
    assert error.code == "abaqus.b31.element_end_frame_variation_unsupported"
    assert error.record["capability"] == "constant_element_frame_only"
    assert tuple(
        item["identity"].local_end for item in error.record["ends"]
    ) == (1, 2)
    assert len(error.locations) >= 3
    assert all(location.path == path for location in error.locations)


def test_missing_orientation_node_is_rejected_with_source_locations(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "missing_orientation_node.inp",
        _beam_lines(connectivity="1, 1, 2, 99"),
    )

    with pytest.raises(abaqus.AbaqusBuildError) as caught:
        abaqus.build_model_with_report(parse_file(path))

    assert caught.value.code == "abaqus.b31.orientation_node_missing"
    assert caught.value.record == {"element": 1, "orientation_node": 99}
    assert caught.value.location is not None
    assert caught.value.location.path == path


def test_normal_targeting_non_b31_element_is_rejected_typed(tmp_path: Path) -> None:
    path = _write_deck(
        tmp_path,
        "normal_non_b31.inp",
        [
            "*Heading",
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "*Element, type=T3D2",
            "1, 1, 2",
            "*Normal, type=ELEMENT",
            "1, 1, 0., 0., 1.",
        ],
    )

    with pytest.raises(abaqus.UnsupportedAbaqusFeatureError) as caught:
        abaqus.build_model_with_report(parse_file(path))

    assert caught.value.code == "abaqus.normal.element_type_unsupported"
    assert caught.value.record == ("1", "1", "0.", "0.", "1.")


def test_normal_targeting_unknown_element_is_rejected_with_source_evidence(
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

    with pytest.raises(abaqus.AbaqusBuildError) as caught:
        abaqus.build_model_with_report(parse_file(path))

    error = caught.value
    assert error.code == "abaqus.b31.normal.element_missing"
    assert error.path == path
    assert error.line == 14
    assert error.keyword == "normal"
    assert error.record == ("99", "1", "0.", "0.", "1.")
    assert error.remediation


@pytest.mark.parametrize(
    ("node_line", "expected_code", "expected_record"),
    (
        (
            "1, 0., 0., , , 0., 1.",
            "abaqus.b31.node_normal_empty",
            ("", "0.", "1."),
        ),
        (
            "1, 0., 0., , 0., 1.",
            "abaqus.b31.node_normal_shape",
            ("0.", "1."),
        ),
    ),
)
def test_node_normal_empty_or_incomplete_components_are_rejected(
    tmp_path: Path,
    node_line: str,
    expected_code: str,
    expected_record: tuple[str, ...],
) -> None:
    path = _write_deck(
        tmp_path,
        f"{expected_code.rsplit('.', 1)[-1]}.inp",
        _beam_lines(
            nodes=(node_line, "2, 1., 0., 0."),
        ),
    )

    with pytest.raises(abaqus.AbaqusBuildError) as caught:
        abaqus.build_model_with_report(parse_file(path))

    error = caught.value
    assert error.code == expected_code
    assert error.path == path
    assert error.line == 3
    assert error.keyword == "node"
    assert error.record == expected_record
    assert error.remediation


def test_normal_conflict_and_invalid_local_end_are_deterministic(
    tmp_path: Path,
) -> None:
    conflict_path = _write_deck(
        tmp_path,
        "normal_conflict.inp",
        _beam_lines(
            extra=(
                "*Normal, type=ELEMENT",
                "1, 1, 0., 0., 1.",
                "1, 1, 0., 1., 0.",
            ),
        ),
    )
    with pytest.raises(abaqus.AbaqusBuildError) as conflict:
        abaqus.build_model_with_report(parse_file(conflict_path))
    assert conflict.value.code == "abaqus.b31.normal.conflict"
    assert len(conflict.value.locations) == 2
    assert {location.line for location in conflict.value.locations} == {14, 15}

    invalid_path = _write_deck(
        tmp_path,
        "normal_invalid_end.inp",
        _beam_lines(
            nodes=(
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 0., 0., 1.",
            ),
            extra=(
                "*Normal, type=ELEMENT",
                "1, 3, 0., 0., 1.",
            ),
        ),
    )
    with pytest.raises(abaqus.AbaqusBuildError) as invalid:
        abaqus.build_model_with_report(parse_file(invalid_path))
    assert invalid.value.code == "abaqus.b31.normal.local_end_invalid"
    assert invalid.value.location is not None
    assert invalid.value.location.path == invalid_path


def test_deck_snapshot_is_detached_and_map_order_is_deterministic(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "snapshot.inp",
        _beam_lines(
            nodes=(
                "2, 1., 0., 0.",
                "1, 0., 0., 0.",
            ),
            connectivity="1, 1, 2",
        ),
    )
    deck = parse_file(path)
    snapshot = deck.snapshot()
    copied = deepcopy(deck)

    assert snapshot is not deck
    assert copied is not deck
    assert snapshot.nodes is not deck.nodes
    assert snapshot.node_records is not deck.node_records
    assert snapshot.normal_records is not deck.normal_records
    assert list(snapshot.nodes) == [1, 2]
    assert list(snapshot.node_records) == [1, 2]

    snapshot.nodes[1] = (99.0, 99.0, 99.0)
    snapshot.normal_records.append("owned mutation")
    assert deck.nodes[1] == (0.0, 0.0, 0.0)
    assert deck.normal_records == []
    assert copied.nodes[1] == (0.0, 0.0, 0.0)


def test_nonfinite_normal_is_rejected_at_parser_boundary(tmp_path: Path) -> None:
    path = _write_deck(
        tmp_path,
        "nonfinite_normal.inp",
        _beam_lines(
            extra=(
                "*Normal, type=ELEMENT",
                "1, 1, 1e999, 0., 1.",
            ),
        ),
    )

    with pytest.raises(abaqus.AbaqusParseError) as caught:
        parse_file(path)
    assert caught.value.code == "abaqus.real.nonfinite"
    assert caught.value.keyword == "normal"


def test_empty_normal_record_is_rejected_at_parser_boundary(tmp_path: Path) -> None:
    path = _write_deck(
        tmp_path,
        "empty_normal.inp",
        _beam_lines(extra=("*Normal, type=ELEMENT", "")),
    )

    with pytest.raises(abaqus.AbaqusParseError) as caught:
        parse_file(path)
    assert caught.value.code == "abaqus.b31.normal.record_shape"
    assert caught.value.keyword == "normal"
