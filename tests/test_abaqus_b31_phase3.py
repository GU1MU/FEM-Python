"""Public-facade coverage for B31 orientation source projection."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

from fem.application import RegionRef, resolve_effective_beam_frames
from fem.assemble import assemble_global_stiffness
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.elements import get_element_kernel
from fem.io import inp
from fem.materials import apply_sections
from fem.solvers.static_linear import solve


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


def test_public_b31_import_transfers_fresh_source_deck_without_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = _write_deck(tmp_path, "single_snapshot.inp", _beam_lines())
    inp.read_with_report(path)
    AbaqusDeck = sys.modules["fem.io._inp.deck"].AbaqusDeck
    original = AbaqusDeck.snapshot
    calls = 0

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(AbaqusDeck, "snapshot", counted)

    result = inp.read_with_report(path)

    assert result.model.mesh.elements[0].type == "Beam2"
    assert calls == 0


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


def test_default_n1_and_generated_normal_are_equivalent_without_old_notice(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "default_n1.inp",
        _beam_lines(section_n1=""),
    )

    result = inp.read_with_report(path)
    frame = _frames(result.model).frames[0]

    assert frame.local_y == pytest.approx((0.0, 0.0, -1.0))
    assert frame.local_z == pytest.approx((0.0, 1.0, 0.0))
    assert tuple(notice.code for notice in result.notices) == (
        "abaqus.b31.linear_timoshenko_support_boundary",
    )


def _order_invariance_lines(*, permuted: bool) -> tuple[str, ...]:
    nodes = (
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "3, 0.984807753012208, 0.17364817766693, 0.",
        "4, 0.766044443118978, 0.642787609686539, 0.",
    )
    elements = (
        "10, 1, 2",
        "20, 1, 3",
        "30, 1, 4",
    )
    if permuted:
        nodes = (nodes[2], nodes[0], nodes[3], nodes[1])
        elements = (elements[2], elements[0], elements[1])
        beam_set = "20, 30, 10"
    else:
        beam_set = "10, 20, 30"
    return (
        "*Heading",
        "Phase 3 order invariance",
        "*Node",
        *nodes,
        "*Element, type=B31",
        *elements,
        "*Elset, elset=BEAMS",
        beam_set,
        "*Nset, nset=FIXED",
        "1",
        "*Material, name=STEEL",
        "*Elastic",
        "210000., 0.3",
        "*Beam Section, elset=BEAMS, material=STEEL, section=RECT",
        "0.2, 0.1",
        "0., 0., 1.",
        "*Step, name=LOAD",
        "*Static",
        "*Boundary",
        "FIXED, 1, 6, 0.",
        "*Dload",
        "BEAMS, P1, 2.5",
        "*End Step",
    )


def _order_invariance_projection(model):
    owned = deepcopy(model)
    apply_sections(owned)
    boundary = boundary_for_step(owned, "LOAD")
    stiffness = assemble_global_stiffness(owned.mesh)
    load = build_load_vector(owned.mesh, boundary)
    displacement = solve(owned, "LOAD").U
    dofs = np.concatenate(
        [
            np.asarray(owned.mesh.node_dofs(node.id), dtype=int)
            for node in sorted(owned.mesh.nodes, key=lambda item: item.id)
        ]
    )
    node_lookup = {node.id: node for node in owned.mesh.nodes}
    resolved_loads = {item.elem_id: item for item in boundary.line_loads}
    recovered = []
    frame_fields = []
    for element in sorted(owned.mesh.elements, key=lambda item: item.id):
        field = element.props["beam_frame_field"]
        frame_fields.append(
            (
                int(element.id),
                field.start.rotation.copy(),
                field.end.rotation.copy(),
            )
        )
        element_load = resolved_loads[int(element.id)]
        kernel = get_element_kernel(element.type)
        equivalent_load = kernel.local_line_load(
            owned.mesh,
            element,
            element_load.vector,
            element_load.coordinate_system,
            node_lookup,
        )
        recovered.append(
            (
                int(element.id),
                kernel.local_end_actions(
                    owned.mesh,
                    element,
                    displacement,
                    equivalent_load,
                    node_lookup,
                ),
            )
        )
    return (
        frame_fields,
        stiffness[np.ix_(dofs, dofs)],
        load[dofs],
        displacement[dofs],
        recovered,
    )


def test_node_element_and_set_permutations_are_identity_deterministic(
    tmp_path: Path,
) -> None:
    forward = inp.read(
        _write_deck(
            tmp_path,
            "forward.inp",
            _order_invariance_lines(permuted=False),
        )
    )
    permuted = inp.read(
        _write_deck(
            tmp_path,
            "permuted.inp",
            _order_invariance_lines(permuted=True),
        )
    )

    expected = _order_invariance_projection(forward)
    actual = _order_invariance_projection(permuted)

    for expected_field, actual_field in zip(expected[0], actual[0], strict=True):
        assert expected_field[0] == actual_field[0]
        np.testing.assert_array_equal(expected_field[1], actual_field[1])
        np.testing.assert_array_equal(expected_field[2], actual_field[2])
    np.testing.assert_array_equal(expected[1], actual[1])
    np.testing.assert_array_equal(expected[2], actual[2])
    np.testing.assert_array_equal(expected[3], actual[3])
    for expected_action, actual_action in zip(expected[4], actual[4], strict=True):
        assert expected_action[0] == actual_action[0]
        np.testing.assert_array_equal(expected_action[1], actual_action[1])


def test_b31_canonical_ordering_does_not_change_non_beam_set_order(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "truss_unsorted_set.inp",
        (
            "*Heading",
            "Phase 3 non-B31 ordering boundary",
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 2., 0., 0.",
            "*Element, type=T3D2, elset=BAR",
            "1, 1, 2",
            "*Nset, nset=ORDERED, unsorted",
            "3, 1, 2",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Solid Section, elset=BAR, material=STEEL",
            "0.1",
        ),
    )

    model = inp.read(path)

    assert model.node_sets["ORDERED"].node_ids == (3, 1, 2)


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
