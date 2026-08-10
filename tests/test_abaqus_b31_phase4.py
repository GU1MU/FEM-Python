"""Public-facade coverage for deterministic B31 normal resolution."""

from __future__ import annotations

from copy import deepcopy
from math import cos, radians, sin
from pathlib import Path

import numpy as np
import pytest

from fem.assemble import assemble_global_stiffness
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.application import RegionRef, resolve_effective_beam_frames
from fem.elements import resolve_beam_frame
from fem.io import inp
from fem.materials import apply_sections


def _write(tmp_path: Path, name: str, lines: tuple[str, ...]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _deck(
    nodes: tuple[str, ...],
    elements: tuple[str, ...],
    *,
    n1: str = "0., 0., 1.",
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    return (
        "*Heading",
        "Phase 4 public facade",
        "*Node",
        *nodes,
        "*Element, type=B31, elset=BEAMS",
        *elements,
        "*Material, name=STEEL",
        "*Elastic",
        "210000., 0.3",
        "*Beam Section, elset=BEAMS, material=STEEL, section=RECT",
        "0.2, 0.1",
        n1,
        *extra,
    )


def _frame_report(model):
    report = resolve_effective_beam_frames(
        model,
        RegionRef("element_set", "BEAMS"),
    )
    assert report.passed
    return report


def _star_deck(angles: tuple[float, ...]) -> tuple[str, ...]:
    nodes = tuple(
        ["0, 0., 0., 0."]
        + [
            (
                f"{index}, {cos(radians(angle))}, {sin(radians(angle))}, "
                f"{0.1 * angle:g}"
            )
            for index, angle in enumerate(angles, start=1)
        ]
    )
    elements = tuple(
        f"{index}, 0, {index}" for index in range(1, len(angles) + 1)
    )
    return _deck(nodes, elements)


def _center_local_z(model) -> dict[int, np.ndarray]:
    return {
        int(element.id): np.asarray(
            element.props["beam_frame_field"].start.local_z,
            dtype=float,
        )
        for element in model.mesh.elements
    }


def _generated_normal(angle: float) -> np.ndarray:
    tangent = _star_tangent(angle)
    normal = np.cross(tangent, np.asarray((0.0, 0.0, 1.0)))
    return normal / np.linalg.norm(normal)


def _star_tangent(angle: float) -> np.ndarray:
    tangent = np.asarray(
        (cos(radians(angle)), sin(radians(angle)), 0.1 * angle),
        dtype=float,
    )
    return tangent / np.linalg.norm(tangent)


def _effective_normal(normal: np.ndarray, angle: float) -> np.ndarray:
    tangent = _star_tangent(angle)
    projected = normal - float(normal @ tangent) * tangent
    return projected / np.linalg.norm(projected)


def _averaged_normal(angles: tuple[float, ...]) -> np.ndarray:
    value = sum((_generated_normal(angle) for angle in angles), np.zeros(3))
    return value / np.linalg.norm(value)


def _covariance_lines(
    connectivity: str,
    dloads: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        "*Heading",
        "Phase 4 covariance",
        "*Node",
        "1, 0., 0., 0.",
        "2, 2., 0., 0.",
        "*Element, type=B31, elset=BEAM",
        connectivity,
        "*Material, name=STEEL",
        "*Elastic",
        "210000., 0.3",
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.2, 0.1",
        "0., 0., 1.",
        "*Step, name=LOAD",
        "*Static",
        "*Dload",
        *dloads,
        "*End Step",
    )


def test_isolated_default_frame_is_valid_and_right_handed(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "isolated.inp",
        _deck(("1, 0., 0., 0.", "2, 1., 0., 0."), ("1, 1, 2",)),
    )

    result = inp.read_with_report(path)
    frame = _frame_report(result.model).frames[0]
    assert frame.local_x == pytest.approx((1.0, 0.0, 0.0))
    assert frame.local_y == pytest.approx((0.0, 0.0, 1.0))
    assert frame.local_z == pytest.approx((0.0, -1.0, 0.0))
    assert (
        result.notices[0].code
        == "abaqus.b31.linear_timoshenko_support_boundary"
    )


def test_kink_and_branch_keep_shared_nodes_and_report_generated_groups(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "branch.inp",
        _deck(
            (
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 2., 0., 0.",
                "4, 1., 1., 0.",
                "5, 1., 0., 1.",
            ),
            ("1, 1, 2", "2, 2, 3", "3, 2, 4", "4, 2, 5"),
            n1="0., 1., 1.",
        ),
    )

    result = inp.read_with_report(path)

    assert result.model.mesh.num_nodes == 5
    assert result.model.mesh.num_elements == 4
    assert tuple(notice.code for notice in result.notices) == (
        "abaqus.b31.linear_timoshenko_support_boundary",
        "abaqus.b31.nodal_normal_generation_approximation",
    )
    report = _frame_report(result.model)
    assert len(report.frames) == 4
    assert len(report.frame_fields) == 4


def test_closed_loop_and_reversed_connectivity_are_publicly_accepted(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "loop.inp",
        _deck(
            (
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 1., 1., 0.",
                "4, 0., 1., 0.",
            ),
            ("1, 1, 2", "2, 3, 2", "3, 3, 4", "4, 1, 4"),
        ),
    )

    result = inp.read_with_report(path)

    assert tuple(
        tuple(element.node_ids) for element in result.model.mesh.elements
    ) == ((1, 2), (3, 2), (3, 4), (1, 4))
    assert len(_frame_report(result.model).frames) == 4


def test_explicit_normals_win_over_generated_grouping(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "explicit.inp",
        _deck(
            ("1, 0., 0., 0.", "2, 1., 0., 0."),
            ("1, 1, 2",),
            n1="0., 1., 0.",
            extra=(
                "*Normal, type=ELEMENT",
                "1, 1, 0., 0., 1.",
                "1, 2, 0., 0., 1.",
            ),
        ),
    )

    result = inp.read_with_report(path)
    frame = _frame_report(result.model).frames[0]
    assert frame.local_y == pytest.approx((0.0, 1.0, 0.0))
    assert result.source_summary is not None
    assert sum(
        occurrence.name == "normal"
        for occurrence in result.source_summary.occurrences
    ) == 1


def test_conflicting_element_normal_is_reported_without_installing_a_model(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "conflict.inp",
        _deck(
            ("1, 0., 0., 0.", "2, 1., 0., 0."),
            ("1, 1, 2",),
            extra=(
                "*Normal, type=ELEMENT",
                "1, 1, 0., 0., 1.",
                "1, 1, 0., 1., 0.",
            ),
        ),
    )

    with pytest.raises(inp.InpBuildError) as caught:
        inp.read_with_report(path)

    assert caught.value.code == "abaqus.b31.normal.conflict"
    assert caught.value.path == path
    assert caught.value.locations


def test_official_pairwise_twenty_degree_group_is_averaged_publicly(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "pairwise_group.inp", _star_deck((0.0, 10.0, 15.0)))

    result = inp.read_with_report(path)
    local_z = _center_local_z(result.model)
    averaged = _averaged_normal((0.0, 10.0, 15.0))

    for index, angle in enumerate((0.0, 10.0, 15.0), start=1):
        np.testing.assert_allclose(
            local_z[index],
            _effective_normal(averaged, angle),
            atol=1e-12,
        )
    assert not np.allclose(local_z[2], _generated_normal(10.0))
    assert any(
        notice.code == "abaqus.b31.nodal_normal_generation_approximation"
        for notice in result.notices
    )
    effective = _frame_report(result.model)
    assert all(
        np.allclose(effective.frames[index].local_z, local_z[index + 1])
        for index in range(3)
    )


def test_official_zero_ten_forty_boundary_is_two_plus_one_publicly(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "two_plus_one.inp", _star_deck((0.0, 10.0, 40.0)))

    local_z = _center_local_z(inp.read_with_report(path).model)
    averaged = _averaged_normal((0.0, 10.0))

    np.testing.assert_allclose(
        local_z[1], _effective_normal(averaged, 0.0), atol=1e-12
    )
    np.testing.assert_allclose(
        local_z[2], _effective_normal(averaged, 10.0), atol=1e-12
    )
    np.testing.assert_allclose(local_z[3], _generated_normal(40.0), atol=1e-12)
    assert not np.allclose(local_z[2], _generated_normal(10.0))


def test_official_non_clique_group_is_fully_split_and_order_stable(
    tmp_path: Path,
) -> None:
    angles = (0.0, 10.0, 30.0)
    forward = _write(tmp_path, "non_clique_forward.inp", _star_deck(angles))
    reversed_lines = list(_star_deck(angles))
    element_start = reversed_lines.index("*Element, type=B31, elset=BEAMS") + 1
    reversed_lines[element_start:] = [
        reversed_lines[element_start + 2],
        reversed_lines[element_start + 1],
        reversed_lines[element_start],
        *reversed_lines[element_start + 3 :],
    ]
    permuted = _write(
        tmp_path,
        "non_clique_permuted.inp",
        tuple(reversed_lines),
    )

    forward_z = _center_local_z(inp.read_with_report(forward).model)
    permuted_z = _center_local_z(inp.read_with_report(permuted).model)

    assert not np.allclose(forward_z[1], forward_z[2])
    assert not np.allclose(forward_z[2], forward_z[3])
    assert not np.allclose(forward_z[1], forward_z[3])
    assert all(np.allclose(forward_z[index], permuted_z[index]) for index in (1, 2, 3))


def test_more_than_thirty_remaining_shared_elements_are_split_publicly(
    tmp_path: Path,
) -> None:
    angles = tuple(index * 0.5 for index in range(31))
    path = _write(tmp_path, "more_than_thirty.inp", _star_deck(angles))

    local_z = _center_local_z(inp.read_with_report(path).model)

    assert len(local_z) == 31
    for index, angle in enumerate(angles, start=1):
        np.testing.assert_allclose(
            local_z[index], _generated_normal(angle), atol=1e-12
        )
    quantized = {tuple(np.round(value, 8)) for value in local_z.values()}
    assert len(quantized) == 31


def test_public_inp_connectivity_reversal_covaries_stiffness_and_line_loads(
    tmp_path: Path,
) -> None:
    global_dloads = ("BEAM, PY, 2.0", "BEAM, PZ, 1.5")
    forward_global_path = _write(
        tmp_path,
        "forward_global.inp",
        _covariance_lines("1, 1, 2", global_dloads),
    )
    reversed_global_path = _write(
        tmp_path,
        "reversed_global.inp",
        _covariance_lines("1, 2, 1", global_dloads),
    )
    forward_global = inp.read(forward_global_path)
    reversed_global = inp.read(reversed_global_path)
    forward_global_materialized = deepcopy(forward_global)
    reversed_global_materialized = deepcopy(reversed_global)
    apply_sections(forward_global_materialized)
    apply_sections(reversed_global_materialized)

    np.testing.assert_allclose(
        assemble_global_stiffness(forward_global_materialized.mesh),
        assemble_global_stiffness(reversed_global_materialized.mesh),
        rtol=1e-10,
        atol=1e-10,
    )
    forward_global_load = build_load_vector(
        forward_global_materialized.mesh,
        boundary_for_step(forward_global_materialized, "LOAD"),
    )
    reversed_global_load = build_load_vector(
        reversed_global_materialized.mesh,
        boundary_for_step(reversed_global_materialized, "LOAD"),
    )
    np.testing.assert_allclose(forward_global_load, reversed_global_load, atol=1e-12)

    global_vector = np.asarray((0.0, 2.0, 1.5))
    forward_local = resolve_beam_frame(
        forward_global.mesh,
        forward_global.mesh.elements[0],
    ).rotation @ global_vector
    reversed_local = resolve_beam_frame(
        reversed_global.mesh,
        reversed_global.mesh.elements[0],
    ).rotation @ global_vector
    assert forward_local[0] == pytest.approx(0.0)
    assert reversed_local[0] == pytest.approx(0.0)

    forward_local_model = inp.read(
        _write(
            tmp_path,
            "forward_local.inp",
            _covariance_lines(
                "1, 1, 2",
                (f"BEAM, P1, {forward_local[1]}", f"BEAM, P2, {forward_local[2]}"),
            ),
        )
    )
    reversed_local_model = inp.read(
        _write(
            tmp_path,
            "reversed_local.inp",
            _covariance_lines(
                "1, 2, 1",
                (f"BEAM, P1, {reversed_local[1]}", f"BEAM, P2, {reversed_local[2]}"),
            ),
        )
    )
    forward_local_materialized = deepcopy(forward_local_model)
    reversed_local_materialized = deepcopy(reversed_local_model)
    apply_sections(forward_local_materialized)
    apply_sections(reversed_local_materialized)
    forward_local_load = build_load_vector(
        forward_local_materialized.mesh,
        boundary_for_step(forward_local_materialized, "LOAD"),
    )
    reversed_local_load = build_load_vector(
        reversed_local_materialized.mesh,
        boundary_for_step(reversed_local_materialized, "LOAD"),
    )
    np.testing.assert_allclose(forward_local_load, forward_global_load, atol=1e-12)
    np.testing.assert_allclose(reversed_local_load, forward_global_load, atol=1e-12)
