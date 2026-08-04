"""Independent Phase 0 characterization of current Abaqus B31 behavior.

These assertions are migration baselines, not permanent rejection contracts.  All
inputs are deliberately small and are written from inline text into pytest's
temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fem import abaqus


def _write_deck(
    tmp_path: Path,
    filename: str,
    *,
    nodes: tuple[str, ...],
    elements: tuple[str, ...],
    preamble: tuple[str, ...] = (),
    node_set: bool = False,
    node_suffix: tuple[str, ...] = (),
    orientation: tuple[float, float, float] = (0.0, 0.0, 1.0),
    tail: tuple[str, ...] = (),
) -> Path:
    lines = [
        "*Heading",
        *preamble,
        "*Node",
        *nodes,
        "*Element, type=B31, elset=BEAM",
        *elements,
    ]
    if node_set:
        lines.extend(("*Nset, nset=ALL", "1, 2"))
    lines.extend(
        (
            "*Material, name=STEEL",
            "*Elastic",
            "2.10E11, 0.30",
            "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
            "0.20, 0.10",
            ", ".join(str(value) for value in orientation),
            *node_suffix,
            *tail,
        )
    )
    path = tmp_path / filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _assert_current_topology_rejection(
    path: Path,
    *,
    reason: str,
) -> None:
    with pytest.raises(abaqus.UnsupportedAbaqusFeatureError) as caught:
        abaqus.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.b31.nodal_normal_averaging_unsupported"
    assert error.path == path
    assert error.line > 0
    assert error.keyword == "element"
    assert error.locations
    assert reason in str(error).casefold()
    assert error.remediation


def test_characterization_preprint_is_currently_a_build_blocker(tmp_path: Path) -> None:
    path = _write_deck(
        tmp_path,
        "preprint.inp",
        preamble=("*Preprint, echo=NO, history=NO, model=NO, contact=NO",),
        nodes=("1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0"),
        elements=("1, 1, 2",),
    )

    with pytest.raises(abaqus.UnsupportedAbaqusFeatureError) as caught:
        abaqus.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.line.keyword_unsupported"
    assert error.keyword == "preprint"
    assert error.path == path
    assert error.remediation


@pytest.mark.parametrize(
    ("filename", "nodes", "elements", "reason"),
    (
        (
            "kink.inp",
            ("1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0", "3, 1.0, 1.0, 0.0"),
            ("1, 1, 2", "2, 2, 3"),
            "kink",
        ),
        (
            "t_junction.inp",
            (
                "1, 0.0, 0.0, 0.0",
                "2, 1.0, 0.0, 0.0",
                "3, 2.0, 0.0, 0.0",
                "4, 1.0, 1.0, 0.0",
            ),
            ("1, 1, 2", "2, 2, 3", "3, 2, 4"),
            "branch or junction",
        ),
        (
            "closed_loop.inp",
            ("1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0", "3, 0.5, 1.0, 0.0"),
            ("1, 1, 2", "2, 2, 3", "3, 3, 1"),
            "closed loop",
        ),
    ),
)
def test_characterization_topology_gate_rejects_currently(
    tmp_path: Path,
    filename: str,
    nodes: tuple[str, ...],
    elements: tuple[str, ...],
    reason: str,
) -> None:
    path = _write_deck(
        tmp_path,
        filename,
        nodes=nodes,
        elements=elements,
    )

    _assert_current_topology_rejection(path, reason=reason)


def test_characterization_orientation_node_is_currently_rejected_during_parse(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "orientation_node.inp",
        nodes=(
            "1, 0.0, 0.0, 0.0",
            "2, 1.0, 0.0, 0.0",
            "3, 0.0, 1.0, 0.0",
        ),
        elements=("1, 1, 2, 3",),
    )

    with pytest.raises(abaqus.UnsupportedAbaqusFeatureError) as caught:
        abaqus.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.b31.orientation_node_unsupported"
    assert error.path == path
    assert error.keyword == "element"
    assert error.record == ("1", "1", "2", "3")
    assert "orientation node" in str(error).casefold()


def test_characterization_nodal_normal_components_are_currently_rejected(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "nodal_normal.inp",
        nodes=(
            "1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0",
            "2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0",
        ),
        elements=("1, 1, 2",),
    )

    with pytest.raises(abaqus.UnsupportedAbaqusFeatureError) as caught:
        abaqus.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.b31.nodal_normals_unsupported"
    assert error.path == path
    assert error.keyword == "node"
    assert error.line == 3
    assert error.record == ("0.0", "1.0", "0.0")
    assert error.remediation


def test_characterization_normal_keyword_is_currently_outside_line_subset(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "normal_keyword.inp",
        nodes=("1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0"),
        elements=("1, 1, 2",),
        tail=("*Normal", "1, 1, 0.0, 1.0, 0.0"),
    )

    with pytest.raises(abaqus.UnsupportedAbaqusFeatureError) as caught:
        abaqus.read_with_report(path)

    error = caught.value
    assert error.code == "abaqus.line.keyword_unsupported"
    assert error.path == path
    assert error.keyword == "normal"


def test_characterization_output_parent_child_is_preserved_without_blocking_import(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "output_parent_child.inp",
        nodes=("1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0"),
        elements=("1, 1, 2",),
        node_set=True,
        tail=(
            "*Step, name=STATIC",
            "*Static",
            "*Output, FIELD, VARIABLE=PRESELECT, FREQUENCY=1",
            "*Node Output, NSET=ALL",
            "U, RF",
            "*Element Output, ELSET=BEAM, DIRECTIONS=YES",
            "S, E",
            "*End Step",
        ),
    )

    result = abaqus.read_with_report(path)

    assert tuple(notice.code for notice in result.notices) == (
        "abaqus.b31.euler_bernoulli_approximation",
    )
    requests = result.model.steps[0].outputs
    assert tuple((item.kind, item.target, item.variables) for item in requests) == (
        ("field", "preselect", ("PRESELECT",)),
        ("field", "node", ("U", "RF")),
        ("field", "element", ("S", "E")),
    )
    assert requests[1].source_evidence is not None
    assert requests[1].source_evidence.parent_flags == ("field",)
    assert requests[1].source_evidence.parent_parameters == (
        ("variable", "PRESELECT"),
        ("frequency", "1"),
    )
    assert requests[1].source_evidence.child_parameters == (("nset", "ALL"),)
    assert requests[2].source_evidence is not None
    assert requests[2].source_evidence.child_parameters == (
        ("elset", "BEAM"),
        ("directions", "YES"),
    )
