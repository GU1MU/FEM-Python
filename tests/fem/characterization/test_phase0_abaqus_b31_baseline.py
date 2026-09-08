"""Independent Phase 0 characterization of current Abaqus B31 behavior.

These assertions are migration baselines, not permanent rejection contracts.  All
inputs are deliberately small and are written from inline text into pytest's
temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fem.io import inp


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


def test_phase1_preprint_is_harmless_and_preserves_source_occurrence(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "preprint.inp",
        preamble=("*Preprint, echo=NO, history=NO, model=NO, contact=NO",),
        nodes=("1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0"),
        elements=("1, 1, 2",),
    )

    result = inp.read_with_report(path)

    assert result.model.mesh.num_nodes == 2
    assert result.model.mesh.num_elements == 1
    assert result.source_summary is not None
    occurrences = result.source_summary.occurrences
    preprints = tuple(
        occurrence for occurrence in occurrences
        if occurrence.name == "preprint"
    )
    assert len(preprints) == 1
    assert preprints[0].category is inp.InpKeywordCategory.HARMLESS_IGNORED
    assert preprints[0].params == (
        ("echo", "NO"),
        ("history", "NO"),
        ("model", "NO"),
        ("contact", "NO"),
    )
    assert preprints[0].location.path == path
    assert preprints[0].location.line == 2


@pytest.mark.parametrize(
    ("filename", "nodes", "elements"),
    (
        (
            "kink.inp",
            ("1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0", "3, 1.0, 1.0, 0.0"),
            ("1, 1, 2", "2, 2, 3"),
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
        ),
        (
            "closed_loop.inp",
            ("1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0", "3, 0.5, 1.0, 0.0"),
            ("1, 1, 2", "2, 2, 3", "3, 3, 1"),
        ),
    ),
)
def test_characterization_topology_is_accepted_through_public_entry(
    tmp_path: Path,
    filename: str,
    nodes: tuple[str, ...],
    elements: tuple[str, ...],
) -> None:
    path = _write_deck(
        tmp_path,
        filename,
        nodes=nodes,
        elements=elements,
    )

    result = inp.read_with_report(path)

    assert result.model.mesh.num_nodes == len(nodes)
    assert result.model.mesh.num_elements == len(elements)
    assert tuple(
        tuple(element.node_ids) for element in result.model.mesh.elements
    ) == tuple(
        tuple(int(value.strip()) for value in element.split(",")[1:])
        for element in elements
    )


def test_characterization_orientation_node_is_accepted_through_public_entry(
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

    result = inp.read_with_report(path)
    assert result.model.mesh.num_nodes == 2
    assert result.model.mesh.num_elements == 1
    assert tuple(result.model.mesh.elements[0].node_ids) == (1, 2)
    assert result.model.mesh.num_dofs == 12
    assert result.source_summary is not None
    assert any(
        occurrence.name == "element"
        for occurrence in result.source_summary.occurrences
    )


def test_characterization_nodal_normal_components_are_accepted_through_public_entry(
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

    result = inp.read_with_report(path)
    assert result.model.mesh.num_nodes == 2
    assert result.model.mesh.num_elements == 1
    assert result.source_summary is not None
    node_occurrence = next(
        occurrence
        for occurrence in result.source_summary.occurrences
        if occurrence.name == "node"
    )
    assert node_occurrence.location.line == 2


def test_characterization_normal_keyword_is_accepted_through_public_entry(
    tmp_path: Path,
) -> None:
    path = _write_deck(
        tmp_path,
        "normal_keyword.inp",
        nodes=("1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0"),
        elements=("1, 1, 2",),
        tail=("*Normal", "1, 1, 0.0, 1.0, 0.0"),
    )

    result = inp.read_with_report(path)
    assert result.model.mesh.num_nodes == 2
    assert result.model.mesh.num_elements == 1
    assert result.source_summary is not None
    normal_occurrence = next(
        occurrence
        for occurrence in result.source_summary.occurrences
        if occurrence.name == "normal"
    )
    assert normal_occurrence.location.keyword == "normal"
    assert normal_occurrence.location.line == 13


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

    result = inp.read_with_report(path)

    assert tuple(notice.code for notice in result.notices) == (
        "abaqus.b31.linear_timoshenko_support_boundary",
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
