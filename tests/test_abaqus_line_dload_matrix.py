from __future__ import annotations

import pytest

from fem.io import inp as abaqus
from tests.helpers.file_builders import write_inp


def _b31_lines(
    *records: str,
    keyword: str = "*Dload",
) -> list[str]:
    return [
        "*Heading",
        "*Node",
        "1, 0.0, 0.0, 0.0",
        "2, 1.0, 0.0, 0.0",
        "*Element, type=B31, elset=BEAM",
        "1, 1, 2",
        "*Material, name=STEEL",
        "*Elastic",
        "210000.0, 0.3",
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.2, 0.1",
        "0.0, 1.0, 0.0",
        "*Step, name=LOAD",
        "*Static",
        keyword,
        *records,
        "*End Step",
    ]


def _t3d2_lines(*records: str) -> list[str]:
    return [
        "*Heading",
        "*Node",
        "1, 0.0, 0.0, 0.0",
        "2, 1.0, 0.0, 0.0",
        "*Element, type=T3D2, elset=TRUSS",
        "1, 1, 2",
        "*Material, name=STEEL",
        "*Elastic",
        "210000.0, 0.3",
        "*Solid Section, elset=TRUSS, material=STEEL",
        "0.01",
        "*Step, name=LOAD",
        "*Static",
        "*Dload",
        *records,
        "*End Step",
    ]


def _step(model):
    return next(step for step in model.steps if step.name == "LOAD")


def test_standard_b31_labels_are_case_insensitive_and_preserve_signed_order(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "signed_components.inp",
        _b31_lines(
            "BEAM, px, -1.0",
            "BEAM, Py, 2.0",
            "BEAM, pZ, -3.0",
            "BEAM, P1, 4.0",
            "BEAM, p2, -5.0",
            "BEAM, P1, 6.0",
        ),
    )

    loads = _step(abaqus.read(path)).line_loads

    assert [
        (load.target, load.vector, load.coordinate_system)
        for load in loads
    ] == [
        ("BEAM", (-1.0, 0.0, 0.0), "global"),
        ("BEAM", (0.0, 2.0, 0.0), "global"),
        ("BEAM", (0.0, 0.0, -3.0), "global"),
        ("BEAM", (0.0, 4.0, 0.0), "local"),
        ("BEAM", (0.0, 0.0, -5.0), "local"),
        ("BEAM", (0.0, 6.0, 0.0), "local"),
    ]


def test_b31_dload_accepts_an_element_id_target(tmp_path) -> None:
    path = write_inp(
        tmp_path,
        "element_target.inp",
        _b31_lines("1, PZ, 3.5"),
    )

    load = _step(abaqus.read(path)).line_loads[0]

    assert load.target == 1
    assert load.vector == pytest.approx((0.0, 0.0, 3.5))
    assert load.coordinate_system == "global"


@pytest.mark.parametrize(
    ("target", "expected_code"),
    (
        ("MISSING", "abaqus.dload.target_undefined"),
        ("EMPTY", "abaqus.dload.target_empty"),
    ),
)
def test_b31_dload_rejects_undefined_or_empty_targets(
    tmp_path,
    target: str,
    expected_code: str,
) -> None:
    lines = _b31_lines(f"{target}, P1, 1.0")
    if target == "EMPTY":
        section_index = lines.index(
            "*Beam Section, elset=BEAM, material=STEEL, section=RECT"
        )
        lines[section_index:section_index] = (
            "*Elset, elset=EMPTY",
        )
    path = write_inp(tmp_path, f"{target.casefold()}_target.inp", lines)

    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    assert caught.value.code == expected_code
    assert caught.value.line > 0


def test_mixed_b31_t3d2_dload_target_is_rejected_by_target_family(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "mixed_line_target.inp",
        [
            "*Heading",
            "*Node",
            "1, 0.0, 0.0, 0.0",
            "2, 1.0, 0.0, 0.0",
            "3, 0.0, 1.0, 0.0",
            "4, 1.0, 1.0, 0.0",
            "*Element, type=B31, elset=BEAM",
            "1, 1, 2",
            "*Element, type=T3D2, elset=TRUSS",
            "2, 3, 4",
            "*Elset, elset=MIXED",
            "1, 2",
            "*Material, name=STEEL",
            "*Elastic",
            "210000.0, 0.3",
            "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
            "0.2, 0.1",
            "0.0, 1.0, 0.0",
            "*Solid Section, elset=TRUSS, material=STEEL",
            "0.01",
            "*Step, name=LOAD",
            "*Static",
            "*Dload",
            "MIXED, P1, 1.0",
            "*End Step",
        ],
    )

    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    assert caught.value.code == "abaqus.target.family_mixed"
    assert "MIXED" in str(caught.value)


def test_t3d2_rejects_non_gravity_distributed_load(tmp_path) -> None:
    path = write_inp(
        tmp_path,
        "truss_line_load.inp",
        _t3d2_lines("TRUSS, PX, 1.0"),
    )

    with pytest.raises(
        abaqus.UnsupportedInpFeatureError,
    ) as caught:
        abaqus.read(path)

    assert caught.value.code == "abaqus.t3d2.line_load_unsupported"
    assert caught.value.remediation


def test_b31_rejects_dsload_and_dload_options(tmp_path) -> None:
    dsload = write_inp(
        tmp_path,
        "beam_dsload.inp",
        _b31_lines("BEAM, P1, 1.0", keyword="*Dsload"),
    )
    follower = write_inp(
        tmp_path,
        "beam_follower.inp",
        _b31_lines("BEAM, P1, 1.0", keyword="*Dload, follower=YES"),
    )

    with pytest.raises(
        abaqus.UnsupportedInpFeatureError,
    ) as dsload_error:
        abaqus.read(dsload)
    with pytest.raises(
        abaqus.UnsupportedInpFeatureError,
    ) as follower_error:
        abaqus.read(follower)

    assert "DSLOAD" in str(dsload_error.value).upper()
    assert follower_error.value.remediation


@pytest.mark.parametrize(
    "record",
    (
        "BEAM, P3, 1.0",
        "BEAM, PXNU, 1.0",
        "BEAM, QGLOBAL, 1.0, 0.0, 0.0",
        "BEAM, QLOCAL, 0.0, 1.0, 0.0",
        "BEAM, PX, 1.0, 2.0",
        "BEAM, P1, NaN",
    ),
)
def test_b31_rejects_retired_nonuniform_or_malformed_records(
    tmp_path,
    record: str,
) -> None:
    path = write_inp(
        tmp_path,
        "unsupported_record.inp",
        _b31_lines(record),
    )

    with pytest.raises(abaqus.InpInputError):
        abaqus.read(path)


def test_continuum_p1_p2_remain_face_pressure_labels(tmp_path) -> None:
    path = write_inp(
        tmp_path,
        "continuum_pressure.inp",
        [
            "*Node",
            "1, 0.0, 0.0",
            "2, 1.0, 0.0",
            "3, 1.0, 1.0",
            "4, 0.0, 1.0",
            "*Element, type=CPS4, elset=SOLID",
            "1, 1, 2, 3, 4",
            "*Step, name=LOAD",
            "*Static",
            "*Dload",
            "SOLID, P1, 2.0",
            "SOLID, P2, -3.0",
            "*End Step",
        ],
    )

    step = _step(abaqus.read(path))

    assert step.line_loads == ()
    assert len(step.edge_loads) == 2
    assert [load.magnitude for load in step.edge_loads] == [
        pytest.approx(2.0),
        pytest.approx(-3.0),
    ]


def test_b31_gravity_keeps_target_and_normalizes_direction(tmp_path) -> None:
    path = write_inp(
        tmp_path,
        "beam_gravity.inp",
        _b31_lines("BEAM, GRAV, 9.81, 0.0, -2.0, 0.0"),
    )

    gravity = _step(abaqus.read(path)).gravity_loads[0]

    assert gravity.target == "BEAM"
    assert gravity.acceleration == pytest.approx((0.0, -9.81, 0.0))


def test_failed_dload_build_does_not_mutate_the_parsed_deck(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "transactional_failure.inp",
        _b31_lines("MISSING, P1, 1.0"),
    )
    with pytest.raises(abaqus.InpInputError):
        abaqus.read(path)
