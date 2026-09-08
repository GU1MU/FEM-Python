from __future__ import annotations

from fem.io import inp as abaqus


B31_TIMOSHENKO_NOTICE = "abaqus.b31.linear_timoshenko_support_boundary"


def test_inline_b31_reports_linear_timoshenko_support_boundary(tmp_path) -> None:
    source = tmp_path / "beam.inp"
    source.write_text(
        "\n".join(
            (
                "*Heading",
                "inline B31 support boundary",
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
            )
        )
        + "\n",
        encoding="utf-8",
    )

    imported = abaqus.read_with_report(source)

    assert imported.model.mesh.elements[0].type == "Beam2"
    assert tuple(notice.code for notice in imported.notices) == (
        B31_TIMOSHENKO_NOTICE,
    )
    message = imported.notices[0].message.casefold()
    assert "linear" in message
    assert "static" in message
    assert "timoshenko" in message
    assert "transverse shear deformation" in message
    assert "element-length slenderness compensation" in message
    assert "integration-point" in message
    assert "rect, circ, and thick pipe" in message
    assert "euler" not in message
    assert "bernoulli" not in message
    for unsupported_boundary in (
        "nonlinear",
        "dynamic",
        "b31h",
        "user-defined transverse shear stiffness",
        "seventh degree of freedom",
        "thin-wall pipe",
        "arbitrary sections",
        "nonuniform line loads",
        "curved elements",
    ):
        assert unsupported_boundary in message
