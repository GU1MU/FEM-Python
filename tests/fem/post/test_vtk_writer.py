from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fem.post.vtk.writer import (
    append_legacy_ascii_unstructured_grid_geometry,
)


def _append(
    lines: list[str],
    **overrides: Any,
) -> None:
    arguments = {
        "title": "canonical result",
        "points": ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0)),
        "cells": ((0, 1),),
        "cell_types": (3,),
        "numeric_declaration": "double",
        "format_float": lambda value: format(float(value), ".17g"),
    }
    arguments.update(overrides)
    append_legacy_ascii_unstructured_grid_geometry(
        lines,
        **arguments,
    )


def test_append_geometry_emits_exact_legacy_ascii_prefix() -> None:
    lines = ["caller-owned-prefix"]

    _append(lines)

    assert lines == [
        "caller-owned-prefix",
        "# vtk DataFile Version 3.0",
        "canonical result",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "POINTS 2 double",
        "0 1 2",
        "3 4 5",
        "CELLS 1 3",
        "2 0 1",
        "CELL_TYPES 1",
        "3",
    ]


def test_append_geometry_supports_point_only_zero_cell_dataset() -> None:
    lines: list[str] = []

    _append(
        lines,
        points=((1.25, -2.5, 0.0),),
        cells=(),
        cell_types=(),
        numeric_declaration="float",
    )

    assert lines[-4:] == [
        "POINTS 1 float",
        "1.25 -2.5 0",
        "CELLS 0 0",
        "CELL_TYPES 0",
    ]


def test_caller_float_formatter_owns_numeric_text() -> None:
    observed: list[object] = []

    def formatter(value: object) -> str:
        observed.append(value)
        return f"<{value}>"

    lines: list[str] = []
    _append(
        lines,
        points=((1, 2, 3),),
        cells=(),
        cell_types=(),
        format_float=formatter,
    )

    assert observed == [1, 2, 3]
    assert "POINTS 1 double\n<1> <2> <3>" in "\n".join(lines)


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    (
        ({"title": ""}, ValueError, "title"),
        ({"title": "two\nlines"}, ValueError, "title"),
        ({"title": "结果"}, ValueError, "ASCII"),
        ({"title": "x" * 257}, ValueError, "256"),
        (
            {"numeric_declaration": 1},
            TypeError,
            "numeric_declaration",
        ),
        (
            {"numeric_declaration": "int"},
            ValueError,
            "float or double",
        ),
        ({"points": ((0.0, 1.0),)}, ValueError, "three"),
        ({"points": ("invalid",)}, TypeError, "coordinate sequences"),
        ({"cells": ((),)}, ValueError, "empty connectivity"),
        ({"cells": ((0, 0),)}, ValueError, "repeat"),
        ({"cells": ((0, 2),)}, ValueError, "unknown point"),
        ({"cells": ((0, True),)}, TypeError, "integers"),
        ({"cell_types": ()}, ValueError, "length"),
        ({"cell_types": (True,)}, TypeError, "integers"),
        ({"cell_types": (0,)}, ValueError, "positive"),
        ({"format_float": None}, TypeError, "callable"),
        (
            {"format_float": lambda _value: "two tokens"},
            ValueError,
            "numeric token",
        ),
    ),
)
def test_validation_is_strict_and_never_partially_appends(
    overrides: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    lines = ["existing"]

    with pytest.raises(error, match=match):
        _append(lines, **overrides)

    assert lines == ["existing"]


@pytest.mark.parametrize(
    "lines",
    (
        (),
        ["valid", 1],
    ),
)
def test_lines_must_be_a_string_list(lines: object) -> None:
    with pytest.raises(TypeError, match="list containing only strings"):
        _append(lines)  # type: ignore[arg-type]


def test_low_level_helper_has_no_application_gui_or_optional_imports() -> None:
    module_path = Path(
        append_legacy_ascii_unstructured_grid_geometry.__code__.co_filename
    )
    source = module_path.read_text(encoding="utf-8")
    forbidden = (
        "fem.application",
        "fem_gui",
        "PySide",
        "pyvista",
        "vtkmodules",
    )

    assert not any(name in source for name in forbidden)
