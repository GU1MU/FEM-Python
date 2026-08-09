from __future__ import annotations

from types import SimpleNamespace

import pytest

from fem_gui.inspection_service import AnalysisScopeHighlight
from fem_gui.main_window import FEMMainWindow


@pytest.mark.parametrize(
    "kind",
    (
        "boundary",
        "cload",
        "surface_load",
        "edge_load",
        "line_load",
        "body_load",
        "gravity_load",
    ),
)
def test_every_analysis_condition_routes_through_scope_highlight(kind: str) -> None:
    scope = AnalysisScopeHighlight(
        "surface",
        node_ids=(1,),
        element_ids=(10,),
        members=(SimpleNamespace(node_ids=(1, 2, 3)),),
    )
    resolved: list[tuple[str, object]] = []
    rendered: list[tuple[tuple[object, ...], dict[str, object]]] = []
    window = SimpleNamespace(
        inspection_service=SimpleNamespace(
            analysis_scope_for=lambda requested_kind, key: (
                resolved.append((requested_kind, key)) or scope
            )
        ),
        viewport=SimpleNamespace(
            highlight_analysis_scope=lambda *args, **kwargs: rendered.append(
                (args, kwargs)
            )
        ),
    )

    FEMMainWindow.highlight_entity(window, kind, ("Load", "Condition"))

    assert resolved == [(kind, ("Load", "Condition"))]
    assert rendered == [
        (
            ("surface",),
            {
                "node_ids": (1,),
                "element_ids": (10,),
                "members": scope.members,
            },
        )
    ]


@pytest.mark.parametrize(
    ("kind", "expected_scope_kind"),
    (
        ("node_set", "node"),
        ("element_set", "element"),
        ("surface", "surface"),
        ("edge", "edge"),
    ),
)
def test_named_scope_tree_items_use_the_same_red_scope_renderer(
    kind: str,
    expected_scope_kind: str,
) -> None:
    member = SimpleNamespace(node_ids=(1, 2, 3))
    rendered: list[tuple[tuple[object, ...], dict[str, object]]] = []
    frame_previews: list[object] = []
    selection = SimpleNamespace(node_ids=(1, 2), element_ids=(10, 20))
    window = SimpleNamespace(
        inspection_service=SimpleNamespace(
            selection_for=lambda _kind, _key: selection
        ),
        document=SimpleNamespace(
            model=SimpleNamespace(
                surfaces={"Scope": SimpleNamespace(faces=(member,))},
                edges={"Scope": SimpleNamespace(edges=(member,))},
            )
        ),
        viewport=SimpleNamespace(
            highlight_analysis_scope=lambda *args, **kwargs: rendered.append(
                (args, kwargs)
            ),
            show_beam_frame_preview=frame_previews.append,
        ),
    )

    FEMMainWindow.highlight_entity(window, kind, "Scope")

    assert rendered[0][0] == (expected_scope_kind,)
    if kind == "node_set":
        assert rendered[0][1]["node_ids"] == (1, 2)
    elif kind == "element_set":
        assert rendered[0][1]["element_ids"] == (10, 20)
        assert len(frame_previews) == 1
    else:
        assert rendered[0][1]["members"] == (member,)
