from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from fem_gui.icons import _ICON_DIR, _PNG_FILES, icon


FIRST_BATCH = (
    "front", "back", "left", "right", "top", "bottom", "iso",
    "job", "nodes", "edges", "symbols", "undeformed", "deformed",
    "overlay", "contour", "select_node", "select_element", "query",
)

GENERATED_BATCH = (
    "view_more", "inspect", "node_ids", "element_ids", "mesh", "node_set",
    "element_set", "surface", "material", "section", "boundary", "load", "output",
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_first_batch_icons_render_at_toolbar_and_ribbon_sizes():
    _application()
    for name in FIRST_BATCH:
        assert name in _PNG_FILES
        assert (_ICON_DIR / _PNG_FILES[name]).is_file()
        for size in (18, 20, 24, 32, 48):
            pixmap = icon(name).pixmap(QSize(size, size))
            assert not pixmap.isNull(), name
            assert pixmap.size() == QSize(size, size), name
            image = pixmap.toImage()
            assert any(
                image.pixelColor(x, y).alpha() > 0
                for y in range(image.height())
                for x in range(image.width())
            ), name


def test_generated_toolbar_and_tree_icons_are_transparent_pngs():
    _application()
    for name in GENERATED_BATCH:
        assert name in _PNG_FILES
        source = _ICON_DIR / _PNG_FILES[name]
        assert source.is_file(), name
        pixmap = icon(name).pixmap(QSize(20, 20))
        assert not pixmap.isNull(), name
        assert pixmap.toImage().hasAlphaChannel(), name


def test_view_actions_map_to_coordinate_plane_pngs():
    assert _PNG_FILES["top"] == "view_xy.png"
    assert _PNG_FILES["bottom"] == "view_yx.png"
    assert _PNG_FILES["front"] == "view_xz.png"
    assert _PNG_FILES["back"] == "view_zx.png"
    assert _PNG_FILES["left"] == "view_yz.png"
    assert _PNG_FILES["right"] == "view_zy.png"
    assert _PNG_FILES["iso"] == "view_xyz.png"


def test_png_sources_have_real_transparent_corners():
    _application()
    for png_name in _PNG_FILES.values():
        source = QPixmap(str(_ICON_DIR / png_name)).toImage()
        assert source.hasAlphaChannel(), png_name
        for point in (
            (0, 0),
            (source.width() - 1, 0),
            (0, source.height() - 1),
            (source.width() - 1, source.height() - 1),
        ):
            assert source.pixelColor(*point).alpha() == 0, png_name
