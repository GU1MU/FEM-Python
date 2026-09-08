from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QPixmap

from fem_gui.icons import _ICON_DIR, _PNG_FILES, icon


ACTION_ICONS = (
    'front', 'back', 'left', 'right', 'top',
    'bottom', 'iso', 'job', 'nodes', 'edges',
    'symbols', 'undeformed', 'deformed', 'overlay', 'contour',
    'select_node', 'select_element', 'query', 'view_more', 'inspect',
    'node_ids', 'element_ids', 'mesh', 'node_set', 'element_set',
    'surface', 'material', 'section', 'boundary', 'load',
    'output', 'sketch', 'extrude', 'geometry_move', 'geometry_rotate',
    'boolean_fuse', 'boolean_cut', 'feature_edit', 'feature_undo', 'geometry_delete',
    'select_geometry_point', 'select_geometry_edge', 'select_geometry_face', 'select_geometry_body', 'named_region_create',
    'named_region_manager', 'mesh_settings', 'mesh_local_control', 'mesh_controls', 'mesh_clear',
    'mesh_verify', 'mesh_statistics', 'mesh_quality', 'new_model', 'open_project',
    'open_inp', 'save_project', 'model_info', 'section_assign', 'step_create',
    'step_info', 'analysis_manager', 'open_result', 'save_result', 'sweep',
)


def test_action_icons_render_at_toolbar_and_ribbon_sizes():
    for name in ACTION_ICONS:
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


def test_standard_view_icons_are_visually_distinct():
    images = [
        icon(name).pixmap(QSize(32, 32)).toImage()
        for name in ("front", "back", "left", "right", "top", "bottom", "iso")
    ]

    for index, image in enumerate(images):
        assert all(image != previous for previous in images[:index])


def test_png_sources_have_real_transparent_corners():
    for png_name in _PNG_FILES.values():
        source = QPixmap(str(_ICON_DIR / png_name)).toImage()
        assert not source.isNull(), png_name
        assert source.hasAlphaChannel(), png_name
        for point in (
            (0, 0),
            (source.width() - 1, 0),
            (0, source.height() - 1),
            (source.width() - 1, source.height() - 1),
        ):
            assert source.pixelColor(*point).alpha() == 0, png_name
