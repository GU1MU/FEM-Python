"""PNG 优先、内嵌 SVG 后备的图标管理。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


_ICON_DIR = Path(__file__).with_name("resources") / "icons"

# A few CAE pictograms are intentionally wide and become visually tiny when
# fitted into a square QToolButton with KeepAspectRatio.  Render those assets
# into a controlled square-relative rectangle instead.  Values are
# (width_factor, height_factor, x_offset, y_offset); offsets are relative to
# the final icon size and keep the visual mass centered.
_PNG_RENDER_LAYOUT = {
    "symbols": (1.00, 0.92, 0.00, -0.04),
    "undeformed": (1.00, 0.76, 0.00, 0.08),
    "deformed": (1.00, 0.82, 0.00, 0.00),
    "overlay": (1.00, 0.82, 0.00, -0.07),
    "contour": (0.98, 0.98, 0.00, 0.00),
    "select_element": (1.00, 1.00, 0.00, 0.00),
}

# 保留既有动作键，避免视口相机和 QAction 调用方感知资源命名变化。
_PNG_FILES = {
    "open": "open.png",
    "reload": "reload.png",
    "close": "close.png",
    "model_info": "model_info.png",
    "model": "model.png",
    "check": "check.png",
    "step": "step.png",
    "resubmit": "resubmit.png",
    "job_manager": "job_manager.png",
    "fit": "fit.png",
    "background": "background.png",
    "field": "field.png",
    "scale": "scale.png",
    "export": "export.png",
    "image": "image.png",
    "orthographic": "orthographic.png",
    "perspective": "perspective.png",
    "clear_selection": "clear_selection.png",
    "settings": "settings.png",
    "front": "view_xz.png",
    "back": "view_zx.png",
    "left": "view_yz.png",
    "right": "view_zy.png",
    "top": "view_xy.png",
    "bottom": "view_yx.png",
    "iso": "view_xyz.png",
    "job": "job.png",
    "nodes": "nodes.png",
    "edges": "edges.png",
    "symbols": "symbols.png",
    "undeformed": "undeformed.png",
    "deformed": "deformed.png",
    "overlay": "overlay.png",
    "contour": "contour.png",
    "select_node": "select_node.png",
    "select_element": "select_element.png",
    "query": "query.png",
    "view_more": "view_more.png",
    "inspect": "inspect.png",
    "node_ids": "node_ids.png",
    "element_ids": "element_ids.png",
    "mesh": "mesh.png",
    "node_set": "node_set.png",
    "element_set": "element_set.png",
    "surface": "surface.png",
    "material": "material.png",
    "section": "section.png",
    "boundary": "boundary.png",
    "load": "load.png",
    "output": "output.png",
    "sketch": "sketch.png",
    "extrude": "extrude.png",
    "geometry_move": "geometry_move.png",
    "geometry_rotate": "geometry_rotate.png",
    "boolean_fuse": "boolean_fuse.png",
    "boolean_cut": "boolean_cut.png",
    "feature_edit": "feature_edit.png",
    "feature_undo": "feature_undo.png",
    "geometry_delete": "geometry_delete.png",
    "select_geometry_point": "select_geometry_point.png",
    "select_geometry_edge": "select_geometry_edge.png",
    "select_geometry_face": "select_geometry_face.png",
    "select_geometry_body": "select_geometry_body.png",
    "named_region_create": "named_region_create.png",
    "named_region_manager": "named_region_manager.png",
    "mesh_settings": "mesh_settings.png",
    "mesh_local_size": "mesh_local_size.png",
    "mesh_controls": "mesh_controls.png",
    "mesh_clear": "mesh_clear.png",
    "mesh_verify": "mesh_verify.png",
    "mesh_statistics": "mesh_statistics.png",
    "mesh_quality": "mesh_quality.png",
    "new_model": "new_model.png",
    "open_project": "open_project.png",
    "open_inp": "open_inp.png",
    "save_project": "save_project.png",
    "section_assign": "section_assign.png",
    "step_create": "step_create.png",
    "step_info": "step_info.png",
    "analysis_manager": "analysis_manager.png",
}


_PATHS = {
    "open": "<path d='M3 7h6l2 2h10l-3 10H4z'/><path d='M3 7V4h7l2 3'/>",
    "reload": "<path d='M19 8a8 8 0 1 0 1 7'/><path d='M19 3v5h-5'/>",
    "close": "<path d='M5 4h14v16H5z'/><path d='M8 8l8 8M16 8l-8 8'/>",
    "run": "<path d='M7 4l11 8-11 8z'/>",
    "check": "<path d='M5 12l4 4 10-10'/><circle cx='12' cy='12' r='9'/>",
    "job": "<rect x='3' y='4' width='13' height='16' rx='.5' fill='#e8eef2' stroke='#455a64' stroke-width='1.8'/><path d='M9.5 4v16M3 12h13' stroke='#71808a' stroke-width='1.35'/><path d='M13 11l8 5-8 5z' fill='#2f80c9' stroke='#ffffff' stroke-width='1.2'/>",
    "resubmit": "<path d='M19 8a8 8 0 1 0 1 7'/><path d='M19 3v5h-5'/><path d='M10 9l5 3-5 3z'/>",
    "job_manager": "<rect x='4' y='4' width='16' height='16'/><path d='M8 9h8M8 13h8M8 17h5'/>",
    "model_info": "<path d='M4 7l8-4 8 4v10l-8 4-8-4z'/><path d='M12 10v6M12 7v.1'/>",
    "inspect": "<circle cx='10' cy='10' r='6'/><path d='M14.5 14.5L21 21M10 7v6M10 6v.1'/>",
    "fit": "<path d='M8 3H3v5M16 3h5v5M3 16v5h5M21 16v5h-5'/>",
    "front": "<path d='M5 19H20M5 19V4' stroke='#455a64' stroke-width='2.1'/><path d='M20 19l-4-2.5v5z' fill='#d84a3a' stroke='none'/><path d='M5 4L2.5 8h5z' fill='#e3a21a' stroke='none'/><circle cx='5' cy='19' r='1.6' fill='#455a64' stroke='none'/><text x='16.2' y='15.4' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>X</text><text x='8' y='7.2' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>Z</text>",
    "back": "<path d='M19 19H4M19 19V4' stroke='#455a64' stroke-width='2.1'/><path d='M4 19l4-2.5v5z' fill='#d84a3a' stroke='none'/><path d='M19 4l-2.5 4h5z' fill='#e3a21a' stroke='none'/><circle cx='19' cy='19' r='1.6' fill='#455a64' stroke='none'/><text x='3' y='15.4' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>X</text><text x='12.5' y='7.2' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>Z</text>",
    "left": "<path d='M5 19H20M5 19V4' stroke='#455a64' stroke-width='2.1'/><path d='M20 19l-4-2.5v5z' fill='#3a9d5d' stroke='none'/><path d='M5 4L2.5 8h5z' fill='#e3a21a' stroke='none'/><circle cx='5' cy='19' r='1.6' fill='#455a64' stroke='none'/><text x='16.2' y='15.4' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>Y</text><text x='8' y='7.2' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>Z</text>",
    "right": "<path d='M19 19H4M19 19V4' stroke='#455a64' stroke-width='2.1'/><path d='M4 19l4-2.5v5z' fill='#3a9d5d' stroke='none'/><path d='M19 4l-2.5 4h5z' fill='#e3a21a' stroke='none'/><circle cx='19' cy='19' r='1.6' fill='#455a64' stroke='none'/><text x='3' y='15.4' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>Y</text><text x='12.5' y='7.2' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>Z</text>",
    "top": "<path d='M5 19H20M5 19V4' stroke='#455a64' stroke-width='2.1'/><path d='M20 19l-4-2.5v5z' fill='#d84a3a' stroke='none'/><path d='M5 4L2.5 8h5z' fill='#3a9d5d' stroke='none'/><circle cx='5' cy='19' r='1.6' fill='#455a64' stroke='none'/><text x='16.2' y='15.4' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>X</text><text x='8' y='7.2' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>Y</text>",
    "bottom": "<path d='M19 19H4M19 19V4' stroke='#455a64' stroke-width='2.1'/><path d='M4 19l4-2.5v5z' fill='#d84a3a' stroke='none'/><path d='M19 4l-2.5 4h5z' fill='#3a9d5d' stroke='none'/><circle cx='19' cy='19' r='1.6' fill='#455a64' stroke='none'/><text x='3' y='15.4' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>X</text><text x='12.5' y='7.2' fill='#66717b' stroke='none' font-size='6.2' font-weight='700'>Y</text>",
    "iso": "<path d='M12 13l9 5M12 13l-9 5M12 13V3' stroke='#455a64' stroke-width='2.1'/><path d='M21 18l-4.6.1 2.2-4z' fill='#d84a3a' stroke='none'/><path d='M3 18l2.4-4 2.2 4z' fill='#3a9d5d' stroke='none'/><path d='M12 3L9.5 7h5z' fill='#e3a21a' stroke='none'/><circle cx='12' cy='13' r='1.6' fill='#455a64' stroke='none'/><text x='18.2' y='14.4' fill='#66717b' stroke='none' font-size='5.7' font-weight='700'>X</text><text x='2' y='14.4' fill='#66717b' stroke='none' font-size='5.7' font-weight='700'>Y</text><text x='15' y='6.2' fill='#66717b' stroke='none' font-size='5.7' font-weight='700'>Z</text>",
    "view_more": "<path d='M5 8l7-4 7 4-7 4z'/><circle cx='7' cy='18' r='1'/><circle cx='12' cy='18' r='1'/><circle cx='17' cy='18' r='1'/>",
    "orthographic": "<rect x='4' y='5' width='16' height='14'/><path d='M8 9h8v6H8'/>",
    "perspective": "<path d='M5 5h14l-3 14H8z'/><path d='M9 9h6l-1 6h-4z'/>",
    "background": "<rect x='3' y='4' width='18' height='16'/><path d='M3 14l5-5 4 4 3-3 6 6'/><circle cx='16.5' cy='8' r='2'/>",
    "nodes": "<rect x='4' y='4' width='16' height='16' fill='#e8eef2' stroke='#71808a' stroke-width='1.5'/><path d='M4 12h16M12 4v16' stroke='#9aa7af' stroke-width='1.2'/><g fill='#2f80c9' stroke='#ffffff' stroke-width='1'><circle cx='4' cy='4' r='2.3'/><circle cx='20' cy='4' r='2.3'/><circle cx='4' cy='20' r='2.3'/><circle cx='20' cy='20' r='2.3'/></g>",
    "select_node": "<rect x='3.5' y='3.5' width='14' height='14' fill='#e8eef2' stroke='#71808a' stroke-width='1.5'/><path d='M3.5 10.5h14M10.5 3.5v14' stroke='#9aa7af' stroke-width='1.1'/><circle cx='10.5' cy='10.5' r='3' fill='#e8872d' stroke='#ffffff' stroke-width='1.2'/><path d='M13.5 13l7.5 3-3.4 1.2-1.3 3.5z' fill='#455a64' stroke='#ffffff' stroke-width='.9'/>",
    "select_element": "<rect x='3.5' y='3.5' width='14' height='14' fill='#e8eef2' stroke='#71808a' stroke-width='1.5'/><path d='M3.5 10.5h14M10.5 3.5v14' stroke='#9aa7af' stroke-width='1.1'/><rect x='10.5' y='10.5' width='7' height='7' fill='#e8872d' stroke='#c86c18' stroke-width='1.2'/><path d='M13.5 13l7.5 3-3.4 1.2-1.3 3.5z' fill='#455a64' stroke='#ffffff' stroke-width='.9'/>",
    "clear_selection": "<path d='M4 5h10v10H4zM12 12l8 8M20 12l-8 8'/>",
    "edges": "<rect x='3' y='4' width='18' height='16' fill='#e8eef2' stroke='#455a64' stroke-width='2.2'/><path d='M12 4v16M3 12h18' stroke='#455a64' stroke-width='2'/>",
    "node_ids": "<circle cx='6' cy='7' r='2'/><path d='M11 5h3v6M11 11h4M18 5h2v6'/><path d='M18 8h2'/>",
    "element_ids": "<path d='M3 5h8v8H3z'/><path d='M14 6h3v6M14 12h4M20 6h1v6'/>",
    "symbols": "<rect x='3' y='12' width='18' height='5' fill='#e8eef2' stroke='#607d8b' stroke-width='1.5'/><path d='M12 3v7' stroke='#d9534f' stroke-width='2.5'/><path d='M12 11L8.5 6.5h7z' fill='#d9534f' stroke='none'/><path d='M4 21l3-4 3 4zM14 21l3-4 3 4z' fill='#2e8b57' stroke='#236c44' stroke-width='1.1'/>",
    "settings": "<circle cx='12' cy='12' r='3'/><path d='M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1'/>",
    "step": "<path d='M4 18h5v-5h5V8h6'/><path d='M16 4l4 4-4 4'/>",
    "undeformed": "<path d='M3 13h18' stroke='#455a64' stroke-width='3.2'/><path d='M5 20l3-7 3 7zM13 20l3-7 3 7z' fill='#e8eef2' stroke='#71808a' stroke-width='1.3'/>",
    "deformed": "<path d='M3 8c4.5 0 5 9 9 9s4.5-9 9-9' stroke='#2f80c9' stroke-width='3.2'/><path d='M3 20l3-10 3 10zM15 20l3-10 3 10z' fill='#e8eef2' stroke='#71808a' stroke-width='1.3'/>",
    "overlay": "<path d='M3 8h18' stroke='#8c969d' stroke-width='1.8' stroke-dasharray='2.5 2'/><path d='M3 8c4.5 0 5 9 9 9s4.5-9 9-9' stroke='#2f80c9' stroke-width='3'/><circle cx='3' cy='8' r='1.5' fill='#2f80c9' stroke='none'/><circle cx='21' cy='8' r='1.5' fill='#2f80c9' stroke='none'/>",
    "contour": "<rect x='3' y='4' width='4.5' height='16' fill='#2878b5' stroke='none'/><rect x='7.5' y='4' width='4.5' height='16' fill='#42b7b1' stroke='none'/><rect x='12' y='4' width='4.5' height='16' fill='#f2c14e' stroke='none'/><rect x='16.5' y='4' width='4.5' height='16' fill='#d9534f' stroke='none'/><rect x='3' y='4' width='18' height='16' stroke='#455a64' stroke-width='1.6'/><path d='M3 12h18' stroke='#ffffff' stroke-opacity='.7' stroke-width='1'/>",
    "field": "<path d='M4 5h16v14H4z'/><path d='M8 9h8M8 13h5M8 17h3'/>",
    "scale": "<path d='M4 17L17 4M7 4h10v10'/><path d='M4 11v9h9'/>",
    "query": "<rect x='3' y='4' width='7' height='7' fill='#2878b5' stroke='none'/><rect x='10' y='4' width='7' height='7' fill='#42b7b1' stroke='none'/><rect x='3' y='11' width='7' height='7' fill='#f2c14e' stroke='none'/><rect x='10' y='11' width='7' height='7' fill='#d9534f' stroke='none'/><rect x='3' y='4' width='14' height='14' stroke='#455a64' stroke-width='1.5'/><circle cx='13.5' cy='14.5' r='4' fill='none' stroke='#455a64' stroke-width='2.2'/><path d='M16.5 17.5L21 22' stroke='#455a64' stroke-width='2.2'/><circle cx='13.5' cy='14.5' r='1.5' fill='#e8872d' stroke='#ffffff' stroke-width='.8'/>",
    "export": "<path d='M5 4h9l5 5v11H5z'/><path d='M14 4v5h5M12 11v7M9 15l3 3 3-3'/>",
    "image": "<rect x='3' y='5' width='18' height='14'/><circle cx='8' cy='10' r='2'/><path d='M5 17l5-5 3 3 2-2 4 4'/>",
}


@lru_cache(maxsize=None)
def icon(name: str) -> QIcon:
    """优先加载受控 PNG 资源，缺失时使用内嵌 SVG。"""
    png_name = _PNG_FILES.get(name)
    if png_name is not None:
        source = QPixmap(str(_ICON_DIR / png_name))
        if not source.isNull():
            result = QIcon()
            for size in (18, 20, 24, 32, 40, 48, 64, 80, 96):
                layout = _PNG_RENDER_LAYOUT.get(name)
                if layout is None:
                    target_width = size
                    target_height = size
                    aspect_mode = Qt.AspectRatioMode.KeepAspectRatio
                    x_offset = y_offset = 0.0
                else:
                    width_factor, height_factor, x_offset, y_offset = layout
                    target_width = max(1, round(size * width_factor))
                    target_height = max(1, round(size * height_factor))
                    aspect_mode = Qt.AspectRatioMode.IgnoreAspectRatio
                scaled = source.scaled(
                    target_width,
                    target_height,
                    aspect_mode,
                    Qt.TransformationMode.SmoothTransformation,
                )
                pixmap = QPixmap(size, size)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.drawPixmap(
                    round((size - scaled.width()) / 2 + size * x_offset),
                    round((size - scaled.height()) / 2 + size * y_offset),
                    scaled,
                )
                painter.end()
                result.addPixmap(pixmap)
            return result

    body = _PATHS.get(name)
    if body is None:
        return QIcon()
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#455a64' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'>{body}</svg>"
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    result = QIcon()
    for size in (18, 24, 32):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.end()
        result.addPixmap(pixmap)
    return result
