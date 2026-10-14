"""Assemble menus and Ribbon pages from shared actions and existing controls."""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenuBar, QWidget

from .widgets.ribbon import RibbonPage, RibbonWidget


_SCOPE_ACTION_NAMES = ("geometry_region", "geometry_regions")


def build_command_menus(
    menu_bar: QMenuBar,
    actions: Mapping[str, QAction],
) -> None:
    """Add the application menus without owning command behavior or state."""

    file_menu = menu_bar.addMenu("File")
    file_menu.setObjectName("menuFile")
    file_menu.addActions([
        actions[name]
        for name in (
            "new_native",
            "open_project",
            "save_project",
            "save_project_as",
            "open",
            "save_result",
            "save_result_as",
            "open_result",
        )
    ])
    file_menu.addSeparator()
    file_menu.addAction(actions["exit"])
    edit_menu = menu_bar.addMenu("Edit")
    edit_menu.setObjectName("menuEdit")
    edit_menu.addActions([
        actions[name]
        for name in (
            "select_point", "select_element", "select_edge", "select_face", "select_body",
        )
    ])
    view_menu = menu_bar.addMenu("View")
    view_menu.setObjectName("menuView")
    view_menu.addActions([actions[name] for name in (
        "fit", "front", "back", "top", "bottom", "left", "right", "iso",
    )])
    view_menu.addSeparator()
    view_menu.addActions([actions[name] for name in ("orthographic", "perspective")])
    view_menu.addSeparator()
    view_menu.addActions([actions[name] for name in ("edges", "nodes", "node_labels", "element_labels", "symbols")])
    view_menu.addSeparator()
    view_menu.addActions([
        actions["symbol_settings"],
        actions["viewport_background"],
        actions["suppressed_part_ghosts"],
    ])
    analysis_menu = menu_bar.addMenu("Analysis")
    analysis_menu.setObjectName("menuAnalysis")
    analysis_menu.addAction(actions["step_info"])
    analysis_menu.addSeparator()
    analysis_menu.addAction(actions["check_model"])
    analysis_menu.addSeparator()
    analysis_menu.addActions([
        actions["submit_job"], actions["resubmit_job"],
        actions["job_manager"],
    ])
    result_menu = menu_bar.addMenu("Results")
    result_menu.setObjectName("menuResult")
    result_menu.addActions([actions[name] for name in ("undeformed", "deformed", "contour")])
    result_menu.addSeparator()
    result_menu.addActions([
        actions[name]
        for name in (
            "overlay",
            "display_settings",
            "scale",
            "contour_options",
            "query",
            "export_csv",
            "export_vtk",
            "screenshot",
        )
    ])
    help_menu = menu_bar.addMenu("Help")
    help_menu.setObjectName("menuHelp")
    help_menu.addAction(actions["about"])


def build_command_ribbon(
    parent: QWidget,
    actions: Mapping[str, QAction],
    *,
    analysis_step_control: QWidget,
    result_field_control: QWidget,
    result_scale_control: QWidget,
) -> RibbonWidget:
    """Arrange existing controls alongside the shared command actions."""

    ribbon = RibbonWidget(parent)
    scope_group = ("Scope", _SCOPE_ACTION_NAMES, ())
    _add_ribbon_page(ribbon, actions, "Project", (
        (
            "File",
            (
                "new_native",
                "delete_model",
                "open_project",
                "save_project",
                "open",
                "save_result",
                "open_result",
                "model_info",
            ),
            ("new_native", "delete_model", "open"),
        ),
        ("Output", ("export_csv", "screenshot"), ()),
    ))
    _add_ribbon_page(ribbon, actions, "Geometry", (
        ("Create", ("geometry_create",), ("geometry_create",)),
        (
            "Features",
            (
                "geometry_extrude",
                "geometry_sweep",
                "geometry_move",
                "geometry_rotate",
                "geometry_fuse",
                "geometry_cut",
            ),
            ("geometry_extrude", "geometry_sweep"),
        ),
        ("Selection", ("select_point", "select_edge", "select_face", "select_body"), ()),
    ))
    _add_ribbon_page(ribbon, actions, "Mesh", (
        (
            "Settings",
            ("mesh_settings", "mesh_local_control", "mesh_controls"),
            ("mesh_settings",),
        ),
        ("Generate", ("mesh_generate", "mesh_clear"), ("mesh_generate",)),
        scope_group,
        ("Inspect", ("mesh_verify", "mesh_statistics"), ()),
    ))
    _add_ribbon_page(ribbon, actions, "Model", (
        ("Definitions", ("material_manager", "section_manager", "section_assign"), ("material_manager",)),
        scope_group,
        (
            "Selection",
            ("select_point", "select_element", "select_edge", "select_face", "select_body"),
            (),
        ),
        ("Display", ("nodes", "edges", "node_labels", "element_labels"), ()),
        ("Symbols", ("symbols", "symbol_settings"), ()),
    ))
    _build_analysis_ribbon_page(ribbon, actions, analysis_step_control)
    _build_result_ribbon_page(
        ribbon,
        actions,
        result_field_control,
        result_scale_control,
    )
    _add_ribbon_page(ribbon, actions, "View", (
        ("Views", ("front", "back", "top", "bottom", "left", "right", "iso"),
         ("front", "back", "top", "bottom", "left", "right", "iso")),
        ("Camera", ("fit", "orthographic", "perspective", "viewport_background"), ()),
        ("Labels", ("nodes", "edges", "node_labels", "element_labels", "symbols"), ()),
    ))
    return ribbon


def _build_analysis_ribbon_page(
    ribbon: RibbonWidget,
    actions: Mapping[str, QAction],
    step_control: QWidget,
) -> None:
    page = ribbon.add_page("Analysis")
    step_group = page.add_group("Step")
    for name in ("step_create", "step_info"):
        step_group.add_action(actions[name])
    step_group.add_widget(step_control)
    step_group.add_action(actions["output_create"], compact=True)
    _add_ribbon_group(page, actions, "Scope", _SCOPE_ACTION_NAMES)
    _add_ribbon_group(page, actions, "Boundary Conditions", ("boundary_create", "load_create"))
    _add_ribbon_group(page, actions, "Jobs", (
        "check_model", "submit_job", "analysis_manager", "job_manager",
    ))


def _build_result_ribbon_page(
    ribbon: RibbonWidget,
    actions: Mapping[str, QAction],
    field_control: QWidget,
    scale_control: QWidget,
) -> None:
    page = ribbon.add_page("Results")
    _add_ribbon_group(page, actions, "Shape", ("undeformed", "deformed"))
    _add_ribbon_group(page, actions, "Contour", ("contour",), ("contour",))
    page.add_group("Primary Variable").add_widget(field_control)
    deformation_group = page.add_group("Deformation")
    deformation_group.add_widget(scale_control)
    deformation_group.add_action(actions["overlay"], large=True)
    _add_ribbon_group(page, actions, "Settings", ("display_settings", "contour_options"))
    _add_ribbon_group(page, actions, "Query and Export", (
        "query", "export_csv", "screenshot",
    ), ("query",))


def _add_ribbon_page(
    ribbon: RibbonWidget,
    actions: Mapping[str, QAction],
    name: str,
    groups: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...],
) -> None:
    page = ribbon.add_page(name)
    for title, action_names, large_names in groups:
        _add_ribbon_group(page, actions, title, action_names, large_names)


def _add_ribbon_group(
    page: RibbonPage,
    actions: Mapping[str, QAction],
    title: str,
    action_names: tuple[str, ...],
    large_names: tuple[str, ...] = (),
) -> None:
    group = page.add_group(title)
    for name in action_names:
        large = name in large_names
        group.add_action(
            actions[name],
            large=large,
            compact=not large and len(action_names) == 1,
        )


__all__ = ["build_command_menus", "build_command_ribbon"]
