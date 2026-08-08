from __future__ import annotations

import numpy as np
import pyvista

from fem_gui.visualization.contour_rendering import (
    CONTOUR_EDGE_ALL,
    CONTOUR_EDGE_EXTERIOR,
    CONTOUR_EDGE_FEATURE,
    CONTOUR_EDGE_FREE,
    CONTOUR_EDGE_GEOMETRY,
    CONTOUR_EDGE_NONE,
    CONTOUR_RENDER_FILLED,
    CONTOUR_RENDER_SHADED,
    contour_surface_options,
    extract_contour_edges,
    style_contour_edges,
)


def test_filled_contours_disable_all_surface_lighting() -> None:
    assert contour_surface_options(
        CONTOUR_RENDER_FILLED,
        is_line_mesh=False,
    ) == {
        "lighting": False,
        "smooth_shading": False,
    }


def test_shaded_contours_use_balanced_diffuse_lighting() -> None:
    options = contour_surface_options(
        CONTOUR_RENDER_SHADED,
        is_line_mesh=False,
    )

    assert options["lighting"]
    assert options["smooth_shading"]
    assert options["ambient"] == 0.35
    assert options["diffuse"] == 0.65
    assert options["specular"] == 0.0


def test_visible_edge_modes_extract_distinct_model_boundaries() -> None:
    grid = pyvista.ImageData(dimensions=(4, 4, 4))

    all_edges = extract_contour_edges(grid, CONTOUR_EDGE_ALL)
    exterior_edges = extract_contour_edges(
        grid,
        CONTOUR_EDGE_EXTERIOR,
    )
    feature_edges = extract_contour_edges(
        grid,
        CONTOUR_EDGE_FEATURE,
    )
    free_edges = extract_contour_edges(grid, CONTOUR_EDGE_FREE)
    geometry_edges = extract_contour_edges(grid, CONTOUR_EDGE_GEOMETRY)

    assert all_edges.n_cells > exterior_edges.n_cells
    assert exterior_edges.n_cells > feature_edges.n_cells > 0
    assert free_edges.n_cells == 0
    assert geometry_edges.n_cells == feature_edges.n_cells
    assert extract_contour_edges(grid, CONTOUR_EDGE_NONE) is None


def test_feature_edges_merge_element_nodal_stress_geometry_first() -> None:
    points = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
            (1, 0, 0),
            (2, 0, 0),
            (2, 1, 0),
            (1, 1, 0),
            (1, 0, 1),
            (2, 0, 1),
            (2, 1, 1),
            (1, 1, 1),
        ),
        dtype=float,
    )
    cells = np.asarray(
        (8, *range(8), 8, *range(8, 16)),
        dtype=np.int64,
    )
    disconnected_stress_grid = pyvista.UnstructuredGrid(
        cells,
        np.asarray((12, 12), dtype=np.uint8),
        points,
    )

    feature_edges = extract_contour_edges(
        disconnected_stress_grid,
        CONTOUR_EDGE_FEATURE,
    )

    assert feature_edges.n_cells == 16


def test_geometry_edges_show_perforated_plate_outline_without_mesh_edges() -> None:
    grid = (
        pyvista.ImageData(dimensions=(4, 4, 1))
        .cast_to_unstructured_grid()
        .extract_cells((0, 1, 2, 3, 5, 6, 7, 8))
    )

    geometry_edges = extract_contour_edges(grid, CONTOUR_EDGE_GEOMETRY)
    all_edges = extract_contour_edges(grid, CONTOUR_EDGE_ALL)

    assert geometry_edges.n_cells == 16
    assert all_edges.n_cells == 24


def test_pyvista_accepts_filled_surface_and_feature_edge_options() -> None:
    grid = pyvista.ImageData(dimensions=(3, 3, 3))
    grid.point_data["value"] = np.linspace(0.0, 1.0, grid.n_points)
    edges = extract_contour_edges(grid, CONTOUR_EDGE_FEATURE)
    plotter = pyvista.Plotter(off_screen=True)
    try:
        surface_actor = plotter.add_mesh(
            grid,
            scalars="value",
            **contour_surface_options(
                CONTOUR_RENDER_FILLED,
                is_line_mesh=False,
            ),
        )
        edge_actor = plotter.add_mesh(
            edges,
            color="white",
            lighting=False,
            pickable=False,
        )
    finally:
        plotter.close()

    assert surface_actor is not None
    assert edge_actor is not None


def test_dashed_contour_styles_create_visible_line_gaps() -> None:
    line = pyvista.lines_from_points(
        np.asarray(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
    )

    dashed = style_contour_edges(line, "dashed")
    short_dashed = style_contour_edges(line, "short_dashed")

    assert dashed.n_cells == 2
    assert short_dashed.n_cells == 3
    assert dashed.points[2, 0] > dashed.points[1, 0]
    assert short_dashed.points[2, 0] > short_dashed.points[1, 0]
    assert np.all(np.diff(dashed.lines.reshape((-1, 3))[:, 1:]) == 1)


def test_solid_and_bold_contour_styles_keep_original_edges() -> None:
    line = pyvista.lines_from_points(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
    )

    assert style_contour_edges(line, "solid") is line
    assert style_contour_edges(line, "bold") is line
