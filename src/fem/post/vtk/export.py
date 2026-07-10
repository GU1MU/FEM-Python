from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from . import cells, fields, polar as polar_fields, writer


def from_result(
    result: Any,
    output_dir: str | Path | None = None,
    name: str | None = None,
    polar: bool = False,
    polar_center: Optional[Sequence[float]] = None,
    overwrite: bool = True,
    threshold: float = 75.0,
) -> None:
    """Export result data to VTK, creating missing CSV files first."""
    mesh = result.model.mesh
    base_name = name or result.name or getattr(result.model, "name", None) or "result"
    output_root = Path(output_dir) if output_dir is not None else Path("results")
    paths = _default_result_paths(output_root, str(base_name))
    stress_paths = _supported_stress_paths(mesh, paths)

    from_csv(
        mesh,
        paths["displacement"],
        stress_paths["element_stress"],
        paths["vtk"],
        stress_paths["nodal_stress"],
        polar=polar,
        polar_center=polar_center,
        U=result.U,
        overwrite=overwrite,
        threshold=threshold,
    )


def from_csv(
    mesh,
    disp_csv_path: str,
    elem_csv_path: Optional[str],
    vtk_path: str,
    nodal_stress_csv_path: Optional[str] = None,
    polar: bool = False,
    polar_center: Optional[Sequence[float]] = None,
    U: Optional[Sequence[float]] = None,
    overwrite: bool = False,
    threshold: float = 75.0,
) -> None:
    """Convert displacement and stress CSV files to VTK."""
    disp_csv_path = Path(disp_csv_path)
    elem_csv_path = Path(elem_csv_path) if elem_csv_path is not None else None
    vtk_path = Path(vtk_path)
    nodal_stress_csv_path = (
        Path(nodal_stress_csv_path)
        if nodal_stress_csv_path is not None
        else None
    )

    if U is not None:
        _export_csvs(
            mesh,
            U,
            disp_csv_path,
            elem_csv_path,
            nodal_stress_csv_path,
            overwrite,
            threshold,
        )

    node_disp = fields.read_displacement(mesh, disp_csv_path)

    if polar:
        if polar_center is None:
            raise ValueError("from_csv: polar_center required when polar=True")
        node_disp = polar_fields.convert_nodal_displacement(mesh, node_disp, polar_center)

    nodal_data = None
    if nodal_stress_csv_path is not None:
        nodal_data = fields.read_nodal_stress_rows(nodal_stress_csv_path)
    if polar and nodal_data is not None:
        nodal_data = polar_fields.convert_nodal_stress_rows(mesh, nodal_data, polar_center)

    field_data = {}
    if elem_csv_path is not None:
        field_data = fields.read_element_stress(elem_csv_path)
    if polar and field_data:
        field_data = polar_fields.convert_element_stress_fields(mesh, field_data, polar_center)

    vtk_path.parent.mkdir(parents=True, exist_ok=True)
    topology = cells.build_result(mesh, () if nodal_data is None else nodal_data.rows)
    if not topology.cells:
        raise ValueError("from_csv: no supported elements")
    nodal_point_fields = (
        {} if nodal_data is None else fields.point_fields(nodal_data, topology.point_rows)
    )

    writer.write(
        mesh=mesh,
        cells=topology.cells,
        cell_types=topology.cell_types,
        elems_for_cell=topology.elems_for_cell,
        node_disp=node_disp,
        field_data=field_data,
        path=vtk_path,
        points=topology.points,
        point_node_ids=topology.point_node_ids,
        nodal_point_fields=nodal_point_fields,
    )


def _default_result_paths(output_dir: Path, name: str) -> dict[str, Path]:
    """Return default result export paths."""
    return {
        "displacement": output_dir / f"{name}_nodal_displacement.csv",
        "element_stress": output_dir / f"{name}_element_stress.csv",
        "nodal_stress": output_dir / f"{name}_nodal_stress.csv",
        "vtk": output_dir / f"{name}.vtk",
    }


def _supported_stress_paths(mesh, paths: dict[str, Path]) -> dict[str, Optional[Path]]:
    """Return default stress paths supported by all mesh element types."""
    from ..stress import dispatch

    try:
        type_keys = dispatch.resolve_type_keys(mesh, None)
        dispatch.stress_group_for_keys(type_keys)
    except ValueError:
        return {"element_stress": None, "nodal_stress": None}

    return {
        "element_stress": (
            paths["element_stress"] if dispatch.element_stress_supported(type_keys) else None
        ),
        "nodal_stress": (
            paths["nodal_stress"] if dispatch.nodal_stress_supported(type_keys) else None
        ),
    }


def _export_csvs(
    mesh,
    U: Sequence[float],
    disp_csv_path: Path,
    elem_csv_path: Optional[Path],
    nodal_stress_csv_path: Optional[Path],
    overwrite: bool,
    threshold: float,
) -> None:
    """Export CSV inputs needed by the VTK writer."""
    from .. import displacement, stress

    disp_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not disp_csv_path.exists():
        displacement.export.nodal(mesh, U, disp_csv_path)

    if elem_csv_path is not None and (overwrite or not elem_csv_path.exists()):
        elem_csv_path.parent.mkdir(parents=True, exist_ok=True)
        stress.export.element(mesh, U, elem_csv_path)

    if nodal_stress_csv_path is not None and (overwrite or not nodal_stress_csv_path.exists()):
        nodal_stress_csv_path.parent.mkdir(parents=True, exist_ok=True)
        stress.export.nodal(mesh, U, nodal_stress_csv_path, threshold=threshold)
