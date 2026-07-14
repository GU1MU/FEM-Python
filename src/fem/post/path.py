import csv
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.mesh import Mesh2D
from ._csv import (
    _NODAL_STRESS_METADATA_FIELDS,
    parse_csv_integer,
    validate_nodal_stress_header,
    validate_nodal_stress_row,
)


def _read_nodal_fields(csv_path: str, *, require_stress_metadata: bool = False):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if require_stress_metadata:
            validate_nodal_stress_header(reader.fieldnames, csv_path)
        elif "node_id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV requires node_id column, got {reader.fieldnames}")

        ignored_fields = {"node_id", "x", "y"}
        if require_stress_metadata:
            ignored_fields.update(_NODAL_STRESS_METADATA_FIELDS)
        field_names = [
            name for name in (reader.fieldnames or []) if name not in ignored_fields
        ]
        data: Dict[int, Dict[str, float]] = {}

        for row in reader:
            if require_stress_metadata:
                node_id, _elem_id, _local_node, _averaged = (
                    validate_nodal_stress_row(row, csv_path, reader.line_num)
                )
            else:
                node_id = parse_csv_integer(
                    row.get("node_id"),
                    csv_path,
                    reader.line_num,
                    "node_id",
                    source="Nodal fields CSV",
                )
            if require_stress_metadata and node_id in data:
                raise ValueError(
                    f"Nodal stress CSV {str(csv_path)} has multiple element-nodal "
                    f"contributions for node_id {node_id}; explicit selection or "
                    "averaging is required"
                )
            values: Dict[str, float] = {}
            for name in field_names:
                val_str = row.get(name, "")
                if val_str == "":
                    continue
                try:
                    values[name] = float(val_str)
                except ValueError:
                    values[name] = 0.0
            data[node_id] = values
        return field_names, data


def extract_path_data(
    mesh: Mesh2D,
    start_id: int,
    end_id: int,
    points: int,
    target: str,
    path: str = "xydata.csv",
    stress_csv_path: Optional[str] = None,
    disp_csv_path: Optional[str] = None,
    normalized: bool = False,
) -> None:
    """Extract path data to CSV."""
    if points < 2:
        raise ValueError("points must be >= 2")
    if stress_csv_path is None and disp_csv_path is None:
        raise ValueError("provide stress_csv_path or disp_csv_path")

    node_lookup = {node.id: node for node in mesh.nodes}
    if start_id not in node_lookup or end_id not in node_lookup:
        raise ValueError("start_id or end_id not in mesh nodes")

    start = np.array([node_lookup[start_id].x, node_lookup[start_id].y], dtype=float)
    end = np.array([node_lookup[end_id].x, node_lookup[end_id].y], dtype=float)
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length == 0.0:
        raise ValueError("start_id and end_id define zero length path")
    direction = vec / length

    disp_fields: List[str] = []
    disp_data: Dict[int, Dict[str, float]] = {}
    if disp_csv_path is not None:
        disp_fields, disp_data = _read_nodal_fields(disp_csv_path)

    stress_fields: List[str] = []
    stress_data: Dict[int, Dict[str, float]] = {}
    if stress_csv_path is not None:
        stress_fields, stress_data = _read_nodal_fields(
            stress_csv_path,
            require_stress_metadata=True,
        )

    source_data = None
    if disp_csv_path is not None and target in disp_fields:
        source_data = disp_data
    elif stress_csv_path is not None and target in stress_fields:
        source_data = stress_data

    if source_data is None:
        raise ValueError(f"target {target} not found in provided CSV files")

    candidates = [
        node for node in mesh.nodes
        if node.id in source_data and target in source_data[node.id]
    ]
    if not candidates:
        raise ValueError("no nodes with target data available")

    selected_ids: List[int] = []
    for i in range(points):
        t = i / (points - 1)
        pos = start + t * vec
        best_id = None
        best_dist = None
        for node in candidates:
            dx = node.x - pos[0]
            dy = node.y - pos[1]
            dist2 = dx * dx + dy * dy
            if best_dist is None or dist2 < best_dist:
                best_dist = dist2
                best_id = node.id
        if best_id is None:
            raise ValueError("failed to select nodes along path")
        selected_ids.append(best_id)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "x", "y", target])

        for node_id in selected_ids:
            node = node_lookup[node_id]
            val = source_data[node_id].get(target)
            if val is None:
                raise ValueError(f"node {node_id} missing target {target}")

            proj = np.dot(np.array([node.x, node.y], dtype=float) - start, direction)
            dist = proj / length if normalized else proj
            writer.writerow([dist, node.x, node.y, val])


def extract_circle_data(
    center: Sequence[float],
    radius: float,
    points: int,
    target: str,
    csv_path: str,
    save_path: str,
) -> None:
    """Extract target data on a circle to CSV."""
    if len(center) != 2:
        raise ValueError("center must have 2 values")
    if points < 2:
        raise ValueError("points must be >= 2")
    if radius <= 0.0:
        raise ValueError("radius must be > 0")

    cx, cy = float(center[0]), float(center[1])

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        if "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError("CSV requires x and y columns")
        if target not in reader.fieldnames:
            raise ValueError(f"target {target} not found in CSV header")

        rows = []
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
            except ValueError:
                continue
            rows.append((x, y, row))

    if not rows:
        raise ValueError("no valid rows in CSV")

    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", target])

        for theta in angles:
            px = cx + radius * np.cos(theta)
            py = cy + radius * np.sin(theta)

            best_row = None
            best_dist = None
            for x, y, row in rows:
                dx = x - px
                dy = y - py
                dist2 = dx * dx + dy * dy
                if best_dist is None or dist2 < best_dist:
                    best_dist = dist2
                    best_row = row

            if best_row is None:
                continue
            writer.writerow([best_row.get("x", ""), best_row.get("y", ""), best_row.get(target, "")])


def extract_nodes_data(
    mesh: Mesh2D,
    node_ids: Sequence[int],
    targets: Sequence[str],
    path: str = "nodes_data.csv",
    stress_csv_path: Optional[str] = None,
    disp_csv_path: Optional[str] = None,
) -> None:
    """Extract nodal target data to CSV."""
    if not node_ids:
        raise ValueError("node_ids is empty")
    if not targets:
        raise ValueError("targets is empty")
    if stress_csv_path is None and disp_csv_path is None:
        raise ValueError("provide stress_csv_path or disp_csv_path")

    node_lookup = {node.id: node for node in mesh.nodes}
    for node_id in node_ids:
        if node_id not in node_lookup:
            raise ValueError(f"node_id {node_id} not in mesh")

    disp_fields: List[str] = []
    disp_data: Dict[int, Dict[str, float]] = {}
    if disp_csv_path is not None:
        disp_fields, disp_data = _read_nodal_fields(disp_csv_path)

    stress_fields: List[str] = []
    stress_data: Dict[int, Dict[str, float]] = {}
    if stress_csv_path is not None:
        stress_fields, stress_data = _read_nodal_fields(
            stress_csv_path,
            require_stress_metadata=True,
        )

    target_sources: Dict[str, Dict[int, Dict[str, float]]] = {}
    for target in targets:
        if disp_csv_path is not None and target in disp_fields:
            target_sources[target] = disp_data
        elif stress_csv_path is not None and target in stress_fields:
            target_sources[target] = stress_data
        else:
            raise ValueError(f"target {target} not found in provided CSV files")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y"] + list(targets))

        for node_id in node_ids:
            node = node_lookup[node_id]
            row = [node_id, node.x, node.y]
            for target in targets:
                source = target_sources[target]
                if node_id not in source or target not in source[node_id]:
                    raise ValueError(f"node {node_id} missing target {target}")
                row.append(source[node_id][target])
            writer.writerow(row)
