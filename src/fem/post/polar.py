import csv
from typing import Sequence

from ._csv import (
    parse_csv_number,
    validate_nodal_stress_header,
    validate_nodal_stress_row,
)


def _basis(x: float, y: float, center: Sequence[float]) -> tuple[float, float]:
    """Return cos/sin of the polar basis at ``(x, y)``."""
    dx = x - float(center[0])
    dy = y - float(center[1])
    radius = (dx * dx + dy * dy) ** 0.5
    if radius == 0.0:
        return 1.0, 0.0
    return dx / radius, dy / radius


def _displacement(c: float, s: float, ux: float, uy: float) -> tuple[float, float]:
    """Rotate Cartesian displacement components into the polar basis."""
    return c * ux + s * uy, -s * ux + c * uy


def _stress(
    c: float,
    s: float,
    sig_x: float,
    sig_y: float,
    tau_xy: float,
) -> tuple[float, float, float]:
    """Rotate Cartesian plane-stress components into the polar basis."""
    sig_r = c * c * sig_x + s * s * sig_y + 2.0 * s * c * tau_xy
    sig_t = s * s * sig_x + c * c * sig_y - 2.0 * s * c * tau_xy
    tau_rt = -s * c * sig_x + s * c * sig_y + (c * c - s * s) * tau_xy
    return sig_r, sig_t, tau_rt


def convert_nodal_solution_into_polar_coord(
    csv_path: str,
    center: Sequence[float],
    out_path: str,
) -> None:
    """Convert nodal displacement or current resolved nodal-stress CSV.

    Stress input requires ``elem_id``, ``local_node``, and ``averaged``
    metadata. Output from ``post.stress.export.element()`` is not accepted.
    """
    if len(center) != 2:
        raise ValueError(f"center must have 2 values, got {len(center)}: {center!r}")
    try:
        cx, cy = float(center[0]), float(center[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"center {center!r} must contain 2 numeric values; expected numeric x/y"
        ) from exc

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Polar conversion CSV {csv_path!r} has no header")

        fields = list(reader.fieldnames)
        has_disp = "ux" in fields and "uy" in fields
        has_stress = "sig_x" in fields and "sig_y" in fields and "tau_xy" in fields

        if has_disp and has_stress:
            raise ValueError(
                f"Polar conversion CSV {csv_path!r} has both displacement and "
                f"stress columns: {fields}"
            )
        if not has_disp and not has_stress:
            raise ValueError(
                f"Polar conversion CSV {csv_path!r} requires either ux,uy or "
                f"sig_x,sig_y,tau_xy columns; got {fields}"
            )

        if has_stress:
            validate_nodal_stress_header(fields, csv_path)

        if "x" not in fields or "y" not in fields:
            raise ValueError(
                f"Polar conversion CSV {csv_path!r} requires x and y columns; "
                f"got {fields}"
            )

        if has_disp:
            mapping = {"ux": "ur", "uy": "ut"}
        else:
            mapping = {"sig_x": "sig_r", "sig_y": "sig_t", "tau_xy": "tau_rt"}

        out_fields = [mapping.get(name, name) for name in fields]
        rows = []
        for row in reader:
            if has_stress:
                _node_id, _elem_id, _local_node, _averaged = (
                    validate_nodal_stress_row(row, csv_path, reader.line_num)
                )
            rows.append((reader.line_num, row))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_fields)

        for line_no, row in rows:
            x = parse_csv_number(
                row.get("x"),
                csv_path,
                line_no,
                "x",
                source="Polar conversion CSV",
            )
            y = parse_csv_number(
                row.get("y"),
                csv_path,
                line_no,
                "y",
                source="Polar conversion CSV",
            )

            c, s = _basis(x, y, (cx, cy))

            ux_val = uy_val = None
            sx_val = sy_val = txy_val = None
            if has_disp:
                ux_val = parse_csv_number(
                    row.get("ux"),
                    csv_path,
                    line_no,
                    "ux",
                    source="Polar conversion CSV",
                )
                uy_val = parse_csv_number(
                    row.get("uy"),
                    csv_path,
                    line_no,
                    "uy",
                    source="Polar conversion CSV",
                )

                ur, ut = _displacement(c, s, ux_val, uy_val)

            if has_stress:
                sx_val = parse_csv_number(
                    row.get("sig_x"),
                    csv_path,
                    line_no,
                    "sig_x",
                    source="Polar conversion CSV",
                )
                sy_val = parse_csv_number(
                    row.get("sig_y"),
                    csv_path,
                    line_no,
                    "sig_y",
                    source="Polar conversion CSV",
                )
                txy_val = parse_csv_number(
                    row.get("tau_xy"),
                    csv_path,
                    line_no,
                    "tau_xy",
                    source="Polar conversion CSV",
                )

                sig_r, sig_t, tau_rt = _stress(c, s, sx_val, sy_val, txy_val)

            out_row = []
            for name in fields:
                if has_disp and name == "ux":
                    out_row.append(ur)
                elif has_disp and name == "uy":
                    out_row.append(ut)
                elif has_stress and name == "sig_x":
                    out_row.append(sig_r)
                elif has_stress and name == "sig_y":
                    out_row.append(sig_t)
                elif has_stress and name == "tau_xy":
                    out_row.append(tau_rt)
                else:
                    out_row.append(row.get(name, ""))

            writer.writerow(out_row)
