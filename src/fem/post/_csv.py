from __future__ import annotations

from collections.abc import Iterable, Mapping


_NODAL_STRESS_METADATA_FIELDS = ("elem_id", "local_node", "averaged")
_NODAL_STRESS_REQUIRED_FIELDS = ("node_id", *_NODAL_STRESS_METADATA_FIELDS)


def parse_csv_integer(
    raw_value: object,
    path: object,
    line_no: int,
    field: str,
    *,
    source: str,
) -> int:
    """Parse one integer CSV field with source context."""
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} {str(path)} line {line_no} field {field} "
            f"has raw value {raw_value!r}; expected an integer"
        ) from exc


def parse_csv_number(
    raw_value: object,
    path: object,
    line_no: int,
    field: str,
    *,
    source: str,
) -> float:
    """Parse one numeric CSV field with source context."""
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} {str(path)} line {line_no} field {field} "
            f"has raw value {raw_value!r}; expected a numeric value"
        ) from exc


def validate_nodal_stress_header(
    fieldnames: Iterable[str] | None,
    path: object,
) -> None:
    """Require all current resolved nodal-stress identifying columns."""
    names = tuple(fieldnames or ())
    missing = [name for name in _NODAL_STRESS_REQUIRED_FIELDS if name not in names]
    if missing:
        raise ValueError(
            f"Nodal stress CSV {str(path)!r} requires columns "
            "node_id, elem_id, local_node, and averaged; "
            f"missing {', '.join(missing)}"
        )


def validate_nodal_stress_row(
    row: Mapping[str, object],
    path: object,
    line_no: int,
) -> tuple[int, int | None, int | None, bool]:
    """Validate and return current resolved nodal-stress row metadata."""
    node_id = parse_csv_integer(
        row.get("node_id"),
        path,
        line_no,
        "node_id",
        source="Nodal stress CSV",
    )
    elem_value = row.get("elem_id")
    local_value = row.get("local_node")
    elem_is_empty = elem_value is None or (
        isinstance(elem_value, str) and elem_value.strip() == ""
    )
    local_is_empty = local_value is None or (
        isinstance(local_value, str) and local_value.strip() == ""
    )
    elem_id = (
        parse_csv_integer(
            elem_value,
            path,
            line_no,
            "elem_id",
            source=f"Nodal stress CSV node {node_id}",
        )
        if not elem_is_empty
        else None
    )
    local_node = (
        parse_csv_integer(
            local_value,
            path,
            line_no,
            "local_node",
            source=f"Nodal stress CSV node {node_id}",
        )
        if not local_is_empty
        else None
    )

    averaged_value = row.get("averaged")
    normalized_averaged = (
        averaged_value.strip().casefold() if isinstance(averaged_value, str) else None
    )
    if normalized_averaged not in {"true", "false"}:
        raise ValueError(
            f"Nodal stress CSV {str(path)} line {line_no} field averaged "
            f"has raw value {averaged_value!r}; expected true or false "
            "(case-insensitive)"
        )

    averaged = normalized_averaged == "true"
    if not averaged:
        missing = []
        if elem_id is None:
            missing.append("elem_id")
        if local_node is None:
            missing.append("local_node")
        if missing:
            raise ValueError(
                f"Nodal stress CSV {str(path)} row {line_no} for node {node_id} has "
                f"averaged=false; missing {', '.join(missing)}"
            )
        if local_node < 1:
            raise ValueError(
                f"Nodal stress CSV {str(path)} row {line_no} for node {node_id} has "
                f"non-one-based local_node {local_node}; local_node must be one-based"
            )

    return node_id, elem_id, local_node, averaged
