# -*- coding: utf-8 -*-
"""Freeze and compare the FEM-Python/Abaqus B31 validation oracle.

Run the program-side export with CPython first, then run this script with
Abaqus Python.  Output is deterministic and must live outside ``data/``.

Example::

    python scripts/export_frame_b31_program_snapshot.py \
        --inp data/portal_frame_b31_wind_snow.inp \
        --output artifacts/validation/frame_b31/program_snapshot.json

    abaqus python scripts/compare_frame_b31_odb.py \
        --odb data/frame_b31.odb \
        --program-snapshot artifacts/validation/frame_b31/program_snapshot.json \
        --program-section-point-number 1 \
        --odb-section-point-number 25

Current FEM-Python beam stress records retain their explicit position.
``INTEGRATION_POINT`` observations are compared formally; historical
``SECTION_END`` rows can only enter the marked cross-position diagnostic.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys


SCRIPT_VERSION = "1.1.0"
REPORT_SCHEMA = "fem-python-b31-validation-report-v1"
PROGRAM_SCHEMA = "fem-python-b31-validation-snapshot-v1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(ROOT, "artifacts", "validation", "frame_b31")
DEFAULT_GROUPS = (
    ("columns", "COLUMNS"),
    ("arch_ribs", "ARCH_RIBS"),
    ("purlins", "PURLINS"),
    ("side_rails", "SIDE_RAILS"),
    ("roof_bracing", "ROOF_BRACING"),
)
NODE_FIELDS = ("U", "UR", "RF", "RM")
STRESS_COMPONENTS = (
    "S11",
    "S22",
    "S12",
    "S13",
    "Mises",
    "MaxPrincipal",
    "MidPrincipal",
    "MinPrincipal",
)
SECTION_COMPONENTS = ("N", "VY", "VZ", "T", "MY", "MZ")
RECT_POINT_BY_SIGNS = {
    (-1, -1): 1,
    (1, -1): 5,
    (-1, 1): 21,
    (1, 1): 25,
}
PY2 = sys.version_info[0] == 2


def _csv_open(path, mode):
    if PY2:
        return open(path, mode + "b")
    return io.open(path, mode, encoding="utf-8", newline="")


def _json_load(path):
    with io.open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _json_write(path, value):
    with io.open(path, "w", encoding="utf-8", newline="\n") as stream:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        if PY2 and not isinstance(text, unicode):  # noqa: F821
            text = text.decode("utf-8")
        text = u"\n".join(
            line.rstrip(u" \t") for line in text.splitlines()
        )
        stream.write(text + u"\n")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a finite number" % label)
    if math.isnan(result) or math.isinf(result):
        raise ValueError("%s must be a finite number" % label)
    return result


def _vector(values, label):
    result = [_finite(value, label) for value in values]
    if len(result) != 3:
        raise ValueError("%s must contain exactly three values" % label)
    return result


def _position_name(value):
    return str(value).upper().replace(" ", "_")


def _repository_item(repository, requested, label, last=False):
    names = list(repository.keys())
    if requested is not None:
        if requested not in names:
            raise ValueError(
                "ODB %s %r not found; available: %s"
                % (label, requested, ", ".join(names))
            )
        name = requested
    elif last:
        name = names[-1]
    elif len(names) == 1:
        name = names[0]
    else:
        raise ValueError("Select an ODB %s from: %s" % (label, ", ".join(names)))
    return name, repository[name]


def _section_point_number(args):
    coordinate_requested = (
        args.section_point_local_y is not None
        or args.section_point_local_z is not None
    )
    coordinate_number = None
    if coordinate_requested:
        if (
            args.section_point_local_y is None
            or args.section_point_local_z is None
        ):
            raise ValueError(
                "section-point coordinate mapping requires both local y and local z"
            )
        if args.section_type.upper() != "RECT":
            raise ValueError(
                "coordinate section-point mapping is currently frozen only for RECT"
            )
        y = _finite(args.section_point_local_y, "section point local y")
        z = _finite(args.section_point_local_z, "section point local z")
        if abs(y) <= args.coordinate_tolerance or abs(z) <= args.coordinate_tolerance:
            raise ValueError("RECT corner coordinates must have non-zero y and z")
        coordinate_number = RECT_POINT_BY_SIGNS[(1 if y > 0 else -1, 1 if z > 0 else -1)]
    if args.odb_section_point_number is None and coordinate_number is None:
        raise ValueError(
            "select the Abaqus section point explicitly by number or local coordinates"
        )
    if (
        args.odb_section_point_number is not None
        and coordinate_number is not None
        and args.odb_section_point_number != coordinate_number
    ):
        raise ValueError(
            "ODB section point %d conflicts with RECT coordinate mapping point %d"
            % (args.odb_section_point_number, coordinate_number)
        )
    return (
        args.odb_section_point_number
        if args.odb_section_point_number is not None
        else coordinate_number
    )


def _component(field, value, name):
    if name == "Mises":
        try:
            return float(value.mises)
        except (AttributeError, TypeError, ValueError):
            return None
    if name == "MaxPrincipal":
        try:
            return float(value.maxPrincipal)
        except (AttributeError, TypeError, ValueError):
            return None
    labels = list(field.componentLabels)
    if name not in labels:
        return None
    return float(value.data[labels.index(name)])


def _element_labels(element_set):
    labels = []
    for item in element_set.elements:
        if hasattr(item, "label"):
            labels.append(int(item.label))
        else:
            labels.extend(int(value.label) for value in item)
    return sorted(set(labels))


def _parse_groups(raw_groups):
    if not raw_groups:
        return list(DEFAULT_GROUPS)
    groups = []
    seen = set()
    for raw in raw_groups:
        if "=" not in raw:
            raise ValueError("group must use NAME=ELSET syntax: %r" % raw)
        name, element_set = [part.strip() for part in raw.split("=", 1)]
        if not name or not element_set or name in seen:
            raise ValueError("group names and element sets must be non-empty and unique")
        seen.add(name)
        groups.append((name, element_set))
    return groups


def _extract_nodal_field(frame, instance_name, field_name):
    if field_name not in frame.fieldOutputs:
        return {}, []
    field = frame.fieldOutputs[field_name]
    components = list(field.componentLabels)
    records = {}
    for value in field.values:
        if value.instance.name != instance_name:
            continue
        node_id = int(value.nodeLabel)
        if node_id in records:
            raise ValueError("duplicate ODB %s node %d" % (field_name, node_id))
        records[node_id] = [float(item) for item in value.data]
    return records, components


def _extract_stress_records(field, instance_name, point_number, position):
    records = []
    seen = set()
    for value in field.values:
        if value.instance.name != instance_name:
            continue
        point = getattr(value, "sectionPoint", None)
        if point is None or int(point.number) != point_number:
            continue
        record = {
            "position": position,
            "element_id": int(value.elementLabel),
            "node_id": (
                int(value.nodeLabel)
                if getattr(value, "nodeLabel", None) is not None
                else None
            ),
            "integration_point": (
                int(value.integrationPoint)
                if getattr(value, "integrationPoint", None) is not None
                else None
            ),
            "section_point": {
                "number": int(point.number),
                "description": str(point.description),
            },
            "components": {},
        }
        for component in STRESS_COMPONENTS:
            result = _component(field, value, component)
            if result is not None:
                record["components"][component] = result
        key = (
            record["element_id"],
            record["node_id"],
            record["integration_point"],
        )
        if key in seen:
            raise ValueError(
                "duplicate ODB %s stress identity %r" % (position, key)
            )
        seen.add(key)
        records.append(record)
    return sorted(
        records,
        key=lambda row: (
            row["element_id"],
            -1 if row["node_id"] is None else row["node_id"],
            -1 if row["integration_point"] is None else row["integration_point"],
        ),
    )


def _merge_section_output(frame, instance_name, records):
    by_key = {
        (row["element_id"], row["integration_point"]): row
        for row in records
    }
    availability = {}
    mappings = (
        ("SF", {"SF1": "N", "SF2": "VY", "SF3": "VZ"}),
        ("SM", {"SM1": "T", "SM2": "MY", "SM3": "MZ"}),
    )
    for field_name, mapping in mappings:
        availability[field_name] = field_name in frame.fieldOutputs
        if field_name not in frame.fieldOutputs:
            continue
        field = frame.fieldOutputs[field_name]
        labels = list(field.componentLabels)
        for value in field.values:
            if value.instance.name != instance_name:
                continue
            key = (int(value.elementLabel), int(value.integrationPoint))
            if key not in by_key:
                continue
            for odb_name, canonical in mapping.items():
                if odb_name in labels:
                    by_key[key]["components"][canonical] = float(
                        value.data[labels.index(odb_name)]
                    )
    return availability


def extract_odb(args, point_number):
    from abaqusConstants import ELEMENT_NODAL, ON
    from odbAccess import openOdb

    odb = openOdb(path=args.odb, readOnly=True)
    try:
        step_name, step = _repository_item(odb.steps, args.step, "step", last=True)
        frame_index = args.frame_index
        if frame_index < 0:
            frame_index += len(step.frames)
        if frame_index < 0 or frame_index >= len(step.frames):
            raise ValueError("ODB frame index is out of range")
        frame = step.frames[frame_index]
        instance_name, instance = _repository_item(
            odb.rootAssembly.instances,
            args.instance,
            "instance",
        )
        coordinates = {
            int(node.label): [float(item) for item in node.coordinates]
            for node in instance.nodes
        }
        connectivity = {
            int(element.label): [int(label) for label in element.connectivity]
            for element in instance.elements
        }
        node_fields = {}
        field_components = {}
        for field_name in NODE_FIELDS:
            records, components = _extract_nodal_field(
                frame,
                instance_name,
                field_name,
            )
            node_fields[field_name] = records
            field_components[field_name] = components

        if "S" not in frame.fieldOutputs:
            raise ValueError("ODB frame does not contain S")
        stored_stress = frame.fieldOutputs["S"]
        stored_stress_position = _position_name(
            stored_stress.locations[0].position
        )
        integration_records = _extract_stress_records(
            stored_stress,
            instance_name,
            point_number,
            "INTEGRATION_POINT",
        )
        if not integration_records:
            available_points = sorted(
                set(
                    (
                        int(value.sectionPoint.number),
                        str(value.sectionPoint.description),
                    )
                    for value in stored_stress.values
                    if value.instance.name == instance_name
                    and getattr(value, "sectionPoint", None) is not None
                )
            )
            raise ValueError(
                "no ODB stress at section point %d; available: %s"
                % (point_number, available_points)
            )
        section_availability = _merge_section_output(
            frame,
            instance_name,
            integration_records,
        )
        extrapolated = stored_stress.getSubset(
            position=ELEMENT_NODAL,
            readOnly=ON,
        )
        element_nodal_records = _extract_stress_records(
            extrapolated,
            instance_name,
            point_number,
            "ELEMENT_NODAL",
        )

        groups = {}
        for group_name, set_name in _parse_groups(args.group):
            if set_name not in instance.elementSets:
                raise ValueError(
                    "ODB element set %r for group %r is unavailable"
                    % (set_name, group_name)
                )
            groups[group_name] = _element_labels(instance.elementSets[set_name])
    finally:
        odb.close()

    return {
        "metadata": {
            "path": os.path.abspath(args.odb),
            "sha256": _sha256(args.odb),
            "step": step_name,
            "frame_index": frame_index,
            "instance": instance_name,
        },
        "coordinates": coordinates,
        "connectivity": connectivity,
        "groups": groups,
        "node_fields": node_fields,
        "field_components": field_components,
        "stress": {
            "stored_position": stored_stress_position,
            "integration_point": integration_records,
            "element_nodal": element_nodal_records,
            "section_output_available": section_availability,
        },
    }


def load_program_snapshot(path, args, point_number):
    snapshot = _json_load(path)
    if snapshot.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("unsupported program snapshot schema")
    nodes = {}
    for row in snapshot.get("nodes", []):
        node_id = int(row["node_id"])
        if node_id in nodes:
            raise ValueError("duplicate program node %d" % node_id)
        nodes[node_id] = {
            "coordinates": _vector(row["coordinates"], "node coordinates"),
            "U": _vector(row["U"], "U"),
            "UR": _vector(row["UR"], "UR"),
            "RF": _vector(row["RF"], "RF"),
            "RM": _vector(row["RM"], "RM"),
        }
    section_results = []
    seen = set()
    for row in snapshot.get("section_results", []):
        section_point = row.get("section_point", {})
        if int(section_point.get("number", -1)) != args.program_section_point_number:
            continue
        section_type = row.get("section", {}).get("type")
        local_y = _finite(section_point.get("local_y"), "program section local y")
        local_z = _finite(section_point.get("local_z"), "program section local z")
        if args.section_point_local_y is not None:
            if abs(local_y - args.section_point_local_y) > args.coordinate_tolerance:
                raise ValueError("program section-point local y does not match selection")
            if abs(local_z - args.section_point_local_z) > args.coordinate_tolerance:
                raise ValueError("program section-point local z does not match selection")
        if (
            args.program_section_point_number == 1
            and section_type == "RECT"
            and local_y > 0.0
            and local_z > 0.0
            and point_number != 25
        ):
            raise ValueError(
                "frozen RECT mapping requires program point 1 (+y,+z) to use Abaqus point 25"
            )
        position = str(row["position"]).upper()
        key = (
            position,
            int(row["element_id"]),
            row.get("integration_point"),
            row.get("local_node"),
        )
        if key in seen:
            raise ValueError("duplicate program section result %r" % (key,))
        seen.add(key)
        normalized = dict(row)
        normalized["position"] = position
        normalized["element_id"] = int(row["element_id"])
        normalized["node_id"] = (
            None if row.get("node_id") is None else int(row["node_id"])
        )
        normalized["components"] = {
            str(name): _finite(value, "program section component")
            for name, value in row.get("components", {}).items()
        }
        section_results.append(normalized)
    return snapshot, nodes, sorted(
        section_results,
        key=lambda row: (
            row["position"],
            row["element_id"],
            -1 if row.get("local_node") is None else row["local_node"],
        ),
    )


def _coordinate_checks(program_nodes, oracle, tolerance):
    errors = []
    for node_id in sorted(set(program_nodes).intersection(oracle["coordinates"])):
        errors.append(
            max(
                abs(left - right)
                for left, right in zip(
                    program_nodes[node_id]["coordinates"],
                    oracle["coordinates"][node_id],
                )
            )
        )
    maximum = max(errors) if errors else None
    if maximum is not None and maximum > tolerance:
        raise ValueError(
            "maximum node coordinate error %.16g exceeds %.16g"
            % (maximum, tolerance)
        )
    return maximum


def _relative_l2(program, reference):
    numerator = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(program, reference))
    )
    denominator = math.sqrt(sum(value * value for value in reference))
    return numerator / denominator if denominator > 0.0 else None


def _node_metrics(program_nodes, odb_nodes, field_name, node_ids):
    matched = sorted(
        set(node_ids).intersection(program_nodes).intersection(odb_nodes)
    )
    if not matched:
        return None
    program_vectors = [program_nodes[node_id][field_name] for node_id in matched]
    reference_vectors = [odb_nodes[node_id] for node_id in matched]
    flat_program = [value for vector in program_vectors for value in vector]
    flat_reference = [value for vector in reference_vectors for value in vector]
    maximum_reference_norm = max(
        math.sqrt(sum(value * value for value in vector))
        for vector in reference_vectors
    )
    significant_errors = []
    for program, reference in zip(program_vectors, reference_vectors):
        reference_norm = math.sqrt(sum(value * value for value in reference))
        if reference_norm >= 0.001 * maximum_reference_norm and reference_norm > 0.0:
            difference_norm = math.sqrt(
                sum((left - right) ** 2 for left, right in zip(program, reference))
            )
            significant_errors.append(difference_norm / reference_norm)
    component_metrics = {}
    for index in range(3):
        program_component = [vector[index] for vector in program_vectors]
        reference_component = [vector[index] for vector in reference_vectors]
        component_metrics[str(index + 1)] = {
            "relative_l2": _relative_l2(program_component, reference_component),
            "max_absolute_error": max(
                abs(left - right)
                for left, right in zip(program_component, reference_component)
            ),
        }
    return {
        "matched_nodes": len(matched),
        "vector_relative_l2": _relative_l2(flat_program, flat_reference),
        "significant_node_count": len(significant_errors),
        "significant_node_mean_vector_relative_error": (
            sum(significant_errors) / len(significant_errors)
            if significant_errors
            else None
        ),
        "significant_node_max_vector_relative_error": (
            max(significant_errors) if significant_errors else None
        ),
        "max_absolute_error": max(
            abs(left - right)
            for left, right in zip(flat_program, flat_reference)
        ),
        "components": component_metrics,
    }


def _node_details(program_nodes, oracle):
    details = []
    worst = []
    all_program = set(program_nodes)
    for field_name in NODE_FIELDS:
        odb_nodes = oracle["node_fields"].get(field_name, {})
        all_odb = set(odb_nodes)
        for node_id in sorted(all_program.union(all_odb)):
            if node_id not in program_nodes:
                status = "missing_in_program"
            elif node_id not in odb_nodes:
                status = "missing_in_odb"
            else:
                status = "matched"
            vector_error = 0.0
            for index in range(3):
                program_value = (
                    program_nodes[node_id][field_name][index]
                    if node_id in program_nodes
                    else None
                )
                abaqus_value = (
                    odb_nodes[node_id][index] if node_id in odb_nodes else None
                )
                difference = (
                    program_value - abaqus_value
                    if status == "matched"
                    else None
                )
                if difference is not None:
                    vector_error += difference * difference
                details.append(
                    {
                        "comparison": "nodal",
                        "field": field_name,
                        "component": "%s%d" % (field_name, index + 1),
                        "position_program": "NODE",
                        "position_abaqus": "NODE",
                        "status": status,
                        "node_id": node_id,
                        "element_id": None,
                        "local_node": None,
                        "integration_point": None,
                        "program_value": program_value,
                        "abaqus_value": abaqus_value,
                        "difference": difference,
                        "absolute_difference": (
                            abs(difference) if difference is not None else None
                        ),
                    }
                )
            if status == "matched":
                worst.append(
                    {
                        "field": field_name,
                        "node_id": node_id,
                        "vector_absolute_error": math.sqrt(vector_error),
                    }
                )
    return details, sorted(
        worst,
        key=lambda row: (-row["vector_absolute_error"], row["field"], row["node_id"]),
    )


def _stress_comparison(program_records, odb_records, position_pair, formal):
    program_position, odb_position = position_pair
    selected_program = [
        row for row in program_records if row["position"] == program_position
    ]
    if program_position == "INTEGRATION_POINT":
        program_lookup = {
            (row["element_id"], int(row["integration_point"])): row
            for row in selected_program
        }
        odb_lookup = {
            (row["element_id"], int(row["integration_point"])): row
            for row in odb_records
        }
    else:
        program_lookup = {
            (row["element_id"], row["node_id"]): row
            for row in selected_program
        }
        odb_lookup = {
            (row["element_id"], row["node_id"]): row
            for row in odb_records
        }
    details = []
    metrics = {}
    worst = []
    keys = sorted(set(program_lookup).union(odb_lookup))
    components = STRESS_COMPONENTS + SECTION_COMPONENTS
    for component in components:
        pairs = []
        reference_scale = max(
            [
                abs(row["components"][component])
                for row in odb_lookup.values()
                if component in row["components"]
            ]
            or [0.0]
        )
        for key in keys:
            program = program_lookup.get(key)
            odb = odb_lookup.get(key)
            if program is None:
                status = "missing_in_program"
            elif odb is None:
                status = "missing_in_odb"
            elif component not in program["components"]:
                status = "component_unavailable_in_program"
            elif component not in odb["components"]:
                status = "component_unavailable_in_odb"
            else:
                status = "matched"
            program_value = (
                program["components"].get(component) if program is not None else None
            )
            abaqus_value = odb["components"].get(component) if odb is not None else None
            difference = (
                program_value - abaqus_value if status == "matched" else None
            )
            element_id = key[0]
            details.append(
                {
                    "comparison": "formal" if formal else "diagnostic_cross_position",
                    "field": "S" if component in STRESS_COMPONENTS else "SECTION_FORCE",
                    "component": component,
                    "position_program": program_position,
                    "position_abaqus": odb_position,
                    "status": status,
                    "node_id": (
                        program.get("node_id")
                        if program is not None
                        else (odb.get("node_id") if odb is not None else None)
                    ),
                    "element_id": element_id,
                    "local_node": program.get("local_node") if program else None,
                    "integration_point": (
                        program.get("integration_point")
                        if program is not None
                        else (odb.get("integration_point") if odb else None)
                    ),
                    "program_value": program_value,
                    "abaqus_value": abaqus_value,
                    "difference": difference,
                    "absolute_difference": (
                        abs(difference) if difference is not None else None
                    ),
                }
            )
            if difference is not None:
                pairs.append((program_value, abaqus_value))
                worst.append(
                    {
                        "comparison": "formal" if formal else "diagnostic_cross_position",
                        "component": component,
                        "element_id": element_id,
                        "absolute_error": abs(difference),
                    }
                )
        significant = [
            pair
            for pair in pairs
            if abs(pair[1]) >= max(1.0e5, 0.001 * reference_scale)
        ] if component in STRESS_COMPONENTS else pairs
        metrics[component] = {
            "matched_rows": len(pairs),
            "mae": (
                sum(abs(left - right) for left, right in pairs) / len(pairs)
                if pairs
                else None
            ),
            "max_absolute_error": (
                max(abs(left - right) for left, right in pairs) if pairs else None
            ),
            "significant_relative_l2": (
                _relative_l2(
                    [left for left, _right in significant],
                    [right for _left, right in significant],
                )
                if significant
                else None
            ),
            "significant_rows": len(significant),
        }
    return details, metrics, worst


def _group_nodes(groups, connectivity):
    result = {}
    for name, element_ids in groups.items():
        nodes = set()
        for element_id in element_ids:
            nodes.update(connectivity[element_id])
        result[name] = sorted(nodes)
    return result


def _totals_from_odb(oracle, field_name):
    force = [0.0, 0.0, 0.0]
    moment = [0.0, 0.0, 0.0]
    field = oracle["node_fields"].get(field_name, {})
    rm = oracle["node_fields"].get("RM", {}) if field_name == "RF" else {}
    for node_id, values in field.items():
        coordinates = oracle["coordinates"][node_id]
        for index in range(3):
            force[index] += values[index]
        cross = (
            coordinates[1] * values[2] - coordinates[2] * values[1],
            coordinates[2] * values[0] - coordinates[0] * values[2],
            coordinates[0] * values[1] - coordinates[1] * values[0],
        )
        for index in range(3):
            moment[index] += cross[index] + rm.get(node_id, (0.0, 0.0, 0.0))[index]
    return {"force": force, "moment_about_origin": moment}


def _difference_totals(left, right):
    return {
        name: [a - b for a, b in zip(left[name], right[name])]
        for name in ("force", "moment_about_origin")
    }


def _negated_totals(value):
    return {
        name: [-item for item in value[name]]
        for name in ("force", "moment_about_origin")
    }


def _legacy_displacement_check(path, program_nodes):
    if path is None:
        return None
    rows = {}
    with _csv_open(path, "r") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames:
            if PY2:
                reader.fieldnames[0] = reader.fieldnames[0].lstrip(
                    "\xef\xbb\xbf"
                )
            else:
                reader.fieldnames[0] = reader.fieldnames[0].lstrip("\ufeff")
        for line, row in enumerate(reader, start=2):
            node_id = int(row["node_id"])
            if node_id in rows:
                raise ValueError("duplicate legacy displacement node %d" % node_id)
            rows[node_id] = [float(row[name]) for name in ("U1", "U2", "U3")]
    missing = sorted(set(program_nodes).difference(rows))
    extra = sorted(set(rows).difference(program_nodes))
    differences = [
        abs(left - right)
        for node_id in sorted(set(rows).intersection(program_nodes))
        for left, right in zip(rows[node_id], program_nodes[node_id]["U"])
    ]
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "matched_nodes": len(set(rows).intersection(program_nodes)),
        "missing_nodes": missing,
        "extra_nodes": extra,
        "max_absolute_difference": max(differences) if differences else None,
    }


def _legacy_stress_check(path, program_records):
    if path is None:
        return None
    program_lookup = {
        (
            row["element_id"],
            int(row["local_node"]),
            int(row["section_point"]["number"]),
        ): row
        for row in program_records
        if row["position"] == "SECTION_END"
    }
    rows = {}
    with _csv_open(path, "r") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames:
            if PY2:
                reader.fieldnames[0] = reader.fieldnames[0].lstrip(
                    "\xef\xbb\xbf"
                )
            else:
                reader.fieldnames[0] = reader.fieldnames[0].lstrip("\ufeff")
        for line, row in enumerate(reader, start=2):
            key = (
                int(row["element_id"]),
                int(row["local_node"]),
                int(row["section_point_number"]),
            )
            if key in rows:
                raise ValueError("duplicate legacy stress identity %r" % (key,))
            rows[key] = row
    matched = sorted(set(rows).intersection(program_lookup))
    component_maximum = {}
    for component in ("S11", "Mises", "MaxPrincipal"):
        differences = [
            abs(
                float(rows[key][component])
                - program_lookup[key]["components"][component]
            )
            for key in matched
        ]
        component_maximum[component] = max(differences) if differences else None
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "matched_rows": len(matched),
        "missing_rows": len(set(program_lookup).difference(rows)),
        "extra_rows": len(set(rows).difference(program_lookup)),
        "max_absolute_difference": component_maximum,
    }


def compare_snapshots(program_snapshot, program_nodes, program_sections, oracle, args, point_number):
    program_ids = set(program_nodes)
    odb_ids = set(oracle["coordinates"])
    program_elements = set(row["element_id"] for row in program_sections)
    odb_elements = set(oracle["connectivity"])
    coordinate_error = _coordinate_checks(
        program_nodes,
        oracle,
        args.coordinate_tolerance,
    )
    details, worst_nodes = _node_details(program_nodes, oracle)
    global_metrics = {
        field_name: _node_metrics(
            program_nodes,
            oracle["node_fields"].get(field_name, {}),
            field_name,
            sorted(program_ids.union(odb_ids)),
        )
        for field_name in NODE_FIELDS
    }
    node_groups = _group_nodes(oracle["groups"], oracle["connectivity"])
    group_metrics = {
        group_name: {
            field_name: _node_metrics(
                program_nodes,
                oracle["node_fields"].get(field_name, {}),
                field_name,
                node_ids,
            )
            for field_name in NODE_FIELDS
        }
        for group_name, node_ids in sorted(node_groups.items())
    }

    formal_details, formal_metrics, formal_worst = _stress_comparison(
        program_sections,
        oracle["stress"]["integration_point"],
        ("INTEGRATION_POINT", "INTEGRATION_POINT"),
        True,
    )
    diagnostic_details, diagnostic_metrics, diagnostic_worst = _stress_comparison(
        program_sections,
        oracle["stress"]["element_nodal"],
        ("SECTION_END", "ELEMENT_NODAL"),
        False,
    )
    for group_name, element_ids in sorted(oracle["groups"].items()):
        element_set = set(element_ids)
        _unused, group_formal, _unused_worst = _stress_comparison(
            [
                row
                for row in program_sections
                if row["element_id"] in element_set
            ],
            [
                row
                for row in oracle["stress"]["integration_point"]
                if row["element_id"] in element_set
            ],
            ("INTEGRATION_POINT", "INTEGRATION_POINT"),
            True,
        )
        _unused, group_diagnostic, _unused_worst = _stress_comparison(
            [
                row
                for row in program_sections
                if row["element_id"] in element_set
            ],
            [
                row
                for row in oracle["stress"]["element_nodal"]
                if row["element_id"] in element_set
            ],
            ("SECTION_END", "ELEMENT_NODAL"),
            False,
        )
        group_metrics[group_name]["formal_integration_point"] = group_formal
        group_metrics[group_name][
            "diagnostic_section_end_vs_element_nodal"
        ] = group_diagnostic
    details.extend(formal_details)
    details.extend(diagnostic_details)
    worst_elements = sorted(
        formal_worst + diagnostic_worst,
        key=lambda row: (
            -row["absolute_error"],
            row["comparison"],
            row["component"],
            row["element_id"],
        ),
    )

    abaqus_reaction = _totals_from_odb(oracle, "RF")
    abaqus_inferred_load = _negated_totals(abaqus_reaction)
    program_totals = program_snapshot["totals"]
    ip_counts = {}
    for row in oracle["stress"]["integration_point"]:
        element_id = row["element_id"]
        ip_counts[element_id] = ip_counts.get(element_id, 0) + 1
    ip_elements = set(ip_counts)
    summary = {
        "schema": REPORT_SCHEMA,
        "script": {
            "name": "compare_frame_b31_odb.py",
            "version": SCRIPT_VERSION,
        },
        "parameters": {
            "step": oracle["metadata"]["step"],
            "frame_index": oracle["metadata"]["frame_index"],
            "instance": oracle["metadata"]["instance"],
            "program_section_point_number": args.program_section_point_number,
            "odb_section_point_number": point_number,
            "section_point_local_y": args.section_point_local_y,
            "section_point_local_z": args.section_point_local_z,
            "section_type": args.section_type,
            "coordinate_tolerance": args.coordinate_tolerance,
            "stress_significant_absolute_floor": 1.0e5,
            "significant_fraction": 0.001,
            "units": program_snapshot.get("units", {}),
            "abaqus_unit_interpretation": "consistent units matching the program snapshot",
        },
        "inputs": {
            "odb": oracle["metadata"],
            "program_snapshot": {
                "path": os.path.abspath(args.program_snapshot),
                "sha256": _sha256(args.program_snapshot),
                "source": program_snapshot.get("input"),
                "producer": program_snapshot.get("producer"),
            },
        },
        "identity": {
            "program_nodes": len(program_ids),
            "odb_nodes": len(odb_ids),
            "matched_nodes": len(program_ids.intersection(odb_ids)),
            "missing_program_nodes": sorted(odb_ids.difference(program_ids)),
            "extra_program_nodes": sorted(program_ids.difference(odb_ids)),
            "program_elements": len(program_elements),
            "odb_elements": len(odb_elements),
            "matched_elements": len(program_elements.intersection(odb_elements)),
            "missing_program_elements": sorted(odb_elements.difference(program_elements)),
            "extra_program_elements": sorted(program_elements.difference(odb_elements)),
            "odb_target_integration_point_rows": len(
                oracle["stress"]["integration_point"]
            ),
            "odb_target_integration_point_missing_elements": sorted(
                odb_elements.difference(ip_elements)
            ),
            "odb_target_integration_point_extra_elements": sorted(
                ip_elements.difference(odb_elements)
            ),
            "odb_target_integration_point_duplicate_elements": sorted(
                element_id
                for element_id, count in ip_counts.items()
                if count != 1
            ),
            "odb_element_nodal_extrapolated_rows": len(
                oracle["stress"]["element_nodal"]
            ),
            "odb_nodal_field_rows": {
                field_name: len(oracle["node_fields"].get(field_name, {}))
                for field_name in NODE_FIELDS
            },
            "maximum_coordinate_difference": coordinate_error,
        },
        "position_contract": {
            "odb_stored_stress_position": oracle["stress"]["stored_position"],
            "formal_stress_pair": ["INTEGRATION_POINT", "INTEGRATION_POINT"],
            "formal_program_rows": len(
                [
                    row
                    for row in program_sections
                    if row["position"] == "INTEGRATION_POINT"
                ]
            ),
            "diagnostic_stress_pair": ["SECTION_END", "ELEMENT_NODAL"],
            "diagnostic_is_acceptance_evidence": False,
            "element_nodal_rows_are_extrapolations_not_independent_integration_points": True,
            "section_output_available": oracle["stress"]["section_output_available"],
        },
        "metrics": {
            "global_nodal": global_metrics,
            "groups": group_metrics,
            "formal_integration_point": formal_metrics,
            "diagnostic_section_end_vs_element_nodal": diagnostic_metrics,
        },
        "totals": {
            "program_applied": program_totals["applied"],
            "program_reaction": program_totals["reaction"],
            "abaqus_reaction": abaqus_reaction,
            "abaqus_load_inferred_from_static_reaction_balance": abaqus_inferred_load,
            "program_minus_abaqus_reaction": _difference_totals(
                program_totals["reaction"],
                abaqus_reaction,
            ),
            "program_applied_minus_abaqus_inferred_load": _difference_totals(
                program_totals["applied"],
                abaqus_inferred_load,
            ),
        },
        "legacy_displacement_csv": _legacy_displacement_check(
            args.legacy_displacement_csv,
            program_nodes,
        ),
        "legacy_stress_csv": _legacy_stress_check(
            args.legacy_stress_csv,
            program_sections,
        ),
    }
    return summary, details, worst_nodes, worst_elements


DETAIL_COLUMNS = (
    "comparison",
    "field",
    "component",
    "position_program",
    "position_abaqus",
    "status",
    "node_id",
    "element_id",
    "local_node",
    "integration_point",
    "program_value",
    "abaqus_value",
    "difference",
    "absolute_difference",
)


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.16g" % value
    return value


def _write_csv(path, columns, rows):
    with _csv_open(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row.get(name)) for name in columns})


def _output_directory(path):
    resolved = os.path.abspath(path)
    data = os.path.abspath(os.path.join(ROOT, "data"))
    try:
        inside = os.path.commonpath([resolved, data]) == data
    except AttributeError:
        inside = resolved == data or resolved.startswith(data + os.sep)
    if inside:
        raise ValueError("validation output directory must be outside data/")
    if not os.path.isdir(resolved):
        os.makedirs(resolved)
    return resolved


def _arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare one FEM-Python B31 snapshot with an Abaqus ODB oracle."
    )
    parser.add_argument("--odb", required=True)
    parser.add_argument("--program-snapshot", required=True)
    parser.add_argument("--output-directory", default=DEFAULT_OUTPUT)
    parser.add_argument("--legacy-displacement-csv", default=None)
    parser.add_argument("--legacy-stress-csv", default=None)
    parser.add_argument("--step", default=None, help="Default: last ODB step")
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--instance", default=None)
    parser.add_argument("--program-section-point-number", type=int, required=True)
    parser.add_argument("--odb-section-point-number", type=int, default=None)
    parser.add_argument("--section-point-local-y", type=float, default=None)
    parser.add_argument("--section-point-local-z", type=float, default=None)
    parser.add_argument("--section-type", default="RECT")
    parser.add_argument("--coordinate-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Repeat NAME=ELSET; defaults to the five portal-frame groups",
    )
    parser.add_argument("--expected-nodes", type=int, default=117)
    parser.add_argument("--expected-elements", type=int, default=276)
    parser.add_argument("--worst-count", type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None):
    args = _arguments(argv)
    if args.coordinate_tolerance <= 0.0:
        raise ValueError("coordinate tolerance must be positive")
    point_number = _section_point_number(args)
    output = _output_directory(args.output_directory)
    program_snapshot, program_nodes, program_sections = load_program_snapshot(
        args.program_snapshot,
        args,
        point_number,
    )
    oracle = extract_odb(args, point_number)
    summary, details, worst_nodes, worst_elements = compare_snapshots(
        program_snapshot,
        program_nodes,
        program_sections,
        oracle,
        args,
        point_number,
    )
    identity = summary["identity"]
    if args.expected_nodes > 0 and (
        identity["program_nodes"] != args.expected_nodes
        or identity["odb_nodes"] != args.expected_nodes
        or identity["matched_nodes"] != args.expected_nodes
        or any(
            count != args.expected_nodes
            for count in identity["odb_nodal_field_rows"].values()
        )
    ):
        raise ValueError("node identity count does not match the frozen expectation")
    if args.expected_elements > 0 and (
        identity["odb_elements"] != args.expected_elements
        or identity["matched_elements"] != args.expected_elements
        or identity["odb_target_integration_point_rows"] != args.expected_elements
        or identity["odb_target_integration_point_missing_elements"]
        or identity["odb_target_integration_point_extra_elements"]
        or identity["odb_target_integration_point_duplicate_elements"]
    ):
        raise ValueError("element/integration-point count does not match the frozen expectation")

    oracle_path = os.path.join(output, "abaqus_oracle_snapshot.json")
    summary_path = os.path.join(output, "summary.json")
    detail_path = os.path.join(output, "detail.csv")
    worst_nodes_path = os.path.join(output, "worst_nodes.csv")
    worst_elements_path = os.path.join(output, "worst_elements.csv")
    _json_write(oracle_path, oracle)
    _json_write(summary_path, summary)
    _write_csv(detail_path, DETAIL_COLUMNS, details)
    _write_csv(
        worst_nodes_path,
        ("field", "node_id", "vector_absolute_error"),
        worst_nodes[: args.worst_count],
    )
    _write_csv(
        worst_elements_path,
        ("comparison", "component", "element_id", "absolute_error"),
        worst_elements[: args.worst_count],
    )
    print("B31 oracle report: %s" % summary_path)
    print("Matched nodes: %d" % identity["matched_nodes"])
    print(
        "Target integration points: %d"
        % identity["odb_target_integration_point_rows"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
