from __future__ import print_function

import json
import os
import sys

from odbAccess import openOdb


def invariant_value(value, name):
    try:
        return float(getattr(value, name))
    except (AttributeError, TypeError, ValueError):
        return None


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_phase5_shear_oracle.py INPUT.odb OUTPUT.json")
    odb_path = os.path.abspath(sys.argv[1])
    output_path = os.path.abspath(sys.argv[2])
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        instance_name = sorted(odb.rootAssembly.instances.keys())[0]
        section_by_element = {
            10: "RECT",
            20: "CIRC",
            30: "THICK PIPE",
            40: "RECT SQUARE",
            50: "RECT WIDE",
            60: "RECT ASPECT 4",
            70: "RECT ASPECT 10",
            80: "RECT ASPECT 1.25",
            90: "RECT ASPECT 1.5",
            100: "RECT ASPECT 3",
            110: "RECT ASPECT 6",
            120: "RECT ASPECT 20",
        }
        output = {
            "schema": "fem-python.abaqus-b31-phase5-shear-oracle.v1",
            "abaqus_release": str(odb.jobData.version),
            "instance": instance_name,
            "section_by_element": section_by_element,
            "steps": {},
        }
        for step_name in ("VY_POS", "VZ_POS", "T_POS", "T_NEG", "COMBINED"):
            frame = odb.steps[step_name].frames[-1]
            stress = frame.fieldOutputs["S"]
            labels = tuple(str(label) for label in stress.componentLabels)
            rows = []
            for value in stress.values:
                if value.instance.name != instance_name:
                    continue
                point = getattr(value, "sectionPoint", None)
                components = {}
                for index, label in enumerate(labels):
                    components[label] = float(value.data[index])
                for label, attribute in (
                    ("Mises", "mises"),
                    ("MaxPrincipal", "maxPrincipal"),
                    ("MidPrincipal", "midPrincipal"),
                    ("MinPrincipal", "minPrincipal"),
                ):
                    result = invariant_value(value, attribute)
                    if result is not None:
                        components[label] = result
                rows.append(
                    {
                        "element_id": int(value.elementLabel),
                        "section": section_by_element[int(value.elementLabel)],
                        "integration_point": int(value.integrationPoint),
                        "section_point": None if point is None else {
                            "number": int(point.number),
                            "description": str(point.description),
                        },
                        "components": components,
                    }
                )
            rows.sort(
                key=lambda row: (
                    row["element_id"],
                    row["integration_point"],
                    -1 if row["section_point"] is None else row["section_point"]["number"],
                )
            )
            output["steps"][step_name] = {
                "component_labels": list(labels),
                "rows": rows,
            }
    finally:
        odb.close()
    with open(output_path, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, separators=(",", ": "))
        stream.write("\n")


if __name__ == "__main__":
    main()
