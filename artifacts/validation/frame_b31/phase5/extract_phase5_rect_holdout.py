from __future__ import print_function

import json
import os
import sys

from odbAccess import openOdb


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_phase5_rect_holdout.py INPUT.odb OUTPUT.json")
    odb = openOdb(path=os.path.abspath(sys.argv[1]), readOnly=True)
    try:
        instance_name = sorted(odb.rootAssembly.instances.keys())[0]
        field = odb.steps["T_POS"].frames[-1].fieldOutputs["S"]
        labels = tuple(str(label) for label in field.componentLabels)
        ratios = {10: 1.37, 20: 5.0, 30: 13.0}
        rows = []
        for value in field.values:
            if value.instance.name != instance_name:
                continue
            point = value.sectionPoint
            rows.append(
                {
                    "aspect_ratio": ratios[int(value.elementLabel)],
                    "element_id": int(value.elementLabel),
                    "integration_point": int(value.integrationPoint),
                    "section_point": {
                        "number": int(point.number),
                        "description": str(point.description),
                    },
                    "components": dict(
                        (label, float(value.data[index]))
                        for index, label in enumerate(labels)
                    ),
                }
            )
        rows.sort(key=lambda row: (row["element_id"], row["section_point"]["number"]))
        output = {
            "schema": "fem-python.abaqus-b31-phase5-rect-holdout.v1",
            "abaqus_release": str(odb.jobData.version),
            "component_labels": list(labels),
            "rows": rows,
        }
    finally:
        odb.close()
    with open(os.path.abspath(sys.argv[2]), "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, separators=(",", ": "))
        stream.write("\n")


if __name__ == "__main__":
    main()
