from __future__ import print_function

import hashlib
import json
import os

from odbAccess import openOdb


ROOT = os.path.dirname(os.path.abspath(__file__))
ODB = os.path.join(ROOT, "minimal_b31_dload_oracle.odb")
INP = os.path.join(ROOT, "minimal_b31_dload_oracle.inp")
OUTPUT = os.path.join(ROOT, "minimal_b31_dload_oracle.json")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _node_rows(field):
    return dict(
        (int(value.nodeLabel), [float(component) for component in value.data])
        for value in field.values
    )


odb = openOdb(ODB, readOnly=True)
try:
    steps = {}
    for name in ("LOCAL_P1", "LOCAL_P2"):
        frame = odb.steps[name].frames[-1]
        steps[name] = {
            "U": _node_rows(frame.fieldOutputs["U"]),
            "UR": _node_rows(frame.fieldOutputs["UR"]),
            "RF": _node_rows(frame.fieldOutputs["RF"]),
            "RM": _node_rows(frame.fieldOutputs["RM"]),
        }
    payload = {
        "schema": "abaqus-b31-minimal-dload-oracle-v1",
        "producer": "Abaqus 2023",
        "input_sha256": _sha256(INP),
        "steps": steps,
    }
finally:
    odb.close()

with open(OUTPUT, "w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(OUTPUT)
