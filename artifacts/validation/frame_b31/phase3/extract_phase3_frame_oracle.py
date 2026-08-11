from __future__ import print_function

import json

from odbAccess import openOdb


JOB_NAME = "phase3_frame_oracle"
STEP_NAMES = ("GX", "GY", "GZ")


def _section_forces(odb, step_name):
    values = odb.steps[step_name].frames[-1].fieldOutputs["SF"].values
    return dict(
        (int(value.elementLabel), [float(item) for item in value.dataDouble])
        for value in values
    )


def main():
    odb = openOdb(JOB_NAME + ".odb", readOnly=True)
    try:
        by_step = dict(
            (step_name, _section_forces(odb, step_name))
            for step_name in STEP_NAMES
        )
        element_ids = sorted(by_step[STEP_NAMES[0]].keys())
        frames = {}
        for element_id in element_ids:
            sf_rows = [
                [
                    by_step[step_name][element_id][component]
                    for step_name in STEP_NAMES
                ]
                for component in range(3)
            ]
            # B31 reports SF=(axial, shear-n2, shear-n1).  The generic Beam2
            # frame rows are (t, n1, n2), hence the 1,3,2 row projection.
            rotation = [sf_rows[0], sf_rows[2], sf_rows[1]]
            frames[str(element_id)] = rotation
        payload = {
            "abaqus_release": "2023",
            "derivation": (
                "For each fixed-free B31, unit tip loads GX/GY/GZ give "
                "SF=(t dot p, n2 dot p, n1 dot p) at the sole longitudinal "
                "integration point. Reordering SF rows 1,3,2 recovers the "
                "global-to-local (t,n1,n2) matrix."
            ),
            "frames": frames,
            "steps": list(STEP_NAMES),
        }
        with open("abaqus_frame_oracle.json", "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    finally:
        odb.close()


if __name__ == "__main__":
    main()
