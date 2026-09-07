"""Frozen beam reference data migrated from existing Abaqus comparison tests.

Values were already recorded in test_abaqus_b31_phase2.py; this move does not
regenerate them or change their source model or comparison tolerances.
"""

# Abaqus 2023, one 2 m B31 element, integrated 0.2 x 0.1 RECT section,
# E=210 GPa, nu=0.3, n1=(0, 1, 0), node 1 ENCASTRE.  Each static step
# uses *DLOAD, OP=NEW so the two local transverse directions are independent.
ABAQUS_B31_DLOAD_ORACLE = {
    "P1": {
        "record": "BEAM, P1, 500.0",
        "tip": (1, 9.036414849106222e-05, 5, 7.142857066355646e-05),
        "reaction": (1, -1000.0, 5, -1000.0),
    },
    "P2": {
        "record": "BEAM, P2, -300.0",
        "tip": (2, -2.155630209017545e-04, 4, 1.714285754133016e-04),
        "reaction": (2, 600.0, 4, -600.0),
    },
}
