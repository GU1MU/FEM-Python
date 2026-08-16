from __future__ import annotations


MALFORMED_H_SLOT_PAYLOAD = {
    "part_function": "2D 平板（带 H 形槽与四角孔）",
    "geometry": {
        "kind": "planar_profiles",
        "profiles": [
            {
                "kind": "rectangle",
                "x": 0,
                "y": 0,
                "width": 300,
                "height": 100,
                "role": "material",
                "operation": "material",
            },
            {
                "kind": "rectangle",
                "x": 110,
                "y": 20,
                "width": 80,
                "height": 60,
                "role": "hole",
                "operation": "cut",
            },
            {
                "kind": "rectangle",
                "x": 140,
                "y": 40,
                "width": 20,
                "height": 20,
                "role": "material",
                "operation": "material",
            },
            *(
                {
                    "kind": "circle",
                    "center_x": x,
                    "center_y": y,
                    "radius": 1,
                    "role": "hole",
                    "operation": "cut",
                }
                for x, y in ((5, 5), (295, 5), (5, 95), (295, 95))
            ),
        ],
    },
}


EXPECTED_H_CONSTRUCTION = {
    "schema_version": 1,
    "name": "带H形槽和四角孔的平板",
    "plane": "XY",
    "nodes": [
        {
            "id": "plate",
            "kind": "rectangle",
            "x": 0,
            "y": 0,
            "width": 300,
            "height": 100,
        },
        {
            "id": "h_left",
            "kind": "rectangle",
            "x": 110,
            "y": 20,
            "width": 10,
            "height": 60,
        },
        {
            "id": "h_cross",
            "kind": "rectangle",
            "x": 110,
            "y": 45,
            "width": 80,
            "height": 10,
        },
        {
            "id": "h_right",
            "kind": "rectangle",
            "x": 180,
            "y": 20,
            "width": 10,
            "height": 60,
        },
        {
            "id": "h_slot",
            "kind": "union",
            "operands": ["h_left", "h_cross", "h_right"],
        },
        {
            "id": "corner_hole",
            "kind": "circle",
            "center_x": 5,
            "center_y": 5,
            "radius": 1,
        },
        {
            "id": "holes",
            "kind": "rectangular_pattern",
            "seed": "corner_hole",
            "count_x": 2,
            "count_y": 2,
            "spacing_x": 290,
            "spacing_y": 90,
        },
        {
            "id": "all_cuts",
            "kind": "union",
            "operands": ["h_slot", "holes"],
        },
        {
            "id": "result",
            "kind": "difference",
            "base": "plate",
            "subtract": ["all_cuts"],
        },
    ],
    "result_node_id": "result",
}


LEGACY_PROFILE_SCHEMA_HASHES = {
    "planar_profiles": (
        "53e31346e57a45b391742d03070a142b04008bef72357490187f0faa0343ae46"
    ),
    "extruded_profiles": (
        "184d46cc2e0367f303693dc209bd1c965ef3841c867ad2af010a9cd3224d5237"
    ),
    "path_swept_profile": (
        "b735fe947d44ea27bb17075c66f19485300356611d278e362e03e01f69bc270f"
    ),
}


H_SLOT_AREA = 1800.0
H_SLOT_BOUNDARY_LINE_COUNT = 12
EXPECTED_PLATE_PROFILE_ROLES = ("outer", "hole")
