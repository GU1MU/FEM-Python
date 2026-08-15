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
        "2038cf1481e79ebb2889b0d41ea98bb1d42c11d766a5d85d6b0ee98dc15ff2d8"
    ),
    "extruded_profiles": (
        "feb2ab9b3565abfcc8b98f651aa47da22cd24c5016daa98822dab77bf935f42d"
    ),
    "path_swept_profile": (
        "f843c6206bdfe2f58b70d91b114fe101737a19a83723434e0d03d780fbea5b89"
    ),
}


H_SLOT_AREA = 1800.0
H_SLOT_BOUNDARY_LINE_COUNT = 12
EXPECTED_PLATE_PROFILE_ROLES = ("outer", "hole")
