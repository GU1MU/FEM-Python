from __future__ import annotations


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


H_SLOT_AREA = 1800.0
H_SLOT_BOUNDARY_LINE_COUNT = 12
EXPECTED_PLATE_PROFILE_ROLES = ("outer", "hole")
