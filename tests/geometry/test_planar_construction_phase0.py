from __future__ import annotations

import hashlib
import json

from fem_agent.authoring_runtime import _LEGACY_PREPARE_GEOMETRY
from tests.helpers.fixtures.planar_construction_phase0 import (
    EXPECTED_H_CONSTRUCTION,
    LEGACY_PROFILE_SCHEMA_HASHES,
    MALFORMED_H_SLOT_PAYLOAD,
)


def _schema_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_phase0_freezes_legacy_profile_schema_hashes() -> None:
    variants = _LEGACY_PREPARE_GEOMETRY.parameters["properties"]["geometry"]["oneOf"]
    actual = {
        variant["properties"]["kind"]["const"]: _schema_hash(variant)
        for variant in variants
        if variant["properties"]["kind"]["const"] in LEGACY_PROFILE_SCHEMA_HASHES
    }

    assert actual == LEGACY_PROFILE_SCHEMA_HASHES


def test_phase0_freezes_malformed_h_payload_and_expected_construction() -> None:
    malformed_profiles = MALFORMED_H_SLOT_PAYLOAD["geometry"]["profiles"]
    assert [profile["operation"] for profile in malformed_profiles[:3]] == [
        "material",
        "cut",
        "material",
    ]
    assert malformed_profiles[1]["width"] == 80
    assert malformed_profiles[2]["width"] == 20

    nodes = EXPECTED_H_CONSTRUCTION["nodes"]
    by_id = {node["id"]: node for node in nodes}
    assert by_id["h_slot"] == {
        "id": "h_slot",
        "kind": "union",
        "operands": ["h_left", "h_cross", "h_right"],
    }
    assert by_id["result"]["kind"] == "difference"
    assert EXPECTED_H_CONSTRUCTION["result_node_id"] == "result"
