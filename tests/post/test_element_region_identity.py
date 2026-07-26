from __future__ import annotations

import numpy as np
import pytest

import fem.post as post
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.post.fields import (
    MATERIAL_SIGNATURE_KEY,
    SECTION_SIGNATURE_KEY,
    ResultRegionSignature,
    result_region_key_for_element,
)
from fem.post.stress import field as stress_field
from fem.post.stress.field import StressPosition, StressRecovery


def _element(props: dict[object, object]) -> Element2D:
    return Element2D(1, [1, 2, 3], "Tri3", props)


@pytest.mark.parametrize(
    ("props", "material_json", "section_json"),
    (
        (
            {
                "material": "Steel",
                "material_id": 17,
                "E": 210000.0,
                "nu": 0.3,
                "rho": 7.8,
                "section_type": "plane_stress",
                "thickness": 2.0,
                "integration_order": 2,
            },
            '["material","Steel"]',
            '["section","plane_stress",{"integration_order":2,"thickness":2.0}]',
        ),
        (
            {
                "material_id": 17,
                "E": 210000.0,
                "nu": 0.3,
                "area": 4.0,
            },
            '["material_id",17]',
            '["section",null,{"area":4.0}]',
        ),
        (
            {
                "E": 70000.0,
                "nu": 0.33,
                "rho": 2.7,
                "section_type": "solid_circle",
                "radius": 0.02,
            },
            '["effective",[["E",70000.0],["nu",0.33],["rho",2.7]]]',
            '["section","solid_circle",{"radius":0.02}]',
        ),
        (
            {},
            '["effective",[]]',
            '["section",null,{}]',
        ),
    ),
)
def test_element_region_identity_preserves_legacy_fallback_precedence(
    props: dict[object, object],
    material_json: str,
    section_json: str,
) -> None:
    region = result_region_key_for_element(_element(props))

    assert region.material_signature.canonical_json == material_json
    assert region.section_signature.canonical_json == section_json


def test_element_region_identity_honors_explicit_assignment_signatures() -> None:
    material_signature = (
        "material",
        "Aluminum",
        (("E", np.float64(70000.0)), ("nu", np.float32(0.25))),
    )
    section_signature = (
        "section",
        "rectangle",
        {"height": np.int64(4), "width": 2.0},
    )
    region = result_region_key_for_element(
        _element(
            {
                MATERIAL_SIGNATURE_KEY: material_signature,
                SECTION_SIGNATURE_KEY: section_signature,
                "material": "ignored",
                "E": 1.0,
                "section_type": "ignored",
                "thickness": 99.0,
            }
        )
    )

    assert region.material_signature.canonical_json == (
        '["material","Aluminum",[["E",70000.0],["nu",0.25]]]'
    )
    assert region.section_signature.canonical_json == (
        '["section","rectangle",{"height":4,"width":2.0}]'
    )
    assert MATERIAL_SIGNATURE_KEY == "_stress_material_signature"
    assert SECTION_SIGNATURE_KEY == "_stress_section_signature"
    assert stress_field.MATERIAL_SIGNATURE_KEY == MATERIAL_SIGNATURE_KEY
    assert stress_field.SECTION_SIGNATURE_KEY == SECTION_SIGNATURE_KEY


def test_element_region_identity_accepts_canonical_signature_instances() -> None:
    material = post.make_result_region_signature({"material": "Steel"})
    section = post.make_result_region_signature({"section": "Solid"})

    region = result_region_key_for_element(
        _element(
            {
                MATERIAL_SIGNATURE_KEY: material,
                SECTION_SIGNATURE_KEY: section,
            }
        )
    )

    assert type(region.material_signature) is ResultRegionSignature
    assert region.material_signature is material
    assert region.section_signature is section


def test_element_region_identity_deep_owns_explicit_and_derived_properties() -> None:
    material_properties = {"history": [{"E": 10.0}]}
    section_properties = {"dimensions": [1.0, {"height": 2.0}]}
    auto_section = {"tags": ["initial"]}
    explicit_element = _element(
        {
            MATERIAL_SIGNATURE_KEY: ("material", material_properties),
            SECTION_SIGNATURE_KEY: ("section", section_properties),
        }
    )
    derived_element = _element(
        {
            "material": "Steel",
            "section_type": "custom",
            "metadata": auto_section,
        }
    )

    explicit = result_region_key_for_element(explicit_element)
    derived = result_region_key_for_element(derived_element)
    explicit_text = (
        explicit.material_signature.canonical_json,
        explicit.section_signature.canonical_json,
    )
    derived_text = derived.section_signature.canonical_json

    material_properties["history"][0]["E"] = 99.0
    section_properties["dimensions"][1]["height"] = 8.0
    auto_section["tags"].append("changed")
    explicit_element.props[MATERIAL_SIGNATURE_KEY] = ("replacement",)
    derived_element.props["metadata"] = {}

    assert (
        explicit.material_signature.canonical_json,
        explicit.section_signature.canonical_json,
    ) == explicit_text
    assert derived.section_signature.canonical_json == derived_text


@pytest.mark.parametrize(
    "props",
    (
        {MATERIAL_SIGNATURE_KEY: {"value": float("nan")}},
        {SECTION_SIGNATURE_KEY: {"value": float("inf")}},
        {MATERIAL_SIGNATURE_KEY: {1: "non-string key"}},
        {SECTION_SIGNATURE_KEY: object()},
        {"E": 1.0, "custom": b"bytes"},
        {"E": 1.0, 3: "non-string section key"},
    ),
)
def test_element_region_identity_rejects_non_json_signatures(
    props: dict[object, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        result_region_key_for_element(_element(props))


def test_element_region_identity_rejects_cyclic_compatible_signature() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(ValueError, match="cyclic"):
        result_region_key_for_element(
            _element({MATERIAL_SIGNATURE_KEY: cyclic})
        )


def test_continuum_recovery_uses_the_neutral_element_region_identity() -> None:
    nodes = [
        Node2D(1, 0.0, 0.0),
        Node2D(2, 1.0, 0.0),
        Node2D(3, 0.0, 1.0),
        Node2D(4, 2.0, 0.0),
        Node2D(5, 3.0, 0.0),
        Node2D(6, 2.0, 1.0),
    ]
    elements = [
        Element2D(
            11,
            [1, 2, 3],
            "Tri3",
            {
                "material_id": 7,
                "E": 100.0,
                "nu": 0.25,
                "plane_type": "stress",
                "thickness": 1.0,
            },
        ),
        Element2D(
            19,
            [4, 5, 6],
            "Tri3",
            {
                MATERIAL_SIGNATURE_KEY: ("material", "Explicit"),
                SECTION_SIGNATURE_KEY: ("section", "explicit"),
                "E": 80.0,
                "nu": 0.2,
                "plane_type": "stress",
                "thickness": 2.0,
            },
        ),
    ]
    mesh = Mesh2D(nodes, elements)
    expected = {
        element.id: result_region_key_for_element(element)
        for element in elements
    }

    recovered = StressRecovery(mesh, np.zeros(mesh.num_dofs)).collect(
        StressPosition.INTEGRATION_POINT
    )

    assert recovered.records
    assert {
        record.elem_id for record in recovered.records
    } == set(expected)
    for record in recovered.records:
        assert record.region_key == expected[record.elem_id]


def test_post_exports_element_region_identity_helper() -> None:
    assert post.result_region_key_for_element is result_region_key_for_element
