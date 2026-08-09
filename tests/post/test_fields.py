from __future__ import annotations

import ast
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys

import pytest

import fem.post as post
from fem.post.fields import (
    ResultRegionKey,
    ResultRegionSignature,
    decode_result_region_key,
    decode_result_region_signature,
    encode_result_region_key,
    make_result_region_signature,
    result_region_sort_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_signature_can_only_be_created_by_factory_or_strict_decoder() -> None:
    with pytest.raises(TypeError, match="must be created"):
        ResultRegionSignature("{}")
    with pytest.raises(TypeError):
        ResultRegionSignature()

    created = make_result_region_signature({"name": "Steel"})
    decoded = decode_result_region_signature(created.canonical_json)

    assert decoded == created


def test_signature_factory_deep_owns_and_canonicalizes_finite_json() -> None:
    payload = {
        "z": [True, None, {"name": "\u94a2"}],
        "a": {"two": 2, "one": 1},
    }

    signature = make_result_region_signature(payload)
    payload["z"][2]["name"] = "changed"
    payload["a"]["zero"] = 0

    assert signature.canonical_json == (
        '{"a":{"one":1,"two":2},"z":[true,null,{"name":"\u94a2"}]}'
    )
    assert "\ufeff" not in signature.canonical_json


def test_signature_is_immutable_hashable_and_deepcopy_stable() -> None:
    first = make_result_region_signature({"b": 2, "a": [1, False]})
    equivalent = make_result_region_signature({"a": [1, False], "b": 2})
    different = make_result_region_signature({"a": [False, 1], "b": 2})

    assert first == equivalent
    assert hash(first) == hash(equivalent)
    assert first != different
    assert len({first, equivalent, different}) == 2
    assert copy(first) is first
    assert deepcopy(first) is first
    with pytest.raises(FrozenInstanceError):
        first.canonical_json = "{}"


@pytest.mark.parametrize(
    "payload",
    (
        (1, 2),
        {1: "non-string key"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": float("-inf")},
        {"value": object()},
        {"value": {1, 2}},
        {"value": b"bytes"},
        {"value": "\ud800"},
    ),
)
def test_signature_factory_rejects_values_outside_strict_json(
    payload: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_result_region_signature(payload)


def test_signature_factory_rejects_list_and_object_cycles() -> None:
    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)
    cyclic_object: dict[str, object] = {}
    cyclic_object["self"] = cyclic_object

    with pytest.raises(ValueError, match="cyclic"):
        make_result_region_signature(cyclic_list)
    with pytest.raises(ValueError, match="cyclic"):
        make_result_region_signature(cyclic_object)


def test_signature_factory_allows_repeated_noncyclic_references() -> None:
    shared = [1, 2]

    signature = make_result_region_signature({"a": shared, "b": shared})

    assert signature.canonical_json == '{"a":[1,2],"b":[1,2]}'


def test_signature_decoder_accepts_only_exact_canonical_utf8_text() -> None:
    canonical = '{"a":[true,1,1.0,null],"b":"\u94a2"}'

    signature = decode_result_region_signature(canonical)

    assert signature.canonical_json == canonical


@pytest.mark.parametrize(
    "encoded",
    (
        "",
        '\ufeff{"a":1}',
        ' {"a":1}',
        '{"a":1}\n',
        '{"b":2,"a":1}',
        '{"a":1, "b":2}',
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":-Infinity}',
        '{"a":1.00}',
        '{"a":-0}',
        '{"a":"\\u94a2"}',
        '{"a":',
    ),
)
def test_signature_decoder_rejects_malformed_or_noncanonical_text(
    encoded: str,
) -> None:
    with pytest.raises(ValueError):
        decode_result_region_signature(encoded)


def test_signature_decoder_rejects_nonstring_input() -> None:
    with pytest.raises(TypeError):
        decode_result_region_signature(b'{"a":1}')


def test_region_key_has_unique_reversible_canonical_projection() -> None:
    material = make_result_region_signature(
        {
            "name": "Steel",
            "properties": {"E": 210000.0},
        }
    )
    section = make_result_region_signature(["solid", {"thickness": 1.0}])
    region = ResultRegionKey(material, section)

    encoded = encode_result_region_key(region)
    decoded = decode_result_region_key(encoded)

    assert encoded == (
        '{"material":{"name":"Steel","properties":{"E":210000.0}},'
        '"section":["solid",{"thickness":1.0}]}'
    )
    assert decoded == region
    assert hash(decoded) == hash(region)
    assert encode_result_region_key(decoded) == encoded
    with pytest.raises(FrozenInstanceError):
        region.material_signature = section


def test_region_key_requires_exact_signature_and_key_types() -> None:
    signature = make_result_region_signature({})

    with pytest.raises(TypeError, match="material_signature"):
        ResultRegionKey("{}", signature)
    with pytest.raises(TypeError, match="section_signature"):
        ResultRegionKey(signature, "{}")
    with pytest.raises(TypeError, match="region_key"):
        encode_result_region_key(object())
    with pytest.raises(TypeError, match="region_key"):
        result_region_sort_key(object())


@pytest.mark.parametrize(
    "encoded",
    (
        "[]",
        '{"material":{}}',
        '{"section":{}}',
        '{"material":{},"other":0,"section":{}}',
        '{"section":{},"material":{}}',
        '{"material": {}, "section":{}}',
        '{"material":{},"material":[],"section":{}}',
        '{"material":{"x":NaN},"section":{}}',
    ),
)
def test_region_key_decoder_rejects_wrong_shape_or_noncanonical_text(
    encoded: str,
) -> None:
    with pytest.raises(ValueError):
        decode_result_region_key(encoded)


def test_region_sort_key_is_material_then_section_canonical_text() -> None:
    material = make_result_region_signature({"material": "Steel"})
    section = make_result_region_signature({"section": "Solid"})
    region = ResultRegionKey(material, section)

    assert result_region_sort_key(region) == (
        material.canonical_json,
        section.canonical_json,
    )


def test_region_encoding_is_stable_in_a_fresh_python_process() -> None:
    material_payload = {"z": [3, 2, 1], "a": {"enabled": True}}
    section_payload = ["solid", {"thickness": 1.25}]
    expected = encode_result_region_key(
        ResultRegionKey(
            make_result_region_signature(material_payload),
            make_result_region_signature(section_payload),
        )
    )
    script = (
        "from fem.post.fields import "
        "ResultRegionKey, encode_result_region_key, "
        "make_result_region_signature as make;"
        "print(encode_result_region_key(ResultRegionKey("
        "make({'z':[3,2,1],'a':{'enabled':True}}),"
        "make(['solid',{'thickness':1.25}]))))"
    )
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )

    assert completed.stdout.rstrip("\r\n") == expected


def test_post_exports_neutral_region_identity_api() -> None:
    assert post.ResultRegionSignature is ResultRegionSignature
    assert post.ResultRegionKey is ResultRegionKey
    assert post.make_result_region_signature is make_result_region_signature
    assert post.decode_result_region_key is decode_result_region_key
    assert post.encode_result_region_key is encode_result_region_key
    assert post.result_region_sort_key is result_region_sort_key


def test_neutral_fields_module_has_no_forbidden_layer_imports() -> None:
    source = Path(post.fields.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module is not None
    )
    forbidden = (
        "fem.application",
        "fem.io",
        "fem_gui",
        "fem_agent",
    )

    assert not {
        module
        for module in imported_modules
        if any(
            module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden
        )
    }
