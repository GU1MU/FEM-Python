from dataclasses import replace

import pytest

from fem.elements import (
    canonical_element_type,
    get_element_capabilities,
    get_element_kernel,
    register_element_kernel,
    registered_element_capabilities,
)
from fem.elements import registry
from fem.elements.triangle import Tri3Kernel


@pytest.fixture
def isolated_registry(monkeypatch):
    # Registration has no public unregister; isolate writes without changing built-ins.
    for name in ("_KERNELS", "_CAPABILITIES", "_CANONICAL_CAPABILITIES"):
        monkeypatch.setattr(registry, name, getattr(registry, name).copy())


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        ("Truss2", "Truss2"),
        ("Beam2", "Beam2"),
        ("cps3", "Tri3"),
        ("CPE3", "Tri3"),
        ("CPE6", "Tri6"),
        ("cps6", "Tri6"),
        ("cps4", "Quad4"),
        ("CPE4", "Quad4"),
        ("CPE8", "Quad8"),
        ("cps8", "Quad8"),
        ("c3d4", "Tet4"),
        ("C3D10", "Tet10"),
        ("c3d8", "Hex8"),
        ("C3D20", "Hex20"),
    ],
    ids=[
        "truss", "beam", "triangle-stress", "triangle-strain",
        "quadratic-triangle-strain", "quadratic-triangle-stress",
        "quad-stress", "quad-strain", "quadratic-quad-strain",
        "quadratic-quad-stress", "tet", "quadratic-tet", "hex", "quadratic-hex",
    ],
)
def test_aliases_resolve_consistently_through_public_registry_queries(name, canonical):
    assert canonical_element_type(name) == canonical
    assert get_element_kernel(name).canonical_type == canonical
    descriptor = get_element_capabilities(name)
    assert descriptor.canonical_type == canonical
    assert descriptor == get_element_capabilities(canonical)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("Unknown42", "Unsupported element type"),
        ("C3D8R", "reduced integration is not implemented"),
        ("C3D20R", "reduced integration is not implemented"),
        ("C3D4T", "coupled temperature-displacement elements are not implemented"),
    ],
    ids=["unknown", "reduced-hex8", "reduced-hex20", "coupled-tet"],
)
def test_unsupported_types_are_rejected_by_public_registry_queries(name, message):
    for query in (get_element_kernel, get_element_capabilities):
        with pytest.raises(NotImplementedError, match=message):
            query(name)


class CustomTriangle(Tri3Kernel):
    canonical_type = "CustomTriangle"
    aliases = ("CustomAlias",)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "requires a capability descriptor"),
        ("canonical", "canonical type must exactly match"),
        ("aliases", "aliases must exactly match"),
        ("conflict", "already registered"),
    ],
    ids=[
        "missing-descriptor", "canonical-mismatch", "alias-mismatch", "existing-alias",
    ],
)
def test_failed_registration_leaves_new_names_unavailable_and_existing_types_usable(
    isolated_registry, failure, message
):
    original_capabilities = registered_element_capabilities()
    kernel = CustomTriangle()
    if failure == "conflict":
        kernel.aliases = ("CustomAlias", "c3d8")
    descriptor = replace(
        get_element_capabilities("Tri3"),
        canonical_type=kernel.canonical_type,
        aliases=kernel.aliases,
    )
    if failure == "missing":
        descriptor = None
    elif failure == "canonical":
        descriptor = replace(descriptor, canonical_type="DifferentIdentity")
    elif failure == "aliases":
        descriptor = replace(descriptor, aliases=("DifferentAlias",))

    with pytest.raises(ValueError, match=message):
        register_element_kernel(kernel, descriptor)

    new_names = ["CustomTriangle", "CustomAlias"]
    if failure == "canonical":
        new_names.append("DifferentIdentity")
    elif failure == "aliases":
        new_names.append("DifferentAlias")
    for name in new_names:
        for query in (get_element_kernel, get_element_capabilities):
            with pytest.raises(NotImplementedError, match="Unsupported element type"):
                query(name)
    assert registered_element_capabilities() == original_capabilities
    assert canonical_element_type("c3d8") == "Hex8"
    assert get_element_kernel("c3d8").canonical_type == "Hex8"
    assert get_element_capabilities("c3d8").canonical_type == "Hex8"


def test_successful_registration_publishes_canonical_name_alias_and_capabilities(isolated_registry):
    descriptor = replace(
        get_element_capabilities("Tri3"),
        canonical_type="CustomTriangle", aliases=("CustomAlias",),
    )
    register_element_kernel(CustomTriangle(), descriptor)

    for name in ("CustomTriangle", "customalias"):
        assert canonical_element_type(name) == "CustomTriangle"
        assert get_element_kernel(name).canonical_type == "CustomTriangle"
        assert get_element_capabilities(name) == descriptor
    assert [item for item in registered_element_capabilities()
            if item.canonical_type == "CustomTriangle"] == [descriptor]
