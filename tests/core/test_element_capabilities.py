from dataclasses import FrozenInstanceError, replace

import pytest

from fem.elements import (
    ElementCapabilityDescriptor,
    ElementCapabilityLimitation,
    ElementCapabilityRequirement,
    ElementCapabilityStatus,
    get_element_capabilities,
    registered_element_capabilities,
)


# Family expectations: topology, space, DOFs, forces, sections, and loads.
_FAMILY_CAPABILITIES = {
    "plane_continuum": (
        2, 2, ("U1", "U2"), ("Fx", "Fy"), ("solid",),
        ("node", "edge", "body", "gravity"),
    ),
    "solid_continuum": (
        3, 3, ("U1", "U2", "U3"), ("Fx", "Fy", "Fz"), ("solid",),
        ("node", "surface", "body", "gravity"),
    ),
    "truss": (
        1, 3, ("U1", "U2", "U3"), ("Fx", "Fy", "Fz"), ("truss",),
        ("node", "body", "gravity"),
    ),
    "beam": (
        1, 3, ("U1", "U2", "U3", "UR1", "UR2", "UR3"),
        ("Fx", "Fy", "Fz", "Mx", "My", "Mz"), ("beam",),
        ("node", "line", "body", "gravity"),
    ),
}
_BUILTIN_ELEMENTS = {
    "Quad4": ("plane_continuum", 4, ("CPS4", "CPE4")),
    "Quad8": ("plane_continuum", 8, ("CPS8", "CPE8")),
    "Tri3": ("plane_continuum", 3, ("CPS3", "CPE3")),
    "Tri6": ("plane_continuum", 6, ("CPS6", "CPE6")),
    "Hex8": ("solid_continuum", 8, ("C3D8",)),
    "Hex20": ("solid_continuum", 20, ("C3D20",)),
    "Tet4": ("solid_continuum", 4, ("C3D4",)),
    "Tet10": ("solid_continuum", 10, ("C3D10",)),
    "Truss2": ("truss", 2, ()),
    "Beam2": ("beam", 2, ()),
}


def test_builtin_catalog_contains_the_supported_element_types():
    assert sorted(
        item.canonical_type for item in registered_element_capabilities()
    ) == sorted(_BUILTIN_ELEMENTS)


@pytest.mark.parametrize("element_type", _BUILTIN_ELEMENTS)
def test_builtin_capabilities_match_element_family_and_node_count(element_type):
    family, nodes, aliases = _BUILTIN_ELEMENTS[element_type]
    topology, spatial, dofs, forces, sections, loads = _FAMILY_CAPABILITIES[family]

    descriptor = get_element_capabilities(element_type)

    assert descriptor.canonical_type == element_type
    assert descriptor.aliases == aliases
    assert descriptor.family == family
    assert descriptor.topological_dimension == topology
    assert descriptor.spatial_dimension == spatial
    assert descriptor.node_count == nodes
    assert descriptor.dofs_per_node == len(dofs)
    assert descriptor.dof_labels == dofs
    assert descriptor.force_labels == forces
    assert descriptor.section_families == sections
    assert descriptor.load_kinds == loads


def test_beam_orientation_requirements_and_descriptor_are_immutable():
    beam = get_element_capabilities("Beam2")

    assert beam.requirements == (
        ElementCapabilityRequirement(
            "beam.orientation.valid", ("section.rectangle", "load.line.local"),
        ),
        ElementCapabilityRequirement("beam.orientation.explicit", ("load.line.local",)),
    )
    assert beam.status == ElementCapabilityStatus.SUPPORTED
    assert beam.limitations == ()
    with pytest.raises(FrozenInstanceError):
        beam.node_count = 3
    with pytest.raises(FrozenInstanceError):
        beam.requirements[0].code = "changed"


def _descriptor():
    return ElementCapabilityDescriptor(
        canonical_type="ExampleTriangle", aliases=(), family="plane_continuum",
        topological_dimension=2, spatial_dimension=2, node_count=3, dofs_per_node=2,
        section_families=("solid",), load_kinds=("node", "edge", "gravity"),
        dof_labels=("U1", "U2"), force_labels=("Fx", "Fy"),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dof_labels": ("U1",)}, "DOF label count"),
        ({"aliases": ("Alias", "alias")}, "unique case-insensitively"),
        ({"family": "unknown"}, "unsupported element family"),
        ({"topological_dimension": 3}, "topological dimension cannot exceed"),
    ],
    ids=["dof-count", "duplicate-alias", "unknown-family", "dimensions"],
)
def test_descriptor_rejects_inconsistent_capability_metadata(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(_descriptor(), **changes)


def test_descriptor_status_uses_the_most_restrictive_immutable_limitation():
    limited = ElementCapabilityLimitation("approximate", ("load.edge",), "Approximate")
    unavailable = ElementCapabilityLimitation(
        "unsupported", ("load.body",), "Unavailable", ElementCapabilityStatus.UNAVAILABLE,
    )
    descriptor = _descriptor()

    assert descriptor.status == ElementCapabilityStatus.SUPPORTED
    assert replace(descriptor, limitations=(limited,)).status == ElementCapabilityStatus.LIMITED
    assert replace(descriptor, limitations=(limited, unavailable)).status == (
        ElementCapabilityStatus.UNAVAILABLE
    )
    assert replace(descriptor, limitations=(unavailable, limited)).status == (
        ElementCapabilityStatus.UNAVAILABLE
    )
    with pytest.raises(FrozenInstanceError):
        limited.status = ElementCapabilityStatus.UNAVAILABLE
