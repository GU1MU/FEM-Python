from dataclasses import FrozenInstanceError

import pytest

from fem.elements import (
    ElementCapabilityDescriptor,
    ElementCapabilityRequirement,
    ElementCapabilityStatus,
    get_element_capabilities,
    get_element_kernel,
    register_element_kernel,
    registered_element_capabilities,
)


_EXPECTED_CAPABILITIES = {
    "Quad4": {
        "aliases": ("CPS4", "CPE4"),
        "family": "plane_continuum",
        "topology": 2,
        "spatial": 2,
        "nodes": 4,
        "dofs": ("U1", "U2"),
        "forces": ("Fx", "Fy"),
        "sections": ("solid",),
        "loads": ("node", "edge", "gravity"),
    },
    "Quad8": {
        "aliases": ("CPS8", "CPE8"),
        "family": "plane_continuum",
        "topology": 2,
        "spatial": 2,
        "nodes": 8,
        "dofs": ("U1", "U2"),
        "forces": ("Fx", "Fy"),
        "sections": ("solid",),
        "loads": ("node", "edge", "gravity"),
    },
    "Tri6": {
        "aliases": ("CPS6", "CPE6"),
        "family": "plane_continuum",
        "topology": 2,
        "spatial": 2,
        "nodes": 6,
        "dofs": ("U1", "U2"),
        "forces": ("Fx", "Fy"),
        "sections": ("solid",),
        "loads": ("node", "edge", "gravity"),
    },
    "Tri3": {
        "aliases": ("CPS3", "CPE3"),
        "family": "plane_continuum",
        "topology": 2,
        "spatial": 2,
        "nodes": 3,
        "dofs": ("U1", "U2"),
        "forces": ("Fx", "Fy"),
        "sections": ("solid",),
        "loads": ("node", "edge", "gravity"),
    },
    "Hex8": {
        "aliases": ("C3D8",),
        "family": "solid_continuum",
        "topology": 3,
        "spatial": 3,
        "nodes": 8,
        "dofs": ("U1", "U2", "U3"),
        "forces": ("Fx", "Fy", "Fz"),
        "sections": ("solid",),
        "loads": ("node", "surface", "gravity"),
    },
    "Hex20": {
        "aliases": ("C3D20",),
        "family": "solid_continuum",
        "topology": 3,
        "spatial": 3,
        "nodes": 20,
        "dofs": ("U1", "U2", "U3"),
        "forces": ("Fx", "Fy", "Fz"),
        "sections": ("solid",),
        "loads": ("node", "surface", "gravity"),
    },
    "Tet4": {
        "aliases": ("C3D4",),
        "family": "solid_continuum",
        "topology": 3,
        "spatial": 3,
        "nodes": 4,
        "dofs": ("U1", "U2", "U3"),
        "forces": ("Fx", "Fy", "Fz"),
        "sections": ("solid",),
        "loads": ("node", "surface", "gravity"),
    },
    "Tet10": {
        "aliases": ("C3D10",),
        "family": "solid_continuum",
        "topology": 3,
        "spatial": 3,
        "nodes": 10,
        "dofs": ("U1", "U2", "U3"),
        "forces": ("Fx", "Fy", "Fz"),
        "sections": ("solid",),
        "loads": ("node", "surface", "gravity"),
    },
    "Truss2": {
        "aliases": (),
        "family": "truss",
        "topology": 1,
        "spatial": 3,
        "nodes": 2,
        "dofs": ("U1", "U2", "U3"),
        "forces": ("Fx", "Fy", "Fz"),
        "sections": ("truss",),
        "loads": ("node", "gravity"),
    },
    "Beam2": {
        "aliases": (),
        "family": "beam",
        "topology": 1,
        "spatial": 3,
        "nodes": 2,
        "dofs": ("U1", "U2", "U3", "UR1", "UR2", "UR3"),
        "forces": ("Fx", "Fy", "Fz", "Mx", "My", "Mz"),
        "sections": ("beam",),
        "loads": ("node", "line", "gravity"),
    },
}


def test_all_registered_kernels_have_complete_capability_descriptors():
    descriptors = registered_element_capabilities()

    assert len(descriptors) == 10
    assert {item.canonical_type for item in descriptors} == set(
        _EXPECTED_CAPABILITIES
    )
    for descriptor in descriptors:
        expected = _EXPECTED_CAPABILITIES[descriptor.canonical_type]
        assert descriptor.aliases == expected["aliases"]
        assert descriptor.family == expected["family"]
        assert descriptor.topological_dimension == expected["topology"]
        assert descriptor.spatial_dimension == expected["spatial"]
        assert descriptor.node_count == expected["nodes"]
        assert descriptor.dofs_per_node == len(expected["dofs"])
        assert descriptor.dof_labels == expected["dofs"]
        assert descriptor.force_labels == expected["forces"]
        assert descriptor.section_families == expected["sections"]
        assert descriptor.load_kinds == expected["loads"]


@pytest.mark.parametrize(
    ("canonical_type", "alias"),
    [
        ("Tri3", "cps3"),
        ("Tri3", "CPE3"),
        ("Tri6", "cps6"),
        ("Quad4", "CPE4"),
        ("Quad8", "cps8"),
        ("Tet4", "c3d4"),
        ("Tet10", "C3D10"),
        ("Hex8", "c3d8"),
        ("Hex20", "C3D20"),
    ],
)
def test_alias_and_canonical_name_return_the_same_descriptor(
    canonical_type,
    alias,
):
    assert get_element_capabilities(alias) is get_element_capabilities(
        canonical_type
    )


def test_descriptors_statuses_requirements_and_limitations_are_immutable():
    beam = get_element_capabilities("Beam2")
    requirement = beam.requirements[0]

    assert isinstance(requirement, ElementCapabilityRequirement)
    assert beam.status is ElementCapabilityStatus.SUPPORTED
    assert beam.limitations == ()
    assert requirement.code == "beam.orientation.explicit"
    assert requirement.operations == ("section.rectangle", "load.line.local")
    assert get_element_capabilities("Truss2").status is (
        ElementCapabilityStatus.SUPPORTED
    )
    with pytest.raises(FrozenInstanceError):
        beam.node_count = 3
    with pytest.raises(FrozenInstanceError):
        requirement.code = "changed"


@pytest.mark.parametrize("element_type", ["Unknown42", "C3D8R", "C3D4T"])
def test_unknown_or_unsupported_element_capability_queries_fail_closed(
    element_type,
):
    with pytest.raises(NotImplementedError, match="Unsupported element type"):
        get_element_capabilities(element_type)


def test_registration_requires_a_descriptor_without_partial_registration():
    class MissingCapabilitiesKernel:
        canonical_type = "MissingCapabilitiesKernel"
        aliases = ("MissingCapabilitiesAlias",)

    with pytest.raises(ValueError, match="requires a capability descriptor"):
        register_element_kernel(MissingCapabilitiesKernel())

    for name in ("MissingCapabilitiesKernel", "MissingCapabilitiesAlias"):
        with pytest.raises(NotImplementedError, match="Unsupported element type"):
            get_element_kernel(name)
        with pytest.raises(NotImplementedError, match="Unsupported element type"):
            get_element_capabilities(name)


def test_registration_rejects_canonical_identity_mismatch_atomically():
    class IdentityKernel:
        canonical_type = "IdentityKernel"
        aliases = ()

    descriptor = _descriptor(canonical_type="DifferentIdentity")

    with pytest.raises(ValueError, match="canonical type must exactly match"):
        register_element_kernel(IdentityKernel(), descriptor)

    with pytest.raises(NotImplementedError, match="Unsupported element type"):
        get_element_kernel("IdentityKernel")


def test_registration_rejects_alias_mismatch_atomically():
    class AliasKernel:
        canonical_type = "AliasKernel"
        aliases = ("KernelAlias",)

    descriptor = _descriptor(
        canonical_type="AliasKernel",
        aliases=("DescriptorAlias",),
    )

    with pytest.raises(ValueError, match="aliases must exactly match"):
        register_element_kernel(AliasKernel(), descriptor)

    for name in ("AliasKernel", "KernelAlias", "DescriptorAlias"):
        with pytest.raises(NotImplementedError, match="Unsupported element type"):
            get_element_capabilities(name)


def test_descriptor_construction_rejects_missing_or_conflicting_metadata():
    with pytest.raises(ValueError, match="DOF label count"):
        _descriptor(dof_labels=("U1",))
    with pytest.raises(ValueError, match="unique case-insensitively"):
        _descriptor(aliases=("Alias", "alias"))
    with pytest.raises(ValueError, match="unsupported element family"):
        _descriptor(family="unknown")
    with pytest.raises(ValueError, match="topological dimension cannot exceed"):
        _descriptor(topological_dimension=3, spatial_dimension=2)
    with pytest.raises(TypeError, match="must be a tuple"):
        _descriptor(load_kinds=["node"])  # type: ignore[arg-type]


def test_registration_rejects_existing_alias_without_partial_registration():
    class ConflictingKernel:
        canonical_type = "CapabilityConflictKernel"
        aliases = ("c3d8",)

    descriptor = _descriptor(
        canonical_type="CapabilityConflictKernel",
        aliases=("c3d8",),
    )

    with pytest.raises(ValueError, match="already registered"):
        register_element_kernel(ConflictingKernel(), descriptor)

    with pytest.raises(NotImplementedError, match="Unsupported element type"):
        get_element_capabilities("CapabilityConflictKernel")
    assert get_element_capabilities("c3d8").canonical_type == "Hex8"


def _descriptor(
    *,
    canonical_type="TestCapabilityKernel",
    aliases=(),
    family="plane_continuum",
    topological_dimension=2,
    spatial_dimension=2,
    node_count=3,
    dofs_per_node=2,
    section_families=("solid",),
    load_kinds=("node", "edge", "gravity"),
    dof_labels=("U1", "U2"),
    force_labels=("Fx", "Fy"),
    limitations=(),
):
    return ElementCapabilityDescriptor(
        canonical_type=canonical_type,
        aliases=aliases,
        family=family,
        topological_dimension=topological_dimension,
        spatial_dimension=spatial_dimension,
        node_count=node_count,
        dofs_per_node=dofs_per_node,
        section_families=section_families,
        load_kinds=load_kinds,
        dof_labels=dof_labels,
        force_labels=force_labels,
        limitations=limitations,
    )
