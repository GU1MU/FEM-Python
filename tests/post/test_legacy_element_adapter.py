from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from fem.elements import get_element_kernel
from fem.post.stress import dispatch, element
from fem.post.stress._common import (
    PLANE_ELEMENT_HEADER,
    nodal_stress,
    node_lookup,
    validated_u,
)
from fem.post.stress.field import StressPosition, collect_stress
from fem.post.stress.invariants import von_mises_plane
from tests.helpers.mesh_builders import (
    make_mixed_tri3_quad4_mesh,
    make_mixed_tri6_quad8_mesh,
    make_quad8_stiffness_mesh,
    make_unit_hex8_mesh,
)


def _legacy_plane_bytes(
    mesh,
    displacement,
    type_keys: set[str],
    gauss_order: int | None,
) -> bytes:
    """Reproduce the pre-Phase-8 plane element writer as a byte oracle."""
    displacement = validated_u(mesh, displacement)
    lookup = node_lookup(mesh)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(PLANE_ELEMENT_HEADER)
    for elem in mesh.elements:
        type_key = dispatch.type_key_from_name(elem.type)
        if type_key not in type_keys:
            continue
        order = (
            gauss_order
            if gauss_order is not None
            else dispatch.default_gauss_order(type_key)
        )
        values, plane_type, poisson_ratio = nodal_stress(
            mesh,
            elem,
            displacement,
            lookup,
            order,
        )
        for local_node, node_id in enumerate(elem.node_ids, start=1):
            sig_x, sig_y, tau_xy = values[local_node - 1].tolist()
            writer.writerow([
                elem.id,
                node_id,
                local_node,
                sig_x,
                sig_y,
                tau_xy,
                von_mises_plane(
                    sig_x,
                    sig_y,
                    tau_xy,
                    plane_type,
                    poisson_ratio,
                ),
            ])
    return stream.getvalue().encode("utf-8")


def test_single_plane_legacy_csv_delegates_and_preserves_bytes(
    tmp_path,
    monkeypatch,
):
    mesh = make_quad8_stiffness_mesh()
    mesh.elements[0].props["plane_type"] = "strain"
    displacement = np.linspace(-0.04, 0.11, mesh.num_dofs)
    target = tmp_path / "quad8-element.csv"
    calls = []
    original = element.collect_stress

    def spy_collect_stress(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(element, "collect_stress", spy_collect_stress)

    element.quad8(mesh, displacement, target, gauss_order=3)

    assert len(calls) == 1
    assert calls[0][1] == {
        "position": StressPosition.ELEMENT_NODAL,
        "element_type": "quad8",
        "gauss_order": 3,
    }
    assert target.read_bytes() == _legacy_plane_bytes(
        mesh,
        displacement,
        {"quad8"},
        3,
    )


@pytest.mark.parametrize(
    ("mesh_builder", "type_keys"),
    (
        (make_mixed_tri3_quad4_mesh, ("tri3", "quad4")),
        (make_mixed_tri6_quad8_mesh, ("tri6", "quad8")),
    ),
)
def test_mixed_plane_legacy_csv_delegates_and_preserves_bytes(
    tmp_path,
    monkeypatch,
    mesh_builder,
    type_keys,
):
    mesh = mesh_builder()
    displacement = np.linspace(-0.03, 0.09, mesh.num_dofs)
    target = tmp_path / f"mixed-{'-'.join(type_keys)}-element.csv"
    calls = []
    original = element.collect_stress

    def spy_collect_stress(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    def reject_legacy_kernel_lookup(*_args, **_kwargs):
        raise AssertionError("plane adapter must not call an element kernel")

    monkeypatch.setattr(element, "collect_stress", spy_collect_stress)
    monkeypatch.setattr(
        element,
        "get_element_kernel",
        reject_legacy_kernel_lookup,
    )

    element.mixed(type_keys, mesh, displacement, target)

    assert len(calls) == 1
    assert calls[0][1] == {
        "position": StressPosition.ELEMENT_NODAL,
        "gauss_order": None,
    }
    assert target.read_bytes() == _legacy_plane_bytes(
        mesh,
        displacement,
        set(type_keys),
        None,
    )


def test_distorted_solid_legacy_representative_point_is_not_centroid_field(
    tmp_path,
):
    mesh = make_unit_hex8_mesh()
    mesh.nodes[2].x = 1.4
    mesh.nodes[2].y = 1.2
    mesh.nodes[5].z = 1.3
    mesh.nodes[6].x = 0.8
    mesh.nodes[7].y = 0.7
    displacement = np.arange(mesh.num_dofs, dtype=float) ** 2 * 0.001
    target = tmp_path / "distorted-hex8-element.csv"

    element.hex8(mesh, displacement, target)

    with target.open(newline="", encoding="utf-8") as stream:
        exported = next(csv.DictReader(stream))
    exported_components = np.asarray([
        float(exported[name])
        for name in ("sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx")
    ])
    representative = np.asarray(
        get_element_kernel("Hex8").stress_at(
            mesh,
            mesh.elements[0],
            displacement,
            0.0,
            0.0,
            0.0,
            node_lookup(mesh),
        )
    )
    centroid = np.asarray(
        collect_stress(
            mesh,
            displacement,
            position=StressPosition.CENTROID,
        ).records[0].components
    )

    assert exported_components == pytest.approx(representative)
    assert np.max(np.abs(representative - centroid)) > 0.4
