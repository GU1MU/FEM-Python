from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.post.stress import truss


# ---------------------------------------------------------------------------
# Truss2 pure-post recovery


def _inclined_truss(
    *,
    reversed_connectivity: bool = False,
) -> tuple[Mesh3D, np.ndarray]:
    node_ids = [20, 10] if reversed_connectivity else [10, 20]
    mesh = Mesh3D(
        nodes=[
            Node3D(10, 0.0, 0.0, 0.0),
            Node3D(20, 2.0, 3.0, 6.0),
        ],
        elements=[
            Element3D(
                70,
                node_ids,
                "Truss2",
                {"E": 200.0, "area": 2.0},
            )
        ],
    )
    displacement = np.zeros(mesh.num_dofs)
    displacement[list(mesh.node_dofs(20))] = (0.2, 0.3, 0.6)
    return mesh, displacement


def test_truss_recovery_returns_analytical_immutable_centroid_row() -> None:
    mesh, displacement = _inclined_truss()

    recovered = truss.recover(mesh, displacement)

    assert isinstance(recovered, truss.TrussStressField)
    assert recovered.position == "centroid"
    assert recovered.component_names == ("LE11", "S11", "Mises")
    assert isinstance(recovered.rows, tuple)
    assert len(recovered.rows) == 1
    row = recovered.rows[0]
    assert row.element_id == 70
    assert row.coordinates == pytest.approx((1.0, 1.5, 3.0))
    assert row.displacement == pytest.approx((0.1, 0.15, 0.3))
    assert row.LE11 == row.le11 == pytest.approx(0.1)
    assert row.S11 == row.s11 == pytest.approx(20.0)
    assert row.Mises == row.mises == pytest.approx(abs(row.S11))
    assert row.values() == {
        "LE11": pytest.approx(0.1),
        "S11": pytest.approx(20.0),
        "Mises": pytest.approx(20.0),
    }
    with pytest.raises(FrozenInstanceError):
        row.S11 = 0.0
    with pytest.raises(FrozenInstanceError):
        recovered.rows = ()


def test_truss_recovery_is_invariant_to_reversed_connectivity() -> None:
    forward_mesh, forward_u = _inclined_truss()
    reversed_mesh, reversed_u = _inclined_truss(reversed_connectivity=True)

    forward = truss.recover(forward_mesh, forward_u).rows[0]
    reversed_row = truss.recover(reversed_mesh, reversed_u).rows[0]

    assert reversed_row.element_id == forward.element_id
    assert reversed_row.coordinates == pytest.approx(forward.coordinates)
    assert reversed_row.displacement == pytest.approx(forward.displacement)
    assert reversed_row.LE11 == pytest.approx(forward.LE11)
    assert reversed_row.S11 == pytest.approx(forward.S11)
    assert reversed_row.Mises == pytest.approx(forward.Mises)


def test_truss_recovery_preserves_noncontiguous_mesh_element_order_and_uses_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = Mesh3D(
        nodes=[
            Node3D(10, 0.0, 0.0, 0.0),
            Node3D(20, 1.0, 0.0, 0.0),
            Node3D(30, 3.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(90, [20, 30], "Truss2", {"E": 100.0, "area": 1.0}),
            Element3D(7, [10, 20], "Truss2", {"E": 100.0, "area": 1.0}),
        ],
    )
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(20, 0)] = 0.1
    displacement[mesh.global_dof(30, 0)] = 0.5
    kernel = truss.get_element_kernel("Truss2")
    kernel_type = type(kernel)
    original = kernel_type.element_stress
    calls: list[int] = []

    def counted(self, mesh_, element, values, lookup):
        calls.append(int(element.id))
        return original(self, mesh_, element, values, lookup)

    monkeypatch.setattr(kernel_type, "element_stress", counted)

    recovered = truss.recover(mesh, displacement)

    assert calls == [90, 7]
    assert [row.element_id for row in recovered.rows] == [90, 7]
    assert [row.coordinates for row in recovered.rows] == pytest.approx(
        [(2.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    )
    assert [row.displacement for row in recovered.rows] == pytest.approx(
        [(0.3, 0.0, 0.0), (0.05, 0.0, 0.0)]
    )
    assert [row.LE11 for row in recovered.rows] == pytest.approx([0.2, 0.1])
    assert [row.S11 for row in recovered.rows] == pytest.approx([20.0, 10.0])
    assert [row.Mises for row in recovered.rows] == pytest.approx([20.0, 10.0])


@pytest.mark.parametrize(
    ("elements", "dofs_per_node", "message"),
    [
        ([], 3, "at least one element"),
        (
            [
                Element3D(
                    1,
                    [1, 2],
                    "Beam2",
                    {
                        "E": 100.0,
                        "nu": 0.25,
                        "section_type": "solid_circle",
                        "radius": 1.0,
                    },
                )
            ],
            3,
            "homogeneous Truss2",
        ),
        (
            [
                Element3D(1, [1, 2], "Truss2", {"E": 100.0, "area": 1.0}),
                Element3D(
                    2,
                    [1, 2],
                    "Beam2",
                    {
                        "E": 100.0,
                        "nu": 0.25,
                        "section_type": "solid_circle",
                        "radius": 1.0,
                    },
                ),
            ],
            3,
            "homogeneous Truss2",
        ),
        (
            [Element3D(1, [1, 2], "Unknown2", {})],
            3,
            "unsupported type",
        ),
        (
            [Element3D(1, [1, 2], "Truss2", {"E": 100.0, "area": 1.0})],
            6,
            "three translational DOFs",
        ),
    ],
)
def test_truss_recovery_rejects_unsupported_or_mixed_meshes(
    elements: list[Element3D],
    dofs_per_node: int,
    message: str,
) -> None:
    mesh = Mesh3D(
        [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
        elements,
        dofs_per_node=dofs_per_node,
    )

    with pytest.raises(ValueError, match=message):
        truss.recover(mesh, np.zeros(mesh.num_dofs))


@pytest.mark.parametrize(
    "displacement",
    [
        np.zeros((2, 3)),
        np.zeros(5),
        np.asarray([0.0, 0.0, 0.0, np.nan, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, np.inf, 0.0, 0.0]),
        np.asarray([False, False, False, False, False, False]),
    ],
)
def test_truss_recovery_rejects_invalid_displacement(
    displacement: np.ndarray,
) -> None:
    mesh, _ = _inclined_truss()

    with pytest.raises((TypeError, ValueError), match="U"):
        truss.recover(mesh, displacement)


@pytest.mark.parametrize(
    ("mesh", "message"),
    [
        (
            Mesh3D(
                [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
                [Element3D(1, [1, 2], "Truss2", {"area": 1.0})],
            ),
            "missing property E",
        ),
        (
            Mesh3D(
                [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
                [
                    Element3D(
                        1,
                        [1, 2],
                        "Truss2",
                        {"E": np.nan, "area": 1.0},
                    )
                ],
            ),
            "property E must be finite",
        ),
        (
            Mesh3D(
                [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 0.0, 0.0, 0.0)],
                [
                    Element3D(
                        1,
                        [1, 2],
                        "Truss2",
                        {"E": 100.0, "area": 1.0},
                    )
                ],
            ),
            "zero length",
        ),
        (
            Mesh3D(
                [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
                [
                    Element3D(
                        1,
                        [1, 3],
                        "Truss2",
                        {"E": 100.0, "area": 1.0},
                    )
                ],
            ),
            "missing mesh nodes",
        ),
    ],
)
def test_truss_recovery_rejects_invalid_element_data(
    mesh: Mesh3D,
    message: str,
) -> None:
    with pytest.raises((KeyError, ValueError), match=message):
        truss.recover(mesh, np.zeros(mesh.num_dofs))


def test_truss_recovery_owns_values_after_caller_mutation() -> None:
    mesh, displacement = _inclined_truss()

    recovered = truss.recover(mesh, displacement)
    row = recovered.rows[0]
    original = (
        row.element_id,
        row.coordinates,
        row.displacement,
        row.LE11,
        row.S11,
        row.Mises,
    )

    displacement[:] = 999.0
    mesh.nodes[0].x = 999.0
    mesh.elements[0].id = 999
    mesh.elements[0].node_ids[:] = [20, 20]
    mesh.elements[0].props["E"] = 999.0

    assert (
        row.element_id,
        row.coordinates,
        row.displacement,
        row.LE11,
        row.S11,
        row.Mises,
    ) == original


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"coordinates": (0.0, 0.0)}, ValueError),
        ({"displacement": (0.0, 0.0, np.nan)}, ValueError),
        ({"LE11": np.inf}, ValueError),
        ({"S11": "1.0"}, TypeError),
        ({"Mises": -1.0}, ValueError),
    ],
)
def test_truss_stress_row_rejects_invalid_typed_values(
    changes: dict[str, object],
    error: type[Exception],
) -> None:
    values: dict[str, object] = {
        "element_id": 1,
        "coordinates": (0.0, 0.0, 0.0),
        "displacement": (0.0, 0.0, 0.0),
        "LE11": 0.0,
        "S11": 0.0,
        "Mises": 0.0,
    }
    values.update(changes)

    with pytest.raises(error):
        truss.TrussStressRow(**values)
