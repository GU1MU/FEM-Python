import numpy as np
import pytest

from fem.elements.hexahedron import hex20_gauss_points, hex20_shape_funcs_grads
from fem.elements.triangle import tri6_gauss_points, tri6_shape_funcs_grads


# Independent natural-node ordering, including edge midpoints.
_TRI6_NODES = np.array([
    (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
    (0.5, 0.0), (0.5, 0.5), (0.0, 0.5),
])
_HEX20_NODES = np.array([
    (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
    (0.0, -1.0, -1.0), (1.0, 0.0, -1.0),
    (0.0, 1.0, -1.0), (-1.0, 0.0, -1.0),
    (0.0, -1.0, 1.0), (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0), (-1.0, 0.0, 1.0),
    (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
    (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0),
])


@pytest.mark.parametrize(
    ("shape", "nodes", "interior_points"),
    [
        (tri6_shape_funcs_grads, _TRI6_NODES, [(0.2, 0.3), (0.1, 0.7)]),
        (hex20_shape_funcs_grads, _HEX20_NODES, [
            (-0.3, 0.2, 0.4), (0.0, 0.0, 0.0), (0.7, -0.5, 0.1),
        ]),
    ],
    ids=["tri6", "hex20"],
)
def test_quadratic_shapes_interpolate_nodes_and_reproduce_affine_fields(
    shape, nodes, interior_points,
):
    interpolated_nodes = np.array([shape(*point)[0] for point in nodes])
    np.testing.assert_allclose(interpolated_nodes, np.eye(len(nodes)), atol=1e-14)

    for point in [*nodes, *interior_points]:
        values, *derivatives = shape(*point)
        gradients = np.array(derivatives)

        assert values.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(gradients.sum(axis=1), 0.0, atol=1e-14)
        # The natural-coordinate fields span every linear field; constants above
        # complete the affine basis. Their derivatives form the identity matrix.
        np.testing.assert_allclose(values @ nodes, point, atol=1e-14)
        np.testing.assert_allclose(gradients @ nodes, np.eye(nodes.shape[1]), atol=1e-14)


@pytest.mark.parametrize(
    ("quadrature", "moments"),
    [
        (tri6_gauss_points, [
            ((0, 0), 0.5), ((1, 0), 1.0 / 6.0),
            ((2, 0), 1.0 / 12.0), ((1, 1), 1.0 / 24.0),
        ]),
        (hex20_gauss_points, [
            ((0, 0, 0), 8.0), ((4, 0, 0), 8.0 / 5.0),
            ((4, 2, 0), 8.0 / 15.0), ((2, 2, 2), 8.0 / 27.0),
            ((3, 2, 0), 0.0),
        ]),
    ],
    ids=["unit-triangle", "reference-cube"],
)
def test_quadrature_reproduces_analytic_monomial_moments(quadrature, moments):
    # Triangle: integral(x^a y^b) = a! b! / (a+b+2)!.
    # Cube [-1,1]^3: product of 2/(p+1) for even p, zero for any odd p.
    points = np.array(quadrature())
    coordinates, weights = points[:, :-1], points[:, -1]

    for powers, expected in moments:
        integrand = np.prod(coordinates ** np.array(powers), axis=1)
        assert weights @ integrand == pytest.approx(expected, abs=1e-14)
