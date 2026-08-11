from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup
from .beam_frame import (
    BeamFrame,
    BeamFrameField,
    resolve_beam_frame_field,
)
from .beam_section import (
    Beam2Section,
    BeamSectionEndForces,
    parse_beam2_section,
)


def _required_float_props(elem: Any, *names: str) -> tuple[float, ...]:
    """Return required, physically admissible element properties."""
    try:
        values = tuple(float(elem.props[name]) for name in names)
    except KeyError as exc:
        raise KeyError(
            f"Element {elem.id} missing property {exc.args[0]}, props={elem.props}"
        ) from exc

    for name, value in zip(names, values):
        if name in {"E", "area"} and (
            not np.isfinite(value) or value <= 0.0
        ):
            raise ValueError(
                f"Element {elem.id} property {name} must be finite and > 0, "
                f"got {elem.props[name]!r}"
            )
    return values


def _validate_optional_rho(elem: Any) -> None:
    """Validate optional non-negative density on a line element."""
    if "rho" not in elem.props:
        return
    try:
        rho = float(elem.props["rho"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Element {elem.id} property rho must be finite and >= 0, "
            f"got {elem.props['rho']!r}"
        ) from exc
    if not np.isfinite(rho) or rho < 0.0:
        raise ValueError(
            f"Element {elem.id} property rho must be finite and >= 0, "
            f"got {elem.props['rho']!r}"
        )


def _body_vector_3d(vector: tuple[float, float, float], element_type: str) -> np.ndarray:
    """Return a validated three-component body-force vector."""
    bvec = np.asarray(vector, dtype=float)
    if bvec.shape != (3,):
        raise ValueError(
            f"{element_type} body force must have 3 components, got {bvec.shape}"
        )
    if not np.all(np.isfinite(bvec)):
        raise ValueError(f"{element_type} body force components must be finite")
    return bvec


def line3d_geometry(
    mesh: Any,
    elem: Any,
    node_lookup: dict[int, Any] | None = None,
) -> tuple[float, np.ndarray]:
    """Return length and spatial unit direction for a 2-node line element."""
    if len(elem.node_ids) != 2:
        raise ValueError(
            f"Line2 element {elem.id} requires 2 nodes, got {len(elem.node_ids)}; "
            f"node_ids={elem.node_ids}"
        )
    if node_lookup is None:
        node_lookup = build_node_lookup(mesh)

    try:
        ni = node_lookup[elem.node_ids[0]]
        nj = node_lookup[elem.node_ids[1]]
    except KeyError as exc:
        raise KeyError(
            f"Element {elem.id} references missing node {exc.args[0]}"
        ) from exc

    delta = np.array(
        [nj.x - ni.x, nj.y - ni.y, nj.z - ni.z],
        dtype=float,
    )
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        raise ValueError(f"Line2 element {elem.id} has zero length")
    return length, delta / length


def _beam3_transformation(rotation: np.ndarray) -> np.ndarray:
    """Return global-to-local Beam2 displacement transformation."""
    transformation = np.zeros((12, 12), dtype=float)
    for start in (0, 3, 6, 9):
        transformation[start : start + 3, start : start + 3] = rotation
    return transformation


def _beam2_local_stiffness(
    length: float,
    E: float,
    area: float,
    Iyy: float,
    Izz: float,
    G: float,
    J: float,
    kGA_y: float,
    kGA_z: float,
) -> np.ndarray:
    """Return the one-point-integrated local B31 stiffness."""

    _, strain = _beam2_b31_interpolation(0.5, length)
    constitutive = np.diag(
        (E * area, G * J, E * Iyy, E * Izz, kGA_y, kGA_z),
    )
    return length * (strain.T @ constitutive @ strain)


def _beam2_shear_flexibilities(
    length: float,
    E: float,
    Iyy: float,
    Izz: float,
    kGA_y: float,
    kGA_z: float,
) -> tuple[float, float]:
    """Return transverse ``(phi_y, phi_z)`` flexibility parameters."""

    return (
        12.0 * E * Izz / (kGA_y * length**2),
        12.0 * E * Iyy / (kGA_z * length**2),
    )


def _beam2_b31_interpolation(
    fraction: float,
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the single B31 first-order interpolation and strain owner."""

    r = float(fraction)
    if not 0.0 <= r <= 1.0:
        raise ValueError("Beam2 B31 interpolation fraction must be in [0, 1]")
    shape = np.zeros((6, 12), dtype=float)
    for component in range(6):
        shape[component, (component, component + 6)] = (1.0 - r, r)
    strain = np.zeros((6, 12), dtype=float)
    strain[0, (0, 6)] = (-1.0 / length, 1.0 / length)
    strain[1, (3, 9)] = (-1.0 / length, 1.0 / length)
    strain[2, (4, 10)] = (1.0 / length, -1.0 / length)
    strain[3, (5, 11)] = (-1.0 / length, 1.0 / length)
    strain[4, (1, 5, 7, 11)] = (
        -1.0 / length,
        -(1.0 - r),
        1.0 / length,
        -r,
    )
    strain[5, (2, 4, 8, 10)] = (
        -1.0 / length,
        1.0 - r,
        1.0 / length,
        r,
    )
    return shape, strain


def _beam2_variable_stiffness(
    field: BeamFrameField,
    E: float,
    area: float,
    Iyy: float,
    Izz: float,
    G: float,
    J: float,
    kGA_y: float,
    kGA_z: float,
) -> np.ndarray:
    """Return B31 stiffness using its sole longitudinal integration point."""

    local_stiffness = _beam2_local_stiffness(
        field.length,
        E,
        area,
        Iyy,
        Izz,
        G,
        J,
        kGA_y,
        kGA_z,
    )
    frame = field.frame_at_fraction(0.5)
    transformation = _beam3_transformation(frame.rotation)
    return transformation.T @ local_stiffness @ transformation


def _beam2_legacy_line_load_shape_matrix(
    fraction: float,
    length: float,
    phi_y: float,
    phi_z: float,
) -> np.ndarray:
    """Return the pre-Phase-2 translational load interpolation."""

    def bending(phi: float) -> np.ndarray:
        r = float(fraction)
        denominator = 1.0 + phi
        return np.array(
            (
                2.0 * r**3 - 3.0 * r**2 - phi * r + denominator,
                length
                * (
                    r**3
                    - (2.0 + phi / 2.0) * r**2
                    + (1.0 + phi / 2.0) * r
                ),
                -2.0 * r**3 + 3.0 * r**2 + phi * r,
                length
                * (
                    r**3
                    - (1.0 - phi / 2.0) * r**2
                    - phi * r / 2.0
                ),
            ),
            dtype=float,
        ) / denominator

    shape = np.zeros((3, 12), dtype=float)
    shape[0, (0, 6)] = (1.0 - float(fraction), float(fraction))
    shape[1, (1, 5, 7, 11)] = bending(phi_y)
    shape[2, (2, 4, 8, 10)] = bending(phi_z) * (1.0, -1.0, 1.0, -1.0)
    return shape


def _beam2_variable_line_load(
    field: BeamFrameField,
    vector: tuple[float, float, float],
    coordinate_system: str,
    phi_y: float,
    phi_z: float,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Integrate one line load using the field's frame samples."""
    line_vector = _body_vector_3d(vector, "line load")
    if coordinate_system not in {"global", "local"}:
        raise ValueError(
            "line load coordinate_system must be 'global' or 'local', "
            f"got {coordinate_system!r}"
        )

    def contribution(_fraction: float, frame: BeamFrame) -> np.ndarray:
        shape = _beam2_legacy_line_load_shape_matrix(
            _fraction,
            field.length,
            phi_y,
            phi_z,
        )
        local_vector = (
            frame.rotation @ line_vector
            if coordinate_system == "global"
            else line_vector
        )
        local_force = shape.T @ (scale * local_vector)
        return _beam3_transformation(frame.rotation).T @ local_force

    return np.asarray(field.integrate(contribution), dtype=float)


def _beam_properties(elem: Any) -> tuple[float, float, Beam2Section]:
    """Return validated Beam2 elastic and section properties."""
    (E,) = _required_float_props(elem, "E")
    try:
        nu = float(elem.props["nu"])
    except KeyError as exc:
        raise KeyError(f"Element {elem.id} missing property nu") from exc
    if not np.isfinite(nu) or not -1.0 < nu < 0.5:
        raise ValueError(
            f"Element {elem.id} property nu must be finite and satisfy -1 < nu < 0.5, "
            f"got {elem.props['nu']!r}"
        )
    return E, nu, parse_beam2_section(elem.props)


class Truss2Kernel:
    """Two-node spatial truss element kernel."""
    canonical_type = "Truss2"
    aliases = ()
    edge_node_indices = ((0, 1),)

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the 6-by-6 spatial truss stiffness matrix."""
        area, E = _required_float_props(elem, "area", "E")
        _validate_optional_rho(elem)
        length, direction = line3d_geometry(mesh, elem, node_lookup)
        block = np.outer(direction, direction)
        return E * area / length * np.block([[block, -block], [-block, block]])

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the consistent spatial truss body-force vector."""
        (area,) = _required_float_props(elem, "area")
        length, _ = line3d_geometry(mesh, elem, node_lookup)
        nodal = _body_vector_3d(vector, "Truss2") * (area * length / 2.0)
        return np.tile(nodal, 2)

    def element_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
    ) -> tuple[float, float, float]:
        """Return axial strain, axial stress, and equivalent stress."""
        (E,) = _required_float_props(elem, "E")
        length, direction = line3d_geometry(mesh, elem, node_lookup)
        ni_id, nj_id = elem.node_ids
        displacement_i = np.asarray(U[mesh.node_dofs(ni_id)], dtype=float)
        displacement_j = np.asarray(U[mesh.node_dofs(nj_id)], dtype=float)
        axial_strain = float(direction @ (displacement_j - displacement_i) / length)
        axial_stress = E * axial_strain
        return axial_strain, float(axial_stress), float(abs(axial_stress))


class Beam2Kernel:
    """Two-node spatial linear-static Timoshenko beam element kernel."""
    canonical_type = "Beam2"
    aliases = ()
    edge_node_indices = ((0, 1),)

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the transformed 12-by-12 Beam2 stiffness matrix."""
        E, nu, section = _beam_properties(elem)
        _validate_optional_rho(elem)
        field = resolve_beam_frame_field(mesh, elem, node_lookup)
        G = E / (2.0 * (1.0 + nu))
        kGA_y, kGA_z = section.abaqus_b31_shear_rigidities(
            G,
            nu,
            field.length,
        )
        return _beam2_variable_stiffness(
            field,
            E,
            section.area,
            section.Iyy,
            section.Izz,
            G,
            section.J,
            kGA_y,
            kGA_z,
        )

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the consistent Beam2 body-force vector."""
        E, nu, section = _beam_properties(elem)
        field = resolve_beam_frame_field(mesh, elem, node_lookup)
        global_vector = _body_vector_3d(vector, "Beam2")
        G = E / (2.0 * (1.0 + nu))
        kGA_y, kGA_z = section.effective_shear_rigidities(G, nu)
        phi_y, phi_z = _beam2_shear_flexibilities(
            field.length,
            E,
            section.Iyy,
            section.Izz,
            kGA_y,
            kGA_z,
        )
        if not field.is_constant:
            return _beam2_variable_line_load(
                field,
                tuple(float(value) for value in global_vector),
                "global",
                phi_y,
                phi_z,
                scale=section.area,
            )
        frame = field.as_constant_frame()
        local_line_load = section.area * (frame.rotation @ global_vector)
        local_force = _beam2_consistent_line_load(
            frame.length,
            local_line_load,
            phi_y,
            phi_z,
        )
        return _beam3_transformation(frame.rotation).T @ local_force

    def line_load(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float, float],
        coordinate_system: str = "global",
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the consistent Beam2 force for a constant line load."""
        E, nu, section = _beam_properties(elem)
        field = resolve_beam_frame_field(mesh, elem, node_lookup)
        G = E / (2.0 * (1.0 + nu))
        kGA_y, kGA_z = section.effective_shear_rigidities(G, nu)
        phi_y, phi_z = _beam2_shear_flexibilities(
            field.length,
            E,
            section.Iyy,
            section.Izz,
            kGA_y,
            kGA_z,
        )
        if not field.is_constant:
            return _beam2_variable_line_load(
                field,
                vector,
                coordinate_system,
                phi_y,
                phi_z,
            )
        frame = field.as_constant_frame()
        local_vector = _beam2_local_line_vector(
            vector,
            coordinate_system,
            frame,
        )
        local_force = _beam2_consistent_line_load(
            frame.length,
            local_vector,
            phi_y,
            phi_z,
        )
        return _beam3_transformation(frame.rotation).T @ local_force

    def local_line_load(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float, float],
        coordinate_system: str = "global",
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the consistent vector used by assembly and recovery.

        For a varying field the vector is the global equivalent nodal load;
        this keeps recovery on the same interpolation owner as assembly.
        Constant fields retain the historical local-vector convention.
        """
        E, nu, section = _beam_properties(elem)
        field = resolve_beam_frame_field(mesh, elem, node_lookup)
        G = E / (2.0 * (1.0 + nu))
        kGA_y, kGA_z = section.effective_shear_rigidities(G, nu)
        phi_y, phi_z = _beam2_shear_flexibilities(
            field.length,
            E,
            section.Iyy,
            section.Izz,
            kGA_y,
            kGA_z,
        )
        if not field.is_constant:
            return _beam2_variable_line_load(
                field,
                vector,
                coordinate_system,
                phi_y,
                phi_z,
            )
        frame = field.as_constant_frame()
        local_vector = _beam2_local_line_vector(
            vector,
            coordinate_system,
            frame,
        )
        return _beam2_consistent_line_load(
            frame.length,
            local_vector,
            phi_y,
            phi_z,
        )

    def local_end_actions(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        equivalent_local_load: np.ndarray | None = None,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return tension-positive (N, My, Mz) at both Beam2 ends."""
        section_forces = self.local_section_end_forces(
            mesh,
            elem,
            U,
            equivalent_local_load,
            node_lookup,
        )
        return np.asarray(
            [
                (forces.axial_force, forces.moment_y, forces.moment_z)
                for forces in section_forces
            ],
            dtype=float,
        )

    def local_section_end_forces(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        equivalent_local_load: np.ndarray | None = None,
        node_lookup: dict[int, Any] | None = None,
    ) -> tuple[BeamSectionEndForces, BeamSectionEndForces]:
        """Return local ``(N, Vy, Vz, My, Mz, T)`` at both Beam2 ends."""
        E, nu, section = _beam_properties(elem)
        field = resolve_beam_frame_field(mesh, elem, node_lookup)
        G = E / (2.0 * (1.0 + nu))
        kGA_y, kGA_z = section.abaqus_b31_shear_rigidities(
            G,
            nu,
            field.length,
        )
        element_displacement = np.asarray(U, dtype=float)[
            list(mesh.element_dofs(elem))
        ]
        if element_displacement.shape != (12,):
            raise ValueError(
                f"Beam2 element {elem.id} displacement requires 12 values"
            )
        if not field.is_constant:
            local_load = (
                np.zeros(12, dtype=float)
                if equivalent_local_load is None
                else np.asarray(equivalent_local_load, dtype=float)
            )
            if local_load.shape != (12,) or not np.all(np.isfinite(local_load)):
                raise ValueError(
                    f"Beam2 element {elem.id} equivalent local load must have "
                    "12 finite values"
                )
            stiffness = _beam2_variable_stiffness(
                field,
                E,
                section.area,
                section.Iyy,
                section.Izz,
                G,
                section.J,
                kGA_y,
                kGA_z,
            )
            action = stiffness @ element_displacement - local_load
            start_force = field.start.rotation @ action[:3]
            start_moment = field.start.rotation @ action[3:6]
            end_force = field.end.rotation @ action[6:9]
            end_moment = field.end.rotation @ action[9:12]
            return (
                BeamSectionEndForces(
                    -start_force[0],
                    -start_moment[1],
                    -start_moment[2],
                    -start_moment[0],
                    shear_y=-start_force[1],
                    shear_z=-start_force[2],
                ),
                BeamSectionEndForces(
                    end_force[0],
                    end_moment[1],
                    end_moment[2],
                    end_moment[0],
                    shear_y=end_force[1],
                    shear_z=end_force[2],
                ),
            )
        frame = field.as_constant_frame()
        local_stiffness = _beam2_local_stiffness(
            frame.length,
            E,
            section.area,
            section.Iyy,
            section.Izz,
            G,
            section.J,
            kGA_y,
            kGA_z,
        )
        local_displacement = (
            _beam3_transformation(frame.rotation) @ element_displacement
        )
        local_load = (
            np.zeros(12, dtype=float)
            if equivalent_local_load is None
            else np.asarray(equivalent_local_load, dtype=float)
        )
        if local_load.shape != (12,) or not np.all(np.isfinite(local_load)):
            raise ValueError(
                f"Beam2 element {elem.id} equivalent local load must have 12 finite values"
            )
        action = local_stiffness @ local_displacement - local_load
        return (
            BeamSectionEndForces(
                -action[0],
                -action[4],
                -action[5],
                -action[3],
                shear_y=-action[1],
                shear_z=-action[2],
            ),
            BeamSectionEndForces(
                action[6],
                action[10],
                action[11],
                action[9],
                shear_y=action[7],
                shear_z=action[8],
            ),
        )


def _beam2_consistent_line_load(
    length: float,
    vector: np.ndarray,
    phi_y: float,
    phi_z: float,
) -> np.ndarray:
    """Return the exact Timoshenko nodal forces for a constant line load."""

    qx, qy, qz = vector
    y_denominator = 1.0 + phi_y
    z_denominator = 1.0 + phi_z
    y_force_integral = length * (0.5 + phi_y / 2.0) / y_denominator
    y_moment_integral = length**2 * (1.0 + phi_y) / (
        12.0 * y_denominator
    )
    z_force_integral = length * (0.5 + phi_z / 2.0) / z_denominator
    z_moment_integral = length**2 * (1.0 + phi_z) / (
        12.0 * z_denominator
    )
    return np.array(
        [
            qx * length / 2.0,
            qy * y_force_integral,
            qz * z_force_integral,
            0.0,
            -qz * z_moment_integral,
            qy * y_moment_integral,
            qx * length / 2.0,
            qy * y_force_integral,
            qz * z_force_integral,
            0.0,
            qz * z_moment_integral,
            -qy * y_moment_integral,
        ],
        dtype=float,
    )


def _beam2_local_line_vector(
    vector: tuple[float, float, float],
    coordinate_system: str,
    frame: BeamFrame,
) -> np.ndarray:
    """Return one validated constant line-load vector in local components."""

    line_vector = _body_vector_3d(vector, "line load")
    if coordinate_system not in {"global", "local"}:
        raise ValueError(
            "line load coordinate_system must be 'global' or 'local', "
            f"got {coordinate_system!r}"
        )
    return (
        frame.rotation @ line_vector
        if coordinate_system == "global"
        else line_vector
    )
