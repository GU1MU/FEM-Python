"""Mesh-owned Gmsh controls, fields, policies, and generation runtime."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, cast

from fem.geometry import EntityRef, FeatureResult, GeometryError, GeometryStateError

from ._configuration import _MeshConfiguration
from ._field_registry import _MeshFieldRegistry, _normalize_active_tags
from ._policies import (
    _GMSH_TOP_CELL_TYPE_NAMES,
    _GenerationSizeMode,
    _MeshGenerationPolicy,
    _automatic_policy,
    _compose_numeric_options,
    _explicit_policy,
)
from ._protocols import _GeometryMeshingPort
from ._validation import (
    _finite_float,
    _integer_at_least,
    _nonnegative_float,
    _positive_float,
    _validate_positive_tag,
)
from .errors import MeshCellShapeError
from .specs import AutoMeshSpec, MeshSpec
from .types import (
    GmshMeshRef,
    MeshFieldRef,
    _prepare_generated_mesh_reference,
)


_FieldType = Literal["Distance", "Threshold", "Min"]
_NativeFieldConfiguration = Callable[[Any, int], None]


class _GmshMeshRuntime:
    """Own one bound model's complete native-meshing workflow."""

    __slots__ = ("_configuration", "_fields", "_port")

    def __init__(self, port: _GeometryMeshingPort) -> None:
        self._port = port
        self._configuration = _MeshConfiguration()
        self._fields = _MeshFieldRegistry(port.model_name)

    def transfinite_curve(
        self,
        curve: EntityRef,
        *,
        num_nodes: int,
    ) -> None:
        operation = "transfinite_curve"
        self._port.validate(operation)
        node_count = _integer_at_least(num_nodes, "num_nodes", minimum=2)
        target = self._prepare_control_target(
            curve,
            dimension=1,
            operation=operation,
        )
        dependency_keys = self._port.boundary_closure(
            (target,),
            operation=operation,
        )

        def apply_control(model: Any) -> None:
            model.mesh.setTransfiniteCurve(target.tag, node_count)

        self._port.native_control(operation, apply_control)
        self._port.register_control_dependencies(
            dependency_keys,
            transform_unsafe=True,
        )
        self._configuration.add_auto_blocker(operation)

    def transfinite_surface(
        self,
        surface: EntityRef,
        *,
        corners: Sequence[EntityRef] = (),
    ) -> None:
        operation = "transfinite_surface"
        self._port.validate(operation)
        normalized_corners = self._normalize_control_corners(
            corners,
            allowed_counts=(0, 3, 4),
            operation=operation,
        )
        target = self._prepare_control_target(
            surface,
            dimension=2,
            operation=operation,
            related_entities=normalized_corners,
        )
        self._port.assert_corners_on_boundary(
            target,
            normalized_corners,
            operation=operation,
        )
        dependency_keys = self._port.boundary_closure(
            (target, *normalized_corners),
            operation=operation,
        )

        def apply_control(model: Any) -> None:
            model.mesh.setTransfiniteSurface(
                target.tag,
                cornerTags=[corner.tag for corner in normalized_corners],
            )

        self._port.native_control(operation, apply_control)
        self._port.register_control_dependencies(
            dependency_keys,
            transform_unsafe=True,
        )
        self._configuration.add_auto_blocker(operation)

    def transfinite_volume(
        self,
        volume: EntityRef,
        *,
        corners: Sequence[EntityRef] = (),
    ) -> None:
        operation = "transfinite_volume"
        self._port.validate(operation)
        normalized_corners = self._normalize_control_corners(
            corners,
            allowed_counts=(0, 6, 8),
            operation=operation,
        )
        target = self._prepare_control_target(
            volume,
            dimension=3,
            operation=operation,
            related_entities=normalized_corners,
        )
        self._port.assert_corners_on_boundary(
            target,
            normalized_corners,
            operation=operation,
        )
        dependency_keys = self._port.boundary_closure(
            (target, *normalized_corners),
            operation=operation,
        )

        def apply_control(model: Any) -> None:
            model.mesh.setTransfiniteVolume(
                target.tag,
                cornerTags=[corner.tag for corner in normalized_corners],
            )

        self._port.native_control(operation, apply_control)
        self._port.register_control_dependencies(
            dependency_keys,
            transform_unsafe=True,
        )
        self._configuration.add_auto_blocker(operation)

    def recombine(self, surface: EntityRef) -> None:
        operation = "recombine"
        self._port.validate(operation)
        target = self._prepare_control_target(
            surface,
            dimension=2,
            operation=operation,
        )
        dependency_keys = self._port.boundary_closure(
            (target,),
            operation=operation,
        )

        def apply_control(model: Any) -> None:
            model.mesh.setRecombine(2, target.tag)

        self._port.native_control(operation, apply_control)
        self._port.register_control_dependencies(
            dependency_keys,
            transform_unsafe=True,
        )
        self._configuration.add_auto_blocker(operation)

    def mesh_size(
        self,
        points: Iterable[EntityRef],
        *,
        size: float,
    ) -> None:
        operation = "mesh_size"
        self._port.validate(operation)
        self._configuration.assert_mesh_size_allowed()
        size_value = _positive_float(size, "size")
        normalized = self._port.normalize_entities(points, operation=operation)
        if any(point.dimension != 0 for point in normalized):
            raise ValueError("mesh_size requires dimension-zero point references")
        self._port.assert_entities_live(normalized, operation=operation)
        point_dim_tags = _dim_tags(normalized)

        def apply_size(model: Any) -> None:
            model.mesh.setSize(list(point_dim_tags), size_value)

        self._port.native_control(operation, apply_size)
        self._configuration.commit_mesh_size()
        self._port.register_control_dependencies(
            point_dim_tags,
            transform_unsafe=True,
        )

    def distance_field(
        self,
        *,
        points: Iterable[EntityRef] = (),
        curves: Iterable[EntityRef] = (),
        surfaces: Iterable[EntityRef] = (),
        sampling: int = 20,
    ) -> MeshFieldRef:
        operation = "distance_field"
        self._port.validate(operation)
        normalized_points = self._port.normalize_optional_entities(
            points,
            operation=operation,
            label="points",
        )
        normalized_curves = self._port.normalize_optional_entities(
            curves,
            operation=operation,
            label="curves",
        )
        normalized_surfaces = self._port.normalize_optional_entities(
            surfaces,
            operation=operation,
            label="surfaces",
        )
        source_groups = (
            ("points", 0, normalized_points),
            ("curves", 1, normalized_curves),
            ("surfaces", 2, normalized_surfaces),
        )
        all_sources = tuple(
            entity for _, _, group in source_groups for entity in group
        )
        if not all_sources:
            raise ValueError("distance_field requires at least one source entity")
        for label, dimension, group in source_groups:
            if any(entity.dimension != dimension for entity in group):
                raise ValueError(
                    f"distance_field {label} must be dimension-{dimension} "
                    "entity references"
                )
        source_keys = [(entity.dimension, entity.tag) for entity in all_sources]
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("distance_field source entities must be duplicate-free")
        sampling_value = _integer_at_least(sampling, "sampling", minimum=2)
        self._port.assert_entities_live(all_sources, operation=operation)
        dependency_keys = self._port.boundary_closure(
            all_sources,
            operation=operation,
        )

        def configure(model: Any, field_tag: int) -> None:
            manager = model.mesh.field
            if normalized_points:
                manager.setNumbers(
                    field_tag,
                    "PointsList",
                    [point.tag for point in normalized_points],
                )
            if normalized_curves:
                manager.setNumbers(
                    field_tag,
                    "CurvesList",
                    [curve.tag for curve in normalized_curves],
                )
            if normalized_surfaces:
                manager.setNumbers(
                    field_tag,
                    "SurfacesList",
                    [surface.tag for surface in normalized_surfaces],
                )
            manager.setNumber(field_tag, "Sampling", sampling_value)

        mesh_field = self._construct_field("Distance", operation, configure)
        self._port.register_control_dependencies(
            dependency_keys,
            transform_unsafe=False,
        )
        return mesh_field

    def threshold_field(
        self,
        distance: MeshFieldRef,
        *,
        size_min: float,
        size_max: float,
        dist_min: float,
        dist_max: float,
    ) -> MeshFieldRef:
        operation = "threshold_field"
        self._port.validate(operation)
        normalized_distance = self._fields.normalize(
            (distance,),
            operation=operation,
        )[0]
        if normalized_distance.field_type != "Distance":
            raise ValueError("threshold_field requires a Distance field")
        size_min_value = _positive_float(size_min, "size_min")
        size_max_value = _positive_float(size_max, "size_max")
        if size_min_value >= size_max_value:
            raise ValueError("size_min must be less than size_max")
        dist_min_value = _nonnegative_float(dist_min, "dist_min")
        dist_max_value = _finite_float(dist_max, "dist_max")
        if dist_max_value <= dist_min_value:
            raise ValueError("dist_max must be greater than dist_min")
        active_tags = self._active_field_tags(operation)
        self._fields.assert_liveness(
            (normalized_distance,),
            active_tags,
            operation=operation,
        )

        def configure(model: Any, field_tag: int) -> None:
            manager = model.mesh.field
            manager.setNumber(field_tag, "InField", normalized_distance.tag)
            manager.setNumber(field_tag, "SizeMin", size_min_value)
            manager.setNumber(field_tag, "SizeMax", size_max_value)
            manager.setNumber(field_tag, "DistMin", dist_min_value)
            manager.setNumber(field_tag, "DistMax", dist_max_value)

        return self._construct_field("Threshold", operation, configure)

    def min_field(self, fields: Sequence[MeshFieldRef]) -> MeshFieldRef:
        operation = "min_field"
        self._port.validate(operation)
        try:
            materialized = tuple(fields)
        except TypeError as exc:
            raise TypeError("min_field fields must be iterable") from exc
        if len(materialized) < 2:
            raise ValueError("min_field requires at least two fields")
        normalized = self._fields.normalize(materialized, operation=operation)
        if any(field.field_type not in {"Threshold", "Min"} for field in normalized):
            raise ValueError("min_field accepts only Threshold and Min size fields")
        active_tags = self._active_field_tags(operation)
        self._fields.assert_liveness(
            normalized,
            active_tags,
            operation=operation,
        )

        def configure(model: Any, field_tag: int) -> None:
            model.mesh.field.setNumbers(
                field_tag,
                "FieldsList",
                [item.tag for item in normalized],
            )

        return self._construct_field("Min", operation, configure)

    def background_field(self, field: MeshFieldRef) -> None:
        operation = "background_field"
        self._port.validate(operation)
        self._configuration.assert_background_allowed()
        normalized = self._fields.normalize((field,), operation=operation)[0]
        if normalized.field_type not in {"Threshold", "Min"}:
            raise ValueError(
                "background_field requires a Threshold or Min size field"
            )
        active_tags = self._active_field_tags(operation)
        self._fields.assert_liveness(
            (normalized,),
            active_tags,
            operation=operation,
        )

        def select_background(model: Any) -> None:
            model.mesh.field.setAsBackgroundMesh(normalized.tag)

        self._port.native_control(operation, select_background)
        self._configuration.commit_background(normalized)

    def structured_extrude(
        self,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
        *,
        num_elements: Sequence[int] = (),
        heights: Sequence[float] = (),
        recombine: bool = False,
    ) -> FeatureResult:
        result = self._port.structured_extrude(
            entities,
            dx,
            dy,
            dz,
            num_elements=num_elements,
            heights=heights,
            recombine=recombine,
        )
        self._configuration.add_auto_blocker("structured_extrude")
        return result

    def generate(self, spec: MeshSpec | AutoMeshSpec) -> GmshMeshRef:
        if isinstance(spec, MeshSpec):
            return self._generate_explicit(
                size=spec.size,
                order=spec.order,
                recombine=spec.recombine,
            )
        if isinstance(spec, AutoMeshSpec):
            return self._generate_automatic(
                level=spec.level,
                cell_shape=spec.cell_shape,
                order=spec.order,
            )
        raise TypeError(
            "spec must be a MeshSpec or AutoMeshSpec, "
            f"got {spec!r}"
        )

    def _generate_explicit(
        self,
        *,
        size: float | None,
        order: Literal[1, 2],
        recombine: bool,
    ) -> GmshMeshRef:
        operation = "MeshSpec generation"
        self._port.validate("generate")
        self._configuration.assert_attempt_available(
            self._port.model_name,
            operation,
        )
        size_value = None if size is None else _positive_float(size, "size")
        self._configuration.assert_uniform_size_allowed(size_value)
        policy = _explicit_policy(
            self._port.dimension,
            order=order,
            recombine=recombine,
        )
        return self._generate_native_mesh(policy=policy, size_value=size_value)

    def _generate_automatic(
        self,
        *,
        level: Literal[1, 2, 3, 4, 5],
        cell_shape: Literal["tri", "tri-quad", "quad", "tet", "hex"] | None,
        order: Literal[1, 2],
    ) -> GmshMeshRef:
        operation = "AutoMeshSpec generation"
        self._port.validate("generate")
        self._configuration.assert_attempt_available(
            self._port.model_name,
            operation,
        )
        dimension = cast(Literal[1, 2, 3], self._port.dimension)
        policy = _automatic_policy(
            dimension,
            level=level,
            cell_shape=cell_shape,
            order=order,
        )
        self._configuration.assert_automatic_allowed(
            self._port.model_name,
            topology_provenance_unknown=self._port.topology_provenance_unknown,
        )
        return self._generate_native_mesh(policy=policy, size_value=None)

    def _generate_native_mesh(
        self,
        *,
        policy: _MeshGenerationPolicy,
        size_value: float | None,
    ) -> GmshMeshRef:
        operation = policy.operation
        top_entities = self._native_entity_pairs(self._port.dimension, operation)
        if not top_entities:
            raise ValueError(
                f"geometry model {self._port.model_name!r}: mesh generation "
                "requires at least one top-dimensional OCC entity"
            )

        self._port.commit_generation_attempt(operation)
        self._configuration.consume_attempt()
        try:
            generation_size_mode: _GenerationSizeMode = self._configuration.size_mode
            if size_value is not None:
                generation_size_mode = "uniform"
                points = self._native_entity_pairs(0, operation)
                if not points:
                    raise GeometryError(
                        f"geometry model {self._port.model_name!r}: uniform mesh "
                        "size requires at least one point entity"
                    )

                def apply_uniform_size(model: Any) -> None:
                    model.mesh.setSize(list(points), size_value)

                self._port.native_control(operation, apply_uniform_size)

            self._apply_mesh_options(policy, size_mode=generation_size_mode)

            def generate_native(model: Any) -> None:
                model.mesh.generate(self._port.dimension)

            self._port.native_control(operation, generate_native)

            def synchronize_and_validate_cells(model: Any) -> None:
                if policy.strict_cell_shape:
                    _validate_generated_top_cell_shape(
                        model,
                        policy,
                        model_name=self._port.model_name,
                        dimension=self._port.dimension,
                    )

            self._port.native_query(operation, synchronize_and_validate_cells)
        except BaseException as error:
            self._port.fail_generation(operation)
            try:
                self._port.restore_numeric_options()
            except BaseException as restore_error:
                error.add_note(
                    f"geometry model {self._port.model_name!r}: additionally "
                    f"failed to restore Gmsh mesh options: {restore_error}"
                )
            error.add_note(
                f"geometry model {self._port.model_name!r}: mesh generation failed"
            )
            raise

        try:
            self._port.restore_numeric_options()
        except BaseException as error:
            self._port.fail_generation(operation)
            raise GeometryError(
                f"geometry model {self._port.model_name!r}: mesh generation "
                "succeeded but restoring global Gmsh options failed"
            ) from error

        try:
            native_borrow = self._port.prepare_native_borrow(operation)
            dimension = cast(Literal[1, 2, 3], self._port.dimension)
            reference, lease = _prepare_generated_mesh_reference(
                native_borrow,
                dimension=dimension,
                model_name=self._port.model_name,
            )
            self._port.complete_generation(operation)
        except BaseException as error:
            try:
                self._port.fail_generation(operation)
            except BaseException as terminal_error:
                error.add_note(
                    f"geometry model {self._port.model_name!r}: additionally "
                    "failed to enter terminal mesh-failure state: "
                    f"{terminal_error}"
                )
            error.add_note(
                f"geometry model {self._port.model_name!r}: mesh generation failed"
            )
            raise

        lease._activate()
        return reference

    def _apply_mesh_options(
        self,
        policy: _MeshGenerationPolicy,
        *,
        size_mode: _GenerationSizeMode,
    ) -> None:
        if self._port.has_pending_numeric_options:
            raise GeometryStateError(
                f"geometry model {self._port.model_name!r}: mesh options "
                "already have a pending restoration"
            )
        requested = _compose_numeric_options(
            policy,
            size_mode=size_mode,
            model_name=self._port.model_name,
        )
        self._port.apply_numeric_options(requested)

    def _prepare_control_target(
        self,
        entity: EntityRef,
        *,
        dimension: int,
        operation: str,
        related_entities: Iterable[EntityRef] = (),
    ) -> EntityRef:
        target = self._port.normalize_entities((entity,), operation=operation)[0]
        if target.dimension != dimension:
            raise ValueError(
                f"{operation} target must be a dimension-{dimension} entity"
            )
        if dimension > self._port.dimension:
            raise ValueError(
                f"{operation} target dimension exceeds the facade dimension"
            )
        self._port.assert_entities_live(
            (target, *related_entities),
            operation=operation,
        )
        return target

    def _normalize_control_corners(
        self,
        corners: Sequence[EntityRef],
        *,
        allowed_counts: tuple[int, ...],
        operation: str,
    ) -> tuple[EntityRef, ...]:
        try:
            materialized = tuple(corners)
        except TypeError as exc:
            raise TypeError(f"{operation} corners must be iterable") from exc
        if len(materialized) not in allowed_counts:
            allowed_text = ", ".join(str(count) for count in allowed_counts)
            raise ValueError(
                f"{operation} requires {allowed_text} corners, got "
                f"{len(materialized)}"
            )
        if not materialized:
            return ()
        normalized = self._port.normalize_entities(
            materialized,
            operation=f"{operation} corners",
        )
        if any(corner.dimension != 0 for corner in normalized):
            raise ValueError(
                f"{operation} corners must be dimension-zero point references"
            )
        return normalized

    def _active_field_tags(self, operation: str) -> frozenset[int]:
        model_name = self._port.model_name

        def query_active_tags(model: Any) -> tuple[int, ...]:
            active_tags = _normalize_active_tags(model.mesh.field.list(), model_name)
            return tuple(sorted(active_tags))

        return frozenset(self._port.native_query(operation, query_active_tags))

    def _construct_field(
        self,
        field_type: _FieldType,
        operation: str,
        configure: _NativeFieldConfiguration,
    ) -> MeshFieldRef:
        model_name = self._port.model_name

        def allocate_and_configure(record_allocated: Callable[[Any], int]) -> None:
            def allocate(model: Any) -> None:
                manager = model.mesh.field
                allocated_tag = record_allocated(manager.add(field_type, tag=-1))
                configure(model, allocated_tag)
                active_tags = _normalize_active_tags(manager.list(), model_name)
                if allocated_tag not in active_tags:
                    raise GeometryError(
                        f"geometry model {model_name!r}: newly allocated "
                        f"{field_type} field {allocated_tag} is not active"
                    )

            self._port.native_control(operation, allocate)

        def rollback(field_tag: int) -> None:
            def remove(model: Any) -> None:
                model.mesh.field.remove(field_tag)

            self._port.native_control(operation, remove)

        return self._fields.construct(
            field_type,
            allocate_and_configure,
            rollback,
        )

    def _native_entity_pairs(
        self,
        dimension: int,
        operation: str,
    ) -> tuple[tuple[int, int], ...]:
        def query_entities(model: Any) -> tuple[tuple[int, int], ...]:
            return tuple(
                sorted(
                    _normalize_dim_tag(pair)
                    for pair in model.getEntities(dimension)
                )
            )

        return self._port.native_query(operation, query_entities)


def _validate_generated_top_cell_shape(
    model: Any,
    policy: _MeshGenerationPolicy,
    *,
    model_name: str,
    dimension: int,
) -> None:
    allowed_types = policy.allowed_top_cell_types
    if allowed_types is None or policy.resolved_cell_shape is None:
        raise GeometryStateError(
            f"geometry model {model_name!r}: strict mesh policy is incomplete"
        )

    raw_blocks = model.mesh.getElements(dimension)
    try:
        raw_types, raw_element_tags, _ = raw_blocks
        element_types = list(raw_types)
        element_tag_blocks = list(raw_element_tags)
    except Exception as exc:
        raise _mesh_cell_shape_error(
            policy,
            "generated malformed top-dimensional element blocks",
            model_name=model_name,
            dimension=dimension,
        ) from exc

    if len(element_types) != len(element_tag_blocks):
        raise _mesh_cell_shape_error(
            policy,
            "generated malformed top-dimensional element blocks "
            f"({len(element_types)} type blocks and "
            f"{len(element_tag_blocks)} element-tag blocks)",
            model_name=model_name,
            dimension=dimension,
        )

    counts: dict[int, int] = {}
    for block_index, (raw_type, raw_tags) in enumerate(
        zip(element_types, element_tag_blocks, strict=True)
    ):
        if isinstance(raw_type, bool):
            raise _mesh_cell_shape_error(
                policy,
                f"generated a non-integer element type in block {block_index}",
                model_name=model_name,
                dimension=dimension,
            )
        try:
            element_type = int(operator.index(raw_type))
        except (TypeError, ValueError, OverflowError) as exc:
            raise _mesh_cell_shape_error(
                policy,
                f"generated a non-integer element type in block {block_index}",
                model_name=model_name,
                dimension=dimension,
            ) from exc
        try:
            element_count = len(raw_tags)
        except Exception as exc:
            raise _mesh_cell_shape_error(
                policy,
                "generated malformed element tags in top-dimensional "
                f"block {block_index}",
                model_name=model_name,
                dimension=dimension,
            ) from exc
        if element_count:
            counts[element_type] = counts.get(element_type, 0) + element_count

    if not counts:
        raise _mesh_cell_shape_error(
            policy,
            "generated no top-dimensional cells",
            model_name=model_name,
            dimension=dimension,
        )
    if set(counts).issubset(allowed_types):
        return

    actual = " and ".join(
        f"{_element_type_diagnostic_name(model, element_type)}={counts[element_type]}"
        for element_type in sorted(counts)
    )
    raise _mesh_cell_shape_error(
        policy,
        f"generated {actual}",
        model_name=model_name,
        dimension=dimension,
    )


def _mesh_cell_shape_error(
    policy: _MeshGenerationPolicy,
    actual_detail: str,
    *,
    model_name: str,
    dimension: int,
) -> MeshCellShapeError:
    allowed_types = policy.allowed_top_cell_types or frozenset()
    expected_names = [
        _GMSH_TOP_CELL_TYPE_NAMES.get(
            element_type,
            f"Gmsh type {element_type}",
        )
        for element_type in sorted(allowed_types)
    ]
    expected = " or ".join(expected_names) if expected_names else "no cell type"
    requested = f"cell_shape={policy.requested_cell_shape!r}"
    if policy.requested_cell_shape is None:
        requested += f" (resolved to {policy.resolved_cell_shape!r})"
    return MeshCellShapeError(
        f"geometry model {model_name!r}: AutoMeshSpec requested "
        f"{requested} for dimension={dimension} and order={policy.order}; "
        f"expected only {expected}, but {actual_detail}; automatic fallback "
        "is disabled"
    )


def _element_type_diagnostic_name(model: Any, element_type: int) -> str:
    try:
        properties = model.mesh.getElementProperties(element_type)
        name = properties[0]
        if not isinstance(name, str) or not name:
            raise ValueError("element name is unavailable")
    except BaseException:
        return f"Gmsh type {element_type}"
    return name


def _normalize_dim_tag(value: Any) -> tuple[int, int]:
    try:
        dimension, tag = value
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"invalid Gmsh entity reference {value!r}") from exc
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension not in range(4)
    ):
        raise ValueError(
            "entity dimension must be an integer from 0 through 3, "
            f"got {dimension!r}"
        )
    return dimension, _validate_positive_tag(tag, "entity tag")


def _dim_tags(entities: Iterable[EntityRef]) -> tuple[tuple[int, int], ...]:
    return tuple((entity.dimension, entity.tag) for entity in entities)
