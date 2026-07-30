"""Headless native geometry-to-canonical-FEM-model workflow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import replace
import math
from typing import Any, Protocol
from collections.abc import Callable

from fem import geometry
from fem.core.mesh import Element2D, Element3D, Mesh2D, Mesh3D, Node2D, Node3D
from fem.core.model import FEMModel
from fem.geometry.measurements import resolve_target_radius
from fem.geometry.body_relations import require_meshable_body_relations
from fem.geometry.recipe_analysis import (
    recipe_characteristic_size,
    supports_structured_hexahedron,
)
from fem.geometry.references import LogicalEntityRef
from fem.geometry.gmsh_coordinator import PROCESS_GMSH_COORDINATOR
from fem.geometry.recipes import (
    MovedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    RotatedGeometry,
    SolidBody,
    WireGeometry,
    geometry_dimension,
)
from fem.geometry.part_namespace import (
    namespace_part_logical_id,
    part_id_from_logical_id,
    strip_part_reference,
)
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.mesh.settings import MeshSettings
from fem.selection import edges as mesh_edges
from fem.selection import faces as mesh_faces

from .recipe_compiler import (
    CompiledRecipeTopology,
    TopologyResolutionError,
    compile_recipe,
)
from .native_mesh_contract import (
    NativeMeshContract,
    require_complete_native_mesh_contract,
)
from .native_regions import validate_native_authoring_context
from .native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
    NATIVE_SCOPE_CATALOG_KEY,
    materialize_native_scopes,
)
from .revisions import MeshTaskSnapshot
from .native_part import NativePart, part_id_sort_key
from .definitions import MeshEntityRef


class RecipeTopologyResolver(Protocol):
    """Application seam for compiling and resolving one-session CAD topology."""

    def build(
        self,
        cad: Any,
        recipe: NativeGeometry,
    ) -> CompiledRecipeTopology:
        """Compile one recipe in the active CAD session."""
        ...

    def resolve(
        self,
        cad: Any,
        recipe: NativeGeometry,
        topology: CompiledRecipeTopology,
        reference: LogicalEntityRef,
    ) -> tuple[Any, ...]:
        """Resolve one logical entity to live CAD references."""
        ...

    def characteristic_size(self, recipe: NativeGeometry) -> float:
        """Return a representative dimension for structured meshing."""
        ...

    def supports_hexahedron(self, recipe: NativeGeometry) -> bool:
        """Return whether the recipe supports the structured Hex workflow."""
        ...


class LogicalRecipeTopologyResolver:
    """Default resolver backed by the tag-independent recipe topology catalog."""

    def build(
        self,
        cad: Any,
        recipe: NativeGeometry,
    ) -> CompiledRecipeTopology:
        return compile_recipe(cad, recipe)

    def resolve(
        self,
        cad: Any,
        recipe: NativeGeometry,
        topology: CompiledRecipeTopology,
        reference: LogicalEntityRef,
    ) -> tuple[Any, ...]:
        del cad, recipe
        return topology.resolve(reference)

    def characteristic_size(self, recipe: NativeGeometry) -> float:
        return recipe_characteristic_size(recipe)

    def supports_hexahedron(self, recipe: NativeGeometry) -> bool:
        return supports_structured_hexahedron(recipe)


DEFAULT_TOPOLOGY_RESOLVER: RecipeTopologyResolver = LogicalRecipeTopologyResolver()


def generate_fem_model(
    recipe_or_snapshot: NativeGeometry | MeshTaskSnapshot,
    settings: MeshSettings | None = None,
    *,
    named_regions: Iterable[Any] | Mapping[str, Any] | None = None,
    resolver: RecipeTopologyResolver | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> FEMModel:
    """Generate a canonical model from explicit inputs or a mesh task snapshot."""
    if (
        isinstance(recipe_or_snapshot, MeshTaskSnapshot)
        and recipe_or_snapshot.parts
        and all(
            type(part) is NativePart
            and part.geometry_recipe is not None
            for part in recipe_or_snapshot.parts
        )
    ):
        if settings is not None:
            raise TypeError(
                "MeshTaskSnapshot 已包含网格设置，不能重复传入 settings"
            )
        if named_regions is not None:
            raise TypeError(
                "MeshTaskSnapshot 已包含命名区域，不能重复传入 named_regions"
            )
        return _generate_multi_part_fem_model(
            recipe_or_snapshot,
            resolver=resolver,
            cancelled=cancelled,
        )
    recipe, mesh_settings, regions, model_name = _normalize_inputs(
        recipe_or_snapshot,
        settings,
        named_regions,
    )
    require_meshable_body_relations(recipe)
    contract = require_complete_native_mesh_contract(recipe, mesh_settings)
    if mesh_settings is None:
        raise TypeError("网格生成需要 MeshSettings")
    validate_native_authoring_context(
        recipe,
        regions,
        local_controls=mesh_settings.local_controls,
        mesh_settings=mesh_settings,
        mesh_contract=contract,
    )
    topology_resolver = resolver or DEFAULT_TOPOLOGY_RESOLVER
    wire_recipe = _wire_recipe(recipe)

    try:
        import gmsh
    except ModuleNotFoundError as error:
        if error.name != "gmsh":
            raise
        raise ModuleNotFoundError(
            "几何与网格功能需要 Gmsh；请安装项目的 cad 可选依赖"
        ) from error

    gmsh_lease = PROCESS_GMSH_COORDINATOR.acquire(
        f"generate FEM model {model_name}",
        cancelled=cancelled,
    )
    owns_session = False
    try:
        owns_session = not bool(gmsh.isInitialized())
        if owns_session:
            # Gmsh can install its SIGINT handler only on the Python main
            # thread. GUI callers intentionally execute this function in a
            # worker, so the application retains signal ownership.
            gmsh.initialize(interruptible=False)
        dimension = contract.dimension
        with geometry.model(model_name, dimension=dimension) as cad:
            topology = topology_resolver.build(cad, recipe)
            mesher = gmsh_meshing.Mesher(cad)
            if (
                wire_recipe is not None
                and contract.line_element_type == "Truss2"
            ):
                _configure_truss_member_mesh(
                    mesher,
                    wire_recipe,
                    topology,
                )
            if mesh_settings.cell_shape == "hexahedron":
                _configure_hexahedral_mesh(
                    cad,
                    mesher,
                    recipe,
                    mesh_settings,
                    topology,
                    topology_resolver,
                )

            mesh_size = mesh_settings.size
            refinements: list[Any] = []
            for control in mesh_settings.local_controls:
                entities = topology_resolver.resolve(
                    cad,
                    recipe,
                    topology,
                    control.target,
                )
                scale = (
                    mesh_settings.size
                    if control.falloff.reference == "global_size"
                    else resolve_target_radius(recipe, control.target)
                )
                distance = mesher.distance_field(
                    **_distance_field_sources(entities),
                    sampling=100,
                )
                refinements.append(
                    mesher.threshold_field(
                        distance,
                        size_min=control.size,
                        size_max=mesh_settings.size,
                        dist_min=scale * control.falloff.start_factor,
                        dist_max=scale * control.falloff.end_factor,
                    )
                )
            if refinements:
                background = (
                    refinements[0]
                    if len(refinements) == 1
                    else mesher.min_field(refinements)
                )
                mesher.background_field(background)
                mesh_size = None

            strict_shape = bool(mesh_settings.strict_cell_shape)
            if strict_shape and not refinements and mesh_settings.auto_level is None:
                mesher.mesh_size(cad.entities(0), size=mesh_settings.size)
                mesh_size = None
            if strict_shape or mesh_settings.auto_level is not None:
                native_mesh = mesher.generate(
                    gmsh_meshing.AutoMeshSpec(
                        # A background field already carries the absolute
                        # effective far-field size derived from AutoMesh
                        # level.  Level 3 keeps MeshSizeFactor at 1 and avoids
                        # scaling local sizes a second time.
                        level=(
                            3
                            if refinements
                            else mesh_settings.auto_level or 3
                        ),
                        cell_shape=_auto_mesh_cell_shape(contract),
                        order=mesh_settings.order,
                    )
                )
            else:
                native_mesh = mesher.generate(
                    gmsh_meshing.MeshSpec(
                        size=mesh_size,
                        order=mesh_settings.order,
                        recombine=contract.cell_shape
                        in {"quadrilateral", "hexahedron"},
                    )
                )
            mesh = gmsh_io.read(
                native_mesh,
                line_element_type=contract.line_element_type,
            )
            if wire_recipe is not None:
                _audit_native_wire_mesh(
                    wire_recipe,
                    topology,
                    native_mesh,
                    mesh,
                    contract,
                )
            model = _build_native_fem_model(
                mesh,
                native_mesh,
                model_name,
                dimension,
                topology,
            )
            if regions:
                model = materialize_native_scopes(
                    model,
                    previous_names=(),
                    regions=regions,
                )
            if dimension == 1 and (model.edges or model.surfaces):
                raise TopologyResolutionError(
                    "native wire model unexpectedly created FEM edge or surface "
                    "collections"
                )
            return model
    finally:
        try:
            if owns_session and bool(gmsh.isInitialized()):
                gmsh.finalize()
        finally:
            gmsh_lease.release()


def _auto_mesh_cell_shape(contract: NativeMeshContract) -> str | None:
    return {
        "line": None,
        "triangle": "tri",
        "quadrilateral": "quad",
        "tetrahedron": "tet",
        "hexahedron": "hex",
    }[str(contract.cell_shape)]


def _generate_multi_part_fem_model(
    snapshot: MeshTaskSnapshot,
    *,
    resolver: RecipeTopologyResolver | None,
    cancelled: Callable[[], bool] | None,
) -> FEMModel:
    """Compile, mesh, and deterministically aggregate active Parts."""

    active_parts = tuple(
        sorted(
            (part for part in snapshot.parts if not part.suppressed),
            key=lambda part: part_id_sort_key(part.id),
        )
    )
    if not active_parts:
        raise ValueError("网格生成至少需要一个未抑制部件")
    dimensions = {
        geometry_dimension(part.geometry_recipe)
        for part in active_parts
    }
    if len(dimensions) != 1:
        details = "、".join(
            f"{part.name} [{part.id}]={part.dimension}D"
            for part in active_parts
        )
        raise ValueError(
            "analysis.mixed-dimension.unsupported: "
            f"当前分析不支持混合维度部件（{details}）"
        )
    dimension = next(iter(dimensions))
    if dimension == 3 and len(active_parts) > 1:
        relation_recipe = MultiBodyGeometry(
            "部件关系预检",
            tuple(
                SolidBody(f"B{index}", part.name, part.geometry_recipe)
                for index, part in enumerate(active_parts, start=1)
            ),
        )
        try:
            require_meshable_body_relations(relation_recipe)
        except ValueError as error:
            raise type(error)(
                str(error)
                .replace("body.overlap.mesh-blocked", "part.overlap.mesh-blocked")
                .replace("Body", "部件")
            ) from error

    per_part_models: list[tuple[NativePart, FEMModel]] = []
    for part in active_parts:
        if part.mesh_settings is None:
            raise TypeError(
                f"部件 {part.name} [{part.id}] 缺少网格设置"
            )
        local_settings = _localize_part_mesh_settings(
            part.id,
            part.mesh_settings,
        )
        require_complete_native_mesh_contract(
            part.geometry_recipe,
            local_settings,
        )
        per_part_models.append(
            (
                part,
                generate_fem_model(
                    part.geometry_recipe,
                    local_settings,
                    named_regions=(),
                    resolver=resolver,
                    cancelled=cancelled,
                ),
            )
        )

    aggregate = _aggregate_part_models(
        str(snapshot.model_name),
        per_part_models,
    )
    active_regions = _active_part_regions(
        snapshot.named_regions,
        frozenset(part.id for part in active_parts),
    )
    if active_regions:
        aggregate = materialize_native_scopes(
            aggregate,
            previous_names=(),
            regions=active_regions,
        )
    return aggregate


def _active_part_regions(
    regions: Iterable[Any],
    active_part_ids: frozenset[str],
) -> tuple[Any, ...]:
    """Project preserved region definitions onto the meshed Part set."""

    projected: list[Any] = []
    for region in regions:
        references = tuple(getattr(region, "references", ()))
        retained = tuple(
            reference
            for reference in references
            if (
                type(reference) is LogicalEntityRef
                and part_id_from_logical_id(reference.logical_id)
                in active_part_ids
            )
            or (
                type(reference) is MeshEntityRef
                and (
                    reference.part_id is None
                    or reference.part_id in active_part_ids
                )
            )
        )
        if retained:
            projected.append(replace(region, references=retained))
    return tuple(projected)


def _localize_part_mesh_settings(
    part_id: str,
    settings: MeshSettings,
) -> MeshSettings:
    controls = tuple(
        replace(
            control,
            target=(
                strip_part_reference(part_id, control.target)
                if part_id_from_logical_id(control.target.logical_id)
                is not None
                else control.target
            ),
        )
        for control in settings.local_controls
    )
    return replace(settings, local_controls=controls)


def _aggregate_part_models(
    model_name: str,
    entries: Iterable[tuple[NativePart, FEMModel]],
) -> FEMModel:
    """Merge Part meshes without merging coincident nodes."""

    rows = tuple(entries)
    if not rows:
        raise ValueError("cannot aggregate an empty Part collection")
    first_mesh = rows[0][1].mesh
    mesh_type = type(first_mesh)
    if mesh_type not in {Mesh2D, Mesh3D}:
        raise TypeError("unsupported native mesh container")
    if any(type(model.mesh) is not mesh_type for _part, model in rows):
        raise ValueError("analysis.mixed-dimension.unsupported")
    if any(
        int(model.mesh.dofs_per_node) != int(first_mesh.dofs_per_node)
        for _part, model in rows
    ):
        raise ValueError("部件网格的每节点自由度不一致")

    nodes: list[Node2D | Node3D] = []
    elements: list[Element2D | Element3D] = []
    catalog: dict[str, dict[str, Any]] = {}
    ownership: dict[str, dict[str, tuple[int, ...]]] = {}
    next_node_id = 1
    next_element_id = 1

    for part, model in rows:
        local_nodes = tuple(
            sorted(model.mesh.nodes, key=lambda node: int(node.id))
        )
        local_elements = tuple(
            sorted(model.mesh.elements, key=lambda element: int(element.id))
        )
        node_map = {
            int(node.id): next_node_id + index
            for index, node in enumerate(local_nodes)
        }
        element_map = {
            int(element.id): next_element_id + index
            for index, element in enumerate(local_elements)
        }
        for node in local_nodes:
            owned_node = deepcopy(node)
            owned_node.id = node_map[int(node.id)]
            nodes.append(owned_node)
        for element in local_elements:
            owned_element = deepcopy(element)
            owned_element.id = element_map[int(element.id)]
            owned_element.node_ids = [
                node_map[int(node_id)] for node_id in element.node_ids
            ]
            elements.append(owned_element)

        source_catalog = model.metadata.get(NATIVE_SCOPE_CATALOG_KEY, {})
        if not isinstance(source_catalog, Mapping):
            raise TypeError("native Part model lacks a scope catalog")
        for logical_id, raw in source_catalog.items():
            if not isinstance(raw, Mapping):
                raise TypeError("native scope catalog entry must be a mapping")
            namespaced = namespace_part_logical_id(
                part.id,
                str(logical_id),
            )
            catalog[namespaced] = _renumber_scope_catalog_entry(
                raw,
                node_map,
                element_map,
            )
        ownership[part.id] = {
            "node_ids": tuple(node_map.values()),
            "element_ids": tuple(element_map.values()),
            "cell_ids": tuple(element_map.values()),
        }
        next_node_id += len(local_nodes)
        next_element_id += len(local_elements)

    aggregate_mesh = mesh_type(
        nodes=nodes,
        elements=elements,
        dofs_per_node=int(first_mesh.dofs_per_node),
    )
    aggregate = FEMModel(mesh=aggregate_mesh, name=model_name)
    aggregate.metadata[NATIVE_SCOPE_CATALOG_KEY] = catalog
    aggregate.metadata[NATIVE_PART_OWNERSHIP_KEY] = ownership
    return aggregate


def _renumber_scope_catalog_entry(
    raw: Mapping[str, Any],
    node_map: Mapping[int, int],
    element_map: Mapping[int, int],
) -> dict[str, Any]:
    def node_id(value: Any) -> int:
        return node_map[int(value)]

    def element_id(value: Any) -> int:
        return element_map[int(value)]

    return {
        "kind": str(raw.get("kind", "")),
        "node_ids": tuple(
            node_id(value) for value in raw.get("node_ids", ())
        ),
        "element_ids": tuple(
            element_id(value) for value in raw.get("element_ids", ())
        ),
        "edges": tuple(
            (
                element_id(element),
                int(local_index),
                tuple(node_id(value) for value in node_ids),
            )
            for element, local_index, node_ids in raw.get("edges", ())
        ),
        "faces": tuple(
            (
                element_id(element),
                int(local_index),
                tuple(node_id(value) for value in node_ids),
            )
            for element, local_index, node_ids in raw.get("faces", ())
        ),
    }


def _normalize_inputs(
    recipe_or_snapshot: NativeGeometry | MeshTaskSnapshot,
    settings: MeshSettings | None,
    named_regions: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[
    NativeGeometry,
    MeshSettings | None,
    tuple[Any, ...],
    str,
]:
    if isinstance(recipe_or_snapshot, MeshTaskSnapshot):
        if settings is not None:
            raise TypeError("MeshTaskSnapshot 已包含网格设置，不能重复传入 settings")
        if named_regions is not None:
            raise TypeError(
                "MeshTaskSnapshot 已包含命名区域，不能重复传入 named_regions"
            )
        recipe = recipe_or_snapshot.geometry_recipe
        mesh_settings = recipe_or_snapshot.mesh_settings
        region_source = recipe_or_snapshot.named_regions
        model_name = str(recipe_or_snapshot.model_name).strip()
    else:
        recipe = recipe_or_snapshot
        mesh_settings = settings
        region_source = () if named_regions is None else named_regions
        model_name = str(getattr(recipe, "name", "")).strip()

    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError("网格生成需要有效的原生几何配方")
    if mesh_settings is not None and type(mesh_settings) is not MeshSettings:
        raise TypeError("网格生成需要 MeshSettings")
    if not model_name:
        raise ValueError("网格生成需要非空模型名称")
    regions = (
        tuple(region_source.values())
        if isinstance(region_source, Mapping)
        else tuple(region_source)
    )
    return recipe, mesh_settings, regions, model_name


def _wire_recipe(recipe: NativeGeometry) -> WireGeometry | None:
    current: object = recipe
    while isinstance(current, (MovedGeometry, RotatedGeometry)):
        current = current.base
    return current if isinstance(current, WireGeometry) else None


def _configure_truss_member_mesh(
    mesher: Any,
    recipe: WireGeometry,
    topology: CompiledRecipeTopology,
) -> None:
    """Generate exactly one Truss2 element for every declared member.

    Subdividing a straight spatial truss member introduces collinear internal
    nodes whose transverse translational DOFs have zero stiffness.  The
    current linear solver therefore treats the authored Wire graph itself as
    the truss discretization.
    """

    for member in recipe.members:
        logical_id = f"edge:{member.name}"
        curves = _unique_entities(
            topology.logical_entities.get(logical_id, ())
        )
        if len(curves) != 1 or curves[0].dimension != 1:
            raise TopologyResolutionError(
                f"{logical_id} must resolve to exactly one CAD curve for "
                "Truss2 member meshing"
            )
        mesher.transfinite_curve(curves[0], num_nodes=2)


def _configure_hexahedral_mesh(
    cad: Any,
    mesher: Any,
    recipe: NativeGeometry,
    settings: MeshSettings,
    topology: CompiledRecipeTopology,
    resolver: RecipeTopologyResolver,
) -> None:
    if not resolver.supports_hexahedron(recipe):
        raise ValueError("六面体结构化网格当前仅支持长方体或矩形草图拉伸体")
    curves = _unique_entities(
        entity
        for entity in cad.boundary(topology.boundary, combined=False)
        if entity.dimension == 1
    )
    node_count = max(
        2,
        int(math.ceil(resolver.characteristic_size(recipe) / settings.size)) + 1,
    )
    for curve in curves:
        mesher.transfinite_curve(curve, num_nodes=node_count)
    for surface in topology.boundary:
        mesher.transfinite_surface(surface)
        mesher.recombine(surface)
    if len(topology.domain) != 1:
        raise TopologyResolutionError("结构化六面体网格要求唯一计算域")
    mesher.transfinite_volume(topology.domain[0])


def _distance_field_sources(
    entities: Iterable[Any],
) -> Mapping[str, tuple[Any, ...]]:
    resolved = _unique_entities(entities)
    if not resolved:
        raise TopologyResolutionError("局部网格控制没有解析到几何实体")
    dimensions = {entity.dimension for entity in resolved}
    if len(dimensions) != 1:
        raise TopologyResolutionError("局部网格控制不能混用不同维度的几何实体")
    dimension = next(iter(dimensions))
    source_name = {0: "points", 1: "curves", 2: "surfaces"}.get(dimension)
    if source_name is None:
        raise TopologyResolutionError("局部网格控制只支持点、边或面")
    return {source_name: resolved}


def _build_native_fem_model(
    mesh: Any,
    native_mesh: Any,
    name: str,
    dimension: int,
    topology: CompiledRecipeTopology,
) -> FEMModel:
    """Return a generated model without any implicit user-facing scopes."""

    model = FEMModel(mesh=mesh, name=name)
    model.metadata[NATIVE_SCOPE_CATALOG_KEY] = _native_scope_catalog(
        mesh,
        native_mesh,
        dimension,
        topology,
    )
    return model


def _native_scope_catalog(
    mesh: Any,
    native_mesh: Any,
    dimension: int,
    topology: CompiledRecipeTopology,
) -> dict[str, dict[str, Any]]:
    """Map selectable CAD topology to mesh entities without creating scopes."""

    boundary_edges = (
        tuple(mesh_edges.boundary(mesh))
        if dimension == 2
        else _unique_element_edges(mesh_edges.all(mesh))
        if dimension == 3
        else ()
    )
    boundary_faces = (
        tuple(mesh_faces.boundary(mesh))
        if dimension == 3
        else ()
    )
    catalog: dict[str, dict[str, Any]] = {}
    mesh_node_ids = {int(node.id) for node in mesh.nodes}
    for logical_id, raw_entities in topology.logical_entities.items():
        entities = _unique_entities(raw_entities)
        if not entities:
            continue
        entity_dimensions = {int(entity.dimension) for entity in entities}
        if len(entity_dimensions) != 1:
            raise TopologyResolutionError(
                f"{logical_id} resolves to mixed CAD dimensions"
            )
        entity_dimension = next(iter(entity_dimensions))
        # Gmsh also meshes construction-only CAD points (such as hole
        # centers) whose nodes belong to no domain element and are dropped
        # from the FEM mesh, so the catalog keeps mesh-backed nodes only.
        node_ids = tuple(
            int(node_id)
            for node_id in gmsh_io.entity_node_ids(native_mesh, entities)
            if int(node_id) in mesh_node_ids
        )
        node_id_set = set(node_ids)
        edge_rows = (
            tuple(
                (
                    int(element_id),
                    int(local_index),
                    tuple(int(node_id) for node_id in row_node_ids),
                )
                for element_id, local_index, row_node_ids in boundary_edges
                if set(row_node_ids).issubset(node_id_set)
            )
            if entity_dimension == 1 and dimension >= 2
            else ()
        )
        face_rows = (
            tuple(
                (
                    int(element_id),
                    int(local_index),
                    tuple(int(node_id) for node_id in row_node_ids),
                )
                for element_id, local_index, row_node_ids in boundary_faces
                if set(row_node_ids).issubset(node_id_set)
            )
            if entity_dimension == 2 and dimension == 3
            else ()
        )
        catalog[str(logical_id)] = {
            "kind": str(logical_id).partition(":")[0],
            "node_ids": node_ids,
            "element_ids": (
                gmsh_io.entity_element_ids(native_mesh, entities)
                if entity_dimension == dimension
                else ()
            ),
            "edges": edge_rows,
            "faces": face_rows,
        }
    return catalog


def _unique_element_edges(
    rows: Iterable[tuple[int, int, Iterable[int]]],
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Keep one deterministic element-local representative per mesh edge."""

    selected: dict[
        tuple[int, int],
        tuple[int, int, tuple[int, ...]],
    ] = {}
    for element_id, local_index, raw_node_ids in rows:
        node_ids = tuple(int(node_id) for node_id in raw_node_ids)
        if len(node_ids) < 2:
            continue
        key = tuple(sorted((node_ids[0], node_ids[-1])))
        candidate = (int(element_id), int(local_index), node_ids)
        current = selected.get(key)
        if current is None or candidate[:2] < current[:2]:
            selected[key] = candidate
    return tuple(selected[key] for key in sorted(selected))


def _audit_native_wire_mesh(
    recipe: WireGeometry,
    topology: CompiledRecipeTopology,
    native_mesh: Any,
    mesh: Any,
    contract: NativeMeshContract,
) -> None:
    """Prove that imported topology still matches the declared wire graph."""

    elements = tuple(getattr(mesh, "elements", ()))
    element_by_id: dict[int, Any] = {}
    for element in elements:
        element_id = int(element.id)
        if element_id in element_by_id:
            raise TopologyResolutionError(
                f"body:domain contains duplicate imported element ID {element_id!r}"
            )
        element_by_id[element_id] = element
    element_ids = set(element_by_id)
    if not elements:
        raise TopologyResolutionError("body:domain 没有生成任何线单元")
    unexpected = tuple(
        element.type
        for element in elements
        if element.type != contract.canonical_element_type
    )
    if unexpected:
        raise TopologyResolutionError(
            f"body:domain 包含未预期的单元类型 {unexpected[0]!r}，"
            f"要求 {contract.canonical_element_type!r}"
        )

    point_nodes: dict[str, int] = {}
    for point in recipe.points:
        logical_id = f"point:{point.name}"
        entities = topology.logical_entities.get(logical_id, ())
        node_ids = gmsh_io.entity_node_ids(native_mesh, entities)
        if len(node_ids) != 1:
            raise TopologyResolutionError(
                f"{logical_id} 应解析为恰好一个网格节点，实际为 {node_ids!r}"
            )
        point_nodes[logical_id] = int(node_ids[0])
    if len(set(point_nodes.values())) != len(point_nodes):
        duplicate = _first_duplicate_value(point_nodes)
        raise TopologyResolutionError(
            f"{duplicate} 与另一个逻辑点共享网格节点，声明的图身份被改变"
        )

    member_element_ids: dict[str, set[int]] = {}
    member_node_ids: dict[str, set[int]] = {}
    member_endpoint_nodes: dict[str, set[int]] = {}
    for member in recipe.members:
        logical_id = f"edge:{member.name}"
        entities = topology.logical_entities.get(logical_id, ())
        ids = set(gmsh_io.entity_element_ids(native_mesh, entities))
        if not ids:
            raise TopologyResolutionError(f"{logical_id} 没有生成任何线单元")
        if not ids.issubset(element_ids):
            raise TopologyResolutionError(
                f"{logical_id} 解析出了不属于导入模型的单元 ID"
            )
        member_element_ids[logical_id] = ids
        for endpoint in (member.start, member.end):
            endpoint_id = point_nodes[f"point:{endpoint}"]
            if not any(
                endpoint_id in tuple(element.node_ids)
                for element in elements
                if int(element.id) in ids
            ):
                raise TopologyResolutionError(
                    f"{logical_id} 的端点 point:{endpoint} 未出现在其单元链中"
                )

    members_by_logical_id = {
        f"edge:{member.name}": member for member in recipe.members
    }
    for logical_id, ids in member_element_ids.items():
        member = members_by_logical_id[logical_id]
        endpoint_nodes = {
            point_nodes[f"point:{member.start}"],
            point_nodes[f"point:{member.end}"],
        }
        member_endpoint_nodes[logical_id] = endpoint_nodes
        adjacency: dict[int, set[int]] = {}
        degree: dict[int, int] = {}
        for element_id in ids:
            element = element_by_id[element_id]
            try:
                node_ids = tuple(int(node_id) for node_id in element.node_ids)
            except (TypeError, ValueError) as error:
                raise TopologyResolutionError(
                    f"{logical_id} contains an imported element with invalid node IDs"
                ) from error
            if len(node_ids) != 2 or node_ids[0] == node_ids[1]:
                raise TopologyResolutionError(
                    f"{logical_id} must resolve to first-order two-node line elements"
                )
            first, second = node_ids
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
            degree[first] = degree.get(first, 0) + 1
            degree[second] = degree.get(second, 0) + 1
        visited: set[int] = set()
        pending = [next(iter(adjacency))]
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            pending.extend(adjacency[node_id] - visited)
        if visited != set(adjacency):
            raise TopologyResolutionError(
                f"{logical_id} is fragmented into disconnected element chains"
            )
        degree_one = {node_id for node_id, count in degree.items() if count == 1}
        if degree_one != endpoint_nodes or any(
            count > 2 for count in degree.values()
        ):
            raise TopologyResolutionError(
                f"{logical_id} is not one endpoint-to-endpoint line chain"
            )
        member_nodes = set(adjacency)
        internal_nodes = member_nodes - endpoint_nodes
        if internal_nodes.intersection(point_nodes.values()):
            raise TopologyResolutionError(
                f"{logical_id} contains a declared point as an undeclared internal node"
            )
        member_node_ids[logical_id] = member_nodes

    element_owners: dict[int, str] = {}
    for logical_id, ids in member_element_ids.items():
        for element_id in ids:
            previous = element_owners.get(element_id)
            if previous is not None:
                raise TopologyResolutionError(
                    f"{logical_id} shares imported element ID {element_id!r} "
                    f"with {previous}"
                )
            element_owners[element_id] = logical_id
    member_items = tuple(member_node_ids.items())
    for index, (logical_id, node_ids) in enumerate(member_items):
        for other_id, other_node_ids in member_items[index + 1 :]:
            shared_nodes = node_ids.intersection(other_node_ids)
            allowed_nodes = member_endpoint_nodes[logical_id].intersection(
                member_endpoint_nodes[other_id]
            )
            unexpected_nodes = shared_nodes - allowed_nodes
            if unexpected_nodes:
                raise TopologyResolutionError(
                    f"{logical_id} and {other_id} share undeclared mesh nodes "
                    f"{sorted(unexpected_nodes)!r}"
                )

    union = set().union(*member_element_ids.values())
    if union != element_ids:
        missing = sorted(element_ids - union)
        extra = sorted(union - element_ids)
        raise TopologyResolutionError(
            "body:domain 与声明 member 单元并集不一致："
            f"missing={missing!r}, extra={extra!r}"
        )
    domain_entities = topology.logical_entities.get("body:domain", ())
    domain_ids = set(gmsh_io.entity_element_ids(native_mesh, domain_entities))
    if domain_ids != union:
        raise TopologyResolutionError(
            "body:domain 的 CAD 所有权与声明 member 单元并集不一致"
        )
    if contract.line_element_type == "Truss2":
        subdivided = tuple(
            logical_id
            for logical_id, ids in member_element_ids.items()
            if len(ids) != 1
        )
        if subdivided:
            raise TopologyResolutionError(
                "native Truss2 requires exactly one element per declared "
                "Wire member; unexpected subdivisions: "
                + ", ".join(subdivided)
            )


def _first_duplicate_value(values: Mapping[str, int]) -> str:
    seen: dict[int, str] = {}
    for logical_id, node_id in values.items():
        previous = seen.get(node_id)
        if previous is not None:
            return f"{logical_id} 与 {previous}"
        seen[node_id] = logical_id
    return "逻辑点"


def _unique_entities(entities: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(entities))


__all__ = [
    "DEFAULT_TOPOLOGY_RESOLVER",
    "LogicalRecipeTopologyResolver",
    "RecipeTopologyResolver",
    "TopologyResolutionError",
    "generate_fem_model",
]
