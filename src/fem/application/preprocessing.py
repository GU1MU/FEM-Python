"""Headless native geometry-to-canonical-FEM-model workflow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any, Protocol

from fem import geometry
from fem.core.model import (
    Edge,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    NodeSet,
    Surface,
)
from fem.geometry.measurements import resolve_target_radius
from fem.geometry.recipe_analysis import (
    recipe_characteristic_size,
    supports_structured_hexahedron,
)
from fem.geometry.references import LogicalEntityRef
from fem.geometry.recipes import (
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    RotatedGeometry,
    WireGeometry,
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
from .native_regions import (
    CompiledDomainRegionSource,
    LogicalReferencesRegionSource,
    NativeRegionDescriptor,
    RecipeRegionSource,
    validate_native_authoring_context,
)
from .revisions import MeshTaskSnapshot


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
) -> FEMModel:
    """Generate a canonical model from explicit inputs or a mesh task snapshot."""
    recipe, mesh_settings, regions = _normalize_inputs(
        recipe_or_snapshot,
        settings,
        named_regions,
    )
    contract = require_complete_native_mesh_contract(recipe, mesh_settings)
    if mesh_settings is None:
        raise TypeError("网格生成需要 MeshSettings")
    region_descriptors = validate_native_authoring_context(
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

    owns_session = not bool(gmsh.isInitialized())
    try:
        if owns_session:
            # Gmsh can install its SIGINT handler only on the Python main
            # thread. GUI callers intentionally execute this function in a
            # worker, so the application retains signal ownership.
            gmsh.initialize(interruptible=False)
        dimension = contract.dimension
        with geometry.model(recipe.name, dimension=dimension) as cad:
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

            entity_groups = {
                descriptor.name: _resolve_region_descriptor(
                    cad,
                    recipe,
                    topology,
                    topology_resolver,
                    descriptor,
                )
                for descriptor in region_descriptors
            }

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
                recipe.name,
                dimension,
                entity_groups,
            )
            if dimension == 1 and (model.edges or model.surfaces):
                raise TopologyResolutionError(
                    "native wire model unexpectedly created FEM edge or surface "
                    "collections"
                )
            return model
    finally:
        if owns_session and bool(gmsh.isInitialized()):
            gmsh.finalize()


def _normalize_inputs(
    recipe_or_snapshot: NativeGeometry | MeshTaskSnapshot,
    settings: MeshSettings | None,
    named_regions: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[NativeGeometry, MeshSettings | None, tuple[Any, ...]]:
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
    else:
        recipe = recipe_or_snapshot
        mesh_settings = settings
        region_source = () if named_regions is None else named_regions

    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError("网格生成需要有效的原生几何配方")
    if mesh_settings is not None and type(mesh_settings) is not MeshSettings:
        raise TypeError("网格生成需要 MeshSettings")
    regions = (
        tuple(region_source.values())
        if isinstance(region_source, Mapping)
        else tuple(region_source)
    )
    return recipe, mesh_settings, regions


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


def _resolve_region_descriptor(
    cad: Any,
    recipe: NativeGeometry,
    topology: CompiledRecipeTopology,
    resolver: RecipeTopologyResolver,
    descriptor: NativeRegionDescriptor,
) -> tuple[Any, ...]:
    source = descriptor.source
    if isinstance(source, CompiledDomainRegionSource):
        return tuple(topology.domain)
    if isinstance(source, RecipeRegionSource):
        entities = tuple(topology.region_bindings.get(source.selector, ()))
        if not entities:
            raise TopologyResolutionError(
                f"内建区域 {descriptor.name!r} 没有对应的 CAD 实体"
            )
        return entities
    if not isinstance(source, LogicalReferencesRegionSource):
        raise TypeError(
            f"不支持的 native region source: {type(source).__name__}"
        )
    entities: list[Any] = []
    for reference in source.references:
        entities.extend(
            resolver.resolve(
                cad,
                recipe,
                topology,
                reference,
            )
        )
    resolved = _unique_entities(entities)
    if not resolved:
        raise TopologyResolutionError(
            f"命名区域 {descriptor.name!r} 没有解析到 CAD 实体"
        )
    return resolved


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
    entity_groups: Mapping[str, tuple[Any, ...]],
) -> FEMModel:
    """Convert CAD entity groups to FEM sets without leaking CAD identifiers."""
    model = FEMModel(mesh=mesh, name=name)
    boundary_edges = mesh_edges.boundary(mesh) if dimension == 2 else ()
    boundary_faces = mesh_faces.boundary(mesh) if dimension == 3 else ()

    for group_name, entities in entity_groups.items():
        if not entities:
            continue
        entity_dimension = entities[0].dimension
        if any(entity.dimension != entity_dimension for entity in entities):
            raise ValueError(f"几何区域 {group_name} 包含不同维度的实体")
        if entity_dimension == dimension:
            element_ids = gmsh_io.entity_element_ids(native_mesh, entities)
            model.element_sets[group_name] = ElementSet(
                group_name,
                element_ids,
            )
            continue

        node_ids = gmsh_io.entity_node_ids(native_mesh, entities)
        model.node_sets[group_name] = NodeSet(group_name, node_ids)
        node_id_set = set(node_ids)
        if dimension == 2 and entity_dimension == 1:
            model.edges[group_name] = Edge(
                group_name,
                [
                    ElementEdge(
                        element_id,
                        local_edge,
                        edge_node_ids,
                    )
                    for (
                        element_id,
                        local_edge,
                        edge_node_ids,
                    ) in boundary_edges
                    if set(edge_node_ids).issubset(node_id_set)
                ],
            )
        elif dimension == 3 and entity_dimension == 2:
            model.surfaces[group_name] = Surface(
                group_name,
                [
                    ElementFace(
                        element_id,
                        local_face,
                        face_node_ids,
                    )
                    for (
                        element_id,
                        local_face,
                        face_node_ids,
                    ) in boundary_faces
                    if set(face_node_ids).issubset(node_id_set)
                ],
            )
    return model


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
