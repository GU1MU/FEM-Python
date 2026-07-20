import ast
import importlib.util
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
GEOMETRY_ROOT = SRC_ROOT / "fem" / "geometry"

GEOMETRY_PUBLIC_API = [
    "BooleanResult",
    "CurveLoopRef",
    "EntityOwnershipError",
    "EntityRef",
    "FeatureResult",
    "GeometryError",
    "GeometryModel",
    "GeometryStateError",
    "LoftContinuity",
    "LoftParametrization",
    "LoftResult",
    "OrientedCurveRef",
    "StaleEntityError",
    "SweepFrame",
    "WireRef",
    "model",
]

GMSH_MESH_PUBLIC_API = [
    "AutoMeshSpec",
    "GmshMeshRef",
    "MeshCellShapeError",
    "MeshControlConflictError",
    "MeshFieldOwnershipError",
    "MeshFieldRef",
    "MeshSpec",
    "Mesher",
    "StaleGmshMeshError",
    "StaleMeshFieldError",
]


def _string_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value.replace("\\", "/")


def _resolved_import_targets(path, module_name):
    """Yield resolved import targets and source lines for a project module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_parts = module_name.split(".")
    package_parts = (
        module_parts if path.name == "__init__.py" else module_parts[:-1]
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            parent_count = node.level - 1
            base_parts = package_parts[: len(package_parts) - parent_count]
            if node.module:
                base_parts += node.module.split(".")
            base = ".".join(base_parts)
        else:
            base = node.module or ""

        for alias in node.names:
            target = f"{base}.{alias.name}" if base else alias.name
            yield target, node.lineno


def _module_name(path):
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _top_level_definition_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            continue
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _is_standard_library_or_gmsh(target):
    root = target.split(".", 1)[0]
    return root == "gmsh" or root in sys.stdlib_module_names


def _project_module_name(path):
    if path.is_relative_to(SRC_ROOT):
        return _module_name(path)
    return ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)


def _call_name(node):
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _nodes_in_scope(scope):
    nested_scopes = (
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.FunctionDef,
        ast.Lambda,
    )
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, nested_scopes):
            continue
        yield child
        yield from _nodes_in_scope(child)


def _node_start(node):
    return node.lineno, node.col_offset


def _node_end(node):
    return node.end_lineno, node.end_col_offset


def _simple_assignment_names(target):
    if isinstance(target, ast.Name):
        yield target.id


def _scope_assignment_events(nodes):
    events = {}
    for node in nodes:
        assignments = ()
        if isinstance(node, ast.Assign):
            assignments = tuple(
                (name, node.value)
                for target in node.targets
                for name in _simple_assignment_names(target)
            )
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                assignments = tuple(
                    (name, node.value)
                    for name in _simple_assignment_names(node.target)
                )
        elif isinstance(node, ast.NamedExpr):
            assignments = tuple(
                (name, node.value)
                for name in _simple_assignment_names(node.target)
            )

        for name, value in assignments:
            events.setdefault(name, []).append(
                (_node_end(node), _node_start(node), value)
            )

    for values in events.values():
        values.sort(key=lambda item: item[0])
    return events


def _is_extrusion_result(expression, events, position, resolving=()):
    while isinstance(expression, (ast.Await, ast.NamedExpr)):
        expression = expression.value

    if _call_name(expression) in {"extrude", "structured_extrude"}:
        return True
    if isinstance(expression, ast.IfExp):
        return _is_extrusion_result(
            expression.body, events, position, resolving
        ) or _is_extrusion_result(
            expression.orelse, events, position, resolving
        )
    if not isinstance(expression, ast.Name) or expression.id in resolving:
        return False

    prior_events = [
        event for event in events.get(expression.id, ()) if event[0] < position
    ]
    if not prior_events:
        return False
    _, evaluation_position, value = prior_events[-1]
    return _is_extrusion_result(
        value,
        events,
        evaluation_position,
        (*resolving, expression.id),
    )


def _extrusion_tuple_consumers(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope_types = (
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.FunctionDef,
        ast.Lambda,
    )
    scopes = (tree, *(node for node in ast.walk(tree) if isinstance(node, scope_types)))
    iterator_calls = {
        "all",
        "any",
        "enumerate",
        "filter",
        "frozenset",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "reversed",
        "set",
        "sorted",
        "sum",
        "tuple",
        "zip",
    }
    offenders = set()

    for scope in scopes:
        nodes = tuple(_nodes_in_scope(scope))
        events = _scope_assignment_events(nodes)

        def is_result(expression):
            return _is_extrusion_result(
                expression,
                events,
                _node_start(expression),
            )

        for node in nodes:
            reason = None
            if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                if is_result(node.iter):
                    reason = "direct iteration"
            elif isinstance(node, ast.Assign):
                if is_result(node.value) and any(
                    isinstance(target, (ast.List, ast.Tuple))
                    for target in node.targets
                ):
                    reason = "direct unpacking"
            elif isinstance(node, ast.Starred) and is_result(node.value):
                reason = "starred iteration"
            elif isinstance(node, ast.Subscript) and is_result(node.value):
                reason = "tuple-style indexing"
            elif isinstance(node, ast.YieldFrom) and is_result(node.value):
                reason = "yield-from iteration"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in iterator_calls and any(
                    is_result(argument) for argument in node.args
                ):
                    reason = f"{node.func.id}() iteration"
            elif isinstance(node, ast.Compare):
                for operator, comparator in zip(
                    node.ops, node.comparators, strict=True
                ):
                    if isinstance(operator, (ast.In, ast.NotIn)) and is_result(
                        comparator
                    ):
                        reason = "membership iteration"
                        break

            if reason is not None:
                offenders.add((node.lineno, node.col_offset, reason))

    return sorted(offenders)


def test_tests_do_not_reference_example_data_or_results_outputs():
    offenders = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        for literal in _string_literals(path):
            if literal == "examples" or literal.startswith("examples/"):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {literal}")
            if literal == "results" or literal.startswith("results/"):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {literal}")

    assert offenders == []


def test_geometry_package_has_no_reverse_or_third_party_runtime_dependencies():
    paths = sorted(GEOMETRY_ROOT.rglob("*.py"))
    assert paths

    offenders = []
    for path in paths:
        for target, lineno in _resolved_import_targets(path, _module_name(path)):
            internal_geometry_import = target == "fem.geometry" or target.startswith(
                "fem.geometry."
            )
            if internal_geometry_import or _is_standard_library_or_gmsh(target):
                continue
            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
            )

    assert offenders == []


def test_geometry_stateless_modules_follow_explicit_dependency_boundaries():
    module_dependencies = {
        "fem.geometry.errors": (),
        "fem.geometry.types": (
            "fem.geometry.errors",
            "fem.geometry._validation",
        ),
        "fem.geometry._validation": (),
        "fem.geometry._gmsh.constants": (),
        "fem.geometry._gmsh.predicates": (
            "fem.geometry._gmsh.constants",
        ),
    }
    missing = []
    offenders = []
    for module_name, allowed_modules in module_dependencies.items():
        relative = Path(*module_name.split(".")).with_suffix(".py")
        path = SRC_ROOT / relative
        if not path.is_file():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
            continue
        for target, lineno in _resolved_import_targets(path, module_name):
            root = target.split(".", 1)[0]
            if root in sys.stdlib_module_names or any(
                target == allowed or target.startswith(f"{allowed}.")
                for allowed in allowed_modules
            ):
                continue
            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
            )

    assert missing == []
    assert offenders == []


def test_geometry_stateful_modules_follow_explicit_dependency_boundaries():
    module_dependencies = {
        "fem.geometry._gmsh.state": (
            "fem.geometry.errors",
        ),
        "fem.geometry._gmsh.session": (
            "fem.geometry._gmsh.backend",
            "fem.geometry.errors",
        ),
        "fem.geometry._gmsh.reference_registry": (
            "fem.geometry._validation",
            "fem.geometry.errors",
            "fem.geometry.types",
        ),
        "fem.geometry._gmsh.control_dependencies": (
            "fem.geometry.errors",
        ),
        "fem.geometry._gmsh.meshing_port": (
            "fem.geometry.types",
        ),
    }
    missing = []
    offenders = []
    for module_name, allowed_modules in module_dependencies.items():
        relative = Path(*module_name.split(".")).with_suffix(".py")
        path = SRC_ROOT / relative
        if not path.is_file():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
            continue
        for target, lineno in _resolved_import_targets(path, module_name):
            root = target.split(".", 1)[0]
            if root in sys.stdlib_module_names or any(
                target == allowed or target.startswith(f"{allowed}.")
                for allowed in allowed_modules
            ):
                continue
            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
            )

    assert missing == []
    assert offenders == []


def test_phase_3_runtime_boundaries_do_not_eagerly_import_external_gmsh():
    modules = (
        "fem.geometry",
        "fem.geometry._gmsh.state",
        "fem.geometry._gmsh.session",
        "fem.geometry._gmsh.reference_registry",
        "fem.geometry._gmsh.control_dependencies",
        "fem.geometry._gmsh.meshing_port",
        "fem.mesh",
        "fem.mesh.gmsh",
    )
    script = f"""
import builtins
import importlib
import sys

sys.path.insert(0, {str(SRC_ROOT)!r})
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "gmsh" or name.startswith("gmsh."):
        raise AssertionError("external gmsh was imported eagerly")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
for module_name in {modules!r}:
    importlib.import_module(module_name)
assert "gmsh" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_geometry_facade_routes_public_stateless_names_to_canonical_modules():
    path = GEOMETRY_ROOT / "__init__.py"
    imports = {
        target for target, _ in _resolved_import_targets(path, "fem.geometry")
    }
    error_names = {
        "EntityOwnershipError",
        "GeometryError",
        "GeometryStateError",
        "StaleEntityError",
    }
    type_names = {
        "BooleanResult",
        "CurveLoopRef",
        "EntityRef",
        "FeatureResult",
        "LoftContinuity",
        "LoftParametrization",
        "LoftResult",
        "OrientedCurveRef",
        "SweepFrame",
        "WireRef",
    }

    assert {f"fem.geometry.errors.{name}" for name in error_names} <= imports
    assert {f"fem.geometry.types.{name}" for name in type_names} <= imports
    assert {
        target
        for target in imports
        if target.startswith("fem.geometry._gmsh.model.")
    } == {
        "fem.geometry._gmsh.model.GeometryModel",
        "fem.geometry._gmsh.model.model",
    }


def test_private_gmsh_model_does_not_redefine_extracted_stateless_names():
    path = GEOMETRY_ROOT / "_gmsh" / "model.py"
    moved_names = {
        "BooleanResult",
        "CurveLoopRef",
        "EntityOwnershipError",
        "EntityRef",
        "FeatureResult",
        "GeometryError",
        "GeometryStateError",
        "LoftContinuity",
        "LoftParametrization",
        "LoftResult",
        "OrientedCurveRef",
        "StaleEntityError",
        "SweepFrame",
        "WireRef",
        "_GeometrySignature",
        "_LOOP_WINDING_REFINEMENTS",
        "_OCC_BOUNDING_BOX_PADDING",
        "_PLANAR_TOLERANCE",
        "_PlaneFrame",
        "_Point2D",
        "_Point3D",
        "_RigidShapeSignature",
        "_coordinate_distance",
        "_finite_float",
        "_integer_at_least",
        "_matches_rigid_shape_signature",
        "_matches_rotated_signature",
        "_matches_translated_coordinate",
        "_matches_translated_signature",
        "_nonnegative_float",
        "_orientation_2d",
        "_plane_frame",
        "_point_axis_distance",
        "_point_segment_distance_2d",
        "_polyline_has_self_contact",
        "_polyline_winding",
        "_positive_feature_vector",
        "_positive_float",
        "_project_plane_point",
        "_project_plane_points",
        "_rotate_point_about_axis",
        "_scale_vector",
        "_segments_contact_2d",
        "_unique_first_seen",
        "_validate_elliptical_arc_geometry",
        "_validate_entity_dimension",
        "_validate_mesh_dimension",
        "_validate_positive_tag",
        "_vector_cross",
        "_vector_difference",
        "_vector_dot",
        "_vector_norm",
    }

    assert sorted(moved_names & _top_level_definition_names(path)) == []

    tree = ast.parse(path.read_text(encoding="utf-8"))
    exported = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            exported = ast.literal_eval(node.value)
    assert exported == ["GeometryModel", "model"]


def test_private_gmsh_model_delegates_extracted_stateful_ownership():
    path = GEOMETRY_ROOT / "_gmsh" / "model.py"
    assert "_State" not in _top_level_definition_names(path)

    imports = {
        target
        for target, _ in _resolved_import_targets(
            path,
            "fem.geometry._gmsh.model",
        )
    }
    required_imports = {
        "fem.geometry._gmsh.control_dependencies._ControlDependencyLedger",
        "fem.geometry._gmsh.meshing_port._BoundMeshingPort",
        "fem.geometry._gmsh.reference_registry._ReferenceRegistry",
        "fem.geometry._gmsh.session._GmshModelSession",
        "fem.geometry._gmsh.state._ModelStateMachine",
        "fem.geometry._gmsh.state._State",
    }
    assert sorted(required_imports - imports) == []

    tree = ast.parse(path.read_text(encoding="utf-8"))
    model_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GeometryModel"
    ]
    assert len(model_classes) == 1

    moved_attributes = {
        "_control_dependency_scope_unknown",
        "_created_model",
        "_curve_loop_dependencies",
        "_curve_loop_tokens",
        "_entity_control_dependencies",
        "_entity_tokens",
        "_mesher_token",
        "_owner_token",
        "_owns_session",
        "_pending_options",
        "_prior_current",
        "_state",
        "_transform_unsafe_control_dependencies",
        "_wire_dependencies",
        "_wire_tokens",
    }
    offenders = sorted(
        (node.lineno, node.attr)
        for node in ast.walk(model_classes[0])
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in moved_attributes
    )

    assert offenders == []


def test_gmsh_meshing_depends_only_on_geometry_and_backend_layers():
    paths = (
        SRC_ROOT / "fem" / "mesh" / "__init__.py",
        SRC_ROOT / "fem" / "mesh" / "gmsh.py",
    )
    offenders = []
    for path in paths:
        for target, lineno in _resolved_import_targets(path, _module_name(path)):
            allowed_fem_target = (
                target == "fem.mesh.gmsh"
                or target == "fem.geometry"
                or target.startswith("fem.geometry.")
            )
            if target == "fem" or (
                target.startswith("fem.") and not allowed_fem_target
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )
            elif not target.startswith("fem.") and not _is_standard_library_or_gmsh(
                target
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_fem_mesh_uses_exactly_one_private_geometry_acquisition_seam():
    paths = sorted((SRC_ROOT / "fem" / "mesh").rglob("*.py"))
    acquisition_calls = []
    forbidden_attributes = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr == "_acquire_meshing_port" and isinstance(
                node.ctx,
                ast.Load,
            ):
                acquisition_calls.append((path, node))
            if (
                node.attr.startswith("_mesher_")
                or node.attr == "_structured_extrude"
                or node.attr in {
                    "_complete",
                    "_complete_mesh_configuration_operation",
                }
            ):
                forbidden_attributes.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    f"{node.attr}"
                )

    assert forbidden_attributes == []
    assert len(acquisition_calls) == 1
    acquisition_path, acquisition = acquisition_calls[0]
    assert acquisition_path == SRC_ROOT / "fem" / "mesh" / "gmsh.py"
    assert isinstance(acquisition.value, ast.Name)
    assert acquisition.value.id == "geometry"

    tree = ast.parse(acquisition_path.read_text(encoding="utf-8"))
    mesher_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Mesher"
    ]
    assert len(mesher_classes) == 1
    mesher_class = mesher_classes[0]
    constructors = [
        node
        for node in mesher_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    assert len(constructors) == 1
    constructor_acquisitions = [
        node
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Attribute)
        and node.attr == "_acquire_meshing_port"
    ]
    assert len(constructor_acquisitions) == 1

    mesher_attributes = {
        node.attr
        for node in ast.walk(mesher_class)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    assert {"_geometry", "_mesher_token"}.isdisjoint(mesher_attributes)

    slot_assignments = [
        node
        for node in mesher_class.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__slots__"
            for target in node.targets
        )
    ]
    assert len(slot_assignments) == 1
    assert ast.literal_eval(slot_assignments[0].value) == ("_port",)


def test_bound_meshing_port_has_only_restricted_transaction_surface():
    path = GEOMETRY_ROOT / "_gmsh" / "meshing_port.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    port_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_BoundMeshingPort"
    ]
    assert len(port_classes) == 1
    port_class = port_classes[0]

    expected_public_methods = {
        "background_field",
        "distance_field",
        "generate_auto_mesh",
        "generate_mesh",
        "mesh_size",
        "min_field",
        "recombine",
        "structured_extrude",
        "threshold_field",
        "transfinite_curve",
        "transfinite_surface",
        "transfinite_volume",
    }
    public_methods = {
        node.name
        for node in port_class.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and not node.name.startswith("_")
    }
    assert public_methods == expected_public_methods

    public_data = {
        target.id
        for node in port_class.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name) and not target.id.startswith("_")
    }
    assert public_data == set()

    slot_assignments = [
        node
        for node in port_class.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__slots__"
            for target in node.targets
        )
    ]
    assert len(slot_assignments) == 1
    assert ast.literal_eval(slot_assignments[0].value) == ("__owner",)

    allowed_owner_transactions = {
        "_mesher_background_field",
        "_mesher_distance_field",
        "_mesher_generate_auto_mesh",
        "_mesher_generate_mesh",
        "_mesher_mesh_size",
        "_mesher_min_field",
        "_mesher_recombine",
        "_mesher_threshold_field",
        "_mesher_transfinite_curve",
        "_mesher_transfinite_surface",
        "_mesher_transfinite_volume",
        "_structured_extrude",
    }
    owner_attributes = {
        node.attr
        for node in ast.walk(port_class)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
        and node.value.attr == "__owner"
    }
    assert owner_attributes == allowed_owner_transactions

    forbidden_internal_handles = {
        "_control_dependencies",
        "_generation_token",
        "_gmsh",
        "_mesh_field_tokens",
        "_references",
        "_session",
        "_states",
        "activate",
        "facade",
        "model",
        "occ",
        "raw_model",
        "raw_occ",
    }
    offenders = sorted(
        (node.lineno, node.attr)
        for node in ast.walk(port_class)
        if isinstance(node, ast.Attribute)
        and node.attr in forbidden_internal_handles
    )
    assert offenders == []


def test_gmsh_io_imports_only_mesh_level_fem_core_types():
    path = SRC_ROOT / "fem" / "io" / "gmsh.py"
    allowed_fem_targets = {
        "fem.core.Element2D",
        "fem.core.Element3D",
        "fem.core.Mesh2D",
        "fem.core.Mesh3D",
        "fem.core.Node2D",
        "fem.core.Node3D",
        "fem.mesh.gmsh.GmshMeshRef",
    }
    resolved_imports = tuple(
        _resolved_import_targets(path, "fem.io.gmsh")
    )
    offenders = []
    for target, lineno in resolved_imports:
        if (
            target == "fem"
            or (
                target.startswith("fem.")
                and target not in allowed_fem_targets
            )
        ):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}")

    assert offenders == []
    assert "fem.mesh.gmsh.GmshMeshRef" in {
        target for target, _ in resolved_imports
    }


def test_fem_runtime_layers_do_not_import_geometry_or_meshing():
    package_roots = (
        SRC_ROOT / "fem" / "core",
        SRC_ROOT / "fem" / "elements",
        SRC_ROOT / "fem" / "assemble",
        SRC_ROOT / "fem" / "solvers",
    )
    missing_roots = [
        str(package_root.relative_to(PROJECT_ROOT))
        for package_root in package_roots
        if not package_root.is_dir()
    ]
    assert missing_roots == []

    paths = [
        path
        for package_root in package_roots
        for path in package_root.rglob("*.py")
    ]

    forbidden_roots = ("fem.geometry", "fem.mesh")
    offenders = []
    for path in sorted(paths):
        for target, lineno in _resolved_import_targets(path, _module_name(path)):
            if any(
                target == root or target.startswith(f"{root}.")
                for root in forbidden_roots
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_fem_geometry_public_api_is_exact_and_model_factory_is_public():
    from fem import geometry
    from fem.geometry import GeometryModel, model

    assert geometry.__name__ == "fem.geometry"
    assert geometry.__all__ == GEOMETRY_PUBLIC_API
    assert all(hasattr(geometry, name) for name in GEOMETRY_PUBLIC_API)
    assert GeometryModel is geometry.GeometryModel
    assert model is geometry.model
    assert isinstance(model("layout-contract", dimension=2), GeometryModel)


def test_fem_mesh_public_api_snapshots_remain_exact():
    from fem import mesh
    from fem.mesh import gmsh as gmsh_meshing

    assert mesh.__all__ == ["gmsh"]
    assert mesh.gmsh is gmsh_meshing
    assert gmsh_meshing.__all__ == GMSH_MESH_PUBLIC_API
    assert all(hasattr(gmsh_meshing, name) for name in GMSH_MESH_PUBLIC_API)

    private_runtime_names = {
        "_BoundMeshingPort",
        "_ControlDependencyLedger",
        "_EntityRegistry",
        "_GmshModelSession",
        "_ModelStateMachine",
        "_ReferenceRegistry",
        "_State",
        "_TopologyReferenceRegistry",
    }
    from fem import geometry

    assert private_runtime_names.isdisjoint(geometry.__all__)
    assert private_runtime_names.isdisjoint(vars(geometry))


def test_fem_geometry_has_no_gmsh_alias_or_legacy_module():
    from fem import geometry

    legacy_module = ".".join(("fem", "geometry", "gmsh"))
    assert "gmsh" not in vars(geometry)
    assert legacy_module not in sys.modules
    assert importlib.util.find_spec(legacy_module) is None
    assert not (GEOMETRY_ROOT / "gmsh.py").exists()
    assert not (GEOMETRY_ROOT / "gmsh").exists()


def test_removed_geometry_gmsh_path_is_absent_from_python_code():
    legacy_module = ".".join(("fem", "geometry", "gmsh"))
    offenders = []
    paths = sorted(
        path
        for root in (SRC_ROOT, TESTS_ROOT, EXAMPLES_ROOT)
        for path in root.rglob("*.py")
    )
    for path in paths:
        relative_path = path.relative_to(PROJECT_ROOT)
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if legacy_module in line:
                offenders.append(f"{relative_path}:{lineno} -> {legacy_module}")
        for target, lineno in _resolved_import_targets(
            path, _project_module_name(path)
        ):
            if target == legacy_module or target.startswith(f"{legacy_module}."):
                offenders.append(f"{relative_path}:{lineno} -> {target}")

    assert offenders == []


def test_geometry_feature_types_are_exported_only_from_the_geometry_layer():
    from fem import geometry
    import fem.mesh as mesh_package
    from fem.mesh import gmsh as gmsh_meshing

    geometry_only_names = {
        "FeatureResult",
        "LoftContinuity",
        "LoftParametrization",
        "LoftResult",
        "SweepFrame",
        "WireRef",
    }
    assert geometry_only_names.issubset(geometry.__all__)
    assert all(getattr(geometry, name) is not None for name in geometry_only_names)

    for module in (gmsh_meshing, mesh_package):
        assert geometry_only_names.isdisjoint(module.__all__)
        assert geometry_only_names.isdisjoint(vars(module))


def test_geometry_feature_methods_are_absent_from_mesher():
    from fem import geometry
    from fem.mesh import gmsh as gmsh_meshing

    operation_names = {"wire", "revolve", "sweep", "loft", "fillet", "chamfer"}
    assert all(
        hasattr(geometry.GeometryModel, operation)
        for operation in operation_names
    )
    assert all(
        not hasattr(gmsh_meshing.Mesher, operation)
        for operation in operation_names
    )


def test_historical_fem_meshing_package_is_not_imported():
    offenders = []
    paths = sorted(
        path
        for root in (SRC_ROOT, TESTS_ROOT, EXAMPLES_ROOT)
        for path in root.rglob("*.py")
    )
    for path in paths:
        for target, lineno in _resolved_import_targets(
            path, _project_module_name(path)
        ):
            if target == "fem.meshing" or target.startswith("fem.meshing."):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_examples_do_not_treat_feature_results_as_flat_tuples():
    offenders = []
    for path in sorted(EXAMPLES_ROOT.rglob("*.py")):
        for lineno, _column, reason in _extrusion_tuple_consumers(path):
            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {reason}"
            )

    assert offenders == []
