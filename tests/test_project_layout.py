import ast
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"
EXAMPLES_ROOT = PROJECT_ROOT / "examples"


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


def test_gmsh_geometry_has_no_reverse_or_third_party_runtime_dependencies():
    paths = (
        SRC_ROOT / "fem" / "geometry" / "__init__.py",
        SRC_ROOT / "fem" / "geometry" / "gmsh.py",
    )
    allowed_fem_targets = {"fem.geometry.gmsh"}
    offenders = []
    for path in paths:
        for target, lineno in _resolved_import_targets(path, _module_name(path)):
            if target == "fem" or (
                target.startswith("fem.") and target not in allowed_fem_targets
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
                or target == "fem.geometry.gmsh"
                or target.startswith("fem.geometry.gmsh.")
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


def test_advanced_feature_types_are_exported_only_from_the_geometry_layer():
    from fem.geometry import gmsh as gmsh_geometry
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
    assert geometry_only_names.issubset(gmsh_geometry.__all__)
    assert gmsh_geometry.FeatureResult.__module__ == "fem.geometry.gmsh"
    assert gmsh_geometry.LoftResult.__module__ == "fem.geometry.gmsh"
    assert gmsh_geometry.WireRef.__module__ == "fem.geometry.gmsh"

    for module in (gmsh_meshing, mesh_package):
        assert geometry_only_names.isdisjoint(module.__all__)
        assert geometry_only_names.isdisjoint(vars(module))


def test_advanced_geometry_features_are_absent_from_mesher():
    from fem.geometry import gmsh as gmsh_geometry
    from fem.mesh import gmsh as gmsh_meshing

    operation_names = {"wire", "revolve", "sweep", "loft", "fillet", "chamfer"}
    assert all(
        hasattr(gmsh_geometry.GeometryModel, operation)
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
