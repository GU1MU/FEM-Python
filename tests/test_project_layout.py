import ast
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"
GEOMETRY_ROOT = SRC_ROOT / "fem" / "geometry"
MESH_ROOT = SRC_ROOT / "fem" / "mesh"
GMSH_MESH_ROOT = MESH_ROOT / "gmsh"
SELECTION_ROOT = SRC_ROOT / "fem" / "selection"
APPLICATION_ROOT = SRC_ROOT / "fem" / "application"
GUI_ROOT = SRC_ROOT / "fem_gui"
AUTHORING_GUI_PATHS = (
    GUI_ROOT / "main_window.py",
    GUI_ROOT / "model_dialogs.py",
    GUI_ROOT / "analysis_definition_dialogs.py",
)


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


def test_selection_package_exports_distinct_mesh_and_geometry_modules():
    from fem import selection

    expected = [
        "curves",
        "edges",
        "elements",
        "faces",
        "nodes",
        "points",
        "surfaces",
        "volumes",
    ]

    assert selection.__all__ == expected
    assert all(
        getattr(selection, name).__name__ == f"fem.selection.{name}"
        for name in expected
    )
    assert selection.faces is not selection.surfaces
    assert "_geometry" not in selection.__all__


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


def test_geometry_selection_depends_only_on_public_geometry_contracts():
    module_names = (
        "fem.selection._geometry",
        "fem.selection.curves",
        "fem.selection.points",
        "fem.selection.surfaces",
        "fem.selection.volumes",
    )
    missing = []
    offenders = []
    for module_name in module_names:
        relative = Path(*module_name.split(".")).with_suffix(".py")
        path = SRC_ROOT / relative
        if not path.is_file():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
            continue
        for target, lineno in _resolved_import_targets(path, module_name):
            root = target.split(".", 1)[0]
            geometry_parts = target.split(".")
            public_geometry_target = (
                target == "fem.geometry"
                or (
                    len(geometry_parts) == 3
                    and geometry_parts[:2] == ["fem", "geometry"]
                    and not geometry_parts[2].startswith("_")
                )
            )
            if (
                root in sys.stdlib_module_names
                or target == "fem.selection"
                or target.startswith("fem.selection.")
                or public_geometry_target
            ):
                continue
            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
            )

    assert missing == []
    assert offenders == []


def test_mesh_selection_modules_do_not_import_geometry():
    module_names = (
        "fem.selection.edges",
        "fem.selection.elements",
        "fem.selection.faces",
        "fem.selection.nodes",
    )
    offenders = []
    for module_name in module_names:
        relative = Path(*module_name.split(".")).with_suffix(".py")
        path = SRC_ROOT / relative
        for target, lineno in _resolved_import_targets(path, module_name):
            if target == "fem.geometry" or target.startswith("fem.geometry."):
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


def test_runtime_boundaries_do_not_eagerly_import_external_gmsh():
    paths = sorted(
        {
            *GEOMETRY_ROOT.rglob("*.py"),
            *MESH_ROOT.rglob("*.py"),
            *SELECTION_ROOT.rglob("*.py"),
            *APPLICATION_ROOT.rglob("*.py"),
            *(SRC_ROOT / "fem" / "io").rglob("*.py"),
        }
    )
    modules = tuple(_module_name(path) for path in paths)
    expected_gmsh_mesh_modules = {
        "fem.mesh.gmsh",
        "fem.mesh.gmsh._configuration",
        "fem.mesh.gmsh._field_registry",
        "fem.mesh.gmsh._policies",
        "fem.mesh.gmsh._protocols",
        "fem.mesh.gmsh._runtime",
        "fem.mesh.gmsh._validation",
        "fem.mesh.gmsh.errors",
        "fem.mesh.gmsh.mesher",
        "fem.mesh.gmsh.specs",
        "fem.mesh.gmsh.types",
    }
    assert expected_gmsh_mesh_modules <= set(modules)
    expected_selection_modules = {
        "fem.selection",
        "fem.selection._geometry",
        "fem.selection.curves",
        "fem.selection.edges",
        "fem.selection.elements",
        "fem.selection.faces",
        "fem.selection.nodes",
        "fem.selection.points",
        "fem.selection.surfaces",
        "fem.selection.volumes",
    }
    assert expected_selection_modules <= set(modules)

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


def test_gmsh_meshing_recursively_depends_only_on_public_geometry_contracts():
    paths = sorted(MESH_ROOT.rglob("*.py"))
    assert paths

    offenders = []
    for path in paths:
        for target, lineno in _resolved_import_targets(path, _module_name(path)):
            allowed_fem_target = target == "fem.mesh" or target.startswith(
                "fem.mesh."
            )
            public_geometry_modules = (
                "fem.geometry.errors",
                "fem.geometry.types",
            )
            geometry_parts = target.split(".")
            public_facade_target = (
                len(geometry_parts) == 3
                and geometry_parts[:2] == ["fem", "geometry"]
                and not geometry_parts[2].startswith("_")
            )
            allowed_fem_target = allowed_fem_target or (
                target == "fem.geometry"
                or public_facade_target
                or any(
                    target == module or target.startswith(f"{module}.")
                    for module in public_geometry_modules
                )
            )
            if target == "fem" or (
                target.startswith("fem.") and not allowed_fem_target
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )
            elif not target.startswith("fem.") and (
                target.split(".", 1)[0] not in sys.stdlib_module_names
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_generated_mesh_reference_has_no_concrete_geometry_backchannel():
    path = GMSH_MESH_ROOT / "types.py"
    imports = {
        target
        for target, _ in _resolved_import_targets(path, "fem.mesh.gmsh.types")
    }
    assert not any(target.startswith("fem.geometry._gmsh") for target in imports)

    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Name) and node.id == "GeometryModel"
        for node in ast.walk(tree)
    )


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


def test_headless_fem_package_does_not_import_gui_modules():
    offenders = []
    for path in sorted((SRC_ROOT / "fem").rglob("*.py")):
        for target, lineno in _resolved_import_targets(
            path,
            _module_name(path),
        ):
            if target == "fem_gui" or target.startswith("fem_gui."):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_application_layer_has_no_qt_pyvista_or_gui_dependency():
    assert APPLICATION_ROOT.is_dir()
    forbidden_roots = {"PySide6", "pyvista", "pyvistaqt", "fem_gui"}
    offenders = []
    for path in sorted(APPLICATION_ROOT.rglob("*.py")):
        for target, lineno in _resolved_import_targets(
            path,
            _module_name(path),
        ):
            if target.split(".", 1)[0] in forbidden_roots:
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_gui_preprocessing_contains_only_display_authoring_helpers():
    path = GUI_ROOT / "preprocessing.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed_fem_modules = {
        "fem.geometry.recipe_topology",
        "fem.geometry.recipes",
        "fem.mesh.settings",
    }
    offenders = []
    for target, lineno in _resolved_import_targets(
        path,
        "fem_gui.preprocessing",
    ):
        allowed_fem_target = any(
            target == module or target.startswith(f"{module}.")
            for module in allowed_fem_modules
        )
        if target == "gmsh" or target.startswith("gmsh."):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}")
        elif target == "fem" or (
            target.startswith("fem.") and not allowed_fem_target
        ):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}")
    forbidden_functions = {
        "generate_fem_model",
        "_build_cad_domain",
        "_build_native_fem_model",
        "_gmsh_entity_node_ids",
        "_gmsh_entity_element_ids",
    }
    offenders.extend(
        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> def {node.name}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in forbidden_functions
    )

    assert offenders == []


def test_recipe_compiler_does_not_map_logical_ids_by_backend_tag_order():
    path = APPLICATION_ROOT / "recipe_compiler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        if function_name != "sorted":
            continue
        if any(
            isinstance(descendant, ast.Attribute) and descendant.attr == "tag"
            for descendant in ast.walk(node)
        ):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert offenders == []


def test_kernel_and_material_layers_do_not_import_application():
    package_roots = (
        SRC_ROOT / "fem" / "core",
        SRC_ROOT / "fem" / "elements",
        SRC_ROOT / "fem" / "materials",
        SRC_ROOT / "fem" / "solvers",
    )
    assert all(path.is_dir() for path in package_roots)
    offenders = []
    for path in sorted(
        source
        for root in package_roots
        for source in root.rglob("*.py")
    ):
        for target, lineno in _resolved_import_targets(
            path,
            _module_name(path),
        ):
            if target == "fem.application" or target.startswith(
                "fem.application."
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def _attribute_chain(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        return () if not parent else (*parent, node.attr)
    if isinstance(node, ast.Subscript):
        return _attribute_chain(node.value)
    return ()


def _assignment_targets(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else (node.target,)
            )
            yield from targets


def _is_snapshot_owned_chain(chain):
    if len(chain) > 2 and chain[:2] == ("self", "document"):
        return True
    return len(chain) > 1 and chain[0] in {
        "document",
        "snapshot",
        "session_snapshot",
    }


def test_gui_session_adapters_do_not_mutate_snapshots_or_legacy_workflow():
    paths = (
        GUI_ROOT / "main_window.py",
        GUI_ROOT / "project_io.py",
    )
    mutating_methods = {
        "add",
        "append",
        "clear",
        "discard",
        "extend",
        "insert",
        "pop",
        "popitem",
        "remove",
        "reverse",
        "setdefault",
        "sort",
        "update",
    }
    offenders = []

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for target in _assignment_targets(tree):
            chain = _attribute_chain(target)
            if _is_snapshot_owned_chain(chain) or "workflow" in chain:
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:"
                    f"{getattr(target, 'lineno', '?')} -> "
                    f"{'.'.join(chain)} assignment"
                )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func,
                ast.Attribute,
            ):
                continue
            owner = _attribute_chain(node.func.value)
            if (
                node.func.attr in mutating_methods
                and (
                    _is_snapshot_owned_chain(owner)
                    or "workflow" in owner
                )
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    f"{'.'.join((*owner, node.func.attr))}()"
                )

    assert offenders == []


def test_production_code_has_no_legacy_workflow_state_booleans():
    offenders = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "WorkflowState":
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    "class WorkflowState"
                )
            elif isinstance(node, ast.Name) and node.id == "WorkflowState":
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    "WorkflowState"
                )
            elif isinstance(node, ast.Attribute) and node.attr in {
                "workflow",
                "model_checked",
                "results_current",
            }:
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    f".{node.attr}"
                )

    assert offenders == []


def test_gui_uses_only_public_model_session_commands():
    path = GUI_ROOT / "main_window.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(
            node.func,
            ast.Attribute,
        ):
            continue
        owner = _attribute_chain(node.func.value)
        if owner == ("self", "session") and node.func.attr.startswith("_"):
            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                f"self.session.{node.func.attr}()"
            )

    assert offenders == []


def _literal_strings(node):
    for descendant in ast.walk(node):
        if (
            isinstance(descendant, ast.Constant)
            and isinstance(descendant.value, str)
        ):
            yield descendant.value


def _contains_casefold_call(node):
    return any(
        isinstance(descendant, ast.Call)
        and isinstance(descendant.func, ast.Attribute)
        and descendant.func.attr == "casefold"
        for descendant in ast.walk(node)
    )


def test_gui_authoring_does_not_infer_element_capabilities_from_type_names():
    family_prefixes = ("beam", "truss", "tri", "quad", "tet", "hex")
    canonical_element_types = {
        "beam2",
        "truss2",
        "tri3",
        "tri6",
        "quad4",
        "quad8",
        "tet4",
        "tet10",
        "hex8",
        "hex20",
    }
    offenders = []

    for path in AUTHORING_GUI_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for value in _literal_strings(tree):
            if value.casefold() in canonical_element_types:
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)} -> "
                    f"canonical element type literal {value!r}"
                )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "startswith"
                and any(
                    value.casefold().startswith(family_prefixes)
                    for argument in node.args
                    for value in _literal_strings(argument)
                )
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    "element-family startswith()"
                )
                continue

            if not isinstance(node, ast.Compare) or not any(
                isinstance(operator, (ast.Eq, ast.NotEq))
                for operator in node.ops
            ):
                continue
            operands = (node.left, *node.comparators)
            if any(
                value.casefold() == "beam2"
                for operand in operands
                for value in _literal_strings(operand)
            ) and any(_contains_casefold_call(operand) for operand in operands):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    "casefold() Beam2 comparison"
                )

    assert offenders == []


def test_gui_authoring_does_not_import_element_kernels():
    offenders = []
    for path in AUTHORING_GUI_PATHS:
        for target, lineno in _resolved_import_targets(
            path,
            _module_name(path),
        ):
            if target == "fem.elements" or target.startswith("fem.elements."):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_main_window_model_check_uses_application_preflight_only():
    path = GUI_ROOT / "main_window.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_calls = {
        "boundary_for_step",
        "validate_constraint_stability",
        "validate_problem",
    }
    model_check_functions = {
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            "model_check" in node.name
            or node.name == "check_current_model"
        )
    }
    assert model_check_functions

    offenders = []
    for function in model_check_functions:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if function_name in forbidden_calls:
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    f"{function_name}()"
                )

    assert offenders == []


def test_session_has_no_private_model_definitions_compiler():
    path = APPLICATION_ROOT / "session.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_functions = {
        "_compile_definitions",
        "_definitions_from_model",
        "_validate_definition_links",
    }
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> {node.name}()"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in forbidden_functions
    ]

    assert offenders == []


def test_production_accept_validation_has_no_passed_escape_hatch():
    offenders = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if function_name != "accept_validation":
                continue
            if any(keyword.arg == "passed" for keyword in node.keywords):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    "accept_validation(..., passed=...)"
                )

    assert offenders == []


def test_gui_model_definitions_shim_is_removed():
    assert not (GUI_ROOT / "model_definitions.py").exists()
