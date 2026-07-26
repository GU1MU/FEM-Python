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
        "fem.geometry",
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
    paths = (GUI_ROOT / "main_window.py",)
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


def test_gui_project_save_gates_only_use_session_can_save_projection():
    path = GUI_ROOT / "main_window.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    update_states = functions["_update_action_states"]
    save_action_call = next(
        node
        for node in ast.walk(update_states)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_set_action_available"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "save_project"
    )
    gate_nodes = {
        "save action": save_action_call.args[1],
        "save handler": functions["save_native_project"],
        "discard confirmation": functions["_confirm_discard_changes"],
    }
    forbidden_attributes = {"source_kind", "geometry_recipe"}

    for label, node in gate_nodes.items():
        chains = {
            _attribute_chain(descendant)
            for descendant in ast.walk(node)
            if isinstance(descendant, ast.Attribute)
        }
        assert ("self", "document", "can_save") in chains, label
        assert not any(
            chain and chain[-1] in forbidden_attributes
            for chain in chains
        ), label


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


def test_beam_frame_domain_module_has_no_upward_or_adapter_dependency():
    path = SRC_ROOT / "fem" / "elements" / "beam_frame.py"
    forbidden_roots = (
        "fem.application",
        "fem_gui",
        "fem.io",
        "fem.abaqus",
        "fem.solvers",
        "fem.post",
    )
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
        for target, lineno in _resolved_import_targets(
            path,
            "fem.elements.beam_frame",
        )
        if any(
            target == root or target.startswith(f"{root}.")
            for root in forbidden_roots
        )
    ]

    assert path.is_file()
    assert offenders == []


def test_beam_frame_has_one_production_resolver_and_no_legacy_helper():
    resolver_definitions = []
    legacy_references = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        legacy_references.extend(
            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "beam3d_geometry"
        )
        resolver_definitions.extend(
            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "resolve_beam_frame"
        )

    assert legacy_references == []
    assert len(resolver_definitions) == 1
    resolver_path, _lineno = resolver_definitions[0].rsplit(":", 1)
    assert Path(resolver_path) == Path(
        "src/fem/elements/beam_frame.py"
    )


def test_gui_beam_frame_consumers_use_application_query_boundary():
    paths = (
        GUI_ROOT / "main_window.py",
        GUI_ROOT / "inspection_service.py",
        GUI_ROOT / "widgets" / "viewport.py",
    )
    forbidden_imports = []
    forbidden_functions = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forbidden_functions.extend(
            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> {node.name}"
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {
                "resolve_beam_frame",
                "beam3d_geometry",
                "resolve_effective_beam_frames",
            }
        )
        for target, lineno in _resolved_import_targets(
            path,
            _module_name(path),
        ):
            if target == "fem.elements" or target.startswith(
                "fem.elements."
            ):
                forbidden_imports.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    main_source = (GUI_ROOT / "main_window.py").read_text(
        encoding="utf-8"
    )
    inspection_source = (GUI_ROOT / "inspection_service.py").read_text(
        encoding="utf-8"
    )
    viewport_source = (
        GUI_ROOT / "widgets" / "viewport.py"
    ).read_text(encoding="utf-8")

    assert forbidden_imports == []
    assert forbidden_functions == []
    assert "resolve_effective_beam_frames" in main_source
    assert "resolve_effective_beam_frames" in inspection_source
    assert "_effective_frame_query" in viewport_source


def test_section_editor_does_not_own_beam_orientation_authoring():
    path = GUI_ROOT / "model_dialogs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    section_editor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SectionEditDialog"
    )

    assert not any(
        value == "beam_orientation"
        or value == "beam_local_y_reference"
        for value in _literal_strings(section_editor)
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {
            "BeamOrientation",
            "beam_orientation",
            "beam_local_y_reference",
        }
        for node in ast.walk(section_editor)
    )


def test_project_v1_writer_contains_explicit_orientation_fail_closed_guard():
    path = SRC_ROOT / "fem" / "io" / "project_v1.py"
    source = path.read_text(encoding="utf-8")

    assert "beam_orientation" in source
    assert "ProjectV1EncodeError" in source
    assert "v1 不支持 Beam orientation" in source


def test_project_v2_production_path_has_no_v1_specific_import():
    path = SRC_ROOT / "fem" / "io" / "project_v2.py"
    imports = {
        target
        for target, _lineno in _resolved_import_targets(
            path,
            "fem.io.project_v2",
        )
    }

    assert not any(
        target == "fem.io.project_v1"
        or target.startswith("fem.io.project_v1.")
        for target in imports
    )


def test_project_field_codecs_have_one_shared_implementation():
    shared_path = SRC_ROOT / "fem" / "io" / "_project_codec.py"
    adapter_paths = (
        SRC_ROOT / "fem" / "io" / "project_v1.py",
        SRC_ROOT / "fem" / "io" / "project_v2.py",
    )
    field_stems = {
        "assignment",
        "boundary",
        "cload",
        "contour",
        "edge_load",
        "geometry",
        "gravity_load",
        "line_load",
        "material",
        "output",
        "section",
        "step",
        "surface_load",
    }
    expected_shared = {
        f"{operation}_{stem}_field"
        for operation in ("decode", "encode")
        for stem in field_stems
    }
    shared_tree = ast.parse(shared_path.read_text(encoding="utf-8"))
    shared_definitions = {
        node.name
        for node in ast.walk(shared_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert expected_shared.issubset(shared_definitions)
    for path in adapter_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        duplicate_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.lstrip("_").removesuffix("_v1").removesuffix("_v2")
            in {
                f"{operation}_{stem}"
                for operation in ("decode", "encode")
                for stem in field_stems
            }
            and node.name not in {
                "decode_assignment_v2",
                "encode_assignment_v2",
            }
        }
        assert duplicate_names == set(), path.relative_to(PROJECT_ROOT)


def test_project_field_codec_policies_are_explicit_per_version():
    v1_source = (
        SRC_ROOT / "fem" / "io" / "project_v1.py"
    ).read_text(encoding="utf-8")
    v2_source = (
        SRC_ROOT / "fem" / "io" / "project_v2.py"
    ).read_text(encoding="utf-8")

    assert "ProjectFieldCodecPolicy" in v1_source
    assert "require_current_fields=False" in v1_source
    assert "assignment_orientation=False" in v1_source
    assert "ProjectFieldCodecPolicy" in v2_source
    assert "require_current_fields=True" in v2_source
    assert "assignment_orientation=True" in v2_source


def test_native_authoring_domain_has_no_integer_geometry_identity():
    topology_path = GEOMETRY_ROOT / "recipe_topology.py"
    compiler_path = APPLICATION_ROOT / "recipe_compiler.py"
    preprocessing_path = APPLICATION_ROOT / "preprocessing.py"
    definitions_path = APPLICATION_ROOT / "definitions.py"
    settings_path = MESH_ROOT / "settings.py"

    topology_tree = ast.parse(topology_path.read_text(encoding="utf-8"))
    topology_methods = {
        node.name
        for node in ast.walk(topology_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "logical_entity" not in topology_methods
    assert "logical_index" not in topology_methods

    compiler_source = compiler_path.read_text(encoding="utf-8")
    preprocessing_source = preprocessing_path.read_text(encoding="utf-8")
    definitions_source = definitions_path.read_text(encoding="utf-8")
    settings_source = settings_path.read_text(encoding="utf-8")
    assert "entity_id" not in compiler_source
    assert "entity_id" not in preprocessing_source
    assert "entity_ids" not in definitions_source
    assert "entity_id" not in settings_source


def test_geometry_preview_pick_tokens_are_viewport_private():
    preview_path = GUI_ROOT / "preprocessing.py"
    preview_tree = ast.parse(preview_path.read_text(encoding="utf-8"))
    preview_class = next(
        node
        for node in preview_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GeometryPreview"
    )
    field_names = {
        node.target.id
        for node in preview_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    forbidden_fields = {
        "face_pick_ids",
        "edge_pick_ids",
        "point_pick_ids",
        "body_pick_id",
    }

    assert field_names.isdisjoint(forbidden_fields)
    assert "geometry_pick_id" not in preview_path.read_text(encoding="utf-8")
    assert all(
        "geometry_pick_id" not in path.read_text(encoding="utf-8")
        for path in GUI_ROOT.rglob("*.py")
        if path != GUI_ROOT / "widgets" / "viewport.py"
    )


def test_native_preprocessing_uses_typed_region_catalog_and_one_control_path():
    compiler_source = (
        APPLICATION_ROOT / "recipe_compiler.py"
    ).read_text(encoding="utf-8")
    preprocessing_source = (
        APPLICATION_ROOT / "preprocessing.py"
    ).read_text(encoding="utf-8")
    settings_source = (MESH_ROOT / "settings.py").read_text(
        encoding="utf-8"
    )

    assert "region_bindings" in compiler_source
    assert "RecipeRegionSelector" in compiler_source
    assert "CompiledDomainRegionSource" in preprocessing_source
    assert "RecipeRegionSource" in preprocessing_source
    assert "LogicalReferencesRegionSource" in preprocessing_source
    assert "describe_native_regions" in (
        APPLICATION_ROOT / "native_regions.py"
    ).read_text(encoding="utf-8")
    assert "topology.groups" not in preprocessing_source
    assert "local_size" not in preprocessing_source
    assert "local_size" not in settings_source


def test_gui_native_region_choices_delegate_to_application_catalog():
    path = GUI_ROOT / "main_window.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "describe_native_regions"
    ]
    assert len(calls) == 1
    assert _attribute_chain(calls[0].func) == (
        "application_api",
        "describe_native_regions",
    )

    built_in_names = {
        "DOMAIN",
        "BOTTOM",
        "RIGHT",
        "TOP",
        "LEFT",
        "FRONT",
        "BACK",
        "OUTER",
        "HOLE",
    }
    hardcoded = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in built_in_names
    }
    assert hardcoded == set()


def test_application_and_gui_avoid_versioned_project_codecs():
    forbidden_roots = {
        "fem.io.project_v1",
        "fem.io.project_v2",
    }
    offenders = []
    for package_root in (APPLICATION_ROOT, GUI_ROOT):
        for path in sorted(package_root.rglob("*.py")):
            for target, lineno in _resolved_import_targets(
                path,
                _module_name(path),
            ):
                if any(
                    target == root or target.startswith(f"{root}.")
                    for root in forbidden_roots
                ):
                    offenders.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> "
                        f"{target}"
                    )

    assert offenders == []


def test_transitional_gui_document_adapter_is_removed():
    adapter = GUI_ROOT / "document.py"
    offenders = []
    for path in sorted(GUI_ROOT.rglob("*.py")):
        for target, lineno in _resolved_import_targets(
            path,
            _module_name(path),
        ):
            if target == "fem_gui.document" or target.startswith(
                "fem_gui.document."
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert not adapter.exists()
    assert offenders == []


def _call_terminal_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _normalized_wire_literal(value):
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().upper().split())


def _zero_subscript(node):
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == 0
    )


def test_abaqus_adapter_is_headless():
    abaqus_root = SRC_ROOT / "fem" / "abaqus"
    paths = sorted(abaqus_root.rglob("*.py"))
    assert paths

    offenders = []
    for path in paths:
        for target, lineno in _resolved_import_targets(
            path,
            _module_name(path),
        ):
            if target == "fem_gui" or target.startswith("fem_gui."):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_abaqus_parser_has_no_application_solver_or_gui_dependency():
    path = SRC_ROOT / "fem" / "abaqus" / "parser.py"
    forbidden_roots = (
        "fem.application",
        "fem.solver",
        "fem.solvers",
        "fem_gui",
    )
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
        for target, lineno in _resolved_import_targets(
            path,
            "fem.abaqus.parser",
        )
        if any(
            target == root or target.startswith(f"{root}.")
            for root in forbidden_roots
        )
    ]

    assert path.is_file()
    assert offenders == []


def test_abaqus_parser_does_not_compute_beam_frames_or_rotations():
    path = SRC_ROOT / "fem" / "abaqus" / "parser.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_calls = {
        "beam3d_geometry",
        "cross",
        "resolve_beam_frame",
    }
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_terminal_name(node.func)
        if name in forbidden_calls or "rotation" in name.casefold():
            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> {name}()"
            )

    assert offenders == []


def test_core_elements_and_materials_do_not_import_abaqus_adapter():
    package_roots = (
        SRC_ROOT / "fem" / "core",
        SRC_ROOT / "fem" / "elements",
        SRC_ROOT / "fem" / "materials",
    )
    assert all(root.is_dir() for root in package_roots)

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
            if target == "fem.abaqus" or target.startswith("fem.abaqus."):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_domain_application_and_gui_do_not_branch_on_abaqus_wire_tokens():
    package_roots = (
        SRC_ROOT / "fem" / "core",
        SRC_ROOT / "fem" / "elements",
        SRC_ROOT / "fem" / "materials",
        SRC_ROOT / "fem" / "application",
        SRC_ROOT / "fem" / "solvers",
        GUI_ROOT,
    )
    adapter_only_tokens = {
        "B31",
        "T3D2",
        "RECT",
        "CIRC",
        "THICK PIPE",
        "PIPE",
        "PX",
        "PY",
        "PZ",
        "P1",
        "P2",
        "QGLOBAL",
        "QLOCAL",
    }
    offenders = []

    for path in sorted(
        source
        for root in package_roots
        for source in root.rglob("*.py")
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        semantic_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(
                node,
                (
                    ast.Compare,
                    ast.Dict,
                    ast.MatchValue,
                    ast.Set,
                ),
            )
            or (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and isinstance(
                    node.value,
                    (ast.Dict, ast.List, ast.Set, ast.Tuple),
                )
            )
        ]
        reported = set()
        for semantic_node in semantic_nodes:
            for literal in ast.walk(semantic_node):
                if not (
                    isinstance(literal, ast.Constant)
                    and isinstance(literal.value, str)
                ):
                    continue
                token = _normalized_wire_literal(literal.value)
                if token not in adapter_only_tokens:
                    continue
                offender = (
                    f"{path.relative_to(PROJECT_ROOT)}:{literal.lineno} -> "
                    f"{literal.value!r}"
                )
                if offender not in reported:
                    offenders.append(offender)
                    reported.add(offender)

    assert offenders == []


def test_non_adapter_semantics_do_not_read_abaqus_type_provenance():
    package_roots = (
        SRC_ROOT / "fem" / "core",
        SRC_ROOT / "fem" / "elements",
        SRC_ROOT / "fem" / "materials",
        SRC_ROOT / "fem" / "application",
        SRC_ROOT / "fem" / "solvers",
    )
    offenders = []

    for path in sorted(
        source
        for root in package_roots
        for source in root.rglob("*.py")
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "abaqus_type"
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    ".get('abaqus_type')"
                )
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "abaqus_type"
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    "['abaqus_type']"
                )

    assert offenders == []


def test_abaqus_formulation_notice_has_one_adapter_owner():
    notice_code = "abaqus.b31.euler_bernoulli_approximation"
    occurrences = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        occurrences.extend(
            (path.relative_to(PROJECT_ROOT), node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and node.value == notice_code
        )

    assert len(occurrences) == 1
    assert occurrences[0][0] == Path("src/fem/abaqus/builder.py")


def test_b31_source_topology_audit_has_one_adapter_implementation():
    topology_code = "abaqus.b31.nodal_normal_averaging_unsupported"
    definitions = []
    code_occurrences = []

    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions.extend(
            (path.relative_to(PROJECT_ROOT), node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_audit_b31_topology"
        )
        code_occurrences.extend(
            (path.relative_to(PROJECT_ROOT), node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and node.value == topology_code
        )

    expected_path = Path("src/fem/abaqus/builder.py")
    assert len(definitions) == 1
    assert definitions[0][0] == expected_path
    assert len(code_occurrences) == 1
    assert code_occurrences[0][0] == expected_path


def test_production_has_no_beam_slenderness_gate():
    suspicious_name_fragments = (
        "beam_aspect_ratio",
        "beam_slender",
        "length_to_depth",
        "length_to_section",
        "slenderness",
    )
    offenders = []

    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ()
            if isinstance(node, ast.Name):
                names = (node.id,)
            elif isinstance(node, ast.Attribute):
                names = (node.attr,)
            elif isinstance(node, ast.arg):
                names = (node.arg,)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = (node.name,)
            if any(
                fragment in name.casefold()
                for name in names
                for fragment in suspicious_name_fragments
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    f"{names[0]}"
                )

    assert offenders == []


def test_abaqus_builder_does_not_dispatch_from_first_mesh_element():
    abaqus_root = SRC_ROOT / "fem" / "abaqus"
    offenders = []

    for path in sorted(abaqus_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not _zero_subscript(node):
                continue
            chain = _attribute_chain(node.value)
            if len(chain) >= 2 and chain[-2:] == ("mesh", "elements"):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    f"{'.'.join(chain)}[0]"
                )

    assert offenders == []


def test_standard_line_subset_excludes_retired_wire_dialect():
    from fem.abaqus.contracts import (
        RETIRED_DLOAD_LABELS,
        RETIRED_ELEMENT_TYPES,
        STANDARD_LINE_SUBSET,
    )

    assert STANDARD_LINE_SUBSET.element_types == frozenset({"B31", "T3D2"})
    assert STANDARD_LINE_SUBSET.section_profiles == frozenset(
        {"RECT", "CIRC", "THICK PIPE"}
    )
    assert STANDARD_LINE_SUBSET.distributed_load_labels == frozenset(
        {"PX", "PY", "PZ", "P1", "P2", "GRAV"}
    )
    assert RETIRED_ELEMENT_TYPES.isdisjoint(
        STANDARD_LINE_SUBSET.element_types
    )
    assert RETIRED_DLOAD_LABELS.isdisjoint(
        STANDARD_LINE_SUBSET.distributed_load_labels
    )
    assert "truss section" not in STANDARD_LINE_SUBSET.executed_keywords


def test_retired_element_aliases_are_not_adapter_mapping_keys():
    retired_aliases = {"BEAM2", "TRUSS2"}
    offenders = []

    for path in sorted((SRC_ROOT / "fem" / "abaqus").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key in node.keys:
                if (
                    isinstance(key, ast.Constant)
                    and key.value in retired_aliases
                ):
                    offenders.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{key.lineno} -> "
                        f"mapping key {key.value!r}"
                    )

    assert offenders == []


def test_abaqus_pipe_profile_is_never_mapped_to_hollow_circle():
    path = SRC_ROOT / "fem" / "abaqus" / "builder.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(
            value == "PIPE" for value in _literal_strings(node.test)
        ):
            if any(
                value == "hollow_circle"
                for statement in node.body
                for value in _literal_strings(statement)
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> "
                    "PIPE branch contains hollow_circle"
                )
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "PIPE"
                and any(
                    literal == "hollow_circle"
                    for literal in _literal_strings(value)
                )
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{key.lineno} -> "
                    "PIPE mapping contains hollow_circle"
                )

    assert offenders == []
