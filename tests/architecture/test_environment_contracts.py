from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = PROJECT_ROOT / "tests"
FIXTURES_ROOT = TESTS_ROOT / "fixtures"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
_SKIP_REASON_PREFIXES = (
    "[slow-opt-in]",
    "[cloud-opt-in]",
    "[platform-capability]",
    "[optional-native-runtime]",
)


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    assert match is not None, requirement
    return match.group(0).casefold().replace("_", "-")


def _requirements_by_name(requirements: list[str]) -> dict[str, str]:
    result = {}
    for requirement in requirements:
        name = _requirement_name(requirement)
        assert name not in result
        result[name] = requirement
    return result


def _git_files(pathspec: str) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", pathspec],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return {
        value.decode("utf-8").replace("\\", "/")
        for value in completed.stdout.split(b"\0")
        if value
    }


def _is_gitignored(path: str) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert completed.returncode in {0, 1}
    return completed.returncode == 0


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _static_reason(call: ast.Call, call_name: tuple[str, ...]) -> str | None:
    reason_node = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "reason"),
        None,
    )
    if reason_node is None and call_name == ("pytest", "skip") and call.args:
        reason_node = call.args[0]
    if isinstance(reason_node, ast.Constant) and isinstance(reason_node.value, str):
        return reason_node.value
    if isinstance(reason_node, ast.JoinedStr):
        return "".join(
            value.value
            for value in reason_node.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
    return None


def test_test_extra_covers_default_gui_and_agent_runtime_with_range_parity():
    extras = _pyproject()["project"]["optional-dependencies"]
    test_requirements = _requirements_by_name(extras["test"])
    product_requirements = _requirements_by_name([
        *extras["gui"],
        *extras["agent"],
    ])

    assert "pytest" in test_requirements
    assert set(product_requirements) < set(test_requirements)
    assert {
        name: test_requirements[name] for name in product_requirements
    } == product_requirements


def test_skip_markers_and_unknown_warning_policy_are_registered():
    pytest_options = _pyproject()["tool"]["pytest"]["ini_options"]
    marker_names = {
        marker.partition(":")[0].strip()
        for marker in pytest_options["markers"]
    }

    assert {"slow", "cloud", "platform", "optional_runtime"} <= marker_names
    assert pytest_options["pythonpath"] == ["."]
    assert pytest_options["filterwarnings"] == [
        "error",
        (
            "ignore:Setting the shape on a NumPy array has been deprecated in "
            "NumPy 2\\.5\\.:DeprecationWarning:vtkmodules\\.util\\.numpy_support$"
        ),
    ]


def test_every_static_test_skip_reason_has_a_stable_category_prefix():
    skip_calls = {
        ("pytest", "skip"),
        ("pytest", "importorskip"),
        ("pytest", "mark", "skip"),
        ("pytest", "mark", "skipif"),
    }
    offenders = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            call_name = _attribute_chain(call.func)
            if call_name not in skip_calls:
                continue
            reason = _static_reason(call, call_name)
            if reason is not None and not reason.startswith(_SKIP_REASON_PREFIXES):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{call.lineno} -> {reason!r}"
                )

    assert offenders == []


def test_fixture_resources_are_under_the_fixture_root_and_tracked_by_git():
    fixture_files = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in FIXTURES_ROOT.rglob("*")
        if path.is_file()
    }

    assert fixture_files
    assert fixture_files == _git_files("tests/fixtures/**")


def test_fixture_allow_rule_does_not_expose_local_resource_trees():
    assert not _is_gitignored("tests/fixtures/_contract_probe.txt")
    assert _is_gitignored("fem-agent.config.json")
    assert _is_gitignored("tests/fem-agent.test.config.json")
    for local_path in (
        "docs/_contract_probe.txt",
        "data/_contract_probe.txt",
        "scripts/_contract_probe.txt",
    ):
        assert _is_gitignored(local_path)


def test_pyproject_is_the_only_dependency_fact_source():
    assert PYPROJECT_PATH.is_file()
    assert not (PROJECT_ROOT / "requirements.txt").exists()
    assert list(PROJECT_ROOT.glob("requirements*.txt")) == []


def test_production_exporters_do_not_call_deprecated_stress_wrappers():
    deprecated = {
        ("stress", "export", "element"),
        ("stress", "export", "nodal"),
    }
    offenders = []
    for relative in (
        Path("src/fem/post/vtk/export.py"),
        Path("src/fem_agent/tools/exports.py"),
    ):
        path = PROJECT_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            chain = _attribute_chain(call.func)
            if any(chain[-len(target):] == target for target in deprecated):
                offenders.append(f"{relative.as_posix()}:{call.lineno}")

    assert offenders == []
