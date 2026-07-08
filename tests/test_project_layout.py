import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = PROJECT_ROOT / "tests"
EXAMPLES_ROOT = PROJECT_ROOT / "examples"


def _string_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value.replace("\\", "/")


def test_legacy_regressions_entrypoint_has_been_removed():
    assert not (TESTS_ROOT / "test_regressions.py").exists()


def test_migrated_tests_do_not_use_legacy_runner_style():
    legacy_runner = "unit" + "test"
    forbidden_patterns = [
        "import " + legacy_runner,
        legacy_runner + "." + "Test" + "Case",
        "Test" + "Case",
        legacy_runner + ".main",
        "self" + ".assert",
    ]

    offenders = []
    for path in TESTS_ROOT.glob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                offenders.append(f"{path.name}: {pattern}")

    assert offenders == []


def test_pytest_tmp_path_can_create_and_read_file(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    output = child / "result.txt"

    output.write_text("ok", encoding="utf-8")

    assert child.is_dir()
    assert output.read_text(encoding="utf-8") == "ok"


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


def test_project_temp_directory_is_ignored():
    ignored = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "temp/" in ignored


def test_examples_use_shared_material_catalog_and_fixed_output_dirs():
    assert (EXAMPLES_ROOT / "examples_data" / "examples_materials.csv").exists()
    assert not (EXAMPLES_ROOT / "examples_data" / "cantilever_beam_materials.csv").exists()

    offenders = []
    for path in EXAMPLES_ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "FEM_MIXED_EXAMPLE_OUTPUT_DIR" in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
        if "cantilever_beam_materials.csv" in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
