import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = PROJECT_ROOT / "tests"
EXAMPLES_ROOT = PROJECT_ROOT / "examples"


def _string_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value.replace("\\", "/")


class ProjectLayoutTests(unittest.TestCase):
    def test_tests_do_not_reference_example_data_or_results_outputs(self):
        offenders = []
        for path in TESTS_ROOT.rglob("test_*.py"):
            if path.name == Path(__file__).name:
                continue
            for literal in _string_literals(path):
                if literal == "examples" or literal.startswith("examples/"):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {literal}")
                if literal == "results" or literal.startswith("results/"):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {literal}")

        self.assertEqual([], offenders)

    def test_project_temp_directory_is_ignored(self):
        ignored = {
            line.strip()
            for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("temp/", ignored)

    def test_examples_use_shared_material_catalog_and_fixed_output_dirs(self):
        self.assertTrue((EXAMPLES_ROOT / "examples_data" / "examples_materials.csv").exists())
        self.assertFalse((EXAMPLES_ROOT / "examples_data" / "cantilever_beam_materials.csv").exists())

        offenders = []
        for path in EXAMPLES_ROOT.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "FEM_MIXED_EXAMPLE_OUTPUT_DIR" in text:
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
            if "cantilever_beam_materials.csv" in text:
                offenders.append(str(path.relative_to(PROJECT_ROOT)))

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
