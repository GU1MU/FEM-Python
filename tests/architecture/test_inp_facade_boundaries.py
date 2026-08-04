"""Architecture guards for the final complete-model INP facade."""

from __future__ import annotations

import ast
from pathlib import Path

from fem.io import inp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
INP_FACADE = SRC_ROOT / "fem" / "io" / "inp.py"
INP_IMPL_ROOT = SRC_ROOT / "fem" / "io" / "_inp"

EXPECTED_PUBLIC_API = [
    "InpBuildError",
    "InpImportNotice",
    "InpImportResult",
    "InpInputError",
    "InpKeywordCategory",
    "InpParseError",
    "InpSourceLocation",
    "InpSourceOccurrence",
    "InpSourceSummary",
    "UnsupportedInpFeatureError",
    "read",
    "read_with_report",
]


def _module_name(path: Path) -> str:
    if SRC_ROOT not in path.parents:
        return ""
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    return ".".join(relative.parts)


def _resolved_import_targets(path: Path, module_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module_name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = package.split(".") if package else []
                if node.level > len(prefix) + 1:
                    target = node.module or ""
                else:
                    base = prefix[: len(prefix) - node.level + 1]
                    target = ".".join((*base, *(node.module or "").split(".")))
            else:
                target = node.module or ""
            yield target, node.lineno


def _production_and_test_sources():
    roots = (SRC_ROOT, PROJECT_ROOT / "tests", PROJECT_ROOT / "examples")
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if path == INP_FACADE or INP_IMPL_ROOT in path.parents:
                continue
            yield path


def test_no_legacy_or_private_inp_imports_outside_the_adapter_boundary():
    offenders = []
    for path in _production_and_test_sources():
        for target, lineno in _resolved_import_targets(path, _module_name(path)):
            if (
                target == "fem.abaqus"
                or target.startswith("fem.abaqus.")
                or target == "fem.io._inp"
                or target.startswith("fem.io._inp.")
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_legacy_package_and_leaf_reader_surface_are_removed():
    assert not (SRC_ROOT / "fem" / "abaqus").exists()
    for name in (
        "read_tri3",
        "read_tri6",
        "read_quad4",
        "read_quad8",
        "read_mixed2d",
        "read_tet4",
        "read_tet10",
        "read_hex8",
        "read_hex20",
    ):
        assert not hasattr(inp, name)


def test_inp_facade_exports_exactly_the_frozen_stable_api():
    assert inp.__all__ == EXPECTED_PUBLIC_API

    for name in inp.__all__:
        value = getattr(inp, name)
        module_name = getattr(value, "__module__", "")
        assert module_name == "fem.io.inp"
        assert not module_name.startswith("fem.io._inp")
        assert not module_name.startswith("fem.abaqus")


def test_inp_facade_does_not_expose_low_level_workflow_symbols():
    for name in (
        "AbaqusDeck",
        "AbaqusParser",
        "build_model",
        "build_model_with_report",
        "parse_file",
        "resolve_b31_orientations",
    ):
        assert not hasattr(inp, name)


def test_inp_facade_has_no_extra_non_private_business_symbols():
    assert {
        name
        for name in vars(inp)
        if not name.startswith("_")
    } == set(EXPECTED_PUBLIC_API)


def test_inp_report_values_have_only_canonical_source_names():
    assert not hasattr(inp.InpSourceOccurrence, "keyword")
    assert not hasattr(inp.InpSourceOccurrence, "span")
    assert not hasattr(inp.InpSourceSummary, "keyword_occurrences")
    assert not hasattr(inp.InpSourceSummary, "source_occurrences")
