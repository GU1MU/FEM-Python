from __future__ import annotations

from pathlib import Path

import pytest


pytest_plugins = ["pytester"]

TESTS_ROOT = Path(__file__).resolve().parents[1]
IMPORTER_NODEID = str(
    TESTS_ROOT / "materials" / "test_section_capabilities.py"
) + "::test_real_importer_internal_section_set_uses_the_same_resolution"


@pytest.mark.parametrize("integration", [False, True], ids=["default", "integration"])
def test_collection_keeps_plain_real_names_and_gates_native_dependencies(
    pytester, monkeypatch, integration
):
    for name in (
        "FEM_RUN_NATIVE_TESTS",
        "FEM_RUN_GUI_NATIVE",
        "FEM_RUN_INTEGRATION_TESTS",
    ):
        monkeypatch.delenv(name, raising=False)
    if integration:
        monkeypatch.setenv("FEM_RUN_INTEGRATION_TESTS", "1")

    # Collect the real importer alongside native dependencies without executing
    # a native runtime. Pytester creates its files beneath pytest's test tree.
    probe = pytester.makepyfile(
        test_native_selection="""
        import pytest

        @pytest.fixture
        def native_model(real_gmsh):
            return real_gmsh

        @pytest.fixture
        def live_gmsh():
            pass

        @pytest.mark.gmsh
        def test_marked_native():
            pass

        def test_fixture_native(real_gmsh):
            pass

        def test_live_fixture_native(live_gmsh):
            pass

        def test_transitive_fixture_native(native_model):
            pass
        """
    )
    result = pytester.runpytest_subprocess(
        IMPORTER_NODEID,
        str(probe),
        "--collect-only", "-q",
        timeout=15,
    )

    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines([
        "*test_section_capabilities.py::test_real_importer_internal_section_set_uses_the_same_resolution",
        "1/5 tests collected (4 deselected)*",
    ])
