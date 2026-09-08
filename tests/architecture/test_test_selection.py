from __future__ import annotations

from pathlib import Path

import pytest


pytest_plugins = ["pytester"]

TESTS_ROOT = Path(__file__).resolve().parents[1]
IMPORTER_NODEID = str(
    TESTS_ROOT / "materials" / "test_section_assignment.py"
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
        "*test_section_assignment.py::test_real_importer_internal_section_set_uses_the_same_resolution",
        "1/5 tests collected (4 deselected)*",
    ])


def test_gui_dependencies_run_by_default_without_legacy_classification(pytester, monkeypatch):
    legacy_flags = (
        "FEM_RUN_NATIVE_TESTS", "FEM_RUN_GUI_NATIVE",
        "FEM_RUN_SLOW_TESTS", "FEM_RUN_SLOW_PERF",
        "FEM_RUN_INTEGRATION_TESTS",
    )
    for name in legacy_flags:
        monkeypatch.delenv(name, raising=False)
    # Use the actual hook in an isolated test tree so fixture closure (including
    # indirect Gmsh use) is resolved by pytest, without loading Qt or native CAD.
    pytester.makeconftest((TESTS_ROOT / "conftest.py").read_text(encoding="utf-8"))
    gui = pytester.path / "gui"
    gui.mkdir()
    (gui / "test_selection.py").write_text(
        '''import pytest

@pytest.fixture
def real_gmsh():
    return object()

@pytest.fixture
def native_model(real_gmsh):
    return real_gmsh

@pytest.fixture
def live_gmsh():
    return object()

def test_plain(request):
    assert not {"gmsh", "integration", "slow", "gui_native"} & {
        marker.name for marker in request.node.iter_markers()
    }

def test_direct(real_gmsh, request):
    test_plain(request)

def test_indirect(native_model, request):
    test_plain(request)

@pytest.mark.usefixtures("real_gmsh")
def test_declared_dependency(request):
    test_plain(request)

def test_live_dependency(live_gmsh, request):
    test_plain(request)
''', encoding="utf-8",
    )
    # The fake runtime fixtures make this check independent of installed Gmsh.
    with (pytester.path / "conftest.py").open("a", encoding="utf-8") as stream:
        stream.write('''
def pytest_sessionstart(session):
    config = session.config
    original = importlib.util.find_spec
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    patch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs:
        object() if name == "gmsh" else original(name, *args, **kwargs))
''')
    for enabled in (False, True):
        if enabled:
            for name in legacy_flags:
                monkeypatch.setenv(name, "1")
        result = pytester.runpytest_subprocess(
            str(gui), "--confcutdir", str(pytester.path), "-q", timeout=15,
        )
        result.assert_outcomes(passed=5)


def test_gui_missing_gmsh_is_skipped_without_backend_classification(monkeypatch):
    from types import SimpleNamespace
    from tests import conftest as selection

    markers = []
    item = SimpleNamespace(
        path=TESTS_ROOT / "gui" / "test_dependency.py",
        fixturenames=("native_model", "real_gmsh"),
        get_closest_marker=lambda name: None,
        add_marker=markers.append,
    )
    monkeypatch.setattr(selection.importlib.util, "find_spec", lambda name: None)
    items = [item]
    selection.pytest_collection_modifyitems(SimpleNamespace(), items)
    assert items == [item]
    assert [marker.name for marker in markers] == ["skip"]
    assert "not installed" in markers[0].kwargs["reason"]
