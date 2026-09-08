from __future__ import annotations

from pathlib import Path

import pytest


pytest_plugins = ["pytester"]
TESTS_ROOT = Path(__file__).resolve().parents[1]


def test_pure_gui_rules_run_without_importing_qt(pytester, monkeypatch):
    pytester.makepyfile(
        block_qt="""
        import sys

        class BlockQt:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "PySide6" or fullname.startswith("PySide6."):
                    raise AssertionError("Pure GUI rules must not import Qt")

        assert "PySide6" not in sys.modules
        sys.meta_path.insert(0, BlockQt())
        """
    )
    monkeypatch.setenv("PYTHONPATH", str(pytester.path))
    result = pytester.runpytest_subprocess(
        "-p", "block_qt", str(TESTS_ROOT / "gui" / "test_action_projection.py"),
        "-q", timeout=30,
    )
    assert result.ret == pytest.ExitCode.OK
    assert result.parseoutcomes()["passed"] > 0


def test_gui_opt_in_reuses_application_and_cleans_each_test(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    pytester.makeini(
        f"[pytest]\npythonpath = {TESTS_ROOT.parent.as_posix()}\n"
        "addopts = -p no:cacheprovider\n"
    )
    pytester.makeconftest(
        (TESTS_ROOT / "gui" / "conftest.py").read_text(encoding="utf-8")
    )
    pytester.makepyfile(
        test_lifecycle="""
        import pytest
        from PySide6.QtWidgets import QApplication, QDialog, QWidget
        from shiboken6 import isValid

        previous = {}

        @pytest.fixture
        def stub_dialog(gui_application, monkeypatch):
            # The runtime guard must run before a test fixture's own stub.
            monkeypatch.setattr(QDialog, "exec", lambda self: 42)

        def test_first(gui_application, stub_dialog):
            previous["application"] = gui_application
            previous["widget"] = QWidget()
            previous["widget"].show()
            assert QDialog().exec() == 42

        def test_second(gui_application):
            assert gui_application is previous["application"]
            assert QApplication.instance() is gui_application
            assert not isValid(previous["widget"])
            previous["widget"] = QWidget()

        def test_dispose_dependency(dispose_gui_widget):
            assert not isValid(previous["widget"])
            assert QApplication.instance() is previous["application"]
            widget = QWidget()
            dispose_gui_widget(widget)
            assert not isValid(widget)
        """
    )
    result = pytester.runpytest_subprocess(
        "--confcutdir", str(pytester.path), "-q", timeout=30,
    )
    result.assert_outcomes(passed=3)


def test_gui_opt_in_reports_unstubbed_modal_dialogs(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    pytester.makeini(
        f"[pytest]\npythonpath = {TESTS_ROOT.parent.as_posix()}\n"
        "addopts = -p no:cacheprovider\n"
    )
    pytester.makeconftest(
        (TESTS_ROOT / "gui" / "conftest.py").read_text(encoding="utf-8")
    )
    pytester.makepyfile(
        test_modal="""
        from PySide6.QtWidgets import QDialog

        def test_modal(gui_application):
            dialog = QDialog()
            assert dialog.exec() == QDialog.DialogCode.Rejected
        """
    )
    result = pytester.runpytest_subprocess(
        "--confcutdir", str(pytester.path), "-q", timeout=30,
    )
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines([
        "*GUI tests must stub modal dialogs before opening them: QDialog.exec*",
    ])
