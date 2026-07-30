from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QWidget

import fem_gui.agent_workspace as workspace_module
from fem_gui.agent_workspace import (
    WorkspaceCommandHandler,
    WorkspaceIndexer,
    WorkspacePathError,
    build_workspace_file_reference,
    normalize_user_workspace,
)
from fem_gui.widgets.agent_chat import (
    AgentChatDrawer,
    ModelViewportOverlayHost,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _handler(
    workspace_root: Path,
    choices: list[str | Path | None],
    *,
    indexer: WorkspaceIndexer | None = None,
) -> WorkspaceCommandHandler:
    remaining = iter(choices)
    return WorkspaceCommandHandler(
        directory_chooser=lambda _parent, _initial: next(remaining),
        indexer=indexer,
        agent_data_root=workspace_root.parent / "agent-private-data",
    )


def _inventory(root: Path) -> set[tuple[str, bool, int]]:
    return {
        (
            path.relative_to(root).as_posix(),
            path.is_dir(),
            0 if path.is_dir() else path.stat().st_size,
        )
        for path in root.rglob("*")
    }


class _ViewportProbe(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.mouse_presses = 0

    def mousePressEvent(self, event) -> None:
        self.mouse_presses += 1
        event.accept()


def test_workspace_menu_and_slash_share_handler_and_cancel_is_stable(
    tmp_path,
):
    application = _application()
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "model.inp").write_text("*Heading\n", encoding="utf-8")
    handler = _handler(workspace, ["", workspace])
    drawer = AgentChatDrawer(workspace_commands=handler)
    drawer.resize(420, 680)
    drawer.show()
    application.processEvents()

    initial_label = drawer.workspace_state.text()
    drawer.workspace_action.trigger()
    assert handler.execution_count == 1
    assert handler.user_workspace is None
    assert drawer.workspace_state.text() == initial_label

    message_spy = QSignalSpy(drawer.messagePreviewRequested)
    drawer.input.setPlainText("/workspace")
    drawer.send_button.click()
    application.processEvents()

    assert handler.execution_count == 2
    assert handler.user_workspace is not None
    assert handler.user_workspace.root == workspace.resolve()
    assert drawer.workspace_state.text() == f"工作区  {workspace.name}"
    assert drawer.composer_hint.text() == "Enter 发送 · Shift+Enter 换行"
    assert drawer.input.toPlainText() == ""
    assert message_spy.count() == 0
    assert [action.text() for action in drawer.add_menu.actions()] == [
        "选择工作区…",
    ]
    drawer.close()


def test_at_candidates_filter_all_extensions_and_insert_typed_reference(
    tmp_path,
):
    application = _application()
    workspace = tmp_path / "mixed-files"
    nested = workspace / "docs"
    nested.mkdir(parents=True)
    (workspace / "frame.inp").write_text("*Heading\n", encoding="utf-8")
    (nested / "设计说明.md").write_text("说明", encoding="utf-8")
    (workspace / "preview.png").write_bytes(b"\x89PNG\r\n")
    (workspace / "LICENSE").write_text("sample", encoding="utf-8")
    handler = _handler(workspace, [workspace])
    drawer = AgentChatDrawer(workspace_commands=handler)
    drawer.resize(440, 700)
    drawer.show()
    drawer.workspace_action.trigger()
    application.processEvents()

    drawer.input.setFocus()
    drawer.input.setPlainText("@")
    application.processEvents()
    candidate_names = {
        drawer.suggestion_list.item(row).text()
        for row in range(drawer.suggestion_list.count())
    }
    assert {
        "frame.inp",
        "docs/设计说明.md",
        "preview.png",
        "LICENSE",
    } <= candidate_names

    drawer.input.setPlainText("@frame")
    drawer.input.setFocus()
    application.processEvents()
    assert drawer.suggestion_list.count() == 1
    QTest.keyClick(drawer.input, Qt.Key.Key_Return)
    assert drawer.input.toPlainText() == "@frame.inp "
    references = drawer.workspace_file_references
    assert len(references) == 1
    reference = references[0]
    assert reference.workspace_id == handler.user_workspace.workspace_id
    assert reference.workspace_root == str(workspace.resolve())
    assert reference.relative_path == "frame.inp"
    assert reference.filename == "frame.inp"
    assert reference.file_type == "inp"
    assert reference.size_bytes == (workspace / "frame.inp").stat().st_size
    assert reference.modified_time_ns > 0
    assert reference.metadata_version
    assert reference.mention_start == 0
    assert reference.mention_end == len("@frame.inp")

    drawer.input.clear()
    drawer.input.setPlainText("@设计")
    drawer.input.setFocus()
    application.processEvents()
    item_rect = drawer.suggestion_list.visualItemRect(
        drawer.suggestion_list.item(0)
    )
    QTest.mouseClick(
        drawer.suggestion_list.viewport(),
        Qt.MouseButton.LeftButton,
        pos=item_rect.center(),
    )
    assert drawer.input.toPlainText() == "@docs/设计说明.md "
    assert (
        drawer.workspace_file_references[0].relative_path
        == "docs/设计说明.md"
    )
    drawer.close()


def test_preview_preserves_reference_ranges_in_emitted_text(tmp_path):
    application = _application()
    workspace = tmp_path / "reference-ranges"
    workspace.mkdir()
    (workspace / "frame.inp").write_text("*Heading\n", encoding="utf-8")
    handler = _handler(workspace, [workspace])
    drawer = AgentChatDrawer(workspace_commands=handler)
    drawer.show()
    drawer.workspace_action.trigger()
    drawer.input.setPlainText("  @frame")
    drawer.input.setFocus()
    application.processEvents()
    QTest.keyClick(drawer.input, Qt.Key.Key_Return)

    message_spy = QSignalSpy(drawer.messagePreviewRequested)
    drawer.send_button.click()

    assert message_spy.count() == 1
    emitted_text, emitted_references = message_spy.at(0)
    assert emitted_text == "  @frame.inp "
    assert len(emitted_references) == 1
    reference = emitted_references[0]
    assert (
        emitted_text[reference.mention_start : reference.mention_end]
        == reference.mention_text
    )
    drawer.close()


def test_reference_rejects_path_escape(tmp_path):
    workspace_root = tmp_path / "workspace"
    outside_root = tmp_path / "outside"
    workspace_root.mkdir()
    outside_root.mkdir()
    (workspace_root / "safe.txt").write_text("safe", encoding="utf-8")
    outside_file = outside_root / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    workspace = normalize_user_workspace(workspace_root)

    with pytest.raises(WorkspacePathError, match="超出用户工作区"):
        build_workspace_file_reference(
            workspace,
            Path("..") / "outside" / "secret.txt",
        )


def test_index_does_not_follow_real_directory_link(tmp_path):
    workspace_root = tmp_path / "workspace"
    outside_root = tmp_path / "outside"
    workspace_root.mkdir()
    outside_root.mkdir()
    (workspace_root / "safe.txt").write_text("safe", encoding="utf-8")
    (outside_root / "secret.txt").write_text("secret", encoding="utf-8")
    workspace = normalize_user_workspace(workspace_root)
    link = workspace_root / "outside-link"
    try:
        link.symlink_to(outside_root, target_is_directory=True)
    except OSError:
        pytest.skip(
            "[platform-capability] 当前 Windows 权限不允许创建符号链接"
        )
    snapshot = WorkspaceIndexer().scan(workspace)
    assert [item.relative_path for item in snapshot.files] == ["safe.txt"]
    assert snapshot.skipped_unsafe_entries >= 1


def test_index_skips_link_and_reparse_metadata(
    tmp_path,
    monkeypatch,
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = normalize_user_workspace(workspace_root)

    @dataclass
    class _FakeStat:
        st_mode: int
        st_file_attributes: int = 0

    class _FakeEntry:
        def __init__(self, name: str, file_stat: _FakeStat) -> None:
            self.path = str(workspace_root / name)
            self._file_stat = file_stat

        def stat(self, *, follow_symlinks: bool):
            assert not follow_symlinks
            return self._file_stat

    class _FakeScandir:
        def __init__(self, entries) -> None:
            self._entries = entries

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def __iter__(self):
            return iter(self._entries)

    entries = (
        _FakeEntry("symbolic", _FakeStat(stat.S_IFLNK)),
        _FakeEntry(
            "junction",
            _FakeStat(
                stat.S_IFDIR,
                workspace_module._WINDOWS_REPARSE_POINT_ATTRIBUTE,
            ),
        ),
    )
    monkeypatch.setattr(
        workspace_module.os,
        "scandir",
        lambda _directory: _FakeScandir(entries),
    )

    snapshot = WorkspaceIndexer().scan(workspace)
    assert snapshot.files == ()
    assert snapshot.skipped_unsafe_entries == 2


def test_selection_reads_only_metadata_and_writes_nothing(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "read-only-context"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("do not read", encoding="utf-8")
    (workspace / ".kept").write_text("existing", encoding="utf-8")
    before = _inventory(workspace)
    private_root = tmp_path / "agent-private-data"
    handler = _handler(workspace, [workspace])

    def fail_content_read(*_args, **_kwargs):
        raise AssertionError("工作区索引不应读取文件内容")

    monkeypatch.setattr(Path, "read_text", fail_content_read)
    monkeypatch.setattr(Path, "read_bytes", fail_content_read)
    monkeypatch.setattr(Path, "open", fail_content_read)
    result = handler.execute("/workspace")

    assert result.succeeded
    assert _inventory(workspace) == before
    assert not private_root.exists()
    assert handler.agent_data_root == private_root.resolve(strict=False)
    assert handler.user_workspace.root == workspace.resolve()
    assert handler.agent_data_root != handler.user_workspace.root


@pytest.mark.parametrize("workspace_position", ("same", "parent", "child"))
def test_selection_rejects_agent_data_directory_overlap(
    tmp_path,
    workspace_position,
):
    agent_data_root = tmp_path / "private" / "fem-agent"
    agent_data_root.mkdir(parents=True)
    if workspace_position == "same":
        selected_workspace = agent_data_root
    elif workspace_position == "parent":
        selected_workspace = agent_data_root.parent
    else:
        selected_workspace = agent_data_root / "nested-workspace"
        selected_workspace.mkdir()

    handler = WorkspaceCommandHandler(
        directory_chooser=lambda _parent, _initial: selected_workspace,
        agent_data_root=agent_data_root,
    )
    result = handler.execute("/workspace")

    assert not result.succeeded
    assert not result.cancelled
    assert result.error == "用户工作区不能与 Agent 私有数据目录重叠"
    assert handler.user_workspace is None
    assert handler.workspace_index is None


def test_bounded_index_reports_truncation_to_ui(tmp_path):
    application = _application()
    workspace = tmp_path / "bounded"
    workspace.mkdir()
    for number in range(5):
        (workspace / f"file-{number}.txt").write_text(
            str(number),
            encoding="utf-8",
        )
    handler = _handler(
        workspace,
        [workspace],
        indexer=WorkspaceIndexer(max_entries=2, max_files=20),
    )
    drawer = AgentChatDrawer(workspace_commands=handler)
    drawer.workspace_action.trigger()
    drawer.input.setPlainText("@")
    application.processEvents()

    assert handler.workspace_index.truncated
    assert len(handler.workspace_index.files) <= 2
    assert drawer.workspace_state.text() == f"工作区  {workspace.name}"
    assert "个文件" not in drawer.workspace_state.text()
    assert "索引已截断" not in drawer.workspace_state.text()
    assert "已选择工作区" not in drawer.composer_hint.text()
    assert drawer.suggestion_title.text() == ""
    assert drawer.suggestion_title.isHidden()


def test_workspace_and_reference_controls_preserve_viewport_geometry(
    tmp_path,
):
    application = _application()
    workspace = tmp_path / "geometry-context"
    workspace.mkdir()
    (workspace / "model.step").write_text("STEP", encoding="utf-8")
    handler = _handler(workspace, [workspace])
    viewport = _ViewportProbe()
    host = ModelViewportOverlayHost(
        viewport,
        workspace_commands=handler,
    )
    host.resize(760, 480)
    host.show()
    application.processEvents()
    baseline_geometry = QRect(viewport.geometry())
    baseline_host_geometry = QRect(host.geometry())

    drawer = host.agent_chat_drawer
    drawer.workspace_action.trigger()
    drawer.input.setPlainText("@")
    drawer.input.setFocus()
    application.processEvents()
    item_rect = drawer.suggestion_list.visualItemRect(
        drawer.suggestion_list.item(0)
    )
    QTest.mouseClick(
        drawer.suggestion_list.viewport(),
        Qt.MouseButton.LeftButton,
        pos=item_rect.center(),
    )
    application.processEvents()

    assert viewport.geometry() == baseline_geometry
    assert host.geometry() == baseline_host_geometry
    assert viewport.mouse_presses == 0
    assert drawer.input.toPlainText() == "@model.step "
    host.close()
