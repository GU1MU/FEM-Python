"""FEM Agent GUI 的本地工作区边界。

本模块只管理用户工作区路径、有限文件元数据索引和本地 ``/workspace`` 命令。
它不读取文件内容、不创建 Agent 会话目录，也不依赖 ``fem_agent``。
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QFileDialog, QWidget


MAX_WORKSPACE_SCAN_ENTRIES = 2_500
MAX_WORKSPACE_INDEX_FILES = 500
MAX_VISIBLE_WORKSPACE_CANDIDATES = 40
_WINDOWS_REPARSE_POINT_ATTRIBUTE = 0x0400


class WorkspacePathError(ValueError):
    """工作区或候选文件不满足本地路径边界。"""


@dataclass(frozen=True, slots=True)
class UserWorkspace:
    """已规范化的用户工作区身份。"""

    workspace_id: str
    root: Path


@dataclass(frozen=True, slots=True)
class WorkspaceFileReference:
    """一项工作区文件元数据及其在输入文本中的引用位置。"""

    workspace_id: str
    workspace_root: str
    relative_path: str
    filename: str
    file_type: str
    size_bytes: int
    modified_time_ns: int
    metadata_version: str
    mention_start: int | None = None
    mention_end: int | None = None

    @property
    def mention_text(self) -> str:
        return f"@{self.relative_path}"

    def at_text_range(
        self,
        start: int,
        end: int,
    ) -> WorkspaceFileReference:
        return replace(
            self,
            mention_start=int(start),
            mention_end=int(end),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceIndexSnapshot:
    """一次有界扫描得到的不可变索引。"""

    workspace: UserWorkspace
    files: tuple[WorkspaceFileReference, ...]
    scanned_entries: int
    skipped_unsafe_entries: int
    truncated: bool

    def matching_files(
        self,
        query: str,
        *,
        limit: int = MAX_VISIBLE_WORKSPACE_CANDIDATES,
    ) -> tuple[WorkspaceFileReference, ...]:
        needle = query.strip().replace("\\", "/").casefold()
        matches = (
            reference
            for reference in self.files
            if not needle
            or needle in reference.relative_path.casefold()
            or needle in reference.filename.casefold()
        )
        return tuple(list(matches)[: max(0, int(limit))])


@dataclass(frozen=True, slots=True)
class WorkspaceSelectionResult:
    """本地工作区命令的结果；取消时保留原有状态。"""

    cancelled: bool
    workspace: UserWorkspace | None
    index: WorkspaceIndexSnapshot | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            not self.cancelled
            and self.error is None
            and self.workspace is not None
            and self.index is not None
        )


DirectoryChooser = Callable[[QWidget | None, str], str | Path | None]


def resolve_agent_data_root(
    explicit_root: str | os.PathLike[str] | None = None,
) -> Path:
    """返回应用私有 Agent 数据根；本函数不会创建目录。"""
    if explicit_root is None:
        local_data = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation
        )
        if not local_data:
            raise WorkspacePathError("Qt 未提供应用私有数据目录")
        explicit_root = Path(local_data) / "fem-agent"
    return Path(explicit_root).expanduser().resolve(strict=False)


def normalize_user_workspace(
    selected_path: str | os.PathLike[str],
) -> UserWorkspace:
    """验证并规范化一个已存在的用户目录。"""
    try:
        root = Path(selected_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkspacePathError("所选工作区不存在或无法访问") from exc
    if not root.is_dir():
        raise WorkspacePathError("所选工作区不是目录")

    identity_source = os.path.normcase(str(root)).replace("\\", "/")
    workspace_id = hashlib.sha256(
        identity_source.encode("utf-8")
    ).hexdigest()[:20]
    return UserWorkspace(workspace_id=workspace_id, root=root)


def _has_reparse_attribute(file_stat: os.stat_result) -> bool:
    attributes = int(getattr(file_stat, "st_file_attributes", 0))
    return bool(attributes & _WINDOWS_REPARSE_POINT_ATTRIBUTE)


def _is_unsafe_link(file_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(file_stat.st_mode) or _has_reparse_attribute(file_stat)


def _relative_path_inside(
    workspace_root: Path,
    candidate: Path,
) -> Path:
    absolute = Path(os.path.abspath(candidate))
    try:
        return absolute.relative_to(workspace_root)
    except ValueError as exc:
        raise WorkspacePathError("文件路径超出用户工作区") from exc


def build_workspace_file_reference(
    workspace: UserWorkspace,
    candidate: str | os.PathLike[str],
) -> WorkspaceFileReference:
    """从普通文件的状态元数据构造引用，不读取文件内容。"""
    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = workspace.root / candidate_path
    relative = _relative_path_inside(workspace.root, candidate_path)
    if not relative.parts:
        raise WorkspacePathError("工作区根目录不是可引用文件")

    current = workspace.root
    final_stat: os.stat_result | None = None
    for part in relative.parts:
        current = current / part
        try:
            final_stat = os.lstat(current)
        except OSError as exc:
            raise WorkspacePathError("文件不存在或无法访问") from exc
        if _is_unsafe_link(final_stat):
            raise WorkspacePathError("不能引用符号链接或 reparse point")

    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(workspace.root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise WorkspacePathError("文件解析后超出用户工作区") from exc
    if final_stat is None or not stat.S_ISREG(final_stat.st_mode):
        raise WorkspacePathError("只能引用普通文件")

    relative_text = relative.as_posix()
    file_type = current.suffix.casefold().lstrip(".") or "file"
    modified_time_ns = int(final_stat.st_mtime_ns)
    size_bytes = int(final_stat.st_size)
    version_source = (
        f"{size_bytes}:{modified_time_ns}:"
        f"{int(getattr(final_stat, 'st_dev', 0))}:"
        f"{int(getattr(final_stat, 'st_ino', 0))}"
    )
    metadata_version = hashlib.sha256(
        version_source.encode("ascii")
    ).hexdigest()[:20]
    return WorkspaceFileReference(
        workspace_id=workspace.workspace_id,
        workspace_root=str(workspace.root),
        relative_path=relative_text,
        filename=current.name,
        file_type=file_type,
        size_bytes=size_bytes,
        modified_time_ns=modified_time_ns,
        metadata_version=metadata_version,
    )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class WorkspaceIndexer:
    """只遍历有限数量目录项的普通文件元数据索引器。"""

    def __init__(
        self,
        *,
        max_entries: int = MAX_WORKSPACE_SCAN_ENTRIES,
        max_files: int = MAX_WORKSPACE_INDEX_FILES,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_files = max(1, int(max_files))

    def scan(
        self,
        workspace: UserWorkspace,
        *,
        excluded_roots: Sequence[Path] = (),
    ) -> WorkspaceIndexSnapshot:
        excluded = tuple(
            Path(path).resolve(strict=False)
            for path in excluded_roots
        )
        pending: deque[Path] = deque((workspace.root,))
        files: list[WorkspaceFileReference] = []
        scanned_entries = 0
        skipped_unsafe_entries = 0
        truncated = False

        while pending and not truncated:
            directory = pending.popleft()
            if directory != workspace.root:
                try:
                    directory_stat = os.lstat(directory)
                except OSError:
                    continue
                if _is_unsafe_link(directory_stat):
                    skipped_unsafe_entries += 1
                    continue
            if any(_path_is_within(directory, root) for root in excluded):
                continue

            try:
                iterator = os.scandir(directory)
            except OSError:
                continue
            with iterator:
                for entry in iterator:
                    if scanned_entries >= self.max_entries:
                        truncated = True
                        break
                    scanned_entries += 1
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if _is_unsafe_link(entry_stat):
                        skipped_unsafe_entries += 1
                        continue

                    entry_path = Path(entry.path)
                    if any(
                        _path_is_within(
                            entry_path.resolve(strict=False),
                            root,
                        )
                        for root in excluded
                    ):
                        continue
                    if stat.S_ISDIR(entry_stat.st_mode):
                        pending.append(entry_path)
                        continue
                    if not stat.S_ISREG(entry_stat.st_mode):
                        continue
                    try:
                        reference = build_workspace_file_reference(
                            workspace,
                            entry_path,
                        )
                    except WorkspacePathError:
                        skipped_unsafe_entries += 1
                        continue
                    files.append(reference)
                    if len(files) >= self.max_files:
                        truncated = True
                        break

        files.sort(key=lambda item: item.relative_path.casefold())
        return WorkspaceIndexSnapshot(
            workspace=workspace,
            files=tuple(files),
            scanned_entries=scanned_entries,
            skipped_unsafe_entries=skipped_unsafe_entries,
            truncated=truncated,
        )


def choose_workspace_directory(
    parent: QWidget | None,
    initial_directory: str,
) -> str:
    """Qt 目录选择边界，便于测试时注入无界面实现。"""
    return QFileDialog.getExistingDirectory(
        parent,
        "选择工作区",
        initial_directory,
        QFileDialog.Option.ShowDirsOnly,
    )


class WorkspaceCommandHandler:
    """``/workspace`` 与加号菜单共享的唯一工作区命令处理器。"""

    def __init__(
        self,
        *,
        directory_chooser: DirectoryChooser | None = None,
        indexer: WorkspaceIndexer | None = None,
        agent_data_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.directory_chooser = (
            directory_chooser or choose_workspace_directory
        )
        self.indexer = indexer or WorkspaceIndexer()
        self.agent_data_root = resolve_agent_data_root(agent_data_root)
        self.user_workspace: UserWorkspace | None = None
        self.workspace_index: WorkspaceIndexSnapshot | None = None
        self.execution_count = 0

    def execute(
        self,
        command: str,
        *,
        parent: QWidget | None = None,
    ) -> WorkspaceSelectionResult:
        """执行一个本地命令；当前只接受 ``/workspace``。"""
        if command.strip().casefold() != "/workspace":
            raise ValueError(f"未知本地命令: {command}")
        self.execution_count += 1
        initial_directory = (
            str(self.user_workspace.root)
            if self.user_workspace is not None
            else ""
        )
        selected = self.directory_chooser(parent, initial_directory)
        if selected is None or not str(selected).strip():
            return WorkspaceSelectionResult(
                cancelled=True,
                workspace=self.user_workspace,
                index=self.workspace_index,
            )

        try:
            workspace = normalize_user_workspace(selected)
            if (
                _path_is_within(workspace.root, self.agent_data_root)
                or _path_is_within(self.agent_data_root, workspace.root)
            ):
                raise WorkspacePathError(
                    "用户工作区不能与 Agent 私有数据目录重叠"
                )
            snapshot = self.indexer.scan(
                workspace,
                excluded_roots=(self.agent_data_root,),
            )
        except WorkspacePathError as exc:
            return WorkspaceSelectionResult(
                cancelled=False,
                workspace=self.user_workspace,
                index=self.workspace_index,
                error=str(exc),
            )

        self.user_workspace = workspace
        self.workspace_index = snapshot
        return WorkspaceSelectionResult(
            cancelled=False,
            workspace=workspace,
            index=snapshot,
        )
