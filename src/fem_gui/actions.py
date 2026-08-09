"""Build shared Qt actions from the single Qt-free descriptor registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtGui import QAction, QActionGroup

from .action_state import ACTION_DESCRIPTORS, GuiActionDescriptor, GuiActionKey
from .icons import icon


def build_actions(owner: Any) -> dict[str, QAction]:
    """Create menu, Ribbon, and viewport actions from the canonical catalog."""

    keys = tuple(descriptor.key for descriptor in ACTION_DESCRIPTORS)
    if len(keys) != len(set(keys)) or set(keys) != set(GuiActionKey):
        raise RuntimeError("action descriptor registry is incomplete or duplicated")

    actions: dict[str, QAction] = {}
    groups: dict[str, QActionGroup] = {}
    for descriptor in ACTION_DESCRIPTORS:
        action = (
            QAction(icon(descriptor.icon_name), descriptor.text, owner)
            if descriptor.icon_name
            else QAction(descriptor.text, owner)
        )
        key = descriptor.key.value
        action.setObjectName(f"action_{key}")
        action.setToolTip(descriptor.text)
        action.setStatusTip(descriptor.text)
        action.setCheckable(descriptor.checkable)
        action.setChecked(descriptor.checked)
        action.triggered.connect(_callback(owner, descriptor))
        actions[key] = action
        if descriptor.group is not None:
            group = groups.get(descriptor.group)
            if group is None:
                group = QActionGroup(owner)
                group.setObjectName(f"{descriptor.group}_action_group")
                group.setExclusive(True)
                groups[descriptor.group] = group
            group.addAction(action)
    return actions


def _callback(owner: Any, descriptor: GuiActionDescriptor) -> Callable[..., Any]:
    if descriptor.argument is None:
        handler = _resolve_handler(owner, descriptor.handler)
        if descriptor.checkable:
            return handler
        # QAction.triggered always emits ``checked``; plain commands do not
        # own that value, even when their handler has an optional argument.
        return lambda _checked=False: handler()
    if descriptor.checked_only:
        return lambda checked=False: (
            _resolve_handler(owner, descriptor.handler)(descriptor.argument)
            if checked
            else None
        )
    return lambda _checked=False: _resolve_handler(
        owner,
        descriptor.handler,
    )(descriptor.argument)


def _resolve_handler(owner: Any, path: str) -> Callable[..., Any]:
    target = owner
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    return getattr(target, parts[-1])


__all__ = ["build_actions"]
