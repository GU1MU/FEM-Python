"""Combo box that preserves the types of Python user data."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QWidget


class ExactDataComboBox(QComboBox):
    """Keep typed Python user data from being coerced by QVariant."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._exact_user_data: list[object] = []

    def addItem(self, text: str, user_data: object = None) -> None:
        super().addItem(text)
        self._exact_user_data.append(user_data)

    def clear(self) -> None:
        super().clear()
        self._exact_user_data.clear()

    def itemData(
        self,
        index: int,
        role: int = Qt.ItemDataRole.UserRole,
    ) -> object:
        if role == Qt.ItemDataRole.UserRole:
            if 0 <= index < len(self._exact_user_data):
                return self._exact_user_data[index]
            return None
        return super().itemData(index, role)

    def currentData(
        self,
        role: int = Qt.ItemDataRole.UserRole,
    ) -> object:
        return self.itemData(self.currentIndex(), role)

    def findData(
        self,
        data: object,
        role: int = Qt.ItemDataRole.UserRole,
        flags: Qt.MatchFlag = (
            Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchCaseSensitive
        ),
    ) -> int:
        if role == Qt.ItemDataRole.UserRole:
            return next(
                (
                    index
                    for index, candidate in enumerate(self._exact_user_data)
                    if type(candidate) is type(data) and candidate == data
                ),
                -1,
            )
        return super().findData(data, role, flags)


__all__ = ["ExactDataComboBox"]
