"""Single-line CAE context status bar."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QStatusBar


class CAEStatusBar(QStatusBar):
    """Display task, selection, object, coordinate, step, and result status."""

    cancelRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("caeStatusBar")
        self.setSizeGripEnabled(False)
        self.setFixedHeight(22)
        self._field_count = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(lambda: self.set_state("Ready"))
        self.state_label = self._add_field(
            "statusState",
            "Status: Ready",
            320,
            stretch=3,
        )
        self.selection_label = self._add_field("statusSelection", "Selection: Nodes", 120)
        self.object_label = self._add_field("statusObject", "Object: —", 130)
        self.coordinate_label = self._add_field("statusCoordinate", "Coordinates: —", 245)
        self.step_label = self._add_field("statusStep", "Step: —", 145)
        self.result_label = self._add_field("statusResult", "Results: —", 180)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setObjectName("cancelTaskButton")
        self.cancel_button.setMinimumWidth(
            self.cancel_button.fontMetrics().horizontalAdvance("Cancelling") + 20
        )
        self.cancel_button.setToolTip("Cancel the current background task")
        self.cancel_button.clicked.connect(self.cancelRequested)
        self.cancel_button.hide()
        self.addPermanentWidget(self.cancel_button)

    def _add_field(
        self,
        name: str,
        text: str,
        maximum: int,
        *,
        stretch: int = 1,
    ) -> QLabel:
        if self._field_count:
            separator = QFrame(self)
            separator.setObjectName("statusSeparator")
            separator.setFrameShape(QFrame.Shape.VLine)
            self.addWidget(separator)
        label = QLabel(text, self)
        label.setObjectName(name)
        label.setMinimumWidth(0)
        label.setMaximumWidth(maximum)
        self.addWidget(label, stretch)
        self._field_count += 1
        return label

    def set_state(self, text: str, timeout: int = 0) -> None:
        self._timer.stop()
        self.state_label.setText(f"Status: {text}")
        self.state_label.setToolTip(str(text))
        if timeout > 0:
            self._timer.start(timeout)

    def set_task_active(
        self,
        active: bool,
        *,
        cancelling: bool = False,
    ) -> None:
        self.cancel_button.setVisible(bool(active))
        self.cancel_button.setEnabled(bool(active) and not cancelling)
        self.cancel_button.setText("Cancelling" if cancelling else "Cancel")

    def set_selection_mode(self, mode: str) -> None:
        labels = {
            "node": "Nodes",
            "element": "Elements",
            "geometry_point": "Points",
            "geometry_edge": "Edges",
            "geometry_face": "Faces",
            "geometry_body": "Bodies",
            "mesh_node": "Nodes",
            "mesh_element": "Elements",
            "mesh_edge": "Edges",
            "mesh_face": "Faces",
            "mesh_body": "Bodies",
        }
        self.selection_label.setText(f"Selection: {labels.get(mode, 'Nodes')}")

    def set_object(self, text: str = "—", coordinates: str = "—") -> None:
        self.object_label.setText(f"Object: {text}")
        self.coordinate_label.setText(f"Coordinates: {coordinates}")

    def set_step(self, step_name: str | None) -> None:
        self.step_label.setText(f"Step: {step_name or '—'}")

    def set_result(self, text: str = "—") -> None:
        self.result_label.setText(f"Results: {text}")

    def reset_document(self) -> None:
        self.set_state("Ready")
        self.set_object()
        self.set_step(None)
        self.set_result()
