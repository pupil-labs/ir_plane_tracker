from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QStackedWidget, QVBoxLayout, QWidget

from .eye_tracking_source_widget import EyeTrackingSourceWidget
from .neon_remote import NeonRemoteWidget


class SourceWidget(QWidget):
    new_device_connected = Signal(object)
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)

        self.selector = QComboBox()
        self.form_stack = QStackedWidget()

        self.selector.currentIndexChanged.connect(self.form_stack.setCurrentIndex)
        self.selector.setVisible(False)

        self.neon_remote_widget = NeonRemoteWidget()
        self.add_source_option("Neon Remote", self.neon_remote_widget)

        layout.addWidget(self.selector)
        layout.addWidget(self.form_stack)

    def add_source_option(self, name: str, widget: EyeTrackingSourceWidget) -> None:
        self.selector.addItem(name)
        self.form_stack.addWidget(widget)

        widget.new_device_connected.connect(
            lambda device: self.new_device_connected.emit(device)
        )
        widget.disconnect_requested.connect(
            lambda: self.disconnect_requested.emit()
        )
