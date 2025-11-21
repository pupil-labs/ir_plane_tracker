from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .hddigital import HDDigitalWidget
from .neon_remote import NeonRemoteWidget


class SourceWidget(QWidget):
    new_device_connected = Signal(object)
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()

        self.neon_usb_widget = HDDigitalWidget()
        self.tabs.addTab(self.neon_usb_widget, "Neon USB")

        self.neon_remote_widget = NeonRemoteWidget()
        self.tabs.addTab(self.neon_remote_widget, "Neon Remote")

        self.hd_digital_widget = HDDigitalWidget()
        self.tabs.addTab(self.hd_digital_widget, "HD Digital")

        # Connections
        for widget in (
            self.neon_usb_widget,
            self.neon_remote_widget,
            self.hd_digital_widget,
        ):
            widget.new_device_connected.connect(
                lambda device: self.new_device_connected.emit(device)
            )
            widget.disconnect_requested.connect(
                lambda: self.disconnect_requested.emit()
            )

        layout.addWidget(self.tabs)
        self.setLayout(layout)
