from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .settings_controler import SettingsController
from .settings_widget import SettingsWidget


class HDDigitalWidget(QWidget):
    new_device_connected = Signal(object)
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_widget = SettingsWidget()
        self.settings_controller = SettingsController(self.settings_widget)
        self.settings_controller.new_device_connected.connect(
            lambda device: self.new_device_connected.emit(device)
        )
        self.settings_controller.disconnect_requested.connect(
            lambda: self.disconnect_requested.emit()
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.settings_widget)
        self.setLayout(layout)
