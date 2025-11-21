from PySide6.QtCore import QObject, Signal

from .settings_widget import SettingsWidget
from .workers import DeviceConnectionWorker


class SettingsController(QObject):
    new_device_connected = Signal(object)
    disconnect_requested = Signal()

    def __init__(self, settings_widget: SettingsWidget, parent=None):
        super().__init__(parent)
        self.settings_widget = settings_widget
        self.settings_widget.connection_requested.connect(self.on_connection_requested)
        self.settings_widget.disconnect_requested.connect(
            lambda: self.disconnect_requested.emit()
        )

        self.device_connection_worker = None

    def on_connection_requested(self):
        self.settings_widget.setDisabled(True)
        self.device_connection_worker = DeviceConnectionWorker()
        self.device_connection_worker.success.connect(self.on_device_connected)
        self.device_connection_worker.failure.connect(self.on_connection_failure)
        self.device_connection_worker.start()

    def on_device_connected(self, device):
        self.settings_widget.setDisabled(False)
        self.settings_widget.set_connected()
        self.new_device_connected.emit(device)

    def on_connection_failure(self):
        self.settings_widget.setDisabled(False)
        self.settings_widget.set_connection_failure()
