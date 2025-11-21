from PySide6.QtCore import QObject, Signal

from ..camera.camera import CameraNotFoundError
from .eye_tracking_source import HDDigital
from .settings_widget import SettingsWidget


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
        try:
            device = HDDigital()
        except CameraNotFoundError:
            self.settings_widget.set_connection_failure()
            return
        self.settings_widget.set_connected()
        self.new_device_connected.emit(device)
