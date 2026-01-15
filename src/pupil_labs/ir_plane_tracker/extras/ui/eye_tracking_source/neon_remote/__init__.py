from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QVBoxLayout

from ..eye_tracking_source_widget import EyeTrackingSourceWidget
from .settings_controler import SettingsController
from .settings_widget import SettingsWidget


class NeonRemoteWidget(EyeTrackingSourceWidget):
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

    def showEvent(self, event: QShowEvent) -> None:
        self.settings_controller.on_device_search_requested()
        return super().showEvent(event)
