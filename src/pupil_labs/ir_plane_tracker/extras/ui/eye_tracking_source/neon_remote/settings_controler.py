from PySide6.QtCore import QObject, Signal

from pupil_labs.realtime_api.simple import Device

from . import workers
from .settings_widget import (
    SettingsWidget,
)


class SettingsController(QObject):
    new_device_connected = Signal(object)
    disconnect_requested = Signal()

    def __init__(self, settings_widget: SettingsWidget, parent=None):
        super().__init__(parent)
        self.settings_widget = settings_widget

        self.device_discovery_worker = workers.DeviceDiscoveryWorker()
        self.device_discovery_worker.finished.connect(self.on_devices_discovered)

        self.device_connection_worker = None

        self.settings_widget.device_search_requested.connect(
            self.on_device_search_requested
        )
        self.settings_widget.connection_requested.connect(self.on_connection_requested)
        self.settings_widget.disconnect_requested.connect(
            lambda: self.disconnect_requested.emit()
        )

    def on_devices_discovered(self, devices):
        device_infos = [
            {
                "phone_ip": device.phone_ip,
                "phone_name": device.phone_name,
                "address": device.address,
                "dns_name": device.dns_name,
                "full_name": device.full_name,
                "port": device.port,
            }
            for device in devices
        ]
        self.settings_widget.update_device_list(device_infos)

    def on_device_search_requested(self):
        self.settings_widget.set_to_search_devices()
        self.device_discovery_worker.start()

    def on_connection_requested(self, host: str, port: int):
        self.settings_widget.setDisabled(True)
        self.device_connection_worker = workers.DeviceConnectionWorker(host, port)
        self.device_connection_worker.success.connect(self.on_device_connected)
        self.device_connection_worker.failure.connect(self.on_device_connection_failed)
        self.device_connection_worker.start()

    def on_device_connected(self, device: Device):
        self.device_connection_worker = None
        self.settings_widget.setDisabled(False)
        self.settings_widget.set_connected(device.address, device.port)
        self.new_device_connected.emit(device)

    def on_device_connection_failed(self, phone_ip: str, port: int):
        self.device_connection_worker = None
        self.settings_widget.setDisabled(False)
        self.settings_widget.set_connection_failure(phone_ip, port)
