from PySide6.QtCore import QThread, Signal

from pupil_labs.ir_plane_tracker.extras.eye_tracking_sources.neon_remote import NeonRemote
from pupil_labs.realtime_api.simple import discover_devices


class DeviceDiscoveryWorker(QThread):
    finished = Signal(object)

    def run(self):
        devices = discover_devices(search_duration_seconds=1.0)
        self.finished.emit(devices)


class DeviceConnectionWorker(QThread):
    success = Signal(object)
    failure = Signal(str, int)

    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port

    def run(self):
        try:
            device = NeonRemote(self.host, self.port)
            self.success.emit(device)
        except Exception as e:
            self.failure.emit(self.host, self.port)
