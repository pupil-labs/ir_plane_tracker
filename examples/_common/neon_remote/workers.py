from PySide6.QtCore import QThread, Signal

from pupil_labs.realtime_api.simple import Device, discover_devices


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
            device = Device(self.host, self.port, start_streaming_by_default=True)
        except Exception:
            device = None

        if device is None:
            print(f"Failed to connect to device at {self.host}:{self.port}.")
            self.failure.emit(self.host, self.port)
            return

        data = None
        print(f"Attempting to receive data from device {device}...")
        counter = 0
        while data is None:
            counter += 1
            print(f"  Attempt {counter}...")
            data = device.receive_matched_scene_video_frame_and_gaze(
                timeout_seconds=1.0
            )
        print("  Success.")
        self.success.emit(device)
