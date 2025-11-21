from PySide6.QtCore import QThread, Signal

from .eye_tracking_source import NeonLocal


class DeviceConnectionWorker(QThread):
    success = Signal(object)
    failure = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            device = NeonLocal()
        except Exception:
            self.failure.emit()
            return

        self.success.emit(device)
