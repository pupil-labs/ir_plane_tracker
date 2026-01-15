from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget


class EyeTrackingSourceWidget(QWidget):
    new_device_connected = Signal(object)
    disconnect_requested = Signal()
    pass
