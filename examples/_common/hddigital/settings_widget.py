from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)


class ConnectionForm(QWidget):
    connection_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("<center><b>Connect to Device</b></center>"))

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._on_connect_clicked)
        main_layout.addWidget(self.connect_button)

        self.setLayout(main_layout)

    def _on_connect_clicked(self):
        self.connection_requested.emit()


class ConnectedDeviceInfo(QWidget):
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.label = QLabel("<center><b>Connected!</b></center>")
        layout.addWidget(self.label)

        self.button = QPushButton("Disconnect")
        self.button.clicked.connect(lambda: self.disconnect_requested.emit())
        layout.addWidget(self.button)

        self.setLayout(layout)

    def set_connected(self):
        self.label.setText("<center><b>Connected!</b></center>")
        self.button.setText("Disconnect")

    def set_connection_failure(self):
        self.label.setText("<center><b>Connection Failed</b></center>")
        self.button.setText("Back")


class SettingsWidget(QWidget):
    connection_requested = Signal()
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.connection_form = ConnectionForm()
        self.connection_form.connection_requested.connect(
            lambda: self.connection_requested.emit()
        )

        self.connected_device_info = ConnectedDeviceInfo()
        self.connected_device_info.disconnect_requested.connect(
            self.on_disconnect_requested
        )

        self.stacked_layout = QStackedLayout()
        self.stacked_layout.addWidget(self.connection_form)
        self.stacked_layout.addWidget(self.connected_device_info)
        self.setLayout(self.stacked_layout)

    def set_connected(self):
        self.connected_device_info.set_connected()
        self.stacked_layout.setCurrentWidget(self.connected_device_info)

    def set_connection_failure(self):
        self.connected_device_info.set_connection_failure()
        self.stacked_layout.setCurrentWidget(self.connected_device_info)

    def on_disconnect_requested(self):
        self.stacked_layout.setCurrentWidget(self.connection_form)
        self.disconnect_requested.emit()
