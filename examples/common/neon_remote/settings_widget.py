from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ConnectionForm(QWidget):
    device_search_requested = Signal()
    connection_requested = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("<center><b>Connect to Device</b></center>"))

        form_layout = QFormLayout()

        device_selector_layout = QHBoxLayout()
        device_selector_layout.setContentsMargins(0, 0, 0, 0)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Manual Selection")
        self.device_combo.currentIndexChanged.connect(self._on_selected_device_changed)

        self.refresh_button = QToolButton()
        self.refresh_button.setText("🔍")
        self.refresh_button.clicked.connect(lambda: self.device_search_requested.emit())

        device_selector_layout.addWidget(self.device_combo)
        device_selector_layout.addWidget(self.refresh_button)

        form_layout.addRow("Device", device_selector_layout)

        self.phone_ip = QLineEdit()
        self.phone_ip.setText("neon.local")
        form_layout.addRow("Host/IP", self.phone_ip)

        self.manual_port = QSpinBox()
        self.manual_port.setRange(1, 65536)
        self.manual_port.setValue(8080)
        form_layout.addRow("Port", self.manual_port)

        main_layout.addLayout(form_layout)
        main_layout.addStretch(1)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._on_connect_clicked)
        main_layout.addWidget(self.connect_button)

        self.setLayout(main_layout)

    def _on_connect_clicked(self):
        host = self.phone_ip.text()
        port = self.manual_port.value()
        self.connection_requested.emit(host, port)

    def clear(self):
        self.device_combo.clear()
        self.phone_ip.clear()
        self.manual_port.setValue(8080)

    def addItem(self, text: str, user_data=None):
        self.device_combo.addItem(text, user_data)
        if self.device_combo.count() >= 2:
            self.device_combo.setCurrentIndex(1)

    def _on_selected_device_changed(self, index: int):
        if index <= 0:
            self.phone_ip.setEnabled(True)
            self.phone_ip.clear()
            self.manual_port.setEnabled(True)
            self.manual_port.setValue(8080)
        else:
            device_info = self.device_combo.itemData(index)
            self.phone_ip.setText(device_info["phone_ip"])
            self.phone_ip.setEnabled(False)
            self.manual_port.setValue(device_info["port"])
            self.manual_port.setEnabled(False)


class ConnectedDeviceInfo(QWidget):
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.label = QLabel("<center><b>Connected!</b></center>")
        layout.addWidget(self.label)

        form_layout = QFormLayout()
        form_layout.setHorizontalSpacing(20)

        self.device_ip_label = QLabel()
        form_layout.addRow("IP", self.device_ip_label)

        self.device_port_label = QLabel()
        form_layout.addRow("Port", self.device_port_label)

        layout.addLayout(form_layout)
        layout.addStretch(1)

        self.button = QPushButton("Disconnect")
        self.button.clicked.connect(lambda: self.disconnect_requested.emit())
        layout.addWidget(self.button)

        self.setLayout(layout)

    def set_connected(self, host: str, port: int):
        self.label.setText("<center><b>Connected!</b></center>")
        self.device_ip_label.setText(host)
        self.device_port_label.setText(str(port))
        self.button.setText("Disconnect")

    def set_connection_failure(self, host: str, port: int):
        self.label.setText("<center><b>Connection Failed</b></center>")
        self.device_ip_label.setText(host)
        self.device_port_label.setText(str(port))
        self.button.setText("Back")


class SettingsWidget(QWidget):
    device_search_requested = Signal()
    connection_requested = Signal(str, int)
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.connection_form = ConnectionForm()
        self.connection_form.device_search_requested.connect(
            lambda: self.device_search_requested.emit()
        )
        self.connection_form.connection_requested.connect(
            lambda host, port: self.connection_requested.emit(host, port)
        )

        self.connected_device_info = ConnectedDeviceInfo()
        self.connected_device_info.disconnect_requested.connect(
            self.on_disconnect_requested
        )

        self.stacked_layout = QStackedLayout()
        self.stacked_layout.addWidget(self.connection_form)
        self.stacked_layout.addWidget(self.connected_device_info)
        self.setLayout(self.stacked_layout)

    def update_device_list(self, device_infos: list[dict]):
        self.connection_form.device_combo.clear()
        self.connection_form.device_combo.addItem("Manual Entry")
        for device_info in device_infos:
            display_name = f"{device_info['phone_name']} ({device_info['dns_name']})"
            self.connection_form.addItem(display_name, device_info)

        self.connection_form.setDisabled(False)

    def set_to_search_devices(self):
        self.connection_form.clear()
        self.connection_form.addItem("Searching...")
        self.connection_form.setDisabled(True)

    def set_connected(self, host: str, port: int):
        self.connected_device_info.set_connected(host, port)
        self.stacked_layout.setCurrentWidget(self.connected_device_info)

    def set_connection_failure(self, host: str, port: int):
        self.connected_device_info.set_connection_failure(host, port)
        self.stacked_layout.setCurrentWidget(self.connected_device_info)

    def on_disconnect_requested(self):
        self.stacked_layout.setCurrentWidget(self.connection_form)
        self.disconnect_requested.emit()
