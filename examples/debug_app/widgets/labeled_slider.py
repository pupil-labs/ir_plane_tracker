from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QSlider, QWidget


class LabeledSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(
        self, label_text="Value", minimum=0, maximum=255, initial=128, parent=None
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label_text)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(initial)
        self.slider.setTickInterval(1)
        self.value_edit = QLineEdit(str(initial))
        self.value_edit.setFixedWidth(60)
        self.value_edit.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_edit)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.value_edit.editingFinished.connect(self._on_edit_finished)

    def _on_slider_changed(self, value):
        self.value_edit.setText(str(value))
        self.valueChanged.emit(value)

    def _on_edit_finished(self):
        try:
            value = int(self.value_edit.text())
        except ValueError:
            value = self.slider.value()
        value = max(self.slider.minimum(), min(self.slider.maximum(), value))
        self.slider.setValue(value)
        self.value_edit.setText(str(value))

    def set_value(self, value: int) -> None:
        if value != self.slider.value():
            self.slider.setValue(value)
            self.value_edit.setText(str(value))
            self.valueChanged.emit(value)

    def value(self) -> int:
        return self.slider.value()
