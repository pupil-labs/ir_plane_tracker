import sys

from PySide6.QtGui import QColor, QFont, QPainter, QPen, QShortcut, Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)


class Ruler(QWidget):
    def __init__(self, orientation, mm_per_pixel, length_mm, parent=None):
        super().__init__(parent)
        self.orientation = orientation  # 'top', 'bottom', 'left', 'right'
        self.mm_per_pixel = mm_per_pixel
        self.length_mm = length_mm
        self.setMinimumSize(20, 20)

    def paintEvent(self, event):  # noqa: C901
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#2b2b2b"))
        painter.setPen(QPen(QColor("#dddddd"), 1))

        font = QFont("Arial", 8)
        painter.setFont(font)

        tick_length_major = 10
        tick_length_minor = 5

        if self.orientation in ("top", "bottom"):
            width_px = self.width()
            step_mm = 10  # major tick every 10mm
            for mm in range(0, int(self.length_mm) + 1, step_mm):
                x = mm / self.mm_per_pixel
                if x > width_px:
                    break
                y1 = (
                    0
                    if self.orientation == "top"
                    else self.height() - tick_length_major
                )
                y2 = tick_length_major if self.orientation == "top" else self.height()
                painter.drawLine(x, y1, x, y2)
                painter.drawText(
                    x + 0, y2 + 10 if self.orientation == "top" else y1 - 2, f"{mm}"
                )
            # Minor ticks every 2mm
            for mm in range(0, int(self.length_mm) + 1, 2):
                if mm % step_mm == 0:
                    continue
                x = mm / self.mm_per_pixel
                if x > width_px:
                    break
                y1 = (
                    0
                    if self.orientation == "top"
                    else self.height() - tick_length_minor
                )
                y2 = tick_length_minor if self.orientation == "top" else self.height()
                painter.drawLine(x, y1, x, y2)

        else:  # left or right
            height_px = self.height()
            step_mm = 10
            for mm in range(0, int(self.length_mm) + 1, step_mm):
                y = mm / self.mm_per_pixel
                if y > height_px:
                    break
                x1 = (
                    0
                    if self.orientation == "left"
                    else self.width() - tick_length_major
                )
                x2 = tick_length_major if self.orientation == "left" else self.width()
                painter.drawLine(x1, y, x2, y)
                painter.save()
                painter.translate(
                    x2 + 10 if self.orientation == "left" else x1 + 2, y + 4
                )
                painter.rotate(-90)
                painter.drawText(0, 0, f"{mm}")
                painter.restore()
            # Minor ticks
            for mm in range(0, int(self.length_mm) + 1, 2):
                if mm % step_mm == 0:
                    continue
                y = mm / self.mm_per_pixel
                if y > height_px:
                    break
                x1 = (
                    0
                    if self.orientation == "left"
                    else self.width() - tick_length_minor
                )
                x2 = tick_length_minor if self.orientation == "left" else self.width()
                painter.drawLine(x1, y, x2, y)


class RulerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        QShortcut(Qt.Key_Escape, self, activated=self.close)  # type: ignore
        self.setWindowTitle("Screen Ruler")
        # screen = QApplication.primaryScreen()
        screen = QApplication.screens()[-1]
        self.setGeometry(screen.geometry())
        self.showFullScreen()

        width_mm = screen.physicalSize().width()
        height_mm = screen.physicalSize().height()
        width_px = screen.size().width()
        height_px = screen.size().height()

        # conversion factors
        mm_per_pixel_x = width_mm / width_px
        mm_per_pixel_y = height_mm / height_px

        # Layout setup
        central = QWidget()
        label = QLabel(
            f"{width_px} x {height_px} px\n{width_mm:.1f} x {height_mm:.1f} mm"
        )
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: black; font-size: 24px; background: transparent;")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        top_ruler = Ruler("top", mm_per_pixel_x, width_mm)
        bottom_ruler = Ruler("bottom", mm_per_pixel_x, width_mm)
        left_ruler = Ruler("left", mm_per_pixel_y, height_mm)
        right_ruler = Ruler("right", mm_per_pixel_y, height_mm)

        middle = QWidget()
        middle_layout = QHBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.addWidget(left_ruler)
        middle_layout.addStretch()
        middle_layout.addWidget(label)
        middle_layout.addStretch()
        middle_layout.addWidget(right_ruler)

        layout.addWidget(top_ruler)
        layout.addWidget(middle, 1)
        layout.addWidget(bottom_ruler)

        self.setCentralWidget(central)


def main():
    app = QApplication(sys.argv)
    window = RulerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
