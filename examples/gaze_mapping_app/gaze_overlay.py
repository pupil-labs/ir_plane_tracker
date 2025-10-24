import platform
import numpy.typing as npt
from common.eye_tracking_sources import EyeTrackingData
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class GazeOverlay(QWidget):
    def __init__(self, target_screen):
        super().__init__()
        self.gaze = []

        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background:transparent;")
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setWindowFlag(Qt.WindowTransparentForInput)

        self.setGeometry(target_screen.geometry())
        self.move(target_screen.geometry().x(), target_screen.geometry().y())

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization | None,
        debug: DebugData,
        gaze_mapped: npt.NDArray | None,
    ):
        if gaze_mapped is not None:
            gaze_mapped = gaze_mapped * (self.size().width(), self.size().height())
        self.gaze = gaze_mapped
        self.update()

    def paintEvent(self, event):
        if self.isVisible():
            with QPainter(self) as painter:
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setBrush(Qt.NoBrush)
                pen = painter.pen()
                pen.setColor(Qt.green)
                pen.setWidth(5)
                painter.setPen(pen)
                radius = 40
                if self.gaze is not None:
                    x, y = int(self.gaze[0]), int(self.gaze[1])
                    painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            if platform.system() == "Darwin":
                self.showMaximized()
            else:
                self.showFullScreen()
            self.update()
