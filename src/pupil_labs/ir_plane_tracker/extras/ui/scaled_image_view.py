import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from .utils import qimage_from_numpy


class ScaledImageView(QWidget):
    image: QImage | QPixmap | None

    def __init__(self, parent=None):
        super().__init__(parent)

        self.image = None
        self.render_rect = None
        self.margin = 0
        self.setMinimumSize(32, 32)
        self.last_image_size = None

    def resizeEvent(self, event):
        self.update_rect()

    def update_rect(self):
        if self.image is None:
            return

        self.render_rect = self.fit_rect(self.image.size())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(0, 0, self.width(), self.height(), Qt.black)

        if self.image is None or self.render_rect is None:
            return

        if isinstance(self.image, QImage):
            painter.drawImage(self.render_rect, self.image)

        elif isinstance(self.image, QPixmap):
            painter.drawPixmap(self.render_rect, self.image)

    def set_image(self, image):
        if isinstance(image, np.ndarray):
            image = qimage_from_numpy(image)

        self.image = image

        if self.image is None:
            return

        if self.last_image_size != self.image.size():
            self.update_rect()
            self.last_image_size = self.image.size()

        self.update()

    def fit_rect(self, source_size):
        if source_size.height() == 0:
            return QRect(0, 0, 1, 1)

        source_ratio = source_size.width() / source_size.height()
        target_ratio = self.width() / self.height()

        resultSize = QSize()

        if source_ratio > target_ratio:
            resultSize.setWidth(self.width() - self.margin * 2)
            resultSize.setHeight(
                source_size.height() * (resultSize.width() / source_size.width())
            )

        else:
            resultSize.setHeight(self.height() - self.margin * 2)
            resultSize.setWidth(
                source_size.width() * (resultSize.height() / source_size.height())
            )

        resultPos = QPoint(
            (self.width() - resultSize.width()) / 2.0,
            (self.height() - resultSize.height()) / 2.0,
        )

        return QRect(resultPos, resultSize)
