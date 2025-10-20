import numpy as np
import numpy.typing as npt
from PySide6.QtGui import QImage


def qimage_from_numpy(frame: npt.NDArray[np.uint8], pix_format=None):
    if frame is None:
        return QImage()

    if len(frame.shape) == 2:
        height, width = frame.shape
        channel = 1
        image_format = QImage.Format_Grayscale8
    else:
        height, width, channel = frame.shape
        image_format = QImage.Format_BGR888

    if pix_format is not None:
        image_format = pix_format

    bytes_per_line = channel * width

    return QImage(frame.data, width, height, bytes_per_line, image_format)
