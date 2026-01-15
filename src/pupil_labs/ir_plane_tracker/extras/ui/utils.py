from PySide6.QtGui import QImage, QPixmap  # noqa: I001


import numpy as np
import numpy.typing as npt
import qimage2ndarray


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


def numpy_from_qpixmap(pixmap: QPixmap) -> npt.NDArray[np.uint8]:
    arr = qimage2ndarray.rgb_view(pixmap.toImage())
    arr = arr[..., :3]
    arr = arr[:, :, ::-1]  # RGB to BGR
    arr = arr.copy()
    return arr
