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
    # channels_count = 4
    # image = pixmap.toImage()
    # size = pixmap.size()
    # h = size.width()
    # w = size.height()
    # s = image.bits().asstring(w * h * channels_count)
    # arr = np.fromstring(s, dtype=np.uint8).reshape((h, w, channels_count))

    arr = qimage2ndarray.rgb_view(pixmap.toImage())

    # Previous custom implementation
    # size = pixmap.size()
    # h = size.width()
    # w = size.height()
    # qimg = pixmap.toImage()
    # qimg = qimg.convertToFormat(qtg.QImage.Format.Format_RGB32)
    # ptr = qimg.constBits()
    # # ptr.setsize(h * w * 4)
    # arr = np.frombuffer(ptr, np.uint8).reshape((w, h, 4)).copy()

    arr = arr[..., :3]
    arr = arr[:, :, ::-1]  # RGB to BGR
    arr = arr.copy()
    return arr
