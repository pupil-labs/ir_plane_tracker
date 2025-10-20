from fcntl import ioctl
from select import select

import pyrav4l2.v4l2 as v4l2
from pyrav4l2.stream import Stream


class V4lStream(Stream):
    def open(self):
        if self.f_cam.closed:
            self._open()

        ioctl(
            self.f_cam,
            v4l2.VIDIOC_STREAMON,
            v4l2.ctypes.c_int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE),
        )
        select((self.f_cam,), (), ())

    def close(self):
        self.f_cam.close()

    def get_frame(self):
        try:
            buf = self.buffers[0][0]
            ioctl(self.f_cam, v4l2.VIDIOC_DQBUF, buf)

            frame = self.buffers[buf.index][1][: buf.bytesused]
            ioctl(self.f_cam, v4l2.VIDIOC_QBUF, buf)
        except Exception:
            self._stop()
        else:
            return frame
