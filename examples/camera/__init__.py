import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import uvc
from pyrav4l2 import Device, v4l2

from .camera import CameraNotFoundError, CameraSpec
from .frame import Frame
from .v4lstream import V4lStream


class CameraBackend(ABC):
    def __init__(self, spec: CameraSpec):
        self.spec = spec

    @abstractmethod
    def get_frame(self) -> Frame:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class UVCBackend(CameraBackend):
    def __init__(self, spec: CameraSpec, extended_controls=None):
        super().__init__(spec)

        self._uvc_capture = None
        self.spec = spec
        self.extended_controls = extended_controls
        self.exposure_controls = None

        uid = self._find_uid()
        capture = uvc.Capture(uid, self.extended_controls)
        capture.bandwidth_factor = self.spec.bandwidth_factor
        self._set_capture_mode(capture)

    def _find_uid(self):
        uid = None
        connected_devices = uvc.device_list()
        for device_info in connected_devices:
            if device_info["name"] == self.spec.name:
                uid = device_info["uid"]
                break

        if uid is None:
            raise CameraNotFoundError(self.spec.name)
        return uid

    def _set_capture_mode(self, capture):
        mode_matched = False
        for mode in capture.all_modes:
            if (mode.width, mode.height, mode.fps) == (
                self.spec.width,
                self.spec.height,
                self.spec.fps,
            ):
                mode_matched = True
                capture.frame_mode = mode
                self.last_frame_timestamp: float = uvc.get_time_monotonic()
                self._uvc_capture = capture

                break

        if not mode_matched:
            capture.close()
            raise OSError(
                f"None of the available modes matched: {capture.available_modes}!"
            )

    def get_frame(self) -> Frame:
        if self._uvc_capture is None:
            raise OSError("Camera not initialized!")

        frame = self._uvc_capture.get_frame(timeout=2.0)
        frame.timestamp = self.last_frame_timestamp = uvc.get_time_monotonic()
        assert frame is not None
        return frame

    def close(self) -> None:
        if self._uvc_capture is not None:
            self._uvc_capture.close()
            del self._uvc_capture
        self._uvc_capture = None


class V4l2Backend(CameraBackend):
    def __init__(self, spec: CameraSpec):  # noqa: C901
        super().__init__(spec)

        self.camera_reinit_timeout = 3
        self.device = None
        self.frame_counter = -1

        for device_path in Path("/dev/").glob("video*"):
            try:
                device = Device(device_path)
            except AttributeError:
                continue
            except FileNotFoundError:
                continue
            except PermissionError:
                continue

            if self.spec.name in device.device_name and device.is_video_capture_capable:
                formats = []
                for color_format, frame_sizes in device.available_formats.items():
                    for frame_size in frame_sizes:
                        for frame_interval in device.get_available_frame_intervals(
                            color_format, frame_size
                        ):
                            formats.append((color_format, frame_size, frame_interval))  # noqa: PERF401

                for color_format, frame_size, frame_interval in formats:
                    fps = frame_interval.denominator / frame_interval.numerator
                    if (frame_size.width, frame_size.height, fps) == (
                        self.spec.width,
                        self.spec.height,
                        self.spec.fps,
                    ):
                        device.set_format(color_format, frame_size)
                        device.set_frame_interval(frame_interval)
                        self.device = device
                        self.stream = V4lStream(self.device)
                        self.stream.open()
                        with open(self.device.path) as fd:
                            self._fd = fd
                            self.color_format, _ = self.device.get_format()

                        break

                else:
                    raise OSError("None of the available modes matched!")

        if self.device is None:
            raise CameraNotFoundError(self.spec.name)

    def get_frame(self) -> Frame:
        buffer = self.stream.get_frame()
        if buffer is None:
            raise TimeoutError

        if self.color_format.pixelformat == v4l2.V4L2_PIX_FMT_GREY:
            pixels = np.frombuffer(buffer, dtype=np.uint8).reshape([
                self.spec.height,
                self.spec.width,
            ])
        elif self.color_format.pixelformat == v4l2.V4L2_PIX_FMT_MJPEG:
            pixels = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
        elif self.color_format.pixelformat == v4l2.V4L2_PIX_FMT_YUYV:
            yuyv = np.frombuffer(buffer, dtype=np.uint8).reshape([
                self.spec.height,
                self.spec.width,
                2,
            ])
            pixels = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)

        self.frame_counter += 1

        return Frame(pixels, time.time(), self.frame_counter)

    def close(self) -> None:
        self._fd.close()


class PNSCam(V4l2Backend):
    def __init__(self):
        spec = CameraSpec(
            name="Tracking Camera",
            vendor_id=10550,
            product_id=4614,
            width=512,
            height=512,
            fps=45,
            bandwidth_factor=1.0,
        )
        super().__init__(spec)
        controls = {c.display_name: c for c in self._uvc_capture.controls}
        controls["Auto Exposure Mode"].value = 1
        controls["Absolute Exposure Time"].value = 10

    def get_frame(self) -> Frame:
        frame = super().get_frame()
        pixels = frame.data
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        pixels = pixels[:, :256]

        # rotate 90 degrees counter clockwise
        pixels = cv2.rotate(pixels, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame = Frame(pixels, frame.timestamp, frame.index)
        return frame


class HDDigitalCam(UVCBackend):
    def __init__(self):
        spec = CameraSpec(
            name="HD USB Camera",
            vendor_id=13028,
            product_id=37424,
            width=640,
            height=480,
            fps=120,
            bandwidth_factor=1.0,
        )
        super().__init__(spec)
        controls = {c.display_name: c for c in self._uvc_capture.controls}
        controls["Auto Exposure Mode"].value = 1
        controls["Absolute Exposure Time"].value = 10
