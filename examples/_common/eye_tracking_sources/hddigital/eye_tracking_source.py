from time import time

import numpy as np

from pupil_labs.neon_usb.cameras.backend import UVCBackend
from pupil_labs.neon_usb.cameras.camera import CameraSpec

from .. import (
    CameraIntrinsics,
    EyeTrackingData,
    EyeTrackingSource,
)


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


class HDDigital(EyeTrackingSource):
    def __init__(self):
        self._cam = HDDigitalCam()

    @property
    def scene_intrinsics(
        self,
    ) -> CameraIntrinsics:
        camera_matrix = np.load("resources/camera_matrix.npy")
        dist_coeffs = np.load("resources/dist_coeffs.npy")
        return CameraIntrinsics(camera_matrix, dist_coeffs)

    def get_sample(self) -> EyeTrackingData:
        frame = self._cam.get_frame()
        img = frame.bgr
        timestamp = time()
        gaze = None

        return EyeTrackingData(timestamp, gaze, img)

    def close(self):
        self._cam.close()
