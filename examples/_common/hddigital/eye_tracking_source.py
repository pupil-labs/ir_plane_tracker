from time import time

import numpy as np

from ..eye_tracking_sources import CameraIntrinsics, EyeTrackingData, EyeTrackingSource


class HDDigital(EyeTrackingSource):
    def __init__(self):
        from common.camera import HDDigitalCam

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
