from time import time

from ..eye_tracking_sources import CameraIntrinsics, EyeTrackingData, EyeTrackingSource


class NeonUSB(EyeTrackingSource):
    def __init__(self):
        from common.camera import SceneCam

        self._cam = SceneCam()

    @property
    def scene_intrinsics(
        self,
    ) -> CameraIntrinsics:
        camera_matrix, dist_coeffs = self._cam.get_intrinsics()
        return CameraIntrinsics(camera_matrix, dist_coeffs)

    def get_sample(self) -> EyeTrackingData:
        frame = self._cam.get_frame()
        img = frame.bgr
        timestamp = time()
        gaze = None

        return EyeTrackingData(timestamp, gaze, img)

    def close(self):
        self._cam.close()
