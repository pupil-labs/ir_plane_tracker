import time

from .. import (
    CameraIntrinsics,
    EyeTrackingData,
    EyeTrackingSource,
)


class NeonLocal(EyeTrackingSource):
    def __init__(self):
        super().__init__()
        from pupil_labs.neon_usb import SceneCamera

        print("Connecting to camera...")
        scene_cam = SceneCamera()
        scene_cam.exposure = 120
        print("Done.")

        print("Loading intrinsics...")
        intrinsics = scene_cam.get_intrinsics()
        intrinsics = CameraIntrinsics(
            intrinsics.camera_matrix, intrinsics.distortion_coefficients
        )
        print("Done.")
        self._scene_cam = scene_cam
        self._intrinsics = intrinsics

    @property
    def scene_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def get_sample(self) -> EyeTrackingData:
        ts = time.time()
        scene_frame = self._scene_cam.get_frame()
        data = EyeTrackingData(time=ts, gaze=None, scene=scene_frame.bgr)
        return data

    def close(self):
        self._scene_cam.close()
