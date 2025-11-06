from ..eye_tracking_sources import (
    CameraIntrinsics,
    EyeTrackingData,
    EyeTrackingSource,
)


class NeonRemote(EyeTrackingSource):
    def __init__(self, device):
        super().__init__()
        self._device = device

    @property
    def scene_intrinsics(self) -> CameraIntrinsics:
        intrinsics = self._device.get_calibration()
        camera_matrix = intrinsics["scene_camera_matrix"]
        distortion_coeffs = intrinsics["scene_distortion_coefficients"]
        return CameraIntrinsics(camera_matrix, distortion_coeffs)

    def get_sample(self) -> EyeTrackingData:
        scene_and_gaze = self._device.receive_matched_scene_video_frame_and_gaze(
            timeout_seconds=1 / 10
        )
        if scene_and_gaze is None:
            raise RuntimeError("No scene and gaze data received")

        scene, gaze = scene_and_gaze
        time = gaze.timestamp_unix_seconds
        return EyeTrackingData(time, (gaze.x, gaze.y), scene.bgr_pixels)

    def close(self):
        self._device.close()
