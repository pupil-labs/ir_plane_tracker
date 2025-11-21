from .. import (
    CameraIntrinsics,
    EyeTrackingData,
    EyeTrackingSource,
)


class RecordingSource(EyeTrackingSource):
    def __init__(self, recording_path):
        import pupil_labs.neon_recording as plr

        self._rec = plr.open(recording_path)
        self._rec_data = zip(
            self._rec.scene.ts,
            self._rec.scene.sample(self._rec.scene.ts),
            self._rec.gaze.sample(self._rec.scene.ts),
            strict=False,
        )

    @property
    def scene_intrinsics(
        self,
    ) -> CameraIntrinsics:
        intrinsics = self._rec.calibration
        assert intrinsics is not None
        camera_matrix = intrinsics.scene_camera_matrix
        distortion_coeffs = intrinsics.scene_distortion_coefficients
        return CameraIntrinsics(camera_matrix, distortion_coeffs)

    def get_sample(self) -> EyeTrackingData:
        ts, frame, gaze = next(self._rec_data)
        return EyeTrackingData(ts, (gaze.x, gaze.y), frame.bgr)

    def close(self):
        pass
