from pathlib import Path

import pandas as pd

from pupil_labs import neon_recording as plr
from pupil_labs.camera import Camera
from pupil_labs.neon_recording import match_ts

from . import (
    EyeTrackingData,
    EyeTrackingSource,
)


class NeonRecording(EyeTrackingSource):
    def __init__(self, recording_folder: str | Path):
        super().__init__()

        self.rec = plr.open(recording_folder)
        assert self.rec.calibration is not None, "Recording has no calibration data."
        self.scene_intrinsics = Camera(
            pixel_width=1600,
            pixel_height=1200,
            camera_matrix=self.rec.calibration.scene_camera_matrix,
            distortion_coefficients=self.rec.calibration.scene_distortion_coefficients,
        )

        matching_df = match_ts(
            self.rec.eye.time,
            self.rec.scene.time,
        )
        matching_df = pd.Series(matching_df, name="scene_idx")
        matching_df.index.name = "eye_idx"
        matching_df = matching_df.reset_index()
        matching_df = matching_df.groupby("scene_idx").agg(list)
        self.matching_iter = matching_df.iterrows()

    def get_sample(self) -> EyeTrackingData:
        scene_idx, eye_idxs = next(self.matching_iter)
        eye_idxs = eye_idxs["eye_idx"]

        scene_frame = self.rec.scene[scene_idx]
        eye_image = self.rec.eye[eye_idxs[-1]].gray
        gaze = self.rec.gaze[eye_idxs].mean(axis=0)

        data = EyeTrackingData(
            time=scene_frame.time,
            gaze_scene_distorted=gaze,
            scene_image_distorted=scene_frame.bgr,
            intrinsics=self.scene_intrinsics,
            eye_image=eye_image,
        )
        return data

    def close(self):
        pass
