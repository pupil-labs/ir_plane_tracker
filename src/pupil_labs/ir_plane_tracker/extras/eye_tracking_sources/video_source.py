import numpy as np

from pupil_labs.video import Reader

from . import (
    EyeTrackingData,
    EyeTrackingSource,
)


class VideoSource(EyeTrackingSource):
    def __init__(self, path: str):
        super().__init__()

        video = Reader(path)
        self._frame_iterator = iter(video)

    def get_sample(self) -> EyeTrackingData:
        frame = next(self._frame_iterator)
        gaze = np.zeros(2, dtype=np.float64)
        data = EyeTrackingData(
            time=frame.time,
            gaze_scene_distorted=gaze,
            scene_image_distorted=frame.bgr,
            intrinsics=self.scene_intrinsics,
            eye_image=None,
        )
        return data

    def close(self):
        self.scene_stop_event.set()
        if self._pipeline is not None:
            self.eye_stop_event.set()
