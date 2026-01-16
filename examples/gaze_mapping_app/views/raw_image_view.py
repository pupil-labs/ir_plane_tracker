import cv2
import numpy.typing as npt

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.ir_plane_tracker.extras.eye_tracking_sources import EyeTrackingData
from pupil_labs.ir_plane_tracker.extras.ui.scaled_image_view import ScaledImageView


class RawImageView(ScaledImageView):
    def update_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization | None,
        debug: DebugData,
        gaze_mapped: npt.NDArray | None,
    ):
        vis = eye_tracking_data.scene_image_distorted.copy()
        if eye_tracking_data.gaze_scene_distorted is not None:
            cv2.circle(
                vis,
                (
                    int(eye_tracking_data.gaze_scene_distorted[0]),
                    int(eye_tracking_data.gaze_scene_distorted[1]),
                ),
                30,
                (0, 0, 255),
                3,
            )

        if plane_localization is not None:
            cv2.polylines(
                vis, [plane_localization.corners.astype(int)], True, (255, 0, 0), 3
            )
        self.set_image(vis)
