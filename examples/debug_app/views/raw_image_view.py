import cv2
from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class RawImageView(ScaledImageView):
    name = "Raw Image"

    def update_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = eye_tracking_data.scene.copy()

        if plane_localization is not None:
            cv2.polylines(
                vis, [plane_localization.corners.astype(int)], True, (255, 0, 0), 3
            )
        self.set_image(vis)
