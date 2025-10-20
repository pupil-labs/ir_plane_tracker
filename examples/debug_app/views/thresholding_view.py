from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class ThresholdingView(ScaledImageView):
    name = "Thresholding"

    def update_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = debug.img_thresholded.copy()
        self.set_image(vis)
