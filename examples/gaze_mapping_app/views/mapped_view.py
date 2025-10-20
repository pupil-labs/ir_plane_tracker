import cv2
import numpy.typing as npt
from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class MappedView(ScaledImageView):
    def __init__(self):
        super().__init__()
        self._background = cv2.imread("gaze_mapping_app/screenshot.png")
        self.set_image(self._background)
        self.gaze = None

    def update_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization | None,
        debug: DebugData,
        gaze_mapped: npt.NDArray | None,
    ):
        if gaze_mapped is not None:
            vis = self._background.copy()
            gaze_mapped = gaze_mapped[:2] * self._background.shape[1::-1]
            cv2.circle(
                vis,
                (int(gaze_mapped[0]), int(gaze_mapped[1])),
                30,
                (0, 0, 255),
                3,
            )

            self.set_image(vis)
