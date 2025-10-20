import cv2
import numpy.typing as npt
from common import utils
from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView
from PySide6.QtGui import QScreen

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class MappedView(ScaledImageView):
    def __init__(self, target_screen: QScreen, parent=None):
        super().__init__()
        self.target_screen = target_screen
        self.gaze = None

    def update_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization | None,
        debug: DebugData,
        gaze_mapped: npt.NDArray | None,
    ):
        vis = self.target_screen.grabWindow(0)
        vis = utils.numpy_from_qpixmap(vis)
        if gaze_mapped is not None:
            gaze_mapped = gaze_mapped[:2] * vis.shape[1::-1]
            cv2.circle(
                vis,
                (int(gaze_mapped[0]), int(gaze_mapped[1])),
                30,
                (0, 0, 255),
                3,
            )

        self.set_image(vis)
