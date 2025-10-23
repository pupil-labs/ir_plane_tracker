from common.eye_tracking_sources import EyeTrackingData
from PySide6.QtWidgets import QWidget

from pupil_labs.ir_plane_tracker.tracker_line_and_dots import (
    DebugData,
    PlaneLocalization,
)


class View(QWidget):
    name = "Base View"

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        pass
