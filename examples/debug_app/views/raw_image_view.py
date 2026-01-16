import cv2
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.ir_plane_tracker.extras.eye_tracking_sources import EyeTrackingData
from pupil_labs.ir_plane_tracker.extras.ui.scaled_image_view import ScaledImageView

from .view import View


class RawImageView(View):
    name = "Raw Image"

    def __init__(self, tracker_param_changed, parent=None):
        super().__init__(parent)
        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        sidebar_layout.addStretch()
        layout.addLayout(sidebar_layout, stretch=1)

        self.setLayout(layout)

        self.make_connections(tracker_param_changed)

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = eye_tracking_data.scene_image_undistorted.copy()

        if plane_localization is not None:
            cv2.polylines(
                vis, [plane_localization.corners.astype(int)], True, (255, 0, 0), 3
            )
        self.image_view.set_image(vis)
