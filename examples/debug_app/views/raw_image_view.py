import cv2
from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView
from debug_app.views import View
from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class RawImageView(View):
    name = "Raw Image"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        self.thresh_c = LabeledSlider("c_thresh", 5, 80, 40)
        sidebar_layout.addWidget(self.thresh_c)

        sidebar_layout.addStretch()
        layout.addLayout(sidebar_layout, stretch=1)

        self.setLayout(layout)

    def set_data(
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
        self.image_view.set_image(vis)
